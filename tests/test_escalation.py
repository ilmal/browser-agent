"""Escalation tests.

These are the load-bearing tests. The whole safety argument for this repo is
that a challenge STOPS a run instead of being retried, so these drive a real
browser against real HTML rather than mocking the detector.
"""

from __future__ import annotations

import http.server
import socket
import sys
import threading
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent.escalation import (  # noqa: E402
    ChallengeKind,
    EscalationRequired,
    detect_challenge,
    looks_logged_out,
)
from browser_agent.recipes._helpers import require_clear  # noqa: E402

CAPTCHA_PAGE = """<html><body><h1>Sign in</h1>
<div class="g-recaptcha"><iframe src="https://www.google.com/recaptcha/api2/anchor"></iframe></div>
</body></html>"""

CLEAN_PAGE = """<html><body><h1>Home</h1>
<div role="textbox" contenteditable="true"></div><button>Post</button>
</body></html>"""

WARNING_PAGE = """<html><body><h1>Unusual activity</h1>
<p>We've temporarily restricted your account. Please confirm your identity.</p></body></html>"""

LOGIN_PAGE = """<html><body><form>
<input type="text" name="user"><input type="password" name="pass"></form></body></html>"""

RATE_LIMIT_PAGE = """<html><body><p>Too many requests. Try again later.</p></body></html>"""

OTP_PAGE = """<html><body><h1>Verify</h1>
<input autocomplete="one-time-code" name="code"></body></html>"""

# A Cloudflare-style wall: the visible block lives in a cross-origin iframe, so
# the main frame contains no captcha selector and no captcha text. "{{ORIGIN}}"
# is substituted with a second hostname (localhost vs 127.0.0.1) so the frame
# is genuinely cross-origin, as a vendor wall is.
VENDOR_WALL_PAGE = """<html><body><h1>Just a moment...</h1>
<iframe src="{{ORIGIN}}/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1"
        width="400" height="300"></iframe>
</body></html>"""

# The wall's own document: a spinner, nothing the detector's selectors match.
VENDOR_WALL_FRAME = """<html><body><div id="spinner">Checking your browser…</div></body></html>"""

# A wall that has not rendered its iframe yet, announced only in the source.
VENDOR_SCRIPT_PAGE = """<html><body><h1>Loading</h1>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js"></script>
</body></html>"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def site():
    pages = {
        "/captcha": CAPTCHA_PAGE,
        "/clean": CLEAN_PAGE,
        "/warning": WARNING_PAGE,
        "/login": LOGIN_PAGE,
        "/ratelimit": RATE_LIMIT_PAGE,
        "/otp": OTP_PAGE,
        "/vendorwall-frame": VENDOR_WALL_FRAME,
        "/vendorwall-script": VENDOR_SCRIPT_PAGE,
    }

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = pages.get(self.path, "<html><body>nope</body></html>").encode()
            self.send_response(200 if self.path in pages else 404)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    port = _free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # Served under two hostnames so a frame can be cross-origin without a
    # second socket. localhost and 127.0.0.1 are distinct origins.
    pages["/vendorwall"] = VENDOR_WALL_PAGE.replace(
        "{{ORIGIN}}", f"http://localhost:{port}"
    )
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
async def page():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        try:
            yield await (await browser.new_context()).new_page()
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_detects_captcha(page, site):
    await page.goto(f"{site}/captcha", wait_until="domcontentloaded")
    found = await detect_challenge(page)
    assert found is not None and found.kind is ChallengeKind.CAPTCHA


@pytest.mark.asyncio
async def test_detects_account_warning(page, site):
    await page.goto(f"{site}/warning", wait_until="domcontentloaded")
    found = await detect_challenge(page)
    assert found is not None and found.kind is ChallengeKind.ACCOUNT_WARNING


@pytest.mark.asyncio
async def test_detects_rate_limit(page, site):
    await page.goto(f"{site}/ratelimit", wait_until="domcontentloaded")
    found = await detect_challenge(page)
    assert found is not None and found.kind is ChallengeKind.RATE_LIMITED


@pytest.mark.asyncio
async def test_detects_two_factor(page, site):
    await page.goto(f"{site}/otp", wait_until="domcontentloaded")
    found = await detect_challenge(page)
    assert found is not None and found.kind is ChallengeKind.TWO_FACTOR


@pytest.mark.asyncio
async def test_clean_page_is_not_a_challenge(page, site):
    """The detector must not cry wolf, or every run would stop."""
    await page.goto(f"{site}/clean", wait_until="domcontentloaded")
    assert await detect_challenge(page) is None


@pytest.mark.asyncio
async def test_login_wall_detected(page, site):
    await page.goto(f"{site}/login", wait_until="domcontentloaded")
    assert await looks_logged_out(page) is True


@pytest.mark.asyncio
async def test_require_clear_raises_on_captcha(page, site):
    """A recipe must be structurally unable to step over a challenge."""
    await page.goto(f"{site}/captcha", wait_until="domcontentloaded")
    with pytest.raises(EscalationRequired) as exc:
        await require_clear(page)
    assert exc.value.challenge.kind is ChallengeKind.CAPTCHA


@pytest.mark.asyncio
async def test_require_clear_passes_clean_page(page, site):
    await page.goto(f"{site}/clean", wait_until="domcontentloaded")
    await require_clear(page)  # must not raise


@pytest.mark.asyncio
async def test_detects_wall_in_cross_origin_iframe(page, site):
    """The vendor wall is invisible to the main frame and must still be caught.

    Regression: detection used to read only the main frame, so a Cloudflare
    interstitial — served from challenges.cloudflare.com in an out-of-process
    iframe — reported a clean page and the run continued into the wall.
    """
    await page.goto(f"{site}/vendorwall", wait_until="domcontentloaded")
    # The guard: the main frame really is clean, which is why this was missed.
    assert await page.locator("div[class*='captcha' i]").count() == 0
    assert "captcha" not in (await page.inner_text("body")).lower()

    found = await detect_challenge(page)
    assert found is not None, "a cross-origin challenge frame went undetected"
    assert found.kind is ChallengeKind.CAPTCHA
    assert "cloudflare" in found.detail.lower()


@pytest.mark.asyncio
async def test_detects_wall_from_source_before_it_renders(page, site):
    """A wall announced in the source but not yet rendered is still a stop."""
    await page.goto(f"{site}/vendorwall-script", wait_until="domcontentloaded")
    found = await detect_challenge(page)
    assert found is not None, "a challenge announced in the source went undetected"
    assert found.kind is ChallengeKind.CAPTCHA
    assert "source" in found.detail.lower()
