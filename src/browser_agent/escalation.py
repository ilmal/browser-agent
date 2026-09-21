"""Challenge detection and human escalation.

The governing rule: when a site presents a captcha, a verification step or an
account warning, the run STOPS and asks a human. It never retries. Blind retry
is the behaviour that gets accounts flagged, so it is a bug here, not a
fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from playwright.async_api import Page

log = logging.getLogger(__name__)


class ChallengeKind(StrEnum):
    CAPTCHA = "captcha"
    TWO_FACTOR = "two_factor"
    ACCOUNT_WARNING = "account_warning"
    LOGIN_REQUIRED = "login_required"
    RATE_LIMITED = "rate_limited"
    UNKNOWN = "unknown"


@dataclass
class Challenge:
    """A blocker a human has to clear before the run can continue."""

    kind: ChallengeKind
    detail: str
    url: str

    def describe(self) -> str:
        return f"{self.kind.value}: {self.detail} ({self.url})"


# Ordered most-specific first. Each entry is (kind, css selector, why).
_CHALLENGE_SELECTORS: list[tuple[ChallengeKind, str, str]] = [
    (ChallengeKind.CAPTCHA, "iframe[src*='recaptcha']", "reCAPTCHA iframe"),
    (ChallengeKind.CAPTCHA, "iframe[src*='hcaptcha']", "hCaptcha iframe"),
    (ChallengeKind.CAPTCHA, "iframe[title*='captcha' i]", "captcha iframe"),
    (ChallengeKind.CAPTCHA, "div[class*='captcha' i]", "captcha element"),
    (ChallengeKind.CAPTCHA, "#captcha", "captcha element"),
    (ChallengeKind.CAPTCHA, "img[src*='captcha' i]", "captcha image"),
    (ChallengeKind.TWO_FACTOR, "input[autocomplete='one-time-code']", "2FA code field"),
    (ChallengeKind.TWO_FACTOR, "input[name*='verification' i]", "verification field"),
    (ChallengeKind.TWO_FACTOR, "input[name*='otp' i]", "OTP field"),
]

# Text that appears when a run must stop rather than continue.
_CHALLENGE_TEXT: list[tuple[ChallengeKind, str]] = [
    (ChallengeKind.RATE_LIMITED, "you've reached the limit"),
    (ChallengeKind.RATE_LIMITED, "too many requests"),
    (ChallengeKind.RATE_LIMITED, "try again later"),
    (ChallengeKind.ACCOUNT_WARNING, "unusual activity"),
    (ChallengeKind.ACCOUNT_WARNING, "we've temporarily restricted"),
    (ChallengeKind.ACCOUNT_WARNING, "verify it's you"),
    (ChallengeKind.ACCOUNT_WARNING, "confirm your identity"),
    (ChallengeKind.ACCOUNT_WARNING, "your account has been locked"),
    (ChallengeKind.ACCOUNT_WARNING, "suspicious login"),
]

# Third-party bot walls (Cloudflare, Akamai, DataDome, PerimeterX) are served
# from the vendor's own origin, so they render in a cross-origin iframe — an
# "out-of-process iframe" the main frame cannot see. page.locator() and
# page.inner_text("body") therefore report a clean page while the human is
# actually staring at "Verifying you are human". Every frame has to be checked.
_CHALLENGE_FRAME_URLS: list[tuple[ChallengeKind, str, str]] = [
    (ChallengeKind.CAPTCHA, "challenges.cloudflare.com", "Cloudflare challenge"),
    (ChallengeKind.CAPTCHA, "challenge-platform", "Cloudflare challenge platform"),
    (ChallengeKind.CAPTCHA, "cdn-cgi/challenge", "Cloudflare challenge"),
    (ChallengeKind.CAPTCHA, "geo.captcha-delivery.com", "DataDome block"),
    (ChallengeKind.CAPTCHA, "captcha.px-cdn.net", "PerimeterX block"),
    (ChallengeKind.CAPTCHA, "px-cloud.net", "PerimeterX block"),
    (ChallengeKind.CAPTCHA, "/akam/", "Akamai bot manager"),
    (ChallengeKind.CAPTCHA, "hcaptcha.com", "hCaptcha"),
    (ChallengeKind.CAPTCHA, "recaptcha", "reCAPTCHA"),
    (ChallengeKind.CAPTCHA, "funcaptcha", "Arkose/FunCaptcha"),
    (ChallengeKind.CAPTCHA, "geetest", "GeeTest"),
    (ChallengeKind.CAPTCHA, "turnstile", "Cloudflare Turnstile"),
]

# Markers that live in the page source of a vendor interstitial but are not
# visible to the DOM query above.
_CHALLENGE_SOURCE_MARKERS: list[tuple[ChallengeKind, str]] = [
    (ChallengeKind.CAPTCHA, "challenges.cloudflare.com"),
    (ChallengeKind.CAPTCHA, "cf-chl-"),
    (ChallengeKind.CAPTCHA, "turnstile"),
    (ChallengeKind.CAPTCHA, "captcha-delivery.com"),
    (ChallengeKind.CAPTCHA, "px-cdn.net"),
]

_LOGIN_MARKERS = [
    "input[type='password']",
    "input[name='session_key']",  # LinkedIn
    "a[href*='/login']",
]


async def _detect_in_frame(frame: Any) -> Challenge | None:
    """Look for a challenge inside one frame's own document."""
    for kind, selector, why in _CHALLENGE_SELECTORS:
        try:
            if await frame.locator(selector).count() > 0:
                return Challenge(kind, why, frame.url)
        except Exception:  # selector invalid here / frame detached mid-check
            continue
    return None


async def detect_challenge(page: Page) -> Challenge | None:
    """Return the blocker on this page, or None when it is safe to continue.

    Every frame is inspected, not just the main one. A vendor bot wall
    (Cloudflare, Akamai, DataDome, PerimeterX) is served from the vendor's own
    origin, so it renders in a cross-origin out-of-process iframe that
    `page.locator()` cannot see — the main frame looks clean while the human is
    staring at "Verifying you are human". The frame URL is also checked because
    the marker may be in a frame whose document has not parsed yet.
    """
    url = page.url

    for frame in page.frames:
        try:
            frame_url = frame.url.lower()
        except Exception:
            frame_url = ""

        for kind, origin, why in _CHALLENGE_FRAME_URLS:
            if origin in frame_url:
                return Challenge(kind, why, url)

        found = await _detect_in_frame(frame)
        if found is not None:
            return found

    # Source-level markers catch a wall whose iframe has not rendered yet.
    try:
        html = (await page.content())[:200_000].lower()
    except Exception:
        html = ""
    for kind, needle in _CHALLENGE_SOURCE_MARKERS:
        if needle in html:
            return Challenge(kind, f"page source: {needle!r}", url)

    # Main-frame text: rate limits and account warnings are served inline.
    try:
        body = (await page.inner_text("body"))[:20_000].lower()
    except Exception:
        body = ""

    for kind, needle in _CHALLENGE_TEXT:
        if needle in body:
            return Challenge(kind, f"page text: {needle!r}", url)

    return None


async def looks_logged_out(page: Page) -> bool:
    """Heuristic: a visible password field usually means a login wall."""
    for selector in _LOGIN_MARKERS:
        try:
            loc = page.locator(selector)
            if await loc.count() > 0 and await loc.first.is_visible():
                return True
        except Exception:
            continue
    return False


class EscalationRequired(Exception):
    """Raised to unwind a task when a human is needed to continue.

    Carrying the Challenge out as an exception (rather than returning a status)
    makes it impossible for a caller to accidentally treat a blocked run as a
    success and retry it.
    """

    def __init__(self, challenge: Challenge) -> None:
        super().__init__(challenge.describe())
        self.challenge = challenge
