"""Persistent browser session.

The context is intentionally *headed* even in a container: it renders into
Xvfb, which x11vnc mirrors, which is what lets a human take over the same
session to solve a captcha. A headless context would be un-takeover-able.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from .config import Settings

log = logging.getLogger(__name__)

# Chrome writes the port it actually bound (the first line) and the browser
# websocket path (the second) into this file inside the user-data-dir, even
# when launched with `--remote-debugging-port=0`. Reading it is how a second
# process attaches without guessing a port or hardcoding one.
_DEVTOOLS_PORT_FILE = "DevToolsActivePort"


def _read_devtools_endpoint(profile_dir: Path, timeout_s: float = 10.0) -> str | None:
    """Return the DevTools HTTP endpoint for the browser owning `profile_dir`.

    Chrome only writes this file while it is running, so its presence is also
    the signal that a live browser already holds the profile.
    """
    path = profile_dir / _DEVTOOLS_PORT_FILE
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            lines = path.read_text().splitlines()
            if lines and lines[0].strip().isdigit():
                return f"http://127.0.0.1:{lines[0].strip()}"
        except OSError:
            pass
        time.sleep(0.25)
    return None


class BrowserSession:
    """Owns one Playwright persistent context and its pages."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("browser not started")
        return self._context

    @property
    def cdp_endpoint(self) -> str | None:
        """DevTools HTTP endpoint for this running browser, or None.

        A second process (the LLM agent) attaches here so it drives the very
        same browser, cookies, tabs and display the human sees over noVNC.
        """
        return _read_devtools_endpoint(self.settings.profile_dir, timeout_s=1.0)

    async def start(self) -> BrowserContext:
        if self._context is not None:
            return self._context

        self.settings.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()

        launch: dict = {
            "user_data_dir": str(self.settings.profile_dir),
            "headless": self.settings.headless,
            "slow_mo": self.settings.slow_mo_ms,
            "viewport": {
                "width": self.settings.screen_width,
                "height": self.settings.screen_height,
            },
            "args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                # Bind a DevTools port on a random free port (0 = let the OS
                # choose) and publish it in the profile. Port 0 also switches
                # off the remote-allow-origins check, so no extra flag is
                # needed. This is how the LLM fallback attaches to *this*
                # browser instead of launching a second, logged-out one.
                "--remote-debugging-port=0",
            ],
            "ignore_default_args": ["--enable-automation"],
        }
        # A persistent context cannot be launched with a proxy per-page; the
        # proxy belongs to the browser. Residential egress (office-proxy) is
        # what keeps these sessions from being flagged as datacenter.
        if self.settings.https_proxy or self.settings.http_proxy:
            launch["proxy"] = {
                "server": self.settings.https_proxy or self.settings.http_proxy,
            }

        self._context = await self._playwright.chromium.launch_persistent_context(**launch)
        self._context.set_default_timeout(30_000)
        log.info(
            "browser started profile=%s dir=%s",
            self.settings.profile,
            self.settings.profile_dir,
        )
        return self._context

    async def page(self) -> Page:
        """Return the active page, creating one if the profile has none."""
        ctx = await self.start()
        if ctx.pages:
            return ctx.pages[0]
        return await ctx.new_page()

    async def goto(self, url: str, *, wait_until: str = "domcontentloaded") -> Page:
        page = await self.page()
        await page.goto(url, wait_until=wait_until)
        return page

    async def stop(self) -> None:
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        log.info("browser stopped profile=%s", self.settings.profile)


def profile_exists(settings: Settings) -> bool:
    """True when this profile has been initialised (i.e. a human logged in once)."""
    path: Path = settings.profile_dir
    return path.is_dir() and any(path.iterdir())
