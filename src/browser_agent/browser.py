"""Persistent browser session.

The context is intentionally *headed* even in a container: it renders into
Xvfb, which x11vnc mirrors, which is what lets a human take over the same
session to solve a captcha. A headless context would be un-takeover-able.
"""

from __future__ import annotations

import logging
from pathlib import Path

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright

from .config import Settings

log = logging.getLogger(__name__)


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
