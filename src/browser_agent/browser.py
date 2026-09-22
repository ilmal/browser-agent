"""Persistent browser session.

The context is intentionally *headed* even in a container: it renders into
Xvfb, which x11vnc mirrors, which is what lets a human take over the same
session to solve a captcha. A headless context would be un-takeover-able.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page, Playwright, async_playwright
from playwright.sync_api import sync_playwright

from .activity import Activity
from .config import Settings

log = logging.getLogger(__name__)

# Chrome writes the port it actually bound (the first line) and the browser
# websocket path (the second) into this file inside the user-data-dir, even
# when launched with `--remote-debugging-port=0`. Reading it is how a second
# process attaches without guessing a port or hardcoding one.
_DEVTOOLS_PORT_FILE = "DevToolsActivePort"

_browser_exe: str | None = None


def chromium_executable() -> str | None:
    """Path to the Chromium Playwright will launch, resolved once.

    Playwright's async and sync APIs expose the same bundled binary; the sync
    entry point is the one that can be queried without a running event loop,
    which is what lets a UA helper stay synchronous.
    """
    global _browser_exe
    if _browser_exe is None:
        try:
            with sync_playwright() as p:
                _browser_exe = p.chromium.executable_path
        except Exception:
            log.debug("could not resolve the Chromium path", exc_info=True)
            _browser_exe = ""
    return _browser_exe or None


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


def _browser_version() -> str | None:
    """The bundled Chromium's own version, e.g. ``153.0.8010.12``.

    Read from the binary so a Playwright upgrade cannot leave a hand-written
    version string behind that no longer matches the engine underneath it.
    """
    try:
        exe = chromium_executable()
        if not exe:
            return None
        proc = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+\.\d+\.\d+\.\d+)", proc.stdout)
    return match.group(1) if match else None


def _user_agent(headless: bool) -> str | None:
    """A UA string that does not announce the browser as automated.

    Sites gate on this. minesweeper.online, for one, serves a script-less shell
    with **zero** game cells to the stock Playwright headless UA
    (``HeadlessChrome/…``) — the page looks loaded and is inert, which reads as
    "the selector broke" rather than "the request was refused". In headless
    mode Playwright also appends ``HeadlessChrome``, so the fix is to present a
    plain Chrome UA; headed mode already sends one and needs no override.
    """
    if not headless:
        return None
    version = _browser_version() or "131.0.0.0"
    return (
        f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{version} Safari/537.36"
    )


def _clear_stale_profile_lock(profile_dir: Path) -> None:
    """Remove a leftover Chrome profile lock when no browser holds it.

    `SingletonLock` is a symlink whose target text is "<hostname>-<pid>". Chrome
    refuses to start with PROFILE_IN_USE when that hostname is not the local
    one. A clean shutdown unlinks the lock, but a SIGKILL, OOM kill or node
    loss does not, so on a PVC-backed profile the next boot can find a lock
    belonging to a pod that no longer exists. Only removed when the DevTools
    port file is absent, which is what distinguishes "crashed" from "a live
    browser owns this profile" — clearing it under a running browser would be
    the very corruption the single-writer rule exists to prevent.
    """
    lock = profile_dir / "SingletonLock"
    if not lock.exists() and not lock.is_symlink():
        return
    if _read_devtools_endpoint(profile_dir, timeout_s=0.1) is not None:
        return
    try:
        lock.unlink()
        log.warning("cleared a stale Chrome profile lock in %s", profile_dir)
    except OSError:
        log.warning("could not clear the stale profile lock in %s", profile_dir, exc_info=True)


class BrowserSession:
    """Owns one Playwright persistent context and its pages."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        # Both live here because the session is the one object every layer
        # already holds, and there is exactly one per pod. Tasks are serialised
        # on a single worker, so a session-scoped activity log is the running
        # task's log; the runner rebinds them per task.
        self.activity = Activity()
        self.control: Any | None = None

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
        _clear_stale_profile_lock(self.settings.profile_dir)
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
        # Present a browser that does not announce itself as automated. Some
        # sites (minesweeper.online) serve a script-less shell to Playwright's
        # headless UA, which looks like a broken selector rather than a refusal.
        agent = _user_agent(self.settings.headless)
        if agent:
            launch["user_agent"] = agent
        # A persistent context cannot be launched with a proxy per-page; the
        # proxy belongs to the browser. Residential egress (office-proxy) is
        # what keeps these sessions from being flagged as datacenter. This is
        # the only place the proxy is configured — it is not exported as
        # HTTP_PROXY, which the LLM client would otherwise pick up.
        if self.settings.browser_proxy:
            launch["proxy"] = {"server": self.settings.browser_proxy}

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
    """True when a human has signed in to this profile.

    Chrome writes the profile directory on its very first launch, so "the
    directory has files in it" is true of a bot that has never been touched —
    which is exactly the state the operator needs to be warned about. The
    roster renders this as its signed-in / not-signed-in pill, so a fresh bot
    claiming to be signed in is worse than no signal at all.

    The honest test is whether a login left cookies behind: Chrome creates the
    empty Cookies database on first launch and fills it only when a site sets
    one. Measured on this deployment — a never-used profile has 0, a signed-in
    one has 11. The count is read over the file's own SQLite, read-only and
    with a short timeout, because the browser holds it open while running.
    """
    import sqlite3

    path: Path = settings.profile_dir
    if not path.is_dir():
        return False
    cookies = path / "Default" / "Cookies"
    if not cookies.is_file():
        # Pre-Chromium-96 layouts kept it under Default/Network. Checked rather
        # than assumed so a profile from an older image still reports right.
        cookies = path / "Default" / "Network" / "Cookies"
    if not cookies.is_file():
        return False
    try:
        # A locked database is not an error worth raising: it means the
        # browser is running, and the count is a roster nicety, not a control
        # path. Fall back to the directory check so a running-but-unknown
        # profile stays visible rather than flipping to "not signed in".
        con = sqlite3.connect(f"file:{cookies}?mode=ro", uri=True, timeout=1.0)
        try:
            n = con.execute("select count(*) from cookies").fetchone()[0]
        finally:
            con.close()
        return bool(n)
    except sqlite3.Error:
        return any(path.iterdir())
