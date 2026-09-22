"""Facebook Page posting recipe.

Posts to a Page the profile administers. Personal-timeline posting is
deliberately not implemented: it is the surface Facebook polices hardest and
the one a human should be present for.
"""

from __future__ import annotations

import logging
from typing import Any

from ..browser import BrowserSession
from ..tasks import register_builtin
from ._config import cfg
from ._helpers import click_first, rate_limited, require_clear

log = logging.getLogger(__name__)

DEFAULT_ENTRY_URL = "https://www.facebook.com/"
DEFAULT_COMPOSER = [
    "div[role='button'][aria-label*=\"What's on your mind\" i]",
    "div[role='button'][aria-label*='Create post' i]",
    "span[data-testid='page_composer_text']",
]
DEFAULT_TEXTBOX = ["div[role='dialog'] div[role='textbox']"]
DEFAULT_SUBMIT = [
    "div[role='dialog'] div[role='button'][aria-label='Post']",
    "div[role='dialog'] div[role='button'][aria-label*='Publish' i]",
]
DEFAULT_MAX_CHARS = 63206


class FacebookPagePost:
    name = "facebook.page_post"
    description = "Post a text update to a Facebook Page the profile administers."

    @property
    def entry_url(self) -> str:
        return cfg("facebook.page_post", "entry_url", DEFAULT_ENTRY_URL)

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        text = (payload.get("text") or "").strip()
        page_id = (payload.get("page_id") or "").strip()
        if not text:
            raise ValueError("payload.text is required")
        max_chars = cfg("facebook.page_post", "max_chars", DEFAULT_MAX_CHARS)
        if len(text) > max_chars:
            raise ValueError(f"text is {len(text)} chars; this page allows {max_chars}")
        if not page_id:
            raise ValueError("payload.page_id is required (numeric Page id)")

        page = await session.page()
        await page.goto(f"https://www.facebook.com/{page_id}", wait_until="domcontentloaded")
        await require_clear(page)

        # Open the Page composer.
        opened = await click_first(
            page, cfg("facebook.page_post", "selectors.composer", DEFAULT_COMPOSER),
            timeout=12000,
        )
        if not opened:
            raise RuntimeError("could not open the Page composer")

        # The dialog lands asynchronously after the click.
        editor = page.locator(
            cfg("facebook.page_post", "selectors.textbox", DEFAULT_TEXTBOX)[0]
        ).first
        await editor.wait_for(state="visible", timeout=15000)
        await editor.click()
        await page.keyboard.type(text, delay=25)

        await require_clear(page)

        posted = await click_first(
            page, cfg("facebook.page_post", "selectors.submit", DEFAULT_SUBMIT),
            timeout=12000,
        )
        if not posted:
            raise rate_limited("Publish button not found or not enabled")

        await require_clear(page)
        log.info("posted to Facebook page %s (%d chars)", page_id, len(text))
        return {"posted": True, "page_id": page_id, "chars": len(text)}


register_builtin(FacebookPagePost())
