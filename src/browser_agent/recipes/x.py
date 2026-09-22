"""X (Twitter) posting recipe.

The composer is reachable from the home timeline, so the entry URL is `/home`.
Selectors are ordered most-stable-first: the data-testid attributes survive X's
styling churn far better than CSS classes, and the aria-label fallback covers
the copy changing.
"""

from __future__ import annotations

import logging
from typing import Any

from ..browser import BrowserSession
from ..tasks import register_builtin
from ._config import cfg
from ._helpers import click_first, rate_limited, require_clear

log = logging.getLogger(__name__)

#: The built-in literals. Each is the default argument to ``cfg`` below, so the
#: recipe behaves exactly as it always did until an operator overrides that key
#: in the recipe library.
DEFAULT_ENTRY_URL = "https://x.com/home"
DEFAULT_COMPOSER = [
    "a[data-testid='SideNav_NewTweet_Button']",
    "a[href='/compose/post']",
    "div[role='button'][aria-label*='Post' i]",
]
DEFAULT_TEXTBOX = [
    "div[data-testid='tweetTextarea_0']",
    "div[role='textbox'][contenteditable='true']",
    "div[contenteditable='true'][aria-label*='Post' i]",
]
DEFAULT_SUBMIT = [
    "button[data-testid='tweetButton']",
    "button[data-testid='tweetButtonInline']",
    "div[role='button'][data-testid='tweetButton']",
]
DEFAULT_MAX_CHARS = 280


class XPost:
    name = "x.post"
    description = "Post a text update to X as the logged-in account."

    @property
    def entry_url(self) -> str:
        return cfg("x.post", "entry_url", DEFAULT_ENTRY_URL)

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        text = (payload.get("text") or "").strip()
        max_chars = cfg("x.post", "max_chars", DEFAULT_MAX_CHARS)
        if not text:
            raise ValueError("payload.text is required")
        if len(text) > max_chars:
            raise ValueError(f"text is {len(text)} chars; X allows {max_chars}")

        page = await session.page()
        await require_clear(page)

        # Open the composer.
        opened = await click_first(
            page, cfg("x.post", "selectors.composer", DEFAULT_COMPOSER)
        )
        if not opened:
            raise RuntimeError("could not open the composer")

        editor = cfg("x.post", "selectors.textbox", DEFAULT_TEXTBOX)
        # Prefer typing over fill(): the editor is contenteditable and fill()
        # bypasses the input events X listens to for enabling the Post button.
        typed = False
        for selector in editor:
            try:
                loc = page.locator(selector).first
                if await loc.count() > 0 and await loc.is_visible(timeout=8000):
                    await loc.click()
                    await page.keyboard.type(text, delay=25)
                    typed = True
                    break
            except Exception:
                continue
        if not typed:
            raise RuntimeError("could not find the post editor")

        await require_clear(page)

        posted = await click_first(
            page, cfg("x.post", "selectors.submit", DEFAULT_SUBMIT)
        )
        if not posted:
            raise rate_limited("Post button not found or not enabled")

        await require_clear(page)
        log.info("posted to X (%d chars)", len(text))
        return {"posted": True, "chars": len(text), "url": page.url}


register_builtin(XPost())
