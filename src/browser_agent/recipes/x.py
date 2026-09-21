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
from ..tasks import register
from ._helpers import click_first, rate_limited, require_clear

log = logging.getLogger(__name__)


class XPost:
    name = "x.post"
    description = "Post a text update to X as the logged-in account."
    entry_url = "https://x.com/home"

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        text = (payload.get("text") or "").strip()
        if not text:
            raise ValueError("payload.text is required")
        if len(text) > 280:
            raise ValueError(f"text is {len(text)} chars; X allows 280")

        page = await session.page()
        await require_clear(page)

        # Open the composer.
        opened = await click_first(
            page,
            [
                "a[data-testid='SideNav_NewTweet_Button']",
                "a[href='/compose/post']",
                "div[role='button'][aria-label*='Post' i]",
            ],
        )
        if not opened:
            raise RuntimeError("could not open the composer")

        editor = [
            "div[data-testid='tweetTextarea_0']",
            "div[role='textbox'][contenteditable='true']",
            "div[contenteditable='true'][aria-label*='Post' i]",
        ]
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
            page,
            [
                "button[data-testid='tweetButton']",
                "button[data-testid='tweetButtonInline']",
                "div[role='button'][data-testid='tweetButton']",
            ],
        )
        if not posted:
            raise rate_limited("Post button not found or not enabled")

        await require_clear(page)
        log.info("posted to X (%d chars)", len(text))
        return {"posted": True, "chars": len(text), "url": page.url}


register(XPost())
