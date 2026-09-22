"""LinkedIn posting recipe.

LinkedIn is the most aggressive of the three about automation, so this recipe
is deliberately conservative: it posts to a Company Page (the surface LinkedIn
sanctions) and treats every unexpected state as a stop rather than something to
click through.

Keep this one in the watched lane. Do not schedule it to run unattended until
it has a long clean history.
"""

from __future__ import annotations

import logging
from typing import Any

from ..browser import BrowserSession
from ..escalation import Challenge, ChallengeKind, EscalationRequired, detect_challenge
from ..tasks import register_builtin
from ._config import cfg
from ._helpers import click_first, require_clear

log = logging.getLogger(__name__)

DEFAULT_ENTRY_URL = "https://www.linkedin.com/feed/"
DEFAULT_COMPOSER = [
    "button[aria-label*='Create' i]",
    "button:has-text('Start a post')",
    "button[class*='share-box' i]",
]
DEFAULT_TEXTBOX = [
    "div[role='dialog'] div[role='textbox'], div.ql-editor[contenteditable='true']",
]
DEFAULT_SUBMIT = [
    "div[role='dialog'] button:has-text('Post')",
    "button[class*='share-actions__primary-action']",
]
DEFAULT_MAX_CHARS = 3000


class LinkedInPagePost:
    name = "linkedin.page_post"
    description = "Post a text update to a LinkedIn Company Page the profile administers."

    @property
    def entry_url(self) -> str:
        return cfg("linkedin.page_post", "entry_url", DEFAULT_ENTRY_URL)

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        text = (payload.get("text") or "").strip()
        admin_url = (payload.get("admin_url") or "").strip()
        if not text:
            raise ValueError("payload.text is required")
        max_chars = cfg("linkedin.page_post", "max_chars", DEFAULT_MAX_CHARS)
        if len(text) > max_chars:
            raise ValueError(f"LinkedIn posts allow {max_chars} characters")
        if not admin_url:
            raise ValueError(
                "payload.admin_url is required, e.g. "
                "https://www.linkedin.com/company/<id>/admin/page-posts/published/"
            )

        page = await session.page()
        await page.goto(admin_url, wait_until="domcontentloaded")
        await require_clear(page)

        # If LinkedIn has redirected us off the admin surface, it has decided
        # something about this session. Stop; do not navigate around it.
        if "/admin/" not in page.url:
            raise EscalationRequired(
                Challenge(
                    ChallengeKind.ACCOUNT_WARNING,
                    f"redirected away from the admin surface to {page.url}",
                    page.url,
                )
            )

        opened = await click_first(
            page, cfg("linkedin.page_post", "selectors.composer", DEFAULT_COMPOSER),
            timeout=12000,
        )
        if not opened:
            raise RuntimeError("could not open the Page composer")

        editor = page.locator(
            cfg("linkedin.page_post", "selectors.textbox", DEFAULT_TEXTBOX)[0]
        ).first
        await editor.wait_for(state="visible", timeout=15000)
        await editor.click()
        await page.keyboard.type(text, delay=25)

        await require_clear(page)

        posted = await click_first(
            page, cfg("linkedin.page_post", "selectors.submit", DEFAULT_SUBMIT),
            timeout=12000,
        )
        if not posted:
            raise RuntimeError("Post button not found or not enabled")

        await require_clear(page)
        # A modal that outlives the click usually means a warning was shown.
        if await detect_challenge(page) is None and await page.locator(
            "div[role='dialog']"
        ).count() > 0:
            log.info("dialog still open after posting; leaving it for review")

        log.info("posted to LinkedIn page (%d chars)", len(text))
        return {"posted": True, "chars": len(text), "url": page.url}


register_builtin(LinkedInPagePost())
