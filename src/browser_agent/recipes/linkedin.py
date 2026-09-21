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
from ..tasks import register
from ._helpers import click_first, require_clear

log = logging.getLogger(__name__)


class LinkedInPagePost:
    name = "linkedin.page_post"
    description = "Post a text update to a LinkedIn Company Page the profile administers."
    entry_url = "https://www.linkedin.com/feed/"

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        text = (payload.get("text") or "").strip()
        admin_url = (payload.get("admin_url") or "").strip()
        if not text:
            raise ValueError("payload.text is required")
        if len(text) > 3000:
            raise ValueError("LinkedIn posts allow 3000 characters")
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
            page,
            [
                "button[aria-label*='Create' i]",
                "button:has-text('Start a post')",
                "button[class*='share-box' i]",
            ],
            timeout=12000,
        )
        if not opened:
            raise RuntimeError("could not open the Page composer")

        editor = page.locator(
            "div[role='dialog'] div[role='textbox'], div.ql-editor[contenteditable='true']"
        ).first
        await editor.wait_for(state="visible", timeout=15000)
        await editor.click()
        await page.keyboard.type(text, delay=25)

        await require_clear(page)

        posted = await click_first(
            page,
            [
                "div[role='dialog'] button:has-text('Post')",
                "button[class*='share-actions__primary-action']",
            ],
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


register(LinkedInPagePost())
