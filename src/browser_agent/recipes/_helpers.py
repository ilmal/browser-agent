"""Shared recipe helpers."""

from __future__ import annotations

from playwright.async_api import Page

from ..escalation import Challenge, ChallengeKind, EscalationRequired, detect_challenge


async def require_clear(page: Page) -> None:
    """Stop the recipe if the page is showing a challenge.

    Called before and after any action that writes to the site, so a captcha
    cannot be stepped over mid-post.
    """
    challenge: Challenge | None = await detect_challenge(page)
    if challenge is not None:
        raise EscalationRequired(challenge)


async def click_first(page: Page, selectors: list[str], *, timeout: int = 8000) -> bool:
    """Click the first selector that matches and is visible."""
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if await loc.count() > 0 and await loc.is_visible(timeout=timeout):
                await loc.click()
                return True
        except Exception:
            continue
    return False


async def fill_first(page: Page, selectors: list[str], text: str, *, timeout: int = 8000) -> bool:
    """Fill the first selector that matches and is visible."""
    for selector in selectors:
        try:
            loc = page.locator(selector).first
            if await loc.count() > 0 and await loc.is_visible(timeout=timeout):
                await loc.fill(text)
                return True
        except Exception:
            continue
    return False


def rate_limited(detail: str = "site refused the action") -> EscalationRequired:
    """Build a stop-and-ask escalation for a rate limit."""
    return EscalationRequired(Challenge(ChallengeKind.RATE_LIMITED, detail, ""))
