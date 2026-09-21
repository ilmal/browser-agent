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

_LOGIN_MARKERS = [
    "input[type='password']",
    "input[name='session_key']",  # LinkedIn
    "a[href*='/login']",
]


async def detect_challenge(page: Page) -> Challenge | None:
    """Return the blocker on this page, or None when it is safe to continue."""
    url = page.url

    for kind, selector, why in _CHALLENGE_SELECTORS:
        try:
            if await page.locator(selector).count() > 0:
                return Challenge(kind, why, url)
        except Exception:  # selector invalid for this page / frame detached
            continue

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
