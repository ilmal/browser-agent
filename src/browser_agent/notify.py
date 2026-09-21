"""Escalation alerts.

Reuses the ops-alert endpoint already used elsewhere, so a blocked run reaches
the same place as other operational failures.
"""

from __future__ import annotations

import logging

import httpx

from .config import Settings
from .escalation import Challenge

log = logging.getLogger(__name__)


async def notify_escalation(settings: Settings, challenge: Challenge, takeover_url: str) -> bool:
    """Tell a human that this profile needs hands. Returns True if delivered.

    Never raises: a failed alert must not mask the escalation itself, but it is
    logged loudly because a silent escalation is an account sitting blocked.
    """
    if not settings.notify_on_escalation:
        log.info("escalation (notify disabled): %s", challenge.describe())
        return False

    if not settings.ops_alert_url:
        log.warning(
            "ESCALATION for profile %s but OPS_ALERT_URL is unset: %s — take over at %s",
            settings.profile,
            challenge.describe(),
            takeover_url,
        )
        return False

    message = (
        f"browser-agent profile '{settings.profile}' needs help\n"
        f"{challenge.describe()}\n"
        f"take over: {takeover_url}"
    )
    # Carry the text under every common webhook key. The receivers disagree —
    # Discord wants `content`, Slack wants `text`, a generic receiver wants
    # `message` — and a mismatch is not an error: Discord answers 400 "Cannot
    # send an empty message" for a payload with no `content`, which
    # raise_for_status would then report as a delivery failure for an alert
    # that never had a chance. Sending all three makes any of them work.
    payload = {"message": message, "content": message, "text": message}
    try:
        # trust_env=False: the alert must not depend on the browser's egress
        # proxy being reachable, and a failure here is swallowed by design —
        # exactly the kind of silent drop this module exists to avoid.
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            resp = await client.post(settings.ops_alert_url, json=payload)
            resp.raise_for_status()
        log.info("escalation alert delivered for %s", settings.profile)
        return True
    except Exception as exc:
        log.error(
            "ALERT FAILED for %s (%s) — blocked run at %s: %s",
            settings.profile,
            exc,
            takeover_url,
            challenge.describe(),
        )
        return False
