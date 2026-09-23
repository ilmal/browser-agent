"""Semantic page readings shared by the plan executor and the agent fallback.

Both paths need the same two readings: a geometry-free fingerprint that means
"the page changed" without also meaning "the page repainted", and a short text
snapshot for the Laya gate. They live in one module so the plan executor's
no-progress latch and the agent's stall latch cannot drift apart.

The fingerprint is ported from jev-ultrafast's ``snapshot.js`` (its
``pageKey``/``marker`` pair), which is why it excludes geometry: a spinner
animating, a hover highlight, or a reflow must not read as progress, and real
progress must not read as stale.
"""

from __future__ import annotations

import asyncio
import hashlib
import time

from playwright.async_api import Page

#: Control state, not just text. Typing into a field leaves ``inner_text``
#: unchanged, so without this three legitimate field-fills would trip the
#: no-change latch.
_FP_CONTROLS_JS = """\
() => [...document.querySelectorAll('input, textarea, select')].map(e =>
  e.type === 'checkbox' || e.type === 'radio'
    ? (e.checked ? '1' : '0')
    : String(e.value == null ? '' : e.value).slice(0, 40)
).join('|')"""


async def state_text(page: Page, limit: int = 900) -> str:
    """Short text snapshot of the page for the Laya gate."""
    try:
        title = await page.title()
    except Exception:
        title = ""
    try:
        body = (await page.inner_text("body"))[:limit]
    except Exception:
        body = ""
    return f"{title}\n{body}".strip()


async def fingerprint(page: Page) -> str | None:
    """Semantic page fingerprint: url + title + visible text + control state.

    ``None`` means "could not read the page" — a navigating document, or one
    that closed mid-read. Callers must treat that as "no evidence", never as
    "unchanged"; a page that cannot be read is not a page that made no
    progress."""
    try:
        title = await page.title()
        body = (await page.inner_text("body"))[:2000]
        controls = await page.evaluate(_FP_CONTROLS_JS)
    except Exception:
        return None
    return hashlib.sha256(f"{page.url}|{title}|{body}|{controls}".encode()).hexdigest()


async def wait_stable(page: Page, budget_s: float = 2.0) -> None:
    """Best-effort quiescence wait: the fingerprint unchanged across one poll
    interval, within budget. A caller acts on what it just observed; clicking
    into a page that is still transitioning is how a correct pick lands on the
    wrong DOM. Bounded — a page that never settles must not stall the caller,
    the done_when/confirm gates still judge the outcome."""
    last = await fingerprint(page)
    if last is None:
        return
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        await asyncio.sleep(0.2)
        cur = await fingerprint(page)
        if cur is None or cur == last:
            return
        last = cur
