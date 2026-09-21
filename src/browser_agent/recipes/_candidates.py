"""Page candidate extraction for the Laya element picker.

Laya classifies text, so the page's interactive elements are enumerated as
numbered lines. The numbering IS the contract: the picker returns an index and
the executor clicks ``locator.nth(index)`` — never a re-match on text, which
two identical buttons would make ambiguous. The ``:visible`` pseudo-classes in
the base selectors keep the filtered set ``locator.nth(i)`` walks aligned with
the lines we showed the model.
"""

from __future__ import annotations

import logging

from playwright.async_api import Locator, Page

log = logging.getLogger(__name__)

CLICK_BASE = (
    "a:visible, button:visible, [role=button]:visible, "
    "input[type=submit]:visible, [onclick]:visible"
)
TYPE_BASE = "input:visible, textarea:visible, [contenteditable='true']:visible"

_ELEMENT_INFO = """\
e => ({
  tag: e.tagName.toLowerCase(),
  text: ((e.innerText || e.value || e.textContent) || '').trim().slice(0, 60),
  aria: e.getAttribute('aria-label') || '',
  href: (e.getAttribute('href') || '').slice(0, 80),
  ph: e.getAttribute('placeholder') || ''
})"""


async def extract_candidates(
    page: Page, base: str, cap: int
) -> tuple[Locator, list[str]]:
    """Enumerate visible interactive elements as numbered lines for Laya.

    Returns the combined locator (index-aligned with the lines) and the lines
    themselves. Truncates at ``cap`` — Laya's context is 512-1024 tokens, so a
    400-element page is useless to it; the truncation is logged. An element
    that cannot be read still emits a placeholder line so indices stay aligned.
    """
    loc = page.locator(base)
    total = await loc.count()
    n = min(total, cap)
    if total > cap:
        log.warning("candidate list truncated to %d of %d elements (LAYA_MAX_CANDIDATES)", cap, total)

    lines: list[str] = []
    for i in range(n):
        el = loc.nth(i)
        try:
            info = await el.evaluate(_ELEMENT_INFO)
            desc = f"<{info['tag']}> '{info['text']}'"
            if info["aria"]:
                desc += f" aria='{info['aria']}'"
            if info["ph"]:
                desc += f" placeholder='{info['ph']}'"
            if info["href"]:
                desc += f" href={info['href']}"
        except Exception:
            desc = "<unreadable>"
        lines.append(f"{i}. {desc}")
    return loc, lines
