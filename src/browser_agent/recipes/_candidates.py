"""Page candidate extraction for the plan.task element pickers.

The page's interactive elements are enumerated as numbered lines. The
numbering IS the contract: a picker returns a line index and the executor
binds ``locator.nth(binds[index])`` — never a re-match on text, which two
identical buttons would make ambiguous.

Three properties are load-bearing, all ported from the System-1 harness
survey (jev-ultrafast's snapshot/question shape, jev-browser-skill's label
cascade, SystemOneHarness's "one act, one name", laya-browser's
current-value term):

* **What the model saw is what nth() binds.** Filtering (visibility,
  disabled, nested duplicates) happens in one in-page pass that records each
  kept element's position in the full match order; the returned ``binds``
  list maps every line to that position. If we hid an element from the model
  but left ``nth(i)`` counting all matches, the binding would silently point
  somewhere else.
* **The line carries the control's meaning**: role, accessible label (a
  cascade — aria-labelledby, aria-label, wrapping label, button value, alt,
  title, inner text, placeholder — mirroring how a human names a control),
  its current value, and state flags (``[checked]``, ``[expanded]``). A
  bare ``<button> ''`` line is what the survey measured as unanswerable.
* **One act, one name.** A control whose nearer ancestor is itself a
  candidate (``<div role=button><a href>``) is dropped as a duplicate of the
  same actionable thing.

Excluded everywhere: ``input[type=file]`` (upload flows go to the agent
fallback) and ``input[type=hidden]``. Password fields stay enumerable — the
typed text comes from the plan, never from the picker, so naming the field
is safe — but their value is never shown in a line.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from playwright.async_api import Locator, Page

log = logging.getLogger(__name__)

#: Candidates shown to the LLM picker. delimiters: laya only ever sees the
#: first ``LAYA_MAX_CANDIDATES`` (its calibrated choice range), but the LLM
#: picker is not bucket-limited, and a target outside the old flat cap of 10
#: used to fall to the agent fallback for no better reason than page size.
PICKER_CAP = 40

CLICK_BASE = (
    "a[href], button, summary, input, [onclick], "
    "[role='button'], [role='link'], [role='menuitem'], [role='tab'], "
    "[role='option'], [role='checkbox'], [role='radio'], [role='switch'], "
    "[role='treeitem']"
)
TYPE_BASE = "input, textarea, [contenteditable='true'], [role='textbox'], [role='searchbox']"

#: One in-page pass over every CSS match, returning the match indexes worth
#: showing. ``null`` marks a drop, with the reason logged only in aggregate:
#: not visible (display/visibility/zero rect/aria-hidden/inert ancestor),
#: disabled, an excluded input type, or a nearer candidate ancestor (one act,
#: one name).
_ENUM_JS = """(args) => {
  const [sel, mode] = args;
  const alwaysExcluded = ['hidden', 'file'];
  const typeExcluded = (t) =>
    alwaysExcluded.includes(t) ||
    (mode === 'type'
      ? ['button', 'submit', 'reset', 'image', 'checkbox', 'radio', 'range',
         'color', 'date', 'datetime-local', 'month', 'week', 'time'].includes(t)
      : t === 'password');
  const visible = (e) => {
    const cs = getComputedStyle(e);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.visibility === 'collapse')
      return false;
    if (e.getAttribute('aria-hidden') === 'true' || e.closest('[inert]')) return false;
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const out = [];
  document.querySelectorAll(sel).forEach((e, i) => {
    const t = (e.getAttribute('type') || '').toLowerCase();
    if (e.disabled || e.getAttribute('aria-disabled') === 'true') { out.push(null); return; }
    if (typeExcluded(t)) { out.push(null); return; }
    if (!visible(e)) { out.push(null); return; }
    let a = e.parentElement, d = 0;
    while (a && d < 3) {
      if (a.matches && a.matches(sel)) { out.push(null); return; }
      a = a.parentElement; d += 1;
    }
    out.push(i);
  });
  return out;
}"""

ELEMENT_INFO = """\
e => {
  const tag = e.tagName.toLowerCase();
  const type = (e.getAttribute('type') || '').toLowerCase();
  let label = '';
  const ids = e.getAttribute('aria-labelledby');
  if (ids) {
    label = ids.trim().split(/\\s+/).slice(0, 3).map(
      id => { const n = document.getElementById(id);
              return n ? (n.innerText || n.textContent || '') : ''; }
    ).join(' ').replace(/\\s+/g, ' ').trim();
  }
  if (!label) label = (e.getAttribute('aria-label') || '').trim();
  if (!label) { const l = e.closest('label');
                if (l) label = (l.innerText || '').replace(/\\s+/g, ' ').trim(); }
  if (!label && tag === 'input' && ['submit', 'button', 'reset'].includes(type))
    label = (e.value || '').trim();
  if (!label) label = (e.getAttribute('alt') || e.getAttribute('title') || '').trim();
  if (!label && e.innerText) label = e.innerText.replace(/\\s+/g, ' ').trim();
  if (!label) label = (e.getAttribute('placeholder') || '').trim();
  let role = e.getAttribute('role') || '';
  if (!role) {
    if (tag === 'button' || tag === 'summary') role = 'button';
    else if (tag === 'a') role = 'link';
    else if (tag === 'select') role = 'combobox';
    else if (tag === 'textarea' || e.isContentEditable) role = 'textbox';
    else if (tag === 'input')
      role = type === 'checkbox' ? 'checkbox'
           : type === 'radio' ? 'radio'
           : ['submit', 'button', 'reset'].includes(type) ? 'button' : 'textbox';
    else role = tag;
  }
  const isSecret = tag === 'input' && type === 'password';
  const value = (!isSecret && 'value' in e && e.value != null)
    ? String(e.value).slice(0, 60) : '';
  const expanded = e.getAttribute('aria-expanded');
  return {
    role,
    label: label.slice(0, 80),
    value,
    checked: !!e.checked,
    expanded: expanded === null || expanded === undefined ? '' : expanded,
    ph: (e.getAttribute('placeholder') || '').trim(),
    href: (e.getAttribute('href') || '').slice(0, 80),
    ident: tag + '|' + role + '|' + label.slice(0, 40) + '|' + value.slice(0, 30),
  };
}"""


@dataclass
class Candidates:
    """The enumeration result: lines for the model, binds for the executor.

    ``lines[j]`` describes candidate ``j``; ``binds[j]`` is its position in
    ``locator``'s full match order (bind with ``locator.nth(binds[j])``);
    ``idents[j]`` is the element identity fingerprint the freshness guard
    re-checks before dispatch.
    """

    locator: Locator
    lines: list[str]
    binds: list[int]
    idents: list[str]


def _line(j: int, info: dict) -> str:
    desc = f"{j}. {info['role']} '{info['label']}'"
    if info.get("value") and info["role"] in {"textbox", "combobox", "searchbox"}:
        desc += f" = '{info['value']}'"
    if info.get("checked"):
        desc += " [checked]"
    if info.get("expanded") != "":
        desc += f" [expanded={info['expanded']}]"
    if info.get("ph"):
        desc += f" placeholder='{info['ph']}'"
    if info.get("href"):
        desc += f" href={info['href']}"
    return desc


async def extract_candidates(
    page: Page, base: str, cap: int, *, mode: str = "click"
) -> Candidates:
    """Enumerate visible interactive elements as numbered lines.

    Returns a :class:`Candidates`. Truncates at ``cap`` — the LLM picker is
    not context-limited at 40 lines, and laya callers slice the calibrated
    first 10 themselves. Dropped elements are logged in aggregate only.
    """
    loc = page.locator(base)
    total = await loc.count()
    if total == 0:
        return Candidates(loc, [], [], [])
    try:
        keeps = await page.evaluate(_ENUM_JS, [base, mode])
    except Exception as exc:
        log.warning("candidate enumeration failed (%s); no candidates", exc)
        return Candidates(loc, [], [], [])
    kept = [i for i in keeps if i is not None]
    if len(kept) > cap:
        log.info(
            "candidate list truncated to %d of %d (base %s)", cap, len(kept), base
        )
        kept = kept[:cap]

    lines: list[str] = []
    binds: list[int] = []
    idents: list[str] = []
    for j, i in enumerate(kept):
        try:
            info = await loc.nth(i).evaluate(ELEMENT_INFO)
        except Exception:
            info = None
        if info is None:
            # Keep the slot with a placeholder line so numbering survives.
            lines.append(f"{j}. <unreadable>")
            binds.append(i)
            idents.append("")
            continue
        lines.append(_line(j, info))
        binds.append(i)
        idents.append(info["ident"])
    return Candidates(loc, lines, binds, idents)
