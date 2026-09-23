"""End an agent run that is grinding: same page, no progress, still spending.

Ported from jev-ultrafast's no-progress latch (``agent.py:153-158``) and
laya-ultrafast's narrower reading of it. The plan executor has had this latch
since it was first written (:mod:`browser_agent.recipes._plan_exec`); the
browser-use fallback did not, which is why a run with nothing left to do loops
until ``agent_max_steps`` — every step is a slow model call, and the operator
watches a Done/Search thrash for a minute.

The signal is the same semantic fingerprint the executor uses, so the two
paths cannot disagree about what "the page changed" means. Three consecutive
mutating steps over an unchanged page stop the run; a step that changes
nothing *and* was told to wait does not count, because waiting for a loading
page is the one legitimate no-op.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

#: How many mutating no-change steps in a row mean "grinding". Three, matching
#: jev and the plan executor: two is still a slow page settling, four wastes a
#: model call to learn nothing new.
STALL_LIMIT = 3

#: browser-use action names that wait rather than mutate. A wait is not
#: evidence of grinding — it is the action for a page that has not finished.
_WAIT_ACTIONS = {"wait", "scroll"}

#: Actions that only ask questions. They cost a model call like any other step,
#: so a run that is only asking is still wasting money — but it cannot be
#: "no change" in the mutation sense, and jev counts them for exactly that
#: reason: a model that re-reads the page forever is also stuck.
_READ_ACTIONS = {"extract_content", "read_file", "search", "find_text", "go_back", "switch_tab"}


def step_mutating(action: Any) -> bool:
    """Whether a browser-use action was expected to change the page.

    Accepts either an action result or its already-extracted name — the caller
    reads the name out of ``last_result`` to build the feed line, and asking it
    to keep the objects too just to answer this would invite the two readings to
    disagree.

    An action we do not recognise counts as mutating: the latch must fire on
    the loops we have seen (repeated clicks) rather than be defeated by a new
    action name browser-use adds.
    """
    if isinstance(action, str):
        name = action
    else:
        name = getattr(action, "name", None)
        if name is None:
            name = getattr(action, "action", None)
        if callable(name):  # some versions expose a method, not an attribute
            try:
                name = name()
            except Exception:
                name = None
    if not isinstance(name, str):
        return True
    return name.strip().lower() not in _WAIT_ACTIONS | _READ_ACTIONS


class AgentStalled(RuntimeError):
    """The agent is spending model calls without changing the page.

    A ``RuntimeError`` subclass on purpose: the caller already treats an
    exception from ``on_step_start`` as a run-ending error (that is how Stop
    and Amend work), and a stall is a failure a human need not act on.
    """


class StallLatch:
    """Accumulate per-step page fingerprints and declare a stall.

    Owns only the counting; reading the page and raising live with the caller.
    Split out so the counting is testable without a browser or browser-use.
    """

    def __init__(self, limit: int = STALL_LIMIT) -> None:
        self.limit = limit
        self.no_change = 0
        self.steps = 0

    def observe(self, fingerprint: str | None, mutating: bool) -> bool:
        """Fold one finished step in. Returns True when the run is stalled.

        ``fingerprint is None`` means the page could not be read — no evidence
        either way, so the counter is held, never incremented and never reset.
        ``mutating=False`` (a wait) is likewise held: a wait is not progress,
        and it is not progress's absence.
        """
        self.steps += 1
        if not mutating or fingerprint is None:
            return self.no_change >= self.limit
        if fingerprint == getattr(self, "last_fingerprint", None):
            self.no_change += 1
        else:
            self.no_change = 0
        self.last_fingerprint = fingerprint
        return self.no_change >= self.limit

    def should_stop(self) -> bool:
        return self.no_change >= self.limit


def looping_question() -> str:
    """The noul question Laya is actually built to answer.

    This is the ``harness`` preset's own question, which is the one thing the
    decision model is calibrated for here — unlike the open "what should the
    browser do next?" head, which measured at chance on this hardware.
    """
    return "Is the agent stuck repeating the same failed action with no page change?"


async def confirm_stall(settings: Any, last_result: Any) -> bool:
    """Ask Laya whether the no-progress pattern really is a stall.

    Best-effort by construction: no gate, no answer, or a low-confidence
    answer all mean "not confirmed", and the latch then rests on the
    deterministic fingerprint count alone. Never raises — a broken gate must
    not decide whether a run continues.
    """
    if not getattr(settings, "laya_enabled", False):
        return False
    try:
        from .laya_gate import LayaGate

        gate = LayaGate(settings)
        p, conf = await gate.yes_no(looping_question(), _history_text(last_result))
    except Exception:
        log.debug("stall confirmation gate failed", exc_info=True)
        return False
    if p is None or conf < settings.laya_min_confidence:
        return False
    return bool(p)


def _history_text(last_result: Any) -> str:
    """A short, honest description of what the last steps did.

    Laya's ``noul`` head reads a state string; the relevant state here is what
    the agent keeps doing, not the page, so the repeated action is the input.
    """
    parts: list[str] = []
    try:
        for r in last_result or []:
            name = getattr(r, "name", None) or getattr(r, "action", None)
            if not name:
                continue
            label = name() if callable(name) else name
            err = getattr(r, "error", None)
            parts.append(f"{label} (failed: {err})" if err else str(label))
    except Exception:
        return "agent repeated the same action with no page change"
    if not parts:
        return "agent repeated the same action with no page change"
    return f"Recent agent actions, all leaving the page unchanged: {', '.join(parts)}"
