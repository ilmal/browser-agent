"""The plan.task recipe: planner once, deterministic executor, Laya gate.

Payload:
  task — the freeform instruction (required)

Failure semantics, deliberately uneven:

* missing ``task`` / planner output that fails validation →
  :class:`~browser_agent.plan_model.PlanRejected` — a caller or model-quality
  bug the runner turns into FAILED, not into an agent run against garbage.
* planner endpoint unreachable → PlannerUnavailable — the runner's normal
  exception path sends the task to the agent fallback (the planner is an
  accelerator, not a gate).
* a plan step failing mid-run → StepFailure — same fallback, briefed with the
  original task text and the plan's entry URL.
"""

from __future__ import annotations

import logging
from typing import Any

from ..browser import BrowserSession
from ..config import load_settings
from ..laya_gate import LayaGate
from ..plan_model import PlanRejected, parse_plan
from ..planner import PlannerClient
from ..tasks import register
from ._plan_exec import run_plan

log = logging.getLogger(__name__)


class PlanTask:
    name = "plan.task"
    description = (
        "Freeform: an LLM plans once, a deterministic executor runs the steps, "
        "Laya picks elements and confirms."
    )
    entry_url = "about:blank"

    def __init__(self, planner: PlannerClient | None = None, laya: LayaGate | None = None,
                 settings: Any = None) -> None:
        # Injectable for tests; lazily built from the environment otherwise, so
        # importing this module never pays for model or client construction.
        self._planner = planner
        self._laya = laya
        self._settings = settings

    def _deps(self) -> None:
        if self._settings is None:
            self._settings = load_settings()
        if self._planner is None:
            self._planner = PlannerClient(self._settings)
        if self._laya is None:
            self._laya = LayaGate(self._settings)

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        self._deps()
        task_text = str(payload.get("task") or "").strip()
        if not task_text:
            raise PlanRejected("plan.task needs a 'task' in the payload")

        raw = await self._planner.plan(task_text)
        plan = parse_plan(raw, max_steps=self._settings.planner_max_steps)
        log.info(
            "plan for %r: %d step(s), entry %s",
            task_text[:80], len(plan.steps), plan.entry_url,
        )
        return await run_plan(session, plan, task_text, self._settings, self._laya)


register(PlanTask())
