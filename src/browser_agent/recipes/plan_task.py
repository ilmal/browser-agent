"""The plan.task recipe: planner once, deterministic executor, Laya gate.

Payload:
  task — the freeform instruction (required; ``text`` accepted as an alias,
  which is what the admin UI's Instructions box sends)

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
from ..picker import ElementPicker
from ..plan_model import PlanRejected, parse_plan
from ..planner import PROMPT, PlannerClient
from ..tasks import register_builtin
from ._config import cfg
from ._plan_exec import run_plan

log = logging.getLogger(__name__)


class PlanTask:
    name = "plan.task"
    description = (
        "Freeform: an LLM plans once, a deterministic executor runs the steps, "
        "Laya picks elements and confirms."
    )

    @property
    def entry_url(self) -> str:
        # "about:blank" is the built-in: plan.task plans its own navigation, and
        # this value is only ever the agent fallback's starting point.
        return cfg("plan.task", "entry_url", "about:blank")

    def __init__(self, planner: PlannerClient | None = None, laya: LayaGate | None = None,
                 settings: Any = None, picker: ElementPicker | None = None) -> None:
        # Injectable for tests; lazily built from the environment otherwise, so
        # importing this module never pays for model or client construction.
        self._planner = planner
        self._laya = laya
        self._settings = settings
        self._picker = picker

    def _deps(self) -> None:
        if self._settings is None:
            self._settings = load_settings()
        if self._planner is None:
            self._planner = PlannerClient(self._settings)
        if self._laya is None:
            self._laya = LayaGate(self._settings)
        if self._picker is None:
            self._picker = ElementPicker(self._settings)

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        self._deps()
        # The admin UI's Instructions box lands in payload["text"] for every
        # recipe, so accept it as an alias — a plan.task typed into the normal
        # form must not fail for field-name reasons (seen live 2026-09-21).
        task_text = str(payload.get("task") or payload.get("text") or "").strip()
        if not task_text:
            raise PlanRejected("plan.task needs a 'task' in the payload")

        raw = await self._planner.plan(
            task_text, prompt=cfg("plan.task", "planner_prompt", PROMPT)
        )
        plan = parse_plan(raw, max_steps=self._settings.planner_max_steps)
        log.info(
            "plan for %r: %d step(s), entry %s",
            task_text[:80], len(plan.steps), plan.entry_url,
        )
        return await run_plan(session, plan, task_text, self._settings, self._laya,
                              picker=self._picker)


register_builtin(PlanTask())
