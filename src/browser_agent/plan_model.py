"""The plan schema, its validation, and the two failure types a plan can raise.

A planner is a model, so its output is untrusted input: everything here either
validates strictly or refuses loudly. The distinction that matters downstream:

* :class:`PlanRejected` — the plan (or the task payload) is unusable. This is a
  model-quality or caller bug, so the task FAILS visibly rather than being
  silently handed to the agent fallback, which would only retry the same
  prompt-shaped garbage.
* :class:`StepFailure` — the plan was fine but the page disagreed mid-run. The
  runner hands these to the agent fallback, which is the whole point of the
  cascade.

Only :class:`~browser_agent.escalation.EscalationRequired` may end in a human
page; nothing in this module can raise one.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

_HTTP_URL = re.compile(r"^https?://\S+$", re.IGNORECASE)


class PlanRejected(Exception):
    """The planner output (or payload) cannot be turned into a usable plan."""


class StepFailure(Exception):
    """A plan step failed against the real page.

    ``goal``/``entry_url`` carry the original task context so the runner can
    brief the agent fallback with what was being attempted, not just what broke.
    """

    def __init__(self, message: str, goal: str = "", entry_url: str = "") -> None:
        super().__init__(message)
        self.goal = goal
        self.entry_url = entry_url


class DoneWhen(BaseModel):
    """Deterministic proof a step did what it claimed. Authoritative over Laya."""

    model_config = ConfigDict(extra="forbid")

    url_contains: str | None = None
    selector_visible: str | None = None
    text_contains: str | None = None


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["navigate", "click", "type", "extract", "wait"]
    #: Semantic description — what this step does. Laya resolves it to an
    #: element on the real page; it is also what the confirmation gate asks about.
    goal: str = ""
    selector: str | None = None
    text: str | None = None
    done_when: DoneWhen | None = None


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_url: str
    steps: list[Step]


def strip_fences(text: str) -> str:
    """Strip markdown code fences a model wrapped around its JSON anyway."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()


def parse_plan(raw: str | dict[str, Any], *, max_steps: int | None = None) -> Plan:
    """Parse planner output into a Plan. Raises PlanRejected, never pydantic."""
    try:
        data = raw if isinstance(raw, dict) else json.loads(strip_fences(raw))
        plan = Plan.model_validate(data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise PlanRejected(f"planner output is not a valid plan: {exc}") from exc
    if max_steps is not None and not 1 <= len(plan.steps) <= max_steps:
        raise PlanRejected(f"plan has {len(plan.steps)} steps; 1..{max_steps} allowed")
    _check(plan)
    return plan


def _check(plan: Plan) -> None:
    if not _HTTP_URL.match(plan.entry_url):
        raise PlanRejected(f"entry_url is not an http(s) URL: {plan.entry_url!r}")
    for i, step in enumerate(plan.steps):
        if step.action == "navigate" and not (step.text and _HTTP_URL.match(step.text)):
            raise PlanRejected(f"step {i}: navigate needs an http(s) URL in `text`")
        if step.action == "click" and not step.goal.strip():
            raise PlanRejected(f"step {i}: click needs a `goal` to resolve on the page")
        if step.action == "type" and (not step.goal.strip() or not step.text):
            raise PlanRejected(f"step {i}: type needs a `goal` and the literal `text` to enter")
        if step.action in {"extract", "wait"} and not step.selector:
            raise PlanRejected(f"step {i}: {step.action} must be deterministic — `selector` required")
