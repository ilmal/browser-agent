"""Task model, the recipe registry, and the run loop.

Flow for one task:

    recipe (Playwright, deterministic)
        -> on failure: agent fallback (browser-use, LLM-driven)
            -> on challenge at any point: STOP, escalate to a human

Tasks run one at a time per profile. Serialising them is deliberate: a profile
is a single browser identity, and concurrent actions in one identity look
nothing like a human.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .activity import Activity
from .browser import BrowserSession
from .config import Settings
from .control import Amended, Cancelled, Control
from .escalation import EscalationRequired, detect_challenge, looks_logged_out
from .llm import LLMClient
from .plan_model import PlanRejected, StepFailure

#: How many times one task may be re-planned by an operator amendment before it
#: is refused. A person steering a run types a few times; a loop means the
#: instruction itself cannot be planned and should be surfaced, not retried.
MAX_AMENDMENTS = 5

log = logging.getLogger(__name__)


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"  # needs a human; never auto-retried


@dataclass
class Task:
    recipe: str
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: TaskStatus = TaskStatus.QUEUED
    detail: str = ""
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    used_agent: bool = False
    #: Set when the operator has changed the instruction mid-run, so the UI can
    #: show that the task it is watching is not the one it started.
    amended_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "recipe": self.recipe,
            "status": self.status.value,
            "detail": self.detail,
            "result": self.result,
            "used_agent": self.used_agent,
            "amended_count": self.amended_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class Recipe(Protocol):
    """A deterministic recipe. Raise to hand the task to the agent fallback."""

    name: str
    #: Human-facing description, surfaced in the admin UI.
    description: str
    #: Starting URL, used when the agent fallback has to take over.
    entry_url: str

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]: ...


#: Pseudo-recipe for freeform instructions. It has no deterministic path: the
#: agent is the implementation.
AGENT_RECIPE = "agent.task"

#: The planner/executor recipe. Named here rather than imported from the recipe
#: module so the registry stays the only import-time coupling.
PLAN_RECIPE = "plan.task"


def _task_text(task: Task) -> str:
    """The instruction as the caller wrote it, whichever field they used."""
    return str(task.payload.get("task") or task.payload.get("text") or "").strip()

_REGISTRY: dict[str, Recipe] = {}


def register(recipe: Recipe) -> Recipe:
    _REGISTRY[recipe.name] = recipe
    return recipe


def get_recipe(name: str) -> Recipe:
    if name not in _REGISTRY:
        raise KeyError(f"unknown recipe {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_recipes() -> list[dict[str, str]]:
    return [
        {"name": r.name, "description": r.description, "entry_url": r.entry_url}
        for r in _REGISTRY.values()
    ]


AgentRunner = Callable[[BrowserSession, str, dict[str, Any]], Awaitable[dict[str, Any]]]


class TaskRunner:
    """Serialises tasks for one profile and owns the browser session."""

    def __init__(
        self,
        settings: Settings,
        session: BrowserSession,
        *,
        agent_runner: AgentRunner | None = None,
    ) -> None:
        self.settings = settings
        self.session = session
        self.llm = LLMClient(settings)
        self._agent_runner = agent_runner
        self.queue: asyncio.Queue[Task] = asyncio.Queue()
        self.tasks: dict[str, Task] = {}
        self.current: Task | None = None
        # Live state for whatever is running now: the control set the API acts
        # on, and the activity log the UI reads. The runner owns them so a task
        # never fails because a session could not hold them (test doubles and
        # any minimal session have no such slots); they are mirrored onto the
        # session, which is how the deeper layers reach them.
        self.control: Control | None = None
        self.activity = Activity()
        self._worker: asyncio.Task | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    # -- submission --------------------------------------------------------

    def submit(self, recipe: str, payload: dict[str, Any]) -> Task:
        get_recipe(recipe)  # fail fast on an unknown recipe
        task = Task(recipe=recipe, payload=payload)
        self.tasks[task.id] = task
        self.queue.put_nowait(task)
        log.info("queued task %s recipe=%s", task.id, recipe)
        return task

    def retry(self, task_id: str, *, payload: dict[str, Any] | None = None) -> Task:
        """Re-queue an existing task. Only legal once a human has cleared it.

        ``payload`` overrides the stored one, which is how an edited instruction
        becomes a real new plan instead of a re-run of the old text.
        """
        old = self.tasks[task_id]
        return self.submit(old.recipe, payload if payload is not None else old.payload)

    # -- live control ------------------------------------------------------

    def running(self, task_id: str) -> Task | None:
        """The task with this id iff it is the one currently executing."""
        if self.current is not None and self.current.id == task_id:
            return self.current
        return None

    def pause(self) -> None:
        if self.control is not None:
            self.control.pause()

    def resume(self) -> None:
        if self.control is not None:
            self.control.resume()

    # -- run loop ----------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            task = await self.queue.get()
            try:
                await self._run(task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let one task kill the worker
                task.status = TaskStatus.FAILED
                task.detail = f"unhandled: {exc}"
                log.exception("task %s crashed", task.id)
            finally:
                task.finished_at = time.time()
                self.current = None
                self.queue.task_done()

    async def _run(self, task: Task) -> None:
        # A fresh activity log and control set per task: the runner outlives the
        # task, so without this the UI would show the previous task's steps and
        # a stale "stopped" flag would cancel the next one at step 0.
        self.activity.reset()
        self.control = Control()
        self._attach(self.activity, self.control)
        try:
            await self._run_task(task)
        finally:
            self._attach(self.activity, None)
            self.control = None

    def _attach(self, activity: Activity, control: Control | None) -> None:
        """Mirror the live state onto the session, best-effort.

        The deeper layers (the plan executor, the agent's step hooks) receive
        the session and nothing else, so the session is where this has to live.
        A session that cannot hold it — a test double, a minimal stand-in —
        must not fail the task for it.
        """
        try:
            self.session.activity = activity
            self.session.control = control
        except Exception:
            log.debug("session cannot hold live-control state", exc_info=True)

    async def _run_task(self, task: Task) -> None:
        recipe = get_recipe(task.recipe)
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.current = task

        # A freeform task has no deterministic path: the agent is the recipe,
        # so it goes straight there. This is the "just do this thing for me"
        # entry point, and it carries the same guardrails and the same
        # challenge-stop rule as the fallback path.
        if task.recipe == AGENT_RECIPE:
            await self._run_freeform(task, recipe)
            return

        # plan.task owns its own navigation and re-planning: an amendment
        # half-way through a step list must re-plan, not resume a list that no
        # longer matches the instruction.
        if task.recipe == PLAN_RECIPE:
            await self._run_plan_recipe(task, recipe)
            return

        page = await self.session.goto(recipe.entry_url)

        # A challenge before we even start means the profile is not usable.
        challenge = await detect_challenge(page)
        if challenge is not None:
            await self._block(task, challenge)
            return

        plan_failure: StepFailure | None = None
        try:
            task.result = await recipe.run(self.session, task.payload)
            task.status = TaskStatus.DONE
            task.detail = "recipe succeeded"
            return
        except Cancelled:
            task.status = TaskStatus.FAILED
            task.detail = "stopped by the operator"
            return
        except EscalationRequired as exc:
            await self._block(task, exc.challenge)
            return
        except Exception as exc:
            log.warning("recipe %s failed for %s: %s", task.recipe, task.id, exc)
            if isinstance(exc, PlanRejected):
                # A plan that never validated is a caller/model-quality bug.
                # Failing visibly beats spending an agent run on the same
                # garbage input the next planner call would likely reproduce.
                task.status = TaskStatus.FAILED
                task.detail = f"plan rejected: {exc}"
                return
            if isinstance(exc, StepFailure):
                plan_failure = exc
            task.detail = f"recipe failed: {exc}"

        # Deterministic path failed. Only now do we spend an LLM call.
        if self._agent_runner is None or not self.llm.configured:
            task.status = TaskStatus.FAILED
            task.detail += " (no agent fallback available)"
            return

        page = await self.session.page()
        challenge = await detect_challenge(page)
        if challenge is not None:
            await self._block(task, challenge)
            return

        # A half-run plan hands the agent the original task and the plan's own
        # starting URL, so it finishes the job instead of retrying the step
        # that broke.
        agent_url = recipe.entry_url
        agent_payload = task.payload
        if plan_failure is not None:
            agent_url = plan_failure.entry_url or agent_url
            agent_payload = {**task.payload, "goal": plan_failure.goal}

        await self._agent_attempt(task, agent_url, agent_payload, prefix="agent fallback")

    async def _agent_attempt(
        self, task: Task, url: str, payload: dict[str, Any], *, prefix: str
    ) -> None:
        """Run the agent fallback and record the outcome on the task.

        Shared by every path that reaches the agent, so the Cancelled /
        amendment / challenge handling is identical whichever way it got here.
        """
        try:
            task.result = await self._agent_runner(self.session, url, payload)
            task.used_agent = True
            task.status = TaskStatus.DONE
            task.detail = f"{prefix} succeeded"
        except EscalationRequired as exc:
            await self._block(task, exc.challenge)
        except Cancelled:
            task.status = TaskStatus.FAILED
            task.detail = "stopped by the operator"
            self.activity.note("error", "stopped by the operator")
        except Amended as exc:
            task.status = TaskStatus.FAILED
            task.detail = f"{prefix}: {exc}; re-plan needed"
            self.activity.note(
                "info", "instruction changed mid-run; press Retry to re-plan it"
            )
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.detail = f"{prefix} failed: {exc}"
            log.error("%s failed for %s: %s", prefix, task.id, exc)

    async def _run_freeform(self, task: Task, recipe: Recipe) -> None:
        """Run a task that is only an instruction, with no deterministic path."""
        if self._agent_runner is None or not self.llm.configured:
            task.status = TaskStatus.FAILED
            task.detail = "agent not available (llm not configured, or browser-use missing)"
            return

        # A freeform task must start somewhere. `entry_url` for this recipe is
        # about:blank, so without a caller-supplied url the agent would be handed
        # a blank page and no site to work on. The documented `url` payload was
        # never read and the session was never navigated, so freeform could only
        # ever operate on whatever page happened to be open.
        url = (task.payload.get("url") or recipe.entry_url or "").strip()
        if not url or url == "about:blank":
            task.status = TaskStatus.FAILED
            task.detail = "freeform tasks need a start url in the payload"
            return
        try:
            page = await self.session.goto(url)
            challenge = await detect_challenge(page)
            if challenge is not None:
                await self._block(task, challenge)
                return
        except Cancelled:
            task.status = TaskStatus.FAILED
            task.detail = "stopped by the operator"
            return
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.detail = f"agent failed: {exc}"
            log.error("freeform task %s failed to open %s: %s", task.id, url, exc)
            return
        await self._agent_attempt(task, url, task.payload, prefix="agent")

    async def _run_plan_recipe(self, task: Task, recipe: Recipe) -> None:
        """plan.task, with the operator's amendment able to re-plan mid-run.

        An amendment means the instruction changed, which makes the running step
        list wrong. Re-running the recipe is therefore the correct response, and
        the planner is called again with the new text — the browser is left
        where the previous attempt stopped, and the task payload carries the
        amended instruction so a retry does not silently revert to the original.
        """
        for _attempt in range(MAX_AMENDMENTS + 1):
            try:
                task.result = await recipe.run(self.session, task.payload)
                task.status = TaskStatus.DONE
                task.detail = "plan succeeded"
                return
            except Cancelled:
                task.status = TaskStatus.FAILED
                task.detail = "stopped by the operator"
                self.activity.note("error", "stopped by the operator")
                return
            except EscalationRequired as exc:
                await self._block(task, exc.challenge)
                return
            except PlanRejected as exc:
                task.status = TaskStatus.FAILED
                task.detail = f"plan rejected: {exc}"
                return
            except Amended as exc:
                # The operator's new instruction replaces the task text; the
                # amended URL, when the agent reported one, becomes the entry.
                task.amended_count += 1
                self.activity.note(
                    "info", f"instruction changed; re-planning ({task.amended_count})"
                )
                task.payload = {**task.payload, "task": exc.instruction, "text": exc.instruction}
                if exc.url and exc.url.startswith("http"):
                    task.payload["entry_url"] = exc.url
                self.session.control.amendments.clear()
                continue
            except StepFailure as exc:
                # The plan was fine, the page disagreed. Hand the job to the
                # agent rather than retrying the broken step blind.
                return await self._agent_attempt(
                    task,
                    exc.entry_url or recipe.entry_url,
                    {**task.payload, "goal": exc.goal or _task_text(task)},
                    prefix="agent fallback",
                )
            except Exception as exc:
                log.warning("plan.task failed for %s: %s", task.id, exc)
                return await self._agent_attempt(
                    task,
                    recipe.entry_url,
                    {**task.payload, "goal": _task_text(task)},
                    prefix="agent fallback",
                )
        task.status = TaskStatus.FAILED
        task.detail = f"too many amendments ({MAX_AMENDMENTS}) without a plan that ran"

    async def _block(self, task: Task, challenge) -> None:
        """Record a blocker and hand it to a human. Never retried automatically."""
        task.status = TaskStatus.BLOCKED
        task.detail = challenge.describe()
        log.warning("task %s BLOCKED: %s", task.id, task.detail)
        from .notify import notify_escalation

        await notify_escalation(self.settings, challenge, takeover_url(self.settings))

    async def logged_in(self) -> bool:
        """Whether this profile currently has a usable session."""
        page = await self.session.page()
        if page.url in {"about:blank", ""}:
            return False
        return not await looks_logged_out(page)


def takeover_url(settings: Settings) -> str:
    """Where a human opens noVNC for this profile."""
    return f"http://{settings.profile}.browser-agent.svc.cluster.local:{settings.novnc_port}/vnc.html"
