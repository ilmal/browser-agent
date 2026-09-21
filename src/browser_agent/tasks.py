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

from .browser import BrowserSession
from .config import Settings
from .escalation import EscalationRequired, detect_challenge, looks_logged_out
from .llm import LLMClient

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "recipe": self.recipe,
            "status": self.status.value,
            "detail": self.detail,
            "result": self.result,
            "used_agent": self.used_agent,
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

    def retry(self, task_id: str) -> Task:
        """Re-queue an existing task. Only legal once a human has cleared it."""
        old = self.tasks[task_id]
        return self.submit(old.recipe, old.payload)

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

        page = await self.session.goto(recipe.entry_url)

        # A challenge before we even start means the profile is not usable.
        challenge = await detect_challenge(page)
        if challenge is not None:
            await self._block(task, challenge)
            return

        try:
            task.result = await recipe.run(self.session, task.payload)
            task.status = TaskStatus.DONE
            task.detail = "recipe succeeded"
            return
        except EscalationRequired as exc:
            await self._block(task, exc.challenge)
            return
        except Exception as exc:
            log.warning("recipe %s failed for %s: %s", task.recipe, task.id, exc)
            task.detail = f"recipe failed: {exc}"

        # Deterministic path failed. Only now do we spend an LLM call.
        if self._agent_runner is None or not self.llm.enabled:
            task.status = TaskStatus.FAILED
            task.detail += " (no agent fallback available)"
            return

        page = await self.session.page()
        challenge = await detect_challenge(page)
        if challenge is not None:
            await self._block(task, challenge)
            return

        try:
            task.result = await self._agent_runner(self.session, recipe.entry_url, task.payload)
            task.used_agent = True
            task.status = TaskStatus.DONE
            task.detail = "agent fallback succeeded"
        except EscalationRequired as exc:
            await self._block(task, exc.challenge)
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.detail = f"agent fallback failed: {exc}"
            log.error("agent fallback failed for %s: %s", task.id, exc)

    async def _run_freeform(self, task: Task, recipe: Recipe) -> None:
        """Run a task that is only an instruction, with no deterministic path."""
        if self._agent_runner is None or not self.llm.enabled:
            task.status = TaskStatus.FAILED
            task.detail = "agent not available (LLM disabled or browser-use missing)"
            return
        try:
            task.result = await self._agent_runner(self.session, recipe.entry_url, task.payload)
            task.used_agent = True
            task.status = TaskStatus.DONE
            task.detail = "agent completed the task"
        except EscalationRequired as exc:
            await self._block(task, exc.challenge)
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.detail = f"agent failed: {exc}"
            log.error("freeform task %s failed: %s", task.id, exc)

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
