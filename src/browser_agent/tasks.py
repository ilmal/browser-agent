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
import contextlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
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
    #: The line of work this attempt belongs to. A first submit is its own
    #: thread; a steer or a steer-then-retry stays in it, which is what turns
    #: "one block, then the retries" into a single conversation instead of a
    #: stack of unrelated History rows.
    thread_id: str = ""
    #: 1 for the first attempt, +1 per follow-up.
    attempt: int = 1
    parent_id: str | None = None
    #: The activity log, snapshotted when the attempt ends. ``_run`` resets the
    #: live log per task, so without this a finished attempt's feed is gone and
    #: the thread has nothing to show for what it actually tried.
    activity: list[dict[str, Any]] = field(default_factory=list)
    #: Whether a message on this attempt would change what it does. ``None``
    #: means "ask the registry" and is resolved once in ``__post_init__``; an
    #: attempt restored from the archive passes the stored answer instead, so a
    #: recipe edited or deleted since cannot rewrite what an old run was.
    reads_instruction: bool | None = None

    def __post_init__(self) -> None:
        if not self.thread_id:
            self.thread_id = self.id
        if self.reads_instruction is None:
            self.reads_instruction = recipe_reads_instruction(self.recipe)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "recipe": self.recipe,
            # Whether a message on this attempt would change anything, so the
            # thread panel can promise the right thing before the operator types.
            "reads_instruction": bool(self.reads_instruction),
            # Carried so the thread can show what each attempt was actually
            # asked to do — an attempt's instruction is the one thing the
            # operator needs in order to tell two attempts apart.
            "payload": self.payload,
            "status": self.status.value,
            "detail": self.detail,
            "result": self.result,
            "used_agent": self.used_agent,
            "amended_count": self.amended_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "thread_id": self.thread_id,
            "attempt": self.attempt,
            "parent_id": self.parent_id,
            "activity": self.activity,
        }


class Recipe(Protocol):
    """A deterministic recipe. Raise to hand the task to the agent fallback."""

    name: str
    #: Human-facing description, surfaced in the admin UI.
    description: str
    #: Starting URL, used when the agent fallback has to take over.
    entry_url: str
    #: Whether ``run`` reads the caller's instruction at all. False for the
    #: thin recipes and the game, which do the same thing however they are
    #: worded — so an operator message asking for something different has to
    #: become an agent run instead (see ``/api/tasks/{id}/say``).
    reads_instruction: bool

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


def _outcome_text(task: Task) -> str | None:
    """The bot's one-line report for a finished attempt.

    None for a status that is not an outcome — a worker cancelled mid-run
    leaves its task RUNNING, and a pod shutdown is not something to speak.
    Normalised the way ThreadStore.say stores text (whitespace collapsed) so
    the dedupe compares against what is actually in the thread.
    """
    if task.status is TaskStatus.DONE:
        body = ""
        if task.result:
            if isinstance(task.result, dict):
                body = json.dumps(task.result, ensure_ascii=False, default=str)
            else:
                body = str(task.result)
        body = body or task.detail or "finished"
        text = f"Done: {body}"
    elif task.status is TaskStatus.BLOCKED:
        text = f"I'm blocked and need you: {task.detail or 'a challenge I cannot pass'}"
    elif task.status is TaskStatus.FAILED:
        detail = task.detail or "no detail"
        if "stopped by the operator" in detail:
            # The operator's own decision is not a failure to report.
            return "Stopped at your request."
        text = f"Failed: {detail}"
    else:
        return None
    return " ".join(text.split())[:300]

_REGISTRY: dict[str, Recipe] = {}

#: Names registered by importing the built-in recipe modules, in registration
#: order. Everything else in the registry came from the operator's library, and
#: the split is what gives ``list_recipes`` an honest ``origin``.
_BUILTIN_NAMES: set[str] = set()

#: Names this process installed *from* the operator's library. Tracked rather
#: than inferred as "every name that is not a built-in": a recipe registered
#: directly through ``register()`` — a test double, an embedding caller — is
#: nobody's to forget, and inferring would delete it on the next library read.
_STORED_NAMES: set[str] = set()


def register(recipe: Recipe) -> Recipe:
    _REGISTRY[recipe.name] = recipe
    return recipe


def register_builtin(recipe: Recipe) -> Recipe:
    _BUILTIN_NAMES.add(recipe.name)
    return register(recipe)


def get_recipe(name: str) -> Recipe:
    _load_library()
    if name not in _REGISTRY:
        raise KeyError(f"unknown recipe {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def get_recipe_or_none(name: str) -> Recipe | None:
    """The registered recipe, or None. Never raises for a name we do not have."""
    _load_library()
    return _REGISTRY.get(name)


def recipe_reads_instruction(name: str) -> bool:
    """Whether ``name`` honours its payload's instruction.

    False for a name that is not in the registry at all — a stored recipe the
    operator deleted while a thread still references it. Absent is not
    instruction-reading, and this must never raise: it is called from
    ``Task.to_dict``, which every list and thread response goes through.
    """
    recipe = _REGISTRY.get(name)
    return bool(getattr(recipe, "reads_instruction", False))


def iter_recipes() -> list[Recipe]:
    """Every registered recipe, in registration order.

    The order is the contract the router relies on: built-ins register by
    importing ``recipes``, so it is stable across processes, and a stored
    recipe registers after them. Anything that claims an instruction *first*
    wins, which is why this returns the live objects rather than the
    ``list_recipes`` dicts — the claim is a method, not a field.
    """
    _load_library()
    return list(_REGISTRY.values())


def list_recipes() -> list[dict[str, str]]:
    _load_library()
    return [
        {
            "name": r.name,
            "description": r.description,
            "entry_url": r.entry_url,
            "origin": "stored" if r.name in _STORED_NAMES else "builtin",
            # Surfaced so the thread panel can promise the right thing: on a
            # recipe that is told nothing, a message redirects the attempt to
            # the agent rather than re-running the same deterministic code.
            "reads_instruction": bool(getattr(r, "reads_instruction", False)),
        }
        for r in _REGISTRY.values()
    ]


def _load_library() -> None:
    """Install the operator's recipes into this registry, once per change.

    Imported here rather than at module scope on purpose: ``recipes.stored`` and
    ``recipe_store`` both import *this* module, so a top-level import would be a
    cycle. The library's own loader is idempotent and swallows a malformed file,
    which matters because this runs under a request and, at boot, under the
    process that serves the control plane.
    """
    try:
        from .recipes.stored import load_stored_recipes

        load_stored_recipes()
    except Exception:  # a broken library must never cost us the registry
        log.exception("could not load the recipe library")


def install_stored(spec: dict) -> Recipe:
    """Register (or refresh) one stored recipe from an already-validated spec."""
    from .recipes.stored import StoredRecipe

    existing = _REGISTRY.get(spec["name"])
    if isinstance(existing, StoredRecipe):
        existing.replace(spec)
        _STORED_NAMES.add(spec["name"])
        return existing
    recipe = StoredRecipe(spec)
    _REGISTRY[recipe.name] = recipe
    _STORED_NAMES.add(recipe.name)
    return recipe


def stored_recipe_names() -> set[str]:
    """Names this process installed from the operator's library."""
    return set(_STORED_NAMES)


def forget_recipe(name: str) -> bool:
    """Drop a stored recipe, so deleting it from the library takes effect."""
    if name not in _STORED_NAMES or name not in _REGISTRY:
        return False
    _STORED_NAMES.discard(name)
    del _REGISTRY[name]
    return True


AgentRunner = Callable[[BrowserSession, str, dict[str, Any]], Awaitable[dict[str, Any]]]


class TaskRunner:
    """Serialises tasks for one profile and owns the browser session."""

    def __init__(
        self,
        settings: Settings,
        session: BrowserSession,
        *,
        agent_runner: AgentRunner | None = None,
        runs: Any = None,
        threads: Any = None,
        active_account: Callable[[], str] | None = None,
    ) -> None:
        self.settings = settings
        self.session = session
        self.llm = LLMClient(settings)
        self._agent_runner = agent_runner
        # The durable run archive, or None. Optional so a test — and any caller
        # that only wants the live behaviour — is not forced to build one; a
        # missing archive costs the record, never the run.
        self.runs = runs
        # The thread store, so a finished attempt can speak its outcome into
        # the conversation. Optional for the same reason as `runs`: a missing
        # store costs the message, never the run.
        self.threads = threads
        # Where "which account should be running" comes from. A callable rather
        # than a value because the answer changes while the pod lives, and a
        # callable rather than a direct import so a test can run the runner with
        # no account store at all.
        self._active_account = active_account
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
        # Held for the duration of a task and, separately, for an account
        # switch. This is what makes "one account runs at a time" true rather
        # than merely intended: a switch cannot land between two steps of a
        # running task, and a task cannot start against a profile being swapped
        # underneath it. Chrome's user-data-dir is single-writer, so the failure
        # this prevents is a corrupted profile, not just a confusing log line.
        self._run_lock = asyncio.Lock()

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

    def submit(
        self,
        recipe: str,
        payload: dict[str, Any],
        *,
        thread_id: str = "",
        attempt: int = 1,
        parent_id: str | None = None,
    ) -> Task:
        get_recipe(recipe)  # fail fast on an unknown recipe
        task = Task(
            recipe=recipe,
            payload=payload,
            thread_id=thread_id,
            attempt=attempt,
            parent_id=parent_id,
        )
        self.tasks[task.id] = task
        self.queue.put_nowait(task)
        log.info(
            "queued task %s recipe=%s thread=%s attempt=%d",
            task.id, recipe, task.thread_id, task.attempt,
        )
        return task

    def retry(
        self,
        task_id: str,
        *,
        payload: dict[str, Any] | None = None,
        recipe: str | None = None,
    ) -> Task:
        """Re-queue an existing task as the next attempt in its thread.

        Only legal once a human has cleared it. ``payload`` overrides the stored
        one, which is how an edited instruction becomes a real new plan instead
        of a re-run of the old text.

        ``recipe`` lets an attempt change *how* it runs, not just what it is
        told. A deterministic recipe has no prose path — it does the same thing
        to the same site however you word the instruction — so an operator who
        says "don't use that site" needs the attempt to become an agent run,
        which does read the instruction. Without this, every message to a
        blocked task re-ran the identical recipe and hit the identical wall.

        The new attempt inherits the thread so the conversation is one row in
        the History, and is briefed with what the earlier attempts tried — an
        "iterate on it" that does not carry the prior failure forward is just a
        second identical roll of the dice.
        """
        old = self.tasks[task_id]
        brief = self._thread_brief(old)
        base = payload if payload is not None else old.payload
        if brief:
            base = {**base, "history": brief}
        return self.submit(
            recipe or old.recipe,
            base,
            thread_id=old.thread_id,
            attempt=old.attempt + 1,
            parent_id=old.id,
        )

    def resubmit(
        self,
        *,
        thread_id: str,
        attempt: int,
        recipe: str,
        payload: dict[str, Any],
        parent_id: str,
    ) -> Task:
        """Queue the next attempt in a thread whose earlier attempt is gone.

        ``retry`` needs the old ``Task`` object, and a task object does not
        survive the pod recreate a deploy performs — but the *record* of it does,
        and "say what to do differently" is most useful on an old run. So the
        archive can hand its own fields here and get a real attempt in the same
        thread. The brief comes from whatever is still in memory for the thread,
        which after a restart is the archived attempts the caller has already
        folded into ``payload["history"]``.
        """
        brief = self._thread_brief_for(thread_id)
        base = {**payload, "history": brief} if brief else payload
        return self.submit(
            recipe,
            base,
            thread_id=thread_id,
            attempt=attempt + 1,
            parent_id=parent_id,
        )

    def _thread_brief_for(self, thread_id: str) -> str:
        """``_thread_brief`` for a thread named by id rather than by a task."""
        task = next((t for t in self.tasks.values() if t.thread_id == thread_id), None)
        return self._thread_brief(task) if task is not None else ""

    def _thread_brief(self, task: Task) -> str:
        """What the earlier attempts in this thread tried, and why they stopped.

        Short on purpose: it is prepended to a prompt whose budget the actual
        instruction needs. The recipe, status and detail carry the signal; the
        last few feed lines say where it got stuck.
        """
        earlier = sorted(
            (t for t in self.tasks.values() if t.thread_id == task.thread_id),
            key=lambda t: t.created_at,
        )
        lines: list[str] = []
        for t in earlier:
            lines.append(f"- attempt {t.attempt} ({t.recipe}) ended {t.status.value}: "
                         f"{t.detail or 'no detail'}")
            for entry in t.activity[-3:]:
                lines.append(f"    · {entry.get('text', '')}")
        if not lines:
            return ""
        return (
            "Earlier attempts at this same task, for context. Do not repeat a "
            "step that already failed this way:\n" + "\n".join(lines)
        )

    # -- live control ------------------------------------------------------

    def account_switch(self) -> AbstractContextManager[None]:
        """Hold the run lock while the running account is changed.

        The caller swaps the browser onto another account inside this block. The
        lock is the whole point: without it a switch could land between two steps
        of a running task, handing a live agent a different Chrome profile than
        the one it launched on — and two writers on one user-data-dir corrupt it.
        Waiting for the current task to finish is the trade: a switch is a
        deliberate, occasional act and the operator can see the queue.
        """
        return self._run_lock

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
                async with self._run_lock:
                    await self._run(task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # never let one task kill the worker
                task.status = TaskStatus.FAILED
                task.detail = f"unhandled: {exc}"
                log.exception("task %s crashed", task.id)
            finally:
                task.finished_at = time.time()
                # Snapshot BEFORE the next task resets the live log: `_run`
                # clears it, so a finished attempt's feed would otherwise be
                # unrecoverable and the thread could not show what it tried.
                with contextlib.suppress(Exception):
                    task.activity = self.activity.as_list()
                # Archive it. After the snapshot, so the record carries the full
                # feed, and after the status is final, so the record is the
                # outcome and not a mid-flight guess. Best-effort by design: the
                # run is already over, and a store failure must not surface as a
                # task failure.
                self._archive(task)
                # The bot's own turn in the conversation it was given: a run
                # that ends says so in its thread, not only in History.
                self._speak_outcome(task)
                self.current = None
                self.queue.task_done()

    def _archive(self, task: Task) -> None:
        if self.runs is None:
            return
        with contextlib.suppress(Exception):
            self.runs.save(task)

    def _speak_outcome(self, task: Task) -> None:
        """Post the attempt's outcome into its thread as a bot message.

        The operator reads a thread as a conversation, so a run that ends says
        so there instead of leaving the outcome only in History. Deduped: the
        worker can pass one terminal task through here more than once (a
        cancelled worker re-runs its ``finally``), and the tail message is
        what the room UI reads. Best-effort like the archive — the run is
        already over, and a broken store must not surface as a new failure.
        """
        if self.threads is None:
            return
        try:
            text = _outcome_text(task)
            if text is None:
                return
            recent = self.threads.for_thread(task.thread_id)
            if recent and recent[-1].role == "bot" and recent[-1].text == text:
                return
            self.threads.say(task.thread_id, "bot", "note", text)
        except Exception:
            log.debug("outcome message for %s not posted", task.id, exc_info=True)

    async def _run(self, task: Task) -> None:
        # Bind the account every task, not once at boot. The store's `active` is
        # the operator's choice and it changes while the pod lives; the session
        # is created before the store is read, so on a pod that restarted after
        # a switch the two would otherwise disagree — the file saying "work"
        # while the browser was launched on "default", which is a task quietly
        # running as the wrong identity. Reading it here also means the
        # reconciliation happens under the run lock, so it cannot race a switch.
        await self._bind_account()
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

    async def _bind_account(self) -> None:
        """Make the session match the account the store says is active.

        Restarting only when the account actually differs keeps the common case
        free: a bot with one account never relaunches, so this changes nothing
        for every existing deployment. A failure to relaunch is logged and left
        for the task itself to surface — a task is a better place to report a
        broken browser than a background reconciliation.
        """
        if self._active_account is None:
            return
        try:
            want = self._active_account()
        except Exception:
            log.warning("could not read the active account", exc_info=True)
            return
        if not want or want == self.session.account:
            return
        log.info("task binding to account %s (was %s)", want, self.session.account)
        self.session.set_account(want)
        await self.session.stop()
        try:
            await self.session.start()
        except Exception:
            log.exception("browser could not start on account %s", want)

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

        try:
            page = await self.session.goto(recipe.entry_url)
        except Exception as exc:
            # `goto` already retried a transient egress failure. Still failing
            # means the hop is down for good, which the agent fallback cannot fix
            # either — it egresses the same tunnel. Fail legibly instead of
            # escaping to the worker as "unhandled", which reads like a crash.
            task.status = TaskStatus.FAILED
            task.detail = f"entry page unreachable: {exc}"
            log.warning("entry page for %s unreachable: %s", task.recipe, exc)
            return

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
            # A learned recipe replays through here (it is a StoredRecipe, not
            # plan.task), so its promotion is counted on this path.
            self._note_replay(recipe, ok=True)
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
            self._note_replay(recipe, ok=False)
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
        self,
        task: Task,
        url: str,
        payload: dict[str, Any],
        *,
        prefix: str,
        propagate_amendment: bool = False,
    ) -> None:
        """Run the agent fallback and record the outcome on the task.

        Shared by every path that reaches the agent, so the Cancelled /
        challenge handling is identical whichever way it got here.

        An amendment is *re-raised* when ``propagate_amendment`` is set: only the
        caller knows what "the instruction changed" means for it. A freeform task
        re-runs the agent, a plan re-plans; the generic fallback has nothing to
        re-interpret, so it fails there and asks for a Retry.
        """
        try:
            task.result = await self._agent_runner(self.session, url, payload)
            task.used_agent = True
            task.status = TaskStatus.DONE
            task.detail = f"{prefix} succeeded"
            self._maybe_learn(task.result)
        except EscalationRequired as exc:
            await self._block(task, exc.challenge)
        except Cancelled:
            task.status = TaskStatus.FAILED
            task.detail = "stopped by the operator"
            self.activity.note("error", "stopped by the operator")
        except Amended:
            if propagate_amendment:
                raise
            task.status = TaskStatus.FAILED
            task.detail = f"{prefix}: instruction changed mid-run; re-plan needed"
            self.activity.note(
                "info", "instruction changed mid-run; press Retry to re-plan it"
            )
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.detail = f"{prefix} failed: {exc}"
            log.error("%s failed for %s: %s", prefix, task.id, exc)

    def _note_replay(self, recipe: Any, *, ok: bool) -> None:
        """Count one replay of a learned recipe. Never raises.

        Only learned recipes are counted: a built-in's clean run says nothing
        about a plan, and an operator's stored recipe is trusted when they
        wrote it. This is the guardrail that makes auto-routing safe — the
        recipe that answers a later request is one that has already been
        replayed against a real page, not one harvested from a single run.
        """
        if getattr(recipe, "origin", "") != "learned":
            return
        from .recipe_store import learned_store_for, record_replay

        store = learned_store_for(self.settings)
        if store is None:
            return
        record_replay(store, recipe.name, ok=ok)

    def _maybe_learn(self, result: Any) -> None:
        """Persist a recipe the agent just earned, if the runner harvested one.

        The runner stays ignorant of the store (a plain dict goes into its
        result), so the write lives here. Never raises: a spec that cannot be
        stored costs a speedup, never the task that produced it.
        """
        if not isinstance(result, dict):
            return
        spec = result.pop("learned_recipe", None)
        if not isinstance(spec, dict):
            return
        from .recipe_store import RecipeError, learned_store_for, save_learned_spec

        store = learned_store_for(self.settings)
        if store is None:
            return
        try:
            save_learned_spec(store, spec)
            log.info("learned recipe stored: %s", spec.get("name"))
            self.activity.note(
                "info",
                f"learned a recipe for this task: {spec.get('name')} "
                f"(needs {store.learned_min_replays()} clean replays before it is reused)",
            )
        except RecipeError as exc:
            # Not every successful run is representable as a plan (a scroll, a
            # file upload, a target with no durable selector). Losing the
            # recipe is the intended outcome, not an error to report.
            log.info("agent run is not replayable as a recipe: %s", exc)

    def _apply_amendment(self, task: Task, exc: Amended) -> None:
        """Adopt the operator's new instruction as the task's own text.

        Written into the payload so a retry does not silently revert to the
        original wording — the task the operator is watching is the amended one.
        """
        task.amended_count += 1
        task.payload = {**task.payload, "task": exc.instruction, "text": exc.instruction}
        # The agent reads `goal` first, so leaving a stale one would make the
        # amendment look ignored.
        task.payload["goal"] = exc.instruction
        self.activity.note(
            "info", f"instruction changed; restarting with it ({task.amended_count})"
        )

    async def _run_freeform(self, task: Task, recipe: Recipe) -> None:
        """Run a task that is only an instruction, with no deterministic path.

        An amendment restarts the agent with the new instruction: there is no
        step list to repair, so "carry on with different words" is the whole
        behaviour. The browser is left where the previous attempt stopped, which
        is what makes a mid-run correction cheap — the site state is intact.
        """
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

        for _attempt in range(MAX_AMENDMENTS + 1):
            # Only the first attempt navigates: a mid-run amendment must not
            # throw away the page the operator is watching.
            if not task.amended_count:
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
            try:
                await self._agent_attempt(
                    task, url, task.payload, prefix="agent", propagate_amendment=True
                )
                return
            except Amended as exc:
                self._apply_amendment(task, exc)
                self.session.control.amendments.clear()
                continue

        task.status = TaskStatus.FAILED
        task.detail = f"too many amendments ({MAX_AMENDMENTS}) without a completed run"

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
                self._note_replay(recipe, ok=True)
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
                self._apply_amendment(task, exc)
                if exc.url and exc.url.startswith("http"):
                    task.payload["entry_url"] = exc.url
                self.session.control.amendments.clear()
                continue
            except StepFailure as exc:
                # The plan was fine, the page disagreed. Hand the job to the
                # agent rather than retrying the broken step blind.
                self._note_replay(recipe, ok=False)
                return await self._agent_attempt(
                    task,
                    exc.entry_url or recipe.entry_url,
                    {**task.payload, "goal": exc.goal or _task_text(task)},
                    prefix="agent fallback",
                )
            except Exception as exc:
                log.warning("plan.task failed for %s: %s", task.id, exc)
                self._note_replay(recipe, ok=False)
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
