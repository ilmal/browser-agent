"""Farms: one task assigned to several bots, started together or on a ladder.

A farm is the hub-side answer to "run this recipe as each of these five
identities, in parallel, staggered, or at ten o'clock". The hub owns it because
the assignment outlives any one pod: the operator composes it on the roster
page, the hub fires each member's own bot API when its start time comes, and
the farm's record — who was assigned what, what came back — is the room the
operator later chats in.

Deliberately *not* the per-bot scheduler (scheduler.py): that is a cron table
inside one pod, evaluated by that pod. A farm needs a coordinator that can
reach every pod, which is the hub, and so the start times are absolute epochs
computed once at creation — a hub restart re-fires whatever is due, which is
the recovery you want, and there is no cron grammar to get wrong.

The store mirrors registry.py: one JSON file on the hub volume, atomic
replace-write, degrade-to-empty when corrupt. The runner is one asyncio loop
whose every failure mode is a member status, never an exception out.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from . import registry

log = logging.getLogger("browser_agent.farms")

#: A member status is one of: waiting for its start time, fired and queued on
#: the pod, observed running, or finished one way or another. ACTIVE is the set
#: a farm is still waiting on; everything else is terminal.
ACTIVE = ("pending", "started", "running")
TERMINAL = ("done", "failed", "blocked", "cancelled", "unreachable")

#: The pod's task statuses (queued|running|done|failed|blocked) mapped onto a
#: member's. "queued" was already reported as "started" when the fire
#: succeeded, so it maps back to itself.
_POD_STATUS = {
    "queued": "started",
    "running": "running",
    "done": "done",
    "failed": "failed",
    "blocked": "blocked",
}

MAX_FARMS = 50
OUTCOME_CHARS = 300


@dataclass
class Member:
    """One environment assigned to a farm: its identity, its start time, and
    what came back. ``task_text`` is composed at assignment time and frozen —
    the member runs what it was assigned, not whatever the registry says later."""

    profile: str
    task_text: str = ""
    start_at: float = 0.0
    status: str = "pending"
    task_id: str = ""
    thread_id: str = ""
    finished_at: float | None = None
    outcome: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Farm:
    id: str
    name: str
    recipe: str
    task: str
    url: str
    mode: str
    stagger_seconds: int
    created_at: float
    status: str = "pending"
    members: list[Member] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["display_name"] = self.name or self.recipe
        return d


def compose_task_text(bot: registry.Bot, task: str) -> str:
    """Frame the farm's task as this identity's own.

    The farm's instruction is shared; the framing is per environment. Omitting
    the background clause when there is none keeps a bare identity's payload
    exactly the operator's words.
    """
    name = bot.name or bot.profile
    if bot.background:
        return f"You are {name}. {bot.background}\n\nTASK: {task}"
    return f"You are {name}.\n\nTASK: {task}"


def build_members(
    profiles: list[str],
    bots: dict[str, registry.Bot],
    *,
    task: str,
    mode: str,
    stagger_seconds: int,
    start_at: float | None,
    now: float,
) -> list[Member]:
    """One Member per assigned profile, with start times from the mode.

    Stagger follows the order the operator assigned — the first profile starts
    now, the second stagger_seconds later, and so on. A scheduled farm starts
    everyone at the one epoch (in the past means immediately).
    """
    members: list[Member] = []
    for i, profile in enumerate(profiles):
        if mode == "stagger":
            at = now + i * stagger_seconds
        elif mode == "schedule":
            at = float(start_at or now)
        else:
            at = now
        members.append(Member(
            profile=profile,
            task_text=compose_task_text(bots[profile], task),
            start_at=at,
        ))
    return members


def cancel_farm(farm: Farm) -> None:
    """Cancel what has not started; leave what is running alone.

    A started member is already somebody's pod doing work — stopping it is a
    decision made on that bot's own page, with its own controls.
    """
    for member in farm.members:
        if member.status == "pending":
            member.status = "cancelled"
    farm.status = "cancelled"


# -- the store ---------------------------------------------------------------


def _clean(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def load(path: Path) -> list[Farm]:
    """Read the farm file; a corrupt one degrades to empty.

    Same reasoning as the roster: the hub's one page must not 500 on a file a
    hand edit broke, and an empty list is recoverable from the pod side.
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    farms: list[Farm] = []
    for item in raw.get("farms", []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        members = []
        for m in item.get("members", []):
            if not isinstance(m, dict) or not m.get("profile"):
                continue
            members.append(Member(
                profile=str(m["profile"]),
                task_text=str(m.get("task_text", "")),
                start_at=float(m.get("start_at") or 0.0),
                status=str(m.get("status", "pending")),
                task_id=str(m.get("task_id", "")),
                thread_id=str(m.get("thread_id", "")),
                finished_at=(float(m["finished_at"]) if m.get("finished_at") is not None else None),
                outcome=str(m.get("outcome", "")),
            ))
        farms.append(Farm(
            id=str(item["id"]),
            name=_clean(item.get("name", ""), 200),
            recipe=str(item.get("recipe", "")),
            task=str(item.get("task", "")),
            url=str(item.get("url", "")),
            mode=str(item.get("mode", "parallel")),
            stagger_seconds=int(item.get("stagger_seconds") or 0),
            created_at=float(item.get("created_at") or 0.0),
            status=str(item.get("status", "pending")),
            members=members,
        ))
    return farms


def save(path: Path, farms: list[Farm]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"farms": [f.to_dict() for f in farms]}, indent=2) + "\n")
    os.replace(tmp, path)  # atomic: a reader sees the old file or the new one


class FarmStore:
    """The farm list, in memory, persisted after every mutation.

    One instance serves both the HTTP routes and the fire loop, so there is
    never a second authority to disagree with; the hub is a single process, and
    each mutation is a synchronous write between awaits.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.farms: list[Farm] = load(path)

    def create(
        self, *, name: str, recipe: str, task: str, url: str,
        mode: str, stagger_seconds: int, members: list[Member],
    ) -> Farm:
        farm = Farm(
            id=uuid.uuid4().hex[:12],
            name=_clean(name, 200),
            recipe=recipe,
            task=task,
            url=url,
            mode=mode,
            stagger_seconds=stagger_seconds,
            created_at=time.time(),
            members=members,
        )
        self.farms.append(farm)
        self._cap()
        self.persist()
        return farm

    def get(self, farm_id: str) -> Farm | None:
        return next((f for f in self.farms if f.id == farm_id), None)

    def all(self) -> list[Farm]:
        return sorted(self.farms, key=lambda f: f.created_at, reverse=True)

    def remove(self, farm_id: str) -> bool:
        farm = self.get(farm_id)
        if farm is None:
            return False
        self.farms.remove(farm)
        self.persist()
        return True

    def persist(self) -> None:
        save(self.path, self.farms)

    def _cap(self) -> None:
        """Keep the file small: drop oldest finished farms first.

        A running assignment is history being made; a finished one is history
        already read. Only when everything is unfinished does the oldest
        pending/running farm go.
        """
        if len(self.farms) <= MAX_FARMS:
            return
        finished = [f for f in self.farms if f.status in ("complete", "cancelled")]
        unfinished = [f for f in self.farms if f.status not in ("complete", "cancelled")]
        overflow = len(self.farms) - MAX_FARMS
        finished.sort(key=lambda f: f.created_at)
        drop = set(id(f) for f in finished[:overflow])
        if overflow > len(finished):
            unfinished.sort(key=lambda f: f.created_at)
            drop.update(id(f) for f in unfinished[: overflow - len(finished)])
        self.farms = [f for f in self.farms if id(f) not in drop]


# -- the fire loop ------------------------------------------------------------


def _outcome_snippet(row: dict[str, Any]) -> str:
    """What the pod's finished task says, condensed for the room.

    A done run reports its result; a failed or blocked one reports why — that
    distinction is the operator's next action (nothing vs. go look at noVNC).
    """
    if row.get("status") == "done":
        text = row.get("result") or row.get("detail") or ""
    else:
        text = row.get("detail") or ""
    return _clean(text, OUTCOME_CHARS)


def fire_body(farm: Farm, text: str) -> dict[str, Any]:
    """The wire shape POST /api/tasks actually takes: ``{recipe, payload}``.

    The instruction lives INSIDE ``payload`` — flat top-level fields are
    silently dropped by TaskRequest, which ran a farm with an empty
    instruction (found live: the pod's attempt carried ``payload: {}``). All
    three aliases on purpose: the agent reads ``goal`` first, so a payload
    missing it lets a stale instruction silently win.
    """
    return {
        "recipe": farm.recipe,
        "payload": {
            "url": farm.url,
            "task": text,
            "text": text,
            "goal": text,
        },
    }


class FarmRunner:
    """Fires due members and folds pod state back into the farm.

    Injected ``post``/``get_state`` stand in for the bot API in tests; the
    defaults are the same httpx shape the roster's probe uses — bearer token,
    ``trust_env=False`` (an ambient proxy must never capture in-cluster calls).
    """

    def __init__(
        self,
        store: FarmStore,
        settings: Any,
        *,
        now: Callable[[], float] = time.time,
        post: Callable[..., Any] | None = None,
        get_state: Callable[..., Any] | None = None,
        interval: float = 2.0,
        refresh_every: float = 5.0,
    ) -> None:
        self._store = store
        self._settings = settings
        self._now = now
        self._post = post if post is not None else self._http_post
        self._get_state = get_state if get_state is not None else self._http_get_state
        self._interval = interval
        self._refresh_every = refresh_every
        self._last_refresh: dict[str, float] = {}
        self._task: asyncio.Task | None = None

    # -- lifecycle

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="farm-runner")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad tick must not end the farm system; the next tick is
                # two seconds away and the farms are all still on disk.
                log.exception("farm tick failed")
            await asyncio.sleep(self._interval)

    # -- the tick

    async def tick(self) -> None:
        now = self._now()
        dirty = False
        for farm in self._store.farms:
            if farm.status not in ("pending", "running"):
                continue
            # Refreshes are throttled per farm, not per member: N members share
            # one clock so a five-member farm costs the pods one poll each
            # interval, not five.
            due = now - self._last_refresh.get(farm.id, 0.0) >= self._refresh_every
            for member in farm.members:
                if member.status == "pending" and member.start_at <= now:
                    if await self._fire(farm, member) and farm.status == "pending":
                        farm.status = "running"
                        dirty = True
                elif due and member.status in ("started", "running"):
                    if await self._refresh(farm, member):
                        dirty = True
            if due:
                self._last_refresh[farm.id] = now
                if self._settle(farm):
                    dirty = True
        if dirty:
            self._store.persist()

    def _settle(self, farm: Farm) -> bool:
        if farm.status in ("pending", "running") and not any(
            m.status in ACTIVE for m in farm.members
        ):
            farm.status = "complete"
            return True
        return False

    async def _fire(self, farm: Farm, member: Member) -> bool:
        try:
            body = await self._post(member.profile, farm)
        except Exception as exc:
            # This member's pod being down is a fact about the member, not a
            # failure of the farm: everyone else still fires.
            member.status = "unreachable"
            member.outcome = f"could not start: {type(exc).__name__}: {exc}"[:OUTCOME_CHARS]
            member.finished_at = self._now()
            return True
        member.task_id = str(body.get("id") or "")
        member.thread_id = str(body.get("thread_id") or "") or member.task_id
        if member.task_id:
            member.status = "started"
        else:
            member.status = "unreachable"
            member.outcome = "pod accepted the request but returned no task id"
            member.finished_at = self._now()
        return True

    async def _refresh(self, farm: Farm, member: Member) -> bool:
        """Fold one pod's task row into the member. True when something changed."""
        try:
            state = await self._get_state(member.profile)
        except Exception:
            # Transport noise is transient by definition; the member keeps its
            # status and the next refresh retries. Flipping it to a terminal
            # state on a blip would end a run that is fine.
            return False
        tasks = state.get("tasks", []) if isinstance(state, dict) else []
        row = next((t for t in tasks if t.get("id") == member.task_id), None)
        if row is None:
            member.status = "unreachable"
            member.outcome = "the pod no longer knows this task"
            member.finished_at = self._now()
            return True
        new_status = _POD_STATUS.get(str(row.get("status", "")))
        if not new_status or new_status == member.status:
            return False
        member.status = new_status
        if new_status in ("done", "failed", "blocked"):
            member.finished_at = self._now()
            member.outcome = _outcome_snippet(row)
        return True

    # -- the real HTTP

    def _bot_url(self, profile: str) -> str:
        return self._settings.bot_url_template.format(
            profile=profile, api_port=self._settings.api_port
        )

    def _headers(self) -> dict[str, str]:
        token = getattr(self._settings, "control_token", "")
        return {"Authorization": f"Bearer {token}"}

    async def _http_post(self, profile: str, farm: Farm) -> dict[str, Any]:
        text = next(m.task_text for m in farm.members if m.profile == profile)
        url = self._bot_url(profile) + "/api/tasks"
        async with httpx.AsyncClient(trust_env=False, timeout=15.0) as client:
            res = await client.post(url, json=fire_body(farm, text),
                                    headers=self._headers())
            res.raise_for_status()
            return res.json()

    async def _http_get_state(self, profile: str) -> dict[str, Any]:
        url = self._bot_url(profile) + "/api/state"
        async with httpx.AsyncClient(trust_env=False, timeout=10.0) as client:
            res = await client.get(url, headers=self._headers())
            res.raise_for_status()
            return res.json()
