"""Durable record of every run, kept so runs can be evaluated after the fact.

The operator's ask: "all steps, all states, all decisions need to be saved so we
can evaluate them — I had a run that was far from perfect, and I need to keep it
to know what to improve." That is a different job from the live feed: the feed is
a bounded in-memory log that exists to answer "what is it doing *now*", and it
dies with the process. This is the archive.

Why a store of its own rather than more columns on a task: tasks are in-memory
and vanish on every pod recreate — which a deploy performs. A record that lives
only as long as the process cannot be "keep this one, I want to study it". So
runs go to SQLite on the profile's PVC, written when an attempt ends, and are
readable by attempt id long after the task object is gone.

The record is deliberately the *whole* attempt, not a summary: the payload it was
given, the snapshotted feed, the result, and — most importantly for the operator's
purpose — every decision the gate made, in order, so a run can be replayed and
scored rather than merely remembered. ``RunStore.decisions`` returns them as a
flat list because that is the shape an evaluation walks.

Retention is bounded by count, not by age: the point is to keep recent runs for
study, and an unbounded table on a 1 GiB volume would eventually fill it. The
newest ``KEEP_PER_PROFILE`` are retained; older rows are pruned on write.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Runs kept per profile. Generous for study, bounded for the 1 GiB data volume:
#: a run row is a few KB, so this is tens of MB at most.
KEEP_PER_PROFILE = 500

#: Messages kept per thread. Far more than a real conversation needs, and each
#: is ~1 KB, so this is not what fills the volume.
KEEP_MESSAGES_PER_THREAD = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    task_id     TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    profile     TEXT NOT NULL,
    recipe      TEXT NOT NULL,
    attempt     INTEGER NOT NULL,
    status      TEXT NOT NULL,
    detail      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    used_agent  INTEGER NOT NULL DEFAULT 0,
    reads_instruction INTEGER NOT NULL DEFAULT 1,
    payload     TEXT NOT NULL,
    result      TEXT,
    activity    TEXT NOT NULL,
    decisions   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_thread ON runs (thread_id);
CREATE INDEX IF NOT EXISTS runs_started ON runs (started_at);

CREATE TABLE IF NOT EXISTS messages (
    id        TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    profile   TEXT NOT NULL,
    at        REAL NOT NULL,
    role      TEXT NOT NULL,
    kind      TEXT NOT NULL,
    text      TEXT NOT NULL,
    meta      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_thread ON messages (thread_id, at);
"""


@dataclass
class Run:
    task_id: str
    thread_id: str
    recipe: str
    attempt: int
    status: str
    detail: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    used_agent: bool
    #: Whether a message would change what this attempt does — a fact about the
    #: recipe at the time it ran, so it is stored rather than re-derived. A
    #: stored recipe can be edited or deleted after the run, and a recipe that
    #: is gone must not retroactively make an old thread claim it was steerable.
    reads_instruction: bool
    payload: dict[str, Any]
    result: dict[str, Any] | None
    activity: list[dict[str, Any]]
    decisions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "thread_id": self.thread_id,
            "recipe": self.recipe,
            "attempt": self.attempt,
            "status": self.status,
            "detail": self.detail,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "used_agent": self.used_agent,
            "reads_instruction": self.reads_instruction,
            "payload": self.payload,
            "result": self.result,
            "activity": self.activity,
            "decisions": self.decisions,
        }


def _decisions_from(activity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The gate's decisions, in order, lifted out of the activity feed.

    A "decision" is a moment the run *chose* rather than merely acted: a Laya
    verdict (kind ``gate``) or an agent step (kind ``agent``). Kept as a separate
    list on the record because evaluation walks decisions, and re-filtering the
    feed at read time in every consumer is the thing that drifts.
    """
    return [e for e in activity if e.get("kind") in ("gate", "agent")]


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _loads(text: str | None, fallback: Any) -> Any:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return fallback


class RunStore:
    """Finished attempts, on disk. Never raises into a run: a save that fails
    must cost the record, not the task — the run is already over by the time we
    write, and its outcome is the operator's, not this store's."""

    def __init__(self, db_path: Path, *, profile: str, keep: int = KEEP_PER_PROFILE) -> None:
        self.db_path = db_path
        self.profile = profile
        self.keep = keep
        self._conn: sqlite3.Connection | None = None
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()
        except Exception as exc:
            # A read-only or missing volume must not stop the agent from running.
            # Log with the fix in it and degrade to "runs are not being saved".
            log.warning("run archive unavailable at %s (%s); runs will not be saved",
                        db_path, exc)
            self._conn = None

    def _migrate(self) -> None:
        """Bring an existing archive up to the current schema.

        ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already
        exists, so a column added later would silently be missing on every
        database written by an older build — and the INSERT names it, so every
        save would fail. Adding the column is the whole migration: the default
        backfills the rows already there, which is right, because a run recorded
        before this column existed was recorded by a build where every recipe's
        steerability was the same question asked live.
        """
        if self._conn is None:
            return
        with contextlib.suppress(Exception):
            have = {r["name"] for r in self._conn.execute("PRAGMA table_info(runs)")}
            if "reads_instruction" not in have:
                self._conn.execute(
                    "ALTER TABLE runs ADD COLUMN reads_instruction "
                    "INTEGER NOT NULL DEFAULT 1"
                )

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def save(self, task: Any) -> bool:
        """Persist one finished attempt. Returns whether it was written."""
        if self._conn is None:
            return False
        activity = list(getattr(task, "activity", []) or [])
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(task_id, thread_id, profile, recipe, attempt, status, detail, "
                "created_at, started_at, finished_at, used_agent, "
                "reads_instruction, payload, result, "
                "activity, decisions) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task.id,
                    getattr(task, "thread_id", "") or task.id,
                    self.profile,
                    task.recipe,
                    int(getattr(task, "attempt", 1) or 1),
                    str(getattr(task.status, "value", task.status)),
                    getattr(task, "detail", "") or "",
                    float(getattr(task, "created_at", time.time())),
                    getattr(task, "started_at", None),
                    getattr(task, "finished_at", None),
                    1 if getattr(task, "used_agent", False) else 0,
                    1 if getattr(task, "reads_instruction", True) else 0,
                    _dumps(getattr(task, "payload", {}) or {}),
                    _dumps(getattr(task, "result", None)),
                    _dumps(activity),
                    _dumps(_decisions_from(activity)),
                ),
            )
            self._conn.commit()
            self._prune()
            return True
        except Exception:
            log.warning("could not archive run %s", getattr(task, "id", "?"),
                        exc_info=True)
            return False

    def _prune(self) -> None:
        """Drop the oldest rows beyond the retention bound."""
        if self._conn is None:
            return
        with contextlib.suppress(Exception):
            self._conn.execute(
                "DELETE FROM runs WHERE task_id IN ("
                "  SELECT task_id FROM runs ORDER BY created_at DESC LIMIT -1 OFFSET ?"
                ")",
                (self.keep,),
            )
            self._conn.commit()

    # -- the conversation --------------------------------------------------

    def save_message(self, msg: Any) -> bool:
        """Persist one thread message. Same contract as ``save``: never raises.

        Messages are stored beside the runs rather than in the in-memory
        ``ThreadStore`` alone, because the operator's first instruction is the
        thing that makes an attempt legible — and the attempt outlives the
        process that received it. A message with no stored text would come back
        as an attempt that appeared from nowhere.
        """
        if self._conn is None:
            return False
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO messages "
                "(id, thread_id, profile, at, role, kind, text, meta) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    msg.id,
                    msg.thread_id,
                    self.profile,
                    float(msg.at),
                    msg.role,
                    msg.kind,
                    msg.text,
                    _dumps(getattr(msg, "meta", {}) or {}),
                ),
            )
            self._conn.commit()
            return True
        except Exception:
            log.warning("could not archive message %s", getattr(msg, "id", "?"),
                        exc_info=True)
            return False

    def messages(self, thread_id: str, *, limit: int = KEEP_MESSAGES_PER_THREAD
                 ) -> list[dict[str, Any]]:
        """This thread's saved messages, oldest first."""
        if self._conn is None:
            return []
        with contextlib.suppress(Exception):
            rows = self._conn.execute(
                "SELECT id, thread_id, at, role, kind, text, meta FROM messages "
                "WHERE thread_id = ? ORDER BY at ASC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "thread_id": r["thread_id"],
                    "at": r["at"],
                    "role": r["role"],
                    "kind": r["kind"],
                    "text": r["text"],
                    "meta": _loads(r["meta"], {}),
                }
                for r in rows
            ]
        return []

    def list(self, *, limit: int = 50, thread_id: str = "") -> list[Run]:
        if self._conn is None:
            return []
        try:
            if thread_id:
                rows = self._conn.execute(
                    "SELECT * FROM runs WHERE thread_id = ? ORDER BY created_at DESC "
                    "LIMIT ?", (thread_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,),
                ).fetchall()
            return [self._row(r) for r in rows]
        except Exception:
            log.warning("could not read run archive", exc_info=True)
            return []

    def get(self, task_id: str) -> Run | None:
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE task_id = ?", (task_id,)
            ).fetchone()
            return self._row(row) if row else None
        except Exception:
            log.warning("could not read run %s", task_id, exc_info=True)
            return None

    def count(self) -> int:
        if self._conn is None:
            return 0
        try:
            return int(self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
        except Exception:
            return 0

    @staticmethod
    def _row(row: sqlite3.Row) -> Run:
        return Run(
            task_id=row["task_id"],
            thread_id=row["thread_id"],
            recipe=row["recipe"],
            attempt=row["attempt"],
            status=row["status"],
            detail=row["detail"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            used_agent=bool(row["used_agent"]),
            # Absent on rows written before the column existed; those runs all
            # predate stored recipes' steerability question, and True is the
            # permissive reading — it never claims a recipe that is not there.
            reads_instruction=bool(row["reads_instruction"])
            if "reads_instruction" in row.keys() else True,
            payload=_loads(row["payload"], {}),
            result=_loads(row["result"], None),
            activity=_loads(row["activity"], []),
            decisions=_loads(row["decisions"], []),
        )
