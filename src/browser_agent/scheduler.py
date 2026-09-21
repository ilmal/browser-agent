"""Schedule store.

Schedules live in a small SQLite file on the profile's PVC so they survive
pod restarts. Kept intentionally tiny: this is a cron table, not an
orchestration engine.

A schedule is a cron expression plus a recipe and payload. It is evaluated
inside the pod by a single loop, so there is no distributed lock to get wrong.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    id          TEXT PRIMARY KEY,
    recipe      TEXT NOT NULL,
    payload     TEXT NOT NULL,
    cron        TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    next_run_at REAL,
    last_run_at REAL,
    last_task_id TEXT,
    created_at  REAL NOT NULL
);
"""


@dataclass
class Schedule:
    id: str
    recipe: str
    payload: dict[str, Any]
    cron: str
    enabled: bool
    next_run_at: float | None
    last_run_at: float | None
    last_task_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "recipe": self.recipe,
            "payload": self.payload,
            "cron": self.cron,
            "enabled": self.enabled,
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "last_task_id": self.last_task_id,
        }


class ScheduleStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        except sqlite3.OperationalError as exc:
            # A bind mount owned by another uid is the usual cause, and the
            # raw sqlite error does not say so. Fail with the fix in it.
            raise RuntimeError(
                f"cannot open {db_path} — is {db_path.parent} writable by uid "
                f"{os.getuid()}? For a bind mount, chown it or run the container "
                f"with `user: \"$(id -u):$(id -g)\"`."
            ) from exc
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add(self, schedule_id: str, recipe: str, payload: dict[str, Any], cron: str) -> Schedule:
        from croniter import croniter

        now = time.time()
        next_run = croniter(cron, now).get_next()
        self._conn.execute(
            "INSERT OR REPLACE INTO schedules "
            "(id, recipe, payload, cron, enabled, next_run_at, last_run_at, "
            "last_task_id, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?, NULL, NULL, ?)",
            (schedule_id, recipe, json.dumps(payload), cron, next_run, now),
        )
        self._conn.commit()
        return self.get(schedule_id)

    def get(self, schedule_id: str) -> Schedule:
        row = self._conn.execute(
            "SELECT * FROM schedules WHERE id = ?", (schedule_id,)
        ).fetchone()
        if row is None:
            raise KeyError(schedule_id)
        return self._row(row)

    def all(self) -> list[Schedule]:
        rows = self._conn.execute("SELECT * FROM schedules ORDER BY created_at").fetchall()
        return [self._row(r) for r in rows]

    def delete(self, schedule_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def set_enabled(self, schedule_id: str, enabled: bool) -> Schedule:
        self._conn.execute(
            "UPDATE schedules SET enabled = ? WHERE id = ?", (1 if enabled else 0, schedule_id)
        )
        self._conn.commit()
        return self.get(schedule_id)

    def due(self, now: float | None = None) -> list[Schedule]:
        now = now or time.time()
        rows = self._conn.execute(
            "SELECT * FROM schedules WHERE enabled = 1 AND next_run_at IS NOT NULL "
            "AND next_run_at <= ? ORDER BY next_run_at",
            (now,),
        ).fetchall()
        return [self._row(r) for r in rows]

    def mark_run(self, schedule_id: str, task_id: str) -> None:
        from croniter import croniter

        sched = self.get(schedule_id)
        now = time.time()
        self._conn.execute(
            "UPDATE schedules SET last_run_at = ?, last_task_id = ?, next_run_at = ? WHERE id = ?",
            (now, task_id, croniter(sched.cron, now).get_next(), schedule_id),
        )
        self._conn.commit()

    @staticmethod
    def _row(row: sqlite3.Row) -> Schedule:
        return Schedule(
            id=row["id"],
            recipe=row["recipe"],
            payload=json.loads(row["payload"]),
            cron=row["cron"],
            enabled=bool(row["enabled"]),
            next_run_at=row["next_run_at"],
            last_run_at=row["last_run_at"],
            last_task_id=row["last_task_id"],
        )
