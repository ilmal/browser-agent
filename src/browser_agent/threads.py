"""The retry conversation: one thread per line of work, one message per turn.

A retry used to mint a brand-new, unlinked task. The operator saw a stack of
unrelated History rows and no way to say "no, try it this way instead" — the
only lever was a button that re-ran the identical instruction. A thread fixes
the linkage; the ``messages`` table fixes the steering.

Deliberately the same SQLite file as the schedules (the pod's PVC, one writer
per profile), so a thread survives a pod restart the way a schedule does.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id        TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    at        REAL NOT NULL,
    role      TEXT NOT NULL,
    kind      TEXT NOT NULL,
    text      TEXT NOT NULL,
    meta      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_thread ON messages (thread_id, at);
"""

#: Who wrote a message. "operator" is the person iterating; "bot" is a
#: deterministic outcome the runner records; "system" is housekeeping.
ROLES = ("operator", "bot", "system")

#: What an operator message *does*. This is a mode the operator picks, not a
#: classifier's guess: an instruction changes the prose, a parameter changes a
#: run knob, a config changes the shared recipe. Guessing would make the three
#: indistinguishable in the log exactly when it mattered.
KINDS = ("instruction", "parameter", "config", "note")


@dataclass
class Message:
    id: str
    thread_id: str
    at: float
    role: str
    kind: str
    text: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "at": self.at,
            "role": self.role,
            "kind": self.kind,
            "text": self.text,
            "meta": self.meta,
        }


class ThreadStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def say(
        self,
        thread_id: str,
        role: str,
        kind: str,
        text: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> Message:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; have {ROLES}")
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; have {KINDS}")
        msg = Message(
            id=uuid.uuid4().hex[:12],
            thread_id=thread_id,
            at=time.time(),
            role=role,
            kind=kind,
            text=" ".join(str(text).split())[:2000],
            meta=meta or {},
        )
        self._conn.execute(
            "INSERT INTO messages (id, thread_id, at, role, kind, text, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg.id, msg.thread_id, msg.at, msg.role, msg.kind, msg.text,
             json.dumps(msg.meta)),
        )
        self._conn.commit()
        return msg

    def for_thread(self, thread_id: str) -> list[Message]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE thread_id = ? ORDER BY at", (thread_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def last_instruction(self, thread_id: str) -> str:
        """The most recent operator instruction, which is the live one.

        Used to brief the next attempt: the thread's newest wording is what the
        operator meant, not whatever the first attempt was queued with.
        """
        row = self._conn.execute(
            "SELECT text FROM messages WHERE thread_id = ? AND kind = 'instruction' "
            "ORDER BY at DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        return row["text"] if row is not None else ""

    def threads_with_tasks(self, tasks: list[Any]) -> list[dict[str, Any]]:
        """Group task dicts by thread, newest thread first.

        The bot page shows one row per *line of work* rather than per attempt,
        which is the operator's actual complaint: "a history with one block,
        then the retries".
        """
        groups: dict[str, list[Any]] = {}
        for t in tasks:
            groups.setdefault(t.thread_id or t.id, []).append(t)
        out: list[dict[str, Any]] = []
        for thread_id, members in groups.items():
            members = sorted(members, key=lambda t: t.created_at)
            out.append(
                {
                    "thread_id": thread_id,
                    "attempts": [m.to_dict() for m in members],
                    "count": len(members),
                    "latest": members[-1].to_dict(),
                }
            )
        out.sort(key=lambda g: g["latest"]["created_at"], reverse=True)
        return out

    @staticmethod
    def _row(row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            thread_id=row["thread_id"],
            at=row["at"],
            role=row["role"],
            kind=row["kind"],
            text=row["text"],
            meta=json.loads(row["meta"] or "{}"),
        )
