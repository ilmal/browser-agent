"""The retry conversation: one thread per line of work, one message per turn.

A retry used to mint a brand-new, unlinked task. The operator saw a stack of
unrelated History rows and no way to say "no, try it this way instead" — the
only lever was a button that re-ran the identical instruction. A thread fixes
the linkage; the messages fix the steering.

Deliberately **in memory**, with exactly the lifetime of the tasks it annotates.
An earlier cut persisted these in SQLite so a thread would survive a pod
restart, which turned out to be worse than useless: tasks are in-memory, so
after a restart the messages came back with no attempts to attach to — a store
outliving the thing it describes, plus rows nothing could ever read. Keeping
both in one process makes that divergence impossible by construction.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

#: Who wrote a message. "operator" is the person iterating; "bot" is a
#: deterministic outcome the runner records; "system" is housekeeping.
ROLES = ("operator", "bot", "system")

#: What an operator message *does*. This is a mode the operator picks, not a
#: classifier's guess: an instruction changes the prose, a parameter changes a
#: run knob, a config changes the shared recipe. Guessing would make the three
#: indistinguishable in the log exactly when it mattered.
KINDS = ("instruction", "parameter", "config", "note")

#: Bounded per thread, so a long conversation cannot grow the process without
#: limit. Far more than any real thread needs.
MAX_PER_THREAD = 500


@dataclass
class Message:
    id: str
    thread_id: str
    at: float
    role: str
    kind: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

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
    """Messages per thread, in process memory."""

    def __init__(self) -> None:
        self._by_thread: dict[str, list[Message]] = {}

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
        bucket = self._by_thread.setdefault(thread_id, [])
        bucket.append(msg)
        if len(bucket) > MAX_PER_THREAD:
            del bucket[: len(bucket) - MAX_PER_THREAD]
        return msg

    def for_thread(self, thread_id: str) -> list[Message]:
        return list(self._by_thread.get(thread_id, ()))

    def last_instruction(self, thread_id: str) -> str:
        """The most recent operator instruction, which is the live one."""
        for msg in reversed(self._by_thread.get(thread_id, ())):
            if msg.kind == "instruction":
                return msg.text
        return ""
