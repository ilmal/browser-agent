"""The retry conversation: one thread per line of work, one message per turn.

A retry used to mint a brand-new, unlinked task. The operator saw a stack of
unrelated History rows and no way to say "no, try it this way instead" — the
only lever was a button that re-ran the identical instruction. A thread fixes
the linkage; the messages fix the steering.

Held in memory, and **mirrored to the run archive** on the way in. An earlier
cut persisted these in SQLite while tasks stayed in memory, which was worse than
useless: the messages came back after a restart with no attempts to attach to.
The arrival of the durable run archive inverts that — the attempts now outlive
the process, so a message that did not would be the odd one out, and the very
first thing an operator types (the instruction that *created* the attempt) is
what makes that attempt legible when it is read back. Memory is still the
authority for the current process; the archive is what a restart reads from.
"""

from __future__ import annotations

import contextlib
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
    """Messages per thread, in process memory and mirrored to the archive."""

    def __init__(self, runs: Any = None) -> None:
        self._by_thread: dict[str, list[Message]] = {}
        #: Threads whose archive rows have already been folded into memory, so a
        #: poll every 2 s does not re-read the table every time.
        self._loaded: dict[str, bool] = {}
        # The durable half, or None. Optional for the same reason the runner's
        # archive is: a caller that only wants the live behaviour should not be
        # forced to build one, and a missing archive costs the record, never the
        # message.
        self._runs = runs

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
        if self._runs is not None:
            # Best-effort, like every other archive write: the message is
            # already in memory and the conversation is already correct for
            # this process. Losing the durable copy must not fail a send.
            with contextlib.suppress(Exception):
                self._runs.save_message(msg)
        return msg

    def for_thread(self, thread_id: str) -> list[Message]:
        """This thread's messages: memory first, the archive behind it.

        The union rather than either alone, keyed by id, because a restart
        empties memory while the archive still holds everything said before it
        — and a thread that is mid-conversation has both. Merging (rather than
        "memory if non-empty") is what makes the first message after a restart
        join the conversation instead of appearing to start a fresh one.
        """
        live = self._by_thread.get(thread_id) or []
        if self._runs is None:
            return list(live)
        # Only load once per thread per process: after the first read, memory
        # holds the archive's rows too, so a second read would be wasted work.
        loaded = self._loaded.setdefault(thread_id, False)
        if not loaded:
            self._loaded[thread_id] = True
            if not live:
                live = []
                self._by_thread[thread_id] = live
            seen = {m.id for m in live}
            for r in self._runs.messages(thread_id):
                if r["id"] in seen:
                    continue
                live.append(Message(
                    id=r["id"],
                    thread_id=r["thread_id"],
                    at=r["at"],
                    role=r["role"],
                    kind=r["kind"],
                    text=r["text"],
                    meta=r.get("meta") or {},
                ))
            live.sort(key=lambda m: m.at)
        return list(live)

    def last_instruction(self, thread_id: str) -> str:
        """The most recent operator instruction, which is the live one."""
        for msg in reversed(self._by_thread.get(thread_id, ())):
            if msg.kind == "instruction":
                return msg.text
        return ""
