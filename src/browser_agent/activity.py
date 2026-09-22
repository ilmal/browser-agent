"""A live, bounded log of what a task is doing, for the admin UI.

The operator's question during a run is "what is it doing right now?", and the
honest answer lives several calls below the task runner: in the plan executor's
per-step outcome and in the agent's per-step hooks. Rather than thread a logger
through every signature, the log rides on the :class:`BrowserSession`, which
every layer already receives and which is exactly one per pod — and because
tasks are serialised on a single worker, a session-scoped log *is* the running
task's log. The runner rebinds it per task.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

#: Kinds the UI styles differently. "step" is a deterministic plan step,
#: "agent" an LLM fallback step, "gate" a Laya verdict, "error" a failure the
#: operator may need to act on, "operator" something the person typed —
#: mirrored into the feed so a steer is visible at the moment it lands, not
#: only in the thread panel.
KINDS = ("info", "step", "agent", "gate", "error", "operator")


@dataclass
class Activity:
    """Append-only, capped so a runaway agent cannot grow the state payload."""

    limit: int = 200
    entries: list[dict[str, Any]] = field(default_factory=list)

    def note(self, kind: str, text: str, **extra: Any) -> None:
        text = " ".join(str(text).split())
        if not text:
            return
        entry: dict[str, Any] = {"at": time.time(), "kind": kind, "text": text[:600]}
        entry.update(extra)
        self.entries.append(entry)
        if len(self.entries) > self.limit:
            del self.entries[: len(self.entries) - self.limit]

    def reset(self) -> None:
        self.entries.clear()

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.entries)


def activity_of(session: Any) -> Activity:
    """The session's log, or a throwaway one.

    Tests build minimal session stand-ins, so this never assumes the attribute
    is there — a missing log must cost the note, not the task.
    """
    found = getattr(session, "activity", None)
    return found if isinstance(found, Activity) else Activity()
