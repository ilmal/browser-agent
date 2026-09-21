"""Mid-run control of the task that is currently executing.

The operator watches a run and wants to intervene *while it works* — pause it,
stop it, or tell it something it got wrong. That needs one object the API can
reach from the request thread and the executing task can read at its own
boundaries, which is what this is.

Two rules keep it honest:

* **Boundaries only.** Nothing here interrupts a step in flight. A step is
  either a deterministic Playwright action or an LLM call, and yanking either
  mid-flight leaves the browser in a state neither layer expects. Pause, cancel
  and amend all land at the next checkpoint.
* **An amendment is a re-plan, not a text edit.** Changing the instruction
  half-way through a fixed step list makes the list wrong, so an amendment
  aborts the current run and the task is re-planned from where the browser
  actually is. "Edit what it's doing" has to mean the plan changes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """The operator stopped this run. Not a failure — a decision."""


class Amended(Exception):
    """The operator changed the instruction mid-run; re-plan with the new one."""

    def __init__(self, instruction: str, url: str = "") -> None:
        super().__init__(f"instruction amended to: {instruction[:120]}")
        self.instruction = instruction
        self.url = url


@dataclass
class Control:
    """Live controls for one running task.

    ``amendments`` is a list because the operator may type more than once while
    the task is between checkpoints; the last one wins, since each is meant as
    the new full instruction rather than a delta.
    """

    paused: bool = False
    cancelled: bool = False
    amendments: list[str] = field(default_factory=list)
    _resume: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def amendment(self) -> str | None:
        return self.amendments[-1] if self.amendments else None

    def pause(self) -> None:
        self.paused = True
        self._resume.clear()

    def resume(self) -> None:
        self.paused = False
        self._resume.set()

    def cancel(self) -> None:
        self.cancelled = True
        # A cancelled run must not stay parked on the pause gate.
        self._resume.set()

    def steer(self, instruction: str) -> None:
        text = " ".join(instruction.split())
        if text:
            self.amendments.append(text)

    async def checkpoint(self, url: str = "") -> None:
        """Called by the executing layer between steps. Raises to unwind."""
        if self.cancelled:
            raise Cancelled("stopped by operator")
        if self.paused:
            log.info("task paused by operator; waiting for resume")
            await self._resume.wait()
            if self.cancelled:
                raise Cancelled("stopped by operator")
        if self.amendment is not None:
            raise Amended(self.amendment, url)

    def to_dict(self) -> dict:
        return {
            "paused": self.paused,
            "cancelled": self.cancelled,
            "amendment": self.amendment,
        }
