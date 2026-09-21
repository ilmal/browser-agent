"""The Laya gate: element picking and step confirmation for plan.task.

Laya is a 322M classifier, not an agent: it answers typed questions about text
in one forward pass (~0.1-0.5 s on the pod's CPU) and can generate nothing.
Two roles, both bounded by that fact:

* *pick*  — "which numbered element should be activated?" over the candidate
  lines :func:`~browser_agent.recipes._candidates.extract_candidates` built.
* *confirm* — "did the step move us toward the goal?" as a yes/no on the page.

Iron rules, enforced by construction:

* Laya NEVER raises and NEVER produces an ``EscalationRequired``. Only the
  deterministic challenge detection may page a human — a Laya false positive
  costs at most a repair attempt, never a noVNC interrupt.
* Any load or inference failure degrades to "off" (logged once). The plan
  executor treats an off/low-confidence gate as a step failure, which falls
  into the existing agent fallback.

Uses the calibrated English checkpoint directly (``Agent``, not ``Router``):
routing page text by language would hand non-English pages to the
multilingual checkpoint, which ships without fitted temperatures —
uncalibrated confidence is worse than a consistent English model.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .config import Settings

log = logging.getLogger(__name__)

#: Page text fed to the model. Laya's sequence budget is 512 tokens shared
#: with the question and its options, so the state stays well under that.
_STATE_CHARS = 900


class LayaGate:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._agent: Any = None
        self._state = "unloaded"  # unloaded | ready | broken
        self.stats = {"calls": 0, "picks": 0, "confirms": 0, "inconclusive": 0}

    @property
    def enabled(self) -> bool:
        return self.settings.laya_enabled and self._state != "broken"

    @property
    def pick_enabled(self) -> bool:
        return self.enabled and self.settings.laya_pick_enabled

    @property
    def summary(self) -> dict[str, Any]:
        return {"state": self._state, **self.stats}

    def _load(self) -> Any:
        """Build the classifier. Override point for tests."""
        from laya import Agent

        return Agent()

    def _ensure(self) -> Any | None:
        if self._state == "ready":
            return self._agent
        if self._state == "broken":
            return None
        try:
            self._agent = self._load()
            self._state = "ready"
            log.info("laya gate loaded (%s)", type(self._agent).__name__)
        except Exception as exc:
            log.warning("laya unavailable (%s); picking and confirmation are off", exc)
            self._state = "broken"
        return self._agent

    async def yes_no(self, question: str, state_text: str) -> tuple[bool | None, float]:
        """Ask a yes/no question. ``(None, 0.0)`` means inconclusive or off."""
        if not self.enabled:
            return None, 0.0
        out = await self._predict(
            state_text,
            {
                "v": {
                    "type": "noul",
                    "instructions": question,
                    "criteria": {"true": "yes", "false": "no"},
                }
            },
        )
        if out is None:
            return None, 0.0
        ans = out["answers"]["v"]
        self.stats["confirms"] += 1
        return float(ans["noul"]) >= 0.5, float(ans["confidence"])

    async def choose(
        self, question: str, lines: list[str], state_text: str
    ) -> tuple[int | None, float]:
        """Pick one of the numbered candidate lines. ``(None, 0.0)`` = no pick."""
        if not self.pick_enabled or not lines:
            return None, 0.0
        out = await self._predict(
            state_text,
            {
                "pick": {
                    "type": "choice",
                    "instructions": question,
                    "criteria": {str(i): line for i, line in enumerate(lines)},
                }
            },
        )
        if out is None:
            return None, 0.0
        ans = out["answers"]["pick"]
        self.stats["picks"] += 1
        try:
            idx = int(ans["choice"])
        except (KeyError, TypeError, ValueError):
            self.stats["inconclusive"] += 1
            return None, float(ans.get("confidence", 0.0))
        if not 0 <= idx < len(lines):
            self.stats["inconclusive"] += 1
            return None, float(ans.get("confidence", 0.0))
        return idx, float(ans["confidence"])

    async def _predict(self, state: str, questions: dict) -> dict | None:
        def _run() -> dict:
            agent = self._ensure()
            if agent is None:
                raise RuntimeError("laya gate is off")
            return agent.predict(state, questions)

        try:
            out = await asyncio.to_thread(_run)
        except Exception as exc:
            log.warning("laya predict failed (%s); treating as inconclusive", exc)
            self.stats["inconclusive"] += 1
            return None
        self.stats["calls"] += 1
        return out
