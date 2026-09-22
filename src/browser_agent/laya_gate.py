"""The Laya gate: element picking and step confirmation for plan.task.

Laya is a decision model, not an agent: it answers typed questions about text
in one encoder pass and can generate nothing. Two roles, both bounded by that:

* *pick*  — "which numbered element should be activated?" over the candidate
  lines :func:`~browser_agent.recipes._candidates.extract_candidates` built.
* *confirm* — "did the step move us toward the goal?" as a yes/no on the page.

Iron rules, enforced by construction:

* Laya NEVER raises and NEVER produces an ``EscalationRequired``. Only the
  deterministic challenge detection may page a human — a Laya false positive
  costs at most a repair attempt, never a noVNC interrupt.
* Any load, transport or inference failure degrades to "off"/inconclusive
  (logged). The plan executor treats an off/low-confidence gate as a step
  failure, which falls into the existing agent fallback.

Two backends, one question schema:

* ``LAYA_DECIDE_URL`` set — the shared decision engine via HTTP (in prod:
  llm-service's ``POST /v1/decide`` proxy to the ggmlc/Vulkan service on cn1).
  The endpoint receives the llm-service auth artifacts (Bearer + ``user``);
  both the proxy and the bare engine tolerate them.
* unset — the in-process ``laya`` pip package (laptop development; the pod
  image does not carry torch).

Confidence semantics are unified across backends because the engines disagree
(measured 2026-09-21): the ggmlc engine reports raw ``p`` as the noul
confidence and a near-zero act-confidence on open-domain choices, while the
pip model reports distance-from-coin-flip. The gate therefore derives
confidence from the answer itself: ``max(p, 1-p)`` for noul, and the chosen
label's own probability for choice. An engine answer is only acted on when
the model itself was clear — otherwise the step is inconclusive.

Uses the ``english`` family with both backends (calibrated checkpoint; the
multilingual family ships without fitted temperatures).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger(__name__)

#: Page text fed to the model. Laya's sequence budget is 512 tokens shared
#: with the question and its options, so the state stays well under that.
_STATE_CHARS = 900

#: The first call on a family loads ~850 MB into VRAM (2-10 s); steady state
#: is ~100-200 ms. The timeout must absorb the cold load, not the warm call.
_DECIDE_TIMEOUT_S = 30.0

_LAYA_FAMILY = "english"


class LayaGate:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._agent: Any = None
        self._state = "unloaded"  # unloaded | ready | broken (local backend only)
        self.stats = {"calls": 0, "picks": 0, "confirms": 0, "inconclusive": 0}

    @property
    def enabled(self) -> bool:
        if self.settings.laya_decide_url:
            return self.settings.laya_enabled
        return self.settings.laya_enabled and self._state != "broken"

    @property
    def pick_enabled(self) -> bool:
        return self.enabled and self.settings.laya_pick_enabled

    @property
    def summary(self) -> dict[str, Any]:
        backend = "http" if self.settings.laya_decide_url else "local"
        return {"state": self._state if backend == "local" else backend, **self.stats}

    def _load(self) -> Any:
        """Build the in-process classifier. Override point for tests."""
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
        try:
            ans = out["answers"]["v"]
            p = float(ans["noul"])
        except (KeyError, TypeError, ValueError):
            self.stats["inconclusive"] += 1
            return None, 0.0
        self.stats["confirms"] += 1
        return p >= 0.5, max(p, 1.0 - p)

    async def choose(
        self, question: str, lines: list[str], state_text: str
    ) -> tuple[int | None, float]:
        """Pick one of the numbered candidate lines. ``(None, 0.0)`` = no pick.

        This does **not** consult ``pick_enabled``: that flag governs the
        *plan* picker, whose wrong answer clicks the wrong element. A caller
        with a different risk profile (the minesweeper tiebreak, where a wrong
        guess costs one life in a game the solver can still win) decides for
        itself. Callers that want the flag honour it — ``_plan_exec`` checks it
        before asking, and its own test pins that refusal.
        """
        if not self.enabled or not lines:
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
        try:
            ans = out["answers"]["pick"]
        except (KeyError, TypeError):
            self.stats["inconclusive"] += 1
            return None, 0.0
        self.stats["picks"] += 1
        label = str(ans.get("choice", ""))
        try:
            idx = int(label)
        except ValueError:
            idx = -1
        if not 0 <= idx < len(lines):
            self.stats["inconclusive"] += 1
            return None, self._choice_confidence(ans, label)
        return idx, self._choice_confidence(ans, label)

    @staticmethod
    def _choice_confidence(ans: dict, label: str) -> float:
        """The chosen label's own probability — the honest confidence in the
        action taken. Falls back to the engine's confidence field."""
        probs = ans.get("probabilities") or {}
        if label in probs:
            return float(probs[label])
        return float(ans.get("confidence") or 0.0)

    async def _predict(self, state: str, questions: dict) -> dict | None:
        if self.settings.laya_decide_url:
            return await self._predict_http(state, questions)
        return await self._predict_local(state, questions)

    async def _predict_local(self, state: str, questions: dict) -> dict | None:
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

    async def _predict_http(self, state: str, questions: dict) -> dict | None:
        headers = {}
        payload: dict[str, Any] = {
            "state": state[:_STATE_CHARS],
            "questions": questions,
            "model": _LAYA_FAMILY,
        }
        if self.settings.llm_api_key:
            # llm-service requires these on every route; the bare engine
            # ignores them, so one payload serves both. The /v1/decide route
            # takes `client_id` literally — unlike the OpenAI-compat routes
            # it has no `user` alias (measured live 2026-09-21: `user` → 400).
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
            headers["X-LLM-Source"] = self.settings.llm_source
            payload["client_id"] = self.settings.llm_client_id
        try:
            # trust_env=False: llm-service/engine live on the tailnet/in-cluster;
            # an ambient HTTP_PROXY must never apply.
            async with httpx.AsyncClient(timeout=_DECIDE_TIMEOUT_S, trust_env=False) as client:
                resp = await client.post(
                    self.settings.laya_decide_url, json=payload, headers=headers
                )
                resp.raise_for_status()
                out = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("laya decide failed (%s); treating as inconclusive", exc)
            self.stats["inconclusive"] += 1
            return None
        if not isinstance(out, dict) or "answers" not in out:
            log.warning("laya decide returned no answers: %s", str(out)[:200])
            self.stats["inconclusive"] += 1
            return None
        self.stats["calls"] += 1
        return out
