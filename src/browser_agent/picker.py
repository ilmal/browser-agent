"""LLM element picker for plan.task's selectorless click/type steps.

Same contract as the laya picker: candidates are enumerated as numbered
lines, the model returns an index, and the executor binds
``locator.nth(index)`` — never a re-match on text, which identical buttons
would make ambiguous. Only the engine changed, on bench numbers (run in the
prod pod 2026-09-21): laya could not discriminate candidates on either head —
its choice head ships near-uniform probabilities on multi-candidate buckets
(4/18 correct, every probability 0.26-0.54) and a per-element yes/no
tournament was equally flat (all p in 0.41-0.55), so no confidence floor can
ever be cleared. deepseek-v4.1-flash on the same fixtures: 18/18, zero parse
failures, ~1 s per pick. See ``scripts/laya_pick_bench.py`` and
``scripts/llm_pick_bench.py``. Laya keeps the confirm role, where it
measures well (0.944 on a clear no).

Failure semantics mirror the rest of the gate: this module never raises and
never escalates — any transport, auth or parse problem reads as "no pick",
which the plan executor turns into a step failure and the agent fallback
repairs.
"""

from __future__ import annotations

import logging
import re

import httpx

from .config import Settings

log = logging.getLogger(__name__)

_SYSTEM = (
    "You pick which numbered page element to activate to accomplish a goal. "
    "Answer with ONLY the element number (an integer). No other text."
)

_FIRST_INT = re.compile(r"\d+")


class ElementPicker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        # Same "configured" gate as the LLM fallback: a flag without a key
        # must not produce calls.
        return self.settings.picker_enabled and bool(self.settings.llm_api_key)

    async def pick(
        self, goal: str, state_text: str, lines: list[str]
    ) -> tuple[int | None, float]:
        """Return ``(index, confidence)`` for the chosen candidate line.

        ``(None, 0.0)`` means no usable pick — disabled, unreachable, or an
        answer that is not an in-range element number. Confidence is 1.0 only
        for a cleanly parsed, in-range index; there is no second signal worth
        trusting from a one-token answer, so range validation IS the gate.
        """
        if not self.enabled or not lines:
            return None, 0.0
        body = {
            "model": self.settings.picker_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Goal: {goal}\n\nPage:\n{state_text}\n\n"
                        f"Elements:\n" + "\n".join(lines) + "\n\nWhich element number?"
                    ),
                },
            ],
            # llm-service authenticates every token-spending call with a
            # client id in the body (the OpenAI-compat `user` field); its own
            # id so the call log separates picker traffic.
            "user": self.settings.picker_client_id,
        }
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        if self.settings.llm_source:
            headers["X-LLM-Source"] = self.settings.llm_source
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.picker_timeout_s, trust_env=False
            ) as client:
                resp = await client.post(
                    f"{self.settings.llm_base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                content = str(resp.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            log.warning("element picker unavailable (%s); step falls back", exc)
            return None, 0.0
        match = _FIRST_INT.search(content)
        if match is None:
            return None, 0.0
        idx = int(match.group())
        if not 0 <= idx < len(lines):
            return None, 0.0
        return idx, 1.0
