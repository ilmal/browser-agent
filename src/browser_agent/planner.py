"""The planner client: freeform task in, JSON plan text out.

Transport only — parsing/validation lives in :mod:`plan_model`. Points at the
same llm-service endpoint and client key as the fallback, with its own model
(a fast text model; the planner never needs to look at a screenshot) and its
own client id so llm-service's call log can tell the two apart.
"""

from __future__ import annotations

import logging

import httpx

from .config import Settings

log = logging.getLogger(__name__)

PLANNER_TIMEOUT_S = 45.0

PROMPT = """\
You convert a freeform browser task into a JSON plan for a deterministic executor.

Return ONLY minified JSON. No markdown fences, no prose, no comments.
Schema:
{"entry_url":"https://...","steps":[{"action":"navigate|click|type|extract|wait",
"goal":"what this step does","selector":"css or null","text":"url or text to type or null",
"done_when":{"url_contains":"...","selector_visible":"css","text_contains":"..."}}]}
Rules:
- 3-12 steps, ordered; the first step navigates to entry_url.
- Describe targets semantically in "goal" (e.g. "open the invitations manager"); leave
  "selector" null unless you are certain of a stable CSS selector (data-testid, id, aria-label).
- "type" carries the literal text to enter in "text". "extract" and "wait" REQUIRE "selector".
- End with a step whose done_when proves the overall goal (url_contains or text_contains).
- Never plan login, captcha, payment, password/email changes, or account creation.
"""


class PlannerUnavailable(RuntimeError):
    """The planner endpoint is down or unconfigured.

    Unlike PlanRejected this hands the task to the agent fallback: the planner
    is an accelerator, and a transport hiccup must not turn an achievable task
    into a failure.
    """


class PlannerClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return (
            self.settings.llm_enabled
            and bool(self.settings.llm_base_url)
            and bool(self.settings.llm_api_key)
        )

    def _headers(self) -> dict[str, str]:
        return {
            "X-LLM-Source": self.settings.llm_source,
            "Authorization": f"Bearer {self.settings.llm_api_key}",
        }

    async def plan(self, task: str) -> str:
        """One planner call. Raises PlannerUnavailable on any transport failure."""
        if not self.enabled:
            raise PlannerUnavailable("planner is not configured (LLM_ENABLED or LLM_API_KEY)")

        payload = {
            "model": self.settings.planner_model,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": task},
            ],
            "temperature": 0,
            "max_tokens": 2048,
            # llm-service requires a client id in the body (the OpenAI-compatible
            # `user` field, no header fallback).
            "user": self.settings.planner_client_id,
        }
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        # trust_env=False: llm-service lives in-cluster, so an ambient
        # HTTP_PROXY (the egress proxy) must never be applied to this call.
        async with httpx.AsyncClient(timeout=PLANNER_TIMEOUT_S, trust_env=False) as client:
            try:
                resp = await client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("planner call failed: %s", exc)
                raise PlannerUnavailable(f"planner transport failed: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise PlannerUnavailable(f"planner returned no choices: {data}")
        return choices[0].get("message", {}).get("content", "") or ""
