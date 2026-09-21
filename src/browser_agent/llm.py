"""LLM fallback client.

Points at any OpenAI-compatible endpoint. In-cluster that is llm-service, so
agent steps cost no per-call fees. Kept deliberately thin: this is the layer
the agent escalates *to* when a deterministic selector no longer works.
"""

from __future__ import annotations

import logging
import time

import httpx

from .config import Settings

log = logging.getLogger(__name__)

#: How long a health result is reused. The admin UI polls /api/state, and the
#: probe is a real completion, so an uncached probe would make every poll wait
#: on the model — turning a status read into the slowest call in the app.
_HEALTH_TTL_S = 45.0
#: The probe is a liveness question, not a real task. Bound it well below the
#: completion timeout so a wedged model cannot stall a status read.
_HEALTH_TIMEOUT_S = 8.0


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._health: bool | None = None
        self._health_at = 0.0

    @property
    def enabled(self) -> bool:
        return self.settings.llm_enabled and bool(self.settings.llm_base_url)

    @property
    def configured(self) -> bool:
        """Enabled *and* able to authenticate.

        llm-service rejects a key-less call with 403, so "enabled" without a key
        is not a usable client. Reported distinctly so the UI can name the real
        cause instead of showing a reachable-looking pill over a broken path.
        """
        return self.enabled and bool(self.settings.llm_api_key)

    def _headers(self) -> dict[str, str]:
        # llm-service requires both: the key authenticates, and the source
        # header attributes the call (it defaults to "openai-compat" otherwise).
        return {
            "X-LLM-Source": self.settings.llm_source,
            "Authorization": f"Bearer {self.settings.llm_api_key}",
        }

    async def complete(
        self, prompt: str, *, system: str = "", max_tokens: int = 1024, timeout: float = 120.0
    ) -> str:
        """Single-turn completion. Raises on transport/HTTP failure."""
        if not self.configured:
            raise RuntimeError("llm fallback is not configured (LLM_ENABLED or LLM_API_KEY)")

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            # llm-service requires a client_id on every call and reads it from
            # the body only — the OpenAI-compatible `user` field, with no
            # header fallback.
            "user": self.settings.llm_client_id,
        }
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        # trust_env=False: the LLM lives in-cluster, so an ambient HTTP_PROXY
        # (egress proxy) must not be applied to this call.
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"llm returned no choices: {data}")
        return choices[0].get("message", {}).get("content", "") or ""

    async def healthy(self, *, max_age: float = _HEALTH_TTL_S) -> bool:
        """True when a *real* call would succeed, cached for ``max_age`` seconds.

        Probing an unauthenticated endpoint is not evidence: llm-service serves
        GET /models without auth, so a ping there returned 200 while every
        actual completion was rejected with 403. This asks the question the
        fallback actually depends on — a one-token completion — and caches the
        answer so a status poll does not wait on the model every time.
        """
        if not self.configured:
            return False
        fresh = self._health is not None and (time.monotonic() - self._health_at) < max_age
        if fresh:
            return bool(self._health)
        try:
            await self.complete("ping", max_tokens=1, timeout=_HEALTH_TIMEOUT_S)
            self._health = True
        except Exception:
            log.debug("llm health probe failed", exc_info=True)
            self._health = False
        self._health_at = time.monotonic()
        return self._health

    async def status(self) -> str:
        """One of ``off`` | ``no-key`` | ``ready`` | ``unreachable``."""
        if not self.enabled:
            return "off"
        if not self.settings.llm_api_key:
            return "no-key"
        return "ready" if await self.healthy() else "unreachable"
