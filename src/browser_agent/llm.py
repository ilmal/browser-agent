"""LLM fallback client.

Points at any OpenAI-compatible endpoint. In-cluster that is llm-service, so
agent steps cost no per-call fees. Kept deliberately thin: this is the layer
the agent escalates *to* when a deterministic selector no longer works.
"""

from __future__ import annotations

import logging

import httpx

from .config import Settings

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.llm_enabled and bool(self.settings.llm_base_url)

    async def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> str:
        """Single-turn completion. Raises on transport/HTTP failure."""
        if not self.enabled:
            raise RuntimeError("llm fallback disabled")

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        url = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"llm returned no choices: {data}")
        return choices[0].get("message", {}).get("content", "") or ""

    async def healthy(self) -> bool:
        """True when the endpoint answers. Used by /healthz, never fatal."""
        try:
            url = self.settings.llm_base_url.rstrip("/") + "/models"
            async with httpx.AsyncClient(timeout=5) as client:
                return (await client.get(url)).status_code < 500
        except Exception:
            return False
