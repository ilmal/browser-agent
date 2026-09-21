"""llm-service's caller contract must be satisfied on the wire.

llm-service rejects a token-spending call for three independent reasons: no key
(403 "Invalid API key."), no ``X-LLM-Source``, or no ``client_id`` in the body
(400). All three were missing at once, so every escalation failed with a 403
that the task layer reported as "the agent could not complete the task" — which
reads like the agent's fault, not a config fault.

Two traps make this easy to reintroduce:

* ``GET /models`` is served *without* auth, so probing it proves nothing. An
  earlier ``healthy()`` did exactly that and reported "ready" while every real
  completion was rejected. These tests assert the probe is authenticated.
* browser-use's ``ChatOpenAI`` cannot express a body field: it rejects an
  unknown ``client_id`` kwarg, its ``model_params`` is a closed set, and
  ``ainvoke(**kwargs)`` drops extras. The id has to be injected where the
  request is actually built.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from browser_agent.config import Settings
from browser_agent.llm import LLMClient


def _settings(**over) -> Settings:
    base = dict(
        profile="test",
        profiles_root=Path("/tmp/p"),
        data_root=Path("/tmp/d"),
        screen_width=1440,
        screen_height=900,
        screen_depth=24,
        novnc_port=6080,
        browser_base_url="",
        headless=True,
        slow_mo_ms=0,
        llm_base_url="http://llm.invalid/v1",
        llm_model="glm-4.5v",
        llm_enabled=True,
        llm_api_key="llm_sk_placeholder",
        llm_source="browser-agent",
        llm_client_id="browser-agent",
        planner_model="deepseek-v4.1-flash",
        planner_max_steps=12,
        planner_step_timeout_s=10,
        planner_client_id="browser-agent-planner",
        agent_max_steps=25,
        agent_timeout_s=600,
        laya_enabled=True,
        laya_decide_url="",
        laya_pick_enabled=True,
        laya_min_confidence=0.75,
        laya_max_candidates=20,
        laya_pick_retries=1,
        api_port=8000,
        control_token="t",
        ops_alert_url="",
        notify_on_escalation=False,
        browser_proxy="",
    )
    base.update(over)
    return Settings(**base)


class _Recorder:
    """Records every request and answers with an OpenAI-shaped body."""

    def __init__(self, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self._status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._status != 200:
            return httpx.Response(self._status, json={"detail": "Invalid API key."})
        return httpx.Response(200, json={"choices": [{"message": {"content": "PONG"}}]})

    @property
    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


def _patch(monkeypatch, rec: _Recorder) -> None:
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(rec)}),
    )


def test_complete_sends_key_source_and_client_id(monkeypatch):
    rec = _Recorder()
    _patch(monkeypatch, rec)

    asyncio.run(LLMClient(_settings()).complete("hello"))

    assert len(rec.requests) == 1
    req = rec.requests[0]

    assert req.headers["authorization"] == "Bearer llm_sk_placeholder"
    assert req.headers["x-llm-source"] == "browser-agent"

    body = json.loads(req.content)
    assert body["user"] == "browser-agent"
    # A body field, not a header: llm-service has no header fallback for it.
    assert not [k for k in req.headers if k.lower() in {"client-id", "client_id"}]


def test_healthy_makes_an_authenticated_call_not_a_models_ping(monkeypatch):
    """/models is unauthenticated, so it cannot be the health signal."""
    rec = _Recorder()
    _patch(monkeypatch, rec)

    assert asyncio.run(LLMClient(_settings()).healthy()) is True

    assert rec.paths == ["/v1/chat/completions"], f"probed the wrong endpoint: {rec.paths}"
    assert rec.requests[0].headers["authorization"] == "Bearer llm_sk_placeholder"


def test_healthy_is_false_when_the_call_is_forbidden(monkeypatch):
    """A 403 must read as unhealthy — that was the whole silent bug."""
    _patch(monkeypatch, _Recorder(status=403))

    assert asyncio.run(LLMClient(_settings()).healthy()) is False


def test_status_distinguishes_no_key_from_unreachable():
    assert asyncio.run(LLMClient(_settings(llm_enabled=False)).status()) == "off"
    assert asyncio.run(LLMClient(_settings(llm_api_key="")).status()) == "no-key"
    # Enabled, key present, but nothing listening on that port.
    assert asyncio.run(LLMClient(_settings(llm_base_url="http://127.0.0.1:1/v1")).status()) == (
        "unreachable"
    )


def test_complete_refuses_without_a_key():
    """Fail with the cause instead of letting llm-service answer 403."""
    with pytest.raises(RuntimeError, match="not configured"):
        asyncio.run(LLMClient(_settings(llm_api_key="")).complete("hello"))

def test_healthy_is_cached_so_polling_does_not_wait_on_the_model(monkeypatch):
    """The UI polls /api/state; an uncached probe would make that the slowest call."""
    rec = _Recorder()
    _patch(monkeypatch, rec)
    client = LLMClient(_settings())

    asyncio.run(client.healthy())
    asyncio.run(client.healthy())
    asyncio.run(client.healthy())

    assert len(rec.requests) == 1, "health probe was not reused"

