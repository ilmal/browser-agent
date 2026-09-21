"""PlannerClient wire-contract tests — no network, AsyncClient faked.

Mirrors test_llm_proxy_isolation.py: the faked client records how it was
constructed and called, so the tests pin the llm-service contract exactly:

* ``trust_env=False`` — an ambient HTTP_PROXY must never touch this call,
* ``Authorization: Bearer`` + ``X-LLM-Source`` headers,
* the ``user`` field (client id) in the body — llm-service rejects without it,
* model/temperature from settings,
* every transport/HTTP failure becomes PlannerUnavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent import planner as planner_mod  # noqa: E402
from browser_agent.config import load_settings  # noqa: E402
from browser_agent.planner import PlannerClient, PlannerUnavailable  # noqa: E402
from browser_agent.plan_model import PlanRejected, parse_plan  # noqa: E402

_COMPLETIONS = {"choices": [{"message": {"content": '{"entry_url":"https://x/","steps":[]}'}}]}


class _Resp:
    def __init__(self, status_code: int = 200, data: dict | None = None):
        self.status_code = status_code
        self._data = data if data is not None else _COMPLETIONS

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("POST", "http://llm.test/v1/chat/completions")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=req, response=httpx.Response(self.status_code)
            )

    def json(self):
        return self._data


class _FakeAsyncClient:
    instances: list["_FakeAsyncClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.post_calls: list[dict] = []
        _FakeAsyncClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self.post_calls.append({"url": url, "json": json, "headers": headers})
        return self._response

    _response = _Resp()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "llm_sk_test")
    monkeypatch.setenv("LLM_SOURCE", "browser-agent-test")
    monkeypatch.setenv("LLM_CLIENT_ID", "browser-agent-test")
    monkeypatch.setenv("PLANNER_MODEL", "deepseek-v4.1-flash")
    monkeypatch.setenv("PLANNER_CLIENT_ID", "browser-agent-planner")
    return load_settings()


@pytest.fixture
def fake_http(monkeypatch):
    _FakeAsyncClient.instances = []
    monkeypatch.setattr(planner_mod.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


async def test_request_contract(settings, fake_http):
    await PlannerClient(settings).plan("go to example.com")
    client = fake_http.instances[0]

    # In-cluster call: the egress proxy must never apply.
    assert client.kwargs["trust_env"] is False

    call = client.post_calls[0]
    assert call["url"] == "http://llm.test/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer llm_sk_test"
    assert call["headers"]["X-LLM-Source"] == "browser-agent-test"
    assert call["json"]["user"] == "browser-agent-planner"  # body, no header fallback
    assert call["json"]["model"] == "deepseek-v4.1-flash"
    assert call["json"]["temperature"] == 0
    assert call["json"]["messages"][-1]["content"] == "go to example.com"


async def test_returns_model_content(settings, fake_http):
    out = await PlannerClient(settings).plan("task")
    assert out == '{"entry_url":"https://x/","steps":[]}'


async def test_http_500_is_planner_unavailable(settings, fake_http):
    fake_http._response = _Resp(status_code=500)
    with pytest.raises(PlannerUnavailable, match="transport failed"):
        await PlannerClient(settings).plan("task")


async def test_connection_error_is_planner_unavailable(settings, fake_http, monkeypatch):
    async def _boom(self, url, json=None, headers=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(fake_http, "post", _boom)
    with pytest.raises(PlannerUnavailable):
        await PlannerClient(settings).plan("task")


async def test_empty_choices_is_planner_unavailable(settings, fake_http):
    fake_http._response = _Resp(data={"choices": []})
    with pytest.raises(PlannerUnavailable, match="no choices"):
        await PlannerClient(settings).plan("task")


def test_enabled_requires_all_three(settings, monkeypatch):
    assert PlannerClient(settings).enabled is True
    monkeypatch.setenv("LLM_API_KEY", "")
    assert PlannerClient(load_settings()).enabled is False
    monkeypatch.setenv("LLM_ENABLED", "false")
    assert PlannerClient(load_settings()).enabled is False


# -- parse_plan validation (what PlanTask does with the planner's text) ------


def _plan(**overrides) -> dict:
    base = {
        "entry_url": "https://x/",
        "steps": [{"action": "extract", "goal": "h", "selector": "h1"}],
    }
    base.update(overrides)
    return base


def test_parse_plan_strips_fences():
    raw = "```json\n" + json_dumps(_plan()) + "\n```"
    plan = parse_plan(raw)
    assert plan.entry_url == "https://x/"


def test_parse_plan_rejects_unknown_top_level_field():
    with pytest.raises(PlanRejected):
        parse_plan(_plan(sneaky="field"))


def test_parse_plan_rejects_unknown_step_field():
    step = {"action": "extract", "goal": "h", "selector": "h1", "xpath": "//h1"}
    with pytest.raises(PlanRejected):
        parse_plan(_plan(steps=[step]))


def test_parse_plan_rejects_non_json():
    with pytest.raises(PlanRejected):
        parse_plan("I will navigate to the site and click the button.")


def test_parse_plan_rejects_oversized():
    steps = [{"action": "extract", "goal": f"g{i}", "selector": "h1"} for i in range(5)]
    with pytest.raises(PlanRejected, match="steps"):
        parse_plan(json_dumps(_plan(steps=steps)), max_steps=4)


def test_parse_plan_rejects_empty_steps():
    with pytest.raises(PlanRejected):
        parse_plan(_plan(steps=[]), max_steps=12)


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj)
