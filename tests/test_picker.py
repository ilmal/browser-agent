"""ElementPicker wire-contract tests — no network, AsyncClient faked.

Same shape as test_planner_client.py: the faked client records how it was
constructed and called, so the tests pin the llm-service contract exactly and
prove every failure mode reads as "no pick", never an exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent import picker as picker_mod  # noqa: E402
from browser_agent.config import load_settings  # noqa: E402
from browser_agent.picker import ElementPicker  # noqa: E402

_LINES = ["0. <a> 'Home' href=/", "1. <a> 'Pricing' href=/pricing"]

_COMPLETIONS = {"choices": [{"message": {"content": "1"}}]}


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
    monkeypatch.setenv("PICKER_ENABLED", "true")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "llm_sk_test")
    monkeypatch.setenv("LLM_SOURCE", "browser-agent-test")
    monkeypatch.setenv("PICKER_CLIENT_ID", "browser-agent-picker-test")
    monkeypatch.setenv("PICKER_MODEL", "deepseek-v4.1-flash")
    return load_settings()


@pytest.fixture
def fake_http(monkeypatch):
    _FakeAsyncClient.instances = []
    monkeypatch.setattr(picker_mod.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


async def test_request_contract(settings, fake_http):
    idx, conf = await ElementPicker(settings).pick("open pricing", "Goal: open pricing", _LINES)
    assert (idx, conf) == (1, 1.0)
    client = fake_http.instances[0]

    # In-cluster call: the egress proxy must never apply.
    assert client.kwargs["trust_env"] is False
    call = client.post_calls[0]
    assert call["url"] == "http://llm.test/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer llm_sk_test"
    assert call["headers"]["X-LLM-Source"] == "browser-agent-test"
    body = call["json"]
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["temperature"] == 0
    assert body["user"] == "browser-agent-picker-test"
    assert "0. <a> 'Home'" in body["messages"][1]["content"]
    assert "Goal: open pricing" in body["messages"][1]["content"]


async def test_disabled_makes_no_call(tmp_path, monkeypatch, fake_http):
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("PICKER_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "llm_sk_test")
    picker = ElementPicker(load_settings())
    assert picker.enabled is False
    assert await picker.pick("goal", "state", _LINES) == (None, 0.0)
    assert fake_http.instances == []


async def test_enabled_without_key_makes_no_call(tmp_path, monkeypatch, fake_http):
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("PICKER_ENABLED", "true")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    picker = ElementPicker(load_settings())
    assert picker.enabled is False
    assert await picker.pick("goal", "state", _LINES) == (None, 0.0)
    assert fake_http.instances == []


async def test_garbled_answer_is_no_pick(settings, fake_http):
    fake_http._response = _Resp(data={"choices": [{"message": {"content": "the Pricing link"}}]})
    assert await ElementPicker(settings).pick("g", "s", _LINES) == (None, 0.0)


async def test_out_of_range_index_is_no_pick(settings, fake_http):
    fake_http._response = _Resp(data={"choices": [{"message": {"content": "7"}}]})
    assert await ElementPicker(settings).pick("g", "s", _LINES) == (None, 0.0)


async def test_http_500_is_no_pick(settings, fake_http):
    fake_http._response = _Resp(status_code=500)
    assert await ElementPicker(settings).pick("g", "s", _LINES) == (None, 0.0)


async def test_transport_error_is_no_pick(settings, fake_http, monkeypatch):
    async def boom(url, json=None, headers=None):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(_FakeAsyncClient, "post", boom)
    assert await ElementPicker(settings).pick("g", "s", _LINES) == (None, 0.0)


async def test_empty_lines_makes_no_call(settings, fake_http):
    assert await ElementPicker(settings).pick("g", "s", []) == (None, 0.0)
    assert fake_http.instances == []
