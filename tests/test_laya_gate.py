"""LayaGate unit tests — everything runs against a stubbed classifier.

The real weights are never downloaded here: ``_load`` is monkeypatched. What
these tests pin is the gate's contract with the plan executor:

* a working gate parses noul/choice answers into ``(verdict, confidence)``,
* any load or inference failure degrades to off/inconclusive — never raises,
* a broken gate loads exactly once (log-once, then stays off),
* the settings switches disable both roles outright.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent.config import load_settings  # noqa: E402
from browser_agent.laya_gate import LayaGate  # noqa: E402


class FakeAgent:
    """Mimics laya.Agent.predict for the two question shapes the gate uses."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict]] = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            raise RuntimeError("inference exploded")
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"noul": 0.9, "confidence": 0.88}
            else:
                answers[qid] = {"choice": "1", "confidence": 0.77}
        return {"answers": answers}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("LAYA_ENABLED", "true")
    monkeypatch.setenv("LAYA_PICK_ENABLED", "true")
    return load_settings()


def test_yes_no_parses_noul(settings):
    agent = FakeAgent()
    gate = LayaGate(settings)
    gate._load = lambda: agent  # type: ignore[method-assign]

    verdict, conf = _run(gate.yes_no("did the step succeed?", "page text"))

    assert verdict is True
    # Confidence is derived from the answer itself: distance from coin-flip.
    assert conf == pytest.approx(0.9)
    assert gate.stats["confirms"] == 1
    assert gate.stats["calls"] == 1
    # The noul question carries the page state and yes/no criteria.
    state, questions = agent.calls[0]
    assert state == "page text"
    assert questions["v"]["criteria"] == {"true": "yes", "false": "no"}


def test_yes_no_confident_false_is_confident(settings):
    agent = FakeAgent()
    gate = LayaGate(settings)
    gate._load = lambda: agent  # type: ignore[method-assign]

    # p=0.05 must read as a CONFIDENT no, not an inconclusive answer.
    async def fake_predict(state, questions):
        return {"answers": {"v": {"noul": 0.05, "confidence": 0.05}}}

    gate._predict = fake_predict  # type: ignore[method-assign]
    verdict, conf = _run(gate.yes_no("q?", "s"))
    assert verdict is False
    assert conf == pytest.approx(0.95)


def test_choose_parses_choice_index(settings):
    agent = FakeAgent()
    gate = LayaGate(settings)
    gate._load = lambda: agent  # type: ignore[method-assign]

    idx, conf = _run(gate.choose("which button?", ["0. b1", "1. b2"], "state"))

    assert idx == 1
    assert conf == pytest.approx(0.77)
    _, questions = agent.calls[0]
    assert questions["pick"]["criteria"] == {"0": "0. b1", "1": "1. b2"}


def test_broken_load_disables_both_roles_once(settings, caplog):
    loads = {"n": 0}

    def _boom():
        loads["n"] += 1
        raise ImportError("no laya installed")

    gate = LayaGate(settings)
    gate._load = _boom  # type: ignore[method-assign]

    assert gate.enabled is True  # not yet known to be broken

    verdict, conf = _run(gate.yes_no("q?", "s"))
    assert verdict is None and conf == 0.0

    picked, pconf = _run(gate.choose("q?", ["0. a"], "s"))
    assert picked is None and pconf == 0.0

    # Broken is sticky: the load was attempted exactly once, then never again.
    assert loads["n"] == 1
    assert gate.enabled is False
    assert gate.pick_enabled is False
    assert gate.summary["state"] == "broken"
    # ``choose`` refuses on ``enabled`` (broken), NOT on ``pick_enabled``: the
    # flag governs the plan picker, and the minesweeper tiebreak calls choose
    # with the flag off. Both refusals look the same from here, which is why
    # test_choose_ignores_the_plan_pick_flag pins the distinction.
    assert gate.stats["inconclusive"] == 1


def test_predict_failure_is_inconclusive_not_fatal(settings):
    agent = FakeAgent(fail=True)
    gate = LayaGate(settings)
    gate._load = lambda: agent  # type: ignore[method-assign]

    verdict, conf = _run(gate.yes_no("q?", "s"))
    assert verdict is None and conf == 0.0
    assert gate.stats["inconclusive"] == 1
    assert gate.stats["calls"] == 0  # only successful predicts count as calls


def test_choose_out_of_range_is_inconclusive(settings, monkeypatch):
    agent = FakeAgent()
    # Force the fake to answer with an index past the end of the list.
    monkeypatch.setattr(
        type(agent), "predict", lambda self, state, questions: {
            "answers": {qid: {"choice": "99", "confidence": 0.9} for qid in questions}
        }
    )
    gate = LayaGate(settings)
    gate._load = lambda: agent  # type: ignore[method-assign]

    picked, conf = _run(gate.choose("q?", ["0. a", "1. b"], "s"))
    assert picked is None
    assert conf == pytest.approx(0.9)
    assert gate.stats["inconclusive"] == 1


def test_settings_disable_both_roles(settings, monkeypatch):
    monkeypatch.setenv("LAYA_ENABLED", "false")
    gate = LayaGate(load_settings())
    gate._load = lambda: pytest.fail("must not load when disabled")  # type: ignore[method-assign]

    assert gate.enabled is False
    assert gate.pick_enabled is False


def test_choose_ignores_the_plan_pick_flag(settings, monkeypatch):
    """The flag is the plan picker's policy, not the gate's.

    Turning PICK off must not blind the minesweeper tiebreak, which shares this
    method but has a different risk profile. The plan picker still refuses —
    it checks the flag itself — so this pins that the *method* is willing.
    """
    monkeypatch.setenv("LAYA_PICK_ENABLED", "false")
    agent = FakeAgent()
    gate = LayaGate(load_settings())
    gate._load = lambda: agent  # type: ignore[method-assign]

    assert gate.pick_enabled is False  # the plan picker's gate is still shut
    idx, conf = _run(gate.choose("which?", ["0. a", "1. b"], "s"))
    assert idx == 1 and conf > 0.0  # …but the tiebreak can still ask

    # And the plan picker honours the flag: with no selector and the flag off,
    # it refuses before ever reaching ``choose``.
    import asyncio

    from browser_agent.recipes._plan_exec import StepFailure, resolve_target
    from browser_agent.plan_model import Step

    monkeypatch.setattr(
        LayaGate, "choose",
        lambda *a, **k: pytest.fail("plan picker asked with the flag off"),
    )
    step = Step(action="click", goal="press the thing", selector=None)
    try:
        asyncio.run(resolve_target(None, step, gate, load_settings(), base="body"))
    except StepFailure as exc:
        assert "no picker is enabled" in str(exc)
    else:
        pytest.fail("plan picker resolved a target with the flag off")


# -- HTTP backend (LAYA_DECIDE_URL set — the prod shape) ---------------------
#
# Mirrors test_planner_client.py: a fake AsyncClient records how it was built
# and called, pinning the llm-service caller contract on the /v1/decide route
# and the measured ggmlc-engine answer shapes (raw noul p, near-zero
# act-confidence with the real distribution in `probabilities`).

from browser_agent import laya_gate as laya_gate_mod  # noqa: E402

_DECIDE = "http://decide.test/v1/decide"


class _Resp:
    def __init__(self, status_code: int = 200, data: dict | None = None):
        self.status_code = status_code
        self._data = data if data is not None else {"answers": {}}

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("POST", _DECIDE)
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
def http_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "llm_sk_test")
    monkeypatch.setenv("LLM_SOURCE", "browser-agent-test")
    monkeypatch.setenv("LLM_CLIENT_ID", "browser-agent-test")
    monkeypatch.setenv("LAYA_ENABLED", "true")
    monkeypatch.setenv("LAYA_PICK_ENABLED", "true")
    monkeypatch.setenv("LAYA_DECIDE_URL", _DECIDE)
    return load_settings()


@pytest.fixture
def fake_http(monkeypatch):
    _FakeAsyncClient.instances = []
    monkeypatch.setattr(laya_gate_mod.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


async def test_http_mode_never_touches_the_local_backend(http_settings, fake_http):
    gate = LayaGate(http_settings)
    gate._load = lambda: pytest.fail("weights must not load in HTTP mode")  # type: ignore[method-assign]

    # Empty answers body → inconclusive, but via HTTP, not the local model.
    verdict, conf = await gate.yes_no("q?", "s")
    assert verdict is None and conf == 0.0
    assert gate._state == "unloaded"
    assert gate.summary["state"] == "http"
    assert gate.enabled is True


async def test_http_contract_key_source_user_model_and_no_proxy(http_settings, fake_http):
    fake_http._response = _Resp(
        data={"answers": {"v": {"noul": 0.5614, "confidence": 0.5614}}}
    )
    gate = LayaGate(http_settings)
    verdict, conf = await gate.yes_no("did it work?", "page text")

    assert verdict is True
    # The ggmlc engine reports raw p as the noul confidence, not max(p, 1-p).
    assert conf == pytest.approx(0.5614)

    client = fake_http.instances[0]
    assert client.kwargs["trust_env"] is False
    # Absorbs the first-call family cold load (2-10 s), not the warm call.
    assert client.kwargs["timeout"] == pytest.approx(30.0)

    call = client.post_calls[0]
    assert call["url"] == _DECIDE
    assert call["headers"]["Authorization"] == "Bearer llm_sk_test"
    assert call["headers"]["X-LLM-Source"] == "browser-agent-test"
    assert call["json"]["client_id"] == "browser-agent-test"  # decide route: no `user` alias
    assert call["json"]["model"] == "english"
    assert call["json"]["state"] == "page text"
    assert call["json"]["questions"]["v"]["type"] == "noul"
    assert call["json"]["questions"]["v"]["criteria"] == {"true": "yes", "false": "no"}


async def test_http_state_is_truncated_to_the_model_budget(http_settings, fake_http):
    gate = LayaGate(http_settings)
    await gate.yes_no("q?", "x" * 5000)
    body = fake_http.instances[0].post_calls[0]["json"]
    assert len(body["state"]) == 900


async def test_http_without_key_sends_a_bare_engine_payload(tmp_path, monkeypatch, fake_http):
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("LAYA_ENABLED", "true")
    monkeypatch.setenv("LAYA_DECIDE_URL", _DECIDE)
    gate = LayaGate(load_settings())

    await gate.yes_no("q?", "s")

    call = fake_http.instances[0].post_calls[0]
    assert "Authorization" not in call["headers"]
    assert "X-LLM-Source" not in call["headers"]
    assert "client_id" not in call["json"]


async def test_http_choice_uses_the_chosen_labels_probability(http_settings, fake_http):
    fake_http._response = _Resp(
        data={
            "answers": {
                "pick": {
                    "choice": "2",
                    "confidence": 0.0035,  # engine's act-confidence: near-zero, ignored
                    "probabilities": {"1": 0.465, "2": 0.535},
                }
            }
        }
    )
    gate = LayaGate(http_settings)
    idx, conf = await gate.choose("which?", ["0. a", "1. b", "2. c"], "state")

    assert idx == 2
    assert conf == pytest.approx(0.535)
    assert gate.stats["picks"] == 1
    body = fake_http.instances[0].post_calls[0]["json"]
    assert body["questions"]["pick"]["type"] == "choice"
    assert body["questions"]["pick"]["criteria"] == {"0": "0. a", "1": "1. b", "2": "2. c"}


async def test_http_500_is_inconclusive(http_settings, fake_http):
    fake_http._response = _Resp(status_code=503)
    gate = LayaGate(http_settings)

    verdict, conf = await gate.yes_no("q?", "s")

    assert verdict is None and conf == 0.0
    assert gate.stats["inconclusive"] == 1
    assert gate.stats["calls"] == 0


async def test_http_missing_answers_is_inconclusive(http_settings, fake_http):
    fake_http._response = _Resp(data={"ok": True})
    gate = LayaGate(http_settings)

    verdict, conf = await gate.yes_no("q?", "s")

    assert verdict is None and conf == 0.0
    assert gate.stats["inconclusive"] == 1


async def test_http_malformed_noul_is_inconclusive(http_settings, fake_http):
    fake_http._response = _Resp(data={"answers": {"v": {"confidence": 0.9}}})
    gate = LayaGate(http_settings)

    verdict, conf = await gate.yes_no("q?", "s")

    assert verdict is None and conf == 0.0
    assert gate.stats["inconclusive"] == 1


def test_decide_url_set_disables_cleanly(http_settings, monkeypatch):
    monkeypatch.setenv("LAYA_ENABLED", "false")
    gate = LayaGate(load_settings())
    assert gate.enabled is False
    assert gate.pick_enabled is False


def _run(coro):
    import asyncio

    return asyncio.run(coro)
