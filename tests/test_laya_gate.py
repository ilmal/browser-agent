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
    assert conf == pytest.approx(0.88)
    assert gate.stats["confirms"] == 1
    assert gate.stats["calls"] == 1
    # The noul question carries the page state and yes/no criteria.
    state, questions = agent.calls[0]
    assert state == "page text"
    assert questions["v"]["criteria"] == {"true": "yes", "false": "no"}


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
    # The yes_no went through predict (inconclusive); choose refused at the door.
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


def _run(coro):
    import asyncio

    return asyncio.run(coro)
