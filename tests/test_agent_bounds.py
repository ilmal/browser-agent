"""The agent fallback must be bounded.

A freeform task has no deterministic path, so the agent *is* the
implementation. browser-use's ``Agent.run`` defaults ``max_steps`` to 500, and
nothing overrode it: on a page with nothing left to do the model keeps choosing
an action, the run never returns, and because the queue has a single worker
every later task sits QUEUED behind it forever. One stuck freeform task took the
whole control plane's task path down.

These tests pin the bound at the seam the runner actually calls, so removing it
fails here rather than in production.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent import agent as agent_mod  # noqa: E402


class _FakeHistory:
    def is_successful(self) -> bool:
        return True

    def has_errors(self) -> bool:
        return False

    def errors(self):
        return []

    def final_result(self):
        return "did the thing"


class _FakeAgent:
    """Records how it was run, and can hang to exercise the wall clock."""

    def __init__(self, *, task, llm, browser):
        self.task = task
        self.llm = llm
        self.browser = browser
        _FakeAgent.last = self
        self.run_calls: list[int] = []

    async def run(self, max_steps: int = 500):
        self.run_calls.append(max_steps)
        if _FakeAgent.hang:
            await asyncio.sleep(3600)
        return _FakeHistory()

    hang = False
    last = None


class _FakeBrowser:
    def __init__(self, **kwargs):
        pass

    async def connect(self):
        pass

    async def close(self):
        pass


class _FakeLLM:
    pass


class _FakePage:
    url = "https://example.com/after"


class _Session:
    cdp_endpoint = "ws://127.0.0.1:9222/devtools/browser/fake"

    async def page(self):
        return _FakePage()


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_PROFILE", "pytest")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "llm_sk_placeholder")
    from browser_agent.config import load_settings

    settings = load_settings()
    monkeypatch.setattr(
        agent_mod,
        "_load_browser_use",
        lambda: (_FakeAgent, _FakeBrowser, object),
    )
    monkeypatch.setattr(agent_mod, "_build_gateway_llm", lambda s, c: _FakeLLM())
    # The post-run challenge check needs a real page; these tests assert on the
    # bound, so stop the run at the history.
    async def _no_challenge(page):
        return None

    monkeypatch.setattr(agent_mod, "detect_challenge", _no_challenge)
    _FakeAgent.hang = False
    return settings


@pytest.mark.asyncio
async def test_run_is_capped_by_agent_max_steps(patched, monkeypatch):
    """The configured step cap reaches Agent.run — not browser-use's 500."""
    monkeypatch.setenv("AGENT_MAX_STEPS", "7")
    from browser_agent.config import load_settings

    settings = load_settings()
    runner = agent_mod.make_agent_runner(settings)

    await runner(_Session(), "https://example.com", {"goal": "scroll"})

    assert _FakeAgent.last.run_calls == [7]


@pytest.mark.asyncio
async def test_run_is_not_unbounded_by_default(patched):
    """Even with nothing configured the cap is finite, and well under 500."""
    runner = agent_mod.make_agent_runner(patched)

    await runner(_Session(), "https://example.com", {"goal": "scroll"})

    (used,) = _FakeAgent.last.run_calls
    assert used < 500, "the fallback fell back to browser-use's unbounded default"


@pytest.mark.asyncio
async def test_a_hung_run_ends_instead_of_blocking_the_queue(patched, monkeypatch):
    """A step that never returns must surface as a failure, not wedge the worker."""
    monkeypatch.setenv("AGENT_TIMEOUT_S", "1")
    from browser_agent.config import load_settings

    settings = load_settings()
    _FakeAgent.hang = True
    try:
        runner = agent_mod.make_agent_runner(settings)
        with pytest.raises(RuntimeError, match="budget"):
            await asyncio.wait_for(
                runner(_Session(), "https://example.com", {"goal": "scroll"}),
                timeout=15,
            )
    finally:
        _FakeAgent.hang = False
