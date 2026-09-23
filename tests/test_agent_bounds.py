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


class _FakeState:
    """The slice of browser-use's AgentState the live-view hooks touch."""

    n_steps = 1
    last_model_output = None
    last_result: list = []


class _FakeAgent:
    """Records how it was run, and can hang to exercise the wall clock."""

    def __init__(self, *, task, llm, browser):
        self.task = task
        self.llm = llm
        self.browser = browser
        _FakeAgent.last = self
        self.run_calls: list[int] = []
        # What the live-view hooks read: browser-use's per-step state.
        self.state = _FakeState()

    async def run(self, max_steps: int = 500, on_step_start=None, on_step_end=None):
        self.run_calls.append(max_steps)
        # The live-view hooks are passed through on every run; the agent calls
        # them between steps, so mirror that here.
        if on_step_start is not None:
            await on_step_start(self)
        if on_step_end is not None:
            await on_step_end(self)
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


# --- the stall latch -------------------------------------------------------
#
# The step cap bounds a runaway but is a terrible way to stop one: a run with
# nothing left to do spends one slow model call per step until it runs out.
# The recorded 26-step trace spent 8 of its 13 model calls on a Done/Search
# loop over a page that never changed. These tests pin the earlier stop.


class _CountingPage:
    """A page whose fingerprint the test controls.

    ``page_state.fingerprint`` reads ``page.url``, the title, the body text and
    the control state through Playwright, so a fake only has to satisfy those
    four calls — and returning a different ``inner_text`` is how a test says
    "the page changed".
    """

    url = "https://example.com/flights"

    def __init__(self, body="form"):
        self.body = body

    async def title(self):
        return "Flights"

    async def inner_text(self, _sel):
        return self.body

    async def evaluate(self, _js):
        return ""


class _LatchSession(_Session):
    def __init__(self, page):
        self._page = page

    async def page(self):
        return self._page


class _StallAgent(_FakeAgent):
    """Drives N steps, optionally changing the page between them.

    Mirrors browser-use's real ordering: ``on_step_start`` before the step,
    ``on_step_end`` after it. The page body mutates *during* the step, so a
    step that changed the page reads as changed at ``on_step_end`` — which is
    what the latch keys on.
    """

    plan: list = []  # per-step: the page body after that step finishes
    actions: list = []  # per-step: the browser-use action names recorded
    page = None  # the _CountingPage the runner reads through the session

    async def run(self, max_steps: int = 500, on_step_start=None, on_step_end=None):
        self.run_calls.append(max_steps)
        for i, body in enumerate(type(self).plan):
            self.state.n_steps = i + 1
            self.state.last_result = [_FakeResult(n) for n in type(self).actions[i]]
            if on_step_start is not None:
                await on_step_start(self)
            type(self).page.body = body
            if on_step_end is not None:
                await on_step_end(self)
        return _FakeHistory()


class _FakeResult:
    def __init__(self, name):
        self.name = name
        self.error = None
        self.is_done = False
        self.extracted_content = None


async def _stall_run(patched, monkeypatch, plan, actions, payload=None):
    """Run the agent over a scripted plan of page bodies and action names.

    Laya is forced off: the fingerprint count is the authority and must stop a
    runaway with no model reachable at all.
    """
    page = _CountingPage()
    _StallAgent.plan = plan
    _StallAgent.actions = actions
    _StallAgent.page = page
    monkeypatch.setattr(
        agent_mod, "_load_browser_use", lambda: (_StallAgent, _FakeBrowser, object)
    )
    monkeypatch.setenv("LAYA_ENABLED", "false")
    from browser_agent.config import load_settings

    runner = agent_mod.make_agent_runner(load_settings())
    body = {"goal": "find a flight"}
    body.update(payload or {})
    return await runner(_LatchSession(page), "https://example.com/flights", body)


@pytest.mark.asyncio
async def test_three_no_change_clicks_end_the_run(patched, monkeypatch):
    """The recorded failure: repeated clicks, the page never moves, stop.

    Five planned steps, because the latch fires at the *next* step's start:
    the third no-change step is the evidence, and the step after it is where
    the run ends — one model call is saved by not making it.
    """
    with pytest.raises(RuntimeError, match="stalled"):
        await _stall_run(
            patched, monkeypatch, ["form"] * 5, [["click_element_by_index"]] * 5
        )


@pytest.mark.asyncio
async def test_waits_do_not_count_as_stalling(patched, monkeypatch):
    """Waiting for a slow page is legitimate and must not end the run."""
    plan = ["form"] * 4 + ["results"]
    actions = [["wait"]] * 4 + [["click_element_by_index"]]
    result = await _stall_run(patched, monkeypatch, plan, actions)
    assert result["agent_result"] == "did the thing"


@pytest.mark.asyncio
async def test_a_page_that_changes_resets_the_counter(patched, monkeypatch):
    """Two no-change clicks, then progress, then two more: never stalls."""
    plan = ["form", "form", "form", "results", "results", "results"]
    actions = [["click_element_by_index"]] * 6
    result = await _stall_run(patched, monkeypatch, plan, actions)
    assert result["agent_result"] == "did the thing"


@pytest.mark.asyncio
async def test_payload_max_steps_overrides_the_configured_cap(patched, monkeypatch):
    """A task may ask for a smaller budget; every step is a slow model call."""
    plan = ["form", "results"]
    actions = [["click_element_by_index"]] * 2
    await _stall_run(patched, monkeypatch, plan, actions, {"max_steps": 4})
    assert _StallAgent.last.run_calls == [4]


@pytest.mark.asyncio
async def test_payload_max_steps_cannot_exceed_the_configured_cap(patched, monkeypatch):
    """The cap is what bounds a runaway; a payload must not raise it."""
    plan = ["form", "results"]
    actions = [["click_element_by_index"]] * 2
    await _stall_run(patched, monkeypatch, plan, actions, {"max_steps": 9999})
    (used,) = _StallAgent.last.run_calls
    assert used == patched.agent_max_steps
