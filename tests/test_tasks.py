"""Task-runner integration tests.

Proves the run loop's central promises:
  * a successful recipe ends DONE,
  * a recipe that hits a challenge ends BLOCKED and is NOT retried,
  * a broken recipe falls through to the agent exactly once,
  * a failed task never silently reports success.
"""

from __future__ import annotations

import http.server
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent import recipes  # noqa: E402,F401  (registers built-ins)
from browser_agent.escalation import EscalationRequired  # noqa: E402
from browser_agent.tasks import (  # noqa: E402
    AGENT_RECIPE,
    TaskRunner,
    TaskStatus,
    register,
)

CAPTCHA_PAGE = """<html><body><h1>Verify</h1>
<img src="/captcha.png" class="captcha-image"></body></html>"""

OK_PAGE = """<html><body><h1>Composer</h1>
<div role="textbox" contenteditable="true" id="box"></div></body></html>"""


def _start_url() -> str:
    """A real page for freeform tests: the start url is now required."""
    return "about:blank#start"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def site():
    pages = {"/ok": OK_PAGE, "/captcha": CAPTCHA_PAGE}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = pages.get(self.path, "<html><body>x</body></html>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    port = _free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE", "pytest")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("HEADLESS", "true")
    # The agent fallback is gated on the LLM being *configured* — enabled and
    # carrying a key, since an unauthenticated call is rejected. Tests inject a
    # stub agent runner, so no call is made, but the gate must be open.
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "llm_sk_placeholder")
    monkeypatch.setenv("NOTIFY_ON_ESCALATION", "false")
    from browser_agent.config import load_settings

    return load_settings()


class _Session:
    """A BrowserSession factory that does not use a persistent profile.

    Tests run against a throwaway headless context; the persistent-profile path
    is exercised in the container, where a real display exists.
    """

    def __init__(self, pw, browser):
        self._pw = pw
        self._browser = browser
        self._ctx = None

    async def start(self):
        if self._ctx is None:
            self._ctx = await self._browser.new_context()
        return self._ctx

    async def page(self):
        ctx = await self.start()
        return ctx.pages[0] if ctx.pages else await ctx.new_page()

    async def goto(self, url, *, wait_until="domcontentloaded"):
        page = await self.page()
        await page.goto(url, wait_until=wait_until)
        return page

    async def stop(self):
        if self._ctx:
            await self._ctx.close()


@pytest.fixture
async def runner_factory(settings, site):
    """Builds a TaskRunner over a throwaway context, plus a call counter."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        made: list[TaskRunner] = []

        def make(agent_runner=None):
            session = _Session(pw, browser)
            r = TaskRunner(settings, session, agent_runner=agent_runner)
            made.append(r)
            return r

        try:
            yield make, site
        finally:
            for r in made:
                await r.stop()
            await browser.close()


async def _drain(runner: TaskRunner, task_id: str, timeout: float = 30) -> Any:
    """Wait for a queued task to reach a terminal state."""
    import asyncio

    runner.start()
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        task = runner.tasks[task_id]
        if task.status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}:
            return task
        await asyncio.sleep(0.05)
    raise TimeoutError(task_id)


class _OkRecipe:
    name = "test.ok"
    description = "succeeds"
    entry_url = ""

    async def run(self, session, payload):
        return {"did": "the thing"}


class _BlockedRecipe:
    name = "test.blocked"
    description = "hits a challenge"
    entry_url = ""

    async def run(self, session, payload):
        page = await session.page()
        from browser_agent.recipes._helpers import require_clear

        await require_clear(page)
        return {"unreachable": True}


class _BrokenRecipe:
    name = "test.broken"
    description = "raises"
    entry_url = ""

    async def run(self, session, payload):
        raise RuntimeError("selector moved")


@pytest.mark.asyncio
async def test_successful_recipe_completes(runner_factory, site):
    make, site_url = runner_factory
    _OkRecipe.entry_url = f"{site_url}/ok"
    register(_OkRecipe())
    runner = make()
    task = runner.submit("test.ok", {})
    done = await _drain(runner, task.id)
    assert done.status is TaskStatus.DONE
    assert done.result == {"did": "the thing"}
    assert done.used_agent is False


@pytest.mark.asyncio
async def test_challenge_blocks_and_does_not_retry(runner_factory, site):
    """The central guarantee: a challenge ends the task, it does not loop."""
    make, site_url = runner_factory
    _BlockedRecipe.entry_url = f"{site_url}/captcha"
    register(_BlockedRecipe())

    calls = {"n": 0}

    def counting_agent(session, url, payload):
        calls["n"] += 1

        async def _run():
            return {"agent": True}

        return _run()

    runner = make(agent_runner=counting_agent)
    task = runner.submit("test.blocked", {})
    blocked = await _drain(runner, task.id)

    assert blocked.status is TaskStatus.BLOCKED
    assert "captcha" in blocked.detail.lower()
    # The agent must NOT have been invoked: a challenge is a human's job.
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_broken_recipe_uses_agent_once(runner_factory, site):
    make, site_url = runner_factory
    _BrokenRecipe.entry_url = f"{site_url}/ok"
    register(_BrokenRecipe())

    calls = {"n": 0}

    async def agent(session, url, payload):
        calls["n"] += 1
        return {"agent": True}

    runner = make(agent_runner=agent)
    task = runner.submit("test.broken", {})
    finished = await _drain(runner, task.id)

    assert finished.status is TaskStatus.DONE
    assert finished.used_agent is True
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_broken_recipe_without_agent_fails(runner_factory, site):
    make, site_url = runner_factory
    _BrokenRecipe.entry_url = f"{site_url}/ok"
    register(_BrokenRecipe())
    runner = make(agent_runner=None)
    task = runner.submit("test.broken", {})
    failed = await _drain(runner, task.id)
    assert failed.status is TaskStatus.FAILED
    assert "no agent fallback" in failed.detail


@pytest.mark.asyncio
async def test_unknown_recipe_rejected(runner_factory):
    make, _ = runner_factory
    runner = make()
    with pytest.raises(KeyError):
        runner.submit("does.not.exist", {})


@pytest.mark.asyncio
async def test_agent_escalation_is_recorded_as_blocked(runner_factory, site):
    """If the agent itself lands on a challenge, the task blocks, not passes."""
    make, site_url = runner_factory
    _BrokenRecipe.entry_url = f"{site_url}/ok"
    register(_BrokenRecipe())

    async def agent_hits_wall(session, url, payload):
        from browser_agent.escalation import Challenge, ChallengeKind

        raise EscalationRequired(Challenge(ChallengeKind.CAPTCHA, "agent saw one", url))

    runner = make(agent_runner=agent_hits_wall)
    task = runner.submit("test.broken", {})
    finished = await _drain(runner, task.id)
    assert finished.status is TaskStatus.BLOCKED


@pytest.mark.asyncio
async def test_freeform_task_goes_straight_to_agent(runner_factory, site):
    """agent.task has no deterministic path: the agent is the implementation."""
    make, site_url = runner_factory

    seen = {}

    async def agent(session, url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return {"agent_result": "did it"}

    runner = make(agent_runner=agent)
    task = runner.submit(AGENT_RECIPE, {"goal": "open settings", "url": site_url})
    finished = await _drain(runner, task.id)

    assert finished.status is TaskStatus.DONE
    assert finished.used_agent is True
    assert seen["payload"]["goal"] == "open settings"


@pytest.mark.asyncio
async def test_freeform_task_without_agent_fails_cleanly(runner_factory):
    make, _ = runner_factory
    from browser_agent.tasks import AGENT_RECIPE

    runner = make(agent_runner=None)
    task = runner.submit(AGENT_RECIPE, {"goal": "do something", "url": _start_url()})
    finished = await _drain(runner, task.id)
    assert finished.status is TaskStatus.FAILED
    assert "agent not available" in finished.detail


@pytest.mark.asyncio
async def test_freeform_task_blocks_on_challenge(runner_factory):
    """The freeform path obeys the same stop-and-ask rule as the fallback."""
    make, _ = runner_factory
    from browser_agent.escalation import Challenge, ChallengeKind
    from browser_agent.tasks import AGENT_RECIPE

    async def agent(session, url, payload):
        raise EscalationRequired(Challenge(ChallengeKind.CAPTCHA, "wall", url))

    runner = make(agent_runner=agent)
    task = runner.submit(AGENT_RECIPE, {"goal": "do something", "url": _start_url()})
    finished = await _drain(runner, task.id)
    assert finished.status is TaskStatus.BLOCKED


@pytest.mark.asyncio
async def test_freeform_requires_a_start_url(runner_factory):
    """A freeform instruction is meaningless without a page to work on.

    entry_url is empty for this recipe and the session is never navigated by the
    deterministic path, so without a caller-supplied url the agent would be
    handed whatever page happened to be open -- about:blank on a fresh pod,
    which is the state the first freeform task actually ran in.
    """
    make, _site = runner_factory

    async def agent(session, url, payload):  # pragma: no cover - must not run
        raise AssertionError("the agent must not be reached without a url")

    runner = make(agent_runner=agent)
    task = runner.submit(AGENT_RECIPE, {"goal": "do a thing"})
    finished = await _drain(runner, task.id)

    assert finished.status is TaskStatus.FAILED
    assert "start url" in finished.detail
    assert finished.used_agent is False


@pytest.mark.asyncio
async def test_freeform_amendment_reruns_the_agent_with_the_new_text(runner_factory, site):
    """Changing the instruction mid-run must not just end the task.

    The operator's whole reason for typing into a running task is to change what
    it does, so the run has to continue with the new wording rather than stop and
    ask for a manual Retry. browser-use raises the amendment out of the step
    hook; the runner is what turns that into "start again with this instead".
    """
    make, site_url = runner_factory
    from browser_agent.control import Amended
    from browser_agent.tasks import AGENT_RECIPE

    calls: list[str] = []

    async def agent(session, url, payload):
        calls.append(payload.get("goal") or payload.get("task") or "")
        if len(calls) == 1:
            # What a running agent does when the operator steers it.
            runner_ref[0].control.steer("just say the title")
            raise Amended("just say the title", url)
        return {"agent_result": "did the amended thing"}

    runner = make(agent_runner=agent)
    runner_ref = [runner]
    task = runner.submit(
        AGENT_RECIPE, {"goal": "read every story", "url": site_url}
    )
    finished = await _drain(runner, task.id)

    assert finished.status is TaskStatus.DONE, finished.detail
    assert calls == ["read every story", "just say the title"]
    assert finished.amended_count == 1


@pytest.mark.asyncio
async def test_freeform_amendment_loop_is_bounded(runner_factory, site):
    """An instruction that can never be satisfied must be refused, not retried
    forever. A person steering a run corrects it a few times; a loop means the
    instruction itself is unworkable and belongs in front of a human."""
    make, site_url = runner_factory
    from browser_agent.control import Amended
    from browser_agent.tasks import AGENT_RECIPE, MAX_AMENDMENTS

    calls: list[int] = []

    async def agent(session, url, payload):
        calls.append(1)
        runner_ref[0].control.steer("still not right")
        raise Amended("still not right", url)

    runner = make(agent_runner=agent)
    runner_ref = [runner]
    task = runner.submit(AGENT_RECIPE, {"goal": "do the thing", "url": site_url})
    finished = await _drain(runner, task.id)

    assert finished.status is TaskStatus.FAILED
    assert "amendments" in finished.detail
    # One initial run plus one per permitted amendment -- and then it stops.
    assert len(calls) == MAX_AMENDMENTS + 1


@pytest.mark.asyncio
async def test_stop_is_failed_not_blocked(runner_factory, site):
    """The operator stopping a run is a decision, not a captcha.

    BLOCKED pages a human and is never auto-retried; a deliberate stop must not
    do either, or stopping a task would itself raise an alert.
    """
    make, site_url = runner_factory
    from browser_agent.control import Cancelled
    from browser_agent.tasks import AGENT_RECIPE

    async def agent(session, url, payload):
        runner_ref[0].control.cancel()
        raise Cancelled("stopped by operator")

    runner = make(agent_runner=agent)
    runner_ref = [runner]
    task = runner.submit(AGENT_RECIPE, {"goal": "do the thing", "url": site_url})
    finished = await _drain(runner, task.id)

    assert finished.status is TaskStatus.FAILED
    assert "operator" in finished.detail
