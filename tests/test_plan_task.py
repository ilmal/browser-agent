"""plan.task integration tests.

Proves the planner/executor loop's central promises:
  * a valid plan runs deterministically — no agent call on the happy path,
  * planner garbage FAILS visibly instead of reaching the agent,
  * a planner that cannot be reached hands the task to the agent fallback,
  * a step failing mid-plan briefs the fallback with the original task,
  * a challenge mid-plan BLOCKs and never reaches the agent,
  * the Laya picker binds by index, and its verdicts gate click/type steps.
"""

from __future__ import annotations

import http.server
import json
import socket
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent import recipes  # noqa: E402,F401  (registers built-ins)
from browser_agent.config import load_settings  # noqa: E402
from browser_agent.planner import PlannerClient, PlannerUnavailable  # noqa: E402
from browser_agent.plan_model import parse_plan  # noqa: E402
from browser_agent.recipes.plan_task import PlanTask  # noqa: E402
from browser_agent.tasks import TaskRunner, TaskStatus, register  # noqa: E402

FORM_PAGE = """<html><body><h1>Form Page</h1><form>
<input id="user" placeholder="Username">
<input id="email" placeholder="Email">
<button id="go" type="button" onclick="document.getElementById('done').textContent='submitted'">Go</button>
</form><div id="done"></div></body></html>"""

TWINS_PAGE = """<html><body><h1>Twins</h1>
<button onclick="location='/picked?which=1'">Submit</button>
<button onclick="location='/picked?which=2'">Submit</button>
</body></html>"""

PICKED_PAGE = """<html><body><h1>Picked</h1><p>landed</p></body></html>"""

MANY_PAGE = "<html><body><h1>Many</h1>" + "".join(
    f"<button>option {i}</button>" for i in range(30)
) + "</body></html>"

CAPTCHA_PAGE = """<html><body><h1>Verify</h1>
<img src="/captcha.png" class="captcha-image"></body></html>"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def site():
    pages = {
        "/form": FORM_PAGE,
        "/twins": TWINS_PAGE,
        "/picked": PICKED_PAGE,
        "/many": MANY_PAGE,
        "/captcha": CAPTCHA_PAGE,
    }

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
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "llm_sk_placeholder")
    monkeypatch.setenv("NOTIFY_ON_ESCALATION", "false")
    monkeypatch.setenv("PLANNER_MAX_STEPS", "12")
    # Fast tests: a selector that never appears must fail in ~1s, not 10.
    monkeypatch.setenv("PLANNER_STEP_TIMEOUT_S", "1")
    monkeypatch.setenv("LAYA_ENABLED", "true")
    monkeypatch.setenv("LAYA_PICK_ENABLED", "true")
    monkeypatch.setenv("LAYA_MIN_CONFIDENCE", "0.75")
    return load_settings()


class FakePlanner:
    enabled = True

    def __init__(self, plan: dict | None = None, fail: bool = False) -> None:
        self._plan = plan
        self._fail = fail
        self.calls: list[str] = []
        #: The system prompt each call was made with, so a test can assert the
        #: operator's ``planner_prompt`` override reached the model.
        self.prompts: list[str | None] = []

    async def plan(self, task: str, *, prompt: str | None = None) -> str:
        self.calls.append(task)
        self.prompts.append(prompt)
        if self._fail:
            raise PlannerUnavailable("planner down")
        return json.dumps(self._plan)


class FakeLaya:
    def __init__(self, pick: int | None = 0, conf: float = 0.95, yes: bool = True) -> None:
        self.pick = pick
        self.conf = conf
        self.yes = yes
        self.seen_lines: list[str] | None = None
        self.stats = {"calls": 0, "picks": 0, "confirms": 0, "inconclusive": 0}

    @property
    def enabled(self) -> bool:
        return True

    @property
    def pick_enabled(self) -> bool:
        return True

    @property
    def summary(self) -> dict[str, Any]:
        return {"state": "ready", **self.stats}

    async def choose(self, question: str, lines: list[str], state: str):
        self.seen_lines = list(lines)
        if self.pick is None:
            return None, 0.0
        return self.pick, self.conf

    async def yes_no(self, question: str, state: str):
        return self.yes, self.conf


class FakePicker:
    """Stands in for the LLM element picker (browser_agent.picker)."""

    enabled = True

    def __init__(self, pick: int | None = 0) -> None:
        self._pick = pick
        self.calls: list[tuple[str, list[str]]] = []

    async def pick(self, goal: str, state: str, lines: list[str]):
        self.calls.append((goal, list(lines)))
        if self._pick is None:
            return None, 0.0
        return self._pick, 1.0


class _Session:
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
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        made: list[TaskRunner] = []

        def make(agent_runner=None, planner=None, laya=None, picker=None):
            register(PlanTask(planner=planner, laya=laya, settings=settings, picker=picker))
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
    import asyncio

    runner.start()
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        task = runner.tasks[task_id]
        if task.status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}:
            return task
        await asyncio.sleep(0.05)
    raise TimeoutError(task_id)


def _happy_plan(site_url: str) -> dict:
    return {
        "entry_url": f"{site_url}/form",
        "steps": [
            {"action": "navigate", "goal": "open the form", "text": f"{site_url}/form"},
            {"action": "type", "goal": "enter the username", "selector": "#user", "text": "nils"},
            {
                "action": "click",
                "goal": "submit the form",
                "selector": "#go",
                "done_when": {"text_contains": "submitted"},
            },
            {"action": "extract", "goal": "page heading", "selector": "h1"},
        ],
    }


async def test_plan_runs_deterministically(runner_factory, site):
    make, site_url = runner_factory
    planner = FakePlanner(_happy_plan(site_url))
    laya = FakeLaya()
    agent_calls = {"n": 0}

    async def agent(session, url, payload):  # pragma: no cover - must not run
        agent_calls["n"] += 1
        return {}

    runner = make(agent_runner=agent, planner=planner, laya=laya)
    task = runner.submit("plan.task", {"task": "fill the form"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is False
    assert agent_calls["n"] == 0
    assert done.result["steps_executed"] == 4
    assert done.result["extracts"]["page heading"] == "Form Page"
    assert planner.calls == ["fill the form"]
    # Explicit selectors were used, so the gate never had to pick.
    assert laya.seen_lines is None


async def test_fenced_planner_output_is_accepted(runner_factory, site):
    make, _site = runner_factory
    plan_dict = _happy_plan(site)
    wrapped = "```json\n" + json.dumps(plan_dict) + "\n```"
    assert "```" in wrapped  # the chatty-model case under test

    class FencedPlanner(FakePlanner):
        async def plan(self, task: str, *, prompt: str | None = None) -> str:
            self.calls.append(task)
            self.prompts.append(prompt)
            return wrapped

    runner = make(planner=FencedPlanner(plan_dict), laya=FakeLaya())
    task = runner.submit("plan.task", {"task": "fill the form"})
    done = await _drain(runner, task.id)
    assert done.status is TaskStatus.DONE
    assert done.used_agent is False


async def test_oversized_plan_fails_visibly(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/form",
        "steps": [{"action": "extract", "goal": f"e{i}", "selector": "h1"} for i in range(13)],
    }
    runner = make(planner=FakePlanner(plan), laya=FakeLaya(), agent_runner=None)
    task = runner.submit("plan.task", {"task": "anything"})
    done = await _drain(runner, task.id)
    assert done.status is TaskStatus.FAILED
    assert "plan rejected" in done.detail
    assert done.used_agent is False


@pytest.mark.parametrize(
    "bad_plan",
    [
        {"entry_url": "ftp://x", "steps": [{"action": "extract", "goal": "e", "selector": "h1"}]},
        {"entry_url": "http://x/", "steps": [{"action": "extract", "goal": "e"}]},
        {"entry_url": "http://x/", "steps": [{"action": "click", "text": "hi"}]},
        {"entry_url": "http://x/", "steps": [{"action": "navigate", "text": "not-a-url"}]},
        {"entry_url": "http://x/", "steps": [{"action": "type", "goal": "g"}]},
        {"entry_url": "http://x/", "steps": [{"action": "explode", "goal": "g"}]},
    ],
)
async def test_invalid_plans_fail_visibly(runner_factory, bad_plan, site):
    make, _ = runner_factory
    runner = make(planner=FakePlanner(bad_plan), laya=FakeLaya(), agent_runner=None)
    task = runner.submit("plan.task", {"task": "anything"})
    done = await _drain(runner, task.id)
    assert done.status is TaskStatus.FAILED
    assert "plan rejected" in done.detail


async def test_missing_task_text_fails(runner_factory, site):
    make, _ = runner_factory
    runner = make(planner=FakePlanner({}), laya=FakeLaya(), agent_runner=None)
    task = runner.submit("plan.task", {})
    done = await _drain(runner, task.id)
    assert done.status is TaskStatus.FAILED
    assert "plan rejected" in done.detail


async def test_ui_text_field_works_as_task_alias(runner_factory, site):
    """The admin UI's Instructions box sends payload["text"], not ["task"]."""
    make, site_url = runner_factory
    planner = FakePlanner(_happy_plan(site_url))
    runner = make(planner=planner, laya=FakeLaya(), agent_runner=None)
    task = runner.submit("plan.task", {"text": "fill the form"})
    done = await _drain(runner, task.id)
    assert done.status is TaskStatus.DONE
    assert done.used_agent is False


async def test_step_failure_briefs_agent_with_original_task(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/form",
        "steps": [{"action": "click", "goal": "open settings", "selector": "#missing"}],
    }
    seen = {}

    async def agent(session, url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return {"agent": True}

    runner = make(agent_runner=agent, planner=FakePlanner(plan), laya=FakeLaya())
    task = runner.submit("plan.task", {"task": "open settings and count things"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    # The agent is briefed with the original task and the plan's entry URL.
    assert seen["payload"]["goal"] == "open settings and count things"
    assert seen["url"] == f"{site_url}/form"


async def test_unreachable_planner_goes_straight_to_agent(runner_factory, site):
    make, site_url = runner_factory
    seen = {}

    async def agent(session, url, payload):
        seen["url"] = url
        return {"agent": True}

    runner = make(agent_runner=agent, planner=FakePlanner(fail=True), laya=FakeLaya())
    task = runner.submit("plan.task", {"task": "do the thing"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    # entry_url for this recipe is about:blank; the agent still gets a usable start.
    assert seen["url"] == "about:blank"


async def test_challenge_mid_plan_blocks_without_agent(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/captcha",
        "steps": [
            {"action": "navigate", "goal": "go", "text": f"{site_url}/captcha"},
            {"action": "click", "goal": "continue", "selector": "#go"},
        ],
    }
    agent_calls = {"n": 0}

    async def agent(session, url, payload):  # pragma: no cover - must not run
        agent_calls["n"] += 1
        return {}

    runner = make(agent_runner=agent, planner=FakePlanner(plan), laya=FakeLaya())
    task = runner.submit("plan.task", {"task": "anything"})
    blocked = await _drain(runner, task.id)
    assert blocked.status is TaskStatus.BLOCKED
    assert agent_calls["n"] == 0


async def test_picker_binds_by_index_not_text(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/twins",
        "steps": [
            {"action": "click", "goal": "submit the form"},
            {"action": "extract", "goal": "heading", "selector": "h1"},
        ],
    }
    laya = FakeLaya(pick=1)
    runner = make(planner=FakePlanner(plan), laya=laya)
    task = runner.submit("plan.task", {"task": "submit"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    # Two identical "Submit" buttons — only the index distinguishes them.
    assert len(laya.seen_lines) == 2
    assert "Submit" in laya.seen_lines[1]
    assert "picked?which=2" in done.result["final_url"]


async def test_llm_picker_resolves_when_laya_declines(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/twins",
        "steps": [
            {"action": "click", "goal": "submit the form"},
            {"action": "extract", "goal": "heading", "selector": "h1"},
        ],
    }
    picker = FakePicker(pick=1)
    runner = make(planner=FakePlanner(plan), laya=FakeLaya(pick=None), picker=picker)
    task = runner.submit("plan.task", {"task": "submit"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is False
    assert "picked?which=2" in done.result["final_url"]
    # The picker saw the same numbered lines laya would have.
    assert len(picker.calls) == 1
    assert len(picker.calls[0][1]) == 2


async def test_picker_decline_too_falls_back_to_agent(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/twins",
        "steps": [{"action": "click", "goal": "submit the form"}],
    }
    agent_calls = {"n": 0}

    async def agent(session, url, payload):
        agent_calls["n"] += 1
        return {"agent": True}

    runner = make(
        agent_runner=agent,
        planner=FakePlanner(plan),
        laya=FakeLaya(pick=None),
        picker=FakePicker(pick=None),
    )
    task = runner.submit("plan.task", {"task": "submit"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    assert agent_calls["n"] == 1


async def test_low_confidence_pick_falls_back_to_agent(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/twins",
        "steps": [{"action": "click", "goal": "submit the form"}],
    }
    agent_calls = {"n": 0}

    async def agent(session, url, payload):
        agent_calls["n"] += 1
        return {"agent": True}

    runner = make(agent_runner=agent, planner=FakePlanner(plan), laya=FakeLaya(pick=0, conf=0.2))
    task = runner.submit("plan.task", {"task": "submit"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    assert agent_calls["n"] == 1


async def test_candidate_list_is_capped(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/many",
        "steps": [{"action": "click", "goal": "press something"}],
    }
    laya = FakeLaya(pick=0)
    runner = make(planner=FakePlanner(plan), laya=laya)
    task = runner.submit("plan.task", {"task": "click one"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert len(laya.seen_lines) == 10  # LAYA_MAX_CANDIDATES (calibrated range), page has 30


async def test_type_step_uses_picker_for_field_not_text(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/form",
        "steps": [{"action": "type", "goal": "enter the email address", "text": "nils@u1.se"}],
    }
    laya = FakeLaya(pick=1)  # second visible input = #email
    runner = make(planner=FakePlanner(plan), laya=laya)
    task = runner.submit("plan.task", {"task": "enter the email"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert "placeholder='Email'" in laya.seen_lines[1]
    page = await runner.session.page()
    value = await page.locator("#email").input_value()
    assert value == "nils@u1.se"


async def test_confident_yes_advances_without_done_when(runner_factory, site):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/form",
        "steps": [{"action": "click", "goal": "submit the form", "selector": "#go"}],
    }
    runner = make(planner=FakePlanner(plan), laya=FakeLaya(yes=True, conf=0.95))
    task = runner.submit("plan.task", {"task": "submit"})
    done = await _drain(runner, task.id)
    assert done.status is TaskStatus.DONE
    assert done.used_agent is False


@pytest.mark.parametrize("yes,conf", [(True, 0.3), (False, 0.95)])
async def test_unconfirmed_click_falls_back_to_agent(runner_factory, site, yes, conf):
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/form",
        "steps": [{"action": "click", "goal": "submit the form", "selector": "#go"}],
    }
    agent_calls = {"n": 0}

    async def agent(session, url, payload):
        agent_calls["n"] += 1
        return {"agent": True}

    runner = make(
        agent_runner=agent, planner=FakePlanner(plan), laya=FakeLaya(yes=yes, conf=conf)
    )
    task = runner.submit("plan.task", {"task": "submit"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    assert agent_calls["n"] == 1


# -- pure model/validation tests (no browser) -------------------------------


def test_parse_plan_accepts_fences():
    raw = "```json\n" + json.dumps(
        {"entry_url": "https://x/", "steps": [{"action": "extract", "goal": "h", "selector": "h1"}]}
    ) + "\n```"
    plan = parse_plan(raw)
    assert plan.steps[0].selector == "h1"


def test_parse_plan_rejects_unknown_fields():
    with pytest.raises(Exception, match="plan rejected|valid plan"):
        parse_plan({"entry_url": "https://x/", "extra": 1, "steps": []})


def test_step_failure_defaults():
    from browser_agent.plan_model import StepFailure

    exc = StepFailure("boom")
    assert exc.goal == ""
    assert exc.entry_url == ""


async def test_transient_network_error_is_retried_once():
    from browser_agent.recipes._plan_exec import _retry_transient

    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Page.goto: net::ERR_NETWORK_CHANGED at https://x/")
        return "ok"

    assert await _retry_transient(flaky, what="navigate") == "ok"
    assert calls["n"] == 2


@pytest.mark.parametrize(
    "message",
    [
        "Page.goto: net::ERR_NAME_NOT_RESOLVED",
        "Page.goto: net::ERR_NETWORK_CHANGED at https://x/",
        "selector moved",
    ],
)
async def test_non_retryable_errors_raise_immediately(message):
    from browser_agent.recipes._plan_exec import _retry_transient

    calls = {"n": 0}

    async def broken():
        calls["n"] += 1
        raise RuntimeError(message)

    with pytest.raises(RuntimeError):
        await _retry_transient(broken, what="navigate")
    # The ERR_NETWORK_CHANGED case consumes its one retry; the others get none.
    assert calls["n"] == (2 if "ERR_NETWORK_CHANGED" in message else 1)


async def test_planner_client_requires_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d"))
    client = PlannerClient(load_settings())
    assert client.enabled is False
    with pytest.raises(PlannerUnavailable):
        await client.plan("x")
