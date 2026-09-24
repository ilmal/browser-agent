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
from browser_agent.plan_model import parse_plan  # noqa: E402
from browser_agent.planner import PlannerClient, PlannerUnavailable  # noqa: E402
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

STUCK_PAGE = """<html><body><h1>Stuck</h1><button>noop</button></body></html>"""

MANY_PAGE = "<html><body><h1>Many</h1>" + "".join(
    f"<button>option {i}</button>" for i in range(30)
) + "</body></html>"

CAPTCHA_PAGE = """<html><body><h1>Verify</h1>
<img src="/captcha.png" class="captcha-image"></body></html>"""

SAVE_PAGE = """<html><body><h1>Save</h1>
<button onclick="location='/picked?which=cancel'">Cancel</button>
<button onclick="location='/picked?which=save'">Save</button>
</body></html>"""

ONE_PAGE = """<html><body><h1>One</h1>
<button onclick="location='/picked?which=one'">Only</button>
</body></html>"""


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
        "/stuck": STUCK_PAGE,
        "/save": SAVE_PAGE,
        "/one": ONE_PAGE,
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
        #: The thread brief each call was made with. Kept separate from
        #: ``prompts`` because the two reach the model by different routes and a
        #: test should be able to tell "the brief was sent" from "the prompt was
        #: overridden" — the bug this guards against sent neither.
        self.histories: list[str] = []

    async def plan(self, task: str, *, prompt: str | None = None,
                   history: str = "") -> str:
        self.calls.append(task)
        self.prompts.append(prompt)
        self.histories.append(history)
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
        self.calls: list[tuple[str, list[str], str]] = []

    async def pick(self, goal: str, state: str, lines: list[str], *, op: str = "click"):
        self.calls.append((goal, list(lines), op))
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
        async def plan(self, task: str, *, prompt: str | None = None,
                       history: str = "") -> str:
            self.calls.append(task)
            self.prompts.append(prompt)
            self.histories.append(history)
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


class RepairPlanner(FakePlanner):
    """Fails validation once, then answers correctly — a model that can be told.

    The reported failure of 2026-09-22 was exactly this shape: a plan whose 11th
    step was an ``extract`` with no ``selector``. One bad field is a quality
    hiccup, and the task had not touched a browser yet.
    """

    def __init__(self, bad: dict, good: dict) -> None:
        super().__init__(good)
        self._bad = bad
        self._good = good
        self.answers = 0

    async def plan(self, task: str, *, prompt: str | None = None,
                   history: str = "") -> str:
        self.calls.append(task)
        self.prompts.append(prompt)
        self.histories.append(history)
        self.answers += 1
        return json.dumps(self._bad if self.answers == 1 else self._good)


async def test_a_plan_that_fails_validation_is_repaired_not_failed(runner_factory, site):
    """The reported case: one bad step must not end a run before it starts.

    A rejection is a model-quality hiccup, and the validator already names the
    offending step and field — so a second call is a correction. Failing instead
    spends nothing and learns nothing, and the operator sees a run that died for
    a reason no browser was ever involved in.
    """
    make, site_url = runner_factory
    bad = {
        "entry_url": f"{site_url}/form",
        "steps": [{"action": "extract", "goal": "read the heading"}],  # no selector
    }
    good = {
        "entry_url": f"{site_url}/form",
        "steps": [{"action": "extract", "goal": "read the heading", "selector": "h1"}],
    }
    planner = RepairPlanner(bad, good)
    runner = make(planner=planner, laya=FakeLaya(), agent_runner=None)
    task = runner.submit("plan.task", {"task": "read the heading"})
    done = await _drain(runner, task.id)

    assert len(planner.calls) == 2, "the planner was not re-asked"
    assert done.status is TaskStatus.DONE
    assert "plan rejected" not in done.detail
    assert done.used_agent is False, "the deterministic path should have carried it"


async def test_the_repair_call_is_told_what_was_wrong(runner_factory, site):
    """A re-roll is not a repair: the retry must carry the validator's error."""
    make, site_url = runner_factory
    bad = {"entry_url": f"{site_url}/form", "steps": [{"action": "extract", "goal": "e"}]}
    good = {
        "entry_url": f"{site_url}/form",
        "steps": [{"action": "extract", "goal": "e", "selector": "h1"}],
    }
    planner = RepairPlanner(bad, good)
    runner = make(planner=planner, laya=FakeLaya(), agent_runner=None)
    task = runner.submit("plan.task", {"task": "read it"})
    await _drain(runner, task.id)

    retry_prompt = planner.prompts[1] or ""
    assert "REJECTED" in retry_prompt
    assert "selector" in retry_prompt, "the field that was missing is not named"
    assert bad["steps"][0]["goal"] in retry_prompt, "the previous answer is not shown"
    assert planner.prompts[0] != retry_prompt, "the retry sent the identical prompt"


async def test_a_planner_that_repeats_its_mistake_still_fails_visibly(runner_factory, site):
    """Repair is bounded at one call, and the guard must survive a broken model.

    Without a bound, a planner that always emits the same bad plan would be
    retried forever; the retry is a correction, not a loop.
    """
    make, site_url = runner_factory
    bad = {"entry_url": f"{site_url}/form", "steps": [{"action": "extract", "goal": "e"}]}
    planner = RepairPlanner(bad, bad)
    runner = make(planner=planner, laya=FakeLaya(), agent_runner=None)
    task = runner.submit("plan.task", {"task": "anything"})
    done = await _drain(runner, task.id)

    assert len(planner.calls) == 2, "more than one repair attempt"
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


async def test_the_thread_brief_reaches_the_planner(runner_factory, site):
    """The bug behind "not remembering between prompts" (2026-09-23).

    ``retry``/``resubmit``/``_say_to_archived`` all write ``payload["history"]``,
    and the agent path has always read it — but ``plan.task`` never passed it on,
    so an "iterate on it" re-planned from scratch and re-decided what the
    previous attempt had already settled. The live thread re-picked its game site
    on every attempt and ended on the wrong one.
    """
    make, site_url = runner_factory
    planner = FakePlanner(_happy_plan(site_url))
    runner = make(planner=planner, laya=FakeLaya(), agent_runner=None)
    brief = "- attempt 1 (plan.task) ended failed: step 4 failed: no element picked"
    task = runner.submit("plan.task", {"task": "fill the form", "history": brief})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert planner.histories == [brief]


async def test_a_first_attempt_plans_with_no_brief(runner_factory, site):
    """No earlier attempt means no brief — and no empty section in the prompt."""
    make, site_url = runner_factory
    planner = FakePlanner(_happy_plan(site_url))
    runner = make(planner=planner, laya=FakeLaya(), agent_runner=None)
    task = runner.submit("plan.task", {"task": "fill the form"})
    await _drain(runner, task.id)

    assert planner.histories == [""]


async def test_the_repair_call_also_carries_the_brief(runner_factory, site):
    """The repair is a second planning attempt, so it needs the same context as
    the first — otherwise the corrected plan is re-decided without it."""
    make, _site = runner_factory
    good = _happy_plan(_site)
    bad = {"entry_url": "not-a-url", "steps": []}
    planner = RepairPlanner(bad, good)
    runner = make(planner=planner, laya=FakeLaya(), agent_runner=None)
    brief = "- attempt 1 (plan.task) ended failed: no such element"
    task = runner.submit("plan.task", {"task": "fill the form", "history": brief})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert planner.histories == [brief, brief]


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
    # Two identical "Submit" labels: one ask, plus the shuffled verify re-ask.
    assert len(picker.calls) == 2
    assert len(picker.calls[0][1]) == 2
    # The question carried the operation premise (CLICK here).
    assert picker.calls[0][2] == "click"


async def test_pick_is_logged_for_the_flywheel(runner_factory, site, settings):
    """Every LLM pick lands in DATA_ROOT/picks.jsonl — raw page context, the
    lines as the model saw them, and the choice. This is the training
    flywheel's raw material (laya-browser logs the page, not the prompt)."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/twins",
        "steps": [{"action": "click", "goal": "submit the form"}],
    }
    picker = FakePicker(pick=1)
    runner = make(planner=FakePlanner(plan), laya=FakeLaya(pick=None), picker=picker)
    task = runner.submit("plan.task", {"task": "submit"})
    done = await _drain(runner, task.id)
    assert done.status is TaskStatus.DONE

    import json

    log_path = Path(settings.data_root) / "picks.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["op"] == "click"
    assert row["goal"] == "submit the form"
    assert row["chosen"] == 1
    assert "Submit" in row["chosen_line"]
    assert len(row["lines"]) == 2
    assert row["url"].endswith("/twins")


async def test_three_no_change_actions_fail_into_the_agent(runner_factory, site):
    """Three consecutive click/type actions that change nothing mean the plan
    is grinding on a page that ignores it — fail into the fallback instead of
    burning the step budget (jev-ultrafast's stuck latch)."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/stuck",
        "steps": [
            {"action": "click", "goal": "open settings"},
            {"action": "click", "goal": "open the profile menu"},
            {"action": "click", "goal": "open the export dialog"},
        ],
    }
    agent_calls = {"n": 0}

    async def agent(session, url, payload):
        agent_calls["n"] += 1
        return {"agent": True}

    runner = make(agent_runner=agent, planner=FakePlanner(plan), laya=FakeLaya(pick=0))
    task = runner.submit("plan.task", {"task": "poke the page"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    assert agent_calls["n"] == 1


async def test_same_action_on_unchanged_page_fails_at_once(runner_factory, site):
    """Repeating the identical action on the identical page state is a loop,
    not persistence — caught on the second occurrence (SystemOneHarness's
    repetition latch)."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/stuck",
        "steps": [
            {"action": "click", "goal": "open settings"},
            {"action": "click", "goal": "open settings"},
        ],
    }
    agent_calls = {"n": 0}

    async def agent(session, url, payload):
        agent_calls["n"] += 1
        return {"agent": True}

    runner = make(agent_runner=agent, planner=FakePlanner(plan), laya=FakeLaya(pick=0))
    task = runner.submit("plan.task", {"task": "poke the page"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    assert agent_calls["n"] == 1


class ContentPicker(FakePicker):
    """Answers by content on every ask — the faithful-picker control."""

    def __init__(self, word: str) -> None:
        super().__init__(pick=None)
        self._word = word

    async def pick(self, goal: str, state: str, lines: list[str], *, op: str = "click"):
        self.calls.append((goal, list(lines), op))
        for i, line in enumerate(lines):
            if self._word in line:
                return i, 1.0
        return None, 0.0


class BiasedPicker(ContentPicker):
    """A position-biased model: the first ask of each round always answers 0,
    later asks answer by content — exactly the failure mode agree-by-two
    exists for, and one the retry loop can wash out."""

    async def pick(self, goal: str, state: str, lines: list[str], *, op: str = "click"):
        self.calls.append((goal, list(lines), op))
        if len(self.calls) % 2 == 1:
            return 0, 1.0
        return await super().pick(goal, state, lines, op=op)


class FlipFlopper(FakePicker):
    """Never stable: alternates between two elements by content, so every
    verify pair disagrees no matter how the order was shuffled."""

    def __init__(self) -> None:
        super().__init__(pick=None)
        self._n = 0

    async def pick(self, goal: str, state: str, lines: list[str], *, op: str = "click"):
        self.calls.append((goal, list(lines), op))
        self._n += 1
        word = "Cancel" if self._n % 2 == 1 else "Save"
        for i, line in enumerate(lines):
            if word in line:
                return i, 1.0
        return None, 0.0


async def test_picker_verify_reasks_and_binds_the_agreed_element(runner_factory, site):
    """A multi-candidate pick is asked twice (second time in shuffled order)
    and only a pick both asks agree on is bound."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/save",
        "steps": [
            {"action": "click", "goal": "save the changes"},
            {"action": "extract", "goal": "heading", "selector": "h1"},
        ],
    }
    picker = ContentPicker("Save")
    runner = make(planner=FakePlanner(plan), laya=FakeLaya(pick=None), picker=picker)
    task = runner.submit("plan.task", {"task": "save"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is False
    assert len(picker.calls) == 2, "the verify re-ask never happened"
    assert "picked?which=save" in done.result["final_url"]


async def test_picker_verify_washes_out_a_first_ask_bias(runner_factory, site):
    """A model biased on its first ask but honest on re-asks ends bound to the
    element its answers agree on — the retry loop + verify correct it."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/save",
        "steps": [
            {"action": "click", "goal": "save the changes"},
            {"action": "extract", "goal": "heading", "selector": "h1"},
        ],
    }
    picker = BiasedPicker("Save")
    runner = make(planner=FakePlanner(plan), laya=FakeLaya(pick=None), picker=picker)
    task = runner.submit("plan.task", {"task": "save"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is False
    assert "picked?which=save" in done.result["final_url"]
    assert len(picker.calls) >= 2


async def test_picker_verify_declines_a_never_stable_model(runner_factory, site):
    """Answers that flip between elements on every ask never agree — the pick
    is declined into the agent fallback, nothing is clicked."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/save",
        "steps": [{"action": "click", "goal": "save the changes"}],
    }
    agent_calls = {"n": 0}

    async def agent(session, url, payload):
        agent_calls["n"] += 1
        return {"agent": True}

    runner = make(
        agent_runner=agent,
        planner=FakePlanner(plan),
        laya=FakeLaya(pick=None),
        picker=FlipFlopper(),
    )
    task = runner.submit("plan.task", {"task": "save"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    assert agent_calls["n"] == 1


async def test_picker_verify_skipped_for_single_candidate(runner_factory, site):
    """One candidate cannot be a position error — no second ask."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/one",
        "steps": [
            {"action": "click", "goal": "press the only button"},
            {"action": "extract", "goal": "heading", "selector": "h1"},
        ],
    }
    picker = ContentPicker("Only")
    runner = make(planner=FakePlanner(plan), laya=FakeLaya(pick=None), picker=picker)
    task = runner.submit("plan.task", {"task": "go"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is False
    assert len(picker.calls) == 1
    assert "picked?which=one" in done.result["final_url"]


async def test_risky_step_without_done_when_is_refused_to_the_agent(runner_factory, site):
    """A click whose wording names something irreversible and which carries no
    done_when is refused BEFORE it acts (cua's risk tiers) — the fallback gets
    the original task, the page untouched by that step."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/form",
        "steps": [{"action": "click", "goal": "delete the account", "selector": "#go"}],
    }
    agent_calls = {"n": 0}

    async def agent(session, url, payload):
        agent_calls["n"] += 1
        return {"agent": True}

    runner = make(agent_runner=agent, planner=FakePlanner(plan), laya=FakeLaya())
    task = runner.submit("plan.task", {"task": "clean up"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    assert agent_calls["n"] == 1


async def test_risky_step_with_done_when_runs(runner_factory, site):
    """Proof attached → the same wording executes deterministically."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/form",
        "steps": [
            {
                "action": "click",
                "goal": "delete the account",
                "selector": "#go",
                "done_when": {"text_contains": "submitted"},
            }
        ],
    }
    runner = make(planner=FakePlanner(plan), laya=FakeLaya())
    task = runner.submit("plan.task", {"task": "clean up"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is False


async def test_final_gate_confident_no_falls_back_to_agent(runner_factory, site):
    """Finish-insist: every step passed its own gates, but the gate says the
    OVERALL task is not satisfied — fail into the fallback with the page where
    the plan left it, instead of reporting a false done."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/form",
        "steps": [
            {
                "action": "click",
                "goal": "submit the form",
                "selector": "#go",
                "done_when": {"text_contains": "submitted"},
            }
        ],
    }
    agent_calls = {"n": 0}

    async def agent(session, url, payload):
        agent_calls["n"] += 1
        return {"agent": True}

    runner = make(
        agent_runner=agent,
        planner=FakePlanner(plan),
        laya=FakeLaya(yes=False, conf=0.95),
    )
    task = runner.submit("plan.task", {"task": "submit the form"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is True
    assert agent_calls["n"] == 1


async def test_final_gate_unsure_never_punishes(runner_factory, site):
    """An inconclusive final check must not fail a plan whose steps all proved
    themselves — unsure advances."""
    make, site_url = runner_factory
    plan = {
        "entry_url": f"{site_url}/form",
        "steps": [
            {
                "action": "click",
                "goal": "submit the form",
                "selector": "#go",
                "done_when": {"text_contains": "submitted"},
            }
        ],
    }
    runner = make(
        planner=FakePlanner(plan),
        laya=FakeLaya(yes=None, conf=0.95),
    )
    task = runner.submit("plan.task", {"task": "submit the form"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert done.used_agent is False


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
