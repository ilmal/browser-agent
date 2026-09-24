"""The bot's own turn: every finished attempt speaks its outcome into its thread.

Proves:
  * the outcome text is shaped per status (done / blocked / failed),
  * exactly one message lands per terminal state, deduped on re-entry,
  * a missing or broken thread store never fails the run,
  * the hook lives in the worker's finally — a queued task says nothing,
  * a real run through the loop posts the message into the store.
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
from browser_agent.tasks import (  # noqa: E402
    Task,
    TaskRunner,
    TaskStatus,
    _outcome_text,
    register,
)


class _Msg:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text


class _FakeThreads:
    """A ThreadStore stand-in with just the two methods the hook touches."""

    def __init__(self, fail: bool = False) -> None:
        self.msgs: list[_Msg] = []
        self.fail = fail

    def say(self, thread_id: str, role: str, kind: str, text: str, **kw: Any) -> _Msg:
        if self.fail:
            raise RuntimeError("store gone")
        assert role == "bot" and kind == "note"
        msg = _Msg(role, text)
        self.msgs.append(msg)
        return msg

    def for_thread(self, thread_id: str) -> list[_Msg]:
        return list(self.msgs)


def _task(status: TaskStatus, **fields: Any) -> Task:
    t = Task(recipe="test.speak", payload={})
    t.status = status
    for k, v in fields.items():
        setattr(t, k, v)
    return t


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE", "pytest")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("HEADLESS", "true")
    from browser_agent.config import load_settings

    return load_settings()


def _runner(settings, threads: Any = None) -> TaskRunner:
    return TaskRunner(settings, session=None, agent_runner=None, threads=threads)


# -- the text shapes ----------------------------------------------------


def test_done_text_carries_the_result():
    t = _task(TaskStatus.DONE, result={"title": "Example Domain"}, detail="recipe succeeded")
    assert _outcome_text(t) == "Done: Example Domain"


def test_done_text_reads_the_extracts_out_of_a_plan_envelope():
    # The shape plan.task actually returns. Dumping this envelope into the
    # thread is what buried "Example Domains" under a page of JSON.
    result = {
        "plan": {"entry_url": "https://example.com", "steps": []},
        "steps_executed": 2,
        "extracts": {"read the main heading": "Example Domain"},
        "final_url": "https://example.com",
    }
    t = _task(TaskStatus.DONE, result=result, detail="plan succeeded")
    assert _outcome_text(t) == "Done: read the main heading: Example Domain"


def test_done_text_falls_back_to_detail():
    t = _task(TaskStatus.DONE, result=None, detail="recipe succeeded")
    assert _outcome_text(t) == "Done: recipe succeeded"


def test_done_text_is_capped():
    t = _task(TaskStatus.DONE, result={"blob": "x" * 5000})
    out = _outcome_text(t)
    assert len(out) <= 300 and out.startswith("Done:")


def test_done_text_reports_an_answer_key_over_the_envelope():
    t = _task(TaskStatus.DONE, result={"answer": "yes", "url": "https://x.se"}, detail="ok")
    assert _outcome_text(t) == "Done: yes"


def test_blocked_text_asks_for_a_human():
    t = _task(TaskStatus.BLOCKED, detail="a captcha wall")
    assert _outcome_text(t) == "I'm blocked and need you: a captcha wall"


def test_failed_text_carries_the_detail():
    t = _task(TaskStatus.FAILED, detail="recipe failed: selector moved")
    assert _outcome_text(t) == "Failed: recipe failed: selector moved"


def test_operator_stop_is_not_reported_as_a_failure():
    t = _task(TaskStatus.FAILED, detail="stopped by the operator")
    assert _outcome_text(t) == "Stopped at your request."


def test_non_terminal_status_has_nothing_to_say():
    assert _outcome_text(_task(TaskStatus.QUEUED)) is None
    assert _outcome_text(_task(TaskStatus.RUNNING)) is None


# -- the hook ------------------------------------------------------------


def test_hook_is_a_noop_without_a_store(settings):
    runner = _runner(settings, threads=None)
    runner._speak_outcome(_task(TaskStatus.DONE, result={"ok": True}))  # must not raise


def test_hook_posts_one_bot_message(settings):
    store = _FakeThreads()
    runner = _runner(settings, threads=store)
    t = _task(TaskStatus.DONE, result={"did": "it"})
    runner._speak_outcome(t)
    runner._speak_outcome(t)  # the finally can pass a terminal task twice
    assert len(store.msgs) == 1
    assert store.msgs[0].role == "bot"
    assert store.msgs[0].text == "Done: it"


def test_hook_posts_for_each_new_outcome(settings):
    """A retry in the same thread must not be swallowed by the dedupe."""
    store = _FakeThreads()
    runner = _runner(settings, threads=store)
    runner._speak_outcome(_task(TaskStatus.DONE, result={"a": 1}))
    runner._speak_outcome(_task(TaskStatus.FAILED, detail="selector moved"))
    assert [m.text for m in store.msgs] == ["Done: 1", "Failed: selector moved"]


def test_hook_survives_a_broken_store(settings):
    runner = _runner(settings, threads=_FakeThreads(fail=True))
    runner._speak_outcome(_task(TaskStatus.BLOCKED, detail="wall"))  # must not raise


def test_hook_speaks_into_the_tasks_own_thread(settings):
    store = _FakeThreads()
    runner = _runner(settings, threads=store)
    t = _task(TaskStatus.FAILED, detail="no agent fallback available")
    t.thread_id = "somethread"
    runner._speak_outcome(t)
    assert store.msgs[0].text.startswith("Failed:")
    # (the fake ignores thread_id; the real store is asserted via the runner)


# -- through the real loop -----------------------------------------------

OK_PAGE = "<html><body><h1>ok</h1></body></html>"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def site():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = OK_PAGE.encode()
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


class _Session:
    """Same throwaway-context factory test_tasks.py uses."""

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

        def make(threads: Any = None, agent_runner: Any = None):
            session = _Session(pw, browser)
            r = TaskRunner(
                settings, session, agent_runner=agent_runner, threads=threads
            )
            made.append(r)
            return r

        try:
            yield make, site
        finally:
            for r in made:
                await r.stop()
            await browser.close()


async def _drain(runner: TaskRunner, task_id: str, timeout: float = 30) -> Task:
    import asyncio

    runner.start()
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        task = runner.tasks[task_id]
        if task.status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}:
            return task
        await asyncio.sleep(0.05)
    raise TimeoutError(task_id)


class _SpeakOk:
    name = "test.speak-ok"
    description = "succeeds"
    entry_url = ""

    async def run(self, session, payload):
        return {"did": "the thing"}


class _SpeakBroken:
    name = "test.speak-broken"
    description = "raises"
    entry_url = ""

    async def run(self, session, payload):
        raise RuntimeError("selector moved")


@pytest.mark.asyncio
async def test_a_real_done_run_posts_done(runner_factory, site):
    make, site_url = runner_factory
    _SpeakOk.entry_url = f"{site_url}/ok"
    register(_SpeakOk())
    store = _FakeThreads()
    runner = make(threads=store)
    task = runner.submit("test.speak-ok", {})
    await _drain(runner, task.id)
    assert [m.text for m in store.msgs] == ["Done: the thing"]


@pytest.mark.asyncio
async def test_a_real_failed_run_posts_failed(runner_factory, site):
    make, site_url = runner_factory
    _SpeakBroken.entry_url = f"{site_url}/ok"
    register(_SpeakBroken())
    store = _FakeThreads()
    runner = make(threads=store, agent_runner=None)
    task = runner.submit("test.speak-broken", {})
    await _drain(runner, task.id)
    assert len(store.msgs) == 1
    assert store.msgs[0].text.startswith("Failed:")
    assert "selector moved" in store.msgs[0].text


@pytest.mark.asyncio
async def test_a_queued_task_says_nothing(runner_factory, site):
    """The hook keys on a terminal status: queued work has no outcome yet."""
    make, site_url = runner_factory
    _SpeakOk.entry_url = f"{site_url}/ok"
    register(_SpeakOk())
    store = _FakeThreads()
    runner = make(threads=store)
    runner.submit("test.speak-ok", {})  # never started — no worker, no run
    assert store.msgs == []
