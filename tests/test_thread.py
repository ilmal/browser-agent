"""The retry conversation: threads, the activity snapshot, and /say.

The operator's complaint was "I have a history with one block, then the
retries", i.e. a retry that is a dead end. These tests pin the four things that
make it a conversation instead:

  * an attempt belongs to a thread, and a follow-up increments it,
  * a finished attempt's feed survives as a snapshot,
  * /say steers a running task and re-queues a finished one,
  * the next attempt is briefed with what the earlier ones tried.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent import recipes  # noqa: E402,F401  (registers built-ins)
from browser_agent.tasks import (  # noqa: E402
    Task,
    TaskRunner,
    TaskStatus,
    recipe_reads_instruction,
    register,
)
from browser_agent.threads import ThreadStore  # noqa: E402


class _OkRecipe:
    name = "thread.ok"
    description = "succeeds"
    entry_url = ""

    async def run(self, session, payload):
        return {"ran": True}


class _FailRecipe:
    name = "thread.fail"
    description = "always fails"
    entry_url = ""

    async def run(self, session, payload):
        raise RuntimeError("selector moved")


# -- the model -------------------------------------------------------------


def test_a_task_is_its_own_thread_until_it_says_otherwise():
    task = Task(recipe="x", payload={})
    assert task.thread_id == task.id
    assert task.attempt == 1
    assert task.parent_id is None
    assert task.to_dict()["thread_id"] == task.id


def test_an_explicit_thread_is_kept():
    task = Task(recipe="x", payload={}, thread_id="t1", attempt=3, parent_id="p0")
    assert (task.thread_id, task.attempt, task.parent_id) == ("t1", 3, "p0")


# -- the store -------------------------------------------------------------


def test_messages_round_trip_in_order(tmp_path):
    store = ThreadStore()
    store.say("t1", "operator", "instruction", "do it differently")
    store.say("t1", "bot", "note", "attempt 1 failed")
    store.say("t2", "operator", "instruction", "unrelated")

    got = store.for_thread("t1")
    assert [m.role for m in got] == ["operator", "bot"]
    assert got[0].text == "do it differently"
    assert got[0].meta == {}


def test_the_last_instruction_is_what_the_operator_meant(tmp_path):
    store = ThreadStore()
    store.say("t1", "operator", "instruction", "first")
    store.say("t1", "operator", "instruction", "second")
    assert store.last_instruction("t1") == "second"
    assert store.last_instruction("nope") == ""


def test_an_unknown_role_or_kind_is_refused(tmp_path):
    store = ThreadStore()
    with pytest.raises(ValueError):
        store.say("t1", "nobody", "instruction", "hi")
    with pytest.raises(ValueError):
        store.say("t1", "operator", "telepathy", "hi")


def test_whitespace_in_a_message_is_collapsed(tmp_path):
    store = ThreadStore()
    msg = store.say("t1", "operator", "instruction", "  line one\n\n  line two  ")
    assert msg.text == "line one line two"


# -- the runner ------------------------------------------------------------


class _StubPage:
    """Enough of a Playwright page for ``detect_challenge`` to say "clear"."""

    url = "about:blank"
    frames: list[Any] = []


class _StubSession:
    """A session that opens nothing but can hold the live-control state.

    ``detect_challenge`` reads only ``page.url`` and iterates ``page.frames``,
    so an empty frame list is the honest way to say "no challenge here".
    """

    def __init__(self) -> None:
        self.activity: Any = None
        self.control: Any = None

    async def goto(self, url, **kw):
        return _StubPage()

    async def page(self):
        return _StubPage()


@pytest.fixture
def runner(tmp_path):
    """A runner whose session never opens a browser."""
    from browser_agent.config import load_settings

    settings = load_settings()
    return TaskRunner(settings, _StubSession(), agent_runner=None)


def test_retry_stays_in_the_thread_and_increments(runner):
    register(_OkRecipe())
    first = runner.submit("thread.ok", {"task": "one"})
    second = runner.retry(first.id)

    assert second.thread_id == first.thread_id
    assert second.attempt == 2
    assert second.parent_id == first.id
    assert second.id != first.id


def test_the_next_attempt_is_briefed_with_what_the_last_one_tried(runner):
    register(_FailRecipe())
    first = runner.submit("thread.fail", {"task": "one"})
    first.status = TaskStatus.FAILED
    first.detail = "recipe failed: selector moved"
    first.activity = [{"at": 1.0, "kind": "error", "text": "clicking #submit"}]

    second = runner.retry(first.id)
    brief = second.payload["history"]
    assert "attempt 1" in brief
    assert "selector moved" in brief
    assert "clicking #submit" in brief


def test_a_first_attempt_gets_no_history_block(runner):
    register(_OkRecipe())
    task = runner.submit("thread.ok", {"task": "one"})
    assert "history" not in task.payload


class _NoisyRecipe:
    name = "thread.noisy"
    description = "notes something"
    entry_url = ""

    async def run(self, session, payload):
        from browser_agent.activity import activity_of

        activity_of(session).note("step", "did the thing")
        return {"ok": True}


async def _drain(runner: TaskRunner, task: Task, timeout: float = 10.0) -> Task:
    import asyncio

    runner.start()
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if task.status in {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED}:
            return task
        await asyncio.sleep(0.02)
    raise TimeoutError(task.id)


async def test_a_finished_attempt_keeps_the_feed_it_ran_with(runner):
    """The snapshot is the whole reason a finished attempt can be read back."""
    register(_NoisyRecipe())
    task = runner.submit("thread.noisy", {"task": "one"})
    await _drain(runner, task)

    assert task.status is TaskStatus.DONE, task.detail
    assert [e["text"] for e in task.activity] == ["did the thing"]
    # A copy, not the runner's live list: the next attempt resets that one.
    assert task.activity is not runner.activity.entries


async def test_the_snapshot_survives_the_next_attempt_resetting_the_live_log(runner):
    register(_NoisyRecipe())
    first = runner.submit("thread.noisy", {"task": "one"})
    await _drain(runner, first)
    assert [e["text"] for e in first.activity] == ["did the thing"]

    second = runner.submit("thread.noisy", {"task": "two"})
    await _drain(runner, second)

    # The live log was reset for the second attempt; the first attempt's feed
    # is still there, which is what the thread panel reads.
    assert [e["text"] for e in first.activity] == ["did the thing"]
    assert first.id != second.id


# -- a message must actually change what the next attempt does -------------
#
# The operator's report: a blocked minesweeper task, told "go to another site.
# dont use the site I'm blocked on", came back blocked on the *same* site one
# second later. Nothing was broken in the plumbing — the message was recorded
# and then ignored, because /say merged it into a payload that
# ``minesweeper.play`` never reads. A deterministic recipe runs the same code
# whatever it is told, so "say what to do differently" was a promise the old
# code could not keep on exactly the recipes an operator most wants to redirect.

DETERMINISTIC = ("minesweeper.play", "x.post", "facebook.page_post",
                 "linkedin.page_post")
INSTRUCTION_READING = ("plan.task", "agent.task")


def test_the_agent_paths_and_stored_recipes_read_the_instruction():
    from browser_agent.recipes.stored import StoredRecipe

    for name in INSTRUCTION_READING:
        assert recipe_reads_instruction(name) is True, name
    assert StoredRecipe.reads_instruction is True


def test_the_deterministic_recipes_do_not():
    for name in DETERMINISTIC:
        assert recipe_reads_instruction(name) is False, name


def test_an_unknown_recipe_reads_nothing_instead_of_raising():
    """A thread can outlive a stored recipe the operator deleted."""
    assert recipe_reads_instruction("deleted.stored.recipe") is False


def test_an_unregistered_recipe_still_serialises():
    """to_dict is on every list and thread response, so it must never raise."""
    assert Task(recipe="x", payload={}).to_dict()["reads_instruction"] is False


def test_to_dict_reports_it_so_the_panel_can_promise_the_truth(runner):
    register(_OkRecipe())
    assert runner.submit("thread.ok", {"task": "one"}).to_dict()[
        "reads_instruction"] is False
    assert Task(recipe="plan.task", payload={}).to_dict()["reads_instruction"] is True


def test_retry_keeps_the_recipe_when_none_is_given(runner):
    register(_OkRecipe())
    first = runner.submit("thread.ok", {"task": "one"})
    assert runner.retry(first.id).recipe == "thread.ok"
