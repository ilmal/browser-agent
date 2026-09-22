"""What a message on a finished deterministic task actually redirects.

The reported failure was an operator typing "go to another site. dont use the
site I'm blocked on" and the next attempt hitting the *same* wall. Two bugs sat
behind that: the message was merged into a payload the recipe never reads (fixed
by escalating to the agent), and the agent run then had no start URL — or worse,
the previous attempt's URL — so the redirect was recorded and quietly dropped.

These pin the second half: where the redirected run begins, and that the
operator's own words beat everything else.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def api_mod(monkeypatch, tmp_path):
    """The control plane, imported fresh so its module-level settings hold."""
    monkeypatch.setenv("AGENT_PROFILE", "pytest")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    for mod in [m for m in list(sys.modules) if m.startswith("browser_agent")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    import browser_agent.api as api

    return api


def _task(api, *, payload, detail="", result=None, recipe="minesweeper.play"):
    from browser_agent.tasks import Task

    t = Task(recipe=recipe, payload=payload, detail=detail, result=result)
    return t


# -- _start_url_for, the fallback chain ------------------------------------


def test_the_url_the_operator_names_wins(api_mod):
    """The reported case, and the whole point: "go to X instead" means X."""
    task = _task(
        api_mod,
        payload={"url": "https://minesweeper.online/"},
        detail="blocked: rate limited",
    )
    got = api_mod._start_url_for("go to https://duckduckgo.com instead", task)
    assert got == "https://duckduckgo.com", "the blocked site was used again"


def test_trailing_punctuation_is_not_part_of_the_url(api_mod):
    task = _task(api_mod, payload={})
    assert api_mod._start_url_for("try https://example.com/x.", task) == (
        "https://example.com/x"
    )
    assert api_mod._start_url_for("(see https://example.com/y)", task) == (
        "https://example.com/y"
    )


def test_a_url_in_the_failure_detail_counts_too(api_mod):
    """A challenge names the page it happened on; that is a real start point."""
    task = _task(api_mod, payload={}, detail="challenge at https://site.example/game")
    assert api_mod._start_url_for("carry on", task) == "https://site.example/game"


def test_the_previous_attempts_url_is_the_fallback(api_mod):
    """No URL named: the run resumes where the last attempt actually was."""
    task = _task(api_mod, payload={}, result={"url": "https://site.example/after"})
    assert api_mod._start_url_for("carry on", task) == "https://site.example/after"


def test_the_previous_payload_is_the_next_fallback(api_mod):
    task = _task(api_mod, payload={"url": "https://site.example/payload"})
    assert api_mod._start_url_for("carry on", task) == "https://site.example/payload"


def test_the_recipes_entry_url_is_the_last_resort(api_mod):
    from browser_agent.tasks import register

    class _Entry:
        name = "thread.entry"
        description = "has an entry url"
        entry_url = "https://recipe.example/entry"
        reads_instruction = False

        async def run(self, session, payload):  # pragma: no cover - never run here
            return {}

    register(_Entry())
    task = _task(api_mod, payload={}, recipe="thread.entry")
    assert api_mod._start_url_for("carry on", task) == "https://recipe.example/entry"


def test_about_blank_is_not_a_start_url(api_mod):
    """The freeform agent rejects a blank page, so it must not be offered one."""
    task = _task(api_mod, payload={"url": "about:blank"}, recipe="agent.task")
    assert api_mod._start_url_for("carry on", task) == ""


def test_a_deleted_recipe_does_not_raise(api_mod):
    """A thread outlives the stored recipe it names; say() must still work."""
    task = _task(api_mod, payload={}, recipe="deleted.stored.recipe")
    assert api_mod._start_url_for("carry on", task) == ""


# -- say(), end to end over the real runner --------------------------------


async def _say(api_mod, task, text):
    """Run the handler with a task already in the runner, as the UI would."""
    api_mod.runner.tasks[task.id] = task
    return await api_mod.say(task.id, api_mod.SayRequest(text=text))


@pytest.mark.asyncio
async def test_a_message_on_a_deterministic_task_becomes_an_agent_run(api_mod):
    """The recipe cannot be told anything, so the attempt has to change hands."""
    from browser_agent.tasks import AGENT_RECIPE

    task = _task(api_mod, payload={"url": "https://minesweeper.online/"})
    out = await _say(api_mod, task, "go to https://duckduckgo.com instead")

    nxt = out["task"]
    assert nxt["recipe"] == AGENT_RECIPE
    assert nxt["attempt"] == 2
    assert nxt["thread_id"] == task.thread_id
    assert nxt["payload"]["url"] == "https://duckduckgo.com", (
        "the agent run was pointed back at the site the operator ruled out"
    )
    assert nxt["status"] == "queued"


@pytest.mark.asyncio
async def test_the_instruction_reaches_every_field_the_agent_reads(api_mod):
    """agent.py reads goal or text; a stale goal would win over the new words."""
    task = _task(api_mod, payload={"url": "https://a.example/", "goal": "old goal"})
    out = await _say(api_mod, task, "search for the thing")

    payload = out["task"]["payload"]
    assert payload["goal"] == "search for the thing"
    assert payload["text"] == "search for the thing"
    assert payload["task"] == "search for the thing"


@pytest.mark.asyncio
async def test_a_recipe_that_reads_its_payload_keeps_its_own_recipe(api_mod):
    """plan.task takes the instruction itself; escalating would be a downgrade."""
    task = _task(api_mod, payload={"task": "one"}, recipe="plan.task")
    out = await _say(api_mod, task, "do it differently")

    assert out["task"]["recipe"] == "plan.task"
    assert out["task"]["payload"]["goal"] == "do it differently"


@pytest.mark.asyncio
async def test_a_message_that_names_no_url_resumes_the_last_attempt(api_mod):
    """"carry on" is not a redirect; it must not blank the start URL."""
    task = _task(
        api_mod,
        payload={"url": "https://a.example/"},
        result={"url": "https://a.example/deep"},
    )
    out = await _say(api_mod, task, "carry on")

    assert out["task"]["payload"]["url"] == "https://a.example/deep"


# -- say() to a run that only exists as a record ---------------------------


def _archive(api_mod, *, task_id="old111", thread_id="t-old", attempt=1,
             recipe="minesweeper.play", payload=None, activity=None,
             detail="blocked: rate limited", created_at=100.0):
    """Put one finished attempt in the archive, with no task object for it.

    That is exactly the post-deploy state: the pod was recreated, every
    in-memory task went with it, and only the record remains.
    """
    from browser_agent.tasks import Task, TaskStatus

    t = Task(
        id=task_id,
        recipe=recipe,
        payload=payload if payload is not None else {"url": "https://minesweeper.online/"},
        thread_id=thread_id,
        attempt=attempt,
        status=TaskStatus.BLOCKED,
        detail=detail,
        created_at=created_at,
    )
    t.activity = activity or []
    api_mod.runs.save(t)
    return t


@pytest.mark.asyncio
async def test_a_message_on_an_archived_run_becomes_the_next_attempt(api_mod):
    """A deploy must not be a dead end for a conversation.

    The operator keeps a bad run so they can study it; being able to answer it
    afterwards is what makes the study actionable, and after a restart the run
    exists only here.
    """
    from browser_agent.tasks import AGENT_RECIPE

    _archive(api_mod)
    assert api_mod.runner.tasks.get("old111") is None, "premise: no live task"

    out = await api_mod.say("old111", api_mod.SayRequest(text="go to https://duckduckgo.com"))

    nxt = out["task"]
    assert out["ran"] is True
    assert nxt["thread_id"] == "t-old", "the answer left the conversation"
    assert nxt["attempt"] == 2
    assert nxt["recipe"] == AGENT_RECIPE, "a deterministic recipe ignores prose"
    assert nxt["payload"]["url"] == "https://duckduckgo.com"
    assert nxt["payload"]["goal"] == "go to https://duckduckgo.com"


@pytest.mark.asyncio
async def test_the_archived_brief_carries_the_earlier_failure_forward(api_mod):
    """An "iterate on it" that does not say what already failed is a re-roll."""
    _archive(
        api_mod,
        activity=[{"at": 1.0, "kind": "error", "text": "clicking #submit timed out"}],
    )
    out = await api_mod.say("old111", api_mod.SayRequest(text="try the menu instead"))

    brief = out["task"]["payload"]["history"]
    assert "attempt 1" in brief
    assert "rate limited" in brief
    assert "clicking #submit timed out" in brief


@pytest.mark.asyncio
async def test_a_note_on_an_archived_run_behaves_like_a_note_on_a_live_one(api_mod):
    """Same meaning either side of a restart — the archive is not a second rule.

    ``kind`` is recorded on the message and does not change what happens, on
    either path. Pinning that here keeps the two branches from drifting into two
    definitions of what a message means.
    """
    _archive(api_mod)
    out = await api_mod.say("old111", api_mod.SayRequest(text="saw this too", kind="note"))

    assert out["ran"] is True
    msgs = api_mod.threads.for_thread("t-old")
    assert [m.text for m in msgs] == ["saw this too"]
    assert msgs[0].kind == "note"


@pytest.mark.asyncio
async def test_an_empty_message_is_refused_before_the_archive_is_touched(api_mod):
    from fastapi import HTTPException

    _archive(api_mod)
    with pytest.raises(HTTPException) as exc:
        await api_mod.say("old111", api_mod.SayRequest(text="   "))
    assert exc.value.status_code == 400
    assert api_mod.threads.for_thread("t-old") == []


@pytest.mark.asyncio
async def test_an_unknown_id_is_still_a_404(api_mod):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await api_mod.say("nope", api_mod.SayRequest(text="hello"))
    assert exc.value.status_code == 404


# -- History reads through to the archive ----------------------------------


def test_history_shows_saved_runs_after_a_restart(api_mod):
    """Keep every run is only true on the page if the page can see them.

    A deploy empties the in-memory task list; without this the operator would
    open the page they kept a run for and find nothing there.
    """
    _archive(api_mod, task_id="gone1", thread_id="t-gone", created_at=5.0)

    rows = api_mod._recent_tasks()

    assert [r["id"] for r in rows] == ["gone1"]
    assert rows[0]["archived"] is True
    assert rows[0]["status"] == "blocked"
    assert rows[0]["payload"]["url"] == "https://minesweeper.online/"
    # Same key the UI already reads off a live task, so one code path serves both.
    assert rows[0]["reads_instruction"] is False


def test_a_live_task_wins_over_its_own_saved_record(api_mod):
    """The archive row must not shadow the live one: only the live one has a control."""
    from browser_agent.tasks import Task, TaskStatus

    _archive(api_mod, task_id="both", thread_id="t-both", created_at=5.0)
    live = Task(recipe="plan.task", payload={"task": "hi"}, id="both",
                thread_id="t-both", status=TaskStatus.RUNNING, created_at=5.0)
    api_mod.runner.tasks[live.id] = live

    rows = api_mod._recent_tasks()

    assert [r["id"] for r in rows] == ["both"]
    assert rows[0]["archived"] is False
    assert rows[0]["status"] == "running"


def test_history_orders_live_and_saved_together_by_age(api_mod):
    """One list, newest first — two lists would make the operator read twice."""
    from browser_agent.tasks import Task, TaskStatus

    _archive(api_mod, task_id="old", thread_id="t1", created_at=1.0)
    live = Task(recipe="plan.task", payload={}, id="new", status=TaskStatus.DONE,
                created_at=9.0)
    api_mod.runner.tasks[live.id] = live

    rows = api_mod._recent_tasks()

    assert [r["id"] for r in rows] == ["new", "old"]


def test_a_bounded_history_is_still_bounded(api_mod):
    """The table is a fixed-height view, so the read-through has a cap too."""
    for i in range(6):
        _archive(api_mod, task_id=f"a{i}", thread_id=f"t{i}", created_at=float(i))

    rows = api_mod._recent_tasks(limit=3)

    assert [r["id"] for r in rows] == ["a5", "a4", "a3"]


@pytest.mark.asyncio
async def test_the_thread_survives_a_restart_through_the_archive(api_mod):
    """Opening an old thread after a deploy must show the run, not "no such".

    The History row is drawn from the in-memory tasks, so a restart empties it —
    but a thread the operator has not finished with has to stay reachable, which
    is what the archive fallback is for.
    """
    _archive(
        api_mod,
        activity=[{"at": 1.0, "kind": "gate", "text": "laya: no (conf 0.41)"}],
    )
    assert not any(t.thread_id == "t-old" for t in api_mod.runner.tasks.values())

    info = await api_mod.get_thread("t-old")

    assert info["archived"] is True
    assert [a["task_id"] for a in info["attempts"]] == ["old111"]
    # The decisions travel with the record, which is the point of archiving:
    # an evaluation walks these, and re-deriving them at read time is what drifts.
    assert [d["text"] for d in info["attempts"][0]["decisions"]] == ["laya: no (conf 0.41)"]


@pytest.mark.asyncio
async def test_a_live_thread_is_not_reported_as_archived(api_mod):
    """In-memory wins when it exists: only a live attempt has a running control."""
    from browser_agent.tasks import Task, TaskStatus

    _archive(api_mod, task_id="same", thread_id="t-live", created_at=1.0)
    live = Task(recipe="thread.ok", payload={}, thread_id="t-live",
                id="same", status=TaskStatus.RUNNING)
    api_mod.runner.tasks[live.id] = live

    info = await api_mod.get_thread("t-live")

    assert info["archived"] is False
    assert [a["id"] for a in info["attempts"]] == ["same"]


@pytest.mark.asyncio
async def test_an_empty_thread_is_a_404(api_mod):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await api_mod.get_thread("never-existed")
    assert exc.value.status_code == 404


def test_the_archive_is_listed_and_readable(api_mod):
    """The operator's ask: keep them, and be able to go back and evaluate them."""
    _archive(api_mod, task_id="r1", thread_id="t1", created_at=1.0)
    _archive(api_mod, task_id="r2", thread_id="t2", created_at=2.0)

    listed = asyncio.run(api_mod.list_runs(limit=50, thread_id=""))
    assert [r["task_id"] for r in listed["runs"]] == ["r2", "r1"]
    assert listed["count"] == 2
    assert listed["enabled"] is True

    one = asyncio.run(api_mod.get_run("r1"))
    assert one["recipe"] == "minesweeper.play"
    assert one["payload"]["url"] == "https://minesweeper.online/"
    assert one["status"] == "blocked"


def test_reading_an_archived_run_that_is_not_there_is_a_404(api_mod):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_mod.get_run("nope"))
    assert exc.value.status_code == 404


# -- the instruction that started the work ---------------------------------


@pytest.mark.asyncio
async def test_the_opening_instruction_is_recorded_in_the_thread(api_mod):
    """The reported gap: the thread opened on an attempt with no visible ask.

    The operator's first sentence is what every later turn is a correction *of*,
    so a thread that starts mid-story is the one thing the panel cannot explain.
    """
    out = await api_mod.create_task(
        api_mod.TaskRequest(recipe="plan.task", payload={"task": "find flights to Oslo"})
    )
    msgs = api_mod.threads.for_thread(out["thread_id"])

    assert [m.text for m in msgs] == ["find flights to Oslo"]
    assert msgs[0].role == "operator"
    assert msgs[0].kind == "instruction"
    # Not "close to": the ask caused the attempt, so it can never sort after it.
    # Dating it at record time made the thread open on an attempt that answered
    # a question nobody had asked yet, which is the bug this pins.
    assert msgs[0].at <= out["created_at"], "the ask must not follow the attempt"


@pytest.mark.asyncio
async def test_a_task_with_no_prose_records_nothing(api_mod):
    """A bare run has nothing to say; an empty turn would be noise in the feed."""
    out = await api_mod.create_task(
        api_mod.TaskRequest(recipe="minesweeper.play", payload={"url": "https://x.example/"})
    )
    assert api_mod.threads.for_thread(out["thread_id"]) == []


def test_the_recorded_instruction_survives_a_restart(api_mod):
    """It is written to the archive, so it is readable after the pod is replaced."""
    out = asyncio.run(api_mod.create_task(
        # The text field is the admin UI's own field name; it must be recorded
        # exactly as a `task` would be — the alias is not a second kind of ask.
        api_mod.TaskRequest(recipe="plan.task", payload={"text": "the alias counts too"})
    ))
    rows = api_mod.runs.messages(out["thread_id"])
    assert [m["text"] for m in rows] == ["the alias counts too"]
