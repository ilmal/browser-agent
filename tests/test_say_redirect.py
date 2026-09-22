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
