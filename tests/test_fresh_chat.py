"""What "New chat" does when the first sentence is sent.

The composer on an empty sheet is the page's front door: one message, no
thread, no form. It has to behave like talking to a person — the sentence
becomes work — and it routes by exactly the rule Run a task uses, so the two
doors can never disagree about what a sentence means.
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


def _live_tasks(api):
    return set(api.runner.tasks)


@pytest.mark.asyncio
async def test_a_sentence_with_a_url_starts_a_freeform_run(api_mod):
    from browser_agent.tasks import AGENT_RECIPE

    out = await api_mod.fresh_chat(
        api_mod.FreshChatRequest(text="Go to https://example.com and tell me what it is for")
    )

    assert out["recipe"] == AGENT_RECIPE
    assert out["status"] == "queued"
    assert out["payload"]["url"] == "https://example.com"
    assert out["payload"]["goal"] == "Go to https://example.com and tell me what it is for"
    assert out["thread_id"], "the run landed outside any conversation"


@pytest.mark.asyncio
async def test_a_sentence_a_recipe_claims_runs_that_recipe(api_mod):
    """The front door and Run a task agree on what a minesweeper ask means.

    A claiming recipe needs no URL in the prose — the freeform-only rule must
    not be applied to text a recipe understood.
    """
    out = await api_mod.fresh_chat(
        api_mod.FreshChatRequest(text="play a game of minesweeper on a site you can reach")
    )

    assert out["recipe"] == "minesweeper.play"


@pytest.mark.asyncio
async def test_a_freeform_sentence_without_a_url_is_refused_before_submit(api_mod):
    """Queueing a task destined to fail is not an answer; refuse at the door."""
    from fastapi import HTTPException

    before = _live_tasks(api_mod)
    with pytest.raises(HTTPException) as exc:
        await api_mod.fresh_chat(
            api_mod.FreshChatRequest(text="tell me what this page is for")
        )
    assert exc.value.status_code == 400
    assert "paste the link" in exc.value.detail
    assert _live_tasks(api_mod) == before, "a doomed task was queued anyway"


@pytest.mark.asyncio
async def test_an_empty_sentence_is_refused(api_mod):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await api_mod.fresh_chat(api_mod.FreshChatRequest(text="   "))
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_each_message_starts_its_own_conversation(api_mod):
    """"Start over" means a new thread, not a second attempt on the old one."""
    first = await api_mod.fresh_chat(
        api_mod.FreshChatRequest(text="Go to https://a.example/ and read it")
    )
    second = await api_mod.fresh_chat(
        api_mod.FreshChatRequest(text="Go to https://b.example/ and read it")
    )

    assert first["thread_id"] != second["thread_id"]
    assert first["id"] != second["id"]


@pytest.mark.asyncio
async def test_the_opening_sentence_is_recorded_in_the_new_thread(api_mod):
    """A fresh sheet opens mid-story otherwise: the ask is the first message."""
    out = await api_mod.fresh_chat(
        api_mod.FreshChatRequest(text="Go to https://example.com and read it")
    )
    msgs = api_mod.threads.for_thread(out["thread_id"])

    assert [m.text for m in msgs] == ["Go to https://example.com and read it"]
    assert msgs[0].role == "operator"
    assert msgs[0].kind == "instruction"
    assert msgs[0].at <= out["created_at"], "the ask must not follow the attempt"


@pytest.mark.asyncio
async def test_trailing_punctuation_is_not_part_of_the_extracted_url(api_mod):
    out = await api_mod.fresh_chat(
        api_mod.FreshChatRequest(text="read https://example.com/docs.")
    )
    assert out["payload"]["url"] == "https://example.com/docs"
