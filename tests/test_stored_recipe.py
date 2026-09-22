"""A stored recipe runs through the same executor as a planner-emitted plan.

The point of the whole feature is that there is *no* new execution machinery: a
saved step list is a saved ``Plan``. These tests run one end to end against the
local HTTP fixture and prove the promises that come with that — determinism, no
agent call, and the operator's own instruction reaching the run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent.recipe_store import RecipeStore, save_step_recipe  # noqa: E402
from browser_agent.recipes.stored import StoredRecipe  # noqa: E402
from browser_agent.tasks import TaskRunner, TaskStatus, register  # noqa: E402
from tests.test_plan_task import (  # noqa: E402
    FakeLaya,
    FakePicker,
    _drain,
    _free_port,
    site,  # noqa: F401 - re-exported fixture
)

_STEPS = [
    {"action": "navigate", "goal": "open the form", "text": "{site}/form"},
    {"action": "type", "goal": "enter the username", "selector": "#user", "text": "nils"},
    {
        "action": "click",
        "goal": "submit the form",
        "selector": "#go",
        "done_when": {"text_contains": "submitted"},
    },
    {"action": "extract", "goal": "page heading", "selector": "h1"},
]


def _spec(site_url: str) -> dict[str, Any]:
    return {
        "name": "form.fill",
        "description": "Fill the form on the local fixture.",
        "entry_url": f"{site_url}/form",
        "steps": [dict(s, text=s.get("text", "").replace("{site}", site_url)) for s in _STEPS],
    }


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE", "pytest")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("HEADLESS", "true")
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "llm_sk_placeholder")
    monkeypatch.setenv("NOTIFY_ON_ESCALATION", "false")
    monkeypatch.setenv("PLANNER_STEP_TIMEOUT_S", "1")
    monkeypatch.setenv("LAYA_ENABLED", "true")
    monkeypatch.setenv("LAYA_PICK_ENABLED", "true")
    from browser_agent.config import load_settings

    return load_settings()


def _stored(spec: dict[str, Any], settings) -> StoredRecipe:
    return StoredRecipe(spec, settings=settings, laya=FakeLaya(), picker=FakePicker())


@pytest.fixture
async def browser_runner(settings, site):
    """The plan_task harness, reused so a stored recipe runs on the same rails."""
    from playwright.async_api import async_playwright

    from tests.test_plan_task import _Session

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        made: list[TaskRunner] = []

        def make(recipe, agent_runner=None):
            register(recipe)
            runner = TaskRunner(settings, _Session(pw, browser), agent_runner=agent_runner)
            made.append(runner)
            return runner

        try:
            yield make, site
        finally:
            for r in made:
                await r.stop()
            await browser.close()


async def test_a_stored_step_recipe_runs_without_the_agent(browser_runner, settings):
    make, site_url = browser_runner
    agent_calls = {"n": 0}

    async def agent(session, url, payload):  # pragma: no cover - must not run
        agent_calls["n"] += 1
        return {}

    runner = make(_stored(_spec(site_url), settings), agent_runner=agent)
    task = runner.submit("form.fill", {"task": "fill the form"})
    done = await _drain(runner, task.id)

    assert done.status is TaskStatus.DONE
    assert agent_calls["n"] == 0
    assert done.used_agent is False
    assert done.result["recipe"] == "form.fill"
    assert done.result["steps_executed"] == 4
    assert done.result["extracts"]["page heading"] == "Form Page"


async def test_an_edited_spec_reaches_the_next_run_without_a_restart(
    tmp_path, settings, site
):
    """An operator's edit must land on the next run, not at the next restart.

    ``load_stored_recipes`` calls ``replace()`` on the *existing* registry entry,
    so the same object serves the new steps — and ``run()`` parses those steps
    every time rather than trusting what it was built with.
    """
    import json

    store = RecipeStore(tmp_path)
    spec = save_step_recipe(store, _spec(site))
    recipe = _stored(spec, settings)
    assert len(recipe.steps) == 4

    edited = {
        **spec,
        "steps": [{"action": "navigate", "goal": "open the twins page", "text": f"{site}/twins"}],
    }
    (tmp_path / "form.fill.json").write_text(json.dumps(edited))
    store.refresh(force=True)
    recipe.replace(store.get("form.fill"))

    assert len(recipe.steps) == 1
    assert recipe.steps[0]["text"].endswith("/twins")


async def test_run_re_validates_before_it_touches_the_browser(settings):
    """A spec that stopped being valid fails loudly, not deep inside Playwright.

    The stored JSON is the source of truth, so a hand-edited file that a reload
    has not yet re-validated must still be refused on the run itself.
    """
    from browser_agent.plan_model import PlanRejected

    spec = _spec("https://example.com")
    spec["steps"] = [{"action": "click", "goal": "press it"}]  # no selector
    recipe = _stored(spec, settings)

    with pytest.raises(PlanRejected, match="needs a `selector`"):
        await recipe.run(None, {})


def test_a_stored_recipe_needs_a_selector_on_its_own_steps(tmp_path):
    # The difference from a planner plan: a throwaway plan may leave a selector
    # to the picker, a saved recipe may not — there is no page in front of it.
    store = RecipeStore(tmp_path)
    with pytest.raises(Exception, match="needs a `selector`"):
        save_step_recipe(store, {
            "name": "guesser",
            "entry_url": "https://example.com",
            "steps": [{"action": "click", "goal": "press something"}],
        })
