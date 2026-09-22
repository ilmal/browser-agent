"""The recipe library: reading, validating, and the overrides that reach recipes.

The load-bearing property is containment. ``api.py`` loads settings at import, so
a malformed file in the library must cost that one recipe and nothing else — a
pod that cannot start has no control plane left to fix it from. Several tests
below are entirely about that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from browser_agent import recipe_store, tasks
from browser_agent.plan_model import PlanRejected
from browser_agent.recipe_store import RecipeError, RecipeStore


@pytest.fixture
def library(tmp_path: Path) -> RecipeStore:
    return RecipeStore(tmp_path)


def _write(store: RecipeStore, name: str, data: object) -> Path:
    path = store.directory / name
    path.write_text(json.dumps(data))
    return path


_STEPS = [
    {"action": "navigate", "goal": "open the site", "text": "https://example.com"},
    {"action": "click", "goal": "press go", "selector": "button.go"},
]


# ---- reading ---------------------------------------------------------------


def test_a_stored_recipe_is_read_with_its_identity(library):
    _write(library, "scrape.json", {
        "name": "scrape", "description": "scrape a page",
        "entry_url": "https://example.com", "steps": _STEPS,
    })
    spec = library.get("scrape")
    assert spec is not None
    assert spec["name"] == "scrape"
    assert spec["description"] == "scrape a page"
    assert [s["action"] for s in spec["steps"]] == ["navigate", "click"]
    assert library.errors == []


def test_a_malformed_file_is_skipped_and_the_rest_survive(library):
    # The crashloop guard: three bad files, one good, and the good one is still
    # there with an honest error naming each bad one.
    _write(library, "reserved.json", {"name": "plan.task", "entry_url": "https://x/", "steps": _STEPS})
    _write(library, "selectorless.json", {
        "name": "guesser", "entry_url": "https://x/",
        "steps": [{"action": "click", "goal": "press it"}],
    })
    (library.directory / "broken.json").write_text("{not json")
    _write(library, "good.json", {"name": "good", "entry_url": "https://x/", "steps": _STEPS})

    assert set(library.specs()) == {"good"}
    assert len(library.errors) == 3
    joined = " ".join(library.errors)
    assert "reserved" in joined
    assert "needs a `selector`" in joined


def test_a_step_recipe_may_not_take_a_routed_name(library):
    with pytest.raises(PlanRejected, match="reserved"):
        recipe_store._validate(
            {"name": "agent.task", "entry_url": "https://x/", "steps": _STEPS},
            source="test",
        )


def test_a_bad_name_is_refused(library):
    with pytest.raises(PlanRejected, match="lowercase"):
        recipe_store._validate(
            {"name": "Not Valid!", "entry_url": "https://x/", "steps": _STEPS},
            source="test",
        )


def test_an_empty_step_list_is_refused(library):
    with pytest.raises(PlanRejected, match="non-empty"):
        recipe_store._validate({"name": "x", "entry_url": "https://x/", "steps": []}, source="test")


def test_a_missing_directory_reads_as_empty(tmp_path):
    store = RecipeStore(tmp_path / "absent")
    assert store.specs() == {}
    assert store.all_overrides() == {}
    assert store.errors == []


# ---- overrides -------------------------------------------------------------


def test_an_override_wins_and_is_type_checked(library):
    recipe_store.save_overrides(library, "x.post", {
        "selectors.submit": ["button.new"],
        "max_chars": 100,
    })
    assert library.cfg("x.post", "selectors.submit", ["old"]) == ["button.new"]
    assert library.cfg("x.post", "max_chars", 280) == 100
    # An untouched key and an unmentioned recipe both fall through untouched.
    assert library.cfg("x.post", "selectors.composer", ["old"]) == ["old"]
    assert library.cfg("linkedin.page_post", "max_chars", 3000) == 3000


def test_a_wrong_typed_override_returns_the_builtin(library):
    # A string where a selector list belongs is a bad edit, not a crash deep in
    # Playwright: the built-in literal is the answer and the mistake is logged.
    recipe_store.save_overrides(library, "x.post", {"selectors.submit": ["button.new"]})
    library._overrides["x.post"]["selectors.submit"] = "button.new"  # simulate a hand-edit
    assert library.cfg("x.post", "selectors.submit", ["old"]) == ["old"]


def test_an_unknown_or_unoverridable_key_is_refused(library):
    with pytest.raises(RecipeError, match="not an overridable key"):
        recipe_store.save_overrides(library, "x.post", {"no.such": 1})
    with pytest.raises(RecipeError, match="no overridable config"):
        recipe_store.save_overrides(library, "agent.task", {"entry_url": "https://x/"})
    with pytest.raises(RecipeError, match="http"):
        recipe_store.save_overrides(library, "x.post", {"entry_url": "javascript:alert(1)"})
    with pytest.raises(RecipeError, match="non-empty list"):
        recipe_store.save_overrides(library, "x.post", {"selectors.submit": []})


def test_saving_an_override_merges_rather_than_replaces(library):
    recipe_store.save_overrides(library, "x.post", {"max_chars": 100})
    merged = recipe_store.save_overrides(library, "x.post", {"selectors.submit": ["a"]})
    assert merged == {"max_chars": 100, "selectors.submit": ["a"]}


def test_deleting_removes_either_kind(library):
    recipe_store.save_overrides(library, "x.post", {"max_chars": 100})
    assert recipe_store.delete_recipe(library, "x.post") == "override"
    assert library.all_overrides() == {}
    assert library.cfg("x.post", "max_chars", 280) == 280

    recipe_store.save_step_recipe(library, {"name": "s", "entry_url": "https://x/", "steps": _STEPS})
    assert recipe_store.delete_recipe(library, "s") == "stored"
    assert library.specs() == {}

    with pytest.raises(RecipeError, match="no stored recipe"):
        recipe_store.delete_recipe(library, "s")


def test_a_saved_recipe_is_one_the_loader_accepts(library):
    # "Saved but unreadable" must be impossible: save goes through the same gate.
    saved = recipe_store.save_step_recipe(
        library, {"name": "s", "entry_url": "https://example.com", "steps": _STEPS}
    )
    assert saved["name"] == "s"
    assert library.get("s") == saved


def test_an_edit_lands_without_a_restart(library, monkeypatch):
    # kubelet syncs the mount; the store notices on its next read.
    monkeypatch.setattr(recipe_store.time, "monotonic", lambda: 1000.0)
    library.refresh(force=True)
    assert library.get("late") is None
    _write(library, "late.json", {"name": "late", "entry_url": "https://x/", "steps": _STEPS})
    monkeypatch.setattr(recipe_store.time, "monotonic", lambda: 2000.0)
    assert library.get("late") is not None


# ---- the registry overlay --------------------------------------------------


def test_installing_a_stored_recipe_keeps_foreign_registrations():
    """A recipe registered directly is nobody's to forget.

    ``stored_recipe_names`` used to mean "everything not built in", which made
    the library's cleanup delete a recipe another caller had registered — a test
    double, or an embedding caller — as soon as the library was read.
    """
    class Foreign:
        name = "foreign.recipe"
        description = "not from the library"
        entry_url = ""

        async def run(self, session, payload):  # pragma: no cover
            return {}

    tasks.register(Foreign())
    try:
        tasks.forget_recipe("foreign.recipe")  # the library's cleanup path
        assert tasks.get_recipe("foreign.recipe").name == "foreign.recipe"
    finally:
        tasks._REGISTRY.pop("foreign.recipe", None)


def test_a_stored_recipe_is_installed_and_can_be_forgotten(monkeypatch, tmp_path):
    from browser_agent.recipes import stored as stored_module

    # The store re-reads on its own clock (a short TTL, then the directory's
    # stamp); the test advances it rather than sleeping, exactly as the "an edit
    # lands without a restart" test does.
    clock = {"t": 1000.0}
    monkeypatch.setattr(recipe_store.time, "monotonic", lambda: clock["t"])

    store = RecipeStore(tmp_path)
    _write(store, "mine.json", {"name": "mine", "entry_url": "https://x/", "steps": _STEPS})
    monkeypatch.setattr(stored_module, "store_for", lambda settings: store)

    stored_module.load_stored_recipes()
    try:
        assert tasks.get_recipe("mine").name == "mine"
        assert any(r["name"] == "mine" and r["origin"] == "stored" for r in tasks.list_recipes())

        # Deleting it from the library must stop it being runnable, without a
        # restart: the next read of the registry forgets it.
        (tmp_path / "mine.json").unlink()
        clock["t"] += 60.0
        with pytest.raises(KeyError):
            tasks.get_recipe("mine")
    finally:
        tasks.forget_recipe("mine")
