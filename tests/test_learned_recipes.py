"""Learned recipes: what a successful agent run becomes, and what may use it.

Nils's idea (2026-09-23) is that the first run of a task pays for the agent and
every later one replays a saved recipe for free. The tests here are almost all
about *refusal*, because that is where the idea is safe or not: a recipe that
misrepresents what the agent did, or that gets trusted before it has been
replayed, hands a later request another task's result while claiming it is the
same one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from browser_agent import recipe_store
from browser_agent.recipe_store import RecipeStore, record_replay, save_learned_spec
from browser_agent.recipes.harvest import harvest, slug_for

# -- fakes for browser-use's history ---------------------------------------
#
# The real shapes (verified in the deployed image, browser-use 0.13.10): an
# action is a pydantic model with exactly one field set, the field name IS the
# **registry** action name, and the element it was dispatched on rides in
# ``state.interacted_element`` at the same index. Modelling those three facts is
# all the harvester reads.
#
# The registry names are NOT the handler or class names, and that distinction is
# the whole bug this file exists to keep caught: ``Tools`` registers ``click``,
# ``input`` and ``navigate`` (``tools/service.py``) and builds the per-action
# models from those keys (``registry/service.py::create_action_model``), so
# ``ClickElementAction`` dumps as ``{"click": {...}}`` — never
# ``click_element_by_index``. A fake that uses a name prod never emits tests the
# mapping against fiction, which is exactly how a harvest that refused every
# real run stayed green. Use ``_REGISTRY_NAMES`` below, not the class names.


#: The action names the installed browser-use actually puts in ``model_dump``.
#: Read from the deployed image: ``sorted(Tools().registry.registry.actions)``
#: minus the non-mutating ones the harvester never maps. Kept as one tuple so a
#: future rename is a one-line change here plus one in ``harvest.py``.
_REGISTRY_NAMES = ("navigate", "click", "input", "scroll", "done", "extract")


class _Action:
    """One browser-use action model.

    ``_Action(click={...})`` — prod's registry vocabulary, the way every test
    should build one. ``_Action(name="click_element_by_index", index=5)`` — an
    explicit ``name``, for pinning a legacy alias the package may emit again.
    """

    def __init__(self, *, name=None, **kwargs):
        if name is not None:
            assert not kwargs, "an explicit name carries no field shorthand"
            self._d = {name: {}}
            return
        assert len(kwargs) == 1, "an action model carries exactly one field"
        (field,) = kwargs
        assert field in _REGISTRY_NAMES, (
            f"{field!r} is not a name the installed browser-use emits "
            f"({_REGISTRY_NAMES}); use name= to pin a legacy alias deliberately"
        )
        self._d = kwargs

    def model_dump(self, **_kw):
        return dict(self._d)


class _Element:
    def __init__(self, attributes=None, x_path="", ax_name=""):
        self.attributes = attributes or {}
        self.x_path = x_path
        self.ax_name = ax_name


class _Output:
    def __init__(self, action, next_goal=""):
        self.action = action
        self.next_goal = next_goal


class _State:
    def __init__(self, elements):
        self.interacted_element = elements


class _Item:
    def __init__(self, output, elements):
        self.model_output = output
        self.state = _State(elements)


class _History:
    def __init__(self, items):
        self.history = items


def _run(*items) -> _History:
    return _History(list(items))


def _step(action, elements=None, goal="") -> _Item:
    return _Item(_Output([action], goal), elements or [None])


# -- harvesting ------------------------------------------------------------


def test_a_click_and_type_run_becomes_a_plan():
    # The plain case the idea is about: open a page, type into a field, press a
    # button. Every target has a durable selector, so a recipe is earned.
    history = _run(
        _step(_Action(navigate={"url": "https://example.com/search"})),
        _step(
            _Action(input={"index": 2, "text": "stockholm"}),
            [_Element(attributes={"name": "q"})],
            goal="type the origin",
        ),
        _step(
            _Action(click={"index": 5}),
            [_Element(attributes={"id": "search-btn"})],
            goal="run the search",
        ),
    )
    spec = harvest(history, entry_url="https://example.com/search", goal="search",
                   task_text="find a flight from stockholm")
    assert spec is not None
    assert [s["action"] for s in spec["steps"]] == ["navigate", "type", "click"]
    assert spec["steps"][0]["text"] == "https://example.com/search"
    assert spec["steps"][1]["selector"] == '[name="q"]'
    assert spec["steps"][1]["text"] == "stockholm"
    assert spec["steps"][2]["selector"] == "#search-btn"
    assert spec["origin"] == "learned"
    assert spec["unverified"] is True
    assert spec["replays"] == 0


def test_the_registry_names_are_what_the_harvester_maps():
    # The bug this file shipped with: the fakes spoke ``click_element_by_index``
    # / ``input_text`` (browser-use's *handler* names) while the installed
    # package emits ``click`` / ``input`` (its *registry* names). Every real run
    # was refused, and the tests were green because they asserted the fiction.
    # Pin the mapping against the vocabulary prod emits.
    from browser_agent.recipes.harvest import _STEP_FOR_ACTION

    for name in ("navigate", "click", "input"):
        assert name in _STEP_FOR_ACTION, (
            f"{name!r} is a registry name the installed browser-use emits; without "
            f"it every run containing that action is refused"
        )


def test_the_legacy_handler_names_still_map():
    # The package is unpinned, so a release that renames ``click`` back to
    # ``click_element_by_index`` must not silently stop harvesting.
    history = _run(
        _step(_Action(name="click_element_by_index"),
              [_Element(attributes={"id": "go"})]),
    )
    spec = harvest(history, entry_url="https://example.com", goal="g")
    assert spec is not None
    assert spec["steps"][0]["selector"] == "#go"


def test_an_unmappable_action_costs_the_whole_recipe():
    # A scroll is not a step the executor has, so the run cannot be represented
    # faithfully. Half a recipe would replay a *different* task.
    history = _run(
        _step(_Action(navigate={"url": "https://example.com"})),
        _step(_Action(scroll={"down": True, "pages": 2})),
        _step(_Action(click={"index": 5}),
              [_Element(attributes={"id": "go"})]),
    )
    assert harvest(history, entry_url="https://example.com", goal="g") is None


def test_a_click_with_no_durable_target_is_not_harvested():
    # browser-use names elements by its own view index, which means nothing to a
    # later run. No selector, no recipe.
    history = _run(
        _step(_Action(click={"index": 5}), [_Element()]),
    )
    assert harvest(history, entry_url="https://example.com", goal="g") is None


def test_the_xpath_is_the_last_resort_selector():
    history = _run(
        _step(_Action(click={"index": 1}),
              [_Element(x_path="/html/body/div[3]/button")]),
    )
    spec = harvest(history, entry_url="https://example.com", goal="g")
    assert spec is not None
    assert spec["steps"][0]["selector"] == "xpath=/html/body/div[3]/button"


def test_a_trailing_done_is_not_a_step_and_mid_run_done_disqualifies():
    # Done at the end is how every successful run finishes; it is not replayable
    # work. Done in the middle means the run continued past it, which we cannot
    # reproduce — so that run is not harvested.
    ok = _run(
        _step(_Action(navigate={"url": "https://example.com"})),
        _step(_Action(done={"text": "finished"})),
    )
    spec = harvest(ok, entry_url="https://example.com", goal="g")
    assert spec is not None
    assert [s["action"] for s in spec["steps"]] == ["navigate"]

    mid = _run(
        _step(_Action(done={"text": "looks done"})),
        _step(_Action(click={"index": 2}),
              [_Element(attributes={"id": "next"})]),
    )
    assert harvest(mid, entry_url="https://example.com", goal="g") is None


def test_a_type_step_with_no_text_is_not_harvested():
    history = _run(
        _step(_Action(input={"index": 2, "text": ""}),
              [_Element(attributes={"name": "q"})]),
    )
    assert harvest(history, entry_url="https://example.com", goal="g") is None


def test_a_goal_too_short_to_be_a_search_cannot_be_a_recipe():
    # No http(s) entry_url: a recipe must have a page to start from.
    assert harvest(_run(), entry_url="", goal="g") is None
    assert harvest(_run(), entry_url="about:blank", goal="g") is None


def test_harvest_never_raises_on_a_structure_it_does_not_know():
    # The dependency is unpinned, so a moved attribute must read as "no recipe".
    class _Junk:
        pass

    assert harvest(_Junk(), entry_url="https://example.com", goal="g") is None
    assert harvest(_run(_Item(_Junk(), [])), entry_url="https://example.com", goal="g") is None


def test_the_name_carries_a_hash_so_two_goals_do_not_collide():
    a = slug_for("find a flight from stockholm")
    b = slug_for("find a flight from stockholm to seattle")
    assert a.startswith("learned-find-a-flight")
    assert a != b


# -- the store's learned half ----------------------------------------------


@pytest.fixture
def learned(tmp_path: Path) -> RecipeStore:
    return RecipeStore(tmp_path)


_LEARNED_SPEC = {
    "name": "learned-search-abc123",
    "description": "find a flight from stockholm",
    "entry_url": "https://example.com/search",
    "steps": [
        {"action": "navigate", "text": "https://example.com/search"},
        {"action": "click", "goal": "run the search", "selector": "#search-btn"},
    ],
    "origin": "learned",
    "unverified": True,
    "replays": 0,
}


def test_a_learned_spec_round_trips_with_its_provenance(learned):
    save_learned_spec(learned, _LEARNED_SPEC)
    spec = learned.get("learned-search-abc123")
    assert spec is not None
    # The provenance must survive validation: dropping it would silently promote
    # an agent-written recipe to one the router trusts.
    assert spec["origin"] == "learned"
    assert spec["unverified"] is True
    assert learned.trusted_learned() == []


def test_replays_promote_a_recipe_only_after_the_floor(learned, monkeypatch):
    monkeypatch.setenv("LEARNED_RECIPE_MIN_REPLAYS", "2")
    save_learned_spec(learned, _LEARNED_SPEC)

    record_replay(learned, "learned-search-abc123", ok=True)
    assert learned.get("learned-search-abc123")["replays"] == 1
    assert learned.trusted_learned() == []  # one clean run is not proof

    record_replay(learned, "learned-search-abc123", ok=True)
    spec = learned.get("learned-search-abc123")
    assert spec["replays"] == 2
    assert spec["unverified"] is False
    assert [s["name"] for s in learned.trusted_learned()] == ["learned-search-abc123"]


def test_a_failed_replay_neither_promotes_nor_punishes(learned, monkeypatch):
    monkeypatch.setenv("LEARNED_RECIPE_MIN_REPLAYS", "2")
    save_learned_spec(learned, _LEARNED_SPEC)
    record_replay(learned, "learned-search-abc123", ok=True)
    record_replay(learned, "learned-search-abc123", ok=False)
    # The failed run did not count toward promotion, and it did not reset the
    # clean one either — the page moved, the recipe is not at fault.
    assert learned.get("learned-search-abc123")["replays"] == 1
    assert learned.trusted_learned() == []


def test_a_selectorless_learned_spec_is_refused(learned):
    bad = dict(_LEARNED_SPEC, steps=[{"action": "click", "goal": "press it"}])
    with pytest.raises(Exception, match="needs a `selector`"):
        save_learned_spec(learned, bad)


def test_a_hand_authored_recipe_is_never_treated_as_learned(tmp_path):
    store = RecipeStore(tmp_path)
    spec = {
        "name": "scrape", "description": "d", "entry_url": "https://example.com",
        "steps": [{"action": "navigate", "text": "https://example.com"}],
    }
    recipe_store.save_step_recipe(store, spec)
    assert store.learned() == []
    assert store.trusted_learned() == []


def test_the_learned_directory_is_separate_from_the_operator_library(tmp_path):
    # The whole point: a learned recipe carries text typed into a live session,
    # so it is written to the pod's own volume and never published cluster-wide.
    from browser_agent.config import Settings

    s = Settings(
        profile="p", profiles_root=tmp_path, data_root=tmp_path,
        screen_width=1, screen_height=1, screen_depth=24, novnc_port=1,
        browser_base_url="", headless=True, slow_mo_ms=0,
        llm_base_url="", llm_model="", llm_enabled=False, llm_api_key="",
        llm_source="", llm_client_id="", planner_model="", planner_max_steps=1,
        planner_step_timeout_s=1, planner_client_id="", agent_max_steps=1,
        agent_timeout_s=1, laya_enabled=False, laya_decide_url="",
        laya_pick_enabled=False, laya_min_confidence=0.75,
        laya_game_min_confidence=0.55, laya_max_candidates=10, laya_pick_retries=0,
        api_port=1, control_token="", ops_alert_url="", notify_on_escalation=False,
        browser_proxy="", recipes_dir=tmp_path / "recipes",
        learned_recipes_dir=tmp_path / "learned",
    )
    assert s.learned_recipes_dir != s.recipes_dir
    store = recipe_store.learned_store_for(s)
    assert store is not None
    assert store.directory == tmp_path / "learned"


def test_learning_is_off_when_the_directory_is_empty(tmp_path):
    from browser_agent.config import Settings

    s = Settings(
        profile="p", profiles_root=tmp_path, data_root=tmp_path,
        screen_width=1, screen_height=1, screen_depth=24, novnc_port=1,
        browser_base_url="", headless=True, slow_mo_ms=0,
        llm_base_url="", llm_model="", llm_enabled=False, llm_api_key="",
        llm_source="", llm_client_id="", planner_model="", planner_max_steps=1,
        planner_step_timeout_s=1, planner_client_id="", agent_max_steps=1,
        agent_timeout_s=1, laya_enabled=False, laya_decide_url="",
        laya_pick_enabled=False, laya_min_confidence=0.75,
        laya_game_min_confidence=0.55, laya_max_candidates=10, laya_pick_retries=0,
        api_port=1, control_token="", ops_alert_url="", notify_on_escalation=False,
        browser_proxy="", learn_recipes=False,
    )
    assert recipe_store.learned_store_for(s) is None


def test_learned_specs_are_valid_json_a_reload_can_read(learned):
    save_learned_spec(learned, _LEARNED_SPEC)
    path = learned.directory / "learned-search-abc123.json"
    assert json.loads(path.read_text())["origin"] == "learned"


# -- the similarity gate ----------------------------------------------------


class _GateStub:
    """Stands in for LayaGate.choose: records the question, returns a verdict."""

    calls: list[dict] = []
    answer: tuple[int | None, float] = (None, 0.0)

    def __init__(self, _settings):
        pass

    async def choose(self, question, lines, state_text):
        _GateStub.calls.append(
            {"question": question, "lines": lines, "state": state_text}
        )
        return _GateStub.answer


def _settings_with_learned(tmp_path, monkeypatch, **over):
    monkeypatch.setenv("LEARNED_RECIPE_MIN_REPLAYS", "1")
    from browser_agent.config import load_settings

    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "d"))
    monkeypatch.setenv("LAYA_ENABLED", "true")
    monkeypatch.setenv("LEARNED_RECIPES_DIR", str(tmp_path / "learned"))
    s = load_settings()
    for k, v in over.items():
        object.__setattr__(s, k, v)
    return s


def _trusted(store):
    spec = dict(_LEARNED_SPEC, unverified=False, replays=2)
    save_learned_spec(store, spec)
    return spec


async def test_the_gate_matches_a_few_trusted_recipes(tmp_path, monkeypatch):
    s = _settings_with_learned(tmp_path, monkeypatch)
    store = recipe_store.learned_store_for(s)
    _trusted(store)
    _GateStub.calls = []
    _GateStub.answer = (0, 0.9)
    monkeypatch.setattr("browser_agent.laya_gate.LayaGate", _GateStub)

    from browser_agent.router import learned_match

    name = await learned_match("find me a flight from stockholm", "plan.task", s)
    assert name == "learned-search-abc123"
    call = _GateStub.calls[0]
    assert call["lines"] == ["find a flight from stockholm"]
    assert "New request: find me a flight from stockholm" == call["state"]


async def test_the_gate_declines_below_the_confidence_floor(tmp_path, monkeypatch):
    s = _settings_with_learned(tmp_path, monkeypatch)
    _trusted(recipe_store.learned_store_for(s))
    _GateStub.answer = (0, 0.4)  # the recorded flat-probability failure shape
    monkeypatch.setattr("browser_agent.laya_gate.LayaGate", _GateStub)

    from browser_agent.router import learned_match

    assert await learned_match("something else entirely", "plan.task", s) is None


async def test_the_gate_is_not_asked_without_a_trusted_recipe(tmp_path, monkeypatch):
    s = _settings_with_learned(tmp_path, monkeypatch)
    save_learned_spec(recipe_store.learned_store_for(s), _LEARNED_SPEC)  # unverified
    _GateStub.calls = []
    monkeypatch.setattr("browser_agent.laya_gate.LayaGate", _GateStub)

    from browser_agent.router import learned_match

    assert await learned_match("find a flight", "plan.task", s) is None
    assert _GateStub.calls == [], "laya was asked to match nothing trusted"


async def test_the_gate_is_not_asked_for_a_named_recipe(tmp_path, monkeypatch):
    s = _settings_with_learned(tmp_path, monkeypatch)
    _trusted(recipe_store.learned_store_for(s))
    _GateStub.calls = []
    monkeypatch.setattr("browser_agent.laya_gate.LayaGate", _GateStub)

    from browser_agent.router import learned_match

    assert await learned_match("post hello", "linkedin.page_post", s) is None
    assert _GateStub.calls == []


async def test_too_many_candidates_are_not_ranked(tmp_path, monkeypatch):
    # Past ~10 options Laya's confidence is uncalibrated; the honest answer is
    # "cannot match", not a guess.
    s = _settings_with_learned(tmp_path, monkeypatch)
    store = recipe_store.learned_store_for(s)
    for i in range(11):
        save_learned_spec(store, dict(
            _LEARNED_SPEC, name=f"learned-x{i:02d}", unverified=False, replays=2,
        ))
    _GateStub.calls = []
    monkeypatch.setattr("browser_agent.laya_gate.LayaGate", _GateStub)

    from browser_agent.router import learned_match

    assert await learned_match("anything", "plan.task", s) is None
    assert _GateStub.calls == []
