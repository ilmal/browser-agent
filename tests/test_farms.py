"""Farms: composition, the store, the fire loop, and the hub routes.

The runner tests drive ``tick()`` by hand with an injected clock and fake pod
endpoints — the loop's real schedule (2s tick, 5s refresh) is a deployment
rhythm, and tests that sleep are tests that flake. The route tests silence the
app-level runner (one-hour tick) so membership state is deterministic at
assertion time.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from browser_agent import config, farms, hub, registry


# -- helpers ----------------------------------------------------------------


def _bot(profile: str, **kw) -> registry.Bot:
    return registry.Bot(profile=profile, **kw)


def _make_runner(store, *, now, post=None, get_state=None, refresh_every=5.0):
    settings = SimpleNamespace(
        api_port=8000,
        control_token="tok",
        bot_url_template="http://profile-{profile}:{api_port}",
    )
    # None stays None on purpose: FarmRunner falls back to its real HTTP
    # methods then, which is exactly what the mock-transport test wants.
    kwargs = {}
    if post is not None:
        kwargs["post"] = post
    if get_state is not None:
        kwargs["get_state"] = get_state
    return farms.FarmRunner(store, settings, now=now, refresh_every=refresh_every,
                            **kwargs)


def _store(tmp_path, farm_list):
    """A real store on disk — the tick persists on change, and a fake path
    would crash the very persistence under test."""
    store = farms.FarmStore(tmp_path / "farms.json")
    store.farms = farm_list
    return store


def _farm(profiles, *, mode="parallel", start_at=None, stagger=0, now=0.0):
    members = farms.build_members(
        profiles, {p: _bot(p) for p in profiles}, task="T", mode=mode,
        stagger_seconds=stagger, start_at=start_at, now=now,
    )
    return farms.Farm(id="farm1", name="", recipe="plan.task", task="T", url="https://x",
                      mode=mode, stagger_seconds=stagger, created_at=now, members=members)


# -- composition ------------------------------------------------------------


class TestComposeTaskText:
    def test_with_background(self):
        bot = _bot("kai", name="Kai", background="Ex-SAS, now a travel writer.")
        text = farms.compose_task_text(bot, "Find a flight to Tokyo.")
        assert text == (
            "You are Kai. Ex-SAS, now a travel writer.\n\nTASK: Find a flight to Tokyo."
        )

    def test_without_background_omits_the_clause(self):
        text = farms.compose_task_text(_bot("kai", name="Kai"), "Find a flight.")
        assert text == "You are Kai.\n\nTASK: Find a flight."

    def test_name_falls_back_to_profile(self):
        assert farms.compose_task_text(_bot("linkedin"), "Post.").startswith(
            "You are linkedin."
        )


class TestBuildMembers:
    def test_parallel_starts_everyone_now(self):
        members = farms.build_members(
            ["a", "b"], {"a": _bot("a"), "b": _bot("b")},
            task="T", mode="parallel", stagger_seconds=60, start_at=None, now=100.0,
        )
        assert [m.start_at for m in members] == [100.0, 100.0]

    def test_stagger_follows_assignment_order(self):
        members = farms.build_members(
            ["a", "b", "c"], {p: _bot(p) for p in "abc"},
            task="T", mode="stagger", stagger_seconds=60, start_at=None, now=100.0,
        )
        assert [m.start_at for m in members] == [100.0, 160.0, 220.0]

    def test_schedule_starts_everyone_at_the_epoch(self):
        members = farms.build_members(
            ["a", "b"], {"a": _bot("a"), "b": _bot("b")},
            task="T", mode="schedule", stagger_seconds=60, start_at=999.0, now=100.0,
        )
        assert [m.start_at for m in members] == [999.0, 999.0]

    def test_task_text_frozen_per_identity(self):
        members = farms.build_members(
            ["a"], {"a": _bot("a", name="Ada", background="bg")},
            task="T", mode="parallel", stagger_seconds=0, start_at=None, now=0.0,
        )
        assert members[0].task_text == "You are Ada. bg\n\nTASK: T"


def test_cancel_farm_spares_running_members():
    farm = farms.Farm(id="f", name="", recipe="r", task="t", url="", mode="stagger",
                      stagger_seconds=60, created_at=0.0, members=[
                          farms.Member(profile="a", status="pending"),
                          farms.Member(profile="b", status="running"),
                          farms.Member(profile="c", status="done"),
                      ])
    farms.cancel_farm(farm)
    assert [m.status for m in farm.members] == ["cancelled", "running", "done"]
    assert farm.status == "cancelled"


# -- the store --------------------------------------------------------------


class TestFarmStore:
    def test_create_persists_and_roundtrips(self, tmp_path):
        path = tmp_path / "farms.json"
        store = farms.FarmStore(path)
        farm = store.create(
            name="Tokyo", recipe="plan.task", task="Fly", url="https://x",
            mode="stagger", stagger_seconds=30,
            members=[farms.Member(profile="a", task_text="You are a.\n\nTASK: Fly",
                                  start_at=5.0)],
        )
        assert path.exists()
        again = farms.FarmStore(path)
        loaded = again.get(farm.id)
        assert loaded is not None
        assert loaded.name == "Tokyo"
        assert loaded.members[0].task_text == "You are a.\n\nTASK: Fly"
        assert loaded.members[0].start_at == 5.0

    def test_all_is_newest_first(self, tmp_path):
        store = farms.FarmStore(tmp_path / "farms.json")
        old = store.create(name="old", recipe="r", task="t", url="", mode="parallel",
                           stagger_seconds=0, members=[])
        old.created_at = 1.0
        new = store.create(name="new", recipe="r", task="t", url="", mode="parallel",
                           stagger_seconds=0, members=[])
        new.created_at = 2.0
        store.persist()
        assert [f.id for f in store.all()] == [new.id, old.id]

    def test_remove(self, tmp_path):
        store = farms.FarmStore(tmp_path / "farms.json")
        farm = store.create(name="", recipe="r", task="t", url="", mode="parallel",
                            stagger_seconds=0, members=[])
        assert store.remove(farm.id) is True
        assert store.remove(farm.id) is False
        assert farms.FarmStore(tmp_path / "farms.json").get(farm.id) is None

    def test_corrupt_file_degrades_to_empty(self, tmp_path):
        path = tmp_path / "farms.json"
        path.write_text("{not json")
        assert farms.FarmStore(path).farms == []

    def test_cap_drops_oldest_finished_first(self, tmp_path):
        store = farms.FarmStore(tmp_path / "farms.json")
        made = []
        for i in range(farms.MAX_FARMS + 2):
            f = store.create(name=str(i), recipe="r", task="t", url="", mode="parallel",
                             stagger_seconds=0, members=[])
            f.created_at = float(i)
            if i < 2:
                f.status = "complete"
            made.append(f)
        store.persist()
        kept = farms.FarmStore(tmp_path / "farms.json")
        assert len(kept.farms) == farms.MAX_FARMS
        assert all(f.id not in {made[0].id, made[1].id} for f in kept.farms)


# -- the fire loop ----------------------------------------------------------


class TestFire:
    def test_not_due_does_not_fire(self, tmp_path):
        farm = _farm(["a"], mode="schedule", start_at=100.0, now=0.0)
        store = _store(tmp_path, [farm])
        calls = []

        async def post(profile, f):
            calls.append(profile)
            return {"id": "t1"}

        runner = _make_runner(store, now=lambda: 0.0, post=post)
        asyncio.run(runner.tick())
        assert calls == []
        assert farm.members[0].status == "pending"
        assert farm.status == "pending"

    def test_fires_composed_payload_with_all_aliases(self, tmp_path, monkeypatch):
        """The agent reads goal first; a payload missing it lets a stale
        instruction win. This drives the real HTTP path against a mock
        transport, so the URL, the bearer header and all three aliases are
        asserted as sent, not as constructed."""
        bot = _bot("a", name="Ada", background="bg words")
        farm = farms.Farm(id="f", name="", recipe="plan.task", task="Find it",
                          url="https://x", mode="parallel", stagger_seconds=0,
                          created_at=0.0,
                          members=farms.build_members(
                              ["a"], {"a": bot}, task="Find it", mode="parallel",
                              stagger_seconds=0, start_at=None, now=0.0))
        store = _store(tmp_path, [farm])
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["json"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "t9", "thread_id": "th9"})

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        class Patched(real_client):
            def __init__(self, **kw):
                kw["transport"] = transport
                super().__init__(**kw)

        monkeypatch.setattr(farms.httpx, "AsyncClient", Patched)

        runner = _make_runner(store, now=lambda: 10.0)
        asyncio.run(runner.tick())

        assert seen["url"] == "http://profile-a:8000/api/tasks"
        assert seen["auth"] == "Bearer tok"
        expected = "You are Ada. bg words\n\nTASK: Find it"
        assert seen["json"]["recipe"] == "plan.task"
        # The instruction rides INSIDE payload: TaskRequest drops flat fields,
        # so the farm would otherwise fire with an empty ask.
        payload = seen["json"]["payload"]
        assert payload["task"] == expected
        assert payload["text"] == expected
        assert payload["goal"] == expected
        assert payload["url"] == "https://x"
        assert farm.members[0].status == "started"
        assert farm.members[0].task_id == "t9"
        assert farm.members[0].thread_id == "th9"
        assert farm.status == "running"

    def test_transport_error_marks_member_unreachable_only(self, tmp_path):
        farm = _farm(["a", "b"])
        store = _store(tmp_path, [farm])
        attempted = []

        async def post(profile, f):
            attempted.append(profile)
            if profile == "a":
                raise ConnectionError("no route")
            return {"id": "t2"}

        runner = _make_runner(store, now=lambda: 10.0, post=post)
        asyncio.run(runner.tick())
        a, b = farm.members
        assert a.status == "unreachable"
        assert "ConnectionError" in a.outcome
        assert b.status == "started"
        assert farm.status == "running"

    def test_response_without_task_id_is_unreachable(self, tmp_path):
        farm = _farm(["a"])
        store = _store(tmp_path, [farm])

        async def post(profile, f):
            return {"unexpected": "shape"}

        runner = _make_runner(store, now=lambda: 10.0, post=post)
        asyncio.run(runner.tick())
        m = farm.members[0]
        assert m.status == "unreachable"
        assert "no task id" in m.outcome

    def test_staggered_members_fire_only_when_due(self, tmp_path):
        farm = _farm(["a", "b"], mode="stagger", stagger=60, now=0.0)
        store = _store(tmp_path, [farm])
        fired = []
        clock = {"t": 30.0}

        async def post(profile, f):
            fired.append(profile)
            return {"id": f"t-{profile}"}

        runner = _make_runner(store, now=lambda: clock["t"], post=post)
        asyncio.run(runner.tick())
        assert fired == ["a"]
        assert {m.profile: m.status for m in farm.members} == {
            "a": "started", "b": "pending",
        }

        clock["t"] = 61.0
        asyncio.run(runner.tick())
        assert fired == ["a", "b"]


class TestRefreshAndSettle:
    def _started(self, tmp_path):
        farm = _farm(["a"])
        farm.status = "running"
        farm.members[0].status = "started"
        farm.members[0].task_id = "t1"
        farm.members[0].start_at = 0.0
        return _store(tmp_path, [farm]), farm

    def test_done_with_result_sets_outcome_and_completes_farm(self, tmp_path):
        store, farm = self._started(tmp_path)

        async def get_state(profile):
            return {"tasks": [{"id": "t1", "status": "done",
                               "result": "Booked flight AY73."}]}

        runner = _make_runner(store, now=lambda: 100.0, get_state=get_state)
        asyncio.run(runner.tick())
        m = farm.members[0]
        assert m.status == "done"
        assert m.outcome == "Booked flight AY73."
        assert m.finished_at == 100.0
        assert farm.status == "complete"

    def test_blocked_uses_detail(self, tmp_path):
        store, farm = self._started(tmp_path)

        async def get_state(profile):
            return {"tasks": [{"id": "t1", "status": "blocked",
                               "detail": "captcha wall"}]}

        runner = _make_runner(store, now=lambda: 100.0, get_state=get_state)
        asyncio.run(runner.tick())
        assert farm.members[0].outcome == "captcha wall"
        assert farm.status == "complete"

    def test_outcome_is_capped(self, tmp_path):
        store, farm = self._started(tmp_path)
        long = "x" * 5000

        async def get_state(profile):
            return {"tasks": [{"id": "t1", "status": "done", "result": long}]}

        runner = _make_runner(store, now=lambda: 100.0, get_state=get_state)
        asyncio.run(runner.tick())
        assert len(farm.members[0].outcome) <= farms.OUTCOME_CHARS

    def test_queued_maps_to_started_running_to_running(self, tmp_path):
        store, farm = self._started(tmp_path)
        states = iter([
            {"tasks": [{"id": "t1", "status": "queued"}]},
            {"tasks": [{"id": "t1", "status": "running"}]},
        ])

        async def get_state(profile):
            return next(states)

        for expected in ("started", "running"):
            runner = _make_runner(store, now=lambda: 100.0, get_state=get_state)
            asyncio.run(runner.tick())
            assert farm.members[0].status == expected
        assert farm.status == "running"

    def test_transport_blip_leaves_status_alone(self, tmp_path):
        store, farm = self._started(tmp_path)

        async def get_state(profile):
            raise TimeoutError("slow")

        runner = _make_runner(store, now=lambda: 100.0, get_state=get_state)
        asyncio.run(runner.tick())
        assert farm.members[0].status == "started"
        assert farm.status == "running"

    def test_missing_task_row_is_unreachable(self, tmp_path):
        store, farm = self._started(tmp_path)

        async def get_state(profile):
            return {"tasks": []}

        runner = _make_runner(store, now=lambda: 100.0, get_state=get_state)
        asyncio.run(runner.tick())
        assert farm.members[0].status == "unreachable"
        assert farm.status == "complete"

    def test_refresh_throttled_per_farm(self, tmp_path):
        store, farm = self._started(tmp_path)
        calls = []
        clock = {"t": 3.0}

        async def get_state(profile):
            calls.append(profile)
            return {"tasks": [{"id": "t1", "status": "running"}]}

        runner = _make_runner(store, now=lambda: clock["t"], get_state=get_state)
        asyncio.run(runner.tick())   # 3 < 5: throttled
        asyncio.run(runner.tick())   # still the same instant
        assert calls == []
        clock["t"] = 5.0
        asyncio.run(runner.tick())
        assert calls == ["a"]

    def test_persist_on_change(self, tmp_path):
        path = tmp_path / "farms.json"
        farm = _farm(["a"], mode="schedule", start_at=50.0, now=0.0)
        store = farms.FarmStore(path)
        store.farms = [farm]

        async def post(profile, f):
            return {"id": "t1"}

        async def get_state(profile):
            return {"tasks": [{"id": "t1", "status": "done", "result": "ok"}]}

        clock = {"t": 0.0}
        runner = _make_runner(store, now=lambda: clock["t"], post=post,
                              get_state=get_state)
        asyncio.run(runner.tick())           # t=0: nothing due
        assert farm.status == "pending"
        assert path.exists() is False        # nothing changed, nothing written

        clock["t"] = 60.0
        asyncio.run(runner.tick())           # fires
        on_disk = json.loads(path.read_text())
        assert on_disk["farms"][0]["status"] == "running"
        assert on_disk["farms"][0]["members"][0]["status"] == "started"

        clock["t"] = 66.0
        asyncio.run(runner.tick())           # refresh -> done, settle -> complete
        on_disk = json.loads(path.read_text())
        assert on_disk["farms"][0]["status"] == "complete"
        assert on_disk["farms"][0]["members"][0]["outcome"] == "ok"


class TestHttpDefaults:
    def test_bot_url_from_template(self):
        runner = _make_runner(farms.FarmStore(tmp_path_none()), now=lambda: 0)
        assert runner._bot_url("kai") == "http://profile-kai:8000"

    def test_headers_carry_bearer(self):
        runner = _make_runner(farms.FarmStore(tmp_path_none()), now=lambda: 0)
        assert runner._headers() == {"Authorization": "Bearer tok"}

    def test_default_post_signature_matches_the_injected_shape(self):
        """The real HTTP method has the same signature the fakes satisfy — a
        drift would make every runner test pass while production could not
        talk to a pod at all."""
        runner = _make_runner(farms.FarmStore(tmp_path_none()), now=lambda: 0)
        assert inspect.iscoroutinefunction(runner._http_post)
        assert list(inspect.signature(runner._http_post).parameters) == ["profile", "farm"]
        assert list(inspect.signature(runner._http_get_state).parameters) == ["profile"]


def tmp_path_none():
    """A store shell for tests that never touch persistence."""

    return Path("/nonexistent-test-farms/farms.json")


# -- the hub routes ---------------------------------------------------------


@pytest.fixture
def farm_env(tmp_path, monkeypatch):
    """The roster fixture pattern, with the app-level runner silenced.

    Silencing matters: TestClient's lifespan starts the real loop, and a farm
    created in ``parallel`` mode would be fired by it mid-test on a real HTTP
    call to a pod that does not exist. A one-hour tick keeps every member
    exactly where the route left it.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("REGISTRY_PATH", str(tmp_path / "data" / "bots.json"))
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("CONTROL_TOKEN", "test-token")
    monkeypatch.setenv("BOT_PROBE_TIMEOUT_S", "0.1")
    monkeypatch.setenv("RECIPES_DIR", str(tmp_path / "recipes"))
    importlib.reload(config)
    importlib.reload(hub)
    hub.farm_runner._interval = 3600.0
    # Seed two environments, one with a persona.
    reg = registry.Registry(bots=[
        registry.Bot(profile="kai", name="Kai", background="A meticulous planner."),
        registry.Bot(profile="lex"),
    ])
    registry.save(hub.settings.registry_path, reg)
    yield tmp_path


AUTH = {"Authorization": "Bearer test-token"}


def _client():
    return TestClient(hub.app)


def _create(client, **over):
    body = {
        "name": "Tokyo flights",
        "recipe": "plan.task",
        "task": "Find a flight to Tokyo under 6000 SEK.",
        "url": "https://flights.example",
        "profiles": ["kai", "lex"],
        "mode": "parallel",
    }
    body.update(over)
    return client.post("/api/farms", json=body, headers=AUTH)


class TestFarmRoutes:
    def test_create_returns_201_with_composed_members(self, farm_env):
        with _client() as client:
            res = _create(client)
            assert res.status_code == 201, res.text
            farm = res.json()["farm"]
            assert farm["display_name"] == "Tokyo flights"
            assert farm["status"] == "pending"
            by_profile = {m["profile"]: m for m in farm["members"]}
            assert by_profile["kai"]["task_text"] == (
                "You are Kai. A meticulous planner.\n\n"
                "TASK: Find a flight to Tokyo under 6000 SEK."
            )
            assert by_profile["lex"]["task_text"].startswith("You are lex.")

    def test_create_validation(self, farm_env):
        with _client() as client:
            assert _create(client, profiles=[]).status_code == 400
            assert _create(client, profiles=["ghost"]).status_code == 400
            assert _create(client, task="  ").status_code == 400
            assert _create(client, recipe=" ").status_code == 400
            assert _create(client, mode="eventually").status_code == 400
            assert _create(client, mode="stagger",
                           stagger_seconds=0).status_code == 400

    def test_create_rejects_an_unknown_recipe(self, farm_env):
        with _client() as client:
            res = _create(client, recipe="no.such.recipe")
            assert res.status_code == 400
            assert "unknown recipe" in res.text

    def test_create_rejects_entry_url_less_recipe_without_a_url(self, farm_env):
        # agent.task has an empty entry_url on purpose: a freeform instruction
        # is meaningless without a page, so the assignment says so up front
        # rather than failing the same way on every member.
        with _client() as client:
            assert _create(client, recipe="agent.task", url="").status_code == 400
            ok = _create(client, recipe="agent.task",
                         url="https://example.com").status_code
            assert ok == 201, ok

    def test_list_newest_first(self, farm_env):
        with _client() as client:
            first = _create(client, name="first").json()["farm"]
            time.sleep(0.01)
            second = _create(client, name="second").json()["farm"]
            listing = client.get("/api/farms", headers=AUTH).json()["farms"]
            assert [f["id"] for f in listing] == [second["id"], first["id"]]

    def test_get_one_and_404(self, farm_env):
        with _client() as client:
            farm = _create(client).json()["farm"]
            got = client.get(f"/api/farms/{farm['id']}", headers=AUTH)
            assert got.status_code == 200
            assert got.json()["farm"]["id"] == farm["id"]
            assert client.get("/api/farms/nope", headers=AUTH).status_code == 404

    def test_cancel_pending_members_only(self, farm_env):
        with _client() as client:
            farm = _create(client, mode="schedule",
                           start_at=time.time() + 10_000).json()["farm"]
            # Simulate one member already started before the cancel.
            hub.farm_store.get(farm["id"]).members[0].status = "started"
            res = client.post(f"/api/farms/{farm['id']}/cancel", headers=AUTH)
            assert res.status_code == 200
            members = res.json()["farm"]["members"]
            assert members[0]["status"] == "started"
            assert members[1]["status"] == "cancelled"
            assert res.json()["farm"]["status"] == "cancelled"

    def test_delete_and_404(self, farm_env):
        with _client() as client:
            farm = _create(client).json()["farm"]
            assert client.delete(f"/api/farms/{farm['id']}",
                                 headers=AUTH).status_code == 200
            assert client.delete(f"/api/farms/{farm['id']}",
                                 headers=AUTH).status_code == 404

    def test_farms_persist_to_the_hub_volume(self, farm_env):
        with _client() as client:
            farm = _create(client).json()["farm"]
        path = hub.settings.registry_path.parent / "farms.json"
        on_disk = json.loads(path.read_text())
        assert [f["id"] for f in on_disk["farms"]] == [farm["id"]]

    def test_routes_require_token(self, farm_env):
        with _client() as client:
            assert client.get("/api/farms").status_code == 401
            assert client.post("/api/farms", json={}).status_code == 401


def test_fire_body_carries_the_instruction_inside_payload():
    """The wire shape the pod's TaskRequest actually takes: {recipe, payload}.

    Flat top-level instruction fields are silently dropped (TaskRequest keeps
    only recipe + payload), which ran a farm with an EMPTY instruction — found
    live when the pod's attempt carried payload: {} and the thread had no ask.
    """
    farm = farms.Farm(id="f", name="", recipe="agent.task", task="T",
                      url="https://example.com", mode="parallel",
                      stagger_seconds=0, created_at=1.0,
                      members=[farms.Member(profile="a", task_text="You are A.\n\nTASK: T")])
    body = farms.fire_body(farm, "You are A.\n\nTASK: T")
    assert body["recipe"] == "agent.task"
    payload = body["payload"]
    assert payload["url"] == "https://example.com"
    # goal wins inside the agent — all three aliases must carry the instruction
    assert payload["task"] == payload["text"] == payload["goal"] == "You are A.\n\nTASK: T"
