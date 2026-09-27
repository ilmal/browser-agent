"""Obstacle memory: every block is written down, and the next run routes around it.

The operator's rule (2026-09-26): a block must never be re-walked. These pin
the store (what is remembered, what is refused, what survives a discovery),
the recipe's consumption of the memory (fast escalation, memory-aware search,
rejection recording, operator override), and the generic recording hook in
``Runner._block``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from browser_agent import site_health
from browser_agent.escalation import Challenge, ChallengeKind, EscalationRequired
from browser_agent.recipes.minesweeper import _SEARCH_URL, Minesweeper
from browser_agent.tasks import TaskRunner as Runner
from browser_agent.tasks import TaskStatus

# ---- local stand-ins (same shapes as test_minesweeper_recipe) --------------


class StubSettings:
    laya_min_confidence = 0.75
    laya_game_min_confidence = 0.55

    def __init__(self, data_root: Any = None) -> None:
        self.data_root = str(data_root) if data_root else ""


class _NoLaya:
    pick_enabled = False


def _recipe(settings: StubSettings) -> Minesweeper:
    return Minesweeper(laya=_NoLaya(), settings=settings)


class _HealPage:
    def __init__(self, links: list[dict], cells_by_url: dict[str, int]) -> None:
        self.links = links
        self.cells_by_url = cells_by_url
        self.visited: list[str] = []
        self._url = ""

    async def goto(self, url, wait_until=None, timeout=None):
        self._url = url
        self.visited.append(url)

    async def wait_for_timeout(self, ms: int) -> None:
        """The board poll between probes (the real Page has it); cells here are
        static so the first read decides."""

    @property
    def url(self) -> str:
        return self._url

    async def evaluate(self, script, arg=None):
        from browser_agent.recipes.minesweeper import _BOARD_PROBE, _COLLECT_LINKS

        if script == _COLLECT_LINKS:
            return self.links
        if script == _BOARD_PROBE:
            return self.cells_by_url.get(self._url, 0)
        raise AssertionError(f"unexpected script: {script[:60]}")


# ---- the store -------------------------------------------------------------


def test_record_blocked_upserts_and_keeps_first_sighting(tmp_path):
    settings = StubSettings(data_root=tmp_path)
    first = site_health.record_blocked(
        settings, "minesweeper.play", host="minesweeper.online",
        reason="Account blocked", url="https://minesweeper.online/new-game")
    again = site_health.record_blocked(
        settings, "minesweeper.play", host="www.minesweeper.online",
        reason="", url="")

    assert first["count"] == 1
    assert again["count"] == 2, "a second sighting must count"
    assert again["first_seen"] == first["first_seen"]
    assert again["last_seen"] >= first["last_seen"]
    assert again["reason"] == "Account blocked", \
        "a blank reason keeps the one that said something"
    assert again["url"] == first["url"]
    stored = site_health.blocked_hosts(settings, "minesweeper.play")
    assert list(stored) == ["minesweeper.online"], "www is folded away"


def test_a_later_timeout_never_demotes_a_recorded_refusal(tmp_path):
    """A refusal, once recorded, is not overwritten by a bare transport failure.

    Live 2026-09-27: a host served its block page (kind "blocked"), then timed
    out on the next run. ``record_blocked`` wrote the new sighting's kind
    verbatim, so "blocked" became "unreachable" and the host fell straight back
    out of ``written_off`` — the recipe walked into the wall it had already
    learned to avoid. The reverse direction is wanted: a timeout later proven to
    be a real refusal does narrow.
    """
    settings = StubSettings(data_root=tmp_path)
    site_health.record_blocked(settings, "minesweeper.play",
                               host="minesweeper.online", reason="Account blocked",
                               kind="blocked")
    after = site_health.record_blocked(
        settings, "minesweeper.play", host="minesweeper.online",
        reason="Page.goto: Timeout 30000ms exceeded", kind="unreachable")

    assert after["kind"] == "blocked", "a timeout must not demote a refusal"
    assert after["count"] == 2, "the sighting is still counted"
    assert after["last_seen"] >= after["first_seen"]
    assert "minesweeper.online" in site_health.written_off(settings, "minesweeper.play")


def test_a_later_refusal_upgrades_a_recorded_timeout(tmp_path):
    """The sticky kind only protects a refusal; it never blocks promotion."""
    settings = StubSettings(data_root=tmp_path)
    site_health.record_blocked(settings, "minesweeper.play",
                               host="minesweeper.online", reason="timeout",
                               kind="unreachable")
    after = site_health.record_blocked(
        settings, "minesweeper.play", host="minesweeper.online",
        reason="Account blocked", kind="blocked")

    assert after["kind"] == "blocked"
    assert "minesweeper.online" in site_health.written_off(settings, "minesweeper.play")


def test_loopback_and_hostless_records_are_refused(tmp_path):
    settings = StubSettings(data_root=tmp_path)
    for host in ("localhost", "127.0.0.1", "", "0.0.0.0"):
        out = site_health.record_blocked(
            settings, "minesweeper.play", host=host, reason="x")
        assert out is None, host
    assert site_health.blocked_hosts(settings, "minesweeper.play") == {}


def test_blocked_hosts_is_empty_when_nothing_is_written(tmp_path):
    settings = StubSettings(data_root=tmp_path)
    assert site_health.blocked_hosts(settings, "minesweeper.play") == {}
    assert site_health.blocked_hosts(settings, "never-heard-of-it") == {}


def test_a_new_discovery_does_not_unblock_the_written_off_hosts(tmp_path):
    settings = StubSettings(data_root=tmp_path)
    site_health.record_blocked(settings, "minesweeper.play",
                               host="minesweeper.online", reason="Account blocked")
    site_health.record(settings, "minesweeper.play",
                       entry_url="https://mines.example/play", note="discovered")

    assert "minesweeper.online" in site_health.blocked_hosts(
        settings, "minesweeper.play")
    assert site_health.entry_url(settings, "minesweeper.play") \
        == "https://mines.example/play"


def test_host_of_strips_www_and_survives_garbage():
    assert site_health.host_of("https://www.Minesweeper.Online/new-game") \
        == "minesweeper.online"
    assert site_health.host_of("about:blank") == ""
    assert site_health.host_of("") == ""


# ---- the recipe consumes the memory ----------------------------------------


def _seed(settings, hosts: dict[str, str]) -> None:
    for host, kind in hosts.items():
        site_health.record_blocked(settings, "minesweeper.play", host=host,
                                   reason="test obstacle", kind=kind)


def test_remembered_block_fast_path_escalates_without_replaying_the_doomed_start(
        monkeypatch, tmp_path):
    """The whole point: the second identical ask stops in seconds, with the
    history in hand — no doomed start_beginner, no re-probing written-off
    candidates."""
    settings = StubSettings(data_root=tmp_path)
    _seed(settings, {"minesweeper.online": "blocked", "mines.example": "rejected"})
    recipe = _recipe(settings=settings)
    page = _HealPage(
        links=[{"href": "https://www.minesweeper.online/", "text": "Mines"}],
        cells_by_url={},
    )
    state = {"starts": 0}

    async def _start(_page, url=None):
        state["starts"] += 1
        raise AssertionError("a remembered block must never be walked into")

    mod = __import__("browser_agent.recipes.minesweeper", fromlist=["x"])
    monkeypatch.setattr(mod, "start_beginner", _start)
    monkeypatch.setattr(mod, "cfg", lambda *a, **k: "")

    try:
        asyncio.run(recipe.run(_HealSession(page), {"task": "play minesweeper"}))
    except EscalationRequired as exc:
        msg = exc.challenge.detail
    else:
        raise AssertionError("an exhausted memory must escalate")

    assert state["starts"] == 0
    assert page.visited == [_SEARCH_URL], \
        "the memory path searches but never probes a written-off host"
    assert "every venue is written off" in msg
    assert "minesweeper.online (seen 1x)" in msg
    assert "1 searched replacement(s) probed and rejected" in msg
    assert site_health.blocked_hosts(settings, "minesweeper.play")[
        "mines.example"]["kind"] == "rejected"


def test_the_memory_path_plays_when_a_genuinely_new_venue_qualifies(
        monkeypatch, tmp_path):
    settings = StubSettings(data_root=tmp_path)
    _seed(settings, {"minesweeper.online": "blocked"})
    recipe = _recipe(settings=settings)
    page = _HealPage(
        links=[
            {"href": "https://minesweeper.online/", "text": "the blocked one"},
            {"href": "https://fresh.example/board", "text": "a new venue"},
        ],
        cells_by_url={"https://fresh.example/board": 81},
    )
    urls: list[str | None] = []
    state = {"starts": 0}

    class _View:
        blocked = False
        cells_ready = True
        n_cells = 81

    async def _start(_page, url=None):
        state["starts"] += 1
        urls.append(url)
        return _View()

    async def _play(*_a):
        return {"outcome": "won"}

    mod = __import__("browser_agent.recipes.minesweeper", fromlist=["x"])
    monkeypatch.setattr(mod, "start_beginner", _start)
    monkeypatch.setattr(recipe, "_play_game", _play)
    monkeypatch.setattr(mod, "cfg", lambda *a, **k: "")

    out = asyncio.run(recipe.run(_HealSession(page), {"task": "play minesweeper"}))

    assert urls == ["https://fresh.example/board"]
    assert out["site"] == "https://fresh.example/board"
    assert site_health.entry_url(settings, "minesweeper.play") \
        == "https://fresh.example/board"
    assert "minesweeper.online" in site_health.blocked_hosts(
        settings, "minesweeper.play"), "the new win does not un-block history"


def test_search_never_returns_a_host_this_bot_wrote_off(tmp_path):
    settings = StubSettings(data_root=tmp_path)
    _seed(settings, {"minesweeper.online": "blocked", "mines.example": "rejected"})
    recipe = _recipe(settings=settings)
    page = _HealPage(
        links=[
            {"href": "https://www.minesweeper.online/new", "text": "blocked"},
            {"href": "https://mines.example/play", "text": "rejected before"},
            {"href": "javascript:void(0)", "text": "junk"},
            {"href": "https://html.duckduckgo.com/l/?uddg="
                     "https%3A%2F%2Fnew.example%2Fboard", "text": "new"},
            {"href": "https://new.example/other", "text": "same host again"},
        ],
        cells_by_url={},
    )

    out = asyncio.run(recipe._search_candidates(page))

    assert out == ["https://new.example/board"]


def test_a_dead_candidate_is_recorded_rejected_so_it_is_never_probed_again(
        monkeypatch, tmp_path):
    settings = StubSettings(data_root=tmp_path)
    recipe = _recipe(settings=settings)
    page = _HealPage(
        links=[
            {"href": "https://dead.example/game", "text": "no board"},
            {"href": "https://live.example/board", "text": "real"},
        ],
        cells_by_url={"https://dead.example/game": 0,
                      "https://live.example/board": 81},
    )

    healed = asyncio.run(recipe._heal(page, _NullLog()))

    assert healed == "https://live.example/board"
    rejected = site_health.blocked_hosts(settings, "minesweeper.play")
    assert rejected["dead.example"]["kind"] == "rejected"
    assert "no 9x9 board contract" in rejected["dead.example"]["reason"]
    assert site_health.entry_url(settings, "minesweeper.play") \
        == "https://live.example/board"


def test_the_operators_override_outranks_the_memory(monkeypatch, tmp_path):
    """A changed egress or lifted ban is the operator's hand: the memory only
    narrows what the bot chooses on its own."""
    from browser_agent.recipes import minesweeper as mod

    settings = StubSettings(data_root=tmp_path)
    _seed(settings, {"minesweeper.online": "blocked"})
    recipe = _recipe(settings=settings)
    page = _HealPage(links=[], cells_by_url={})
    urls: list[str | None] = []

    class _View:
        blocked = False
        cells_ready = True
        n_cells = 81

    async def _start(_page, url=None):
        urls.append(url)
        return _View()

    async def _play(*_a):
        return {"outcome": "won"}

    monkeypatch.setattr(mod, "start_beginner", _start)
    monkeypatch.setattr(recipe, "_play_game", _play)
    monkeypatch.setattr(mod, "cfg",
                        lambda *a, **k: "https://operator.example/new-game")

    out = asyncio.run(recipe.run(_HealSession(page), {"task": "play minesweeper"}))

    assert urls == ["https://operator.example/new-game"], \
        "the override is played directly; no search, no memory consult"
    assert page.visited == []
    assert out["site"] == "https://operator.example/new-game"


class _HealSession:
    activity = None

    def __init__(self, page: Any) -> None:
        from browser_agent.activity import Activity

        self.activity = Activity()
        self._page = page

    async def page(self):
        return self._page


class _NullLog:
    def note(self, *a, **k):
        pass


# ---- the generic hook in Runner._block --------------------------------------


class _StubRunner:
    """Only what _block touches: its settings."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings


class _StubTask:
    id = "t-test"
    recipe = "minesweeper.play"
    status = None
    detail = None


class _StubSettingsForBlock:
    data_root = ""
    notify_on_escalation = False
    profile = "test"
    novnc_port = 6900

    def __init__(self, data_root: Any) -> None:
        self.data_root = str(data_root)


def test_a_rate_limited_block_is_written_down_for_the_recipe(tmp_path):
    settings = _StubSettingsForBlock(tmp_path)
    task = _StubTask()
    challenge = Challenge(ChallengeKind.RATE_LIMITED, "Account blocked",
                          "https://www.minesweeper.online/new-game")

    asyncio.run(Runner._block(_StubRunner(settings), task, challenge))

    assert task.status == TaskStatus.BLOCKED
    stored = site_health.blocked_hosts(settings, "minesweeper.play")
    assert stored["minesweeper.online"]["reason"] == "Account blocked"
    assert stored["minesweeper.online"]["kind"] == "blocked"


def test_captcha_and_hostless_escalations_are_not_memorized(tmp_path):
    settings = _StubSettingsForBlock(tmp_path)

    asyncio.run(Runner._block(
        _StubRunner(settings), _StubTask(),
        Challenge(ChallengeKind.CAPTCHA, "solve me",
                  "https://challenges.example/x")))
    asyncio.run(Runner._block(
        _StubRunner(settings), _StubTask(),
        Challenge(ChallengeKind.RATE_LIMITED, "blank", "about:blank")))

    assert site_health.blocked_hosts(settings, "minesweeper.play") == {}


# ---- the entry-unreachable seam (runner-level goto died) -------------------


class _HookSession:
    """Session whose goto fails for the dead canonical, succeeds elsewhere."""

    def __init__(self, dead_url: str, cells_by_url: dict[str, int]) -> None:
        self.dead_url = dead_url
        self.cells_by_url = cells_by_url
        self.gotos: list[str] = []
        self.page = _HealPage(links=[], cells_by_url=cells_by_url)

    async def goto(self, url, wait_until=None, timeout=None):
        self.gotos.append(url)
        if url == self.dead_url:
            raise TimeoutError("Page.goto: Timeout 30000ms exceeded")
        return f"PAGE::{url}"


def test_entry_unreachable_records_the_obstacle_and_fails_legibly(tmp_path):
    """A recipe without the recovery hook: obstacle written down, fail as
    before, exactly one goto (no doubled timeout)."""
    settings = StubSettings(data_root=tmp_path)
    task = _StubTask()
    recipe = _PlainRecipe("https://dead.example/new-game")
    session = _HookSession("https://dead.example/new-game", {})
    stub = _StubRunner(settings)
    stub.session = session

    out = asyncio.run(Runner._entry_unreachable(
        stub, task, recipe, TimeoutError("Timeout 30000ms exceeded")))

    assert out is None
    assert task.status == TaskStatus.FAILED
    assert "entry page unreachable" in task.detail
    assert session.gotos == [], "no hook → no extra navigation, no doubled timeout"
    stored = site_health.blocked_hosts(settings, task.recipe)
    assert stored["dead.example"]["kind"] == "unreachable"
    assert "Timeout 30000ms" in stored["dead.example"]["reason"]


def test_entry_unreachable_lets_the_recipe_heal_and_retries_the_new_entry(
        tmp_path):
    """With the hook, a found replacement is recorded and the runner navigates
    there instead of the dead hop."""
    settings = StubSettings(data_root=tmp_path)
    task = _StubTask()
    recipe = _HealingRecipe("https://dead.example/new-game",
                            "https://alive.example/board", settings)
    session = _HookSession("https://dead.example/new-game", {})
    session.page.links = [{"href": "https://alive.example/board", "text": "x"}]
    session.page.cells_by_url = {"https://alive.example/board": 81}
    stub = _StubRunner(settings)
    stub.session = session

    out = asyncio.run(Runner._entry_unreachable(
        stub, task, recipe, TimeoutError("Timeout 30000ms exceeded")))

    assert out == "PAGE::https://alive.example/board"
    assert task.status is None, "the task went on to run, not failed"
    assert session.gotos == ["https://alive.example/board"], \
        "the retry navigates the recorded replacement (the dead goto already happened in _run_task)"
    assert site_health.entry_url(settings, task.recipe) \
        == "https://alive.example/board"


def test_a_failed_recovery_still_fails_the_task_without_crashing(tmp_path):
    settings = StubSettings(data_root=tmp_path)
    task = _StubTask()
    recipe = _BoomRecipe("https://dead.example/new-game")
    session = _HookSession("https://dead.example/new-game", {})
    stub = _StubRunner(settings)
    stub.session = session

    out = asyncio.run(Runner._entry_unreachable(
        stub, task, recipe, TimeoutError("Timeout")))

    assert out is None
    assert task.status == TaskStatus.FAILED
    assert session.gotos == ["https://dead.example/new-game"], \
        "one retry after the (failed) hook; the original goto is not the helper's"


class _PlainRecipe:
    def __init__(self, url: str) -> None:
        self.entry_url = url


class _HealingRecipe(_PlainRecipe):
    """entry_url re-reads the store, exactly like Minesweeper's property."""

    def __init__(self, url: str, replacement: str, settings: Any) -> None:
        self._default = url
        self._replacement = replacement
        self._settings = settings

    @property
    def entry_url(self) -> str:
        return site_health.entry_url(self._settings, "minesweeper.play") \
            or self._default

    async def on_entry_unreachable(self, session, exc) -> bool:
        site_health.record(self._settings, "minesweeper.play",
                           entry_url=self._replacement, note="healed")
        return True


class _BoomRecipe(_PlainRecipe):
    async def on_entry_unreachable(self, session, exc) -> bool:
        raise RuntimeError("search blew up")


def test_the_recipe_hook_finds_a_replacement_and_records_it(tmp_path):
    """minesweeper.on_entry_unreachable: heal succeeds → True + recorded;
    nothing new → False + nothing recorded."""
    settings = StubSettings(data_root=tmp_path)
    site_health.record_blocked(settings, "minesweeper.play",
                               host="dead.example", kind="unreachable",
                               reason="timeout")
    recipe = _recipe(settings)
    page = _HealPage(
        links=[{"href": "https://alive.example/board", "text": "real"}],
        cells_by_url={"https://alive.example/board": 81},
    )

    ok = asyncio.run(
        recipe.on_entry_unreachable(_HealSession(page), TimeoutError("t")))
    assert ok is True
    assert site_health.entry_url(settings, "minesweeper.play") \
        == "https://alive.example/board"

    page2 = _HealPage(
        links=[{"href": "https://dead.example/", "text": "remembered"}],
        cells_by_url={},
    )

    assert asyncio.run(
        recipe.on_entry_unreachable(_HealSession(page2), TimeoutError("t"))
    ) is False


def test_a_dead_egress_is_nobodys_obstacle(tmp_path):
    """ERR_PROXY_CONNECTION_FAILED means the tunnel is down: fail legibly,
    record nothing (a venue reachable once the tunnel returns must not be
    slandered in the memory), heal nothing, retry nothing."""
    settings = StubSettings(data_root=tmp_path)
    task = _StubTask()
    recipe = _HealingRecipe("https://dead.example/new-game",
                            "https://alive.example/board", settings)
    session = _HookSession("https://dead.example/new-game", {})
    stub = _StubRunner(settings)
    stub.session = session

    out = asyncio.run(Runner._entry_unreachable(
        stub, task, recipe,
        TimeoutError("Page.goto: net::ERR_PROXY_CONNECTION_FAILED")))

    assert out is None
    assert task.status == TaskStatus.FAILED
    assert "egress unreachable" in task.detail
    assert session.gotos == [], "no retry while the tunnel is down"
    assert site_health.blocked_hosts(settings, "minesweeper.play") == {}
    assert site_health.entry_url(settings, "minesweeper.play") is None, \
        "the hook never ran, so no replacement was recorded"


# ---- the narrowing rule: only a REFUSAL writes a venue off -------------------
#
# The defect this round removes (2026-09-27): a single 30s navigation timeout
# was memorized as kind "unreachable", and the recipe's fast path treated mere
# membership as a permanent write-off — so a healthy canonical (curl: 200 in
# 0.26s) was never tried again, and every later run escalated into a fresh
# "needs you" row. A transient sighting must be recorded and shown, but it must
# not narrow the choice.


def test_written_off_excludes_the_transient_unreachable_kind(tmp_path):
    settings = StubSettings(data_root=tmp_path)
    _seed(settings, {"a.example": "unreachable", "b.example": "blocked",
                     "c.example": "rejected"})

    narrow = site_health.written_off(settings, "minesweeper.play")

    assert set(narrow) == {"b.example", "c.example"}, \
        "only refusals narrow; a bare timeout is advisory"
    # The full store still carries every sighting — nothing is forgotten.
    assert set(site_health.blocked_hosts(settings, "minesweeper.play")) == \
        {"a.example", "b.example", "c.example"}


def test_a_transient_unreachable_record_does_not_divert_the_fast_path(
        monkeypatch, tmp_path):
    """The live bug, pinned: the canonical was recorded unreachable by a
    timeout, then a later run with a healthy canonical must try it — not jump
    straight to _heal."""
    settings = StubSettings(data_root=tmp_path)
    _seed(settings, {"minesweeper.online": "unreachable"})
    recipe = _recipe(settings=settings)
    page = _HealPage(links=[], cells_by_url={})
    seen: list[str | None] = []

    class _View:
        blocked = False
        cells_ready = True
        n_cells = 81

    async def _start(_page, url=None):
        seen.append(url)
        return _View()

    async def _play(*_a):
        return {"outcome": "won"}

    mod = __import__("browser_agent.recipes.minesweeper", fromlist=["x"])
    monkeypatch.setattr(mod, "start_beginner", _start)
    monkeypatch.setattr(recipe, "_play_game", _play)
    monkeypatch.setattr(mod, "cfg", lambda *a, **k: "")

    asyncio.run(recipe.run(_HealSession(page), {"task": "play minesweeper"}))

    assert seen == [recipe.entry_url], "the canonical is tried, not skipped"
    assert page.visited == [], "no search when the canonical is merely advisory"
    assert site_health.blocked_hosts(settings, "minesweeper.play")[
        "minesweeper.online"]["kind"] == "unreachable"


def test_a_genuine_block_still_diverts_the_fast_path(monkeypatch, tmp_path):
    """The mandate holds: a real refusal is routed around, never re-walked."""
    settings = StubSettings(data_root=tmp_path)
    _seed(settings, {"minesweeper.online": "blocked"})
    recipe = _recipe(settings=settings)
    page = _HealPage(
        links=[{"href": "https://fresh.example/board", "text": "new"}],
        cells_by_url={"https://fresh.example/board": 81},
    )
    seen: list[str | None] = []

    class _View:
        blocked = False
        cells_ready = True
        n_cells = 81

    async def _start(_page, url=None):
        seen.append(url)
        return _View()

    async def _play(*_a):
        return {"outcome": "won"}

    mod = __import__("browser_agent.recipes.minesweeper", fromlist=["x"])
    monkeypatch.setattr(mod, "start_beginner", _start)
    monkeypatch.setattr(recipe, "_play_game", _play)
    monkeypatch.setattr(mod, "cfg", lambda *a, **k: "")

    out = asyncio.run(recipe.run(_HealSession(page), {"task": "play minesweeper"}))

    assert seen == ["https://fresh.example/board"], "the block is routed around"
    assert out["site"] == "https://fresh.example/board"


def test_search_re_probes_an_unreachable_host_but_never_a_refused_one(tmp_path):
    """A transient host stays a candidate (the run is searching BECAUSE it did
    not answer); a refused host stays excluded."""
    settings = StubSettings(data_root=tmp_path)
    _seed(settings, {"minesweeper.online": "unreachable",
                     "blocked.example": "blocked"})
    recipe = _recipe(settings=settings)
    page = _HealPage(
        links=[{"href": "https://minesweeper.online/new", "text": "canonical"},
               {"href": "https://blocked.example/play", "text": "refused"},
               {"href": "https://new.example/board", "text": "new"}],
        cells_by_url={},
    )

    out = asyncio.run(recipe._search_candidates(page))

    assert out == ["https://minesweeper.online/new", "https://new.example/board"]


# ---- the escalation must not memorize a non-venue ---------------------------


def test_exhaustion_escalates_against_the_venue_not_the_search_page(tmp_path):
    """_exhausted carried page.url — usually the DuckDuckGo SERP or the last
    probed candidate — and Runner._block then memorized that host as a BLOCKED
    venue. It must name the venue the run was working on."""
    settings = StubSettings(data_root=tmp_path)
    _seed(settings, {"minesweeper.online": "unreachable"})
    recipe = _recipe(settings=settings)

    exc = recipe._exhausted("https://minesweeper.online/new-game")

    assert exc.challenge.url == "https://minesweeper.online/new-game"
    assert "duckduckgo" not in exc.challenge.url


def test_an_exhausted_escalation_is_not_memorized_as_a_venue_refusal(tmp_path):
    """blocked=False: "no venue could be found" is RATE_LIMITED yet refuses
    nothing, so Runner._block must write nothing. Memorizing it promoted the
    healthy canonical to a permanent kind="blocked" write-off."""
    settings = _StubSettingsForBlock(tmp_path)
    task = _StubTask()
    recipe = _recipe(StubSettings(data_root=tmp_path))
    exc = recipe._exhausted("https://minesweeper.online/new-game")

    assert exc.blocked is False

    asyncio.run(Runner._block(_StubRunner(settings), task, exc.challenge,
                              memory_block=exc.blocked))

    assert task.status == TaskStatus.BLOCKED, "the operator is still asked"
    assert site_health.blocked_hosts(settings, "minesweeper.play") == {}, \
        "an absence of venues is not a venue refusal"


def test_a_genuine_refusal_escalation_is_still_memorized(tmp_path):
    """blocked=True (the default) keeps today's behaviour for a real block."""
    settings = _StubSettingsForBlock(tmp_path)
    task = _StubTask()
    exc = EscalationRequired(Challenge(
        ChallengeKind.RATE_LIMITED, "the replacement page is blocked too",
        "https://blocked.example/x"))

    assert exc.blocked is True

    asyncio.run(Runner._block(_StubRunner(settings), task, exc.challenge,
                              memory_block=exc.blocked))

    assert "blocked.example" in site_health.blocked_hosts(
        settings, "minesweeper.play")
