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
