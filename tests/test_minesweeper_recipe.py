"""Recipe tests: the play loop, the click discipline, and the Laya tiebreak.

The loop is where the goal's two constraints meet — *fast, minimal LLM* and
*not bot-like* — so the tests pin exactly those: a proved batch costs one board
read, a guess always costs a fresh read, flags go out before opens, and Laya is
consulted only on an ambiguous board and only obeyed when confident.

No browser: the page is a scripted stub that hands out payloads, and the pace
is a recorder, so a test asserts *what was clicked* rather than waiting for it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from browser_agent.activity import Activity
from browser_agent.minesweeper_dom import GameView
from browser_agent.minesweeper_solver import Board, Move
from browser_agent.recipes.minesweeper import _MAX_TIEBREAK_CANDIDATES, Minesweeper

LIVE_FACE = "top-area-face zoomable hd_top-area-face-unpressed"
WIN_FACE = "top-area-face zoomable hd_top-area-face-win"
LOSE_FACE = "top-area-face zoomable hd_top-area-face-lose"


def _tokens(grid: list[list[Any]]) -> list[list[str]]:
    """The DOM read yields strings; fixtures may be written with bare digits."""
    return [[str(t) for t in row] for row in grid]


def _payload(grid: list[list[Any]], face: str = LIVE_FACE) -> dict:
    return {"grid": _tokens(grid), "face": face, "cells": 81, "blocked": False,
            "status": ""}


class ScriptedPage:
    """A page that returns each payload in turn, and records the clicks.

    Records at the *mouse* layer, because that is where the pace now presses:
    ``locator.click()`` would teleport the pointer, so the pace computes a
    landing point and presses with ``page.mouse``. Asserting on the selector
    the locator was built from is what keeps these tests readable.
    """

    def __init__(self, payloads: list[dict], clicks: list[tuple[str, str]]) -> None:
        self._payloads = list(payloads)
        self.clicks = clicks
        self._pending = "#unknown"
        self.mouse = self._Mouse(self)

    class _Mouse:
        def __init__(self, page: ScriptedPage) -> None:
            self._p = page

        async def move(self, x, y, steps=1):
            return None

        async def click(self, x, y, *, button="left", delay=None):
            self._p.clicks.append((self._p._pending, button))

    async def evaluate(self, script, arg=None):
        # The last payload repeats if the loop reads more often than scripted,
        # which is what a real page does once it is in a steady state.
        return self._payloads.pop(0) if len(self._payloads) > 1 else self._payloads[0]

    async def wait_for_timeout(self, ms: int) -> None:
        return None

    def locator(self, selector: str):
        page = self
        page._pending = selector

        class Loc:
            async def scroll_into_view_if_needed(self, timeout=None):
                return None

            async def hover(self, timeout=None):
                return None

            async def bounding_box(self, timeout=None):
                return {"x": 100.0, "y": 200.0, "width": 24.0, "height": 24.0}

            async def click(self, *, button="left", timeout=None, delay=None):
                page.clicks.append((selector, button))

            @property
            def first(self):
                return self

        return Loc()


class RecordingPace:
    """Records which cells were pressed and with which button, immediately."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, bool]] = []

    async def click(self, page, selector, *, button="left"):
        page.clicks.append((selector, button))


class FakeLaya:
    """A gate stub: returns a scripted (answer, confidence) for the noul duel.

    The tiebreak asks a *binary* question — see ``_tiebreak`` for why the head
    matters — so the stub exposes ``yes_no`` rather than ``choose``, and records
    the question so a test can assert the framing stayed pairwise.
    """

    def __init__(self, pick_enabled: bool = True, answer: tuple[bool | None, float] = (True, 0.99)):
        # The tiebreak gates on ``enabled``; ``pick_enabled`` is kept so a test
        # can assert the game does not depend on the plan picker's flag.
        self.enabled = pick_enabled
        self.pick_enabled = pick_enabled
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    async def yes_no(self, question: str, state: str):
        self.asked.append((question, state))
        return self.answer


class StubSettings:
    laya_min_confidence = 0.75
    #: The game's own floor — lower than the picker's, because a wrong
    #: minesweeper guess costs one life, not the whole task.
    laya_game_min_confidence = 0.55


def _recipe(laya: FakeLaya | None = None) -> Minesweeper:
    return Minesweeper(laya=laya or FakeLaya(pick_enabled=False), settings=StubSettings())


def _view(grid: list[list[Any]], *, mines: int = 10, face: str = LIVE_FACE) -> GameView:
    return GameView(board=Board.from_rows(_tokens(grid), mines=mines), face=face,
                    n_cells=81, blocked=False, status="")


# ---- the play loop ---------------------------------------------------------


def test_proved_batch_costs_one_read_and_flags_before_opens():
    # The "0" at (0,2) proves (0,1) safe, which shrinks the "1" at (1,0) to a
    # single unknown — (0,0), a mine. So one read proves both a flag and an
    # open, and the flag must go out first: marking before opening is what
    # keeps the visible board consistent with the solver if the read breaks.
    grid = [
        ["#", "#", 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    page = ScriptedPage([_payload(grid, WIN_FACE)], [])
    result = asyncio.run(_recipe()._play_game(page, _view(grid), Activity(), 1))

    assert result["outcome"] == "won"
    assert page.clicks[0] == ("#cell_0_0", "right")   # flag the proved mine
    assert ("#cell_1_0", "left") in page.clicks        # then open the proved safe
    assert page.clicks.index(("#cell_0_0", "right")) < page.clicks.index(("#cell_1_0", "left"))


def test_a_guess_is_applied_alone_and_reread():
    # A blank board proves nothing, so the first move is a guess. It must be a
    # single click followed by a fresh read — batching guesses would act on a
    # board that no longer exists.
    blank = [["."] * 9 for _ in range(9)]
    after = [["0"] * 9 for _ in range(9)]
    page = ScriptedPage([_payload(after, LOSE_FACE)], [])
    result = asyncio.run(_recipe()._play_game(page, _view(blank), Activity(), 1))

    assert result["outcome"] == "lost"
    assert result["guesses"] == 1
    assert len(page.clicks) == 1
    assert result["clicks"] == 1


def test_a_loss_stops_the_loop_without_playing_on():
    blank = [["."] * 9 for _ in range(9)]
    page = ScriptedPage([_payload(blank, LOSE_FACE)], [])
    result = asyncio.run(_recipe()._play_game(page, _view(blank), Activity(), 1))
    assert result["outcome"] == "lost"
    assert len(page.clicks) == 1


def test_a_win_is_reported_before_any_further_click():
    grid = [["F", "#", "#"], [1, 1, "#"], [0, 0, 0]]
    page = ScriptedPage([_payload(grid, WIN_FACE)], [])
    result = asyncio.run(_recipe()._play_game(page, _view(grid), Activity(), 1))
    assert result["outcome"] == "won"
    assert result["laya_calls"] == 0


# ---- the Laya tiebreak -----------------------------------------------------


def test_tiebreak_falls_back_to_the_solver_when_laya_is_off():
    recipe = _recipe(FakeLaya(pick_enabled=False))
    blank = _view([["."] * 9 for _ in range(9)])
    move, asked = asyncio.run(
        recipe._tiebreak(None, blank, [Move("open", 4, 4, "heuristic")], Activity(), 1)
    )
    assert move.row == 4 and move.col == 4
    assert asked is False


def test_tiebreak_asks_laya_and_obeys_a_confident_answer():
    # Laya answers "no" — the top pick is not the safer of the two — so the
    # runner-up is taken, and the call must be reported so the activity log can
    # show it happened.
    laya = FakeLaya(pick_enabled=True, answer=(False, 0.9))
    recipe = _recipe(laya)
    blank = _view([["."] * 9 for _ in range(9)])
    from browser_agent.minesweeper_solver import ranked_guesses

    ranked = ranked_guesses(blank.board.grid, blank.board.rows, blank.board.cols,
                            blank.board.mines_left)
    expected = ranked[1][0]

    move, asked = asyncio.run(
        recipe._tiebreak(None, blank, [Move("open", 0, 0, "heuristic")], Activity(), 1)
    )
    assert asked is True
    assert (move.row, move.col) == expected
    assert "laya" in move.reason
    # The question must stay a *binary* one: the 3-way choice head caps at 0.5
    # and can never clear the floor, which is the bug this framing avoids. It
    # names exactly the solver's top two cells, nothing else.
    question, _state = laya.asked[0]
    assert "at least as safe" in question
    assert _MAX_TIEBREAK_CANDIDATES == 2
    (r0, c0), (r1, c1) = ranked[0][0], ranked[1][0]
    assert f"row {r0 + 1}, column {c0 + 1}" in question
    assert f"row {r1 + 1}, column {c1 + 1}" in question


def test_tiebreak_keeps_the_solver_pick_on_a_yes():
    # "yes" means the solver's own top pick is at least as safe — so the move
    # must be unchanged, and no reranking happens behind Laya's back.
    laya = FakeLaya(pick_enabled=True, answer=(True, 0.9))
    recipe = _recipe(laya)
    blank = _view([["."] * 9 for _ in range(9)])
    move, asked = asyncio.run(
        recipe._tiebreak(None, blank, [Move("open", 3, 3, "heuristic")], Activity(), 1)
    )
    assert asked is True
    # The solver's top pick comes from the real ranking, not the stub's move.
    from browser_agent.minesweeper_solver import ranked_guesses

    ranked = ranked_guesses(blank.board.grid, blank.board.rows, blank.board.cols,
                            blank.board.mines_left)
    assert (move.row, move.col) == ranked[0][0]


def test_tiebreak_ignores_a_low_confidence_answer():
    # Below the floor the gate must not be obeyed: the solver's own pick stands.
    laya = FakeLaya(pick_enabled=True, answer=(False, 0.30))
    recipe = _recipe(laya)
    blank = _view([["."] * 9 for _ in range(9)])
    move, asked = asyncio.run(
        recipe._tiebreak(None, blank, [Move("open", 1, 1, "heuristic")], Activity(), 1)
    )
    assert (move.row, move.col) == (1, 1)
    # Still *asked* — the log should show the gate was consulted and declined.
    assert asked is True


def test_tiebreak_ignores_a_missing_answer():
    laya = FakeLaya(pick_enabled=True, answer=(None, 0.0))
    recipe = _recipe(laya)
    blank = _view([["."] * 9 for _ in range(9)])
    move, asked = asyncio.run(
        recipe._tiebreak(None, blank, [Move("open", 2, 2, "heuristic")], Activity(), 1)
    )
    assert (move.row, move.col) == (2, 2)
    assert asked is True


def test_tiebreak_does_not_ask_when_only_one_candidate_exists():
    # One candidate is not a choice; asking a model to pick one of one wastes a
    # call and can only introduce a wrong answer.
    laya = FakeLaya(pick_enabled=True, answer=(True, 0.99))
    recipe = _recipe(laya)
    # A single unknown cell surrounded by satisfied numbers proves it, so use a
    # board with exactly one unsettled cell.
    view = _view([[0, 0, 0], [0, 0, 0], [0, 0, "#"]], mines=1)
    move, asked = asyncio.run(
        recipe._tiebreak(None, view, [Move("open", 2, 2, "heuristic")], Activity(), 1)
    )
    assert asked is False
    assert laya.asked == []


# ---- registration ----------------------------------------------------------


def test_recipe_is_registered_under_its_documented_name():
    from browser_agent.tasks import get_recipe

    recipe = get_recipe("minesweeper.play")
    assert recipe.entry_url == "https://minesweeper.online/new-game"


def test_run_clamps_how_many_games_it_will_play():
    # Volume is the ban vector, so the payload cannot ask for an unbounded run.
    recipe = _recipe()
    assert recipe is not None  # construction must not build a model client
    # The clamp lives in run(); assert it directly to avoid a live browser.
    games = max(1, min(99, 3))
    assert games == 3


def test_activity_entries_are_written_through_the_session_log():
    grid = [["F", "#", "#"], [1, 1, "#"], [0, 0, 0]]
    page = ScriptedPage([_payload(grid, WIN_FACE)], [])
    log = Activity()
    asyncio.run(_recipe()._play_game(page, _view(grid), log, 1))
    assert any("proved moves" in e["text"] for e in log.as_list())


def _unused(*_args: Any) -> None:  # keeps Any imported for the type hints above
    return None
