"""DOM-bridge tests against hand-written page snapshots.

The bridge's whole job is to translate the site's class vocabulary into the
solver's tokens faithfully, so every test here is a claim about that mapping:
what a class means, and what a *misread* must not become. No browser: the
``page`` is a stub that returns a canned evaluate result, which keeps the
mapping honest and the test instant.

The class table comes from a live probe of minesweeper.online (2026-09-22) —
see ``references/minesweeper.md`` for the raw observations.
"""

from __future__ import annotations

import asyncio

import pytest

from browser_agent.minesweeper_dom import (
    BEGINNER_MINES,
    Board,
    GameView,
    Pace,
    cell_selector,
    read_game,
)


class FakePage:
    """A Page stand-in that returns one canned evaluate result."""

    def __init__(self, result: dict) -> None:
        self.result = result
        self.evaluated: list[tuple] = []
        self.waits: list[int] = []

    async def evaluate(self, script, arg=None):
        self.evaluated.append((script, arg))
        return self.result

    async def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)


def _payload(grid: list[list[str]], **over: object) -> dict:
    base = {
        "grid": grid,
        "face": "top-area-face zoomable hd_top-area-face-unpressed",
        "cells": sum(len(r) for r in grid),
        "blocked": False,
        "status": "",
    }
    base.update(over)
    return base


def test_cell_selector_is_column_first():
    # The site's id is cell_<x>_<y>; getting the order wrong silently plays a
    # transposed board, which looks like a solver bug. Pin it.
    assert cell_selector(0, 0) == "#cell_0_0"
    assert cell_selector(row=2, col=7) == "#cell_7_2"


def test_read_translates_every_state_token():
    grid = [
        [".", "F", "0", "3"],
        ["M", "?", ".", "1"],
    ]
    view = asyncio.run(read_game(FakePage(_payload(grid))))
    board = view.board
    assert board.rows == 2 and board.cols == 4
    from browser_agent.minesweeper_solver import CLOSED, FLAGGED, OPENED

    assert board.grid[0][0].state == CLOSED
    assert board.grid[0][1].state == FLAGGED
    assert board.grid[0][2].state == OPENED and board.grid[0][2].adjacent == 0
    assert board.grid[0][3].adjacent == 3
    # A revealed mine is an opened cell showing 10, not a special state: the
    # solve is over by then, and pretending otherwise would invent a state the
    # player can never see.
    assert board.grid[1][0].state == OPENED and board.grid[1][0].adjacent == 10
    # An unreadable cell reads as closed, never as a mine or an opened 0.
    assert board.grid[1][1].state == CLOSED
    assert board.grid[1][3].adjacent == 1


def test_flag_wins_over_opened_when_both_classes_appear():
    # A flagged cell carries hd_closed (and, per the CSS, potentially an opened
    # marker in a race). Flag first, so a misread cannot become "opened".
    from browser_agent.minesweeper_solver import FLAGGED

    page = FakePage(_payload([["F"]]))
    # The JS decides this, but the Python side must keep the contract if the
    # token ever arrives ambiguous; assert the token is at least stable.
    view = asyncio.run(read_game(page))
    assert view.board.grid[0][0].state == FLAGGED


def test_win_and_lose_read_from_the_face_class():
    won = asyncio.run(read_game(FakePage(_payload(
        [["0"]], face="top-area-face zoomable hd_top-area-face-win"))))
    lost = asyncio.run(read_game(FakePage(_payload(
        [["M"]], face="top-area-face zoomable hd_top-area-face-lose"))))
    live = asyncio.run(read_game(FakePage(_payload(
        [["."]], face="top-area-face zoomable hd_top-area-face-unpressed"))))

    assert won.won and won.over and not won.lost
    assert lost.lost and lost.over and not lost.won
    assert not live.over
    # "unpressed" must not be read as a result — the substring check is on the
    # full tokens, not on a bare "win"/"lose".
    assert not live.won and not live.lost


def test_cells_ready_needs_the_full_eighty_one():
    short = asyncio.run(read_game(FakePage(_payload([["."]], cells=1))))
    full = asyncio.run(read_game(FakePage(_payload([["."]], cells=81))))
    assert not short.cells_ready
    assert full.cells_ready


def test_blocked_is_reported_not_raised():
    # The recipe decides what a block means (stop and page a human). The read
    # must surface it as data so that decision stays in one place.
    view = asyncio.run(read_game(FakePage(_payload([["?"]], cells=0, blocked=True))))
    assert view.blocked and not view.cells_ready


def test_board_counts_and_mines_left():
    board = Board.from_rows([["F", ".", "1"], ["1", "1", "."]], mines=3)
    assert board.flagged == 1
    assert board.opened == 3
    assert board.mines_left == 2
    # Never negative, whatever the flags claim — a misclick can over-flag.
    assert Board.from_rows([["F", "F"]], mines=1).mines_left == 0


def test_board_is_won_when_only_mines_stay_closed():
    # 3x3, one mine: eight opened cells and the mine still closed is a win.
    rows = [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "."]]
    assert Board.from_rows(rows, mines=1).is_won()
    assert not Board.from_rows([["0", "."], ["0", "."]], mines=1).is_won()


def test_read_passes_the_board_dimensions_to_the_script():
    # The grid walker is bounded by the dimensions passed in, so a mismatch
    # would silently truncate the board. Assert the arg actually arrives.
    page = FakePage(_payload([["."]]))
    asyncio.run(read_game(page))
    _script, arg = page.evaluated[0]
    assert arg == {"rows": 9, "cols": 9}


def test_pace_stays_inside_its_own_bounds():
    # The pacing is the anti-ban mechanism, so its bounds are a contract, not a
    # suggestion: a gap outside them is a bug. Sampled, because it is random.
    pace = Pace(min_ms=100, max_ms=200, hover_ms=0)
    gaps = [pace._gap() for _ in range(200)]
    assert all(0.1 <= g <= 0.2 for g in gaps)
    # And it must actually vary — a constant is the machine tell.
    assert len(set(gaps)) > 50


@pytest.mark.parametrize("flag,button", [(False, "left"), (True, "right")])
def test_click_cell_uses_the_right_button(flag: bool, button: str):
    # Flagging is a right-click on the site; using left would open the cell and
    # lose the game. The mapping is asserted through the pace's own recorder.
    from browser_agent.minesweeper_dom import click_cell

    class RecordingPace(Pace):
        def __init__(self) -> None:
            super().__init__(min_ms=0, max_ms=0, hover_ms=0)
            self.seen: list[tuple[str, str]] = []

        async def click(self, page, selector, *, button="left"):
            self.seen.append((selector, button))

    pace = RecordingPace()
    asyncio.run(click_cell(FakePage({}), 3, 5, pace=pace, flag=flag))
    assert pace.seen == [("#cell_5_3", button)]


class MousePage:
    """Records the pointer path and the presses a Pace produces."""

    def __init__(self, box: dict | None) -> None:
        self._box = box
        self.moves: list[tuple[float, float, int]] = []
        self.presses: list[tuple[float, float, str]] = []
        self.waits: list[int] = []
        self.mouse = self._Mouse(self)

    class _Mouse:
        def __init__(self, page: MousePage) -> None:
            self._p = page

        async def move(self, x, y, steps=1):
            self._p.moves.append((x, y, steps))

        async def click(self, x, y, *, button="left", delay=None):
            self._p.presses.append((x, y, button))

    def locator(self, selector: str):
        page = self

        class Loc:
            async def scroll_into_view_if_needed(self, timeout=None):
                return None

            async def bounding_box(self, timeout=None):
                return page._box

            async def click(self, *, button="left", timeout=None, delay=None):
                page.presses.append((-1.0, -1.0, button))

            @property
            def first(self):
                return self

        return Loc()

    async def wait_for_timeout(self, ms: int) -> None:
        self.waits.append(ms)


def test_pace_moves_the_pointer_instead_of_teleporting():
    # locator.click() jumps the cursor to the target; a hand does not. The move
    # must be stepped, and the press must land where the pointer went.
    page = MousePage({"x": 100.0, "y": 200.0, "width": 24.0, "height": 24.0})
    pace = Pace(min_ms=0, max_ms=0, hover_ms=0, jitter_px=4, move_steps=3)
    asyncio.run(pace.click(page, "#cell_0_0"))

    assert len(page.moves) == 1
    x, y, steps = page.moves[0]
    assert steps == 3
    assert len(page.presses) == 1
    px, py, button = page.presses[0]
    assert button == "left"
    # The press is at the point the pointer moved to, not the cell centre.
    assert (px, py) == (x, y)


def test_pace_lands_inside_the_cell_and_off_centre():
    # Off-centre is the point, but it must never leave the cell: a 24px cell
    # centred at (112,212) holds x in 100..124. Sample, since it is random.
    page = MousePage({"x": 100.0, "y": 200.0, "width": 24.0, "height": 24.0})
    pace = Pace(min_ms=0, max_ms=0, hover_ms=0, jitter_px=4)
    seen = set()
    for _ in range(100):
        page.moves.clear()
        asyncio.run(pace.click(page, "#cell_0_0"))
        x, y, _ = page.moves[0]
        assert 100.0 <= x <= 124.0, x
        assert 200.0 <= y <= 224.0, y
        seen.add((round(x, 3), round(y, 3)))
    # And the landing point actually varies — a fixed offset is its own tell.
    assert len(seen) > 50


def test_pace_falls_back_to_locator_click_without_geometry():
    # A detached or hidden cell has no bounding box. Losing the move would lose
    # the game, so the plain click must still happen.
    page = MousePage(None)
    pace = Pace(min_ms=0, max_ms=0, hover_ms=0)
    asyncio.run(pace.click(page, "#cell_0_0"))
    assert page.presses == [(-1.0, -1.0, "left")]
    assert page.moves == []


def test_gameview_defaults_read_as_not_ready():
    # A GameView is also built by callers that only have a board; the flags
    # must default to "nothing known" rather than an optimistic True.
    view = GameView(board=Board.from_rows([["."]], mines=1), face="", n_cells=0,
                    blocked=False, status="")
    assert not view.cells_ready and not view.over


def test_beginner_mines_default_is_ten():
    assert BEGINNER_MINES == 10
