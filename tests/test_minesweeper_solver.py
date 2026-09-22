"""Solver tests against hand-built boards with known answers.

The boards are small on purpose: each one isolates a single rule, so a failure
points at the rule rather than at the reading of some fixture. Where a board is
ambiguous by construction the test asserts a *property* (never both an open and
a flag on one cell) rather than a specific move, because there is no single
right answer to assert.
"""

from __future__ import annotations

from browser_agent.minesweeper_solver import (
    CLOSED,
    FLAGGED,
    OPENED,
    Cell,
    constraints,
    guess,
    guess_key,
    is_won,
    logic_moves,
    make_board,
    neighbors,
    next_moves,
    ranked_guesses,
    render,
)


def _grid(rows: list[list[int | str]]) -> list[list[Cell]]:
    """Build a board from a sketch: "#" closed, "F" flagged, digits opened."""
    out = []
    for row in rows:
        cells = []
        for token in row:
            if token == "#":
                cells.append(Cell(CLOSED))
            elif token == "F":
                cells.append(Cell(FLAGGED))
            else:
                cells.append(Cell(OPENED, int(token)))
        out.append(cells)
    return out


def _proved(grid: list[list[Cell]], rows: int, cols: int) -> tuple[set, set]:
    """(safe, mines) as the solver proves them, for direct comparison."""
    safe: set = set()
    mines: set = set()
    for move in logic_moves(grid, rows, cols):
        (mines if move.kind == "flag" else safe).add((move.row, move.col))
    return safe, mines


def test_neighbors_clips_at_edges():
    assert len(neighbors(3, 3, 0, 0)) == 3
    assert len(neighbors(3, 3, 1, 1)) == 8
    assert len(neighbors(3, 3, 2, 2)) == 3
    assert set(neighbors(3, 3, 0, 1)) == {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}


def test_single_point_all_neighbours_are_mines():
    # A "1" at the end of a one-row board has exactly one neighbour, so it must
    # be a mine. (A "2" there would need a second neighbour in a cell that does
    # not exist, which is why the boundary is stated with a "1".)
    grid = _grid([[0, 1, "#"]])
    safe, mines = _proved(grid, 1, 3)
    assert mines == {(0, 2)}
    assert safe == set()


def test_single_point_already_satisfied_opens_the_rest():
    # The "1" is satisfied by the flag, so its other closed neighbour is safe.
    grid = _grid([["F", "#", "#"], [1, 1, "#"], [0, 0, 0]])
    safe, mines = _proved(grid, 3, 3)
    assert (0, 1) in safe and (0, 2) in safe
    assert mines == set()


def test_subset_elimination_finds_the_mine_that_single_point_cannot():
    # Classic 1-2 wall. The 1 owns {x0,x1}, the 2 owns {x0,x1,x2}, so the cell
    # the 2 owns alone carries the difference: one mine. Neither cell's own
    # count proves anything, which is exactly why single-point logic stalls here.
    # The "0" beside the 2 also proves one of its neighbours safe, which is the
    # other half of what this shape yields.
    grid = _grid([["#", "#", "#"], [1, 2, "#"], [0, 0, 0]])
    safe, mines = _proved(grid, 3, 3)
    assert mines == {(0, 2)}
    assert safe == {(1, 2)}


def test_chained_subset_elimination_solves_a_121_wall():
    # A 1-2-3-2-1 row over a wall. Proving the middle mine has to shrink the
    # neighbouring constraints for the outer cells to fall, so a solver that runs
    # each phase once instead of to a fixpoint stops after one mine.
    grid = _grid(
        [
            ["#", "#", "#", "#", "#"],
            [1, 2, 3, 2, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    safe, mines = _proved(grid, 3, 5)
    assert mines == {(0, 1), (0, 2), (0, 3)}, render(grid)
    assert safe == {(0, 0), (0, 4)}


def test_constraints_are_clamped_not_negative():
    # Four flags around a "2" is a misread (or a stale flag). The cell must
    # produce no constraint at all rather than a negative mine count.
    grid = _grid([["F", "F", "F"], ["F", 2, 0], [0, 0, 0]])
    assert constraints(grid, 3, 3) == []
    assert logic_moves(grid, 3, 3) == []


def test_contradictory_board_never_yields_two_moves_for_one_cell():
    # A "0" and a "1" over the same two closed cells is impossible input: one
    # proves both safe, the other needs one to be a mine. Whatever the solver
    # concludes, it must not tell the caller to both open and flag the same cell.
    grid = _grid([["#", "#"], [1, 0]])
    safe, mines = _proved(grid, 2, 2)
    assert not (safe & mines)

    proven_safe = _grid([[1, "#"], ["#", "#"], [0, 0]])
    safe, mines = _proved(proven_safe, 3, 2)
    assert not (safe & mines)


def test_guess_returns_a_closed_cell_when_logic_is_exhausted():
    # One mine among four unknown cells, all equally likely. The guess must pick
    # one of the genuinely unknown cells, never something already opened.
    grid = _grid([["#", "#", "#", "#"], [1, 1, 1, "#"], [0, 0, 0, 0]])
    move = guess(grid, 3, 4, mines_left=3)
    assert move is not None and move.kind == "open"
    assert grid[move.row][move.col].state == CLOSED


def test_guess_key_orders_by_risk_then_information():
    # The ranking is a policy, so it is asserted directly on hand-chosen cells
    # with known neighbour counts. Going through a fixture board instead would
    # make the test depend on which cells that board happens to leave open —
    # and every small board tried turned out to be fully settled by logic, i.e.
    # to yield no guesses at all, which would have hidden a broken tie-break.
    grid = _grid(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, "#"],
        ]
    )
    # (0,3) has 3 opened neighbours and is a corner; (1,3) has 3 and is not;
    # (2,3) has 1 and is a corner.
    corner_edge = guess_key(grid, 3, 4, (0, 3), 0.25)
    plain_edge = guess_key(grid, 3, 4, (1, 3), 0.25)
    lower_risk = guess_key(grid, 3, 4, (1, 3), 0.10)

    # Same risk, same neighbour count: the non-corner wins.
    assert plain_edge < corner_edge
    # Lower risk always wins, whatever the neighbour counts.
    assert lower_risk < plain_edge
    # More opened neighbours wins at equal risk.
    assert guess_key(grid, 3, 4, (1, 3), 0.25) < guess_key(grid, 3, 4, (2, 3), 0.25)


def test_ranked_guesses_is_a_permutation_of_the_unsettled_cells():
    # The ranking must contain each unsettled closed cell exactly once, and none
    # of the cells the logic already settled.
    grid = _grid(
        [
            ["#", "#", "#", "#", "#"],
            [1, 2, 1, 2, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    ranked = ranked_guesses(grid, 3, 5, mines_left=3)
    coords = [rc for rc, _ in ranked]
    assert len(coords) == len(set(coords))

    safe, mines = _proved(grid, 3, 5)
    assert not (set(coords) & (safe | mines))

    closed = {
        (r, c)
        for r in range(3)
        for c in range(5)
        if grid[r][c].state == CLOSED
    }
    assert set(coords) == closed - (safe | mines)


def test_guess_never_returns_a_cell_the_logic_already_proved():
    grid = _grid([["F", "#", "#"], [1, 1, "#"], [0, 0, 0]])
    safe, _ = _proved(grid, 3, 3)
    move = guess(grid, 3, 3, mines_left=1)
    assert move is None or (move.row, move.col) not in safe


def test_next_moves_reports_whether_it_proved_them():
    proved = _grid([["F", "#"], [1, "#"], [0, 0]])
    moves, proven = next_moves(proved, 3, 2, mines_left=1)
    assert proven is True and moves

    # Nothing opened yet: no constraint exists, so any move is a gamble.
    blank = make_board(3, 3)
    moves, proven = next_moves(blank, 3, 3, mines_left=1)
    assert proven is False
    assert len(moves) == 1 and moves[0].kind == "open"


def test_next_moves_batches_every_proved_move():
    # Both "1"s are already satisfied by flags, so their remaining neighbours are
    # safe, and the batch applies them together instead of one per board read.
    grid = _grid([["F", "#", "#", "F"], [1, 1, "#", 1], [0, 0, 0, 0]])
    moves, proven = next_moves(grid, 3, 4, mines_left=2)
    assert proven is True
    assert {(m.row, m.col) for m in moves} >= {(0, 1), (0, 2)}


def test_a_guess_is_returned_alone():
    # A guess must not be batched with anything: the board has to be re-read
    # before the next decision, because the guess may lose the game outright.
    blank = make_board(4, 4)
    moves, proven = next_moves(blank, 4, 4, mines_left=3)
    assert proven is False
    assert len(moves) == 1


def test_is_won_when_only_mines_remain_closed():
    # Flags are cosmetic: the win condition is "every non-mine cell is open".
    assert is_won(_grid([["#"], [0]]), 2, 1, mines=1) is True
    assert is_won(_grid([[0], [0]]), 2, 1, mines=0) is True
    assert is_won(_grid([["#"], ["#"], [0]]), 3, 1, mines=1) is False


def test_render_uses_readable_tokens():
    assert render(_grid([["#", "F"], [0, 3]])) == "# F\n. 3"
