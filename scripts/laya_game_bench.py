"""Does Laya actually help at minesweeper? Measured against ground truth.

The goal asks the game to lean on Laya, but the solver already computes real
constraint-derived probabilities while Laya is a small model reading a *text*
rendering of the board. Those can disagree, and the solver is the one doing
arithmetic, so "lean on Laya" is only a good idea if Laya's picks are at least
as safe. This bench answers that with ground truth rather than opinion.

Method: generate random Beginner boards (9x9/10) with known mine positions,
open the mandatory first cell, then repeatedly play the cells the solver
*proves*. Stop at the first genuine guess — that is the only place the tiebreak
ever runs — and ask Laya to choose among the same shortlist the recipe builds.
Score both Laya's pick and the solver's own top pick against the real board.

Run inside the browser-agent pod, where ``LAYA_DECIDE_URL`` and the key are the
live ones (``python /tmp/laya_game_bench.py``). No browser and no egress: the
boards are synthetic, so this is safe to run while every exit IP is banned.

Usage: ``python scripts/laya_game_bench.py [--boards 40] [--floor 0.55]``
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import random
import sys

from browser_agent.config import load_settings
from browser_agent.laya_gate import LayaGate
from browser_agent.minesweeper_solver import (
    CLOSED,
    FLAGGED,
    OPENED,
    Board,
    Cell,
    ranked_guesses,
)

ROWS, COLS, MINES = 9, 9, 10
SHORTLIST = 3
QUESTION = "Which numbered cell is the safest to open next?"


def _random_board(rng: random.Random) -> list[list[bool]]:
    mines = set()
    while len(mines) < MINES:
        mines.add((rng.randrange(ROWS), rng.randrange(COLS)))
    return [[(r, c) in mines for c in range(COLS)] for r in range(ROWS)]


def _count(grid: list[list[bool]], r: int, c: int) -> int:
    return sum(
        grid[y][x]
        for y in range(max(0, r - 1), min(ROWS, r + 2))
        for x in range(max(0, c - 1), min(COLS, c + 2))
    )


def _fresh(mines: list[list[bool]]) -> list[list[Cell]]:
    return [[Cell(CLOSED) for _ in range(COLS)] for _ in range(ROWS)]


def _open(grid: list[list[bool]], cells: list[list[Cell]], r: int, c: int,
          mines_left: int) -> int:
    """Open (r,c) and flood-fill blanks, exactly as the site does. -1 = mine."""
    if grid[r][c]:
        return -1
    seen = [(r, c)]
    while seen:
        y, x = seen.pop()
        if cells[y][x].state == OPENED:
            continue
        n = _count(grid, y, x)
        cells[y][x] = Cell(OPENED, n)
        if n == 0:
            for yy in range(max(0, y - 1), min(ROWS, y + 2)):
                for xx in range(max(0, x - 1), min(COLS, x + 2)):
                    if cells[yy][xx].state == CLOSED:
                        seen.append((yy, xx))
    return 0


def _play_until_guess(grid, cells, rng) -> bool:
    """Play proved moves to a fixpoint. False if a mine was hit."""
    for _ in range(400):
        board = Board(grid=cells, rows=ROWS, cols=COLS, mines=MINES)
        moves, proved = board.next_moves()
        if not moves:
            return False
        if not proved:
            return True  # a genuine guess is due
        for m in moves:
            if m.kind == "flag":
                cells[m.row][m.col] = Cell(FLAGGED)
                continue
            if _open(grid, cells, m.row, m.col, board.mines_left) < 0:
                return False
    return False


async def one_board(gate: LayaGate, rng: random.Random, floor: float) -> dict | None:
    grid = _random_board(rng)
    cells = _fresh(grid)
    r0, c0 = rng.randrange(ROWS), rng.randrange(COLS)
    while grid[r0][c0]:  # first press never loses, as the site guarantees
        r0, c0 = rng.randrange(ROWS), rng.randrange(COLS)
    _open(grid, cells, r0, c0, MINES)
    if not _play_until_guess(grid, cells, rng):
        return None  # no guess needed, or a solver bug lost it — not our case

    board = Board(grid=cells, rows=ROWS, cols=COLS, mines=MINES)
    ranked = ranked_guesses(cells, ROWS, COLS, board.mines_left)
    if len(ranked) < 2:
        return None
    shortlist = ranked[:SHORTLIST]

    lines = [f"row {r + 1}, column {c + 1} (risk about {p:.0%})" for (r, c), p in shortlist]
    state = (
        f"{board.render()}\n"
        f"Minesweeper, {ROWS}x{COLS}, {board.mines_left} mines unmarked. "
        f"Nothing is provable; one of these cells must be opened."
    )
    idx, conf = await gate.choose(QUESTION, lines, state)

    (sr, sc), _ = shortlist[0]                      # the solver's own top pick
    solver_safe = not grid[sr][sc]
    laya_safe = None
    if idx is not None and 0 <= idx < len(shortlist):
        (lr, lc), _ = shortlist[idx]
        laya_safe = not grid[lr][lc]
    return {
        "acted": idx is not None and conf >= floor,
        "conf": conf,
        "solver_safe": solver_safe,
        "laya_safe": laya_safe,
        "agrees": idx == 0,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=40)
    ap.add_argument("--floor", type=float, default=0.55)
    args = ap.parse_args()

    settings = dataclasses.replace(load_settings(), laya_enabled=True,
                                   laya_pick_enabled=True)
    gate = LayaGate(settings)
    if not gate.enabled:
        print("gate disabled — set LAYA_ENABLED/LAYA_DECIDE_URL")
        return 2

    rng = random.Random(20260922)
    rows = []
    for i in range(args.boards):
        r = await one_board(gate, rng, args.floor)
        if r is None:
            continue
        r["n"] = len(rows) + 1
        rows.append(r)
        mark = "act" if r["acted"] else "  -"
        laya = "-" if r["laya_safe"] is None else ("safe" if r["laya_safe"] else "MINE")
        print(f"  board {r['n']:2}  conf={r['conf']:.3f} {mark}  "
              f"laya={laya:4} solver={'safe' if r['solver_safe'] else 'MINE':4} "
              f"agree={r['agrees']}")

    if not rows:
        print("no ambiguous boards produced")
        return 1

    acted = [r for r in rows if r["acted"]]
    def pct(n, d): return f"{100.0 * n / d:.0f}%" if d else "n/a"

    solver_ok = sum(r["solver_safe"] for r in rows)
    laya_ok = sum(1 for r in rows if r["laya_safe"])
    print()
    print(f"boards with a real guess: {len(rows)}")
    print(f"solver top pick safe:     {solver_ok}/{len(rows)} ({pct(solver_ok, len(rows))})")
    print(f"laya pick safe:           {laya_ok}/{len(rows)} ({pct(laya_ok, len(rows))})")
    print(f"laya agreed with solver:  {sum(r['agrees'] for r in rows)}/{len(rows)}")
    print(f"cleared floor {args.floor}:      {len(acted)}/{len(rows)} ({pct(len(acted), len(rows))})")
    if acted:
        a_ok = sum(1 for r in acted if r["laya_safe"])
        s_ok = sum(1 for r in acted if r["solver_safe"])
        print(f"  on those, laya safe:    {a_ok}/{len(acted)} ({pct(a_ok, len(acted))})")
        print(f"  on those, solver safe:  {s_ok}/{len(acted)} ({pct(s_ok, len(acted))})")
        worse = sum(1 for r in acted if not r["laya_safe"] and r["solver_safe"])
        print(f"  acted and WORSE than the solver: {worse}/{len(acted)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
