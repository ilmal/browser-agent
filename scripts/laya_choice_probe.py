"""Why does Laya never pick index 0 at minesweeper? A controlled probe.

The game bench recorded agreement **0/17**: Laya never once chose the solver's
top-ranked cell. That is not "picks at random" — random would agree ~1/3 of the
time. It is a systematic avoidance of the first option.

Measured in-pod against the live ``/v1/decide``, 2026-09-22, n=24 boards:

    position control (neutral 3-way question, NO board): {0: 13, 1: 1, 2: 1}
    board question, today's wording ("risk about P%"):   pick-0  0/24
    board question, coordinates only:                    pick-0  1/24
    board question, A/B/C labels:                        pick-0  8/24

So the model is **strongly first-option-biased in the abstract** (13/15 picked
index 0 when nothing was at stake) and that bias **inverts on a board question**.
It lands on index 1 or 2 almost exclusively, regardless of how the options are
worded or whether numbers are present. The 0/17 was this collapse, not a
reasoning failure about risk and not a wording failure.

What this means for the goal ("rely heavy on laya"): the tiebreak cannot be
improved by prompting. Three framings moved pick-0 from 0 to 1 to 8 of 24, all
far from a model that would actually rank. Confidence is the real wall: clearing
the 0.55 game floor stayed at 2-4/24 (8-17%) across every framing, because the
choice probabilities sit near 0.5. Laya's own safe-pick rate (21-22/24) tracks
the solver's 24/24 closely, so it is not *wrong* when it answers — it just
almost never commits. The solver's own top pick is the better and free decision.

Kept because the negative result is the finding, and because it is cheap to
re-run after a model or checkpoint change (no browser, no egress — every exit IP
is banned, and this needs none).

``python scripts/laya_choice_probe.py --boards 24`` inside the pod.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
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


def _random_board(rng: random.Random) -> list[list[bool]]:
    mines = set()
    while len(mines) < MINES:
        mines.add((rng.randrange(ROWS), rng.randrange(COLS)))
    return [[(r, c) in mines for c in range(COLS)] for r in range(ROWS)]


def _count(grid, r, c) -> int:
    return sum(
        grid[y][x]
        for y in range(max(0, r - 1), min(ROWS, r + 2))
        for x in range(max(0, c - 1), min(COLS, c + 2))
    )


def _open(grid, cells, r, c, mines_left) -> int:
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


def _play_until_guess(grid, cells) -> bool:
    for _ in range(400):
        board = Board(grid=cells, rows=ROWS, cols=COLS, mines=MINES)
        moves, proved = board.next_moves()
        if not moves:
            return False
        if not proved:
            return True
        for m in moves:
            if m.kind == "flag":
                cells[m.row][m.col] = Cell(FLAGGED)
                continue
            if _open(grid, cells, m.row, m.col, board.mines_left) < 0:
                return False
    return False


def _boards(rng: random.Random, want: int):
    """(grid, shortlist, state_text) for boards that reach a real guess."""
    out = []
    while len(out) < want:
        grid = _random_board(rng)
        cells = [[Cell(CLOSED) for _ in range(COLS)] for _ in range(ROWS)]
        r0, c0 = rng.randrange(ROWS), rng.randrange(COLS)
        while grid[r0][c0]:
            r0, c0 = rng.randrange(ROWS), rng.randrange(COLS)
        _open(grid, cells, r0, c0, MINES)
        if not _play_until_guess(grid, cells):
            continue
        board = Board(grid=cells, rows=ROWS, cols=COLS, mines=MINES)
        ranked = ranked_guesses(cells, ROWS, COLS, board.mines_left)
        if len(ranked) < 2:
            continue
        shortlist = ranked[:SHORTLIST]
        state = (
            f"{board.render()}\n"
            f"Minesweeper, {ROWS}x{COLS}, {board.mines_left} mines unmarked. "
            f"Nothing is provable; one of these cells must be opened."
        )
        out.append((grid, shortlist, state))
    return out


def _framings(shortlist):
    """The three candidate wordings."""
    ranked = [
        f"row {r + 1}, column {c + 1} (risk about {p:.0%})" for (r, c), p in shortlist
    ]
    plain = [f"row {r + 1}, column {c + 1}" for (r, c), _ in shortlist]
    labelled = [
        f"option {L}: row {r + 1}, column {c + 1}"
        for L, ((r, c), _) in zip("ABC", shortlist, strict=False)
    ]
    return {"ranked": ranked, "plain": plain, "labelled": labelled}


async def _position_control(gate: LayaGate, n: int) -> collections.Counter:
    """A neutral 3-way question with no board; is the pick uniform?"""
    picks: collections.Counter = collections.Counter()
    for i in range(n):
        lines = [f"option {L}: candidate {j}" for j, L in enumerate("ABC")]
        idx, _ = await gate.choose(
            "Which numbered option should be selected?",
            lines,
            f"A neutral list. Selection number {i + 1}.",
        )
        picks[idx] += 1
    return picks


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=24)
    ap.add_argument("--control", type=int, default=15)
    args = ap.parse_args()

    settings = dataclasses.replace(load_settings(), laya_enabled=True,
                                   laya_pick_enabled=True)
    gate = LayaGate(settings)
    if not gate.enabled:
        print("gate disabled — set LAYA_ENABLED/LAYA_DECIDE_URL")
        return 2

    rng = random.Random(20260922)
    boards = _boards(rng, args.boards)
    print(f"boards reaching a real guess: {len(boards)}\n")

    tally = {k: {"pick0": 0, "safe": 0, "acted": 0, "picks": collections.Counter()}
             for k in ("ranked", "plain", "labelled")}

    for nth, (grid, shortlist, state) in enumerate(boards, 1):
        solver_safe = not grid[shortlist[0][0][0]][shortlist[0][0][1]]
        row = [f"solver={'safe' if solver_safe else 'MINE'}"]
        for name, lines in _framings(shortlist).items():
            idx, conf = await gate.choose(
                "Which numbered cell is the safest to open next?", lines, state
            )
            t = tally[name]
            t["picks"][idx] += 1
            if idx == 0:
                t["pick0"] += 1
            if idx is not None and 0 <= idx < len(shortlist):
                (lr, lc), _ = shortlist[idx]
                if not grid[lr][lc]:
                    t["safe"] += 1
                if conf >= 0.55:
                    t["acted"] += 1
            row.append(f"{name}: idx={idx} conf={conf:.2f}")
        print(f"  board {nth:2}  " + "  ".join(row))

    n = len(boards)
    print(f"\n=== by framing (n={n}) ===")
    for name, t in tally.items():
        print(f"  {name:9} picked index 0: {t['pick0']:2}/{n}   "
              f"safe: {t['safe']:2}/{n}   cleared 0.55: {t['acted']:2}/{n}")
        print(f"            pick distribution: {dict(t['picks'])}")

    print("\n=== position control (no board) ===")
    picks = await _position_control(gate, args.control)
    print(f"  {dict(picks)}  (uniform would be ~{args.control // 3} each)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
