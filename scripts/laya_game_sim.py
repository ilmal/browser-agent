"""Does obeying Laya actually lose games? Play whole games, both ways.

``laya_game_bench.py`` scores a *single* guess per board, so its "safe 22/24"
says nothing about the games those picks sit inside — the runner-up is often
safe too, so a weak pick can still be a winning one. This plays the game out.

At every point where nothing is provable, one policy picks the cell to open:

* ``solver`` — the solver's own top-ranked guess (today's behaviour).
* ``laya``   — one ``noul`` head-to-head question, "is A at least as safe as B?",
  between the solver's top two. That framing cleared the 0.55 floor on **20/24**
  guesses in ``laya_pairwise_probe.py``, against 2-4/24 for the 3-way ``choice``
  head the game used before. Obey only above the floor; otherwise fall back.

Win rate is the number that matters: the goal asks for a game that is won, fast,
with minimal LLM thinking. A pick that is *safe* but never *committed* changes
nothing, which is exactly what the choice head was doing.

``python scripts/laya_game_sim.py --games 200`` inside the pod.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import random
import sys

from laya_game_bench import (
    COLS,
    MINES,
    ROWS,
    _fresh,
    _open,
    _random_board,
)

from browser_agent.config import load_settings
from browser_agent.laya_gate import LayaGate
from browser_agent.minesweeper_solver import (
    CLOSED,
    FLAGGED,
    Board,
    Cell,
    ranked_guesses,
)

SHORTLIST = 3


def _won(grid, cells) -> bool:
    return all(
        cells[r][c].state != CLOSED
        for r in range(ROWS)
        for c in range(COLS)
        if not grid[r][c]
    )


async def _laya_pick(gate, grid, cells, board, shortlist, floor: float):
    """One head-to-head noul question; None when it does not commit."""
    (ri, ci), _ = shortlist[0]
    (rj, cj), _ = shortlist[1]
    state = (
        f"{board.render()}\n"
        f"Minesweeper, {ROWS}x{COLS}, {board.mines_left} mines unmarked. "
        f"Nothing is provable; one of these cells must be opened."
    )
    question = (
        f"Is choosing row {ri + 1}, column {ci + 1} at least as safe "
        f"as choosing row {rj + 1}, column {cj + 1}?"
    )
    picked, conf = await gate.yes_no(question, state)
    if picked is None or conf < floor:
        return None, conf
    return (0 if picked else 1), conf


async def play(gate, rng, floor: float, use_laya: bool) -> dict:
    grid = _random_board(rng)
    cells = _fresh(grid)
    r0, c0 = rng.randrange(ROWS), rng.randrange(COLS)
    while grid[r0][c0]:
        r0, c0 = rng.randrange(ROWS), rng.randrange(COLS)
    _open(grid, cells, r0, c0, MINES)
    clicks, guesses, obey = 1, 1, 0

    for _ in range(600):
        if _won(grid, cells):
            return {"won": True, "clicks": clicks, "guesses": guesses, "obey": obey}
        board = Board(grid=cells, rows=ROWS, cols=COLS, mines=MINES)
        moves, proved = board.next_moves()
        if not moves:
            break
        if proved:
            for m in moves:
                if m.kind == "flag":
                    cells[m.row][m.col] = Cell(FLAGGED)
                    clicks += 1
                    continue
                if _open(grid, cells, m.row, m.col, board.mines_left) < 0:
                    return {"won": False, "clicks": clicks, "guesses": guesses, "obey": obey}
                clicks += 1
            continue

        ranked = ranked_guesses(cells, ROWS, COLS, board.mines_left)
        if not ranked:
            break
        pick = 0
        if use_laya and len(ranked) >= 2:
            idx, _ = await _laya_pick(gate, grid, cells, board, ranked[:SHORTLIST], floor)
            if idx is not None:
                pick = idx
                obey += 1
        (gr, gc), _ = ranked[pick]
        guesses += 1
        if _open(grid, cells, gr, gc, board.mines_left) < 0:
            return {"won": False, "clicks": clicks, "guesses": guesses, "obey": obey}
        clicks += 1

    return {"won": _won(grid, cells), "clicks": clicks, "guesses": guesses, "obey": obey}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--floor", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    settings = dataclasses.replace(load_settings(), laya_enabled=True,
                                   laya_pick_enabled=True)
    gate = LayaGate(settings)
    if not gate.enabled:
        print("gate disabled — set LAYA_ENABLED/LAYA_DECIDE_URL")
        return 2

    # Same board sequence for both policies, so the comparison is paired: the
    # only thing that differs between run i of each policy is the guess policy.
    runs = {}
    for label, use_laya in (("solver", False), ("laya", True)):
        rng = random.Random(args.seed)
        results, laya_questions = [], 0
        for _ in range(args.games):
            before = gate.stats["confirms"]
            r = await play(gate, rng, args.floor, use_laya)
            laya_questions += gate.stats["confirms"] - before
            results.append(r)
        wins = sum(r["won"] for r in results)
        clicks = sum(r["clicks"] for r in results)
        guesses = sum(r["guesses"] for r in results)
        obey = sum(r["obey"] for r in results)
        runs[label] = results
        print(
            f"{label:7} won {wins:3}/{len(results)} ({100.0 * wins / len(results):.0f}%)  "
            f"guesses {guesses:4}  clicks {clicks:5}  "
            f"mean clicks/win {clicks / max(wins, 1):.1f}  "
            f"laya committed on {obey}/{guesses} guesses  questions={laya_questions}"
        )

    s, la = runs["solver"], runs["laya"]
    only_laya = sum(1 for a, b in zip(s, la, strict=True) if b["won"] and not a["won"])
    only_solver = sum(1 for a, b in zip(s, la, strict=True) if a["won"] and not b["won"])
    print(f"\npaired on the same {len(s)} boards: "
          f"laya wins {only_laya} the solver loses, solver wins {only_solver} laya loses")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
