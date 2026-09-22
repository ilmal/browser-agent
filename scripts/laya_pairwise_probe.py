"""Can a *pairwise* (noul yes/no) framing break the 0.5 confidence wall?

``laya_choice_probe.py`` showed the 3-way ``choice`` head is the problem: its
chosen-label probabilities sit at **0.5**, so the 0.55 game floor is cleared on
only 2-4/24 guesses under every wording. That is a property of the head, not of
the question — the gate's other answer type, ``noul`` (binary yes/no), derives
confidence as ``max(p, 1-p)``, which is >= 0.5 by construction and clears 0.55 as
soon as the model is even slightly decided.

So the same shortlist is re-asked as a bracket of pairwise comparisons:

    "Is row R, column C at least as safe to open as row R2, column C2?"

and the winner is the cell that survives. If the noul head is decisive where the
choice head is not, the tiebreak becomes usable at the existing floor — no new
threshold, no new model, no fresh egress. That is the whole question this
answers, and it is answerable offline.

Scores each framing the same way the game does: is the pick safe (ground truth),
and does it clear ``LAYA_GAME_MIN_CONFIDENCE``.

``python scripts/laya_pairwise_probe.py --boards 24`` inside the pod.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import random
import sys

from laya_choice_probe import (
    _boards,
    _framings,
)

from browser_agent.config import load_settings
from browser_agent.laya_gate import LayaGate


async def _duel(gate: LayaGate, shortlist, state: str, i: int, j: int):
    """One head-to-head noul question between shortlist[i] and shortlist[j]."""
    (ri, ci), _ = shortlist[i]
    (rj, cj), _ = shortlist[j]
    question = (
        f"Is choosing row {ri + 1}, column {ci + 1} at least as safe "
        f"as choosing row {rj + 1}, column {cj + 1}?"
    )
    picked, conf = await gate.yes_no(question, state)
    return (i if picked else j) if picked is not None else None, conf


async def _pairwise(gate: LayaGate, shortlist, state: str):
    """Bracket the shortlist with noul comparisons; return (index, min_conf).

    Round-robin: every pair is asked once, the cell winning more duels takes it.
    Ties break toward the solver's own ranking, which is the status quo anyway.
    """
    n = len(shortlist)
    wins = [0] * n
    confs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            winner, conf = await _duel(gate, shortlist, state, i, j)
            confs.append(conf)
            if winner is not None:
                wins[winner] += 1
    best = max(wins)
    if best == 0:
        return None, (min(confs) if confs else 0.0)
    # First index with the top count — ties favour the solver's ranking.
    return wins.index(best), min(confs)


async def _one_duel(gate: LayaGate, shortlist, state: str):
    """The minimum-thinking variant: the solver's top pick against its runner-up.

    One question, and only the winner's own confidence gates it — the goal asks
    for minimal LLM thinking, so the cheapest framing that still beats "never
    commit" is the one that matters.
    """
    winner, conf = await _duel(gate, shortlist, state, 0, 1)
    return winner, conf


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=24)
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

    tally = {
        k: {"pick0": 0, "safe": 0, "acted": 0, "picks": collections.Counter()}
        for k in ("ranked", "plain", "labelled", "pairwise", "one_duel")
    }

    for nth, (grid, shortlist, state) in enumerate(boards, 1):
        row = [f"solver={'safe' if not grid[shortlist[0][0][0]][shortlist[0][0][1]] else 'MINE'}"]
        for name, lines in _framings(shortlist).items():
            idx, conf = await gate.choose(
                "Which numbered cell is the safest to open next?", lines, state
            )
            _score(tally[name], idx, conf, shortlist, grid, len(shortlist), 0.55)
            row.append(f"{name[:4]}: idx={idx} c={conf:.2f}")
        for name in ("pairwise", "one_duel"):
            fn = _pairwise if name == "pairwise" else _one_duel
            idx, conf = await fn(gate, shortlist, state)
            _score(tally[name], idx, conf, shortlist, grid, len(shortlist), 0.55)
            row.append(f"{name[:4]}: idx={idx} c={conf:.2f}")
        print(f"  board {nth:2}  " + "  ".join(row))

    n = len(boards)
    print(f"\n=== by framing (n={n}) ===")
    for name, t in tally.items():
        print(f"  {name:9} picked index 0: {t['pick0']:2}/{n}   "
              f"safe: {t['safe']:2}/{n}   cleared 0.55: {t['acted']:2}/{n}")
        print(f"            pick distribution: {dict(t['picks'])}")
    return 0


def _score(t, idx, conf, shortlist, grid, n, floor) -> None:
    t["picks"][idx] += 1
    if idx == 0:
        t["pick0"] += 1
    if idx is not None and 0 <= idx < n:
        (lr, lc), _ = shortlist[idx]
        if not grid[lr][lc]:
            t["safe"] += 1
        if conf >= floor:
            t["acted"] += 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
