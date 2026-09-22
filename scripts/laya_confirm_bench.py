"""Fixture accuracy bench for the plan.task Laya CONFIRM role (noul).

``confirm_step`` asks laya one yes/no — "Did executing '<desc>' move the page
closer to completing the overall task '<task>'?" — and advances the plan only on
a confident yes. The two errors are not symmetric: a wrong *yes* lets a step that
did nothing through, and a wrong *no* throws a good plan into the agent fallback.
So the floor that decides which answers count is a precision knob, not a taste
call: sweep it and read the table rather than guessing.

Runs against whatever backend ``Settings`` points at, but the number that matters
is the one the pod's own engine produces, so pass the URL the pod uses.

**The engine matters more than the floor.** Measured 2026-09-22: the same twelve
cases score 0/12 on cn1's ``vulkan:1`` build and 12/12 on ``--device cpu``. The
ggmlc Vulkan kernels miscompute ModernBERT-class models on RDNA1, so a floor
swept against the vulkan engine is a table of noise. Confirm the device first
(``GET /health`` reports it).

Usage: ``python scripts/laya_confirm_bench.py [decide-url]``
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from dataclasses import dataclass

from browser_agent.config import load_settings
from browser_agent.laya_gate import LayaGate


@dataclass
class Case:
    task: str
    desc: str
    state: str
    expect: bool


CASES = [
    Case(
        "go to minesweeper and play a game until you win",
        "clicked the 'Beginner' link to start a beginner game",
        "Minesweeper Online - Beginner 9x9, 10 mines. A grid of 81 cells is "
        "displayed and the timer is running at 00:03.",
        True,
    ),
    Case(
        "read the news",
        "accepted the cookie consent dialog",
        "Today's headlines from around the region. The article list is visible "
        "and no consent dialog is present.",
        True,
    ),
    Case(
        "find the pricing",
        "navigated to the pricing page",
        "Pricing - Plans and pricing. Three tiers: Free, Pro and Team, with a "
        "monthly/annual toggle.",
        True,
    ),
    Case(
        "sign in to the account",
        "typed the email address into the sign-in form",
        "Sign in. The email address field contains user@example.com, the "
        "password field is empty, and a Sign in button is present.",
        True,
    ),
    Case(
        "buy a widget",
        "submitted the order",
        "Order confirmed. Thank you for your purchase. Order number 10428.",
        True,
    ),
    Case(
        "buy a widget",
        "submitted the order",
        "Your card was declined. Please try another payment method.",
        False,
    ),
    Case(
        "sign in to the account",
        "clicked the sign in button",
        "Sign in. The email address field is empty, the password field is "
        "empty, and the form is still asking for credentials.",
        False,
    ),
    Case(
        "read the news",
        "dismissed the cookie consent dialog",
        "We use cookies. Accept all | Reject all | Settings. The consent dialog "
        "is still displayed over the article.",
        False,
    ),
    Case(
        "go to minesweeper and play a game until you win",
        "clicked a cell to begin playing",
        "Minesweeper Online. All 81 cells are still closed. No cell has been "
        "opened and the timer has not started.",
        False,
    ),
    Case(
        "open account settings",
        "navigated to the account settings page",
        "404 Not Found - the page you requested could not be found.",
        False,
    ),
    Case(
        "search for running shoes",
        "submitted the search",
        "Store. The search box is empty and no results are shown.",
        False,
    ),
    Case(
        "buy a widget",
        "confirmed the payment",
        "Payment failed. Please check your card details and try again.",
        False,
    ),
]

#: The floors worth distinguishing. Below ~0.6 the model is guessing.
FLOORS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def question(case: Case) -> str:
    """The exact question ``confirm_step`` asks."""
    return (
        f"Did executing '{case.desc}' move the page closer to completing the "
        f"overall task '{case.task}'?"
    )


async def main() -> int:
    settings = load_settings()
    if len(sys.argv) > 1:
        settings = dataclasses.replace(
            settings, laya_decide_url=sys.argv[1], laya_enabled=True
        )
    gate = LayaGate(settings)

    rows: list[tuple[Case, bool | None, float]] = []
    for case in CASES:
        verdict, conf = await gate.yes_no(question(case), case.state)
        rows.append((case, verdict, conf))

    print(f"\nbackend: {gate.summary}")
    print(f"{'expect':>6} {'verdict':>7} {'conf':>5}  ok   desc")
    for case, verdict, conf in rows:
        ok = verdict is not None and verdict == case.expect
        print(f"{str(case.expect):>6} {str(verdict):>7} {conf:5.3f}  {str(ok):<4} {case.desc[:50]}")

    correct = sum(1 for c, v, _ in rows if v is not None and v == c.expect)
    print(f"\ntotal {len(rows)} | correct {correct} ({correct / len(rows):.0%})")

    print("\nfloor sweep (a wrong answer at or above the floor is the failure that matters):")
    best: float | None = None
    for floor in FLOORS:
        acted = [(c, v, conf) for c, v, conf in rows if v is not None and conf >= floor]
        wrong = [(c, v, conf) for c, v, conf in acted if v != c.expect]
        yes = sum(1 for _, v, _ in acted if v)
        note = "PRECISION 1.0" if acted and not wrong else ("no answers acted" if not acted else "ERRORS")
        print(f"  floor {floor:.2f}: acted {len(acted):2d}/{len(rows)}  yes {yes:2d}  wrong {len(wrong):2d}  {note}")
        if best is None and acted and not wrong:
            best = floor

    print(f"\nlowest floor with zero errors: {best if best is not None else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
