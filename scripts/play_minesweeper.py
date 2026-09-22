"""Play a real Beginner game and print what happened — the live smoke test.

Runs the shipped recipe path (``Minesweeper._play_game``) against a real
browser, so a green run means the DOM bridge, the solver and the pacing all
work together, not just in unit tests. Not part of the app; it exists so
"does the whole thing actually play?" has a one-command answer.

    python scripts/play_minesweeper.py [--proxy socks5://...] [--games N]

Deliberately low volume: one game, real clicks, human-ish gaps. The site bans
per egress IP on interaction volume, so do not loop this.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from playwright.async_api import async_playwright  # noqa: E402

from browser_agent.activity import Activity  # noqa: E402
from browser_agent.browser import _user_agent  # noqa: E402
from browser_agent.config import load_settings  # noqa: E402
from browser_agent.laya_gate import LayaGate  # noqa: E402
from browser_agent.minesweeper_dom import Pace, start_beginner  # noqa: E402
from browser_agent.recipes.minesweeper import Minesweeper  # noqa: E402

log = logging.getLogger("play-minesweeper")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="", help="SOCKS5/HTTP proxy for the browser")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--pace-min", type=int, default=220)
    ap.add_argument("--pace-max", type=int, default=700)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    recipe = Minesweeper(laya=LayaGate(settings), settings=settings,
                         pace=Pace(min_ms=args.pace_min, max_ms=args.pace_max))
    activity = Activity()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True,
                                          args=["--no-sandbox", "--disable-dev-shm-usage"])
        ctx_kwargs = {"viewport": {"width": 1440, "height": 900}, "locale": "en-US"}
        ua = _user_agent(headless=True)
        if ua:
            ctx_kwargs["user_agent"] = ua
        if args.proxy:
            ctx_kwargs["proxy"] = {"server": args.proxy}
        ctx = await browser.new_context(**ctx_kwargs)
        page = await ctx.new_page()

        for game_no in range(1, args.games + 1):
            view = await start_beginner(page)
            print(f"\n=== game {game_no}: cells={view.n_cells} blocked={view.blocked} ===")
            if view.blocked:
                print("BLOCKED by the site — this egress IP is banned.")
                await browser.close()
                return 2
            if not view.cells_ready:
                print(f"board never rendered ({view.n_cells} cells) — UA gate or network.")
                await browser.close()
                return 3
            print(view.board.render())
            result = await recipe._play_game(page, view, activity, game_no)
            print(f"result: {result['outcome']} clicks={result['clicks']} "
                  f"guesses={result['guesses']} laya_calls={result['laya_calls']}")
            if "board" in result:
                print(result["board"])

        print("\n--- activity ---")
        for entry in activity.as_list():
            print(f"  [{entry['kind']}] {entry['text']}")
        await browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
