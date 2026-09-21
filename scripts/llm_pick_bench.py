"""Fixture accuracy bench for the LLM element picker (browser_agent.picker).

Same fixtures, scoring and verdict rule as ``laya_pick_bench.py`` — this is
the second half of the engine decision the two benches document: on this
corpus the laya engine scored 4/18 (flat probabilities on both heads) while
deepseek-v4.1-flash scored 18/18 at ~1 s/pick, which is why the picker role
moved to the LLM and laya kept only the confirm role.

Runs against whatever ``Settings`` points at, so run it inside the
browser-agent pod for the number that matters (prod llm-service path). The
picker flag is forced on for the run; the bench measures the picker, not the
deployment flag.

Usage (in the pod): ``python scripts/llm_pick_bench.py``
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import time

from playwright.async_api import async_playwright

from browser_agent.config import load_settings
from browser_agent.picker import ElementPicker
from browser_agent.recipes._candidates import extract_candidates
from browser_agent.recipes._plan_exec import state_text as page_state_text

from laya_pick_bench import FIXTURES


async def main() -> int:
    settings = dataclasses.replace(load_settings(), picker_enabled=True)
    picker = ElementPicker(settings)

    rows: list[tuple[str, str, bool, bool]] = []
    started = time.time()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for fx in FIXTURES:
            await page.set_content(fx.html)
            _, lines = await extract_candidates(page, fx.base, settings.laya_max_candidates)
            idx, _ = await picker.pick(
                fx.goal, await page_state_text(page, 400), lines
            )
            if idx is None or not 0 <= idx < len(lines):
                rows.append((fx.name, "<no pick>", False, True))
                continue
            picked_line = lines[idx]
            correct = any(marker in picked_line for marker in fx.expect_any)
            rows.append((fx.name, picked_line, correct, False))
        await browser.close()
    elapsed = time.time() - started

    correct = sum(1 for r in rows if r[2])
    no_pick = sum(1 for r in rows if r[1] == "<no pick>")
    total = len(rows)

    print(f"picker: {settings.picker_model} | floor semantics: parse+range only")
    print(f"{'fixture':<18} correct  picked line")
    for name, line, ok, badp in rows:
        flag = "BAD-PARSE" if badp else ""
        print(f"{name:<18} {str(ok):<7} {flag:<9} {line[:70]}")
    print(f"\ntotal {total} | correct {correct} ({correct / total:.0%}) | no-pick {no_pick} "
          f"| {elapsed / total:.2f}s avg")

    ok = correct == total and no_pick == 0
    print(f"\nverdict: {'PASS — picker may be enabled' if ok else 'FAIL — keep the picker off'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
