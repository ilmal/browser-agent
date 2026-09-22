"""Does the "Talk to it" composer survive a poll?

Written 2026-09-22 after the operator reported "page is constantyl refershing so
cant write or anything". The thread panel re-renders every 2 s, and the bug was
that the whole panel — textarea included — was rewritten on every tick, so the
caret was destroyed mid-sentence and nothing could be typed.

The regression this guards against is subtle: ``set()`` dedupes on
``el.dataset.sig``, but if the markup embeds anything that changes over time
(an ``ago(ts)`` string is the obvious one) the signature changes every tick and
the node is rewritten anyway. This probe does not read the diff — it *types* and
checks the text is still there after several real poll cycles, which is the only
thing the operator actually cares about.

    python scripts/ui_thread_probe.py https://browser.ilmal.se/b/flight-planner/ \
        --user you@example.com --password '...' --task-id <id>
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--ticks", type=int, default=4, help="poll cycles to wait")
    ap.add_argument("--task-id", default="", help="open this thread's composer")
    ap.add_argument("--text", default="try the date grid instead")
    args = ap.parse_args()

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            http_credentials=(
                {"username": args.user, "password": args.password} if args.user else None
            ),
        )
        page = await ctx.new_page()
        page.on("console", lambda m: print("  [console]", m.type, m.text[:160]))
        page.on("pageerror", lambda e: print("  [pageerror]", str(e)[:200]))
        await page.goto(args.url, wait_until="networkidle")

        # Open the thread panel the way the operator does: the History row's
        # "Talk to it" button. Fall back to the banner if there is no row.
        if args.task_id:
            clicked = await page.evaluate(
                """(tid) => {
                    const b = document.querySelector(`[data-talk="${tid}"]`)
                        || [...document.querySelectorAll('[data-talk]')][0];
                    if (!b) return false;
                    b.click();
                    return true;
                }""",
                args.task_id,
            )
        else:
            clicked = await page.evaluate(
                """() => {
                    const b = document.querySelector('[data-talk]');
                    if (!b) return false;
                    b.click();
                    return true;
                }"""
            )
        print(f"opened thread: {clicked}")
        await page.wait_for_timeout(1500)

        composer = await page.query_selector("#thread-form textarea")
        if composer is None:
            print("NO COMPOSER FOUND — panel did not open")
            await browser.close()
            return 1

        # Type with real keystrokes, then leave the caret in the box.
        await composer.click()
        await composer.type(args.text, delay=20)
        before = await composer.input_value()
        print(f"typed           : {before!r}")

        # Sit through several poll cycles without touching the page. The poll
        # interval is 2 s, so wait generously past it.
        stale = []
        for tick in range(args.ticks):
            await page.wait_for_timeout(2500)
            el = await page.query_selector("#thread-form textarea")
            if el is None:
                stale.append(f"tick {tick}: textarea GONE")
                break
            val = await el.input_value()
            focused = await page.evaluate(
                "() => document.activeElement?.tagName === 'TEXTAREA'"
            )
            print(f"  tick {tick}: value={val!r} focused={focused}")
            if val != before:
                stale.append(f"tick {tick}: value changed to {val!r}")
            if not focused:
                stale.append(f"tick {tick}: lost focus")

        # And a message typed later must still append, not replace.
        el = await page.query_selector("#thread-form textarea")
        if el is not None:
            await el.click()
            await el.type(" and also", delay=20)
            after = await el.input_value()
            print(f"after more typing: {after!r}")

        await browser.close()

    if stale:
        print("\nFAIL — the composer was disturbed while idle:")
        for s in stale:
            print("  ", s)
        return 1
    print("\nPASS — the composer held its value across every poll")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
