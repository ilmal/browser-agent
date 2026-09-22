"""Measure horizontal overflow on a bot page, with a real run's data loaded.

Written 2026-09-22 after the operator reported "stuff is overflowing in the
flight-planner page after a run". The guess was a long result string in the
tasks table; the point of scripting it is to *measure* which element is wider
than its container rather than patch the first suspect.

Run it against the live page (needs the basic-auth credentials):

    python scripts/ui_overflow_check.py https://browser.ilmal.se/b/flight-planner/ \
        --user you@example.com --password '...'

It prints, per element, the overflow in pixels: ``scrollWidth`` beyond
``clientWidth`` for a block, and ``right`` beyond the viewport for anything.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

_PROBE = """
() => {
  const out = [];
  const vw = document.documentElement.clientWidth;
  for (const el of document.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    const over = Math.round(r.right - vw);
    const scroll = el.scrollWidth - el.clientWidth;
    const wide = Math.round(scroll);
    if (over > 1 || wide > 1) {
      out.push({
        tag: el.tagName.toLowerCase(),
        id: el.id || '',
        cls: (el.className && el.className.toString().slice(0, 60)) || '',
        over,
        wide,
        text: (el.textContent || '').trim().slice(0, 60),
      });
    }
  }
  // Widest offenders first: the element that actually pushes the layout.
  out.sort((a, b) => Math.max(b.over, b.wide) - Math.max(a.over, a.wide));
  const de = document.documentElement;
  return {vw, docScroll: de.scrollWidth, bodyScroll: document.body.scrollWidth,
          out: out.slice(0, 40)};
}
"""


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--wait", type=float, default=4.0)
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
        await page.goto(args.url, wait_until="networkidle")
        await page.wait_for_timeout(args.wait * 1000)
        # Open every collapsed <details> — the result block is the suspected
        # offender and it is closed by default, so a probe that skips this
        # measures a page the operator never sees.
        await page.evaluate(
            "() => document.querySelectorAll('details').forEach(d => d.open = true)"
        )
        await page.wait_for_timeout(400)
        res = await page.evaluate(_PROBE)
        await browser.close()

    print(f"viewport {res['vw']}  documentScrollWidth {res['docScroll']}  "
          f"bodyScrollWidth {res['bodyScroll']}")
    if res["docScroll"] <= res["vw"]:
        print("\nno page-level horizontal overflow")
    print()
    for e in res["out"]:
        print(f"  over={e['over']:>5}  inner={e['wide']:>5}  "
              f"<{e['tag']}{'#' + e['id'] if e['id'] else ''}"
              f"{'.' + e['cls'] if e['cls'] else ''}>  {e['text']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
