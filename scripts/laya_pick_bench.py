"""Fixture accuracy bench for the plan.task Laya picker.

Serves known-target fixture pages in a throwaway headless Chromium, enumerates
candidates exactly like the plan executor (``extract_candidates`` + the same
CLICK/TYPE bases), asks the Laya gate to pick with the same question and state
text ``_plan_exec.resolve_target`` uses, and scores each pick against a
known-correct marker embedded in the candidate's visible text, aria-label,
placeholder or href.

Runs against whatever backend ``Settings`` points at. The number that gates
the deployment flag is the prod backend — llm-service's ``/v1/decide`` proxy —
so run this inside the browser-agent pod, where ``LAYA_DECIDE_URL``, the key
and the network path are the live ones. The pip backend on a laptop measures
a different engine and proves nothing about prod.

The picker flag is forced on for the run (``dataclasses.replace`` — Settings
is frozen); the bench measures the picker itself, not the deployment flag.

Verdict rule printed at the end: the picker may be trusted when every acted-on
pick (confidence >= floor) is correct (precision 1.0) and at least 80% of
fixtures get an acted-on pick — a misclick that survives the floor clicks the
wrong element, which is the one failure mode the fallback cannot undo.

Usage (in the pod): ``python scripts/laya_pick_bench.py``
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys

from playwright.async_api import async_playwright

from browser_agent.config import load_settings
from browser_agent.laya_gate import LayaGate
from browser_agent.recipes._candidates import CLICK_BASE, TYPE_BASE, extract_candidates
from browser_agent.recipes._plan_exec import state_text as page_state_text

QUESTION = "Which numbered element should be activated to accomplish the page goal?"


def _page(title: str, body: str) -> str:
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


_NAV = (
    '<nav><a href="/">Home</a> <a href="/about">About</a> '
    '<a href="/pricing">Pricing</a> <a href="/contact">Contact</a></nav>'
)

_ORDER_FORM = (
    "<form>"
    '<button type="button">Save draft</button> '
    '<button type="button">Cancel</button> '
    '<button type="submit">Submit order</button> '
    '<button type="button">Delete order</button>'
    "</form>"
)

_LOGIN_FORM = (
    "<form>"
    '<input type="email" placeholder="Email address" name="email"> '
    '<input type="password" placeholder="Password" name="password"> '
    "<button>Sign in</button>"
    "</form>"
)


@dataclasses.dataclass
class Fixture:
    name: str
    goal: str
    html: str
    base: str
    expect_any: list[str]


FIXTURES = [
    Fixture(
        "example-more-info",
        "learn more about example domains",
        _page(
            "Example Domain",
            "<h1>Example Domain</h1>"
            "<p>This domain is for use in illustrative examples in documents.</p>"
            '<p><a href="https://www.iana.org/domains/example">More information...</a></p>',
        ),
        CLICK_BASE,
        ["More information"],
    ),
    Fixture("nav-pricing", "open the pricing page", _page("Welcome", _NAV), CLICK_BASE, ["Pricing"]),
    Fixture("nav-about", "read about the company", _page("Welcome", _NAV), CLICK_BASE, ["About"]),
    Fixture(
        "order-submit",
        "place the order",
        _page("Checkout", f"<h1>Checkout</h1>{_ORDER_FORM}"),
        CLICK_BASE,
        ["Submit order"],
    ),
    Fixture(
        "order-cancel",
        "cancel and go back",
        _page("Checkout", f"<h1>Checkout</h1>{_ORDER_FORM}"),
        CLICK_BASE,
        ["Cancel"],
    ),
    Fixture(
        "hero-signup",
        "create a new account",
        _page(
            "Build faster",
            '<header><a href="/login">Log in</a></header>'
            "<section><h1>Build faster</h1><p>Ship your project in minutes.</p>"
            '<a class="cta" href="/signup">Sign up free</a></section>',
        ),
        CLICK_BASE,
        ["Sign up free", "/signup"],
    ),
    Fixture(
        "docs-changelog",
        "see what changed between versions",
        _page(
            "Documentation",
            "<h1>Documentation</h1><ul>"
            '<li><a href="/docs/install">Installation</a></li>'
            '<li><a href="/docs/usage">Usage</a></li>'
            '<li><a href="/docs/api">API Reference</a></li>'
            '<li><a href="/docs/changelog">Changelog</a></li>'
            "</ul>",
        ),
        CLICK_BASE,
        ["Changelog"],
    ),
    Fixture(
        "close-dialog",
        "close the dialog without saving",
        _page(
            "Settings",
            '<div class="dialog"><h2>Settings</h2><p>Preferences panel</p>'
            '<button aria-label="Close dialog">×</button>'
            '<button aria-label="Save preferences">Save</button></div>',
        ),
        CLICK_BASE,
        ["Close dialog"],
    ),
    Fixture(
        "icon-settings",
        "open account settings",
        _page(
            "Dashboard",
            '<header><a href="/">⌂</a> <a href="/settings">⚙</a> <a href="/mail">✉</a></header>'
            "<h1>Dashboard</h1><p>Overview of your workspace.</p>",
        ),
        CLICK_BASE,
        ["settings"],
    ),
    Fixture(
        "buy-any-widget",
        "buy a widget",
        _page(
            "Shop",
            "<h1>Shop</h1>"
            "<div><button>Buy</button><p>Widget A, blue</p></div>"
            "<div><button>Buy</button><p>Widget B, red</p></div>",
        ),
        CLICK_BASE,
        ["Buy"],
    ),
    Fixture(
        "pagination-next",
        "go to the next page of results",
        _page(
            "Search results",
            "<h1>Search results</h1><p>results 11-20 of 340</p>"
            '<div><a href="/s?page=1">‹ Prev</a> <a href="/s?page=3">Next ›</a></div>',
        ),
        CLICK_BASE,
        ["Next"],
    ),
    Fixture(
        "cookie-accept",
        "accept all cookies",
        _page(
            "News",
            "<h1>News</h1><p>Today's headlines from around the region.</p>"
            '<div class="consent"><p>We use cookies</p>'
            "<button>Settings</button> <button>Reject all</button> <button>Accept all</button></div>",
        ),
        CLICK_BASE,
        ["Accept all"],
    ),
    Fixture(
        "plan-team",
        "select the Team plan",
        _page(
            "Pricing",
            "<h1>Pricing</h1>"
            "<div><h2>Free</h2><button>Choose Free</button></div>"
            "<div><h2>Pro</h2><button>Choose Pro</button></div>"
            "<div><h2>Team</h2><button>Choose Team</button></div>",
        ),
        CLICK_BASE,
        ["Team"],
    ),
    Fixture(
        "download-report",
        "download the quarterly report",
        _page(
            "Q3 Report",
            "<h1>Q3 Report</h1><p>Long analysis of the quarter's results and outlook.</p>"
            "<button>Share</button> <button>Print</button> "
            '<a href="/reports/q3.pdf">Download report (PDF)</a>',
        ),
        CLICK_BASE,
        ["Download report"],
    ),
    Fixture(
        "login-email",
        "enter the email address",
        _page("Sign in", f"<h1>Sign in</h1>{_LOGIN_FORM}"),
        TYPE_BASE,
        ["Email address"],
    ),
    Fixture(
        "login-password",
        "type the password",
        _page("Sign in", f"<h1>Sign in</h1>{_LOGIN_FORM}"),
        TYPE_BASE,
        ["Password"],
    ),
    Fixture(
        "search-products",
        "search for running shoes",
        _page(
            "Store",
            "<h1>Store</h1>"
            '<input type="search" placeholder="Search products"> '
            '<input type="email" placeholder="Newsletter email"> '
            "<button>Subscribe</button>",
        ),
        TYPE_BASE,
        ["Search products"],
    ),
    Fixture(
        "comment-box",
        "write a comment",
        _page(
            "Article",
            "<h1>Article</h1><p>Body text of the article being discussed.</p>"
            '<input placeholder="Your name"> '
            '<textarea placeholder="Write a comment"></textarea> '
            "<button>Post</button>",
        ),
        TYPE_BASE,
        ["Write a comment"],
    ),
]


async def main() -> int:
    settings = dataclasses.replace(load_settings(), laya_pick_enabled=True)
    floor = settings.laya_min_confidence
    gate = LayaGate(settings)

    rows: list[tuple[str, str, float, bool, bool]] = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        for fx in FIXTURES:
            await page.set_content(fx.html)
            cands = await extract_candidates(
                page, fx.base, settings.laya_max_candidates,
                mode="type" if fx.base == TYPE_BASE else "click",
            )
            lines = cands.lines
            idx, conf = await gate.choose(
                QUESTION, lines, f"Goal: {fx.goal}\n{await page_state_text(page, 400)}"
            )
            if idx is None or not 0 <= idx < len(lines):
                rows.append((fx.name, "<no pick>", conf, False, conf >= floor))
                continue
            picked_line = lines[idx]
            correct = any(marker in picked_line for marker in fx.expect_any)
            rows.append((fx.name, picked_line, conf, correct, conf >= floor))
        await browser.close()

    acted = [r for r in rows if r[4]]
    acted_correct = sum(1 for r in acted if r[3])
    correct_all = sum(1 for r in rows if r[3])
    no_pick = sum(1 for r in rows if r[1] == "<no pick>")
    total = len(rows)

    print(f"\nbackend: {gate.summary}")
    print(f"confidence floor (LAYA_MIN_CONFIDENCE): {floor}")
    print(f"{'fixture':<18} {'conf':>5}  {'acted':<5} correct  picked line")
    for name, line, conf, correct, act in rows:
        print(f"{name:<18} {conf:5.3f}  {str(act):<5} {str(correct):<7} {line[:70]}")
    print(f"\ntotal {total} | correct {correct_all} ({correct_all / total:.0%}) | no-pick {no_pick}")
    if acted:
        print(f"acted (>= floor): {len(acted)} | acted-correct {acted_correct} "
              f"(precision {acted_correct / len(acted):.0%})")
    else:
        print("acted: none")

    ok = acted and acted_correct == len(acted) and len(acted) >= 0.8 * total and no_pick <= 2
    print(f"\nverdict: {'PASS — picker may be enabled' if ok else 'FAIL — keep the picker off'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
