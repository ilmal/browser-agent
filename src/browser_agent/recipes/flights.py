"""The flights.search recipe: Google Flights, deterministically, no LLM calls.

The instruction is parsed *locally* into (origin, destination, month, trip
length). A fixed-date round trip is encoded as a ``tfs`` protobuf and loaded
directly — one page load, no typing, no autocomplete, no date picker. The
"Date grid" tab then exposes the whole month as ``aria-label`` text carrying
``price, tag, "Jan D to Jan D"`` per cell, so the cheapest fare *for the
requested trip length* is a local minimum over a list of strings.

Why this shape (measured live 2026-09-22; see the ``browser-agent`` skill's
``references/google-flights.md``):

* The month's cheapest is **not** the globally cheapest cell. That cell belongs
  to whatever trip length is cheapest overall — a 4–5 day trip — which is
  exactly the bug that made the agent answer 5,459 for an 8-day question. It is
  the cheapest cell whose trip length matches the instruction.
* ``tfs`` encodes origin/destination and the date directly; the page pre-fills
  the form from it, so a single Search click runs the search.
* The grid is anchored on both dates of the loaded ``tfs``: it returns outbound
  dates within ±3 days of the departure and return dates within ±3 days of the
  return, which is what makes a 7-day window of trip lengths 2–14 fall out of
  one page load.

Failure semantics match the rest of the repo: anything unparseable or absent
raises :class:`~browser_agent.plan_model.StepFailure`, which the runner turns
into the agent fallback briefed with the original instruction.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from datetime import date, timedelta
from typing import Any

from ..activity import activity_of
from ..browser import BrowserSession
from ..plan_model import StepFailure
from ..tasks import register_builtin
from ._config import cfg

log = logging.getLogger(__name__)

#: Consent cookie — without it the EU interstitial returns a page with no
#: results at all. Set on the context before the first navigation.
SOCS = "CAESHAgCEhJnd3MfMjAyMzA4MTAtMF9SQzIaAmVuIAEaBgiA_LyaBg"

#: The fixed "flights and prices, no filters" request constant.
TFU = "EgQIABABIgA"

_SEARCH_URL = "https://www.google.com/travel/flights?tfs={tfs}&tfu=" + TFU
_SEARCH_BUTTONS = (
    'button[aria-label="Search for flights"]',
    'button[aria-label="Search"]',
)
_DATE_GRID_TABS = (
    'button:has-text("Date grid")',
    '[role="tab"]:has-text("Date grid")',
)
_ACCEPT_BUTTONS = ('button:has-text("Accept all")', 'button:has-text("Accept")')

#: Departure-day anchors; the grid covers ±3 days around each, so these tile a
#: whole month with one page load each. Five is the minimum for a 31-day month.
_ANCHORS = (4, 11, 18, 25, 29)

#: A date-grid cell: "SEK 5,071, cheapest price, Jan 14 to Jan 22, selected".
_CELL_RE = re.compile(
    r"SEK[\s ]*([\d,]+).*?\b([A-Z][a-z]{2})\s+(\d{1,2})\s+to\s+([A-Z][a-z]{2})\s+(\d{1,2})"
)

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MONTH_WORDS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

#: Instruction place names to the codes the transport understands: a 3-letter
#: IATA airport, or an ``/m/`` city kgmid. Extend as needed.
PLACES = {
    "stockholm": "/m/06mxs",
    "arlanda": "ARN",
    "arn": "ARN",
    "seattle": "/m/04jr0",
    "sea": "SEA",
    "gothenburg": "/m/04d0k",
    "copenhagen": "/m/01lrt",
    "oslo": "/m/05dq_",
    "london": "/m/04jpl",
    "new york": "/m/02_286",
    "nyc": "/m/02_286",
    "paris": "/m/05qtj",
    "tokyo": "/m/07dfk",
    "bangkok": "/m/0h8my",
    "palma": "/m/04lz2",
}


# --------------------------------------------------------------------------
# tfs encoding — the whole URL-addressable search
# --------------------------------------------------------------------------

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _field(num: int, wire: int, payload: bytes) -> bytes:
    tag = _varint((num << 3) | wire)
    return tag + (_varint(len(payload)) + payload if wire == 2 else payload)


def _endpoint(code: str) -> bytes:
    """Origin/destination: airport as {1:1, 2:code}, city as {1:3, 2:kgmid}."""
    kind = 1 if len(code) == 3 and code.isalpha() else 3
    return _field(1, 0, _varint(kind)) + _field(2, 2, code.encode())


def _leg(dep: str, origin: str, dest: str) -> bytes:
    """One FlightData: {2: ISO date, 13: origin, 14: destination}."""
    return (
        _field(2, 2, dep.encode())
        + _field(13, 2, _endpoint(origin))
        + _field(14, 2, _endpoint(dest))
    )


def encode_tfs(dep: str, ret: str, origin: str, dest: str) -> str:
    """A round-trip ``tfs`` for two fixed dates, base64url (no padding).

    The minimum that returns real results: trip type, the two legs, and the
    trailing sentinel/echo fields. ``Airport.flag``, passengers and seat are all
    optional — verified byte-consistent with a capture from the real form.
    """
    body = (
        _field(1, 0, _varint(28))
        + _field(2, 0, _varint(1))
        + _field(3, 2, _leg(dep, origin, dest))
        + _field(3, 2, _leg(ret, dest, origin))
        + _field(8, 0, _varint(1))
        + _field(9, 0, _varint(1))
        + _field(14, 0, _varint(1))
        + _field(16, 2, _field(1, 0, _varint(2**64 - 1)))
        + _field(19, 0, _varint(1))
    )
    return base64.urlsafe_b64encode(body).decode().rstrip("=")


# --------------------------------------------------------------------------
# instruction parsing — pure, so it is cheap to test
# --------------------------------------------------------------------------

def parse_trip_length(text: str) -> int | None:
    """Days of vacation from prose. None if absent — the caller must not invent.

    "8 day vacation" -> 8; "8-14 days" -> 8; "1 week" -> 7; "two weeks" -> 14.
    """
    t = text.lower()
    # The range first: "4-8 days" is the low end, and a single-number pattern
    # with an optional dash would otherwise match the "8 days" inside it.
    m = re.search(r"(\d{1,2})\s*[\-–]\s*(\d{1,2})\s*(?:day|night)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{1,2})\s*[-–]?\s*(?:day|night)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{1,2})\s*week", t)
    if m:
        return int(m.group(1)) * 7
    for word, n in (("one", 1), ("two", 2), ("three", 3), ("four", 4)):
        if re.search(rf"\b{word}\s+week", t):
            return n * 7
    return None


def resolve_month(text: str, today: date) -> tuple[int, int]:
    """The (year, month) the instruction refers to. An ISO ``2027-01`` wins;
    otherwise a named month means its next future occurrence."""
    m = re.search(r"\b(20\d{2})-(\d{2})\b", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    for word, mon in _MONTH_WORDS.items():
        if re.search(rf"\b{word[:3]}", text.lower()):
            year = today.year + (1 if (mon, 1) <= (today.month, 1) else 0)
            return year, mon
    raise StepFailure(f"no month in the instruction: {text!r}")


def resolve_place(name: str) -> str:
    """An instruction's place name to a GF code. Unknown -> StepFailure."""
    key = name.strip().lower()
    if key in PLACES:
        return PLACES[key]
    if re.fullmatch(r"[a-z]{3}", key):
        return key.upper()
    raise StepFailure(f"unknown place {name!r} — add it to PLACES")


def parse_market(text: str) -> tuple[str | None, str | None]:
    """(origin, destination) instruction names, or (None, None).

    Handles "from X to Y", "X to Y", "X-Y". Deliberately greedy only up to the
    preposition that ends the phrase, so "from stockholm to seattle in january"
    yields ("stockholm", "seattle").
    """
    m = re.search(
        r"from\s+([A-Za-zÀ-ɏ ]+?)\s+to\s+([A-Za-zÀ-ɏ ]+?)"
        r"(?:\s+in\b|\s+for\b|\s*,|\s*$)",
        text, re.I,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"\b([A-Za-zÀ-ɏ]+)\s+to\s+([A-Za-zÀ-ɏ]+)\b", text, re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


def parse_cells(labels: list[str], year: int) -> list[tuple[int, date, date, int]]:
    """Date-grid aria-labels to (price, depart, return, days).

    Labels that do not match the cell shape are skipped rather than raising: the
    same page carries unrelated ``SEK`` labels (the currency button, card price
    nodes) and one of them must not cost the search.
    """
    out: list[tuple[int, date, date, int]] = []
    for lbl in labels:
        m = _CELL_RE.search(lbl or "")
        if not m:
            continue
        try:
            price = int(m.group(1).replace(",", ""))
            dep = date(year, MONTHS[m.group(2).lower()], int(m.group(3)))
            ret = date(year, MONTHS[m.group(4).lower()], int(m.group(5)))
        except (KeyError, ValueError):
            continue
        days = (ret - dep).days
        if days > 0:
            out.append((price, dep, ret, days))
    return out


def pick_cheapest(
    cells: list[tuple[int, date, date, int]], trip_length: int | None
) -> tuple[int, date, date, int] | None:
    """The cheapest cell for the requested trip length (None = any length).

    Ties break on the earlier departure, which the real grid usually ties
    anyway — the Jan-2027 ARN->SEA 8-day minimum is 5,071 on four dates.
    """
    pool = cells if trip_length is None else [c for c in cells if c[3] == trip_length]
    return min(pool, key=lambda c: (c[0], c[1])) if pool else None


# --------------------------------------------------------------------------
# the page flow
# --------------------------------------------------------------------------

async def _accept_consent(page: Any) -> None:
    for sel in _ACCEPT_BUTTONS:
        el = page.locator(sel).first
        try:
            if await el.count() and await el.is_visible():
                await el.click(timeout=5000)
                await page.wait_for_timeout(2500)
                return
        except Exception:
            continue


async def _click_first(page: Any, selectors: tuple[str, ...], last: bool = False) -> bool:
    for sel in selectors:
        el = page.locator(sel)
        el = el.last if last else el.first
        try:
            if await el.count():
                await el.click(timeout=8000)
                return True
        except Exception:
            continue
    return False


async def _open_search(session: BrowserSession, tfs: str) -> Any:
    """Load one ``tfs``, dismiss consent, click Search, wait for prices."""
    page = await session.goto(_SEARCH_URL.format(tfs=tfs))
    await page.wait_for_timeout(2500)
    # Distinguish the rate wall from a page that merely has no results: every
    # later symptom ("no Search button", "no prices") is the same wall wearing a
    # different mask, and naming it is what tells an operator to back off rather
    # than to go looking for a broken selector. Seen live 2026-09-22 after ~20
    # probes from one IP.
    if "/sorry/" in page.url:
        raise StepFailure("google is rate-limiting this IP (/sorry/ wall) — back off")
    await _accept_consent(page)
    await page.wait_for_timeout(2500)
    if "/sorry/" in page.url:
        raise StepFailure("google is rate-limiting this IP (/sorry/ wall) — back off")
    if not await _click_first(page, _SEARCH_BUTTONS, last=True):
        raise StepFailure("no Search button on the flights page")
    for _ in range(10):
        await page.wait_for_timeout(3000)
        if re.search(r"SEK[\s ]*[\d,]{3,9}", await page.inner_text("body")):
            return page
    raise StepFailure("no prices rendered after Search")


async def _grid_cells(session: BrowserSession, tfs: str, log_: Any) -> list[str]:
    """One anchor's whole Date grid, as raw ``aria-label`` strings."""
    page = await _open_search(session, tfs)
    if not await _click_first(page, _DATE_GRID_TABS):
        raise StepFailure("no Date grid tab (results may be an error page)")
    await page.wait_for_timeout(6000)
    labels = await page.evaluate(
        """() => [...document.querySelectorAll('[aria-label]')]
             .map(e => e.getAttribute('aria-label'))
             .filter(a => a && a.includes('SEK') && a.includes(' to '))"""
    )
    log_.note("info", f"date grid: {len(labels or [])} cells")
    return list(labels or [])


async def _confirm(session: BrowserSession, tfs: str) -> str | None:
    """Load the winning pair and return the cheapest card's itinerary line.

    The grid price is the fare; the card is what carries the airline, the times
    and the stop count, which is what makes the answer checkable by hand.
    """
    page = await _open_search(session, tfs)
    card = page.locator("li").filter(has_not_text=" ").first
    el = page.locator("li").filter(has_text="SEK").first
    try:
        if await el.count():
            txt = await el.inner_text()
            if "SEK" in txt:
                return " | ".join(p.strip() for p in txt.split("\n") if p.strip())[:400]
    except Exception:
        pass
    return None


class FlightSearch:
    name = "flights.search"
    description = (
        "Google Flights: parse the instruction locally, drive the search from a "
        "tfs URL, read the month's Date grid, report the cheapest trip."
    )
    reads_instruction = True

    @property
    def entry_url(self) -> str:
        return cfg("flights.search", "entry_url", "https://www.google.com/travel/flights")

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        task_text = str(payload.get("task") or payload.get("text") or "").strip()
        if not task_text:
            raise StepFailure("flights.search needs a 'task' in the payload")
        log_ = activity_of(session)

        o_name, d_name = parse_market(task_text)
        if not o_name or not d_name:
            raise StepFailure(f"could not read origin/destination from {task_text!r}")
        origin = resolve_place(o_name)
        dest = resolve_place(d_name)
        trip = parse_trip_length(task_text)
        year, month = resolve_month(task_text, date.today())
        log_.note("info", f"{origin} -> {dest} {year}-{month:02d}, trip={trip}d")

        # The consent cookie is what makes the results appear at all.
        try:
            await session.context.add_cookies(
                [{"name": "SOCS", "value": SOCS, "domain": ".google.com", "path": "/"}]
            )
        except Exception as exc:  # a stand-in context in a test, or a closed one
            log_.note("warn", f"could not set SOCS: {exc}")

        nights = trip or 8
        cells: list[tuple[int, date, date, int]] = []
        walled = 0
        for anchor in _ANCHORS:
            dep = date(year, month, anchor)
            ret = dep + timedelta(days=nights)
            tfs = encode_tfs(dep.isoformat(), ret.isoformat(), origin, dest)
            try:
                cells += parse_cells(await _grid_cells(session, tfs, log_), year)
            except StepFailure as exc:
                log_.note("warn", f"anchor {dep}: {exc}")
                if "/sorry/" in str(exc):
                    walled += 1
            # One page load per anchor is five loads for a month; a short gap
            # keeps that under Google's published 10 req/s and reads as a person.
            await asyncio.sleep(cfg("flights.search", "anchor_gap_s", 2.5))
        if not cells:
            if walled:
                raise StepFailure(
                    f"google rate-limited every anchor ({walled}/{len(_ANCHORS)})"
                )
            raise StepFailure(f"no Date grid cells for {year}-{month:02d}")

        best = pick_cheapest(cells, trip)
        if best is None:
            raise StepFailure(f"no {trip}-day trip in the grid for {year}-{month:02d}")
        price, depart, ret, days = best
        tied = sorted({c[0] for c in cells if c[3] == days})[:4]

        detail = None
        try:
            detail = await _confirm(session, encode_tfs(
                depart.isoformat(), ret.isoformat(), origin, dest))
        except Exception as exc:
            log_.note("warn", f"confirm failed (grid number stands): {exc}")

        text = (
            f"Cheapest {days}-day round trip {o_name.title()} -> {d_name.title()} in "
            f"{depart.strftime('%B %Y')}: SEK {price:,}, departing {depart} and "
            f"returning {ret}."
        )
        if detail:
            text += f" Itinerary: {detail}"
        text += (
            f" (Lowest for a {days}-day trip: {', '.join(f'SEK {p:,}' for p in tied)}.)"
        )
        return {
            "ok": True,
            "text": text,
            "price": price,
            "currency": "SEK",
            "depart": depart.isoformat(),
            "return": ret.isoformat(),
            "days": days,
            "origin": origin,
            "destination": dest,
            "itinerary": detail,
            "cells": len(cells),
        }


register_builtin(FlightSearch())
