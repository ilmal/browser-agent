"""The local minesweeper board keeps the DOM contract the adapter was built on.

minesweeper.online IP-blocked the pod's egress (2026-09-25), so the recipe's
default site is now the agent's own /minesweeper.html. That only works if the
page really is the DOM minesweeper_dom.py reads — cell ids ``cell_<x>_<y>``,
digits as ``hd_type<N>`` classes, flags as ``hd_flag``, the face signalling
``face-win``/``face-lose``. These tests drive the real page in a real browser
through the adapter's own read, so a contract drift fails here and not in a
live game.
"""

from __future__ import annotations

import http.server
import socket
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent.minesweeper_dom import read_game  # noqa: E402
from browser_agent.minesweeper_solver import CLOSED, FLAGGED  # noqa: E402

_PAGE = (Path(__file__).resolve().parents[1] / "src" / "browser_agent" / "minesweeper_local.html")
_HTML = _PAGE.read_text()


def test_the_page_carries_the_dom_contract_as_plain_text():
    """A cheap string check so an accidental rewrite fails fast, no browser."""
    assert 'c.id = "cell_" + x + "_" + y' in _HTML, "cells are cell_<x>_<y>, column first"
    assert "ROWS = 9" in _HTML and "COLS = 9" in _HTML and "MINES = 10" in _HTML
    assert 'id="top_area_face"' in _HTML
    assert "face-win" in _HTML and "face-lose" in _HTML
    assert "hd_opened" in _HTML and "hd_flag" in _HTML and "hd_type11" in _HTML
    assert "UserBlockedBlock" not in _HTML, "nothing blocks a local page"


@pytest.fixture(scope="module")
def local_site():
    """Serve the page the way the agent's API serves it."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/minesweeper.html"
    server.shutdown()


async def _open(pw, url):
    browser = await pw.chromium.launch(args=["--no-sandbox"])
    try:
        ctx = await browser.new_context()
        page = await ctx.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        return page
    finally:
        await browser.close()


async def _open_all_safe(page):
    """Click every non-mine cell; the first click starts the (seeded) game.

    Mines are placed lazily on the first left click, so one click must land
    before the mine set can be read at all.
    """
    await page.click("#cell_4_4")
    mines = set(await page.evaluate("() => window.__msDebug.mines()"))
    assert mines, "the first click placed the mines"
    for y in range(9):
        for x in range(9):
            if y * 9 + x in mines:
                continue
            await page.click(f"#cell_{x}_{y}")
    return mines


@pytest.mark.asyncio
async def test_a_seeded_board_reads_ready_and_first_click_opens_safely(local_site):
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        try:
            page = await browser.new_page()
            await page.goto(f"{local_site}?seed=7", wait_until="domcontentloaded")
            view = await read_game(page)
            assert view.cells_ready and not view.blocked
            assert not view.over, "a fresh board is neither won nor lost"
            assert all(cell.state == CLOSED for row in view.board.grid for cell in row)

            await page.click("#cell_4_4")
            view = await read_game(page)
            assert any(cell.state != CLOSED for row in view.board.grid for cell in row), (
                "the first click opens at least its own cell and never loses"
            )
            assert not view.over
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_playing_every_safe_cell_wins_and_a_mine_loses(local_site):
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        try:
            page = await browser.new_page()
            await page.goto(f"{local_site}?seed=3", wait_until="domcontentloaded")
            await _open_all_safe(page)
            view = await read_game(page)
            assert view.won, "opening every safe cell is the win condition"

            page2 = await browser.new_page()
            await page2.goto(f"{local_site}?seed=3", wait_until="domcontentloaded")
            await page2.click("#cell_4_4")
            mines = set(await page2.evaluate("() => window.__msDebug.mines()"))
            assert mines, "the first click placed the mines"
            first = next(iter(mines))
            mine_x, mine_y = first % 9, first // 9
            await page2.click(f"#cell_{mine_x}_{mine_y}")
            view = await read_game(page2)
            assert view.lost, "clicking a mine is the loss condition"
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_a_right_click_flags_and_the_adapter_reads_an_f(local_site):
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--no-sandbox"])
        try:
            page = await browser.new_page()
            await page.goto(f"{local_site}?seed=3", wait_until="domcontentloaded")
            await page.click("#cell_0_0", button="right")
            view = await read_game(page)
            assert view.board.grid[0][0].state == FLAGGED, "the flag is read from hd_flag"
        finally:
            await browser.close()
