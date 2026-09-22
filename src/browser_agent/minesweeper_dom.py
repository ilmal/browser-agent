"""The minesweeper.online DOM bridge: read the board, click like a person.

Split from :mod:`~browser_agent.minesweeper_solver` on purpose. The solver is
pure logic with no browser in it; everything that touches the page — the read,
the click, the pacing, the win/lose signal — is here, so the solver's tests
never need a browser and this layer's single job is to be a faithful,
un-clever adapter.

Three site facts shape all of it (verified live 2026-09-22):

* **The digit is a CSS class, not text.** An opened cell carries
  ``hd_opened hd_type<N>`` and has empty ``innerText`` *and* ``innerHTML``, so
  the number can only be read from the class. ``hd_type10`` is a revealed
  mine, ``hd_type11`` the one that was just clicked.
* **Ids are ``cell_<x>_<y>``** — column first, then row — under ``#CellsBlock``.
* **The site UA-gates its JS.** A default Playwright UA gets a script-less
  shell with zero cells, so the caller must present a real Chrome UA. That is
  the browser's job, not this module's, but a zero-cell read is the symptom.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from playwright.async_api import Page

from .minesweeper_solver import Board

log = logging.getLogger(__name__)

#: Beginner board: 9x9 with 10 mines. The recipe only plays Beginner — a bigger
#: board is more clicks, and the point is a fast, low-volume game.
BEGINNER_ROWS = 9
BEGINNER_COLS = 9
BEGINNER_MINES = 10

#: Boot test. The site renders 81 cells for Beginner and nothing until its
#: script has run, so this is both "is it loaded" and "did the UA gate us".
_EXPECTED_CELLS = BEGINNER_ROWS * BEGINNER_COLS

#: One evaluate for the whole board: per-cell state token, plus the game result
#: and the blocked flag. Keyed on the id so a partially-rendered board reads as
#: ``?`` rather than shifting every row.
_READ_JS = """(dims) => {
  const cells = {};
  for (const c of document.querySelectorAll('[id^=cell_]')) cells[c.id] = c;
  const grid = [];
  for (let y = 0; y < dims.rows; y++) {
    const row = [];
    for (let x = 0; x < dims.cols; x++) {
      const c = cells[`cell_${x}_${y}`];
      if (!c) { row.push('?'); continue; }
      const cls = String(c.className);
      const t = (cls.match(/hd_type(\\d+)/) || [])[1];
      // Flag first: a flagged cell also carries hd_closed, and hd_opened is
      // never present with hd_flag, so this order cannot mislabel either.
      row.push(/hd_flag/.test(cls)   ? 'F'
             : /hd_opened/.test(cls) ? (t === '10' || t === '11' ? 'M' : (t || '0'))
             : '.');
    }
    grid.push(row);
  }
  const face = document.querySelector('#top_area_face');
  return {
    grid,
    face: face ? String(face.className) : '',
    cells: Object.keys(cells).length,
    blocked: !!document.querySelector('#UserBlockedBlock'),
    status: (document.body.innerText || '').slice(0, 200).replace(/\\s+/g, ' ').trim(),
  };
}"""


@dataclass(frozen=True)
class GameView:
    """One read of the page: the board, plus what the chrome says."""

    board: Board
    face: str
    n_cells: int
    blocked: bool
    status: str

    @property
    def cells_ready(self) -> bool:
        return self.n_cells == _EXPECTED_CELLS

    @property
    def won(self) -> bool:
        return "face-win" in self.face

    @property
    def lost(self) -> bool:
        return "face-lose" in self.face

    @property
    def over(self) -> bool:
        return self.won or self.lost


async def read_game(page: Page, *, mines: int = BEGINNER_MINES) -> GameView:
    """Read the whole board and the game chrome in one round trip."""
    raw = await page.evaluate(_READ_JS, {"rows": BEGINNER_ROWS, "cols": BEGINNER_COLS})
    board = Board.from_rows(raw["grid"], mines=mines)
    return GameView(
        board=board,
        face=raw.get("face") or "",
        n_cells=int(raw.get("cells") or 0),
        blocked=bool(raw.get("blocked")),
        status=raw.get("status") or "",
    )


async def wait_for_board(page: Page, *, tries: int = 30, gap_ms: int = 1000) -> GameView:
    """Poll until the 81 cells render, or give up and return the last read.

    The board appears only after the site's script runs, so a read taken at
    ``domcontentloaded`` is legitimately empty. Returning the empty view rather
    than raising lets the caller tell "not loaded yet" from "blocked" — the
    two need different handling.
    """
    view = await read_game(page)
    for _ in range(tries):
        if view.cells_ready or view.blocked:
            return view
        await page.wait_for_timeout(gap_ms)
        view = await read_game(page)
    return view


@dataclass
class Pace:
    """Human-ish timing for every interaction.

    The ban this defends against keys on interaction volume and machine
    regularity, so two properties matter and both are deliberate: a randomised
    gap (never a fixed sleep) and a hover-then-pause before each press, which
    is what a hand does and what a scripted ``dispatchEvent`` does not. The
    click itself is Playwright's real trusted input — never a synthetic event,
    which the site can and does distinguish.
    """

    min_ms: int = 220
    max_ms: int = 700
    #: Pause between moving onto a cell and pressing it.
    hover_ms: int = 90

    def _gap(self) -> float:
        return random.uniform(self.min_ms, self.max_ms) / 1000.0

    async def settle(self, page: Page) -> None:
        """Wait a random beat, as a person would between decisions."""
        await page.wait_for_timeout(int(self._gap() * 1000))

    async def click(self, page: Page, selector: str, *, button: str = "left") -> None:
        """Hover, pause, then press — one real trusted click."""
        loc = page.locator(selector).first
        await loc.scroll_into_view_if_needed(timeout=5000)
        await loc.hover(timeout=5000)
        if self.hover_ms:
            await page.wait_for_timeout(self.hover_ms)
        await loc.click(button=button, timeout=8000, delay=random.randint(30, 90))
        await self.settle(page)


def cell_selector(row: int, col: int) -> str:
    """The site's id is ``cell_<x>_<y>`` — column first."""
    return f"#cell_{col}_{row}"


async def click_cell(page: Page, row: int, col: int, *, pace: Pace, flag: bool = False) -> None:
    """Open (left) or flag (right) one cell with real input and human pacing."""
    await pace.click(page, cell_selector(row, col), button="right" if flag else "left")


async def start_beginner(page: Page, *, timeout_ms: int = 60_000) -> GameView:
    """Navigate to a fresh Beginner game and wait for its board."""
    await page.goto("https://minesweeper.online/new-game",
                    wait_until="domcontentloaded", timeout=timeout_ms)
    return await wait_for_board(page)


async def is_blocked(page: Page) -> bool:
    """Cheap re-check for the IP block, which arrives after boot, not at it."""
    try:
        return await page.evaluate("() => !!document.querySelector('#UserBlockedBlock')")
    except Exception:
        return False
