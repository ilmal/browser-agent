"""Minesweeper logic: a pure solver over a grid of cell states.

No browser, no LLM, no I/O, so every rule is unit-testable against hand-built
boards. The recipe reads the DOM into this shape, asks for moves, and writes
them back as real clicks. Keeping the reasoning here is what makes a game cost
~30 browser actions and zero model calls.

Two layers of deduction, in order:

1. **Single point.** An opened cell showing *n* with *n* closed neighbours means
   those neighbours are all mines; with *n* already flagged, every other closed
   neighbour is safe.
2. **Subset elimination.** If one cell's unknown set is contained in another's,
   the difference carries the difference in mine counts. This is what cracks the
   "1-2-1" and "1-2-2-1" walls that single-point logic cannot.

When neither applies the position is genuinely ambiguous, and :func:`guess`
returns the least dangerous cell — a heuristic, not a proof. The recipe routes
*that* decision through the gate rather than trusting the heuristic blindly.
"""

from __future__ import annotations

from dataclasses import dataclass

CLOSED = "closed"
OPENED = "opened"
FLAGGED = "flagged"

Coord = tuple[int, int]


@dataclass(frozen=True)
class Cell:
    state: str = CLOSED
    #: Mines among the eight neighbours. Only meaningful when ``state`` is OPENED.
    adjacent: int = 0


@dataclass(frozen=True)
class Move:
    kind: str  # "open" | "flag"
    row: int
    col: int
    reason: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.kind} ({self.row},{self.col}) — {self.reason}"


def make_board(rows: int, cols: int) -> list[list[Cell]]:
    return [[Cell() for _ in range(cols)] for _ in range(rows)]


@dataclass
class Board:
    """A grid plus its dimensions, and the counts the solver needs.

    Bundling the three keeps every call site from threading ``rows``/``cols``
    and re-tallying flags; the recipe builds one per DOM read and asks it for
    moves.
    """

    grid: list[list[Cell]]
    rows: int
    cols: int
    #: Total mines on the board (Beginner = 10), used for the density fallback
    #: in :func:`_probabilities` when no constraint covers a cell.
    mines: int

    @classmethod
    def from_rows(cls, rows: list[str], *, mines: int) -> Board:
        """Build from display rows: ``#`` closed, ``F`` flagged, ``M`` mine,
        a digit an opened cell, ``?`` unreadable (treated as closed)."""
        grid: list[list[Cell]] = []
        for row in rows:
            cells: list[Cell] = []
            for token in row:
                if token == "F":
                    cells.append(Cell(FLAGGED))
                elif token == "M":
                    # Only visible once the game is lost, and then the solve is
                    # over; reading it as an opened 10 keeps the tally honest
                    # without pretending it is a player-visible number.
                    cells.append(Cell(OPENED, 10))
                elif token.isdigit():
                    cells.append(Cell(OPENED, int(token)))
                else:
                    cells.append(Cell(CLOSED))
            grid.append(cells)
        return cls(grid=grid, rows=len(grid), cols=len(grid[0]) if grid else 0, mines=mines)

    @property
    def flagged(self) -> int:
        return sum(1 for row in self.grid for cell in row if cell.state == FLAGGED)

    @property
    def opened(self) -> int:
        return sum(1 for row in self.grid for cell in row if cell.state == OPENED)

    @property
    def mines_left(self) -> int:
        """Mines not yet accounted for by a flag. Never negative."""
        return max(0, self.mines - self.flagged)

    def is_won(self) -> bool:
        return is_won(self.grid, self.rows, self.cols, self.mines)

    def next_moves(self) -> tuple[list[Move], bool]:
        return next_moves(self.grid, self.rows, self.cols, self.mines_left)

    def render(self) -> str:
        return render(self.grid)


def neighbors(rows: int, cols: int, r: int, c: int) -> list[Coord]:
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                out.append((rr, cc))
    return out


def _unknown(grid: list[list[Cell]], rows: int, cols: int, r: int, c: int) -> set[Coord]:
    return {
        (rr, cc)
        for rr, cc in neighbors(rows, cols, r, c)
        if grid[rr][cc].state == CLOSED
    }


def _flagged(grid: list[list[Cell]], rows: int, cols: int, r: int, c: int) -> int:
    return sum(
        1
        for rr, cc in neighbors(rows, cols, r, c)
        if grid[rr][cc].state == FLAGGED
    )


def constraints(
    grid: list[list[Cell]], rows: int, cols: int
) -> list[tuple[frozenset[Coord], int]]:
    """One (unknown cells, mines among them) pair per informative opened cell."""
    out: list[tuple[frozenset[Coord], int]] = []
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell.state != OPENED:
                continue
            unknown = _unknown(grid, rows, cols, r, c)
            if not unknown:
                continue
            remaining = cell.adjacent - _flagged(grid, rows, cols, r, c)
            if remaining < 0:
                # More flags than the number allows: the board was misread, or a
                # flag is wrong. Say nothing rather than emit a bogus deduction.
                continue
            out.append((frozenset(unknown), remaining))
    return out


def _deduce(
    con: list[tuple[frozenset[Coord], int]],
) -> tuple[set[Coord], set[Coord]]:
    """Run both deduction layers to a fixpoint; return (safe, mines).

    The three phases have to interleave, not run once each: a mine proved by
    subset elimination shrinks a *neighbouring* constraint's unknown set, which
    can turn a "1 of 2" into a "0 of 1" and hand back a safe cell. That chaining
    is what solves a 1-2-1 wall, so the loop rewrites the constraint list with
    everything proved so far and goes round again until nothing changes.
    """
    con = list(con)
    safe: set[Coord] = set()
    mines: set[Coord] = set()

    # A guard, not the exit: the exit is "a full pass changed nothing".
    for _ in range(512):
        changed = False

        for unknown, remaining in con:
            if remaining <= 0:
                for coord in unknown:
                    if coord not in safe:
                        safe.add(coord)
                        changed = True
            elif remaining >= len(unknown):
                for coord in unknown:
                    if coord not in mines:
                        mines.add(coord)
                        changed = True

        rewritten: list[tuple[frozenset[Coord], int]] = []
        for unknown, remaining in con:
            known_mines = unknown & mines
            shrunk = frozenset(unknown - mines - safe)
            left = remaining - len(known_mines)
            if left < 0 or (not shrunk and left != 0):
                # A constraint that cannot hold means the board was misread.
                # Drop it rather than draw a conclusion from bad input.
                changed = True
                continue
            if not shrunk:
                continue
            if shrunk != unknown or left != remaining:
                changed = True
            rewritten.append((shrunk, left))
        con = rewritten

        additions: list[tuple[frozenset[Coord], int]] = []
        for i in range(len(con)):
            unknown_a, mines_a = con[i]
            for j in range(i + 1, len(con)):
                unknown_b, mines_b = con[j]
                if unknown_a < unknown_b:
                    diff, diff_mines = unknown_b - unknown_a, mines_b - mines_a
                elif unknown_b < unknown_a:
                    diff, diff_mines = unknown_a - unknown_b, mines_a - mines_b
                else:
                    continue
                if diff and all(diff != u for u, _ in con):
                    additions.append((frozenset(diff), diff_mines))
        if additions:
            con.extend(additions)
            changed = True

        if not changed:
            break

    # A cell can be proved safe by one constraint and mine by another only if the
    # reading was wrong; safety wins, because opening a cell is recoverable from
    # and flagging a non-mine silently corrupts the rest of the solve.
    return safe - mines, mines - safe


def logic_moves(grid: list[list[Cell]], rows: int, cols: int) -> list[Move]:
    """Every move the board proves right now. Empty means a guess is needed."""
    safe, mines = _deduce(constraints(grid, rows, cols))
    moves = [
        Move("flag", r, c, "all remaining neighbours are mines")
        for r, c in sorted(mines)
        if grid[r][c].state == CLOSED
    ]
    moves += [
        Move("open", r, c, "no mine left among its neighbours")
        for r, c in sorted(safe)
        if grid[r][c].state == CLOSED
    ]
    return moves


def _probabilities(
    grid: list[list[Cell]], rows: int, cols: int, mines_left: int
) -> dict[Coord, float]:
    """Rough mine probability per closed cell.

    Per constraint the chance is mines/unknown; a cell on several constraints
    takes the worst of them, which is the conservative choice — guessing is
    where games are lost, so overestimate danger rather than understate it.
    Cells on no constraint fall back to the board's average density.
    """
    closed = [
        (r, c) for r in range(rows) for c in range(cols) if grid[r][c].state == CLOSED
    ]
    probs = {coord: 0.0 for coord in closed}
    for unknown, remaining in constraints(grid, rows, cols):
        p = remaining / len(unknown)
        for coord in unknown:
            probs[coord] = max(probs[coord], p)
    if not closed:
        return probs
    density = max(0.0, mines_left) / len(closed)
    for coord, p in probs.items():
        if p == 0.0:
            probs[coord] = density
    return probs


def guess_key(
    grid: list[list[Cell]], rows: int, cols: int, coord: Coord, prob: float
) -> tuple:
    """The gamble-ranking key for one cell, lowest-is-best.

    Spelled out as its own function because it is a *policy*, not arithmetic:
    the tests assert it directly rather than inferring it from whichever move a
    fixture happens to produce. In order:

    1. lowest estimated mine probability,
    2. then the most opened neighbours — the cell most constrained by what is
       already visible, so surviving it unlocks the most of the board,
    3. then corners last, since a corner touches fewer cells,
    4. then row/column, purely so the choice is deterministic.
    """
    r, c = coord
    opened_neighbours = sum(
        1
        for rr, cc in neighbors(rows, cols, r, c)
        if grid[rr][cc].state == OPENED
    )
    corner = r in (0, rows - 1) and c in (0, cols - 1)
    return (prob, -opened_neighbours, 0 if corner else 1, r, c)


def ranked_guesses(
    grid: list[list[Cell]], rows: int, cols: int, mines_left: int
) -> list[tuple[Coord, float]]:
    """Closed, unsettled cells ordered best-guess-first, with their risk.

    Cells the logic has already settled are excluded, so every entry here is a
    genuine gamble.
    """
    safe, mines = _deduce(constraints(grid, rows, cols))
    settled = safe | mines
    probs = {
        coord: p
        for coord, p in _probabilities(grid, rows, cols, mines_left).items()
        if coord not in settled
    }
    return sorted(
        probs.items(),
        key=lambda item: guess_key(grid, rows, cols, item[0], item[1]),
    )


def guess(
    grid: list[list[Cell]], rows: int, cols: int, mines_left: int
) -> Move | None:
    """The least dangerous cell to open when logic is exhausted, or None.

    None means every closed cell is already settled by logic, so the caller has
    no gamble left to take — which only happens on a board where the deduction
    already decided everything.
    """
    ranked = ranked_guesses(grid, rows, cols, mines_left)
    if not ranked:
        return None
    (r, c), p = ranked[0]
    return Move("open", r, c, f"no proof available; least likely ({p:.0%})")


def next_moves(
    grid: list[list[Cell]], rows: int, cols: int, mines_left: int
) -> tuple[list[Move], bool]:
    """Moves to make now, and whether they are proved (False means a guess).

    Proved moves are returned as a batch: they are all safe to apply together,
    because each was derived from the board as it stands. A guess is returned
    alone, and the caller must re-read the board before deciding again.
    """
    moves = logic_moves(grid, rows, cols)
    if moves:
        return moves, True
    fallback = guess(grid, rows, cols, mines_left)
    return ([fallback] if fallback is not None else []), False


def is_won(grid: list[list[Cell]], rows: int, cols: int, mines: int) -> bool:
    """Every non-mine cell opened. Flags are cosmetic and deliberately ignored."""
    closed = sum(
        1 for r in range(rows) for c in range(cols) if grid[r][c].state == CLOSED
    )
    return closed == mines


def render(grid: list[list[Cell]]) -> str:
    """The board as text, for logs and test failures."""
    lines: list[str] = []
    for row in grid:
        cells = []
        for cell in row:
            if cell.state == FLAGGED:
                cells.append("F")
            elif cell.state == OPENED:
                cells.append(str(cell.adjacent) if cell.adjacent else ".")
            else:
                cells.append("#")
        lines.append(" ".join(cells))
    return "\n".join(lines)
