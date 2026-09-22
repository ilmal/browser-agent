"""Minesweeper: a whole game driven by the solver, with Laya only as a tiebreak.

The division of labour this recipe exists to prove:

* **The solver moves.** Every proved move is computed in-process from the board
  read — no model call, no planner, no LLM. A Beginner game is ~30-40 clicks.
* **The board is read once per decision, moved on, and re-read.** A batch of
  proved moves is applied together; a guess is applied alone and the board is
  re-read, because a wrong guess ends the game.
* **Laya is asked exactly one narrow question** — which of the two or three
  genuinely-ambiguous frontier cells is safest — and only when the solver has
  no proof at all. It is a tiebreak on a judgement call, not a driver. If the
  gate is off or unsure, the solver's own heuristic decides; the game never
  stops for a model.
* **Every press is real trusted input with randomised pacing** (see
  :class:`~browser_agent.minesweeper_dom.Pace`), so the run does not read as a
  bot. That is also why the click count is kept low: a Beginner board's worth
  of clicks is fine, hundreds of rapid ones is what gets an IP banned.
"""

from __future__ import annotations

import logging
from typing import Any

from ..activity import activity_of
from ..browser import BrowserSession
from ..config import load_settings
from ..escalation import Challenge, ChallengeKind, EscalationRequired
from ..laya_gate import LayaGate
from ..minesweeper_dom import (
    GameView,
    Pace,
    click_cell,
    read_game,
    start_beginner,
)
from ..minesweeper_solver import Move
from ..tasks import register_builtin
from ._config import cfg

log = logging.getLogger(__name__)

#: Boards to play before returning. One game proves the loop; more than a
#: couple is exactly the interaction volume the site bans for.
DEFAULT_GAMES = 1

#: The board's URL. Overridable because minesweeper.online moves its entry path
#: between game modes, and a moved URL must not be a code deploy.
DEFAULT_ENTRY_URL = "https://minesweeper.online/new-game"

#: A ceiling on clicks per game, so a solver bug cannot turn into a click storm.
#: A Beginner board needs ~40; 200 is generous headroom, not a target.
MAX_CLICKS_PER_GAME = 200

#: Ambiguous frontier cells worth asking Laya about. Three keeps the choice
#: inside the range the gate is calibrated for and keeps the question small.
_MAX_TIEBREAK_CANDIDATES = 3


class Minesweeper:
    name = "minesweeper.play"
    description = "Play a full Beginner game of minesweeper.online to a win, solver-driven."

    def __init__(self, laya: LayaGate | None = None, settings: Any = None,
                 pace: Pace | None = None) -> None:
        self._settings = settings if settings is not None else load_settings()
        # Constructed lazily-but-once, like plan.task: importing the module must
        # not build a model client, but a run should reuse one gate.
        self._laya = laya if laya is not None else LayaGate(self._settings)
        self._base_pace = pace if pace is not None else Pace()

    @property
    def entry_url(self) -> str:
        return cfg("minesweeper.play", "entry_url", DEFAULT_ENTRY_URL)

    @property
    def _pace(self) -> Pace:
        """The pacing, with the operator's overrides applied over the defaults.

        The gap is what the site's ban heuristics actually look at, so it is the
        one lever worth exposing — and with nothing overridden it is the exact
        ``Pace`` the recipe has always used.
        """
        return Pace(
            min_ms=cfg("minesweeper.play", "pace.min_ms", self._base_pace.min_ms),
            max_ms=cfg("minesweeper.play", "pace.max_ms", self._base_pace.max_ms),
            hover_ms=self._base_pace.hover_ms,
            jitter_px=self._base_pace.jitter_px,
            move_steps=self._base_pace.move_steps,
        )

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        games = int(payload.get("games") or DEFAULT_GAMES)
        games = max(1, min(games, 3))
        max_clicks = cfg("minesweeper.play", "max_clicks", MAX_CLICKS_PER_GAME)
        log_ = activity_of(session)

        page = await session.page()
        results: list[dict[str, Any]] = []

        for game_no in range(1, games + 1):
            log_.note("info", f"game {game_no}/{games}: starting a Beginner board")
            view = await start_beginner(page, url=self.entry_url)

            if view.blocked:
                # The IP block is app-side and arrives after boot. Retrying is
                # what makes it worse, so this stops and asks for a human.
                raise EscalationRequired(
                    Challenge(ChallengeKind.RATE_LIMITED,
                              "minesweeper.online blocked this egress IP (Account blocked)",
                              page.url)
                )
            if not view.cells_ready:
                raise RuntimeError(
                    f"board never rendered: {view.n_cells} cells "
                    f"(expected 81) — the site UA-gates its JS, so check the "
                    f"browser's user agent and the egress"
                )

            results.append(await self._play_game(page, view, log_, game_no, max_clicks))

            if results[-1]["outcome"] == "lost":
                # A loss is a legitimate outcome, but playing on after one
                # spends clicks on nothing. Stop the run and report.
                break

        wins = sum(1 for r in results if r["outcome"] == "won")
        log_.note("info", f"done: {wins}/{len(results)} won")
        return {"games": results, "wins": wins}

    async def _play_game(
        self, page: Any, view: GameView, log_: Any, game_no: int,
        max_clicks: int = MAX_CLICKS_PER_GAME,
    ) -> dict[str, Any]:
        clicks = 0
        guesses = 0
        laya_calls = 0
        # The first press on a fresh board is always a guess (no constraint
        # exists yet), and the site guarantees it opens an area rather than
        # losing, so it is played without ceremony.
        first_press = True

        while clicks < max_clicks:
            if view.won:
                log_.note("step", f"game {game_no}: won in {clicks} clicks")
                return {"outcome": "won", "clicks": clicks, "guesses": guesses,
                        "laya_calls": laya_calls}
            if view.lost:
                log_.note("error", f"game {game_no}: lost on a guess after {clicks} clicks")
                return {"outcome": "lost", "clicks": clicks, "guesses": guesses,
                        "laya_calls": laya_calls,
                        "board": view.board.render()}

            moves, proved = view.board.next_moves()
            if not moves:
                # No proof and no gamble left: the board is fully settled, which
                # only happens once every non-mine cell is open — i.e. a win the
                # face check should already have caught.
                log_.note("info", f"game {game_no}: no moves left")
                break

            if not proved:
                guesses += 1
                move, asked = await self._tiebreak(page, view, moves, log_, game_no)
                laya_calls += 1 if asked else 0
                if first_press:
                    # Opening the very first cell is the site's own free move;
                    # its "safest" ranking is meaningless before anything is
                    # visible, so take the centre-ish cell it picks and move on.
                    log_.note("info", f"game {game_no}: opening move")
                else:
                    log_.note(
                        "gate" if asked else "step",
                        f"game {game_no}: guess {move.row},{move.col} — {move.reason}",
                    )
                await click_cell(page, move.row, move.col, pace=self._pace)
                clicks += 1
                first_press = False
                view = await read_game(page)
                continue

            first_press = False
            # Proved moves are all safe from this one read, so they go out as a
            # batch. Flags first: a flag is what proves the next cell safe, and
            # marking before opening keeps the visible board consistent with the
            # solver's model if the read is interrupted.
            ordered = sorted(moves, key=lambda m: 0 if m.kind == "flag" else 1)
            for move in ordered:
                if clicks >= max_clicks:
                    break
                await click_cell(page, move.row, move.col,
                                 pace=self._pace, flag=move.kind == "flag")
                clicks += 1
            log_.note("step",
                      f"game {game_no}: applied {len(ordered)} proved moves "
                      f"({sum(1 for m in ordered if m.kind == 'flag')} flags)")
            view = await read_game(page)

        return {"outcome": "unfinished", "clicks": clicks, "guesses": guesses,
                "laya_calls": laya_calls, "board": view.board.render()}

    async def _tiebreak(
        self, page: Any, view: GameView, moves: list[Move], log_: Any, game_no: int
    ) -> tuple[Move, bool]:
        """Choose among ambiguous cells; ask Laya only if she is enabled.

        Laya is a *tiebreak*, so the question is deliberately narrow — among a
        few candidate cells, which is safest — and the answer is only taken
        when the gate is confident. Anything else (gate off, unsure, transport
        failure) leaves the solver's own risk ranking in charge. Laya never
        raises and never escalates; a bad call costs a guess, not a game, since
        the fallback is the same heuristic either way.
        """
        from ..minesweeper_solver import ranked_guesses

        ranked = ranked_guesses(view.board.grid, view.board.rows, view.board.cols,
                                view.board.mines_left)
        if not ranked:
            return moves[0], False

        # Only the cells nearest the frontier are ambiguous in any interesting
        # way; the rest are board-density fill and all rank the same.
        shortlist = ranked[:_MAX_TIEBREAK_CANDIDATES]
        # Gate on ``enabled``, not ``pick_enabled``: the latter protects the
        # *plan* picker, where a wrong pick clicks the wrong element. Here a
        # wrong guess costs one life in a game the solver can still win, so the
        # game is allowed to use the gate while the plan picker stays off.
        if len(shortlist) <= 1 or not self._laya.enabled:
            return moves[0], False

        lines = [
            f"row {r + 1}, column {c + 1} (risk about {p:.0%})"
            for (r, c), p in shortlist
        ]
        state = (
            f"{view.board.render()}\n"
            f"Minesweeper, {view.board.rows}x{view.board.cols}, "
            f"{view.board.mines_left} mines unmarked. "
            f"Nothing is provable; one of these cells must be opened."
        )
        idx, conf = await self._laya.choose(
            "Which numbered cell is the safest to open next?", lines, state
        )
        threshold = self._settings.laya_game_min_confidence
        if idx is None or conf < threshold:
            log_.note("gate", f"game {game_no}: tiebreak inconclusive (conf {conf:.2f})")
            return moves[0], True

        (r, c), p = shortlist[idx]
        log_.note("gate",
                  f"game {game_no}: laya chose {r + 1},{c + 1} "
                  f"(conf {conf:.2f}, risk {p:.0%})")
        return Move("open", r, c, f"laya tiebreak (conf {conf:.2f})"), True


register_builtin(Minesweeper())
