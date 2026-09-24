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
import re
from datetime import date
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

#: Boards to play when the instruction says "until you win": losses are retried
#: rather than reported, so a run keeps going until a win or this budget. Three
#: is the same ceiling ``run()`` has always enforced on an explicit ``games``
#: payload — the site bans for interaction volume, and a Beginner game is ~40
#: clicks, so a long replay streak is exactly what gets the egress IP blocked.
#: Reaching it with no win is reported as a loss, never as a win.
WIN_GAMES = 3

#: The board's URL. Overridable because minesweeper.online moves its entry path
#: between game modes, and a moved URL must not be a code deploy.
DEFAULT_ENTRY_URL = "https://minesweeper.online/new-game"

#: A ceiling on clicks per game, so a solver bug cannot turn into a click storm.
#: A Beginner board needs ~40; 200 is generous headroom, not a target.
MAX_CLICKS_PER_GAME = 200

#: "Play minesweeper" in the operator's own words. The router's contract is that
#: a claim means "this recipe can do it completely" — and for a Beginner board
#: that is true: the solver plays it with no model in the loop. Until this
#: predicate existed (2026-09-23) no sentence could reach the recipe at all; the
#: UI's dropdown was the only way in, so "play a game of minesweeper until you
#: win" went to plan.task, planned a Google search, and burned three attempts and
#: 24 agent steps on a site it had picked for itself.
_MINESWEEPER_RE = re.compile(r"\bminesweeper\b|\bmine ?sweeper\b", re.IGNORECASE)


def _wants_a_win(text: str) -> bool:
    """Whether the instruction asks to keep playing until a board is won.

    Narrow on purpose: "until you win", "until it's won", "and win". A plain
    "play minesweeper" plays one board, which is what the recipe has always done
    and what the site's own ban budget allows.
    """
    return bool(re.search(r"\bwin\b|\bbeat\b", text or "", re.IGNORECASE))


#: A sentence that names the game for a reason other than playing it. Kept tiny
#: and literal: these are the forms that actually appeared, and a broader net
#: would start refusing real plays.
_NOT_A_PLAY_RE = re.compile(
    r"\b(explain|describe|what is|what's|how does|how do|read about|wikipedia"
    r"|source code|documentation|install|implement|write|solve this|help me "
    r"understand)\b",
    re.IGNORECASE,
)


#: A verb that makes the sentence an instruction to play. "play", "game",
#: "win", or an explicit start.
_PLAY_RE = re.compile(
    r"\b(play|playing|game|win|won|start|begin|beating?|beat)\b",
    re.IGNORECASE,
)

#: Ambiguous frontier cells the noul duel needs. Exactly two, because the
#: question is binary and the measured framing is the solver's top pick against
#: its runner-up — a third cell has nothing to contribute to one yes/no.
_MAX_TIEBREAK_CANDIDATES = 2


class Minesweeper:
    name = "minesweeper.play"
    #: Deterministic: the same code runs whatever it is told, so an operator
    #: instruction is not an input it has. See /api/tasks/{id}/say.
    reads_instruction = False

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

    @classmethod
    def understands(cls, text: str, today: date) -> bool:
        """Whether the instruction is a minesweeper game this recipe can play.

        Strict in the one way that matters: it claims only a *game*, never a
        sentence that merely mentions the word — "explain the minesweeper
        algorithm" and "read the minesweeper Wikipedia page" are not plays, and
        claiming them would hand a reading task to a click loop. A play verb (or
        the word "game") has to be present alongside the game's name.
        """
        low = text or ""
        if not _MINESWEEPER_RE.search(low):
            return False
        if _NOT_A_PLAY_RE.search(low):
            return False
        return bool(_PLAY_RE.search(low))

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
        instruction = str(payload.get("task") or payload.get("text") or "")
        # "Play a game of minesweeper until you win" is an outcome, not a count:
        # a lost board is retried, up to WIN_GAMES, because a single loss is a
        # legitimate way for a Beginner board to end and reporting it as the run's
        # result is not what the operator asked for. An explicit ``games`` payload
        # still wins over the prose, so a scheduled run that says "3" means 3.
        until_win = _wants_a_win(instruction)
        default_games = WIN_GAMES if until_win else DEFAULT_GAMES
        games = int(payload.get("games") or default_games)
        games = max(1, min(games, WIN_GAMES))
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

            outcome = await self._play_game(page, view, log_, game_no, max_clicks)
            results.append(outcome)

            if outcome["outcome"] == "won":
                break
            if outcome["outcome"] == "lost":
                if not until_win:
                    # A loss is a legitimate outcome, but playing on after one
                    # spends clicks on nothing. Stop the run and report.
                    break
                # "Until you win": a loss is the reason to play another board,
                # not the result. The click budget (WIN_GAMES) is what stops a
                # losing streak; reaching it is reported as the loss it is.
                if game_no < games:
                    log_.note("info", f"game {game_no} lost — playing another board")

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

        Laya is a *tiebreak*, so the question is deliberately narrow — the
        solver's top pick against its runner-up — and the answer is only taken
        when the gate is confident. Anything else (gate off, unsure, transport
        failure) leaves the solver's own risk ranking in charge. Laya never
        raises and never escalates; a bad call costs a guess, not a game, since
        the fallback is the same heuristic either way.

        The question is a **binary** one, and that is load-bearing, not style.
        The 3-way ``choice`` head reports the chosen label's own probability,
        which sits at 0.5, so ``laya_game_min_confidence`` was structurally
        unreachable through it — measured 2-4/24 cleared under every wording
        (``scripts/laya_choice_probe.py``). The binary ``noul`` head reports
        ``max(p, 1-p)``, which clears the same floor as soon as the model is
        slightly decided: 20/24 on one duel, and whole games at 354/400 against
        the solver's 349/400 (``scripts/laya_game_sim.py``). Two candidates is
        the whole shortlist, because one yes/no has nothing to say about a
        third cell.
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

        (r0, c0), _ = shortlist[0]
        (r1, c1), _ = shortlist[1]
        state = (
            f"{view.board.render()}\n"
            f"Minesweeper, {view.board.rows}x{view.board.cols}, "
            f"{view.board.mines_left} mines unmarked. "
            f"Nothing is provable; one of these cells must be opened."
        )
        picked, conf = await self._laya.yes_no(
            f"Is choosing row {r0 + 1}, column {c0 + 1} at least as safe "
            f"as choosing row {r1 + 1}, column {c1 + 1}?",
            state,
        )
        threshold = self._settings.laya_game_min_confidence
        if picked is None or conf < threshold:
            log_.note("gate", f"game {game_no}: tiebreak inconclusive (conf {conf:.2f})")
            return moves[0], True

        # "yes" keeps the solver's own top pick; "no" takes the runner-up.
        (r, c), p = shortlist[0] if picked else shortlist[1]
        log_.note("gate",
                  f"game {game_no}: laya chose {r + 1},{c + 1} "
                  f"(conf {conf:.2f}, risk {p:.0%})")
        return Move("open", r, c, f"laya tiebreak (conf {conf:.2f})"), True


register_builtin(Minesweeper())
