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
from urllib.parse import urlsplit

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
#: between game modes, and a moved URL must not be a code deploy. This is the
#: canonical real site. When it stops working (it IP-blocked this deployment's
#: egress on 2026-09-25), the recipe does not fall back to a stand-in page — the
#: operator was explicit: "not on local host — find a real page". Instead it
#: searches for a real alternative that serves the board contract, verifies it,
#: and records it (see ``site_health``), which *is* the recipe update. Precedence:
#: the operator's config override, then what the bot itself discovered, then this
#: default.
DEFAULT_ENTRY_URL = "https://minesweeper.online/new-game"

#: How to find a real replacement when the canonical site blocks this egress: a
#: plain search for online minesweeper, read like a person would read it.
_SEARCH_URL = "https://html.duckduckgo.com/html/?q=play+minesweeper+online"

#: Only pages that carry this exact board contract count as replacements: a
#: 9x9 ``cell_<x>_<y>`` grid under ``#CellsBlock`` — the same DOM the solver
#: reads on the canonical site. Anything else would need a new adapter, which
#: is a human decision, not something to improvise mid-run.
_BOARD_PROBE = (
    "() => { const b = document.querySelector('#CellsBlock');"
    " return b ? b.querySelectorAll('[id^=cell_]').length : 0; }"
)

#: Search-result links, kept as (href, visible text) so junk can be filtered.
_COLLECT_LINKS = (
    "() => Array.from(document.querySelectorAll('a[href]')).map(a => ({"
    "href: a.href, text: (a.textContent || '').trim() }))"
)

#: Hosts that are never candidates: the blocked canonical, the agent's own
#: loopback (operator-excluded), and the search engine itself.
_EXCLUDED_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "duckduckgo.com", "html.duckduckgo.com"}
)

#: How many distinct real hosts to try before concluding nothing qualifies.
_MAX_CANDIDATES = 4


def _real_url(href: str) -> str | None:
    """A navigable http(s) URL from a search link, or None.

    DuckDuckGo's HTML endpoint wraps results in ``/l/?uddg=<encoded>``; unwrap
    that. javascript:, relative and non-http links are not pages to play on.
    """
    href = (href or "").strip()
    if not href:
        return None
    if href.startswith("//"):
        href = "https:" + href
    parts = urlsplit(href)
    if parts.scheme not in ("http", "https"):
        return None
    if parts.path.startswith("/l/") and parts.query:
        from urllib.parse import parse_qs

        target = parse_qs(parts.query).get("uddg", [""])[0]
        if not target:
            return None
        return _real_url(target)
    return href

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
        """Where to play. The operator's explicit override (the recipe library)
        wins over what this bot itself discovered after the canonical site
        broke; both win over the built-in default."""
        override = cfg("minesweeper.play", "entry_url", "")
        if override:
            return override
        from ..site_health import entry_url as discovered

        return discovered(self._settings, self.name) or DEFAULT_ENTRY_URL

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
        url = self.entry_url
        override = bool(cfg("minesweeper.play", "entry_url", ""))

        # Obstacle memory, consulted before anything is navigated (2026-09-26).
        # Every block and every rejected replacement is written down for good,
        # so a venue this bot has already written off is never walked into
        # again: the run goes straight to looking for a genuinely new venue,
        # and when search yields nothing new it stops in seconds carrying the
        # whole history, instead of re-playing a doomed path for minutes. The
        # operator's explicit override outranks the memory — a changed egress
        # or a lifted ban is the operator's hand, and the memory only ever
        # narrows what the bot chooses on its own.
        if not override:
            from ..site_health import host_of, written_off

            if host_of(url) in written_off(self._settings, self.name):
                log_.note(
                    "info",
                    f"{host_of(url)} is remembered-blocked; searching only for "
                    f"venues this bot has not already ruled out",
                )
                healed = await self._heal(page, log_)
                if healed is None:
                    raise self._exhausted(url)
                url = healed

        for game_no in range(1, games + 1):
            log_.note("info", f"game {game_no}/{games}: starting a Beginner board")
            view = await start_beginner(page, url=url)

            if view.blocked:
                # The IP block is app-side and arrives after boot. Retrying the
                # blocked site only makes it worse — but neither does giving up
                # while a real alternative may exist. The recipe updates itself:
                # the block is written down first (so this sighting counts even
                # if the heal succeeds elsewhere), then it finds a real page
                # that serves the board contract, verifies it, records it, and
                # plays there. Only when no real page qualifies does this stop
                # and ask for a human.
                from ..site_health import host_of, record_blocked

                record_blocked(self._settings, self.name, host=host_of(url),
                               reason="the site served its block page "
                                      "(Account blocked)", url=url)
                healed = await self._heal(page, log_)
                if healed is None:
                    raise self._exhausted(url)
                url = healed
                view = await start_beginner(page, url=url)
                if view.blocked:
                    record_blocked(self._settings, self.name, host=host_of(url),
                                   reason="the replacement served its block "
                                          "page too", url=url)
                    raise EscalationRequired(
                        Challenge(ChallengeKind.RATE_LIMITED,
                                  "the replacement page is blocked too", page.url)
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
        lost = sum(1 for r in results if r["outcome"] == "lost")
        blocked = sum(1 for r in results if r["outcome"] == "blocked")
        stalled = sum(1 for r in results if r["outcome"] == "stalled")
        unfinished = len(results) - wins - lost - blocked - stalled
        host = urlsplit(url).hostname or url
        bits = [f"{wins} won"]
        if lost:
            bits.append(f"{lost} lost on a guess")
        if blocked:
            bits.append(f"{blocked} blocked mid-game")
        if stalled:
            bits.append(f"{stalled} settled without a win signal")
        if unfinished:
            bits.append(f"{unfinished} unfinished")
        summary = f"played {len(results)} board(s) on {host}: " + ", ".join(bits)
        log_.note("info", summary)
        # "summary" is what the thread note renders — "Done: wins: 0" told the
        # operator nothing about where it played or why nothing was won.
        return {"games": results, "wins": wins, "blocked": blocked,
                "site": url, "summary": summary}

    async def on_entry_unreachable(self, session: Any, exc: Exception) -> bool:
        """The runner-level entry navigation died before run() got control.

        This is the recipe's one chance to find its own way in: search for a
        real replacement (obstacle memory keeps the search off every host
        already written off — the unreachable canonical included), verify the
        board contract, and record it. The runner then re-reads ``entry_url``
        — now the replacement — and navigates there instead of the dead hop.
        """
        log_ = activity_of(session)
        page = await session.page()
        return await self._heal(page, log_) is not None

    def _exhausted(self, url: str, *, blocked: bool = False) -> EscalationRequired:
        """The stop for "every venue this bot knows about is written off".

        The message IS the memory — which hosts blocked this egress and how
        often, how many searched replacements were probed and rejected — so
        the escalation carries the whole history and the operator decides
        with everything in view, once, instead of re-deriving it each run.

        ``url`` is the venue this run was working on, NOT ``page.url``: the
        page usually ended on the search engine or the last probed candidate,
        and escalating against those memorized DuckDuckGo as a blocked venue.

        ``blocked`` marks a genuine refusal of that venue (it served its block
        page). Without it the challenge is RATE_LIMITED with blocked=False —
        "we could not finish" is not "this site refused us", and the runner
        must not write a permanent refusal from it. That inversion is what
        turned a single transient timeout into a site written off for good.
        """
        from ..site_health import blocked_hosts

        blocked_hosts_ = blocked_hosts(self._settings, self.name)
        names: list[str] = []
        rejected = 0
        for host, b in sorted(blocked_hosts_.items()):
            if b.get("kind") == "rejected":
                rejected += 1
            elif b.get("kind") in ("blocked", "unreachable"):
                names.append(f"{host} (seen {b.get('count', 1)}x)")
        msg = ("every venue is written off: " + ", ".join(names)) if names \
            else "no playable venue is known"
        if rejected:
            msg += (f"; {rejected} searched replacement(s) probed and rejected "
                    f"(no board contract)")
        msg += (". The local test board is operator-excluded. Needs a different "
                "egress, or an adapter for another site.")
        return EscalationRequired(
            Challenge(ChallengeKind.RATE_LIMITED, msg, url), blocked=blocked)

    async def _heal(self, page: Any, log_: Any) -> str | None:
        """Update the recipe after its canonical site stopped working.

        The operator's rule (2026-09-25): a recipe whose site breaks gets
        updated, not abandoned and not replaced with a stand-in. This searches
        the real web for another playable board, keeps only pages that carry
        the exact DOM contract the solver reads (a 9x9 ``cell_<x>_<y>`` grid
        under ``#CellsBlock``), verifies the match with a real navigation, and
        records the winner via ``site_health`` so every later run starts from
        the learned answer. A candidate without the contract is written down
        as rejected, so the next search never re-probes it. Returns the new
        URL, or None when nothing real qualifies — honesty over a fake board.
        """
        from ..site_health import host_of, record, record_blocked, written_off

        for url in await self._search_candidates(page, skip=written_off(
                self._settings, self.name)):
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            except Exception:
                log_.note("error", f"candidate {url} could not be checked")
                continue
            # The board paints only after the site's script runs, so a single
            # un-waited read at domcontentloaded legitimately sees an empty
            # grid — and writing that off as "rejected" shrinks the candidate
            # pool toward exhaustion on a page that was merely slow. Poll for
            # the contract before judging; a real absence still records.
            cells = 0
            for _ in range(10):
                try:
                    cells = await page.evaluate(_BOARD_PROBE)
                except Exception:
                    cells = 0
                if cells == 81:
                    break
                await page.wait_for_timeout(1000)
            if cells != 81:
                record_blocked(self._settings, self.name, host=host_of(url),
                               kind="rejected",
                               reason="searched as a replacement but its page "
                                      "has no 9x9 board contract", url=url)
                continue
            record(self._settings, self.name, entry_url=url,
                   note="the canonical site is unusable from this egress; "
                        f"{url} discovered and verified as a replacement")
            log_.note("step", f"recipe updated: canonical site is blocked; "
                              f"playing on {url}")
            return url
        return None

    async def _search_candidates(self, page: Any,
                                 skip: dict | None = None) -> list[str]:
        """Real minesweeper pages found by search, best guess first.

        Navigates a search engine like a person would, collects the result
        links, and filters to real http(s) pages — never a host this bot has
        written off as a REFUSAL (``skip``, the narrowing set: blocked or
        rejected), never the agent's own loopback (operator-excluded), one per
        host.

        A transient ``unreachable`` host is deliberately NOT skipped: the run
        is searching precisely because a venue did not answer, and the
        canonical is the first candidate worth re-probing. Excluding it here
        would make the transient record act like a permanent write-off — the
        exact defect this round removes.
        """
        from ..site_health import written_off as _written_off

        try:
            await page.goto(_SEARCH_URL, wait_until="domcontentloaded", timeout=45_000)
            found = await page.evaluate(_COLLECT_LINKS)
        except Exception:
            log.warning("site search failed; no candidates", exc_info=True)
            return []

        remembered = skip if skip is not None else _written_off(
            self._settings, self.name)
        seen: set[str] = set()
        out: list[str] = []
        for link in found or []:
            url = _real_url(str(link.get("href") or ""))
            if url is None:
                continue
            host = urlsplit(url).hostname or ""
            host = host[4:] if host.startswith("www.") else host
            if not host or host in seen:
                continue
            if host in remembered or host in _EXCLUDED_HOSTS:
                continue
            seen.add(host)
            out.append(url)
            if len(out) >= _MAX_CANDIDATES:
                break
        return out

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
            if view.blocked:
                # The IP block is app-side and can arrive *mid-game*, after the
                # board has booted. It is read on every ``read_game`` but was
                # only ever checked at game start, so a board that got blocked
                # partway kept clicking a block page until the click budget ran
                # out and reported "unfinished" — a false non-result that hid
                # the real cause (2026-09-27). Record the sighting and stop.
                from ..site_health import host_of, record_blocked

                record_blocked(self._settings, self.name, host=host_of(page.url),
                               reason="the site served its block page mid-game",
                               url=page.url)
                log_.note("error", f"game {game_no}: blocked mid-game after "
                                   f"{clicks} clicks")
                return {"outcome": "blocked", "clicks": clicks, "guesses": guesses,
                        "laya_calls": laya_calls}
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
                # No proof and no gamble left: the solver considers the board
                # fully settled, which should mean every non-mine cell is open —
                # a win the face check ought to have caught already. It is not
                # reported as a win, because the face says otherwise and a false
                # win is worse than an honest stall; it is reported as its own
                # outcome so the summary can say the board settled without a win
                # signal rather than lumping it into "unfinished" with a board
                # abandoned mid-play (2026-09-27).
                log_.note("error", f"game {game_no}: board settled with no win "
                                   f"signal after {clicks} clicks")
                return {"outcome": "stalled", "clicks": clicks, "guesses": guesses,
                        "laya_calls": laya_calls, "board": view.board.render()}

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
