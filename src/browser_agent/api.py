"""Control-plane API + admin UI.

Each profile pod serves this on its own port. The admin UI is a single page
served from here, so there is no separate frontend build to keep in sync.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import recipes  # noqa: F401  (importing registers every built-in recipe)
from .accounts import AccountStore, load_accounts, remove_account, valid_account
from .agent import make_agent_runner
from .browser import BrowserSession, profile_exists
from .config import Settings, load_settings
from .escalation import detect_challenge
from .router import claim, learned_match, route
from .runstore import RunStore
from .scheduler import ScheduleStore
from .tasks import (
    _URL_RE,
    AGENT_RECIPE,
    TaskRunner,
    TaskStatus,
    _start_url_for,
    _task_text,
    history_brief,
    list_recipes,
    recipe_reads_instruction,
)
from .threads import ThreadStore

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings: Settings = load_settings()
session = BrowserSession(settings)
store = ScheduleStore(settings.state_db)
# Every finished run, on the profile's PVC. Separate from the schedule store and
# from the in-memory task objects on purpose: the point of the archive is to
# outlive a deploy, and a task is recreated (and lost) by one.
runs = RunStore(settings.runs_db, profile=settings.profile)
# The conversation is mirrored into that same archive, so the instruction that
# created an attempt is still there when the attempt is read back.
threads = ThreadStore(runs)
# The runner asks where the active account is rather than being told once, so a
# pod that restarted after a switch reconciles before its first task instead of
# running silently as the previous identity.
runner = TaskRunner(
    settings,
    session,
    agent_runner=make_agent_runner(settings),
    runs=runs,
    threads=threads,
    active_account=lambda: load_accounts(
        settings.profile_root, settings.accounts_path
    ).active_account(),
)

# Shown on a freshly started pod so noVNC opens on a real page instead of a
# blank X root window. A data URL on purpose: it cannot fail on DNS, on the
# egress proxy, or on a site being down, and it costs no network to paint.
_IDLE_PAGE = (
    "data:text/html;charset=utf-8,"
    "<meta charset=utf-8><title>browser-agent</title>"
    "<style>html,body{height:100%;margin:0}"
    "body{display:flex;align-items:center;justify-content:center;"
    "background:#14161a;color:#e6e6e6;"
    "font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}"
    "main{text-align:center}h1{font-size:17px;font-weight:600;margin:0 0 6px}"
    "p{margin:0;color:#8b93a1}</style>"
    "<main><h1>browser-agent</h1><p>Ready &mdash; no task running.</p></main>"
)


async def _schedule_loop() -> None:
    """Fire due schedules. One loop per pod, so no distributed lock is needed."""
    import asyncio

    while True:
        # Per schedule, not around the whole batch. One schedule naming a
        # recipe that no longer exists — a stored recipe deleted from the
        # library, say — raises out of submit(), and with the guard outside the
        # loop that aborted every *later* schedule on every pass, forever.
        # Each entry is independent, so each gets its own guard.
        for sched in store.due():
            try:
                # A schedule carries its instruction too, so it routes by the
                # same rule as a typed one — a nightly "cheapest flight to X"
                # schedule lands on the deterministic recipe, not the agent.
                text = str(sched.payload.get("task") or sched.payload.get("text") or "")
                task = runner.submit(route(text, sched.recipe), sched.payload)
                store.mark_run(sched.id, task.id)
                log.info("schedule %s fired task %s", sched.id, task.id)
            except Exception:
                log.exception("schedule %s could not be fired", sched.id)
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    # Migrate before anything touches the disk, and in particular before the
    # browser pre-warm below. Order matters here and is not obvious: this build
    # keeps Chrome's user-data-dir one level deeper than the last one
    # (/profiles/<bot>/accounts/<account>), so a bot upgraded from the old
    # layout has its login sitting at /profiles/<bot>/. Launching first would
    # have Chrome create a *fresh* profile in the new location — and the
    # migration would then merge the old files into a directory Chrome had
    # already written, which is how a profile gets corrupted rather than merely
    # reported wrong. So: move, then start.
    try:
        session.set_account(store_accounts().active_account())
    except Exception:
        log.exception("could not prepare this bot's accounts; the profile may be unmigrated")

    # Start the browser now rather than on first use. Everything a human sees
    # over noVNC is this X display, and the browser was only launched lazily by
    # the first task: a freshly started pod therefore served noVNC straight to
    # an empty black root window, which reads as "the browser is broken" until
    # something happens to run. Pre-warming also means a takeover has something
    # to take over. A failure here must not take the control plane down with it.
    try:
        await session.start()
        page = await session.page()
        if page.url in ("", "about:blank"):
            await page.goto(_IDLE_PAGE)
    except Exception:
        log.exception("browser pre-warm failed; the desktop may be empty until a task runs")

    # Read the recipe library once at boot so a freshly started pod serves the
    # operator's recipes on its first request. Later reads are lazy (the store
    # caches on the directory mtime), but kubelet's sync of a ConfigMap edit is
    # not instant, so this is also what makes a pod that just mounted the
    # projection agree with the hub.
    try:
        from .recipes.stored import load_stored_recipes

        installed = load_stored_recipes()
        if installed:
            log.info("recipe library: %s", ", ".join(installed))
    except Exception:
        log.exception("could not read the recipe library at boot")

    runner.start()
    loop = asyncio.create_task(_schedule_loop())
    log.info(
        "control plane up profile=%s recipes=%s",
        settings.profile,
        [r["name"] for r in list_recipes()],
    )
    try:
        yield
    finally:
        loop.cancel()
        await runner.stop()
        await session.stop()


app = FastAPI(title="browser-agent", lifespan=lifespan)


# -- the path this bot is reached under ------------------------------------

# When a roster proxies several bots from one domain, each is mounted at a
# prefix (/b/<profile>) that nginx strips before forwarding. Routing does not
# need to know that — every route here is still "/" from the pod's side — but
# the UI does, because it builds its own live-view URL. The proxy passes the
# prefix in X-Forwarded-Prefix (the convention uvicorn's --proxy-headers
# understands) and it takes precedence over the static setting.
_REACHED_PREFIX = ContextVar("reached_prefix", default="")

# SEC-BA-005 (2026-09-27): X-Forwarded-Prefix is client-controlled and used to
# reach an inline <script> in the served UI verbatim, so a crafted header was
# stored XSS on the bot's own origin. The value is only ever a URL path prefix
# (nginx sends /b/<profile>; the standalone default is the operator's own
# BROWSER_URL_PREFIX), so anything outside this character set is not a prefix
# and is dropped in favour of the configured one. The render site escapes too —
# this validates the stored value, the sink is still escaped.
_PREFIX_RE = re.compile(r"^/[A-Za-z0-9._~/-]*$")


@app.middleware("http")
async def _capture_prefix(request: Request, call_next):
    raw = request.headers.get("x-forwarded-prefix", "").rstrip("/")
    if raw and not _PREFIX_RE.match(raw):
        log.warning("dropping non-prefix X-Forwarded-Prefix value")
        raw = ""
    token = _REACHED_PREFIX.set(raw or settings.url_prefix)
    try:
        return await call_next(request)
    finally:
        _REACHED_PREFIX.reset(token)


# -- auth ------------------------------------------------------------------


async def require_token(request: Request) -> None:
    """Bearer auth.

    SEC-BA-002 (2026-09-26): fails CLOSED. An unset CONTROL_TOKEN used to
    disable auth on every route; it now refuses the request instead.

    SEC-BA-004: a browser that passed nginx basic auth has the bearer injected
    for it, so without an origin check a page on another site could drive the
    control plane with the operator's own credentials. Refuse mutating
    requests that arrive cross-site.
    """
    if not settings.control_token:
        raise HTTPException(status_code=503, detail="CONTROL_TOKEN is not configured")
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        # The browser's own verdict is authoritative and cannot be forged by
        # page script, so when it is present it is the whole check. The
        # Origin-vs-Host comparison is only the fallback for clients that send
        # no Sec-Fetch-Site: any proxy that rewrites Host (nginx stops
        # inheriting proxy_set_header the moment a location sets its own)
        # defeats it, which refused every legitimate same-origin POST live
        # on 2026-09-27.
        site = request.headers.get("sec-fetch-site", "")
        if site:
            if site not in ("same-origin", "none"):
                raise HTTPException(status_code=403, detail="cross-site request refused")
        else:
            origin = request.headers.get("origin", "")
            if origin and origin.split("://", 1)[-1] != request.headers.get("host", ""):
                raise HTTPException(status_code=403, detail="cross-origin request refused")
    header = request.headers.get("authorization", "")
    if header.removeprefix("Bearer ").strip() != settings.control_token:
        raise HTTPException(status_code=401, detail="unauthorized")


# -- models ----------------------------------------------------------------


class TaskRequest(BaseModel):
    recipe: str
    payload: dict[str, Any] = {}


class ScheduleRequest(BaseModel):
    id: str
    recipe: str
    cron: str
    payload: dict[str, Any] = {}
    enabled: bool = True


class LoginRequest(BaseModel):
    url: str


class SteerRequest(BaseModel):
    instructions: str


class RetryRequest(BaseModel):
    instructions: str = ""


# -- status ----------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "profile": settings.profile,
        "account": session.account,
        "profile_initialised": profile_exists(settings, session.account),
        "llm_enabled": runner.llm.enabled,
    }


@app.get("/api/whoami", dependencies=[Depends(require_token)])
async def whoami() -> dict[str, Any]:
    """Who this bot is, cheaply.

    Deliberately separate from /api/state: the roster needs one cheap identity
    line per bot, and /api/state is the expensive one (it probes the model and
    serialises up to 50 tasks). Nothing here touches the browser, the store or
    the network, so a roster can poll every bot at once without waking any of
    them up. It is also the only endpoint that does not require the profile to
    have been initialised — a roster has to see a bot before it is signed in.

    The account fields stay deliberately shallow for that reason: the roster
    shows *which* identity is live, but the per-account detail (cookie counts
    for each of four accounts, on every poll of every bot) belongs to the bot's
    own page, and /api/state carries it.
    """
    accounts = store_accounts()
    running = accounts.get(session.account)
    return {
        "profile": settings.profile,
        "display_name": settings.display_name,
        "job": settings.display_job,
        "account": session.account,
        # Falls back to the slug when the running account is not in the store —
        # reachable by hand-editing accounts.json under a live pod.
        "account_label": running.to_dict()["display_label"] if running else session.account,
        "accounts": len(accounts.accounts),
        "profile_initialised": profile_exists(settings, session.account),
        "url_prefix": _REACHED_PREFIX.get(),
    }


def _recent_tasks(limit: int = 50) -> list[dict[str, Any]]:
    """The History list: this process's tasks, then the archive behind them.

    The in-memory list is authoritative while a task exists — it carries live
    status and the running control — but it is emptied by every deploy, because
    a deploy recreates the pod. Since the archive exists precisely to outlive
    that, the History table reads through to it: otherwise "keep every run so we
    can evaluate them" would be true of the disk and false of the page.

    An archived row is shaped like a live one so the UI needs no second code
    path, and is marked ``archived`` so "Read" is offered instead of controls
    that would 404.
    """
    live = sorted(runner.tasks.values(), key=lambda t: t.created_at, reverse=True)
    out = [{**t.to_dict(), "archived": False} for t in live[:limit]]
    if len(out) >= limit:
        return out
    seen = {t["id"] for t in out}
    for run in runs.list(limit=limit):
        if run.task_id in seen:
            continue
        d = run.to_dict()
        d["id"] = d.pop("task_id")
        # ``reads_instruction`` rides along from the record — the archive stored
        # it when the run happened. Re-deriving it here would ask today's recipe
        # registry a question about last week's run, so a stored recipe that was
        # edited or deleted since would silently rewrite the answer.
        d["amended_count"] = 0
        d["archived"] = True
        out.append(d)
        if len(out) >= limit:
            break
    return sorted(out, key=lambda t: t["created_at"], reverse=True)


@app.get("/api/state", dependencies=[Depends(require_token)])
async def state() -> dict[str, Any]:
    """Everything the admin UI needs, in one call."""
    tasks = _recent_tasks()
    current = runner.current.to_dict() if runner.current else None
    if current is not None and runner.control is not None:
        # Rides on the task because the UI's question is "what is this task
        # doing", and the controls only exist while one is running.
        current["control"] = runner.control.to_dict()
    return {
        "profile": settings.profile,
        "profile_initialised": profile_exists(settings, session.account),
        # Which account this bot is running as, and every account it holds.
        # Accounts are cheap to read (a small JSON file plus, per account, one
        # read-only SQLite count), and a switch has to be reflected here
        # immediately — the UI's question after clicking "switch" is "did it
        # take", and a cached answer would say no.
        "account": session.account,
        "accounts": store_accounts().to_dict(),
        # Whether the browser behind noVNC is actually up. The window can be
        # closed from inside the live view, and until now nothing said so: the
        # next task just failed with a connection error. Recovery is automatic
        # on the next run (BrowserSession.start relaunches), and this is what
        # lets the UI say so and offer a restart instead of looking broken.
        "browser_running": session.is_running(),
        "novnc_port": settings.novnc_port,
        # Where the browser UI is reachable from the operator's machine. Empty
        # when the pod port is forwarded directly; set via BROWSER_BASE_URL when
        # reached through a reverse proxy (then it is the domain, not :port).
        "browser_base_url": settings.browser_base_url,
        # Whether the fallback can actually reach the model, not just whether it
        # is switched on. The two diverged silently once (an ambient proxy sent
        # the call through SOCKS), which made every escalation look like a
        # captcha. This is deliberately NOT in /healthz: that backs the k8s
        # liveness probe, and an LLM outage must not restart the pod.
        "llm_reachable": await runner.llm.healthy(),
        # off | no-key | ready | unreachable — distinguishes a missing key from
        # an unreachable service, because they need different fixes. Both are
        # cached (see LLMClient.healthy), so polling never waits on the model.
        "llm_status": await runner.llm.status(),
        "llm_enabled": runner.llm.enabled,
        "recipes": list_recipes(),
        "current": current,
        "tasks": tasks,
        "schedules": [s.to_dict() for s in store.all()],
        "queue_depth": runner.queue.qsize(),
        # Prefixed, so a bot reached under /b/<profile> points at its OWN live
        # view and not the roster's. Empty prefix (the default) leaves today's
        # behaviour exactly as it was.
        "url_prefix": _REACHED_PREFIX.get(),
        "takeover_url": f"{_REACHED_PREFIX.get()}/vnc.html",
    }


@app.get("/api/activity", dependencies=[Depends(require_token)])
async def activity(task_id: str = "") -> dict[str, Any]:
    """What the running task is doing, step by step.

    Separate from /api/state because it changes far faster: the state poll is
    about the pod, this is about the run, and the operator watching a task wants
    the second one second-by-second.

    ``?task_id=`` reads a *finished* attempt's feed from the snapshot the runner
    takes when the attempt ends. Without it the thread panel could show the live
    run and nothing else, which is precisely the state the operator complained
    about — "one block, then the retries", with no record of what was tried.
    """
    if task_id:
        task = runner.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="no such task")
        live = task.id == (runner.current.id if runner.current else None)
        return {
            "task_id": task.id,
            "status": task.status.value,
            "control": runner.control.to_dict() if live and runner.control else None,
            "detail": task.detail,
            "entries": runner.activity.as_list() if live else task.activity,
            "page_url": _current_page_url() if live else "",
        }

    current = runner.current
    return {
        "task_id": current.id if current else None,
        "status": current.status.value if current else None,
        "control": runner.control.to_dict() if runner.control else None,
        "detail": current.detail if current else "",
        "entries": runner.activity.as_list(),
        "page_url": _current_page_url(),
    }


def _current_page_url() -> str:
    """The live page URL, best-effort. Never let a status read fail on it."""
    try:
        ctx = session._context
        if ctx is None or not ctx.pages:
            return ""
        return ctx.pages[0].url
    except Exception:
        return ""


@app.get("/api/challenge", dependencies=[Depends(require_token)])
async def challenge() -> dict[str, Any]:
    """Live check of what is on screen right now."""
    try:
        page = await session.page()
    except Exception as exc:
        return {"page_open": False, "detail": str(exc)}
    found = await detect_challenge(page)
    return {
        "page_open": True,
        "url": page.url,
        "challenge": None if found is None else {"kind": found.kind.value, "detail": found.detail},
    }


# -- tasks -----------------------------------------------------------------


@app.post("/api/tasks", dependencies=[Depends(require_token)])
async def create_task(req: TaskRequest) -> dict[str, Any]:
    # The recipe is picked here rather than trusted blindly: the instruction is
    # what the operator actually wrote, and a deterministic recipe that can
    # answer it outright must not lose to a 135-second LLM plan just because the
    # dropdown was left on the default. See router.py.
    text = str(req.payload.get("task") or req.payload.get("text") or "")
    # A learned recipe the agent earned may answer a freeform request, but only
    # once it has replayed cleanly (see Settings.learned_recipe_min_replays).
    # Consulted before the sync route() so a hand-authored predicate still wins.
    matched = await learned_match(text, req.recipe, settings)
    try:
        task = runner.submit(route(text, req.recipe, learned_match=matched), req.payload)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # The instruction that STARTED the work is part of the conversation, and it
    # is the one an operator most needs back: "Talk to it" opens on an attempt
    # whose whole reason for existing was this sentence. Recorded for every
    # recipe, because a deterministic one is asked for by prose too — the run
    # just does not read it — and hiding it would make the thread open mid-story.
    #
    # Stamped with the task's own creation time, not "now". Saying it after
    # ``submit`` returns would date the ask a few hundred microseconds *later*
    # than the attempt it caused, and the thread sorts by time — so the
    # conversation would open with an attempt that answered a question nobody
    # had asked yet.
    opening = _task_text(task)
    if opening:
        threads.say(
            task.thread_id, "operator", "instruction", opening, at=task.created_at
        )
    return task.to_dict()


class FreshChatRequest(BaseModel):
    text: str


@app.post("/api/chat", dependencies=[Depends(require_token)])
async def fresh_chat(req: FreshChatRequest) -> dict[str, Any]:
    """Start a brand-new thread from one message — the chat's "start over".

    "New chat" has to behave like talking to a person: an empty conversation is
    not an error state that redirects to a form, it accepts a sentence and work
    begins. This is the composer's path when the open thread has no attempts,
    and it routes prose by exactly the same rule as create_task: a recipe whose
    ``understands`` claims the sentence runs it, a learned recipe may, and
    otherwise the freeform agent takes it — which is the one case that needs a
    start URL, extracted from the message itself here. Refusing at submit time
    (instead of queueing a task destined to fail) is the point: the composer
    can put the reason next to the text the operator just typed.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    matched = await learned_match(text, AGENT_RECIPE, settings)
    recipe = route(text, AGENT_RECIPE, learned_match=matched)
    payload: dict[str, Any] = {"task": text, "text": text, "goal": text}
    if recipe == AGENT_RECIPE:
        found = _URL_RE.search(text)
        if not found:
            raise HTTPException(
                status_code=400,
                detail="No recipe claimed that, and a freeform run needs a page "
                "to work on — paste the link into the message, or use Run a task.",
            )
        payload["url"] = found.group(0).rstrip(".,;")
    task = runner.submit(recipe, payload)
    opening = _task_text(task)
    if opening:
        threads.say(
            task.thread_id, "operator", "instruction", opening, at=task.created_at
        )
    return task.to_dict()


@app.post("/api/tasks/{task_id}/retry", dependencies=[Depends(require_token)])
async def retry_task(task_id: str, req: RetryRequest | None = None) -> dict[str, Any]:
    """Re-queue a task. Intended for use after a human has cleared a block.

    An optional ``instructions`` field replaces the stored text, so "change what
    it does and try again" is one call rather than a submit-and-delete.
    """
    if task_id not in runner.tasks:
        raise HTTPException(status_code=404, detail="no such task")
    if runner.tasks[task_id].status is TaskStatus.RUNNING:
        raise HTTPException(status_code=409, detail="task is still running")
    payload = None
    if req is not None and req.instructions.strip():
        text = req.instructions.strip()
        # ``goal`` too: the agent reads it FIRST (agent.py), so setting only
        # task/text would let a stale goal silently win and the operator's new
        # instruction would look ignored.
        payload = {**runner.tasks[task_id].payload, "task": text, "text": text, "goal": text}
        threads.say(runner.tasks[task_id].thread_id, "operator", "instruction", text)
    return runner.retry(task_id, payload=payload).to_dict()


# -- live control while a task runs ----------------------------------------


def _running_or_404(task_id: str):
    task = runner.running(task_id)
    if task is None:
        raise HTTPException(status_code=409, detail="that task is not the one running")
    return task


@app.post("/api/tasks/{task_id}/pause", dependencies=[Depends(require_token)])
async def pause_task(task_id: str) -> dict[str, Any]:
    """Hold the run at its next checkpoint. The browser stays open."""
    _running_or_404(task_id)
    runner.pause()
    return {"task_id": task_id, "control": runner.control.to_dict()}


@app.post("/api/tasks/{task_id}/resume", dependencies=[Depends(require_token)])
async def resume_task(task_id: str) -> dict[str, Any]:
    _running_or_404(task_id)
    runner.resume()
    return {"task_id": task_id, "control": runner.control.to_dict()}


@app.post("/api/tasks/{task_id}/stop", dependencies=[Depends(require_token)])
async def stop_task(task_id: str) -> dict[str, Any]:
    """End the run at its next checkpoint. The task is NOT marked blocked:
    a decision by the operator is not a captcha, and must not page one."""
    _running_or_404(task_id)
    runner.control.cancel()
    with contextlib.suppress(Exception):
        runner.activity.note("info", "stop requested; ending at the next step")
    return {"task_id": task_id, "control": runner.control.to_dict()}


@app.post("/api/tasks/{task_id}/steer", dependencies=[Depends(require_token)])
async def steer_task(task_id: str, req: SteerRequest) -> dict[str, Any]:
    """Replace the instruction for a running task.

    A running plan's step list becomes wrong the moment the instruction changes,
    so this ends the current attempt at its next checkpoint and the runner
    re-plans from the new text — see ``TaskRunner._run_plan_recipe``.
    """
    task = _running_or_404(task_id)
    text = req.instructions.strip()
    if not text:
        raise HTTPException(status_code=400, detail="instructions is empty")
    runner.control.steer(text)
    threads.say(task.thread_id, "operator", "instruction", text)
    runner.activity.note("operator", f"you: {text[:200]}")
    return {"task_id": task.id, "instruction": text, "control": runner.control.to_dict()}


# -- the retry conversation ------------------------------------------------


def _recipe_reads_payload(name: str) -> bool:
    """Whether this recipe actually honours its payload's instruction.

    The built-ins that do are the two agent paths and the stored step recipes
    (whose steps carry their own goal text). Everything else — the thin social
    recipes, the game — runs the same code whatever it is told.
    """
    return recipe_reads_instruction(name)


class SayRequest(BaseModel):
    text: str
    #: "instruction" changes the wording for the next attempt; "note" only
    #: records it in the thread. Parameters and config edits go through their
    #: own endpoints so the shared-library blast radius is stated once, in the
    #: confirm the operator actually reads.
    kind: str = "instruction"


async def _claimed_recipe(text: str, thread_recipe: str) -> str | None:
    """The recipe a fresh say-instruction should run, or None for thread logic.

    A message typed into a finished thread is still a new instruction, and the
    router treats it exactly like one. "play a game of minesweeper on a site you
    can reach" names a deterministic recipe's whole job — but typed into a
    thread whose recipe was a refused freeform attempt, it inherited
    ``agent.task`` and died on the no-URL rule (measured live 2026-09-25). The
    router, not the thread the box happened to be open on, decides what a fresh
    instruction means; create_task routes prose by the same rule.

    The thread's *own* recipe counts as a claimant too: re-asked to do its job,
    a deterministic recipe re-runs itself. Measured live 2026-09-25, "play a
    game of minesweeper..." typed into a ``minesweeper.play`` thread escalated
    to the freeform agent instead — route() returns ``requested`` both when that
    recipe claims and when nothing does, so only a direct claim check can tell
    a re-play from an override. The escalate rule below stays for text no
    recipe claims ("don't use that site"), which is the case it exists for.
    """
    matched = await learned_match(text, thread_recipe, settings)
    claimed = route(text, thread_recipe, learned_match=matched)
    if claimed != thread_recipe:
        return claimed
    return thread_recipe if claim(text) == thread_recipe else None


@app.get("/api/runs", dependencies=[Depends(require_token)])
async def list_runs(limit: int = 50, thread_id: str = "") -> dict[str, Any]:
    """The durable archive of finished runs, newest first.

    Distinct from ``/api/state``, which lists the *in-memory* tasks of this
    process and therefore forgets everything on the next deploy. The operator's
    ask was to keep a run — especially a bad one — so it can be studied later;
    that requires a record that outlives the process, which lives here.

    ``?thread_id=`` narrows to one line of work, which is how the thread panel
    reaches runs whose task objects are gone.
    """
    rows = runs.list(limit=max(1, min(limit, 500)), thread_id=thread_id)
    return {
        "profile": settings.profile,
        "count": runs.count(),
        "enabled": runs.enabled,
        "runs": [r.to_dict() for r in rows],
    }


@app.get("/api/runs/{task_id}", dependencies=[Depends(require_token)])
async def get_run(task_id: str) -> dict[str, Any]:
    """One archived run, in full: payload, feed, result, and its decisions.

    The decisions are lifted into their own list because that is what an
    evaluation walks — every Laya verdict and every agent step, in order, is the
    record of what the run *chose*, as opposed to what merely happened.
    """
    found = runs.get(task_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no archived run with that id")
    return found.to_dict()


@app.get("/api/threads/{thread_id}", dependencies=[Depends(require_token)])
async def get_thread(thread_id: str) -> dict[str, Any]:
    """One thread: every attempt in it, the messages, and the live feed.

    The attempts carry their own snapshotted activity, so the panel can show
    what every earlier attempt tried without a second request per attempt.

    A thread whose attempts are no longer in memory — the pod was recreated by a
    deploy — falls back to the archive, so opening an old thread after a restart
    shows the run instead of "no such thread". Attempts still in memory win,
    because only they carry live status and a running attempt's current control.
    """
    members = sorted(
        (t for t in runner.tasks.values() if t.thread_id == thread_id),
        key=lambda t: t.created_at,
    )
    if not members:
        archived = sorted(runs.list(limit=200, thread_id=thread_id),
                          key=lambda r: r.created_at)
        if not archived:
            raise HTTPException(status_code=404, detail="no such thread")
        return {
            "thread_id": thread_id,
            "attempts": [r.to_dict() for r in archived],
            "messages": [m.to_dict() for m in threads.for_thread(thread_id)],
            "current": None,
            "archived": True,
        }
    return {
        "thread_id": thread_id,
        "attempts": [t.to_dict() for t in members],
        "messages": [m.to_dict() for m in threads.for_thread(thread_id)],
        "current": runner.current.to_dict() if runner.current else None,
        "archived": False,
    }


@app.post("/api/tasks/{task_id}/say", dependencies=[Depends(require_token)])
async def say(task_id: str, req: SayRequest) -> dict[str, Any]:
    """Say something to a task, whether it is running or finished.

    The operator's ask was a retry that is a conversation rather than a
    dead-end button, and the whole difficulty is that "talk to it" means two
    different things depending on when you type:

    * **still running** — steer it in place, exactly like /steer. The browser is
      where the attempt left it, which is what makes a mid-run correction cheap.
    * **finished** — the message becomes the next attempt's instruction, in the
      same thread, briefed with what the earlier attempts tried.

    One endpoint rather than two, because the operator should not have to know
    which state the task is in before they can say what they mean.
    """
    task = runner.tasks.get(task_id)
    if task is None:
        # The run outlived the task object — a deploy recreates every pod, and
        # every in-memory task with it — but the archive still has it, and
        # talking to a recorded run is the whole reason the archive exists.
        return await _say_to_archived(task_id, req)
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    if req.kind not in ("instruction", "note"):
        raise HTTPException(status_code=400, detail="kind must be instruction or note")
    threads.say(task.thread_id, "operator", req.kind, text)

    if task.status is TaskStatus.RUNNING:
        runner.control.steer(text)
        runner.activity.note("operator", f"you: {text[:200]}")
        return {"task_id": task.id, "ran": False, "control": runner.control.to_dict()}

    payload = {**task.payload, "task": text, "text": text, "goal": text}

    claimed = await _claimed_recipe(text, task.recipe)
    if claimed is not None:
        nxt = runner.retry(task.id, payload=payload, recipe=claimed)
        return {"task_id": nxt.id, "ran": True, "task": nxt.to_dict()}

    # A deterministic recipe reads no instruction: it does the same thing to the
    # same site however the operator words it, so "don't use that site" would be
    # recorded and then ignored, and the attempt would hit the same wall. Those
    # recipes are exactly the ones the freeform agent exists to take over, so the
    # attempt becomes an agent run — which *does* read the instruction — while
    # anything that genuinely reads its payload (plan.task, agent.task, a stored
    # step recipe) is left alone.
    recipe = task.recipe
    if not _recipe_reads_payload(recipe):
        # Assignment, not setdefault: a URL the operator names must beat the one
        # the previous attempt left in the payload, or "go to <other site>
        # instead" would be recorded and then quietly ignored. _start_url_for
        # already carries the fallback chain, so an empty result means there is
        # genuinely nothing to start from and the agent's own error is the
        # honest outcome.
        url = _start_url_for(text, task)
        if url:
            payload["url"] = url
        recipe = AGENT_RECIPE

    nxt = runner.retry(task.id, payload=payload, recipe=recipe)
    return {"task_id": nxt.id, "ran": True, "task": nxt.to_dict()}


def _archived_brief(thread_id: str, at: float) -> str:
    """The same brief ``TaskRunner._thread_brief`` writes, built from records.

    Needed because after a deploy there is no task object to walk, and an
    "iterate on it" that does not carry the earlier failure forward is just a
    second identical roll of the dice — the whole reason the brief exists.

    ``at`` is inclusive, matching the live path: ``retry`` briefs the next
    attempt with the attempt it is replacing, so the run being answered is the
    one whose failure matters most. Corrections are filtered only against the
    thread's *first* attempt — the original ask is the instruction dated at its
    creation, everything later is a correction. In particular the say() that
    triggers this brief has just recorded the operator's message, dated now,
    and it binds the attempt that message creates; filtering against ``at``
    instead would drop exactly the words that matter most.
    """
    in_thread = runs.list(limit=200, thread_id=thread_id)
    earlier = sorted((r for r in in_thread if r.created_at <= at),
                     key=lambda r: r.created_at)
    first_created = min((r.created_at for r in in_thread), default=0.0)
    lines: list[str] = []
    for r in earlier:
        lines.append(f"- attempt {r.attempt} ({r.recipe}) ended {r.status}: "
                     f"{r.detail or 'no detail'}")
        for entry in r.activity[-3:]:
            lines.append(f"    · {entry.get('text', '')}")
    corrections = [
        m["text"] for m in runs.messages(thread_id)
        if m.get("kind") == "instruction" and m.get("role") == "operator"
        and m.get("text") and m.get("at", 0.0) > first_created
    ]
    return history_brief(lines, corrections=corrections)


async def _say_to_archived(task_id: str, req: SayRequest) -> dict[str, Any]:
    """``say`` for an attempt that only exists as a record.

    The run archive is what makes an old run studyable; letting the operator also
    *answer* it is what makes the study actionable, and it is the same endpoint
    either way so the UI never has to ask which kind of task it is looking at.
    """
    run = runs.get(task_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no such task")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    if req.kind not in ("instruction", "note"):
        raise HTTPException(status_code=400, detail="kind must be instruction or note")
    threads.say(run.thread_id, "operator", req.kind, text)

    # Deliberately the same shape as the live branch above, ``kind`` included: a
    # note queues an attempt either way. Branching on kind here and not there
    # would be a second rule about what a message means, differing only by
    # whether the pod happens to have restarted since the run.
    payload = {**run.payload, "task": text, "text": text, "goal": text}
    history = _archived_brief(run.thread_id, run.created_at)
    if history:
        payload["history"] = history

    claimed = await _claimed_recipe(text, run.recipe)
    if claimed is not None:
        nxt = runner.resubmit(
            thread_id=run.thread_id,
            attempt=run.attempt,
            recipe=claimed,
            payload=payload,
            parent_id=run.task_id,
        )
        return {"task_id": nxt.id, "ran": True, "task": nxt.to_dict()}

    recipe = run.recipe
    if not _recipe_reads_payload(recipe):
        # Same rule as a live retry: a deterministic recipe ignores the prose, so
        # the message becomes an agent run on this thread instead of a re-run of
        # the identical steps.
        url = _start_url_for(text, run)
        if url:
            payload["url"] = url
        recipe = AGENT_RECIPE

    nxt = runner.resubmit(
        thread_id=run.thread_id,
        attempt=run.attempt,
        recipe=recipe,
        payload=payload,
        parent_id=run.task_id,
    )
    return {"task_id": nxt.id, "ran": True, "task": nxt.to_dict()}


# -- login / takeover ------------------------------------------------------


@app.post("/api/login", dependencies=[Depends(require_token)])
async def start_login(req: LoginRequest) -> dict[str, Any]:
    """Open a URL in the live browser so the human can log in over noVNC.

    This is the bootstrap path: a fresh profile has no session, so the first
    thing it needs is a human to sign in once. After that the profile persists.
    """
    page = await session.goto(req.url)
    return {"opened": page.url, "takeover_url": "/vnc.html"}


@app.post("/api/browser/restart", dependencies=[Depends(require_token)])
async def restart_browser() -> dict[str, Any]:
    """Close and relaunch the browser behind noVNC.

    The window can be closed from inside the live view, which leaves the
    desktop empty and the profile lock behind. Recovery is automatic on the next
    task, but a human watching noVNC wants it now rather than after queueing
    something — and the session is one browser per profile, so this is also the
    honest "start over" when a page has wedged.
    """
    await session.stop()
    try:
        await session.start()
        page = await session.page()
        if page.url in ("", "about:blank"):
            await page.goto(_IDLE_PAGE)
    except Exception as exc:
        # SEC-BA-006: the exception text stays in the log, not the response.
        log.exception("browser restart failed")
        raise HTTPException(status_code=500, detail="browser restart failed") from exc
    return {"browser_running": session.is_running(), "url": page.url}


# -- accounts --------------------------------------------------------------


class AccountRequest(BaseModel):
    name: str
    label: str = ""
    email: str = ""
    notes: str = ""


class AccountSwitchRequest(BaseModel):
    name: str


@app.get("/api/accounts", dependencies=[Depends(require_token)])
async def list_accounts() -> dict[str, Any]:
    """Every account on this bot, and which of them is running.

    Cheap and browser-free, like /api/whoami: the UI polls it beside the state.
    """
    return store_accounts().to_dict()


@app.post("/api/accounts", dependencies=[Depends(require_token)])
async def create_account(req: AccountRequest) -> dict[str, Any]:
    """Add an account. It exists immediately but is not signed in.

    Creating one does *not* switch to it: signing in is a deliberate act over
    noVNC, and silently swapping the running identity under a bot that is
    mid-schedule would be the wrong default even though it is technically safe
    (the switch is serialised).
    """
    name = _clean_account(req.name)
    if not valid_account(name):
        raise HTTPException(status_code=400, detail="invalid account name")
    acc = store_accounts()
    if acc.get(name) is not None:
        raise HTTPException(status_code=409, detail="account already exists")
    acc.add(name, label=req.label, email=req.email, notes=req.notes)
    acc.save()
    return acc.to_dict()


@app.post("/api/accounts/switch", dependencies=[Depends(require_token)])
async def switch_account(req: AccountSwitchRequest) -> dict[str, Any]:
    """Make one account the running identity, restarting the browser onto it.

    Serialised against running tasks by the runner's lock: the switch waits for
    the current task to finish rather than tearing a browser out from under it,
    because two writers on one Chrome user-data-dir corrupt the profile.

    A failure to relaunch is reported but the *store* keeps the new choice: the
    operator asked for this account, and silently reverting would leave the UI
    disagreeing with the file. A restart from the UI is the retry path.
    """
    name = _clean_account(req.name)
    accounts = store_accounts()
    if accounts.get(name) is None:
        raise HTTPException(status_code=404, detail="no such account")

    async with runner.account_switch():
        accounts.set_active(name)
        accounts.save()
        session.set_account(name)
        await session.stop()
        error = ""
        page_url = ""
        try:
            await session.start()
            page = await session.page()
            if page.url in ("", "about:blank"):
                await page.goto(_IDLE_PAGE)
            page_url = page.url
        except Exception as exc:
            log.exception("browser could not start on account %s", name)
            error = str(exc)

    payload = accounts.to_dict()
    payload.update({
        "browser_running": session.is_running(),
        "url": page_url,
        "error": error,
    })
    return payload


@app.delete("/api/accounts/{name}", dependencies=[Depends(require_token)])
async def delete_account(name: str, purge: bool = False) -> dict[str, Any]:
    """Remove an account from the list, optionally deleting its profile dir.

    ``purge`` is the destructive half and defaults OFF: dropping the name is
    reversible (add it back and the Chrome directory is still there, still
    signed in), whereas deleting the directory *is* logging out of everything
    that identity holds. Deleting the running account is refused outright — the
    browser is holding that directory open.

    The last account is never removable: a bot with no account has no
    user-data-dir to launch, so every later task would fail with a confusing
    error instead of an honest "add an account first".
    """
    name = _clean_account(name)
    accounts = store_accounts()
    if accounts.get(name) is None:
        raise HTTPException(status_code=404, detail="no such account")
    if name == session.account:
        raise HTTPException(status_code=409, detail="account is running; switch away first")
    try:
        removed = accounts.remove(name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    accounts.save()
    purged = remove_account(settings.profile_root, removed.name) if purge else False
    payload = accounts.to_dict()
    payload["purged"] = purged
    return payload


def store_accounts() -> AccountStore:
    """Read this bot's accounts from disk, fresh.

    Read per request rather than cached: the file is a few lines, and a cache
    would have to be invalidated on every write — including writes by whatever
    the operator does by hand over kubectl exec, which is the whole reason the
    file is human-editable.
    """
    return load_accounts(settings.profile_root, settings.accounts_path)


def _clean_account(name: str) -> str:
    return str(name or "").strip().lower()


# -- schedules -------------------------------------------------------------


@app.post("/api/schedules", dependencies=[Depends(require_token)])
async def create_schedule(req: ScheduleRequest) -> dict[str, Any]:
    try:
        sched = store.add(req.id, req.recipe, req.payload, req.cron)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not req.enabled:
        sched = store.set_enabled(req.id, False)
    return sched.to_dict()


@app.delete("/api/schedules/{schedule_id}", dependencies=[Depends(require_token)])
async def delete_schedule(schedule_id: str) -> dict[str, Any]:
    if not store.delete(schedule_id):
        raise HTTPException(status_code=404, detail="no such schedule")
    return {"deleted": schedule_id}


@app.post("/api/schedules/{schedule_id}/toggle", dependencies=[Depends(require_token)])
async def toggle_schedule(schedule_id: str) -> dict[str, Any]:
    try:
        sched = store.get(schedule_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="no such schedule") from exc
    return store.set_enabled(schedule_id, not sched.enabled).to_dict()


# -- admin UI --------------------------------------------------------------

_UI = Path(__file__).parent / "ui" / "index.html"


def _prefix_script(value: str) -> str:
    """The inline bootstrap that hands the UI its prefix, escaped for its context.

    SEC-BA-005 (2026-09-27): the value lands inside a <script> block in served
    HTML, and json.dumps alone leaves "<" and ">" literal — so a crafted
    X-Forwarded-Prefix used to close the tag and run script on the bot's own
    origin. The capture middleware already allowlists the header; this escaping
    is the second half, so no path to the sink can emit markup.
    """
    escaped = (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return f'<script>window.BA_PREFIX={escaped};</script>'


@app.get("/", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    # The page is served from the SAME prefix it will talk to, so the prefix has
    # to be in it before the first request goes out: the UI fetches "/api/state"
    # root-absolute, and under a roster that resolves against the ORIGIN — the
    # hub — not this bot. It would then render the roster's state (no profile,
    # no tasks) while looking perfectly healthy. Handing it the prefix here is
    # what makes the same file work both standalone and behind /b/<bot>.
    # SEC-BA-005: the prefix is escaped for the JS-in-HTML context even after
    # the capture-side allowlist, so a value that closes the script tag cannot
    # reach the page through any path (json.dumps leaves "<" and ">" literal).
    html = _UI.read_text().replace("<!--PREFIX-->", _prefix_script(_REACHED_PREFIX.get()))
    return HTMLResponse(html)


@app.get("/vnc.html", response_class=HTMLResponse)
async def vnc_redirect() -> HTMLResponse:
    return HTMLResponse(
        "<html><body style='margin:0;background:#111;color:#ddd;font:14px system-ui'>"
        "<p style='padding:12px'>noVNC is served by the pod's websockify on port "
        f"{settings.novnc_port}. If you reached this page through a proxy that does not "
        "forward that port, open it directly.</p></body></html>"
    )


_MS_LOCAL = Path(__file__).parent / "minesweeper_local.html"


@app.get("/minesweeper.html", response_class=HTMLResponse)
async def minesweeper_local() -> HTMLResponse:
    """The local Beginner board the minesweeper recipe plays by default.

    minesweeper.online IP-blocked the pod's egress (2026-09-25); this page
    mimics its DOM contract so the same solver drives either one.
    """
    return HTMLResponse(_MS_LOCAL.read_text())


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    # SEC-BA-006: the response carries only a generic body plus a request id
    # the log line shares — str(exc) has leaked subprocess stderr and library
    # internals to whoever was calling. The detail stays here, in the log.
    rid = uuid.uuid4().hex[:12]
    log.exception("unhandled error on %s [req=%s]", request.url.path, rid)
    return JSONResponse(
        status_code=500, content={"detail": "internal error", "request_id": rid}
    )
