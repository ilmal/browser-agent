"""Control-plane API + admin UI.

Each profile pod serves this on its own port. The admin UI is a single page
served from here, so there is no separate frontend build to keep in sync.
"""

from __future__ import annotations

import contextlib
import json
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import recipes  # noqa: F401  (importing registers every built-in recipe)
from .agent import make_agent_runner
from .browser import BrowserSession, profile_exists
from .config import Settings, load_settings
from .escalation import detect_challenge
from .scheduler import ScheduleStore
from .tasks import (
    AGENT_RECIPE,
    TaskRunner,
    TaskStatus,
    list_recipes,
    recipe_reads_instruction,
)
from .threads import ThreadStore

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings: Settings = load_settings()
session = BrowserSession(settings)
store = ScheduleStore(settings.state_db)
threads = ThreadStore()
runner = TaskRunner(settings, session, agent_runner=make_agent_runner(settings))

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
                task = runner.submit(sched.recipe, sched.payload)
                store.mark_run(sched.id, task.id)
                log.info("schedule %s fired task %s", sched.id, task.id)
            except Exception:
                log.exception("schedule %s could not be fired", sched.id)
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

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


@app.middleware("http")
async def _capture_prefix(request: Request, call_next):
    token = _REACHED_PREFIX.set(
        request.headers.get("x-forwarded-prefix", "").rstrip("/") or settings.url_prefix
    )
    try:
        return await call_next(request)
    finally:
        _REACHED_PREFIX.reset(token)


# -- auth ------------------------------------------------------------------


async def require_token(request: Request) -> None:
    """Bearer auth. Disabled only when no CONTROL_TOKEN is configured."""
    if not settings.control_token:
        return
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
        "profile_initialised": profile_exists(settings),
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
    """
    return {
        "profile": settings.profile,
        "display_name": settings.display_name,
        "job": settings.display_job,
        "profile_initialised": profile_exists(settings),
        "url_prefix": _REACHED_PREFIX.get(),
    }


@app.get("/api/state", dependencies=[Depends(require_token)])
async def state() -> dict[str, Any]:
    """Everything the admin UI needs, in one call."""
    tasks = sorted(runner.tasks.values(), key=lambda t: t.created_at, reverse=True)[:50]
    current = runner.current.to_dict() if runner.current else None
    if current is not None and runner.control is not None:
        # Rides on the task because the UI's question is "what is this task
        # doing", and the controls only exist while one is running.
        current["control"] = runner.control.to_dict()
    return {
        "profile": settings.profile,
        "profile_initialised": profile_exists(settings),
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
        "tasks": [t.to_dict() for t in tasks],
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
    try:
        task = runner.submit(req.recipe, req.payload)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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


def _last_url(task: Any) -> str:
    """The URL the previous attempt was working on, as a start point.

    Better than the recipe's ``entry_url`` when the operator is redirecting the
    task: it is where the work actually is, so "go to another site" begins from
    a real page rather than the site that just blocked us.
    """
    for source in (task.result or {}, task.payload, {"detail": task.detail or ""}):
        url = source.get("url") if isinstance(source, dict) else None
        if isinstance(url, str) and url.startswith("http"):
            return url
    return ""


class SayRequest(BaseModel):
    text: str
    #: "instruction" changes the wording for the next attempt; "note" only
    #: records it in the thread. Parameters and config edits go through their
    #: own endpoints so the shared-library blast radius is stated once, in the
    #: confirm the operator actually reads.
    kind: str = "instruction"


@app.get("/api/threads/{thread_id}", dependencies=[Depends(require_token)])
async def get_thread(thread_id: str) -> dict[str, Any]:
    """One thread: every attempt in it, the messages, and the live feed.

    The attempts carry their own snapshotted activity, so the panel can show
    what every earlier attempt tried without a second request per attempt.
    """
    members = sorted(
        (t for t in runner.tasks.values() if t.thread_id == thread_id),
        key=lambda t: t.created_at,
    )
    if not members:
        raise HTTPException(status_code=404, detail="no such thread")
    return {
        "thread_id": thread_id,
        "attempts": [t.to_dict() for t in members],
        "messages": [m.to_dict() for m in threads.for_thread(thread_id)],
        "current": runner.current.to_dict() if runner.current else None,
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
        raise HTTPException(status_code=404, detail="no such task")
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

    # A deterministic recipe reads no instruction: it does the same thing to the
    # same site however the operator words it, so "don't use that site" would be
    # recorded and then ignored, and the attempt would hit the same wall. Those
    # recipes are exactly the ones the freeform agent exists to take over, so the
    # attempt becomes an agent run — which *does* read the instruction — while
    # anything that genuinely reads its payload (plan.task, agent.task, a stored
    # step recipe) is left alone.
    recipe = task.recipe
    if not _recipe_reads_payload(recipe):
        payload.setdefault("url", _last_url(task))
        recipe = AGENT_RECIPE

    nxt = runner.retry(task.id, payload=payload, recipe=recipe)
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
        log.exception("browser restart failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"browser_running": session.is_running(), "url": page.url}


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


@app.get("/", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    # The page is served from the SAME prefix it will talk to, so the prefix has
    # to be in it before the first request goes out: the UI fetches "/api/state"
    # root-absolute, and under a roster that resolves against the ORIGIN — the
    # hub — not this bot. It would then render the roster's state (no profile,
    # no tasks) while looking perfectly healthy. Handing it the prefix here is
    # what makes the same file work both standalone and behind /b/<bot>.
    html = _UI.read_text().replace(
        "<!--PREFIX-->", f'<script>window.BA_PREFIX={json.dumps(_REACHED_PREFIX.get())};</script>'
    )
    return HTMLResponse(html)


@app.get("/vnc.html", response_class=HTMLResponse)
async def vnc_redirect() -> HTMLResponse:
    return HTMLResponse(
        "<html><body style='margin:0;background:#111;color:#ddd;font:14px system-ui'>"
        "<p style='padding:12px'>noVNC is served by the pod's websockify on port "
        f"{settings.novnc_port}. If you reached this page through a proxy that does not "
        "forward that port, open it directly.</p></body></html>"
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})
