"""Control-plane API + admin UI.

Each profile pod serves this on its own port. The admin UI is a single page
served from here, so there is no separate frontend build to keep in sync.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
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
from .tasks import TaskRunner, TaskStatus, list_recipes

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings: Settings = load_settings()
session = BrowserSession(settings)
store = ScheduleStore(settings.state_db)
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
        try:
            for sched in store.due():
                task = runner.submit(sched.recipe, sched.payload)
                store.mark_run(sched.id, task.id)
                log.info("schedule %s fired task %s", sched.id, task.id)
        except Exception:
            log.exception("schedule loop iteration failed")
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


# -- status ----------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "profile": settings.profile,
        "profile_initialised": profile_exists(settings),
        "llm_enabled": runner.llm.enabled,
    }


@app.get("/api/state", dependencies=[Depends(require_token)])
async def state() -> dict[str, Any]:
    """Everything the admin UI needs, in one call."""
    tasks = sorted(runner.tasks.values(), key=lambda t: t.created_at, reverse=True)[:50]
    return {
        "profile": settings.profile,
        "profile_initialised": profile_exists(settings),
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
        "current": runner.current.to_dict() if runner.current else None,
        "tasks": [t.to_dict() for t in tasks],
        "schedules": [s.to_dict() for s in store.all()],
        "queue_depth": runner.queue.qsize(),
        "takeover_url": "/vnc.html",
    }


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
async def retry_task(task_id: str) -> dict[str, Any]:
    """Re-queue a task. Intended for use after a human has cleared a block."""
    if task_id not in runner.tasks:
        raise HTTPException(status_code=404, detail="no such task")
    if runner.tasks[task_id].status is TaskStatus.RUNNING:
        raise HTTPException(status_code=409, detail="task is still running")
    return runner.retry(task_id).to_dict()


# -- login / takeover ------------------------------------------------------


@app.post("/api/login", dependencies=[Depends(require_token)])
async def start_login(req: LoginRequest) -> dict[str, Any]:
    """Open a URL in the live browser so the human can log in over noVNC.

    This is the bootstrap path: a fresh profile has no session, so the first
    thing it needs is a human to sign in once. After that the profile persists.
    """
    page = await session.goto(req.url)
    return {"opened": page.url, "takeover_url": "/vnc.html"}


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
    return HTMLResponse(_UI.read_text())


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
