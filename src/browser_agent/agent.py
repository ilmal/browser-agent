"""The LLM fallback: hand the task to browser-use when a recipe breaks.

The agent is given the *same* persistent profile and renders to the same Xvfb
display, so a human watching over noVNC sees the agent working and can take
over mid-flight. That shared-session property is the whole point: it is what
makes "agent gets stuck, human helps" a one-screen operation.

browser-use is an optional dependency (the `agent` extra). If it is absent the
task fails cleanly rather than silently doing nothing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, NoReturn

from .activity import activity_of
from .browser import BrowserSession
from .config import Settings
from .escalation import Challenge, ChallengeKind, EscalationRequired, detect_challenge

log = logging.getLogger(__name__)

_TASK_TEMPLATE = """\
You are operating a real browser session on behalf of its owner.

Goal: {goal}

Starting page: {url}

Context: {payload}
{history}
Rules you must follow:
- If you see a captcha, an SMS/email code prompt, an "unusual activity" or
  "verify it's you" warning, or you are asked to confirm your identity, STOP
  immediately. Do not attempt it, do not try to work around it, and do not
  retry. Report that a human is needed.
- Do not create accounts, add payment methods, change passwords or email
  addresses, or accept any terms on the owner's behalf.
- Do not send direct messages to individuals or post comments on other
  people's content.
- Stay on the target site. Do not navigate elsewhere.
- Perform the goal once, then stop.
"""


def _load_browser_use():
    """Import browser-use, tolerating the two module layouts it has shipped."""
    try:
        from browser_use import Agent, Browser  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "browser-use is not installed; install the 'agent' extra "
            "(uv sync --extra agent)"
        ) from exc

    try:
        from browser_use import ChatOpenAI  # type: ignore
    except ImportError:
        try:
            from browser_use.llm import ChatOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("could not locate browser-use's ChatOpenAI client") from exc

    return Agent, Browser, ChatOpenAI


def _build_gateway_llm(settings: Settings, ChatOpenAI):
    """A browser-use chat client that satisfies llm-service's caller contract.

    browser-use's ChatOpenAI can set static headers (``default_headers``) but
    has no way to put ``client_id`` in the request body: it rejects an unknown
    ``client_id`` kwarg with TypeError, its ``model_params`` is a closed set, and
    ``ainvoke(**kwargs)`` accepts and silently drops extras. llm-service reads
    the client id from the body only (OpenAI ``user``), with no header fallback,
    so the id has to be injected at the one point that controls the wire format.
    """

    client = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        # A real key is required; `api_key="not-required"` was the 403.
        api_key=settings.llm_api_key,
        default_headers={"X-LLM-Source": settings.llm_source},
    )

    original_get_client = client.get_client

    def get_client(*args, **kwargs):
        underlying = original_get_client(*args, **kwargs)
        create = underlying.chat.completions.create

        async def create_with_client_id(*a, **kw):
            kw.setdefault("user", settings.llm_client_id)
            return await create(*a, **kw)

        underlying.chat.completions.create = create_with_client_id
        return underlying

    client.get_client = get_client
    return client


async def _current_url(session: BrowserSession) -> str:
    """The page URL, or "" — a feed annotation must never fail the run."""
    try:
        return (await session.page()).url
    except Exception:
        return ""


def _record_agent_step(agent: Any, log_: Any) -> None:
    """Turn one finished agent step into a line a human can read.

    browser-use reports what the model decided and what happened in separate
    places, so this reads both: ``last_model_output`` carries the model's own
    next_goal (the useful "why"), and ``last_result`` the outcome.
    """
    try:
        output = agent.state.last_model_output
        goal = getattr(output, "next_goal", "") or ""
        results = agent.state.last_result or []
        actions: list[str] = []
        for r in results:
            if getattr(r, "error", None):
                actions.append(f"failed: {r.error}")
            elif getattr(r, "extracted_content", None):
                actions.append(str(r.extracted_content)[:200])
            elif getattr(r, "is_done", False):
                actions.append("done")
            else:
                actions.append("ok")
        text = goal or "step"
        if actions:
            text += " → " + "; ".join(actions)
        log_.note("agent", text, step=agent.state.n_steps)
    except Exception:  # pragma: no cover - defensive; the feed is not load-bearing
        log.debug("could not record agent step", exc_info=True)


def _raise_for_no_result(history: Any, url: str) -> NoReturn:
    """Classify an agent run that produced no result.

    Only "the agent could not do it" is a failure a human need not act on. An
    unreachable LLM or an errored run raises RuntimeError (FAILED); anything
    else means the agent stopped for a reason a person must resolve, which is
    BLOCKED. Never silently report success.
    """
    if history.is_successful() is False or history.has_errors():
        reasons = "; ".join(str(e) for e in (history.errors() or []) if e)
        raise RuntimeError(
            f"agent could not complete the task: {reasons[:300] or 'no reason given'}"
        )
    raise EscalationRequired(
        Challenge(ChallengeKind.UNKNOWN, "agent finished without a result", url)
    )


def make_agent_runner(settings: Settings):
    """Build the async runner the TaskRunner calls after a recipe failure."""

    async def run(session: BrowserSession, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        Agent, Browser, ChatOpenAI = _load_browser_use()

        goal = payload.get("goal") or payload.get("text") or "Complete the task"
        # The runner's thread briefing, when this is a retry: what the earlier
        # attempts tried and why they stopped. Its own section, and out of the
        # generic dump — a repr of the raw list would be worse than useless.
        history = str(payload.get("history") or "").strip()
        task_prompt = _TASK_TEMPLATE.format(
            goal=goal,
            url=url,
            history=f"\n{history}\n" if history else "",
            payload={
                k: v for k, v in payload.items()
                if k not in {"goal", "text", "history", "entry_url"}
            },
        )

        # Attach to the browser the recipe was already using rather than
        # launching a second one. Chrome permits a second context on the same
        # user_data_dir — it simply starts logged out — so a fresh launch would
        # leave the agent driving a different, unauthenticated browser while the
        # human watches the real one over noVNC.
        #
        # Ensure one is actually up first. Closing the browser window used to
        # leave a stale DevTools port file behind, which the agent then attached
        # to and failed with "All connection attempts failed" — a recoverable
        # state reported as a broken task. start() is idempotent and relaunches
        # when the previous browser is gone, so a closed window now just means
        # the next run reopens it. Tolerated when absent: a session that cannot
        # start (a stub in a test) still exposes cdp_endpoint, and failing here
        # would report "no browser" for a reason that is not ours to diagnose.
        start = getattr(session, "start", None)
        if start is not None:
            await start()
        cdp_url = session.cdp_endpoint
        if not cdp_url:
            raise RuntimeError(
                "no DevTools endpoint for the running browser; the agent must "
                "attach to the session a human can see"
            )

        if not settings.llm_api_key:
            # Fail with the cause rather than letting llm-service answer 403 and
            # reporting it as "the agent could not complete the task".
            raise RuntimeError("LLM_API_KEY is not set; llm-service rejects every call")

        llm = _build_gateway_llm(settings, ChatOpenAI)
        browser = Browser(cdp_url=cdp_url, is_local=True, headless=settings.headless)
        await browser.connect()

        agent = Agent(task=task_prompt, llm=llm, browser=browser)

        # The live feed and the operator's controls ride on the agent's own step
        # hooks. browser-use calls on_step_start BEFORE its step try/except
        # (verified in _execute_step), so an exception raised here unwinds the
        # whole run rather than being swallowed as a failed step — which is
        # exactly what a stop or an amendment needs.
        log_ = activity_of(session)
        control = getattr(session, "control", None)

        async def on_step_start(a) -> None:
            if control is not None:
                await control.checkpoint(await _current_url(session))
            log_.note("agent", f"step {a.state.n_steps}: thinking")

        async def on_step_end(a) -> None:
            _record_agent_step(a, log_)
        log.info(
            "agent fallback attaching to %s for %s (max_steps=%d, timeout=%ds)",
            cdp_url,
            url,
            settings.agent_max_steps,
            settings.agent_timeout_s,
        )
        try:
            # browser-use defaults max_steps to 500, so an unbounded run is the
            # default: a task with nothing left to do loops until the model
            # happens to stop, and since the queue has ONE worker every later
            # task sits QUEUED behind it forever. The step cap ends the loop;
            # the wall clock is the backstop for a step hung in the browser.
            history = await asyncio.wait_for(
                agent.run(
                    max_steps=settings.agent_max_steps,
                    on_step_start=on_step_start,
                    on_step_end=on_step_end,
                ),
                timeout=settings.agent_timeout_s,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"agent exceeded its {settings.agent_timeout_s}s budget"
            ) from None
        finally:
            # Detach only. Killing the browser here would close the tab the
            # recipe owns and take down the human's noVNC view with it.
            try:
                await browser.close()
            except Exception:  # pragma: no cover - best-effort detach
                log.debug("browser-use detach failed", exc_info=True)

        # Check what the agent left on screen before trusting its report: it
        # may have hit a challenge on its last step.
        page = await session.page()
        challenge = await detect_challenge(page)
        if challenge is not None:
            raise EscalationRequired(challenge)

        result = history.final_result()
        if result is not None:
            log.info("agent fallback finished")
            return {"agent_result": str(result), "url": page.url}

        _raise_for_no_result(history, page.url)

    return run
