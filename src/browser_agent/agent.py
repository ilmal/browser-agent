"""The LLM fallback: hand the task to browser-use when a recipe breaks.

The agent is given the *same* persistent profile and renders to the same Xvfb
display, so a human watching over noVNC sees the agent working and can take
over mid-flight. That shared-session property is the whole point: it is what
makes "agent gets stuck, human helps" a one-screen operation.

browser-use is an optional dependency (the `agent` extra). If it is absent the
task fails cleanly rather than silently doing nothing.
"""

from __future__ import annotations

import logging
from typing import Any

from .browser import BrowserSession
from .config import Settings
from .escalation import Challenge, ChallengeKind, EscalationRequired, detect_challenge

log = logging.getLogger(__name__)

_TASK_TEMPLATE = """\
You are operating a real browser session on behalf of its owner.

Goal: {goal}

Starting page: {url}

Context: {payload}

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


def make_agent_runner(settings: Settings):
    """Build the async runner the TaskRunner calls after a recipe failure."""

    async def run(session: BrowserSession, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        Agent, Browser, ChatOpenAI = _load_browser_use()

        goal = payload.get("goal") or payload.get("text") or "Complete the task"
        task_prompt = _TASK_TEMPLATE.format(
            goal=goal,
            url=url,
            payload={k: v for k, v in payload.items() if k not in {"goal", "text"}},
        )

        llm = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key="not-required",
        )
        # Same profile, same display: the human can watch and intervene.
        browser = Browser(
            user_data_dir=str(settings.profile_dir),
            headless=settings.headless,
        )

        agent = Agent(task=task_prompt, llm=llm, browser=browser)
        log.info("agent fallback starting for %s", url)
        history = await agent.run()

        # Check what the agent left on screen before trusting its report: it
        # may have hit a challenge on its last step.
        page = await session.page()
        challenge = await detect_challenge(page)
        if challenge is not None:
            raise EscalationRequired(challenge)

        result = history.final_result()
        if result is None:
            raise EscalationRequired(
                Challenge(
                    ChallengeKind.UNKNOWN,
                    "agent finished without a result",
                    page.url,
                )
            )
        log.info("agent fallback finished")
        return {"agent_result": str(result), "url": page.url}

    return run
