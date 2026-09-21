"""The deterministic plan executor.

Runs a validated :class:`~browser_agent.plan_model.Plan` against the live page.
Per step the gates run in a fixed order:

1. the action itself (explicit selector first, Laya picker as resolver),
2. ``require_clear`` — challenge detection; may raise EscalationRequired, the
   only path to a human,
3. ``done_when`` — the authoritative proof a step worked,
4. Laya confirmation — only for click/type steps with no ``done_when``.

A step that fails any gate raises :class:`StepFailure`, which the runner hands
to the agent fallback together with the original task text — the agent's job
is to finish the task, not to retry the broken step blind.
"""

from __future__ import annotations

import logging

from playwright.async_api import Locator, Page

from ..activity import activity_of
from ..browser import BrowserSession
from ..config import Settings
from ..escalation import EscalationRequired  # noqa: F401  (re-raised, never caught here)
from ..laya_gate import LayaGate
from ..picker import ElementPicker
from ..plan_model import DoneWhen, Plan, Step, StepFailure
from ._candidates import CLICK_BASE, TYPE_BASE, extract_candidates
from ._helpers import require_clear

log = logging.getLogger(__name__)


async def state_text(page: Page, limit: int = 900) -> str:
    """Short text snapshot of the page for the Laya gate."""
    try:
        title = await page.title()
    except Exception:
        title = ""
    try:
        body = (await page.inner_text("body"))[:limit]
    except Exception:
        body = ""
    return f"{title}\n{body}".strip()


async def check_done_when(page: Page, dw: DoneWhen) -> bool:
    """Every predicate present must hold. A broken check reads as unmet."""
    try:
        if dw.url_contains and dw.url_contains.lower() not in page.url.lower():
            return False
        if dw.selector_visible:
            loc = page.locator(dw.selector_visible).first
            if await loc.count() == 0 or not await loc.is_visible():
                return False
        if dw.text_contains:
            body = (await page.inner_text("body"))[:20_000].lower()
            if dw.text_contains.lower() not in body:
                return False
    except Exception:
        return False
    return True


# Host-level interface churn (vpn/tailscale/nic) aborts an in-flight navigation
# with these Chrome errors. The page is fine; one retry is cheaper than burning
# the whole plan into the agent fallback for a flake the host caused.
_TRANSIENT_NET_ERRORS = ("net::ERR_NETWORK_CHANGED", "net::ERR_CONNECTION_RESET")


async def _retry_transient(op, *, what: str):
    for attempt in range(2):
        try:
            return await op()
        except Exception as exc:
            if attempt or not any(t in str(exc) for t in _TRANSIENT_NET_ERRORS):
                raise
            log.warning("transient network error during %s, retrying once: %s", what, exc)


async def resolve_target(
    page: Page,
    step: Step,
    laya: LayaGate,
    settings: Settings,
    *,
    base: str,
    picker: ElementPicker | None = None,
) -> Locator:
    """The element a click/type step acts on.

    Explicit selector first (deterministic beats clever). Without one, the
    pickers choose among the page's visible candidates and the executor binds
    by index — never by re-matching text, which identical buttons would make
    ambiguous. Laya is asked first (cheap, and decisive on single-candidate
    pages); the LLM picker answers when laya declines.
    """
    timeout_ms = settings.planner_step_timeout_s * 1000
    if step.selector:
        loc = page.locator(step.selector).first
        try:
            await loc.wait_for(state="visible", timeout=timeout_ms)
        except Exception as exc:
            raise StepFailure(f"selector never became visible: {step.selector!r}") from exc
        return loc

    if not laya.pick_enabled and not (picker is not None and picker.enabled):
        raise StepFailure(f"no selector and no picker is enabled for: {step.goal!r}")
    floor = settings.laya_min_confidence
    for _ in range(settings.laya_pick_retries + 1):
        loc, lines = await extract_candidates(page, base, settings.laya_max_candidates)
        if not lines:
            break
        state = f"Goal: {step.goal}\n{await state_text(page, 400)}"
        # Laya first: ~50 ms warm vs ~1 s for the LLM picker. It stays in the
        # chain even though the bench shows it cannot clear the floor on
        # multi-candidate pages — the single-candidate case it can do, and the
        # LLM picker only pays when laya declines.
        if laya.pick_enabled:
            idx, conf = await laya.choose(
                "Which numbered element should be activated to accomplish the page goal?",
                lines,
                state,
            )
            if idx is not None and conf >= floor:
                return loc.nth(idx)
        if picker is not None and picker.enabled:
            idx, _ = await picker.pick(step.goal or "", state, lines)
            if idx is not None:
                return loc.nth(idx)
    raise StepFailure(f"no element picked for: {step.goal!r}")


async def exec_step(
    page: Page,
    step: Step,
    laya: LayaGate,
    settings: Settings,
    extracts: dict[str, str],
    picker: ElementPicker | None = None,
) -> str:
    timeout_ms = settings.planner_step_timeout_s * 1000
    if step.action == "navigate":
        async def _go() -> None:
            await page.goto(step.text or "", wait_until="domcontentloaded", timeout=timeout_ms)

        await _retry_transient(_go, what=f"navigate to {step.text!r}")
        return f"navigated to {page.url}"
    if step.action == "click":
        target = await resolve_target(page, step, laya, settings, base=CLICK_BASE, picker=picker)

        async def _click() -> None:
            await target.click(timeout=timeout_ms)

        await _retry_transient(_click, what=f"click for {step.goal!r}")
        return f"clicked for: {step.goal or step.selector}"
    if step.action == "type":
        target = await resolve_target(page, step, laya, settings, base=TYPE_BASE, picker=picker)
        await target.fill(step.text or "")
        return f"typed into: {step.goal or step.selector}"
    if step.action == "extract":
        texts = await page.locator(step.selector or "body").all_inner_texts()
        key = step.goal or step.selector or "body"
        extracts[key] = "\n".join(t.strip() for t in texts if t.strip())[:2_000]
        return f"extracted {len(texts)} node(s) into {key!r}"
    if step.action == "wait":
        await page.wait_for_selector(step.selector or "body", timeout=timeout_ms)
        return f"waited for {step.selector}"
    raise StepFailure(f"unknown action {step.action!r}")  # pydantic Literal guards this


async def confirm_step(
    page: Page, laya: LayaGate, settings: Settings, task_text: str, desc: str
) -> bool:
    """Laya yes/no gate: advance only on a confident yes; one re-ask when unsure."""
    for _ in range(settings.laya_pick_retries + 1):
        verdict, conf = await laya.yes_no(
            f"Did executing '{desc}' move the page closer to completing the overall "
            f"task '{task_text}'?",
            await state_text(page),
        )
        if conf >= settings.laya_min_confidence and verdict is not None:
            return verdict
    return False


async def run_plan(
    session: BrowserSession,
    plan: Plan,
    task_text: str,
    settings: Settings,
    laya: LayaGate,
    picker: ElementPicker | None = None,
) -> dict:
    log_ = activity_of(session)
    log_.note("info", f"plan: {len(plan.steps)} step(s), entry {plan.entry_url}")
    page = await session.goto(plan.entry_url)
    extracts: dict[str, str] = {}
    executed = 0

    for i, step in enumerate(plan.steps):
        desc = step.goal or f"{step.action} {step.selector or step.text or ''}".strip()
        # Between steps, not during one: the operator's pause/stop/amend land
        # here, where neither Playwright nor the LLM is mid-action.
        control = getattr(session, "control", None)
        if control is not None:
            await control.checkpoint(page.url)
        log_.note("step", f"{i + 1}/{len(plan.steps)} {step.action}: {desc}", step=i + 1)
        try:
            outcome = await exec_step(page, step, laya, settings, extracts, picker=picker)
        except StepFailure as exc:
            log_.note("error", f"step {i + 1} failed: {exc}")
            raise StepFailure(f"step {i} ({desc}): {exc}", task_text, plan.entry_url) from exc
        except Exception as exc:
            log_.note("error", f"step {i + 1} failed: {exc}")
            raise StepFailure(f"step {i} ({desc}): {exc}", task_text, plan.entry_url) from exc
        log.info("plan step %d ok: %s", i, outcome)
        log_.note("step", f"step {i + 1} ok: {outcome}", step=i + 1)

        # A captcha between steps stops the run — never stepped over, never retried.
        await require_clear(page)

        if step.done_when is not None:
            if not await check_done_when(page, step.done_when):
                raise StepFailure(
                    f"step {i} done_when unmet after: {desc}", task_text, plan.entry_url
                )
        elif step.action in {"click", "type"} and laya.enabled:
            if not await confirm_step(page, laya, settings, task_text, desc):
                raise StepFailure(
                    f"step {i}: not confirmed by the gate after: {desc}",
                    task_text,
                    plan.entry_url,
                )
        executed += 1

    return {
        "plan": plan.model_dump(),
        "steps_executed": executed,
        "extracts": extracts,
        "final_url": page.url,
        "gate": laya.summary,
    }
