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

import json
import logging
import random
import re
import time
import uuid
from pathlib import Path

from playwright.async_api import Locator, Page

from ..activity import activity_of
from ..browser import BrowserSession
from ..config import Settings
from ..escalation import EscalationRequired  # noqa: F401  (re-raised, never caught here)
from ..laya_gate import LayaGate
from ..page_state import fingerprint as _fingerprint
from ..page_state import state_text, wait_stable as _wait_stable
from ..picker import ElementPicker
from ..plan_model import DoneWhen, Plan, Step, StepFailure
from ._candidates import (
    ELEMENT_INFO as _ELEMENT_INFO,
    CLICK_BASE,
    TYPE_BASE,
    PICKER_CAP,
    Candidates,
    extract_candidates,
)
from ._helpers import require_clear

log = logging.getLogger(__name__)

#: Risk tiers (trycua/cua's confirmation policy, tightened to what can do
#: damage from here): words naming money, destruction or outbound publication.
#: A click/type whose wording hits this and which carries NO done_when is
#: refused before it acts — the planner was told to attach proof to exactly
#: these steps, so a risky step without proof is a plan defect, and the
#: remedy is the agent fallback, not an unverified destructive action.
#: Deliberately excludes "submit"/"confirm" (every form's happy path says
#: them) — this gate is for irreversible, not for routine.
_RISKY_RE = re.compile(
    r"\b(pay|payment|paying|checkout|purchase|purchasing|buy|buying|order"
    r"|delete|deleting|remove|removing|transfer|password"
    r"|send|sending|post|posting|publish|publishing)\b",
    re.IGNORECASE,
)


def _is_risky_unproven(step: Step) -> bool:
    if step.done_when is not None:
        return False
    haystack = " ".join(
        part for part in (step.goal, step.selector, step.text) if part
    )
    return bool(_RISKY_RE.search(haystack))


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


async def _fresh(cands: Candidates, idx: int) -> bool:
    """Re-check the picked element's identity right before dispatch.

    Between enumeration and the model's answer (~1 s for the LLM picker) a
    dynamic page can re-render and shift the match order; ``locator.nth``
    re-resolves at dispatch and would silently click a different element.
    jev-ultrafast refuses a stale decision for the same reason. A mismatch
    costs one re-enumeration, not a wrong click.
    """
    if not cands.idents or cands.idents[idx] == "":
        return True
    try:
        info = await cands.locator.nth(cands.binds[idx]).evaluate(_ELEMENT_INFO)
    except Exception:
        return False
    return bool(info) and info.get("ident") == cands.idents[idx]

#: The pick flywheel: one JSONL row per pick — the raw page context, the lines
#: as the model saw them, and the choice (laya-browser's DAgger logging rule:
#: log the page, not the prompt, so training can re-render in any future
#: format). Rotated at 5 MB; never allowed to break a run.
_PICK_LOG_MAX = 5_000_000


def _log_pick(settings: Settings, row: dict) -> None:
    try:
        path = Path(settings.data_root) / "picks.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > _PICK_LOG_MAX:
            path.rename(path.with_suffix(".jsonl.1"))
        with path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - best effort only
        log.debug("pick log unavailable (%s)", exc)


def _line_body(line: str) -> str:
    """The line minus its leading number — the element's description."""
    return re.sub(r"^\d+\.\s*", "", line)


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
    pages); the LLM picker answers when laya declines. A pick is bound only
    if the element is still the one the model chose (``_fresh``); otherwise
    the loop re-enumerates and asks again.
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
    mode = "type" if base == TYPE_BASE else "click"
    for _ in range(settings.laya_pick_retries + 1):
        cands = await extract_candidates(page, base, PICKER_CAP, mode=mode)
        if not cands.lines:
            break
        state = f"Goal: {step.goal}\n{await state_text(page, 400)}"
        # Laya first: ~50 ms warm vs ~1 s for the LLM picker. It stays in the
        # chain even though the bench shows it cannot clear the floor on
        # multi-candidate pages — the single-candidate case it can do, and the
        # LLM picker only pays when laya declines. Laya sees only its
        # calibrated bucket range (LAYA_MAX_CANDIDATES); the numbering is
        # shared, so an index answers for the same line either way.
        if laya.pick_enabled:
            idx, conf = await laya.choose(
                "Which numbered element should be activated to accomplish the page goal?",
                cands.lines[: settings.laya_max_candidates],
                state,
            )
            if idx is not None and conf >= floor and await _fresh(cands, idx):
                return cands.locator.nth(cands.binds[idx])
        if picker is not None and picker.enabled:
            idx, conf = await picker.pick(step.goal or "", state, cands.lines, op=mode)
            verified: bool | None = None
            if idx is not None and settings.picker_verify and len(cands.lines) > 1:
                # Agree-by-two: ask again with the order shuffled. A first
                # answer that survives re-asking in a different position is
                # evidence about the ELEMENT; one that flips with the order
                # was position bias. The second ask failing (transport) does
                # not discard the first — only a disagreement does.
                order = list(range(len(cands.lines)))
                random.shuffle(order)
                idx2, _ = await picker.pick(
                    step.goal or "",
                    state,
                    [cands.lines[k] for k in order],
                    op=mode,
                )
                if idx2 is not None:
                    verified = _line_body(cands.lines[idx]) == _line_body(
                        cands.lines[order[idx2]]
                    )
                    if not verified:
                        log.warning(
                            "picker verify disagreed (%r vs %r); declining the pick",
                            _line_body(cands.lines[idx]),
                            _line_body(cands.lines[order[idx2]]),
                        )
                        idx = None
            if idx is not None:
                if not await _fresh(cands, idx):
                    log.warning("picked element went stale; re-enumerating")
                    continue
                _log_pick(
                    settings,
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "op": mode,
                        "goal": step.goal or "",
                        "url": page.url,
                        "state": state,
                        "lines": cands.lines,
                        "chosen": idx,
                        "chosen_line": cands.lines[idx],
                        "conf": conf,
                        "verified": verified,
                        "model": settings.picker_model,
                    },
                )
                return cands.locator.nth(cands.binds[idx])
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
    mutated = False
    # Stuck-loop latches (ported from jev-ultrafast / SystemOneHarness):
    # consecutive click/type actions that change nothing, and the same action
    # on the same page state twice, both mean the plan is grinding — fail into
    # the agent fallback while there is still budget to repair.
    no_change = 0
    seen: set[tuple[str, str, str]] = set()

    for i, step in enumerate(plan.steps):
        desc = step.goal or f"{step.action} {step.selector or step.text or ''}".strip()
        # Between steps, not during one: the operator's pause/stop/amend land
        # here, where neither Playwright nor the LLM is mid-action.
        control = getattr(session, "control", None)
        if control is not None:
            await control.checkpoint(page.url)
        log_.note("step", f"{i + 1}/{len(plan.steps)} {step.action}: {desc}", step=i + 1)
        mutates = step.action in {"click", "type"}
        if mutates:
            await _wait_stable(page)
        fp_before = await _fingerprint(page) if mutates else None
        if fp_before is not None:
            key = (step.action, step.goal or "", fp_before)
            if key in seen:
                raise StepFailure(
                    f"step {i} ({desc}): same action on an unchanged page",
                    task_text,
                    plan.entry_url,
                )
            seen.add(key)
        if step.action in {"click", "type"} and _is_risky_unproven(step):
            raise StepFailure(
                f"step {i} ({desc}): risky action without a done_when proof",
                task_text,
                plan.entry_url,
            )
        t0 = time.monotonic()
        try:
            outcome = await exec_step(page, step, laya, settings, extracts, picker=picker)
        except StepFailure as exc:
            log_.note("error", f"step {i + 1} failed: {exc}")
            raise StepFailure(f"step {i} ({desc}): {exc}", task_text, plan.entry_url) from exc
        except Exception as exc:
            log_.note("error", f"step {i + 1} failed: {exc}")
            raise StepFailure(f"step {i} ({desc}): {exc}", task_text, plan.entry_url) from exc
        log.info("plan step %d ok: %s", i, outcome)
        log_.note(
            "step",
            f"step {i + 1} ok ({time.monotonic() - t0:.1f}s): {outcome}",
            step=i + 1,
        )
        if mutates:
            mutated = True
        if mutates and fp_before is not None:
            fp_after = await _fingerprint(page)
            no_change = no_change + 1 if fp_after == fp_before else 0
            if no_change >= 3:
                raise StepFailure(
                    f"step {i} ({desc}): page did not change for 3 consecutive actions",
                    task_text,
                    plan.entry_url,
                )

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

    # Finish-insist (SystemOneHarness's completion rule): per-step proof says
    # every step worked; one more question says the OVERALL task is done. A
    # confident no means the plan technically succeeded and did not finish the
    # job — the agent fallback gets the original task with the page already
    # where the plan left it, which is exactly the repair position. Unsure
    # never punishes: the per-step gates already passed.
    if mutated and laya.enabled:
        verdict, conf = await laya.yes_no(
            f"Does the page now satisfy the overall task '{task_text}'?",
            await state_text(page),
        )
        label = "yes" if verdict else "no" if verdict is not None else "unsure"
        log_.note("gate", f"final check: {label} (conf {conf:.2f})")
        if verdict is False and conf >= settings.laya_min_confidence:
            raise StepFailure(
                "plan ran to its end but the page does not satisfy the task yet",
                task_text,
                plan.entry_url,
            )

    return {
        "plan": plan.model_dump(),
        "steps_executed": executed,
        "extracts": extracts,
        "final_url": page.url,
        "gate": laya.summary,
    }
