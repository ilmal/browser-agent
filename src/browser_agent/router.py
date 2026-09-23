"""Send an instruction to the recipe that already knows how to do it.

The recipes are deterministic but they are *selected*, not inferred: the run
form posts the dropdown's value, so a sentence typed into the box only reaches
``flights.search`` if an operator picks that name first. Measured against the
goal prompt 2026-09-22, that is exactly what went wrong — ``plan.task`` ran a
12-step LLM plan, fell into the agent, and spent 135 s to answer 5,459 with the
4-day-trip bug still in it. The deterministic recipe that answers the question
in zero LLM calls was sitting unused in the registry.

This is the missing dispatch, and it is deliberately the smallest thing that
can work: no model, no scoring, no learned classifier. A recipe opts in by
exposing an ``understands(text, today) -> bool`` predicate (see
:meth:`~browser_agent.recipes.flights.FlightSearch.understands`); the router
takes the *first* recipe that claims the text, in registry order, and otherwise
leaves the requested recipe alone.

Why first-claim rather than best-match: every predicate here is a conjunction
that a sentence either satisfies outright or not at all (it names two known
places, a month and a duration, or it does not). Ranking such claims would be
inventing a tie-break for a case that cannot arise, and a wrong reroute is more
expensive than no reroute — the operator asked for a specific recipe by name,
and the fallback for "nothing claims this" is the thing they actually chose.
"""

from __future__ import annotations

import logging
from datetime import date

from .tasks import iter_recipes

log = logging.getLogger(__name__)


def route(
    text: str,
    requested: str,
    *,
    today: date | None = None,
    learned_match: str | None = None,
) -> str:
    """The recipe that should run ``text``, given the one that was requested.

    Returns ``requested`` unchanged when no recipe claims the text — including
    when ``requested`` itself claims it, so an explicit ``flights.search`` pick
    is never rerouted away. An empty instruction routes nowhere.

    Hand-authored predicates are consulted first and always win: a human wrote
    ``understands`` to mean "I can do this completely", which outranks a
    pattern the agent happened to produce once.

    ``learned_match`` is the name of a learned recipe a caller has *already*
    decided this text matches (the Laya similarity gate, which needs an await
    and so cannot live in this function). It is only honoured when the operator
    asked for something their own text does not name — a freeform or
    agent-bound request — never to override a recipe the text itself claimed.
    """
    text = (text or "").strip()
    if not text:
        return requested
    day = today or date.today()
    for recipe in iter_recipes():
        claims = getattr(recipe, "understands", None)
        if not callable(claims):
            continue
        try:
            if claims(text, day):
                if recipe.name != requested:
                    log.info(
                        "router: %r -> %s (asked for %s)", text[:80], recipe.name, requested
                    )
                return recipe.name
        except Exception:
            # A predicate is third-party-ish code (a stored recipe could grow
            # one). A recipe that cannot decide must not cost the run: skip it
            # and let the next claimant, or the requested recipe, answer.
            log.exception("router: %s.understands() raised; skipping", recipe.name)
    # No hand-authored recipe claimed it. A learned one may, but only for a
    # request the operator did not aim at a specific deterministic recipe: if
    # they picked one by name, that pick is the instruction.
    if learned_match and requested in _FREE_REQUESTS:
        if learned_match != requested:
            log.info(
                "router: %r -> learned recipe %s (asked for %s)",
                text[:80], learned_match, requested,
            )
        return learned_match
    return requested


#: The recipes an operator lands on by *default* rather than by choosing — the
#: freeform agent, and the planner that falls back to it. Only for these does a
#: learned recipe get to answer, because only for these is there no name the
#: operator picked. A request aimed at a stored recipe, a built-in, or anything
#: else is honoured as written.
_FREE_REQUESTS = frozenset({"agent.task", "plan.task"})

#: Options Laya is given to match a request against. At or below
#: ``laya_max_candidates`` so the question stays inside the choice range the
#: checkpoint is actually calibrated for; above it the honest answer is "too
#: many to match", i.e. no match.
_MATCH_CAP = 10


async def learned_match(text: str, requested: str, settings) -> str | None:
    """The trusted learned recipe that answers ``text``, or None.

    Nils's second-run path (2026-09-23): "I first use laya to figure out if my
    new run is similar to older run, if yes we load it in and let laya do
    basically all steps". This is the "similar" question, asked as the one kind
    of question Laya is measured to answer well — a *few* options, not an open
    one.

    Three refusals, all deliberate:

    * Nothing to match, or nothing trusted yet → None, without a model call.
    * More than :data:`_MATCH_CAP` candidates → None. Laya's confidence is
      uncalibrated past ~10 options; a bucket that large is a question it
      cannot honestly answer.
    * Below the confidence floor → None. A wrong match runs a *different*
      task's steps and reports its result as this request's, which is the one
      failure mode worth a missed speedup.

    Never raises, and never escalates: the gate is an accelerator, so any
    failure means the requested recipe runs as it would have anyway.
    """
    if requested not in _FREE_REQUESTS or not (text or "").strip():
        return None
    from .recipe_store import learned_store_for

    store = learned_store_for(settings)
    if store is None:
        return None
    candidates = [s for s in store.trusted_learned() if s.get("name") != requested]
    if not candidates:
        return None
    if len(candidates) > _MATCH_CAP:
        log.info(
            "router: %d learned recipes to match against; more than laya can rank, "
            "not matching", len(candidates),
        )
        return None
    try:
        from .laya_gate import LayaGate

        lines = [str(s.get("description") or s["name"]) for s in candidates]
        # No "or N if none" escape hatch: it biases the answer toward a fixed
        # label, and the confidence floor is what actually says "none of these".
        # Flat probabilities on an unrelated request are the signal, and they
        # only show up if the model is free to answer with what it believes.
        question = "Which numbered saved task is this new request asking for?"
        gate = LayaGate(settings)
        idx, conf = await gate.choose(question, lines, f"New request: {text}")
        if idx is None or conf < settings.laya_min_confidence:
            log.info("router: learned match declined (conf %.2f)", conf)
            return None
        return str(candidates[idx]["name"])
    except Exception:
        log.debug("router: learned match failed", exc_info=True)
        return None
