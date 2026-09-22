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


def route(text: str, requested: str, *, today: date | None = None) -> str:
    """The recipe that should run ``text``, given the one that was requested.

    Returns ``requested`` unchanged when no recipe claims the text — including
    when ``requested`` itself claims it, so an explicit ``flights.search`` pick
    is never rerouted away. An empty instruction routes nowhere.
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
    return requested
