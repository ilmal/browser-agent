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

#: How many URL-matching candidates Laya will be asked to judge in one decision.
#: A bucket larger than this is more than the gate should be answering at once;
#: above it the honest answer is "too many to match", i.e. no match.
_MATCH_CAP = 10


async def learned_match(text: str, requested: str, settings) -> str | None:
    """The trusted learned recipe that answers ``text``, or None.

    Nils's second-run path (2026-09-23): "I first use laya to figure out if my
    new run is similar to older run, if yes we load it in and let laya do
    basically all steps".

    The "similar" question is asked twice over, and the cheap half goes first: a
    freeform request only ever names its own URL, and replaying a recipe navigates
    to *that recipe's* entry URL, so a candidate whose entry URL the request does
    not name cannot be the one. Measured live 2026-09-24, the match itself cannot
    tell either: "Go to https://www.iana.org/about and tell me what IANA is
    responsible for" was routed to an example-domains recipe, whose page was then
    scraped and reported as the answer to a question about a different page. The
    URL is a fact about the request, so it is checked before any model call and
    the model is only asked about the candidates that survive — each on its own,
    so no fixed option set can bias it.

    Four refusals, all deliberate:

    * Nothing to match, or nothing trusted yet → None, without a model call.
    * No candidate whose entry URL the request names → None, without a model
      call. This is the precondition above.
    * More than :data:`_MATCH_CAP` candidates → None. A bucket that large is
      more than the gate should be judging in one decision.
    * No candidate the gate calls the same task, above the confidence floor →
      None. A wrong match runs a *different* task's steps and reports its result
      as this request's, which is the one failure mode worth a missed speedup.

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
    named = _named_by_url(text, candidates)
    if not named:
        return None
    if len(named) > _MATCH_CAP:
        log.info(
            "router: %d learned recipes to match against; more than the gate should "
            "judge at once, not matching", len(named),
        )
        return None
    try:
        from .laya_gate import LayaGate

        gate = LayaGate(settings)
        for spec in named:
            saved = str(spec.get("description") or spec["name"])
            same, conf = await gate.same_task(saved, text)
            if same and conf >= settings.laya_min_confidence:
                log.info(
                    "router: learned match %s (conf %.2f) for %r",
                    spec["name"], conf, text[:80],
                )
                return str(spec["name"])
            log.info(
                "router: learned candidate %s declined (conf %.2f)",
                spec["name"], conf,
            )
        return None
    except Exception:
        log.debug("router: learned match failed", exc_info=True)
        return None


def _named_by_url(text: str, candidates: list[dict]) -> list[dict]:
    """Candidates whose entry URL the request itself names.

    A learned recipe's whole effect is the page it navigates to (``_validate``
    requires a learned spec to carry one), so a request that names a different
    URL cannot be asking for it — no matter how similar the wording. Checked by
    host + path, forgiving of the scheme, ``www.``, a trailing slash and a
    trailing sentence, so a request that writes the URL the way a person would
    still matches the one the harvest recorded.
    """
    low = (text or "").lower()
    return [spec for spec in candidates if _url_named(low, str(spec.get("entry_url") or ""))]


def _url_named(text: str, url: str) -> bool:
    """Does ``text`` name ``url``'s host, and its path when the URL has one?

    A URL with no path (``https://example.com``) is named by its host alone; one
    with a path needs the path too, so ``/about`` and ``/help/example-domains``
    on the same host stay distinct.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host or host not in text:
        return False
    path = parts.path.rstrip("/").lower()
    return not path or path in text
