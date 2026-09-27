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


def claim(text: str, *, today: date | None = None) -> str | None:
    """The first recipe whose ``understands`` claims the text, or None.

    This is ``route`` without a fallback: a name comes back **only** when some
    hand-authored predicate actually claimed the sentence, which is the
    distinction ``route``'s return value cannot make (it returns ``requested``
    both when that recipe claims and when nothing does).
    """
    day = today or date.today()
    for recipe in iter_recipes():
        claims = getattr(recipe, "understands", None)
        if not callable(claims):
            continue
        try:
            if claims(text, day):
                return recipe.name
        except Exception:
            # A predicate is third-party-ish code (a stored recipe could grow
            # one). A recipe that cannot decide must not cost the run: skip it
            # and let the next claimant, or the requested recipe, answer.
            log.exception("router: %s.understands() raised; skipping", recipe.name)
    return None


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
    found = claim(text, today=day)
    if found is not None:
        if found != requested:
            log.info(
                "router: %r -> %s (asked for %s)", text[:80], found, requested
            )
        return found
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


async def learned_match(text: str, requested: str, settings, *, url: str = "") -> str | None:
    """The trusted learned recipe that answers ``text``, or None.

    Nils's second-run path (2026-09-23): "I first use laya to figure out if my
    new run is similar to older run, if yes we load it in and let laya do
    basically all steps".

    The "similar" question is asked twice over, and the cheap half goes first:
    replaying a recipe navigates to *that recipe's* entry URL, so a candidate
    whose entry URL is not the page this request goes to cannot be the one.
    Measured live 2026-09-24, the match itself cannot tell either: "Go to
    https://www.iana.org/about and tell me what IANA is responsible for" was
    routed to an example-domains recipe, whose page was then scraped and
    reported as the answer to a question about a different page. The URL is a
    fact about the request, so it is checked before any model call and the
    model is only asked about the candidates that survive — each on its own,
    so no fixed option set can bias it.

    ``url`` is the request's structured target (the task payload's ``url``
    field) when the caller has one; the URLs written in the request's own text
    are extracted as well, which is how the composer passes a start page. The
    precondition is an EQUALITY on scheme-less host+path between the request's
    targets and a candidate's entry URL — a prose *mention* of a host or path
    that is not the page being asked for qualifies nothing (SEC-BA-010: the
    original containment test let any sentence naming the host put the
    decision wholly to the gate).

    Five refusals, all deliberate:

    * Nothing to match, or nothing trusted yet → None, without a model call.
    * No navigation target anywhere in the request → None, without a model
      call. A request that names no page cannot be asking for a replay, whose
      whole effect is the page it opens.
    * No candidate whose entry URL equals a target → None, without a model
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
    named = _named_by_url(_request_targets(url, text), candidates)
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


def _request_targets(url: str, text: str) -> list[str]:
    """The URLs this request would navigate to, structured field first.

    ``url`` is the payload's ``url`` field — the target the caller stated in
    the only field the runner reads for a start page. The URLs written in the
    request's own text come second, because the composer has exactly one box
    and passes its start page inside the sentence; trailing sentence
    punctuation is stripped the same way ``api.fresh_chat`` strips it before
    storing the field.
    """
    from .tasks import _URL_RE

    targets = [u.rstrip(".,;") for u in _URL_RE.findall(text or "")]
    if (url or "").strip():
        targets.insert(0, url.strip())
    return targets


def _url_key(target: str) -> tuple[str, str]:
    """Scheme-less, ``www.``-less, trailing-slash-less host+path identity.

    Two URLs are the same page for the precondition's purpose when this pair
    is equal — https and http, ``www.``, an explicit ``:443`` and a trailing
    slash are spellings, not destinations. A different path on the same host
    is a different page.
    """
    from urllib.parse import urlsplit

    parts = urlsplit((target or "").strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host, parts.path.rstrip("/").lower()


def _named_by_url(targets: list[str], candidates: list[dict]) -> list[dict]:
    """Candidates whose entry URL IS one of the request's navigation targets.

    A learned recipe's whole effect is the page it navigates to (``_validate``
    requires a learned spec to carry one), so a request whose destination is a
    different page cannot be asking for it — no matter how similar the
    wording. Equality on the :func:`_url_key` pair, never containment: a
    host-only entry matches only a host-only target, and a URL merely
    mentioned in passing is not a destination.
    """
    keys = {key for key in (_url_key(t) for t in targets) if key[0]}
    return [
        spec for spec in candidates if _url_key(str(spec.get("entry_url") or "")) in keys
    ]
