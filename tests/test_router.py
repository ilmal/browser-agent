"""The router: prose reaches the deterministic recipe instead of the agent.

The measured failure this guards (2026-09-22): the goal prompt ran as
``plan.task``, spent 135 s and one LLM plan, and answered 5,459 — the
4-day-trip bug — while ``flights.search`` sat unused in the registry because
nothing mapped the sentence onto it.
"""

from __future__ import annotations

from datetime import date

from browser_agent.recipes.flights import FlightSearch
from browser_agent.router import route

BENCH = "Find the cheapest roundtrip flights from stockholm to seattle in january, 8 day vacation"
TODAY = date(2026, 9, 22)


def test_benchmark_prompt_routes_to_the_deterministic_recipe():
    assert route(BENCH, "plan.task", today=TODAY) == "flights.search"


def test_a_recipe_that_claims_the_text_is_left_alone():
    # An explicit pick must never be rerouted *away* — the operator asked for it.
    assert route(BENCH, "flights.search", today=TODAY) == "flights.search"


def test_an_unrelated_instruction_is_not_hijacked():
    for text in (
        "post hello world on x",
        "play minesweeper",
        "report the page title",
        "",
    ):
        assert route(text, "plan.task", today=TODAY) == "plan.task"


def test_partial_matches_are_not_claimed():
    # Each of these is missing a piece run() would need, so the honest answer is
    # "no claim" and the requested recipe runs.
    for text in (
        "find cheap flights to seattle in january for 8 days",  # no origin
        "find cheap flights from stockholm to seattle in january",  # no duration
        "find cheap flights from stockholm to seattle for 8 days",  # no month
        "find cheap flights from stockholm to atlantis in january for 8 days",  # unknown
        "book a table from stockholm to seattle in january for 8 days",  # not a flight
    ):
        assert route(text, "plan.task", today=TODAY) == "plan.task", text


def test_understand_cannot_disagree_with_run():
    # understands() is only allowed to claim a text run() can actually parse, so
    # the two must agree on the benchmark prompt's parts.
    assert FlightSearch.understands(BENCH, TODAY) is True
    assert FlightSearch.understands("find cheap flights to seattle for 8 days", TODAY) is False
