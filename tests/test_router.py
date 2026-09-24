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
        "report the page title",
        "",
    ):
        assert route(text, "plan.task", today=TODAY) == "plan.task"


def test_playing_minesweeper_routes_to_the_recipe_that_can_play_it():
    """The live failure this guards (2026-09-23).

    "play a game of minesweeper until you win" reached ``plan.task`` — the
    dropdown's default — because no predicate claimed it. The plan searched
    Google, picked tic-tac-toe, and the agent spent 24 steps on a page the
    deterministic solver was already able to beat. The recipe existed the whole
    time; nothing routed to it.
    """
    assert route("play a game of minesweeper until you win", "plan.task",
                 today=TODAY) == "minesweeper.play"
    assert route("play minesweeper", "plan.task", today=TODAY) == "minesweeper.play"


def test_naming_minesweeper_without_playing_it_is_not_a_play():
    """A reading task must not be handed to a click loop."""
    for text in (
        "explain the minesweeper algorithm",
        "read the minesweeper wikipedia page",
        "what is minesweeper",
        "help me understand minesweeper source code",
    ):
        assert route(text, "plan.task", today=TODAY) == "plan.task", text


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


# -- learned recipes --------------------------------------------------------
#
# The Laya similarity gate needs an await and so lives outside route(); these
# pin the contract between the two: a hand-authored claim always wins, and a
# learned match only ever answers a request the operator did not aim at a
# specific recipe.


def test_a_learned_match_answers_a_freeform_request():
    assert (
        route("do the thing I did yesterday", "plan.task",
              today=TODAY, learned_match="learned-thing-abc123")
        == "learned-thing-abc123"
    )


def test_a_learned_match_never_overrides_a_named_recipe():
    # The operator picked plan.task by name *and* told it something: their pick
    # is the instruction, and rerouting it would run a different task.
    assert (
        route("do the thing", "linkedin.page_post",
              today=TODAY, learned_match="learned-thing-abc123")
        == "linkedin.page_post"
    )


def test_a_hand_authored_claim_beats_a_learned_match():
    # The benchmark prompt names a recipe that can do it completely; a pattern
    # the agent produced once must not outrank that.
    assert (
        route(BENCH, "plan.task", today=TODAY, learned_match="learned-bench-zzz")
        == "flights.search"
    )


def test_no_learned_match_leaves_the_request_alone():
    assert route("do the thing", "plan.task", today=TODAY) == "plan.task"
