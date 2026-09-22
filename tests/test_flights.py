"""The flights recipe's pure half: the tfs encoder and the grid arithmetic.

The page flow needs a browser; everything below is deterministic and is where
the actual bug lived — reading the globally cheapest cell instead of the
cheapest cell *for the requested trip length*.
"""

from __future__ import annotations

from datetime import date

from browser_agent.recipes.flights import (
    _endpoint,
    encode_tfs,
    parse_cells,
    parse_market,
    parse_trip_length,
    pick_cheapest,
    resolve_month,
    resolve_place,
)

# A capture from the real Google Flights form: ARN/SEA, 2027-01-14 -> 2027-01-22.
REAL_TFS = (
    "CBwQARoeEgoyMDI3LTAxLTE0agcIARIDQVJOcgcIARIDU0VBGh4SCjIwMjctMDEtMjJqBwgBEgNTRUFy"
    "BwgBEgNBUk5AAUgBcAGCAQsI____________AZgBAQ"
)


def test_encode_tfs_is_byte_identical_to_a_real_capture():
    assert encode_tfs("2027-01-14", "2027-01-22", "ARN", "SEA") == REAL_TFS


def test_encode_tfs_airport_vs_city_endpoint():
    # IATA -> kind 1; a /m/ kgmid -> kind 3. The tag byte is what GF reads.
    assert _endpoint("ARN") == b"\x08\x01\x12\x03ARN"
    assert _endpoint("/m/06mxs") == b"\x08\x03\x12\x08/m/06mxs"


def test_encode_tfs_roundtrips_dates_and_airports():
    import base64

    raw = base64.urlsafe_b64decode(REAL_TFS + "==")
    assert b"2027-01-14" in raw and b"2027-01-22" in raw
    assert b"ARN" in raw and b"SEA" in raw


def test_parse_trip_length():
    assert parse_trip_length("8 day vacation") == 8
    assert parse_trip_length("an 8-day trip") == 8
    assert parse_trip_length("4-8 day vacation") == 4  # low end
    assert parse_trip_length("1 week") == 7
    assert parse_trip_length("two weeks") == 14
    assert parse_trip_length("somewhere warm") is None


def test_resolve_month_prefers_iso_then_next_occurrence():
    assert resolve_month("in january 2027", date(2026, 9, 22)) == (2027, 1)
    assert resolve_month("2027-01", date(2026, 9, 22)) == (2027, 1)
    # A month already past this year means next year.
    assert resolve_month("in january", date(2026, 9, 22)) == (2027, 1)
    assert resolve_month("in december", date(2026, 9, 22)) == (2026, 12)


def test_resolve_place():
    assert resolve_place("Stockholm") == "/m/06mxs"
    assert resolve_place("ARN") == "ARN"
    assert resolve_place("xyz") == "XYZ"  # a bare 3-letter code passes through


def test_parse_market_handles_the_benchmark_prompt():
    o, d = parse_market(
        "Find the cheapest roundtrip flights from stockholm to seattle in january, 8 day vacation"
    )
    assert (o.lower(), d.lower()) == ("stockholm", "seattle")


def test_parse_cells_reads_the_real_grid_labels():
    labels = [
        "SEK 5,075, low price, Jan 14 to Jan 21",
        "SEK 5,071, cheapest price, Jan 14 to Jan 22, selected",
        "SEK 10,236, Jan 12 to Jan 22",
        "Currency SEK",  # an unrelated label on the same page
        "",
    ]
    cells = parse_cells(labels, 2027)
    assert len(cells) == 3
    assert (5071, date(2027, 1, 14), date(2027, 1, 22), 8) in cells


def test_pick_cheapest_ignores_a_cheaper_shorter_trip():
    # The regression this recipe exists for: the globally cheapest cell is a
    # 4-day trip; the 8-day question must not take it.
    cells = [
        (4846, date(2027, 1, 11), date(2027, 1, 15), 4),  # cheapest overall
        (5071, date(2027, 1, 14), date(2027, 1, 22), 8),
        (5118, date(2027, 1, 13), date(2027, 1, 20), 7),
    ]
    assert pick_cheapest(cells, 8)[0] == 5071
    assert pick_cheapest(cells, 4)[0] == 4846
    assert pick_cheapest(cells, 9) is None
    # No trip length given -> the global minimum, which is the old behaviour.
    assert pick_cheapest(cells, None)[0] == 4846


def test_pick_cheapest_breaks_ties_on_the_earlier_departure():
    cells = [
        (5071, date(2027, 1, 15), date(2027, 1, 23), 8),
        (5071, date(2027, 1, 13), date(2027, 1, 21), 8),
    ]
    assert pick_cheapest(cells, 8)[1] == date(2027, 1, 13)
