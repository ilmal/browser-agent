"""The needs-you banner must retire, not accumulate.

Found live 2026-09-27: every exhausted run minted a BLOCKED row, nothing ever
retired one, and the banner showed a two-day-old minesweeper block beside a
dozen identical copies. A blocked row is "needs you" only while it is the
latest thing that happened to its recipe.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    yield


def _retire():
    from browser_agent import api

    importlib.reload(api)
    return api._retire_superseded_blocked


def _row(recipe: str, status: str, created_at: float) -> dict:
    return {"recipe": recipe, "status": status, "created_at": created_at}


def test_a_later_success_retires_an_older_block(tmp_path):
    out = _retire()([
        _row("minesweeper.play", "blocked", 300.0),
        _row("minesweeper.play", "done", 500.0),
    ])

    assert out == [_row("minesweeper.play", "done", 500.0)]


def test_a_block_newer_than_the_last_success_survives(tmp_path):
    out = _retire()([
        _row("minesweeper.play", "done", 100.0),
        _row("minesweeper.play", "blocked", 400.0),
    ])

    assert _row("minesweeper.play", "blocked", 400.0) in out


def test_retirement_is_scoped_per_recipe():
    """A success of one recipe must never hide a still-blocked other one."""
    out = _retire()([
        _row("minesweeper.play", "blocked", 300.0),
        _row("agent.task", "done", 900.0),
    ])

    assert _row("minesweeper.play", "blocked", 300.0) in out


def test_many_identical_blocks_all_retire_after_one_success():
    """The wall case: a dozen copies, all retired by a single later success."""
    rows = [_row("minesweeper.play", "blocked", float(i)) for i in range(1, 13)]
    rows.append(_row("minesweeper.play", "done", 99.0))

    assert _retire()(rows) == [_row("minesweeper.play", "done", 99.0)]


def test_failed_and_done_rows_are_never_retired():
    rows = [
        _row("minesweeper.play", "failed", 100.0),
        _row("minesweeper.play", "done", 200.0),
    ]

    assert _retire()(rows) == rows


def test_a_block_with_no_later_success_is_kept():
    rows = [_row("minesweeper.play", "blocked", 100.0)]

    assert _retire()(rows) == rows


# -- the banner's obstacle key (served JS, run for real) --------------------
#
# ``normDetail``/``obstacleKey`` live in the served index.html, so no Python
# test would otherwise notice them drifting. They are the whole reason the
# banner stops re-opening: an obstacle whose headline churns between runs is
# read as a NEW obstacle and un-hides the wall. These run the page's own
# functions through node against the exact strings the live bot produced.

_PAGE = Path(__file__).resolve().parents[1] / "src" / "browser_agent" / "ui" / "index.html"


def _obstacle_key(details: list[str]) -> list[str]:
    import re
    import shutil
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node is not installed; cannot run the page's key function")
    js = _PAGE.read_text()
    helper = re.search(r"function normDetail\(detail\) \{.*?\n\}", js, re.S)
    keyer = re.search(r"function obstacleKey\(t\) \{.*?\n\}", js, re.S)
    assert helper and keyer, "the obstacle-key functions are gone from index.html"
    script = (
        helper.group(0) + "\n" + keyer.group(0) + "\n"
        "const details = " + repr(list(details)).replace("'", '"') + ";\n"
        "process.stdout.write(JSON.stringify("
        "details.map(d => obstacleKey({recipe: 'minesweeper.play', detail: d}))));"
    )
    proc = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    import json

    return json.loads(proc.stdout)


def test_one_obstacle_keeps_one_key_as_its_counts_grow():
    """The live case: re-running the recipe raises the seen/replacement counts.

    Both rows are the same blocked canonical, so the banner must read them as
    ONE obstacle — not two — and the dismiss must hold across the change.
    """
    a = ("rate_limited: every venue is written off: minesweeper.online (seen 2x); "
         "4 searched replacement(s) probed and rejected (no board contract). "
         "The local test board is operator-excluded. Needs a different egress, or "
         "an adapter for another site. (https://minesweeper.online/new-game)")
    b = a.replace("4 searched", "8 searched").replace("(seen 2x)", "(seen 9x)")

    keys = _obstacle_key([a, b])

    assert keys[0] == keys[1], "a churning count re-opened the banner"


def test_two_different_hosts_stay_distinct():
    a = "rate_limited: every venue is written off: minesweeper.online (seen 1x). (https://minesweeper.online/new-game)"
    b = "rate_limited: every venue is written off: other.example (seen 1x). (https://other.example/new-game)"

    keys = _obstacle_key([a, b])

    assert keys[0] != keys[1], "two venues collapsed into one obstacle"
