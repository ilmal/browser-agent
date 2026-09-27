"""The needs-you banner must retire, not accumulate.

Found live 2026-09-27: every exhausted run minted a BLOCKED row, nothing ever
retired one, and the banner showed a two-day-old minesweeper block beside a
dozen identical copies. A blocked row is "needs you" only while it is the
latest thing that happened to its recipe.
"""

from __future__ import annotations

import importlib

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
