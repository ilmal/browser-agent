"""Several identities per bot.

A bot was one Chrome profile. It now owns a directory of them and runs exactly
one at a time, because a user-data-dir is single-writer and there is one X
display per pod. Two things have to hold for that to be safe rather than merely
convenient:

* **the migration must be resumable** — every bot that exists today has Chrome's
  files written directly into ``/profiles/<bot>/``, and a half-moved profile
  reads to Chrome as an empty one, i.e. logged out, which is the failure that
  matters; and
* **the store must never leave a bot with no account** — a bot with no
  user-data-dir cannot launch, so every task would fail confusingly instead of
  saying "add an account first".
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent.accounts import (  # noqa: E402
    ACCOUNTS_FILE,
    DEFAULT_ACCOUNT,
    account_dir,
    ensure_layout,
    load_accounts,
    remove_account,
    signed_in,
    valid_account,
)


def _chrome_files(root: Path) -> None:
    """The shape of a profile Chrome has written into, pre-accounts."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "Default").mkdir(exist_ok=True)
    (root / "Default" / "Preferences").write_text("{}")
    (root / "Local State").write_text("{}")
    (root / "SingletonLock").symlink_to("browser-agent-1")


def _cookies(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("create table cookies (host_key text, name text)")
    for i in range(rows):
        con.execute("insert into cookies values (?,?)", (f"e{i}.com", "c"))
    con.commit()
    con.close()


# -- names -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["default", "nils", "work-2", "a"])
def test_plain_slugs_are_valid(name):
    assert valid_account(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Nils",          # uppercase: the directory name must be predictable
        "..",            # a path traversal dressed as an account
        "../escape",
        "a/b",
        "-leading",
        "trailing-",
        "sp ace",
        "dot.name",
    ],
)
def test_anything_that_is_not_a_slug_is_refused(name):
    """These become directory names, so a bad one is a traversal, not a typo."""
    assert valid_account(name) is False


# -- migration -------------------------------------------------------------


def test_legacy_profile_is_moved_under_an_account(tmp_path):
    _chrome_files(tmp_path)
    (tmp_path / "Default" / "Cookies").write_text("x")

    moved = ensure_layout(tmp_path)

    assert set(moved) == {"Default", "Local State", "SingletonLock"}
    dest = account_dir(tmp_path, DEFAULT_ACCOUNT)
    assert (dest / "Default" / "Cookies").read_text() == "x"
    # The root must be left holding only our own entries: Chrome opens this
    # directory, and a leftover Default/ beside accounts/ would be ambiguous.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["accounts"]


def test_migration_is_resumable_half_way(tmp_path):
    """An interrupted move must converge, not need a marker.

    This is the state a pod killed mid-migration leaves: the move creates
    ``accounts/default/`` *first*, so a run interrupted before the loop finishes
    has ``accounts/`` present while Chrome's files are still at the root. An
    "is ``accounts/`` absent?" test would call that migrated and leave the bot
    logged out, with its login in a directory nothing points at.
    """
    _chrome_files(tmp_path)
    dest = account_dir(tmp_path, DEFAULT_ACCOUNT)
    dest.mkdir(parents=True)                     # created before the loop
    (tmp_path / "Default").rename(dest / "Default")   # first entry moved, then killed

    moved = ensure_layout(tmp_path)

    assert set(moved) == {"Local State", "SingletonLock"}
    assert (dest / "Local State").is_file()
    assert (dest / "SingletonLock").is_symlink()
    assert [p.name for p in tmp_path.iterdir() if p.name != "accounts"] == []

    # And a second pass is a no-op rather than a re-shuffle.
    assert ensure_layout(tmp_path) == []


def test_migration_retries_an_entry_that_could_not_move(tmp_path, monkeypatch):
    """One un-movable entry must not abort the rest.

    Leaving the root populated is the state that breaks Chrome, so getting most
    of it moved is strictly better than stopping at the first failure.
    """
    _chrome_files(tmp_path)
    real_move = __import__("shutil").move

    def flaky(src, dst):
        if Path(src).name == "Local State":
            raise OSError("nope")
        return real_move(src, dst)

    monkeypatch.setattr("browser_agent.accounts.shutil.move", flaky)
    moved = ensure_layout(tmp_path)

    assert "Default" in moved and "Local State" not in moved
    assert (tmp_path / "Local State").exists()          # still there, retryable
    assert (account_dir(tmp_path, DEFAULT_ACCOUNT) / "Default").is_dir()


def test_already_migrated_profile_is_left_alone(tmp_path):
    (tmp_path / "accounts" / "default").mkdir(parents=True)
    (tmp_path / "accounts" / "default" / "Default").mkdir()

    assert ensure_layout(tmp_path) == []
    assert (tmp_path / "accounts" / "default" / "Default").is_dir()


def test_our_own_metadata_file_is_never_migrated(tmp_path):
    """accounts.json is ours, not Chrome's.

    Sweeping it into accounts/default/ would put a stray JSON inside a
    user-data-dir, and the store would then find no file and re-default.
    """
    _chrome_files(tmp_path)
    (tmp_path / ACCOUNTS_FILE).write_text("{}")

    ensure_layout(tmp_path)

    assert (tmp_path / ACCOUNTS_FILE).is_file()
    assert not (account_dir(tmp_path, DEFAULT_ACCOUNT) / ACCOUNTS_FILE).exists()


# -- store -----------------------------------------------------------------


def test_a_fresh_bot_gets_one_default_account(tmp_path):
    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)

    assert store.names == [DEFAULT_ACCOUNT]
    assert store.active_account() == DEFAULT_ACCOUNT
    assert (tmp_path / ACCOUNTS_FILE).is_file()


def test_the_legacy_login_lands_on_the_default_account(tmp_path):
    """An existing bot keeps its login, under a name the operator can see."""
    _chrome_files(tmp_path)
    _cookies(tmp_path / "Default" / "Cookies", rows=5)

    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)

    assert store.names == [DEFAULT_ACCOUNT]
    assert signed_in(account_dir(tmp_path, DEFAULT_ACCOUNT)) is True


def test_accounts_round_trip_through_the_file(tmp_path):
    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)
    store.add("work", label="Work X", email="nils@u1.se")
    store.add("personal")
    store.set_active("work")
    store.save()

    again = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)

    assert again.names == [DEFAULT_ACCOUNT, "work", "personal"]
    assert again.active_account() == "work"
    assert again.get("work").email == "nils@u1.se"


def test_adding_an_account_makes_its_directory(tmp_path):
    """So the operator can see it exists and is merely not signed in yet."""
    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)
    store.add("work")

    assert account_dir(tmp_path, "work").is_dir()
    assert signed_in(account_dir(tmp_path, "work")) is False


def test_a_corrupt_file_degrades_to_the_default(tmp_path):
    """This runs at import of the control plane: a raise would crashloop the pod."""
    (tmp_path / ACCOUNTS_FILE).write_text("{not json")

    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)

    assert store.names == [DEFAULT_ACCOUNT]
    assert store.active_account() == DEFAULT_ACCOUNT


def test_an_account_with_a_traversing_name_is_dropped(tmp_path):
    (tmp_path / ACCOUNTS_FILE).write_text(json.dumps({
        "accounts": [{"name": "../escape"}, {"name": "ok"}],
        "active": "ok",
    }))

    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)

    assert store.names == ["ok"]


def test_active_falls_back_when_it_names_a_missing_account(tmp_path):
    """A hand-edited file must not make every task fail with "no such account"."""
    (tmp_path / ACCOUNTS_FILE).write_text(json.dumps({
        "accounts": [{"name": "a"}, {"name": "b"}],
        "active": "gone",
    }))

    assert load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE).active_account() == "a"


def test_last_account_cannot_be_removed(tmp_path):
    """A bot with no account has no user-data-dir, so it cannot run at all."""
    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)

    with pytest.raises(ValueError):
        store.remove(DEFAULT_ACCOUNT)
    assert store.names == [DEFAULT_ACCOUNT]


def test_removing_the_active_account_moves_active_to_a_remaining_one(tmp_path):
    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)
    store.add("work")
    store.set_active("work")

    store.remove("work")

    assert store.active_account() == DEFAULT_ACCOUNT
    assert store.names == [DEFAULT_ACCOUNT]


def test_set_active_refuses_an_unknown_account(tmp_path):
    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)

    assert store.set_active("nope") is None
    assert store.active_account() == DEFAULT_ACCOUNT


def test_removing_an_account_can_delete_its_chrome_directory(tmp_path):
    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)
    store.add("work")
    _cookies(account_dir(tmp_path, "work") / "Default" / "Cookies", rows=2)
    assert signed_in(account_dir(tmp_path, "work")) is True

    assert remove_account(tmp_path, "work") is True

    assert not account_dir(tmp_path, "work").exists()
    assert remove_account(tmp_path, "work") is False


def test_the_metadata_file_holds_no_credentials(tmp_path):
    """The repo is public and this is rendered in HTML.

    Guards the invariant rather than the mechanism: an account is a slug the
    operator chose, and the free-text fields are notes. Nothing here may become
    a place a password or a session token gets written.
    """
    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)
    store.add("work", label="Work", email="nils@u1.se", notes="the work X login")
    store.save()

    body = json.loads((tmp_path / ACCOUNTS_FILE).read_text())

    assert set(body) == {"accounts", "active"}
    for acc in body["accounts"]:
        assert set(acc) <= {
            "name", "label", "email", "notes", "created_at", "display_label"
        }


def test_summary_reports_each_account_separately(tmp_path):
    """The UI's question: which of these is the one that is signed in."""
    store = load_accounts(tmp_path, tmp_path / ACCOUNTS_FILE)
    store.add("work")
    _cookies(account_dir(tmp_path, "work") / "Default" / "Cookies", rows=3)

    rows = {r["name"]: r for r in store.summary()}

    assert rows["work"]["signed_in"] is True
    assert rows["work"]["active"] is False
    assert rows[DEFAULT_ACCOUNT]["signed_in"] is False
    assert rows[DEFAULT_ACCOUNT]["active"] is True
