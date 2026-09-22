"""A fresh bot must not claim to be signed in.

Chrome creates its profile directory — and the empty Cookies database inside
it — on the very first launch, so "the profile directory is non-empty" is true
of a bot nobody has ever logged into. The roster renders that answer as a
signed-in / not-signed-in pill, and telling the operator a bot is ready when it
has never been logged in is the one thing that pill must never do.

The discriminating fact is whether a login left cookies behind. Measured on the
real deployment: a never-used profile reports 0, and a logged-in one reports 11.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent.browser import profile_exists  # noqa: E402


@dataclass
class _Settings:
    """Just the fields profile_exists reads.

    A bot's Chrome directories live one level below its profile root now — one
    per account — so the fake mirrors that: `profile_dir_for` is the only thing
    the real Settings and this stand-in have to agree on.
    """

    profiles_root: Path
    profile: str
    account: str = ""

    @property
    def profile_root(self) -> Path:
        return self.profiles_root / self.profile

    def profile_dir_for(self, account: str) -> Path:
        return self.profile_root / "accounts" / account


def _settings(tmp_path: Path, name: str = "x") -> _Settings:
    return _Settings(profiles_root=tmp_path, profile=name)


def _account_dir(tmp_path: Path, account: str = "default") -> Path:
    return tmp_path / "x" / "accounts" / account


def _make_cookies(path: Path, rows: int) -> None:
    """Write a Cookies database with `rows` entries, as Chrome would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        "create table cookies (host_key text, name text, value text, "
        "path text, expires_utc integer, is_secure integer, is_httponly integer)"
    )
    for i in range(rows):
        con.execute(
            "insert into cookies values (?,?,?,?,?,?,?)",
            (f"example{i}.com", f"c{i}", "v", "/", 0, 0, 0),
        )
    con.commit()
    con.close()


def test_absent_profile_is_not_signed_in(tmp_path):
    assert profile_exists(_settings(tmp_path)) is False


def test_directory_without_a_cookie_database_is_not_signed_in(tmp_path):
    """A directory Chrome has begun writing into, but with no Cookies db."""
    (_account_dir(tmp_path) / "Default").mkdir(parents=True)
    (_account_dir(tmp_path) / "Default" / "Preferences").write_text("{}")

    assert profile_exists(_settings(tmp_path)) is False


def test_empty_cookie_database_is_not_signed_in(tmp_path):
    """The exact state a freshly created bot is in.

    Chrome has launched, made the profile and the empty Cookies db, and no site
    has set anything. This reported True before the fix, which is what put a
    "signed in" pill on a bot nobody had ever logged into.
    """
    _make_cookies(_account_dir(tmp_path) / "Default" / "Cookies", rows=0)

    assert profile_exists(_settings(tmp_path)) is False


def test_cookies_mean_signed_in(tmp_path):
    _make_cookies(_account_dir(tmp_path) / "Default" / "Cookies", rows=11)

    assert profile_exists(_settings(tmp_path)) is True


def test_legacy_network_cookies_path_is_understood(tmp_path):
    """Pre-Chromium-96 profiles kept it under Default/Network."""
    _make_cookies(_account_dir(tmp_path) / "Default" / "Network" / "Cookies", rows=3)

    assert profile_exists(_settings(tmp_path)) is True


def test_locked_database_falls_back_to_saying_used(tmp_path):
    """A running browser holds the database open.

    The count is a roster nicety, not a control path, so an unreadable db must
    not flip a running profile to "not signed in".
    """
    profile = _account_dir(tmp_path) / "Default"
    profile.mkdir(parents=True)
    # A file that is present but not a SQLite database: opening it raises.
    (profile / "Cookies").write_bytes(b"not a database")

    assert profile_exists(_settings(tmp_path)) is True


def test_a_named_account_is_read_not_the_default(tmp_path):
    """The whole point of the argument: ask about a specific identity.

    Two accounts on one bot, only the second signed in — asking about the
    default must not report the other one's cookies, or the roster would show
    every account as ready once any one of them was.
    """
    _make_cookies(_account_dir(tmp_path, "work") / "Default" / "Cookies", rows=4)

    assert profile_exists(_settings(tmp_path), "work") is True
    assert profile_exists(_settings(tmp_path), "default") is False
