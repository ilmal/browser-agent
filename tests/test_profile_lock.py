"""Stale Chrome profile locks must not stop a restart.

`SingletonLock` is a symlink whose target text is "<hostname>-<pid>". Chrome
exits with PROFILE_IN_USE when that hostname is not the local one. A pod's
hostname is its pod name, which changes on every recreate, while the profile
lives on a stable PVC — so any unclean shutdown leaves a lock that blocks the
next boot. A live browser must never have its lock cleared.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent.browser import _clear_stale_profile_lock  # noqa: E402


def _make_lock(profile_dir: Path, target: str) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    lock = profile_dir / "SingletonLock"
    if lock.is_symlink() or lock.exists():
        lock.unlink()
    lock.symlink_to(target)
    return lock


def test_clears_lock_from_a_hostname_that_no_longer_exists(tmp_path):
    """The restart case: pod name changed, leftover lock from the old pod."""
    profile = tmp_path / "x"
    lock = _make_lock(profile, "profile-x-5b9cbbfcc7-mqw6s-42")

    _clear_stale_profile_lock(profile)

    assert not lock.is_symlink() and not lock.exists()


def test_keeps_lock_when_a_browser_is_running(tmp_path, monkeypatch):
    """A live browser owns the profile; its lock must survive.

    Clearing it here would let a second Chrome into the same user_data_dir —
    the corruption the single-writer rule exists to prevent.

    Liveness is "something is listening on the port", not "the port file
    exists": Chrome writes the file at launch and never removes it on exit, so
    the file alone is also what a *closed* window leaves behind.
    """
    from browser_agent import browser as browser_mod

    profile = tmp_path / "x"
    lock = _make_lock(profile, "browser-agent-42")
    (profile / "DevToolsActivePort").write_text("9222\n/devtools/browser/abc\n")
    monkeypatch.setattr(browser_mod, "_port_is_live", lambda port, timeout_s=0.4: True)

    _clear_stale_profile_lock(profile)

    assert lock.is_symlink(), "a running browser's lock was cleared"


def test_clears_the_lock_when_the_window_was_closed(tmp_path):
    """The stale-file case: the window is gone, so the lock is stale too.

    Keeping it would refuse the relaunch with PROFILE_IN_USE, which is how a
    closed browser turned into a bot that could not recover.
    """
    profile = tmp_path / "x"
    lock = _make_lock(profile, "browser-agent-42")
    (profile / "DevToolsActivePort").write_text("9222\n/devtools/browser/abc\n")

    _clear_stale_profile_lock(profile)

    assert not lock.is_symlink(), "a closed browser's lock left the profile unusable"


def test_no_lock_is_a_noop(tmp_path):
    profile = tmp_path / "x"
    profile.mkdir(parents=True)
    _clear_stale_profile_lock(profile)  # must not raise
    assert not (profile / "SingletonLock").exists()


def test_keeps_lock_when_port_file_is_unreadable_port(tmp_path):
    """A truncated port file means 'no live browser', so the lock goes."""
    profile = tmp_path / "x"
    lock = _make_lock(profile, "old-pod-1")
    (profile / "DevToolsActivePort").write_text("")  # zero-length, as after a kill

    _clear_stale_profile_lock(profile)

    assert not lock.exists()


def test_a_stale_port_file_is_not_an_endpoint(tmp_path):
    """The exact reported failure: a closed browser, a port nobody owns.

    /profiles/x/DevToolsActivePort survived the window being closed, so
    cdp_endpoint kept handing out http://127.0.0.1:37445 and the agent attached
    to it, failing with "All connection attempts failed" — a recoverable state
    reported as a broken task.
    """
    from browser_agent.browser import _read_devtools_endpoint

    profile = tmp_path / "x"
    profile.mkdir()
    (profile / "DevToolsActivePort").write_text("37445\n/devtools/browser/abc\n")

    assert _read_devtools_endpoint(profile, timeout_s=0.1) is None


def test_a_live_port_file_is_an_endpoint(tmp_path, monkeypatch):
    from browser_agent import browser as browser_mod
    from browser_agent.browser import _read_devtools_endpoint

    profile = tmp_path / "x"
    profile.mkdir()
    (profile / "DevToolsActivePort").write_text("37445\n/devtools/browser/abc\n")
    monkeypatch.setattr(browser_mod, "_port_is_live",
                        lambda port, timeout_s=0.4: port == 37445)

    assert _read_devtools_endpoint(profile, timeout_s=0.1) == "http://127.0.0.1:37445"


def test_no_port_file_is_no_endpoint(tmp_path):
    from browser_agent.browser import _read_devtools_endpoint

    profile = tmp_path / "x"
    profile.mkdir()
    assert _read_devtools_endpoint(profile, timeout_s=0.1) is None
