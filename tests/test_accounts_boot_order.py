"""The upgrade path for a bot that already exists.

The dangerous part of several-accounts-per-bot is not the feature, it is the
first boot on a bot whose login was written by the *previous* build, directly
into ``/profiles/<bot>/``. Chrome must not be launched before those files are
moved: launching creates a fresh profile in the new location, and the migration
then merges the old files into a directory Chrome has already written — which is
how a profile gets corrupted, not merely reported wrong.

This pins that ordering. It fails if the migration is left lazy: the pre-warm
launches the browser onto ``accounts/default/`` before anything has swept the
root.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def legacy_bot(tmp_path: Path, monkeypatch):
    """A bot laid out the way the previous build left it, with a live app."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("CONTROL_TOKEN", "test-token")
    monkeypatch.setenv("AGENT_PROFILE", "x")

    # The legacy layout: Chrome's files directly in the bot's directory, and a
    # login among them — which is the thing that must survive the upgrade.
    root = tmp_path / "profiles" / "x"
    (root / "Default").mkdir(parents=True)
    (root / "Default" / "Preferences").write_text("{}")
    (root / "Local State").write_text("{}")

    con = sqlite3.connect(root / "Default" / "Cookies")
    con.execute("create table cookies (host_key text, name text)")
    con.execute("insert into cookies values ('x.com', 'auth_token')")
    con.commit()
    con.close()

    import importlib

    from browser_agent import api, config

    importlib.reload(config)
    importlib.reload(api)

    launches: list[tuple[str, list[str]]] = []

    class _Page:
        url = "data:text/html,idle"

    class _Session:
        """Stands in for Chrome creating its user-data-dir.

        Records the account it was told to start on *and* what was on disk at
        the bot's root at that moment — which is the fact the ordering
        assertion is really about.
        """

        def __init__(self):
            self.account = "default"

        def set_account(self, name):
            self.account = name

        async def stop(self):
            pass

        async def start(self):
            launches.append((self.account, sorted(p.name for p in root.iterdir())))
            api.settings.profile_dir_for(self.account).mkdir(parents=True, exist_ok=True)

        async def page(self):
            return _Page()

        def is_running(self):
            return True

    stub = _Session()
    api.session = stub
    api.runner.session = stub
    return api, root, launches


def test_the_migration_happens_before_the_browser_is_launched(legacy_bot):
    """The ordering the whole upgrade depends on.

    The root must already be swept clean of Chrome's files by the time the
    browser starts, or Chrome writes into a directory the migration is about to
    merge into.
    """
    api, root, launches = legacy_bot
    from fastapi.testclient import TestClient

    with TestClient(api.app) as c:
        c.get("/api/healthz")

    assert launches, "the browser should have been pre-warmed"
    account, root_at_launch = launches[0]
    assert account == "default"
    # Everything Chrome owns must be gone from the root by now. `accounts.json`
    # is our own metadata file and legitimately stays; the assertion is that no
    # Chrome file does, because those are what a later merge would collide with.
    chrome_entries = [n for n in root_at_launch if n not in ("accounts", "accounts.json")]
    assert chrome_entries == [], (
        f"the browser started while Chrome's files were still at the root: {chrome_entries}"
    )


def test_a_legacy_bot_keeps_its_login_across_the_upgrade(legacy_bot):
    api, root, _ = legacy_bot
    from fastapi.testclient import TestClient

    with TestClient(api.app) as c:
        body = c.get("/api/accounts", headers={"Authorization": "Bearer test-token"}).json()

    assert [a["name"] for a in body["accounts"]] == ["default"]
    # The login is now under the account, and reports as signed in — the whole
    # point of migrating rather than starting fresh.
    assert body["accounts"][0]["signed_in"] is True
    assert (root / "accounts" / "default" / "Local State").is_file()


def test_the_boot_picks_the_account_the_file_says_is_active(legacy_bot):
    """A pod restarting after a switch must come up on the switched account.

    Otherwise it launches on ``default`` while the file says ``work``, and every
    task runs silently as the wrong identity.
    """
    api, root, launches = legacy_bot
    from fastapi.testclient import TestClient

    with TestClient(api.app) as c:
        c.post("/api/accounts", headers={"Authorization": "Bearer test-token"},
               json={"name": "work"})
        c.post("/api/accounts/switch", headers={"Authorization": "Bearer test-token"},
               json={"name": "work"})

    # A restarted pod, simulated the way the other fixtures simulate a fresh
    # process: reload the module. That is what gives a genuinely new session —
    # whose account is the default, because a fresh process knows only that —
    # over the same files on disk.
    import importlib

    from browser_agent import api as api_mod, config

    importlib.reload(config)
    api = importlib.reload(api_mod)

    class _Page:
        url = "data:text/html,idle"

    class _Restarted:
        def __init__(self):
            self.account = "default"

        def set_account(self, name):
            self.account = name

        async def stop(self):
            pass

        async def start(self):
            launches.append((self.account, sorted(p.name for p in root.iterdir())))

        async def page(self):
            return _Page()

        def is_running(self):
            return True

    launches.clear()
    restarted = _Restarted()
    api.session = restarted
    api.runner.session = restarted

    with TestClient(api.app) as c:
        c.get("/api/healthz")

    assert restarted.account == "work"
    assert launches and launches[0][0] == "work"
