"""The account endpoints, over the real app.

The unit tests cover the store; these cover the wiring, which is where the two
mistakes that matter would live: an endpoint that acts on an account the store
does not have, and a *switch* that reports success without the session actually
changing — the UI's whole question after clicking it is "did it take".

The browser is stubbed. A switch does restart Chrome for real, so a test that
did not stub it would launch one — slow, and it would need a display.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def bot(tmp_path: Path, monkeypatch):
    """The bot app with a temp profile root and a stubbed browser."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("CONTROL_TOKEN", "test-token")

    import importlib

    from browser_agent import api, config

    importlib.reload(config)
    importlib.reload(api)

    started: list[str] = []

    class _Page:
        url = "data:text/html,idle"

    class _Session:
        """Only what the account endpoints touch."""

        def __init__(self):
            self.account = "default"

        def set_account(self, name):
            self.account = name

        async def stop(self):
            pass

        async def start(self):
            started.append(self.account)

        async def page(self):
            return _Page()

        def is_running(self):
            return True

    stub = _Session()
    api.session = stub
    api.runner.session = stub
    return api, stub, started


def _client(api):
    from fastapi.testclient import TestClient

    return TestClient(api.app)


H = {"Authorization": "Bearer test-token"}


def test_a_fresh_bot_lists_one_default_account(bot):
    api, _, _ = bot
    with _client(api) as c:
        res = c.get("/api/accounts", headers=H)
    assert res.status_code == 200
    body = res.json()
    assert [a["name"] for a in body["accounts"]] == ["default"]
    assert body["active"] == "default"


def test_adding_an_account_does_not_switch_to_it(bot):
    """Signing in is a deliberate act over noVNC; adding a name is not.

    Silently swapping the running identity when the operator types a new name
    would move a scheduled bot onto a logged-out profile.
    """
    api, stub, started = bot
    with _client(api) as c:
        # The boot pre-warm starts the browser once; nothing after that should.
        started.clear()
        res = c.post("/api/accounts", headers=H,
                     json={"name": "work", "email": "nils@u1.se"})
        assert res.status_code == 200
        body = res.json()

    assert body["active"] == "default"
    assert [a["name"] for a in body["accounts"]] == ["default", "work"]
    assert stub.account == "default"
    assert started == []


def test_an_invalid_account_name_is_refused(bot):
    """These become directory names, so this is a traversal guard, not a typo check."""
    api, _, _ = bot
    with _client(api) as c:
        res = c.post("/api/accounts", headers=H, json={"name": "../escape"})
    assert res.status_code == 400


def test_a_duplicate_account_is_a_conflict(bot):
    api, _, _ = bot
    with _client(api) as c:
        c.post("/api/accounts", headers=H, json={"name": "work"})
        res = c.post("/api/accounts", headers=H, json={"name": "work"})
    assert res.status_code == 409


def test_switching_restarts_the_browser_on_the_new_account(bot):
    api, stub, started = bot
    with _client(api) as c:
        c.post("/api/accounts", headers=H, json={"name": "work"})
        # Past the boot pre-warm: what follows is the switch and nothing else.
        started.clear()
        res = c.post("/api/accounts/switch", headers=H, json={"name": "work"})
        assert res.status_code == 200
        body = res.json()

    assert body["active"] == "work"
    assert body["error"] == ""
    assert stub.account == "work"
    assert started == ["work"]          # stopped and started, on the new one


def test_switching_survives_a_restart_of_the_pod(bot):
    """The choice is on disk, so it outlives the process that made it.

    This is what makes a switch a restart rather than a redeploy: the file is
    the record, and the next boot reads it.
    """
    api, _, _ = bot
    with _client(api) as c:
        c.post("/api/accounts", headers=H, json={"name": "work"})
        c.post("/api/accounts/switch", headers=H, json={"name": "work"})

    from browser_agent.accounts import load_accounts

    again = load_accounts(api.settings.profile_root, api.settings.accounts_path)
    assert again.active_account() == "work"


def test_switching_to_an_unknown_account_is_a_404(bot):
    api, stub, _ = bot
    with _client(api) as c:
        res = c.post("/api/accounts/switch", headers=H, json={"name": "nope"})
    assert res.status_code == 404
    assert stub.account == "default"


def test_deleting_the_running_account_is_refused(bot):
    """The browser holds that directory open; deleting it out from under Chrome
    is the corruption the single-writer rule exists to prevent."""
    api, stub, _ = bot
    with _client(api) as c:
        c.post("/api/accounts", headers=H, json={"name": "work"})
        res = c.delete("/api/accounts/default", headers=H)
    assert res.status_code == 409


def test_deleting_an_idle_account_forgets_it_but_keeps_the_profile(bot):
    """`purge` is the destructive half and defaults OFF.

    Forgetting the name is reversible — add it back and the login is still
    there — whereas deleting the directory *is* logging out of everything that
    identity holds. So the safer of the two is what happens by default.
    """
    api, stub, _ = bot
    from browser_agent.accounts import account_dir

    with _client(api) as c:
        c.post("/api/accounts", headers=H, json={"name": "work"})
        res = c.delete("/api/accounts/work", headers=H)
        assert res.status_code == 200
        body = res.json()

    assert [a["name"] for a in body["accounts"]] == ["default"]
    assert body["purged"] is False
    assert account_dir(api.settings.profile_root, "work").is_dir()


def test_purging_deletes_the_chrome_directory(bot):
    api, _, _ = bot
    from browser_agent.accounts import account_dir

    with _client(api) as c:
        c.post("/api/accounts", headers=H, json={"name": "work"})
        res = c.delete("/api/accounts/work?purge=true", headers=H)

    assert res.json()["purged"] is True
    assert not account_dir(api.settings.profile_root, "work").exists()


def test_the_last_account_cannot_be_deleted(bot):
    """A bot with no account has no user-data-dir, so it cannot run at all."""
    api, _, _ = bot
    with _client(api) as c:
        res = c.delete("/api/accounts/default", headers=H)
    assert res.status_code == 409


def test_state_reports_the_running_account_and_all_of_them(bot):
    """The UI's poll: which is live, and what the alternatives are."""
    api, _, _ = bot
    with _client(api) as c:
        c.post("/api/accounts", headers=H, json={"name": "work"})
        body = c.get("/api/state", headers=H).json()

    assert body["account"] == "default"
    assert body["accounts"]["active"] == "default"
    assert [a["name"] for a in body["accounts"]["accounts"]] == ["default", "work"]


def test_account_endpoints_require_the_token(bot):
    api, _, _ = bot
    with _client(api) as c:
        assert c.get("/api/accounts").status_code == 401
        assert c.post("/api/accounts", json={"name": "work"}).status_code == 401
