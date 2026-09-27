"""SEC-BA-005 and SEC-BA-006: what the control plane echoes back.

Two disclosure bugs, both on the bot's own API: a client-controlled
X-Forwarded-Prefix that reached an inline <script> in the served page, and a
global exception handler that answered 500s with raw str(exc) — subprocess
stderr, paths and library internals included.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

sys_path = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def bot_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("CONTROL_TOKEN", "test-token")
    monkeypatch.chdir(tmp_path)
    import sys

    if str(sys_path) not in sys.path:
        sys.path.insert(0, str(sys_path))
    from browser_agent import api, config

    importlib.reload(config)
    importlib.reload(api)
    return api


AUTH = {"Authorization": "Bearer test-token"}

# The exact payload the audit proved escapes json.dumps unharmed.
HOSTILE_PREFIX = "</script><img src=x onerror=alert(1)>"


class TestForwardedPrefix:
    def test_hostile_prefix_is_dropped_at_capture(self, bot_env):
        """A non-prefix header value never reaches _REACHED_PREFIX at all."""
        from fastapi.testclient import TestClient

        with TestClient(bot_env.app) as client:
            res = client.get(
                "/api/state", headers={**AUTH, "X-Forwarded-Prefix": HOSTILE_PREFIX}
            )
        assert res.status_code == 200
        assert res.json()["url_prefix"] == ""

    def test_prefix_shaped_values_with_markup_metacharacters_are_dropped(self, bot_env):
        """The allowlist is the defence; values that only LOOK like a prefix
        (query, spaces, quotes, semicolons) fall back to the configured one."""
        from fastapi.testclient import TestClient

        bad = [
            "/b/x?next=<script>",
            '/b/x" onerror="y',
            "/b/x; exit 1",
            "/b/x <steer>",
        ]
        with TestClient(bot_env.app) as client:
            for value in bad:
                res = client.get(
                    "/api/state", headers={**AUTH, "X-Forwarded-Prefix": value}
                )
                assert res.json()["url_prefix"] == "", value

    def test_a_legit_roster_prefix_still_flows(self, bot_env):
        """The fix must not break the one value nginx actually sends."""
        from fastapi.testclient import TestClient

        with TestClient(bot_env.app) as client:
            res = client.get(
                "/api/state", headers={**AUTH, "X-Forwarded-Prefix": "/b/linkedin"}
            )
        assert res.json()["url_prefix"] == "/b/linkedin"

    def test_prefix_script_escapes_its_value(self, bot_env):
        """The sink itself escapes for the JS-in-HTML context, so even a value
        that got past the allowlist could not close the script tag."""
        out = bot_env._prefix_script(HOSTILE_PREFIX)
        assert out.startswith('<script>window.BA_PREFIX="')
        assert out.endswith('";</script>')
        # No literal markup survives inside the assignment.
        assert "</script><img" not in out
        assert "\\u003c/script\\u003e" in out
        assert "\\u003cimg src=x onerror=alert(1)\\u003e" in out
        # And the clean value is byte-identical to what the page always had.
        assert bot_env._prefix_script("/b/linkedin") == (
            '<script>window.BA_PREFIX="/b/linkedin";</script>'
        )

    def test_served_page_never_carries_the_hostile_value(self, bot_env):
        from fastapi.testclient import TestClient

        with TestClient(bot_env.app) as client:
            page = client.get(
                "/", headers={**AUTH, "X-Forwarded-Prefix": HOSTILE_PREFIX}
            )
        assert page.status_code == 200
        assert "onerror" not in page.text
        assert '<script>window.BA_PREFIX="";</script>' in page.text


class TestUnhandledErrors:
    def test_500s_carry_a_generic_body_and_a_request_id(self, bot_env):
        """SEC-BA-006: the handler answers "internal error" plus an id shared
        with the log line — never str(exc)."""
        from fastapi.testclient import TestClient

        marker = "SECRET-DETAIL /var/lib/browser-agent/imploded"

        @bot_env.app.get("/_boom_test")
        async def _boom() -> dict:  # pragma: no cover - raises by design
            raise RuntimeError(f"subprocess died: {marker}")

        try:
            with TestClient(bot_env.app, raise_server_exceptions=False) as client:
                res = client.get("/_boom_test", headers=AUTH)
        finally:
            bot_env.app.router.routes = [
                r for r in bot_env.app.router.routes
                if getattr(r, "path", None) != "/_boom_test"
            ]

        assert res.status_code == 500
        body = res.json()
        assert body["detail"] == "internal error"
        assert re.fullmatch(r"[0-9a-f]{12}", body["request_id"])
        assert marker not in res.text

    def test_browser_restart_failure_names_nothing(self, bot_env, monkeypatch):
        """The one inline detail=str(exc) on a 500 (browser restart) is gone too.

        stop() is patched as a no-op because it sits before the route's try —
        the failure has to come from start() to reach the guarded raise.
        """
        from fastapi.testclient import TestClient

        async def quiet_stop():
            return None

        async def broken_start():
            raise RuntimeError("chromium --flag-secret-value crashed")

        monkeypatch.setattr(bot_env.session, "stop", quiet_stop)
        monkeypatch.setattr(bot_env.session, "start", broken_start)

        with TestClient(bot_env.app, raise_server_exceptions=False) as client:
            res = client.post("/api/browser/restart", headers=AUTH)

        assert res.status_code == 500
        assert res.json()["detail"] == "browser restart failed"
        assert "chromium" not in res.text
