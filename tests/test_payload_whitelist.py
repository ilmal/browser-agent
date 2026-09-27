"""SEC-BA-011: the run form's extra-JSON box used to merge its object into
the task payload verbatim, so any key downstream code treats as trusted
could be set from the client. The API now drops every payload key outside
the allowlist at the door. These tests pin both halves: the sanitizer
itself, and the route that hands a client payload to the runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

sys_path = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def bot_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("CONTROL_TOKEN", "test-token")
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.chdir(tmp_path)
    import importlib
    import sys

    if str(sys_path) not in sys.path:
        sys.path.insert(0, str(sys_path))
    from browser_agent import api, config

    importlib.reload(config)
    importlib.reload(api)
    return api


AUTH = {"Authorization": "Bearer test-token"}


class TestSanitizer:
    def test_every_key_a_recipe_or_the_runner_reads_survives(self, bot_env):
        payload = {
            "task": "t", "text": "t", "goal": "g", "url": "https://example.com",
            "max_steps": 3, "page_id": "p", "games": 2,
            "admin_url": "https://example.com/admin",
        }
        assert bot_env._sanitize_payload(payload) == payload

    def test_unknown_keys_are_dropped(self, bot_env):
        out = bot_env._sanitize_payload(
            {"task": "t", "start_url": "https://evil.test", "trusted": True,
             "recipe_override": {"entry_url": "https://evil.test"}}
        )
        assert out == {"task": "t"}

    def test_a_non_dict_payload_degrades_to_empty(self, bot_env):
        assert bot_env._sanitize_payload(None) == {}
        assert bot_env._sanitize_payload([("task", "t")]) == {}

    def test_dropping_is_logged(self, bot_env, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="browser_agent.api"):
            bot_env._sanitize_payload({"task": "t", "rogue": 1})
        assert any("rogue" in r.message for r in caplog.records)


class TestTaskRoute:
    class _FakeTask:
        thread_id = "thread-1"
        created_at = "2026-09-27T00:00:00"
        payload: dict = {}

        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return {"id": "t1", "thread_id": self.thread_id, "status": "queued"}

    def test_rogue_keys_never_reach_the_runner(self, bot_env, monkeypatch):
        from fastapi.testclient import TestClient

        seen: dict = {}

        def fake_submit(recipe, payload):
            seen["recipe"], seen["payload"] = recipe, payload
            return self._FakeTask(dict(payload))

        monkeypatch.setattr(bot_env.runner, "submit", fake_submit)
        monkeypatch.setattr(bot_env.threads, "say", lambda *a, **k: None)
        with TestClient(bot_env.app) as client:
            res = client.post(
                "/api/tasks",
                headers=AUTH,
                json={
                    "recipe": "agent.task",
                    "payload": {
                        "task": "open the page",
                        "url": "https://example.com",
                        # The mass-assignment surface: keys the UI never sends.
                        "start_url": "https://evil.test",
                        "trusted": True,
                        "entry_url": "https://evil.test",
                        "history": ["forged"],
                    },
                },
            )
        assert res.status_code == 200
        assert seen["payload"]["url"] == "https://example.com"
        assert seen["payload"]["task"] == "open the page"
        assert "start_url" not in seen["payload"]
        assert "trusted" not in seen["payload"]
        assert "entry_url" not in seen["payload"]
        assert "history" not in seen["payload"]

    def test_legitimate_recipe_keys_reach_the_runner(self, bot_env, monkeypatch):
        from fastapi.testclient import TestClient

        seen: dict = {}

        def fake_submit(recipe, payload):
            seen["recipe"], seen["payload"] = recipe, payload
            return self._FakeTask(dict(payload))

        monkeypatch.setattr(bot_env.runner, "submit", fake_submit)
        monkeypatch.setattr(bot_env.threads, "say", lambda *a, **k: None)
        with TestClient(bot_env.app) as client:
            res = client.post(
                "/api/tasks",
                headers=AUTH,
                json={
                    "recipe": "minesweeper.play",
                    "payload": {"task": "play", "games": 3, "max_steps": 12},
                },
            )
        assert res.status_code == 200
        assert seen["payload"]["games"] == 3
        assert seen["payload"]["max_steps"] == 12
