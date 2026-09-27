"""The mutating-request origin guard in require_token (hub and bot API).

Found live 2026-09-27: nginx stops inheriting the server-level
``proxy_set_header Host $host`` the moment a location declares its own header,
so the pod saw ``Host: 10.98.9.57:8000`` while the browser sent
``Origin: https://browser.ilmal.se`` — and the Origin-vs-Host comparison
refused every legitimate same-origin POST as "cross-origin request refused",
while GETs (which skip the check) kept working.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from browser_agent import config, hub


@pytest.fixture
def guard_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("REGISTRY_PATH", str(tmp_path / "data" / "bots.json"))
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("CONTROL_TOKEN", "test-token")
    monkeypatch.setenv("BOT_PROBE_TIMEOUT_S", "0.1")
    monkeypatch.setenv("RECIPES_DIR", str(tmp_path / "recipes"))
    importlib.reload(config)
    importlib.reload(hub)
    hub.farm_runner._interval = 3600.0
    return tmp_path


AUTH = {"Authorization": "Bearer test-token"}
# The live shape: a same-origin browser whose request was proxied so the pod's
# Host is the ClusterIP, not the public origin.
PROXIED_SAME_ORIGIN = {
    "Origin": "https://browser.ilmal.se",
    "Sec-Fetch-Site": "same-origin",
}


def _post(client, headers):
    """An invalid body on purpose: validation (422) proves the guard passed,
    with no side effects, while a refusal is 403."""
    return client.post("/api/farms", json={}, headers={**AUTH, **headers})


class TestOriginGuard:
    def test_same_origin_verdict_survives_a_rewritten_host(self, guard_env):
        with TestClient(hub.app) as client:
            res = _post(client, PROXIED_SAME_ORIGIN)
        assert res.status_code == 422

    def test_cross_site_verdict_is_refused(self, guard_env):
        with TestClient(hub.app) as client:
            res = _post(client, {"Origin": "https://evil.example",
                                 "Sec-Fetch-Site": "cross-site"})
        assert res.status_code == 403
        assert res.json()["detail"] == "cross-site request refused"

    def test_same_site_verdict_is_still_refused(self, guard_env):
        with TestClient(hub.app) as client:
            res = _post(client, {"Origin": "https://sibling.ilmal.se",
                                 "Sec-Fetch-Site": "same-site"})
        assert res.status_code == 403

    def test_without_fetch_header_a_foreign_origin_is_refused(self, guard_env):
        with TestClient(hub.app) as client:
            res = _post(client, {"Origin": "https://evil.example"})
        assert res.status_code == 403
        assert res.json()["detail"] == "cross-origin request refused"

    def test_without_fetch_header_a_matching_origin_passes(self, guard_env):
        with TestClient(hub.app) as client:
            res = _post(client, {"Origin": "http://testserver"})
        assert res.status_code == 422

    def test_get_is_never_origin_refused(self, guard_env):
        headers = {**AUTH, "Origin": "https://evil.example",
                   "Sec-Fetch-Site": "cross-site"}
        with TestClient(hub.app) as client:
            res = client.get("/api/bots", headers=headers)
        assert res.status_code == 200
