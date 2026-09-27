"""SEC-BA-008: POST /api/login navigates the LIVE, logged-in browser to a
caller-supplied URL, so that URL is validated before Chrome ever sees it —
http(s) only, and nothing that resolves into private, loopback, link-local
or metadata address space. These tests pin the guard directly; the DNS
dependency is stubbed at ``urlguard._resolve`` so the suite never touches
the network.
"""

from __future__ import annotations

import pytest

from browser_agent.urlguard import UrlRejected, validate_public_http_url


@pytest.fixture
def dns(monkeypatch):
    """A stubbable resolver table: host -> addresses (missing = NXDOMAIN)."""
    table: dict[str, list[str]] = {}
    monkeypatch.setattr(
        "browser_agent.urlguard._resolve", lambda host: list(table.get(host, []))
    )
    return table


class TestSchemeAndShape:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "chrome://settings",
            "view-source:https://example.com",
            "javascript:alert(1)",
            "data:text/html,hello",
            "ftp://example.com/pub",
            "//example.com/path",
            "example.com/page",
            "  ",
            "",
        ],
    )
    def test_non_http_targets_are_refused(self, url):
        with pytest.raises(UrlRejected):
            validate_public_http_url(url)

    def test_http_and_https_are_accepted_verbatim(self, dns):
        dns["example.com"] = ["93.184.216.34"]
        assert validate_public_http_url("https://example.com") == "https://example.com"
        assert (
            validate_public_http_url("http://example.com/page?q=1")
            == "http://example.com/page?q=1"
        )

    def test_the_url_is_returned_for_navigation(self, dns):
        dns["example.com"] = ["93.184.216.34"]
        assert (
            validate_public_http_url("https://example.com/login")
            == "https://example.com/login"
        )


class TestLiteralAddresses:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8000/api/tasks",
            "http://localhost/",
            "https://10.244.231.211/",  # a pod IP — the flat cluster network
            "http://172.17.0.1/admin",
            "http://192.168.3.106/router",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://100.64.0.3/",  # CGNAT/Tailscale — non-global
            "http://0.0.0.0/",
            "http://[::1]/",
            "http://[fd00::5]/",
            "http://[fe80::1]/",
            "http://224.0.0.1/",
        ],
    )
    def test_private_loopback_and_metadata_literals_are_refused(self, url):
        with pytest.raises(UrlRejected):
            validate_public_http_url(url)

    def test_a_public_literal_is_accepted(self):
        assert validate_public_http_url("http://8.8.8.8/") == "http://8.8.8.8/"


class TestResolvedHosts:
    def test_a_host_resolving_into_private_space_is_refused(self, dns):
        # The rebinding shape: a public-looking name that answers with an
        # internal address.
        dns["internal.example.test"] = ["10.1.2.3"]
        with pytest.raises(UrlRejected, match="non-public"):
            validate_public_http_url("https://internal.example.test/")

    def test_loopback_resolution_is_refused(self, dns):
        dns["localhost"] = ["127.0.0.1", "::1"]
        with pytest.raises(UrlRejected):
            validate_public_http_url("http://localhost/login")

    def test_every_resolved_address_must_be_public(self, dns):
        # One bad address in the answer is enough: the browser could land on
        # any of them.
        dns["mixed.example.test"] = ["93.184.216.34", "192.168.0.9"]
        with pytest.raises(UrlRejected):
            validate_public_http_url("https://mixed.example.test/")

    def test_an_unresolvable_host_is_refused_fail_closed(self, dns):
        with pytest.raises(UrlRejected, match="does not resolve"):
            validate_public_http_url("https://no-such-host.example.test/")

    def test_a_public_resolution_is_accepted(self, dns):
        dns["example.com"] = ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]
        assert validate_public_http_url("https://example.com/")


class TestLoginEndpoint:
    @pytest.fixture
    def bot_env(self, tmp_path, monkeypatch):
        """The same reload pattern the disclosure tests use."""
        import importlib
        import sys
        from pathlib import Path

        monkeypatch.setenv("DATA_ROOT", str(tmp_path))
        monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
        monkeypatch.setenv("CONTROL_TOKEN", "test-token")
        monkeypatch.setenv("LLM_ENABLED", "false")
        monkeypatch.chdir(tmp_path)
        sys_path = str(Path(__file__).resolve().parents[1] / "src")
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from browser_agent import api, config

        importlib.reload(config)
        importlib.reload(api)
        return api

    AUTH = {"Authorization": "Bearer test-token"}

    def test_the_endpoint_refuses_an_internal_target(self, bot_env, monkeypatch):
        """No navigation is attempted for a refused URL — the session stub
        would record a goto."""
        from fastapi.testclient import TestClient

        got: list[str] = []

        class Page:
            url = ""

        class Session:
            # The lifespan touches these on startup and shutdown; the test
            # only cares that /api/login validates before any goto happens.
            async def set_account(self, account):
                pass

            async def start(self):
                pass

            async def stop(self):
                pass

            async def page(self):
                return Page()

            async def goto(self, url):
                got.append(url)
                return Page()

        monkeypatch.setattr(bot_env, "session", Session())
        with TestClient(bot_env.app) as client:
            res = client.post(
                "/api/login", headers=self.AUTH, json={"url": "http://127.0.0.1:8000/"}
            )
        assert res.status_code == 400
        assert "refused" in res.json()["detail"]
        assert got == [], "the browser navigated to a refused URL"

    def test_the_endpoint_navigates_to_a_validated_public_url(self, bot_env, monkeypatch):
        from fastapi.testclient import TestClient

        got: list[str] = []

        class Page:
            url = "https://example.com/login"

        class Session:
            async def set_account(self, account):
                pass

            async def start(self):
                pass

            async def stop(self):
                pass

            async def page(self):
                return Page()

            async def goto(self, url):
                got.append(url)
                return Page()

        monkeypatch.setattr(bot_env, "session", Session())
        monkeypatch.setattr(
            "browser_agent.urlguard._resolve", lambda host: ["93.184.216.34"]
        )
        with TestClient(bot_env.app) as client:
            res = client.post(
                "/api/login", headers=self.AUTH, json={"url": "https://example.com/login"}
            )
        assert res.status_code == 200
        assert res.json()["opened"] == "https://example.com/login"
        assert got == ["https://example.com/login"]
