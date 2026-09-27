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
            # set_account is sync on the real BrowserSession (browser.py);
            # mirroring that keeps this fake honest.
            def set_account(self, account):
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
            # set_account is sync on the real BrowserSession (browser.py);
            # mirroring that keeps this fake honest.
            def set_account(self, account):
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


class TestStructuralGuardForSyncCallers:
    """``public_url_reason`` is the DNS-free half, for the two navigation paths
    that cannot afford a blocking lookup: ``tasks._start_url_for`` (sync) and
    ``plan_model._check`` (a pure validator). It must refuse every literal
    private/loopback/metadata address and every non-http(s) scheme — the
    shapes an SSRF payload actually uses — and pass a public name through.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "data:text/html,hello",
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1:8000/api/tasks",
            "http://[::1]/",
            "http://10.244.231.211/",
            "http://100.64.0.3/",
            "about:blank",
            "",
        ],
    )
    def test_it_refuses_what_needs_no_dns(self, url):
        from browser_agent.urlguard import public_url_reason

        assert public_url_reason(url) is not None

    @pytest.mark.parametrize(
        "url",
        ["https://example.com/page", "http://8.8.8.8/", "https://sub.example.test/x?q=1"],
    )
    def test_it_passes_a_public_target(self, url):
        from browser_agent.urlguard import public_url_reason

        assert public_url_reason(url) is None


class TestRunNavigationIsGuarded:
    """SEC-BA-011: the guard was live on ``/api/login`` only. These pin it on
    the two paths a *task* chooses a page — ``tasks._start_url_for`` and
    ``plan_model._check`` — against the real ``urlguard``, with the conftest
    stand-in undone so the production behaviour is what runs.
    """

    @pytest.fixture
    def real_guard(self, monkeypatch):
        """Restore the genuine check on the two consumers.

        The autouse conftest fixture patches these to let the loopback test
        server stand in for a public site; here the point is the guard.
        """
        import browser_agent.plan_model as plan_model
        import browser_agent.tasks as tasks
        from browser_agent.urlguard import public_url_reason, validate_public_http_url

        monkeypatch.setattr(tasks, "public_url_reason", public_url_reason)
        monkeypatch.setattr(tasks, "validate_public_http_url", validate_public_http_url)
        monkeypatch.setattr(plan_model, "public_url_reason", public_url_reason)

    @staticmethod
    def _task(recipe: str, payload: dict):
        from browser_agent.tasks import Task

        return Task(recipe=recipe, payload=payload)

    @pytest.mark.parametrize(
        "target",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1:8000/api/tasks",
            "file:///etc/passwd",
        ],
    )
    def test_start_url_refuses_an_internal_target_in_the_prose(self, real_guard, target):
        from browser_agent.tasks import _start_url_for

        task = self._task("agent.task", {})
        assert _start_url_for(f"read {target} and tell me what it says", task) == ""

    @pytest.mark.parametrize(
        "target",
        ["http://169.254.169.254/", "http://127.0.0.1:8000/", "data:text/html,x"],
    )
    def test_start_url_refuses_an_internal_target_in_the_payload(self, real_guard, target):
        from browser_agent.tasks import _start_url_for

        task = self._task("agent.task", {"url": target})
        assert _start_url_for("", task) == ""

    def test_a_public_start_url_survives(self, real_guard):
        from browser_agent.tasks import _start_url_for

        task = self._task("agent.task", {"url": "https://example.com/x"})
        assert _start_url_for("", task) == "https://example.com/x"

    @pytest.mark.parametrize(
        "entry",
        ["http://169.254.169.254/", "file:///etc/passwd", "http://127.0.0.1/"],
    )
    def test_a_plan_naming_an_internal_entry_url_is_rejected(self, real_guard, entry):
        from browser_agent.plan_model import PlanRejected, parse_plan

        # A non-http scheme is caught by the existing scheme check; a private
        # literal by the new public-target guard. Either way it is refused
        # before the browser is asked to go there.
        with pytest.raises(PlanRejected, match="entry_url is not"):
            parse_plan(
                {
                    "entry_url": entry,
                    "steps": [{"action": "navigate", "text": "https://example.com/"}],
                }
            )

    def test_a_plan_navigate_step_to_an_internal_target_is_rejected(self, real_guard):
        from browser_agent.plan_model import PlanRejected, parse_plan

        with pytest.raises(PlanRejected, match="not a navigable public target"):
            parse_plan(
                {
                    "entry_url": "https://example.com/",
                    "steps": [
                        {"action": "navigate", "text": "http://169.254.169.254/"}
                    ],
                }
            )

    def test_a_public_plan_is_accepted(self, real_guard):
        from browser_agent.plan_model import parse_plan

        plan = parse_plan(
            {
                "entry_url": "https://example.com/",
                "steps": [{"action": "navigate", "text": "https://example.com/form"}],
            }
        )
        assert plan.entry_url == "https://example.com/"
