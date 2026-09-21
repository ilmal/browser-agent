"""The browser must expose a DevTools endpoint so the LLM agent can attach.

Regression guard for a silent failure: launching a *second* browser on the same
profile succeeds but starts logged out, so the agent would drive a different,
unauthenticated session than the one the human sees over noVNC.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent.browser import BrowserSession  # noqa: E402


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE", "cdp")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("HEADLESS", "true")
    monkeypatch.setenv("HTTP_PROXY", "")
    monkeypatch.setenv("HTTPS_PROXY", "")
    from browser_agent.config import load_settings

    return load_settings()


@pytest.mark.asyncio
async def test_running_browser_exposes_cdp_endpoint(settings):
    session = BrowserSession(settings)
    try:
        await session.start()
        endpoint = session.cdp_endpoint
        assert endpoint is not None, "no DevTools endpoint; the agent cannot attach"
        assert endpoint.startswith("http://127.0.0.1:")

        # It must actually be reachable — an attach is only real if the port
        # answers.
        import json
        import urllib.request

        with urllib.request.urlopen(f"{endpoint}/json/version", timeout=5) as r:
            assert json.load(r).get("Browser")
    finally:
        await session.stop()


@pytest.mark.asyncio
async def test_cdp_endpoint_absent_before_start(settings):
    session = BrowserSession(settings)
    assert session.cdp_endpoint is None


class _FakeHistory:
    """Minimal stand-in for browser-use's AgentHistoryList."""

    def __init__(self, *, final=None, successful=None, errors=None):
        self._final = final
        self._successful = successful
        self._errors = errors or []

    def final_result(self):
        return self._final

    def is_successful(self):
        return self._successful

    def has_errors(self):
        return bool(self._errors)

    def errors(self):
        return self._errors


def test_agent_llm_failure_is_a_failure_not_a_blocker():
    """An unreachable LLM must not read as 'a human must solve a captcha'."""
    from browser_agent.agent import _raise_for_no_result

    with pytest.raises(RuntimeError, match="agent could not complete"):
        _raise_for_no_result(
            _FakeHistory(successful=False, errors=["Connection refused"]),
            "https://x.test/",
        )


def test_agent_error_run_is_a_failure():
    from browser_agent.agent import _raise_for_no_result

    with pytest.raises(RuntimeError):
        _raise_for_no_result(_FakeHistory(errors=["step 3 blew up"]), "https://x.test/")


def test_agent_stopping_without_result_blocks():
    """Stopping cleanly but without a result is the human's call."""
    from browser_agent.agent import _raise_for_no_result
    from browser_agent.escalation import EscalationRequired

    with pytest.raises(EscalationRequired):
        _raise_for_no_result(_FakeHistory(successful=None), "https://x.test/")
