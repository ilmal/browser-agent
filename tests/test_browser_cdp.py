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
