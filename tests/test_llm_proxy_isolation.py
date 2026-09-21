"""The in-cluster LLM call must not inherit the browser's egress proxy.

Regression: the pod exported HTTP_PROXY/HTTPS_PROXY for Chrome's residential
egress. httpx honours those implicitly, so the LLM call was routed through a
SOCKS proxy and died with "socksio package is not installed" — the fallback
layer was silently dead. The proxy is now BROWSER_PROXY (applied to Chrome at
launch only), and the LLM client additionally sets trust_env=False.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent.config import load_settings  # noqa: E402
from browser_agent.llm import LLMClient  # noqa: E402


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm-service.llm-service.svc.cluster.local:8001/v1")
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    return load_settings()


def test_conventional_proxy_env_does_not_reach_settings(settings, monkeypatch):
    """HTTP_PROXY/HTTPS_PROXY must not populate any setting.

    If they did, httpx would apply them to the LLM call.
    """
    monkeypatch.setenv("HTTP_PROXY", "socks5://egress.example:1080")
    monkeypatch.setenv("HTTPS_PROXY", "socks5://egress.example:1080")
    reloaded = load_settings()

    assert reloaded.browser_proxy == ""
    for field in ("browser_proxy",):
        assert "egress.example" not in getattr(reloaded, field)


def test_browser_proxy_is_read_from_its_own_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWSER_PROXY", "socks5://office-proxy.crawl.svc.cluster.local:1080")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    assert load_settings().browser_proxy == (
        "socks5://office-proxy.crawl.svc.cluster.local:1080"
    )


@pytest.mark.asyncio
async def test_llm_client_ignores_ambient_proxy(settings, monkeypatch):
    """An ambient HTTP_PROXY must not be applied to the LLM request.

    httpx with trust_env=True would pick up the proxy and, for a socks5://
    scheme, fail before any connection. trust_env=False keeps it direct.
    """
    monkeypatch.setenv("HTTP_PROXY", "socks5://egress.example:1080")
    monkeypatch.setenv("HTTPS_PROXY", "socks5://egress.example:1080")

    seen: dict = {}

    class _FakeClient:
        def __init__(self, *a, **kw):
            seen.update(kw)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url):
            seen["url"] = url

            class _R:
                status_code = 200

            return _R()

    monkeypatch.setattr("browser_agent.llm.httpx.AsyncClient", _FakeClient)

    assert await LLMClient(settings).healthy() is True
    assert seen.get("trust_env") is False, "LLM client must not honour ambient proxies"
