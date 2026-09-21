"""The browser must be running when the pod is up, not when a task first runs.

Everything a human sees over noVNC is the pod's X display, and the browser was
started lazily by the first task. A freshly started pod therefore served noVNC
straight to an empty black root window: the display was fine, the client
connected, and there was simply nothing on screen — which reads as "it takes
forever to see anything". These tests pin the pre-warm at startup, and pin that
a broken browser cannot take the control plane down with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _FakePage:
    def __init__(self):
        self.url = "about:blank"
        self.gotos: list[str] = []

    async def goto(self, url, **kw):
        self.gotos.append(url)
        self.url = url
        return self


class _FakeSession:
    def __init__(self, *, raise_on_start=False):
        self.started = 0
        self.stopped = 0
        self.page_obj = _FakePage()
        self.raise_on_start = raise_on_start

    async def start(self):
        self.started += 1
        if self.raise_on_start:
            raise RuntimeError("chrome blew up")

    async def page(self):
        return self.page_obj

    async def stop(self):
        self.stopped += 1


class _FakeRunner:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1


@pytest.fixture
def api_mod(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_PROFILE", "pytest")
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    for mod in [m for m in list(sys.modules) if m.startswith("browser_agent")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    import browser_agent.api as api

    return api


@pytest.mark.asyncio
async def test_startup_starts_the_browser_not_just_the_runner(api_mod, monkeypatch):
    session = _FakeSession()
    runner = _FakeRunner()
    monkeypatch.setattr(api_mod, "session", session)
    monkeypatch.setattr(api_mod, "runner", runner)

    async with api_mod.lifespan(api_mod.app):
        assert session.started == 1, "the browser was left to start lazily"
        assert runner.started == 1


@pytest.mark.asyncio
async def test_startup_paints_something_instead_of_a_blank_screen(api_mod, monkeypatch):
    session = _FakeSession()  # its page starts at about:blank
    monkeypatch.setattr(api_mod, "session", session)
    monkeypatch.setattr(api_mod, "runner", _FakeRunner())

    async with api_mod.lifespan(api_mod.app):
        gotos = session.page_obj.gotos
        assert gotos, "a blank about:blank page was left on screen"
        assert gotos[0].startswith("data:text/html"), (
            "the idle page must not depend on the network"
        )


@pytest.mark.asyncio
async def test_a_broken_browser_does_not_take_the_control_plane_down(api_mod, monkeypatch):
    session = _FakeSession(raise_on_start=True)
    runner = _FakeRunner()
    monkeypatch.setattr(api_mod, "session", session)
    monkeypatch.setattr(api_mod, "runner", runner)

    async with api_mod.lifespan(api_mod.app):
        # The API must still come up: the human can drive noVNC by hand even
        # when the automated browser failed to start.
        assert runner.started == 1
