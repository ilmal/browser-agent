"""Environment-driven configuration.

One process = one profile. The profile name and its data directory come from
the environment so that the same image backs every pod in the deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for a single profile container."""

    profile: str
    profiles_root: Path
    data_root: Path

    # Display / noVNC
    screen_width: int
    screen_height: int
    screen_depth: int
    novnc_port: int

    # Public base URL for the admin UI + browser view, when the pod is reached
    # through a reverse proxy (e.g. https://browser.ilmal.se). Empty means the
    # caller reaches the pod ports directly.
    browser_base_url: str

    # Browser
    headless: bool
    slow_mo_ms: int

    # LLM fallback (OpenAI-compatible). Points at llm-service by default.
    llm_base_url: str
    llm_model: str
    llm_enabled: bool

    # Control plane
    api_port: int
    control_token: str

    # Alerts
    ops_alert_url: str
    notify_on_escalation: bool

    # Egress
    http_proxy: str
    https_proxy: str

    @property
    def profile_dir(self) -> Path:
        """Persistent Chrome user-data-dir for this profile."""
        return self.profiles_root / self.profile

    @property
    def state_db(self) -> Path:
        return self.data_root / "state.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_root / "artifacts" / self.profile


def load_settings() -> Settings:
    return Settings(
        profile=_env("AGENT_PROFILE", "default"),
        profiles_root=Path(_env("PROFILES_ROOT", "/profiles")),
        data_root=Path(_env("DATA_ROOT", "/data")),
        screen_width=_env_int("SCREEN_WIDTH", 1440),
        screen_height=_env_int("SCREEN_HEIGHT", 900),
        screen_depth=_env_int("SCREEN_DEPTH", 24),
        novnc_port=_env_int("NOVNC_PORT", 6080),
        browser_base_url=_env("BROWSER_BASE_URL"),
        headless=_env("HEADLESS", "false").lower() in {"1", "true", "yes"},
        slow_mo_ms=_env_int("SLOW_MO_MS", 0),
        llm_base_url=_env("LLM_BASE_URL", "http://llm-service.llm-service.svc.cluster.local:8001/v1"),
        llm_model=_env("LLM_MODEL", "glm-4.5v"),
        llm_enabled=_env("LLM_ENABLED", "true").lower() in {"1", "true", "yes"},
        api_port=_env_int("API_PORT", 8000),
        control_token=_env("CONTROL_TOKEN"),
        ops_alert_url=_env("OPS_ALERT_URL"),
        notify_on_escalation=_env("NOTIFY_ON_ESCALATION", "true").lower() in {"1", "true", "yes"},
        http_proxy=_env("HTTP_PROXY"),
        https_proxy=_env("HTTPS_PROXY"),
    )
