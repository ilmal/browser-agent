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


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
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
    # through a reverse proxy. Empty means the caller reaches the pod ports
    # directly. The real value is deployment-specific, so it is applied at
    # deploy time rather than committed.
    browser_base_url: str

    # Browser
    headless: bool
    slow_mo_ms: int

    # LLM fallback (OpenAI-compatible). Points at llm-service by default.
    llm_base_url: str
    llm_model: str
    llm_enabled: bool
    # llm-service is authenticated: every token-spending call needs a key, the
    # X-LLM-Source header, and a client_id in the body (the OpenAI-compatible
    # `user` field — llm-service has no header fallback for it). All three were
    # missing, which surfaced as 403 "Invalid API key." on every escalation.
    llm_api_key: str
    llm_source: str
    llm_client_id: str

    # Planner (the plan.task recipe). Same llm-service endpoint and client key
    # as the fallback — the planner is an accelerator in front of the same
    # control plane, not a second service — but a faster text model: it only
    # has to emit JSON, not look at screenshots.
    planner_model: str
    planner_max_steps: int
    planner_step_timeout_s: int
    planner_client_id: str

    # The freeform agent fallback. Without a bound it runs until the model
    # decides to stop, which on a page with nothing to do is never — a task sat
    # in RUNNING indefinitely and, because the queue has a single worker, every
    # later task stayed QUEUED behind it. A step cap ends the loop; the wall
    # clock is the backstop for a step that hangs inside the browser.
    agent_max_steps: int
    agent_timeout_s: int

    # Laya — a 322M local classifier used by plan.task to pick which page
    # element to activate and to confirm step outcomes. An optional dependency
    # whose every failure mode degrades to "off": a broken gate must cost a
    # repair attempt at most, never a crash and never a human escalation.
    laya_enabled: bool
    laya_pick_enabled: bool
    laya_min_confidence: float
    laya_max_candidates: int
    laya_pick_retries: int

    # Control plane
    api_port: int
    control_token: str

    # Alerts
    ops_alert_url: str
    notify_on_escalation: bool

    # Browser egress proxy. Deliberately its own variable rather than the
    # conventional HTTP_PROXY/HTTPS_PROXY: those are honoured implicitly by
    # every httpx/requests client in the process, so the in-cluster LLM call
    # would be routed through the egress proxy and fail. This value is passed
    # to Chrome explicitly at launch instead.
    browser_proxy: str

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
        llm_api_key=_env("LLM_API_KEY"),
        llm_source=_env("LLM_SOURCE", "browser-agent"),
        llm_client_id=_env("LLM_CLIENT_ID", "browser-agent"),
        # deepseek-v4.1-flash: fastest planner on llm-service's ollama_cloud
        # backend that returns clean JSON (probed 2026-09-21). NOTE: the
        # dash-spelled gpt-oss-120b routes to a backend our key cannot use.
        planner_model=_env("PLANNER_MODEL", "deepseek-v4.1-flash"),
        planner_max_steps=_env_int("PLANNER_MAX_STEPS", 12),
        planner_step_timeout_s=_env_int("PLANNER_STEP_TIMEOUT_S", 10),
        agent_max_steps=_env_int("AGENT_MAX_STEPS", 25),
        agent_timeout_s=_env_int("AGENT_TIMEOUT_S", 600),
        planner_client_id=_env("PLANNER_CLIENT_ID", "browser-agent-planner"),
        laya_enabled=_env("LAYA_ENABLED", "true").lower() in {"1", "true", "yes"},
        laya_pick_enabled=_env("LAYA_PICK_ENABLED", "true").lower() in {"1", "true", "yes"},
        laya_min_confidence=_env_float("LAYA_MIN_CONFIDENCE", 0.75),
        # 10, not more: laya warns that confidence for choice buckets >=11 is
        # uncalibrated (its checkpoint ships clamp-distorted temperatures
        # there), and 10 options fit the 512-token budget beside the page
        # state. A target outside the top 10 falls to the agent fallback.
        laya_max_candidates=_env_int("LAYA_MAX_CANDIDATES", 10),
        laya_pick_retries=_env_int("LAYA_PICK_RETRIES", 1),
        api_port=_env_int("API_PORT", 8000),
        control_token=_env("CONTROL_TOKEN"),
        ops_alert_url=_env("OPS_ALERT_URL"),
        notify_on_escalation=_env("NOTIFY_ON_ESCALATION", "true").lower() in {"1", "true", "yes"},
        browser_proxy=_env("BROWSER_PROXY"),
    )
