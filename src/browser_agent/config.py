"""Environment-driven configuration.

One process = one bot. A bot owns several Chrome profiles — the accounts in
accounts.py — and runs one of them at a time; the bot's own identity and data
directory come from the environment, so the same image backs every pod in the
deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .accounts import ACCOUNTS_FILE, DEFAULT_ACCOUNT, account_dir


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


def _env_opt_int(name: str) -> int | None:
    """An int that may legitimately be absent — "" and unset are the same thing."""
    raw = _env(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


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

    # Laya — a decision model used by plan.task to pick which page element to
    # activate and to confirm step outcomes. Two backends: the shared
    # decision engine over HTTP (LAYA_DECIDE_URL, prod = llm-service's
    # /v1/decide proxy) or the in-process pip package when no URL is set.
    # Every failure mode degrades to "off": a broken gate must cost a repair
    # attempt at most, never a crash and never a human escalation.
    laya_enabled: bool
    laya_decide_url: str
    laya_pick_enabled: bool
    laya_min_confidence: float
    #: Confidence floor for the minesweeper tiebreak, deliberately separate
    #: from ``laya_min_confidence``. The two roles differ in what a wrong
    #: answer costs: a wrong *pick* clicks the wrong element and is
    #: unrecoverable, whereas a wrong minesweeper guess costs one life in a
    #: game the solver can still win. Reusing the picker's floor left the
    #: tiebreak silent (the engine's choice confidences rarely reach 0.75),
    #: so this starts lower and is tuned from measurement, not taste.
    laya_game_min_confidence: float
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

    # Element picker for plan.task's selectorless click/type steps. Same
    # numbered-lines/bind-by-index contract as the laya picker, but the model
    # is an LLM: benched in the prod pod 2026-09-21, laya could not
    # discriminate candidates on either head (choice 4/18 with flat
    # probabilities, yes/no tournament all p in 0.41-0.55) while
    # deepseek-v4.1-flash went 18/18 at ~1 s/pick — see
    # scripts/laya_pick_bench.py and scripts/llm_pick_bench.py. Laya keeps
    # the confirm role, where it measures well. Off by default; prod turns
    # it on explicitly.
    picker_enabled: bool = False
    picker_model: str = "deepseek-v4.1-flash"
    picker_client_id: str = "browser-agent-picker"
    picker_timeout_s: float = 20.0
    # Agree-by-two: on a page with more than one candidate the pick is re-asked
    # in shuffled order and both answers must name the same element before it
    # is bound. The harness survey measured position bias as the LLM-picker
    # failure mode that a one-token answer cannot self-report; ~1 s more per
    # multi-candidate pick is cheap against a wrong click burning a whole run.
    picker_verify: bool = True

    # ---- what the roster calls this bot. All optional: a profile generated by
    # hand or run from a laptop needs none of them, and each has a sane
    # fallback (the profile name, and the pod's own origin).
    #
    # bot_name/bot_job are what the roster shows: a profile name is an
    # identifier ("x", "linkedin"), while a bot is named for a job.
    bot_name: str = ""
    bot_job: str = ""
    # Which account of this bot the process starts on. One bot owns several
    # Chrome user-data-dirs (see accounts.py) and runs exactly one at a time: a
    # user-data-dir is single-writer, and there is one X display and one x11vnc
    # per pod, so a second simultaneous identity is not something the display
    # could even show. Normally empty — the account store's `active` decides,
    # which is what makes switching an account a restart rather than a
    # redeploy — and set only to pin a pod to one identity.
    account: str = ""
    # The path this bot is reached under, when a roster proxies several bots
    # from one domain: "/b/x". Empty in the plain one-domain-per-profile
    # deployment, which is the default. Purely cosmetic — it only affects the
    # URL the bot advertises for its own live view, never how it is routed.
    url_prefix: str = ""

    # The operator's recipe library. One directory, shared by every bot: the hub
    # writes it and each profile pod reads it — in the cluster, a k8s ConfigMap
    # mounted as a directory. It carries the built-in recipes' config overrides
    # and the step-recipes the operator composed, so changing a recipe no longer
    # means a code deploy. An absent or empty directory leaves every built-in
    # literal standing, which is the whole behaviour before this existed: a pod
    # without the mount is exactly as capable as it was.
    recipes_dir: Path = Path("/recipes")

    #: Hub only: the ConfigMap the library is published to, and which every bot
    #: pod mounts read-only at ``recipes_dir``. The hub's own ``recipes_dir`` is
    #: a plain volume — this is the projection step that makes an edit visible to
    #: every profile. Named here rather than inline so the manifest, the RBAC and
    #: the code all point at one name.
    recipes_configmap: str = "browser-agent-recipes"

    # ---- learned recipes (Nils, 2026-09-23) ----
    # Where a recipe the *agent* earned is written. Deliberately NOT the
    # operator's ``recipes_dir``: that directory is a ConfigMap the hub owns and
    # every pod mounts read-only, and a learned recipe carries the literal text
    # the agent typed into a logged-in session, which must stay on the pod's own
    # volume rather than be published to the cluster. Empty disables harvesting.
    learned_recipes_dir: Path = Path("")
    #: Whether a successful agent run is written down as a replayable recipe at
    #: all. On by default: writing a spec costs nothing, and a spec that never
    #: replays is inert. It is the *routing* below that needs the guardrail.
    learn_recipes: bool = True
    #: How many clean replays a learned recipe needs before the router will pick
    #: it for a sentence that no hand-authored recipe claims. Two, not one: a
    #: single replay only proves the page had not moved that morning, and the
    #: cost of being wrong is a run that reports a different task's result as
    #: this one's. Below the count it is offered, never auto-routed.
    learned_recipe_min_replays: int = 2

    #: Hub only, and OFF by default. Publishing the library to the cluster is a
    #: deployment-specific act — whether this hub has a cluster to publish to,
    #: and whether it is allowed to — so the real value is set at deploy time
    #: rather than assumed. A hub run from a laptop, or a test, must never reach
    #: for someone's cluster because a write endpoint was called.
    recipes_publish: bool = False

    # ---- hub (roster) only. A bot process never reads these. ----
    # The roster lives on a PVC: it must survive a pod restart, and a bot's
    # profile is the wrong place for it (several bots share one roster).
    registry_path: Path = Path("/data/bots.json")
    bot_probe_timeout_s: float = 3.0
    # How the hub reaches a bot pod's API: {profile} and {api_port} are
    # substituted per call. Bare service DNS resolves from inside the cluster
    # (the hub pod has the namespace search path); the override exists for a
    # hub run from a laptop, where the pod names are unreachable.
    bot_url_template: str = "http://profile-{profile}:{api_port}"
    namespace: str = "browser-agent"
    kubectl_bin: str = "kubectl"
    # Where the "new bot" button finds the manifest generator. Baked into the
    # image at the repo root, so the default is right inside the container and
    # the repo-relative path is right outside it.
    add_profile_script: str = ""

    @property
    def profile_root(self) -> Path:
        """This bot's profile volume directory: /profiles/<profile>.

        Holds the account metadata file and one Chrome user-data-dir per
        account. It is **not** a user-data-dir itself any more, and Chrome must
        never be pointed at it — that is the whole reason ``profile_dir``
        reaches one level deeper.
        """
        return self.profiles_root / self.profile

    @property
    def accounts_path(self) -> Path:
        return self.profile_root / ACCOUNTS_FILE

    def profile_dir_for(self, account: str) -> Path:
        """The Chrome user-data-dir of one named account.

        Everything that launches, attaches to, locks or inspects a browser goes
        through here, so switching accounts is a change of directory and nothing
        else — no second code path, and no chance of one caller still reading
        the pre-accounts layout.
        """
        return account_dir(self.profile_root, account)

    @property
    def profile_dir(self) -> Path:
        """Persistent Chrome user-data-dir for the *running* account.

        The account named on the environment if the operator pinned one,
        otherwise ``default`` — which is what every bot migrated from the
        single-profile layout has, and what keeps a bot that predates accounts
        working with no redeploy. This is deliberately not the store's
        ``active``: config is built once at import and the store is mutable, so
        resolving the store's choice here would freeze it at boot and make a
        switch silently do nothing. The runner resolves ``active`` explicitly
        and rebinds the session (see ``BrowserSession.set_account``).
        """
        return self.profile_dir_for(self.account or DEFAULT_ACCOUNT)

    @property
    def state_db(self) -> Path:
        return self.data_root / "state.db"

    @property
    def runs_db(self) -> Path:
        """The durable run archive — see ``runstore``. On the data volume, so it
        survives the pod recreate every deploy performs."""
        return self.data_root / "runs.db"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_root / "artifacts" / self.profile

    @property
    def display_name(self) -> str:
        return self.bot_name or self.profile

    @property
    def display_job(self) -> str:
        return self.bot_job


def load_settings() -> Settings:
    return Settings(
        profile=_env("AGENT_PROFILE", "default"),
        # Normally unset: the store's `active` decides, and switching is a
        # restart, not a redeploy. Set only to pin a pod to one account.
        account=_env("AGENT_ACCOUNT"),
        bot_name=_env("BOT_NAME"),
        bot_job=_env("BOT_JOB"),
        url_prefix=_env("BROWSER_URL_PREFIX").rstrip("/"),
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
        # deepseek-v4.1-flash: the ollama_cloud leg, which is the one vision-
        # capable model that actually answers (probed 2026-09-21: 200 in 0.79s,
        # and it describes an image_url part correctly). The previous default,
        # glm-4.5v, is zai-ONLY — llm-service's explicit_chain_for gives it no
        # twin, because the ollama_cloud catalogue has no vision GLM — so when
        # zai wedged (240s budget, 0 content chars) every escalation hung and
        # the UI showed "llm: unreachable". A model with no second leg has no
        # failover; pick one on the leg that is up.
        llm_model=_env("LLM_MODEL", "deepseek-v4.1-flash"),
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
        laya_decide_url=_env("LAYA_DECIDE_URL"),
        laya_pick_enabled=_env("LAYA_PICK_ENABLED", "true").lower() in {"1", "true", "yes"},
        laya_min_confidence=_env_float("LAYA_MIN_CONFIDENCE", 0.75),
        laya_game_min_confidence=_env_float("LAYA_GAME_MIN_CONFIDENCE", 0.55),
        # 10, not more: laya warns that confidence for choice buckets >=11 is
        # uncalibrated (its checkpoint ships clamp-distorted temperatures
        # there), and 10 options fit the 512-token budget beside the page
        # state. A target outside the top 10 falls to the agent fallback.
        laya_max_candidates=_env_int("LAYA_MAX_CANDIDATES", 10),
        laya_pick_retries=_env_int("LAYA_PICK_RETRIES", 1),
        registry_path=Path(_env("REGISTRY_PATH", "/data/bots.json")),
        bot_probe_timeout_s=_env_float("BOT_PROBE_TIMEOUT_S", 3.0),
        bot_url_template=_env("BOT_URL_TEMPLATE", "http://profile-{profile}:{api_port}"),
        namespace=_env("AGENT_NAMESPACE", "browser-agent"),
        kubectl_bin=_env("KUBECTL_BIN", "kubectl"),
        add_profile_script=_env(
            "ADD_PROFILE_SCRIPT",
            str(Path(__file__).resolve().parent.parent.parent / "scripts" / "add-profile.sh"),
        ),
        api_port=_env_int("API_PORT", 8000),
        control_token=_env("CONTROL_TOKEN"),
        recipes_dir=Path(_env("RECIPES_DIR", "/recipes")),
        recipes_configmap=_env("RECIPES_CONFIGMAP", "browser-agent-recipes"),
        recipes_publish=_env("RECIPES_PUBLISH", "false").lower() in {"1", "true", "yes"},
        # Default lives under the pod's own data PVC, beside the pick log: it is
        # written data about this bot, not part of the operator's shared library.
        learned_recipes_dir=Path(_env("LEARNED_RECIPES_DIR", "/data/learned-recipes")),
        learn_recipes=_env("LEARN_RECIPES", "true").lower() in {"1", "true", "yes"},
        learned_recipe_min_replays=_env_int("LEARNED_RECIPE_MIN_REPLAYS", 2),
        ops_alert_url=_env("OPS_ALERT_URL"),
        notify_on_escalation=_env("NOTIFY_ON_ESCALATION", "true").lower() in {"1", "true", "yes"},
        browser_proxy=_env("BROWSER_PROXY"),
        picker_enabled=_env("PICKER_ENABLED", "false").lower() in {"1", "true", "yes"},
        picker_model=_env("PICKER_MODEL", "deepseek-v4.1-flash"),
        picker_client_id=_env("PICKER_CLIENT_ID", "browser-agent-picker"),
        picker_timeout_s=_env_float("PICKER_TIMEOUT_S", 20.0),
        picker_verify=_env("PICKER_VERIFY", "true").lower() in {"1", "true", "yes"},
    )
