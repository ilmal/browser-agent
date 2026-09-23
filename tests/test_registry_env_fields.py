"""Registry environment fields (background, login_notes) and the hub's
bot-url template setting.

These are the contracts the farm feature stands on: an identity's persona and
login notes are per-bot registry fields that survive partial updates, and the
runner's target URL comes from a Settings knob so a laptop hub can be pointed
at reachable addresses.
"""

from __future__ import annotations

import dataclasses

import pytest

from browser_agent import config, registry


class TestEnvFieldsOnBot:
    def test_default_to_empty(self):
        bot = registry.Bot(profile="kai")
        assert bot.background == ""
        assert bot.login_notes == ""

    def test_load_reads_both(self, tmp_path):
        path = tmp_path / "bots.json"
        path.write_text(
            '{"bots": [{"profile": "kai", "background": "A meticulous planner.",'
            ' "login_notes": "kai@x.co / 2FA via authenticator"}]}'
        )
        reg = registry.load(path)
        bot = reg.get("kai")
        assert bot.background == "A meticulous planner."
        assert bot.login_notes == "kai@x.co / 2FA via authenticator"

    def test_load_tolerant_of_missing_keys(self, tmp_path):
        path = tmp_path / "bots.json"
        path.write_text('{"bots": [{"profile": "kai"}]}')
        bot = registry.load(path).get("kai")
        assert bot.background == ""
        assert bot.login_notes == ""

    def test_save_roundtrip(self, tmp_path):
        path = tmp_path / "bots.json"
        reg = registry.Registry()
        registry.upsert(reg, "kai", background="bg text", login_notes="login text")
        registry.save(path, reg)
        again = registry.load(path)
        assert again.get("kai").background == "bg text"
        assert again.get("kai").login_notes == "login text"

    def test_upsert_sets_and_limits(self):
        reg = registry.Registry()
        registry.upsert(reg, "kai",
                        background="b" * 3000, login_notes="l" * 3000)
        bot = reg.get("kai")
        assert len(bot.background) == 2000
        assert len(bot.login_notes) == 2000

    def test_upsert_never_clears_by_omission(self):
        """Patching the name must not wipe the login notes — these fields hold
        the persona and the credentials story, and an editor that saves one
        field alone is the normal case."""
        reg = registry.Registry()
        registry.upsert(reg, "kai", background="keep me", login_notes="and me")
        registry.upsert(reg, "kai", name="Kai Prime")
        bot = reg.get("kai")
        assert bot.background == "keep me"
        assert bot.login_notes == "and me"
        assert bot.name == "Kai Prime"

    def test_to_dict_carries_both(self):
        d = registry.Bot(profile="kai", background="b", login_notes="l").to_dict()
        assert d["background"] == "b"
        assert d["login_notes"] == "l"


class TestPatchBotModel:
    def test_accepts_the_new_fields(self):
        from browser_agent.hub import PatchBot

        req = PatchBot(background="who they are", login_notes="how to sign in")
        assert req.background == "who they are"
        assert req.login_notes == "how to sign in"
        assert req.name is None  # untouched fields stay None, not ""


class TestBotUrlTemplate:
    def test_default_targets_in_cluster_services(self):
        settings = config.load_settings()
        assert settings.bot_url_template == "http://profile-{profile}:{api_port}"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("BOT_URL_TEMPLATE", "http://localhost:{profile}:{api_port}")
        settings = config.load_settings()
        rendered = settings.bot_url_template.format(profile="kai", api_port=8000)
        assert rendered == "http://localhost:kai:8000"

    def test_settings_dataclass_still_constructs_minimally(self):
        """Other tests build Settings(**base) with only the required fields; a
        new field without a default would break every one of them."""
        fields = {f.name: f for f in dataclasses.fields(config.Settings)}
        for name in ("bot_url_template",):
            assert fields[name].default != dataclasses.MISSING
        s = config.Settings(
            profile="p", profiles_root=None, data_root=None,
            screen_width=1, screen_height=1, screen_depth=24, novnc_port=1,
            browser_base_url="", headless=False, slow_mo_ms=0,
            llm_base_url="", llm_model="", llm_enabled=False, llm_api_key="",
            llm_source="", llm_client_id="", planner_model="",
            planner_max_steps=1, planner_step_timeout_s=1, planner_client_id="",
            agent_max_steps=1, agent_timeout_s=1, laya_enabled=False,
            laya_decide_url="", laya_pick_enabled=False,
            laya_min_confidence=0.0, laya_game_min_confidence=0.0,
            laya_max_candidates=1, laya_pick_retries=1,
            api_port=8000, control_token="", ops_alert_url="",
            notify_on_escalation=False, browser_proxy="",
        )
        assert s.bot_url_template == "http://profile-{profile}:{api_port}"
