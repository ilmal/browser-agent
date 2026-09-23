"""The roster: naming, persistence, and the paths it hands out.

The roster is what turns "a pile of pods" into "my bots", so the things that
break it are quiet ones: a name that is accepted here but rejected by the
manifest generator, a bot whose live view points at the roster instead of
itself, or a create that registers a bot whose pod was never applied. Each of
those looks fine on the page and is wrong in a way only the operator notices.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_agent import registry  # noqa: E402


def test_profile_names_agree_with_the_generator():
    """A name the roster accepts must also be one add-profile.sh will accept.

    The two rules live in different languages, so this pins them to the same
    set: anything the API takes is something the generator can actually build,
    and neither can drift into accepting a name the other rejects.
    """
    import re
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "add-profile.sh"
    # Same regex as the shell guard, written once here as the contract.
    shell_rule = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")

    for name in ["x", "linkedin", "intercom-2", "a", "bot123", "a-b-c"]:
        assert registry.valid_profile(name), name
        assert shell_rule.match(name), name
        # And the generator really does accept it, run for real.
        out = subprocess.run([str(script), name], capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, (name, out.stderr)

    for name in ["", "Bad", "UPPER", "-lead", "trail-", "has space", "a_b", "bot.name"]:
        assert not registry.valid_profile(name), name
        assert not shell_rule.match(name), name
        out = subprocess.run([str(script), name], capture_output=True, text=True, timeout=30)
        assert out.returncode != 0, name


def test_generated_env_keeps_the_bot_naming_and_prefix():
    """The generator must emit the roster's fields, not just the bot's ones.

    Without BROWSER_URL_PREFIX the bot advertises the roster's /vnc.html, and
    without BOT_NAME the roster has nothing to show but the slug.
    """
    import subprocess

    script = Path(__file__).resolve().parents[1] / "scripts" / "add-profile.sh"
    out = subprocess.run(
        [str(script), "linkedin"],
        capture_output=True, text=True, timeout=30,
        env={"PATH": "/usr/bin:/bin", "BOT_NAME": "LinkedIn outreach",
             "BOT_JOB": "Post and reply as me: never without approval"},
    )
    assert out.returncode == 0, out.stderr
    assert '- { name: BROWSER_URL_PREFIX, value: "/b/linkedin" }' in out.stdout
    # Quoted in the YAML flow mapping: the job contains a colon, and bare it
    # would parse as a nested mapping and the whole manifest would fail.
    assert (
        '- { name: BOT_JOB, value: "Post and reply as me: never without approval" }' in out.stdout
    )

    import yaml

    dep = [d for d in yaml.safe_load_all(out.stdout) if d and d["kind"] == "Deployment"][0]
    env = {e["name"]: e["value"] for e in dep["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["BOT_JOB"] == "Post and reply as me: never without approval"
    assert env["BROWSER_URL_PREFIX"] == "/b/linkedin"


def test_registry_roundtrip_and_display_fallback(tmp_path: Path):
    path = tmp_path / "bots.json"
    reg = registry.Registry()
    registry.upsert(reg, "linkedin", name="LinkedIn outreach", job="Post and reply")
    registry.upsert(reg, "x", job="Post to X")
    registry.save(path, reg)

    back = registry.load(path)
    assert [b.profile for b in back.bots] == ["linkedin", "x"]
    assert back.get("x").to_dict()["display_name"] == "x"      # no name -> the slug
    assert back.get("linkedin").to_dict()["url"] == "/b/linkedin/"
    assert back.get("x").job == "Post to X"


def test_upsert_never_clears_a_field_by_omission(tmp_path: Path):
    """Editing the job must not wipe the notes.

    The roster's edit form sends only what changed, so a missing key has to
    mean "leave it alone" rather than "set it empty".
    """
    reg = registry.Registry()
    registry.upsert(reg, "x", name="X bot", job="Post", notes="logs in as @nils")
    registry.upsert(reg, "x", job="Post and reply")
    bot = reg.get("x")
    assert bot.job == "Post and reply"
    assert bot.name == "X bot"
    assert bot.notes == "logs in as @nils"


def test_a_corrupt_roster_does_not_take_the_hub_down(tmp_path: Path):
    """A half-written or hand-broken file must degrade to an empty roster.

    The hub has one page; a 500 on it means the operator cannot see or fix
    anything, including the file that is broken.
    """
    path = tmp_path / "bots.json"
    path.write_text("{ not json")
    assert registry.load(path).bots == []

    path.write_text(json.dumps({"bots": [
        {"profile": "good", "name": "Fine"},
        {"profile": "Bad Name"},        # rejected: not a valid profile
        "not even an object",
        {"profile": "alsogood"},
    ]}))
    assert [b.profile for b in registry.load(path).bots] == ["good", "alsogood"]


def test_text_is_cleaned_before_it_is_stored(tmp_path: Path):
    """Operator text is trimmed to one line and bounded.

    It is rendered into HTML, where the template escapes it; this is about the
    file staying readable and a pasted essay not becoming the roster.
    """
    reg = registry.Registry()
    registry.upsert(reg, "x", job="  post\n\n  and   reply  ", notes="y" * 900)
    bot = reg.get("x")
    assert bot.job == "post and reply"
    assert len(bot.notes) == 500


@pytest.fixture
def hub_env(tmp_path: Path, monkeypatch):
    """The hub app, with a temp registry and no real pods to probe."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("REGISTRY_PATH", str(tmp_path / "bots.json"))
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("CONTROL_TOKEN", "test-token")
    monkeypatch.setenv("BOT_PROBE_TIMEOUT_S", "0.2")
    # The recipe library the hub writes for every bot to read. A fresh one per
    # test, so an edit in one test cannot be seen by the next.
    monkeypatch.setenv("RECIPES_DIR", str(tmp_path / "recipes"))

    import importlib

    from browser_agent import config, hub, recipe_store

    importlib.reload(config)
    importlib.reload(recipe_store)
    importlib.reload(hub)
    return hub


@pytest.fixture
def bot_env(tmp_path: Path, monkeypatch):
    """A bot's own app. Separate from the hub: they are different processes in
    production and, importantly, different apps."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PROFILES_ROOT", str(tmp_path / "profiles"))
    monkeypatch.setenv("CONTROL_TOKEN", "test-token")

    import importlib

    from browser_agent import api, config

    importlib.reload(config)
    # Reloaded, not imported: another test module has almost certainly already
    # imported this, which would have bound the real /data path.
    importlib.reload(api)
    return api


def test_roster_reports_a_bot_with_no_pod_as_unreachable(hub_env, tmp_path: Path):
    """A roster line whose pod is not up is data, not an error.

    This is the state a just-created bot is in for its first few seconds, and
    the page has to render it rather than fail to load.
    """
    from fastapi.testclient import TestClient

    reg = hub_env.registry.Registry()
    hub_env.registry.upsert(reg, "linkedin", name="LinkedIn outreach", job="Post and reply")
    hub_env.registry.save(hub_env.settings.registry_path, reg)

    with TestClient(hub_env.app) as client:
        res = client.get("/api/bots", headers={"Authorization": "Bearer test-token"})
        assert res.status_code == 200
        bots = res.json()["bots"]
        assert len(bots) == 1
        assert bots[0]["display_name"] == "LinkedIn outreach"
        assert bots[0]["url"] == "/b/linkedin/"
        assert bots[0]["reachable"] is False
        assert bots[0]["signed_in"] is False


def test_roster_requires_the_token(hub_env):
    from fastapi.testclient import TestClient

    with TestClient(hub_env.app) as client:
        assert client.get("/api/bots").status_code == 401
        assert client.get("/api/bots", headers={"Authorization": "Bearer wrong"}).status_code == 401
        ok = client.get("/api/bots", headers={"Authorization": "Bearer test-token"})
        assert ok.status_code == 200


def test_duplicate_bot_is_refused_before_touching_the_cluster(hub_env):
    """Creating a bot that already exists must fail early and say so.

    The failure mode to avoid is kubectl applying a manifest that changes an
    existing bot's PVC, which is how a live login gets destroyed.
    """
    from fastapi.testclient import TestClient

    reg = hub_env.registry.Registry()
    hub_env.registry.upsert(reg, "x")
    hub_env.registry.save(hub_env.settings.registry_path, reg)

    with TestClient(hub_env.app) as client:
        res = client.post("/api/bots", headers={"Authorization": "Bearer test-token"},
                          json={"profile": "x", "name": "again"})
        assert res.status_code == 409


def test_an_invalid_name_is_refused(hub_env):
    from fastapi.testclient import TestClient

    with TestClient(hub_env.app) as client:
        res = client.post("/api/bots", headers={"Authorization": "Bearer test-token"},
                          json={"profile": "Bad Name"})
        assert res.status_code == 400


def test_remove_takes_the_bot_out_and_says_which_one(tmp_path: Path):
    """Removing has to report *what* it removed, so the caller can tell a real
    deletion from a request for a bot that was never there."""
    reg = registry.Registry()
    registry.upsert(reg, "x", name="X poster", job="Post to X")
    registry.upsert(reg, "linkedin", name="LinkedIn outreach")

    gone = registry.remove(reg, "x")
    assert gone is not None and gone.name == "X poster"
    assert [b.profile for b in reg.bots] == ["linkedin"]

    # Nothing left to remove: the caller answers 404 rather than reporting a
    # deletion that did not happen.
    assert registry.remove(reg, "x") is None
    assert registry.remove(reg, "never-existed") is None


def test_remove_persists(tmp_path: Path):
    path = tmp_path / "bots.json"
    reg = registry.Registry()
    registry.upsert(reg, "x", job="Post to X")
    registry.upsert(reg, "linkedin", job="Post and reply")
    registry.save(path, reg)

    back = registry.load(path)
    registry.remove(back, "x")
    registry.save(path, back)

    assert [b.profile for b in registry.load(path).bots] == ["linkedin"]


@pytest.fixture
def fake_cluster(tmp_path: Path, monkeypatch):
    """A hub whose generator and kubectl are harmless stand-ins.

    The generator records the env it was handed (that is what carries BOT_NAME
    and BOT_JOB into the pod), and kubectl records the argv of each call so a
    test can assert what was and was not deleted.
    """
    gen = tmp_path / "add-profile.sh"
    gen.write_text(
        "#!/bin/sh\n"
        "printf -- '- { name: BOT_NAME, value: \"%s\" }\\n' \"$BOT_NAME\"\n"
        "printf -- '- { name: BOT_JOB, value: \"%s\" }\\n' \"$BOT_JOB\"\n"
        "printf -- '- { name: BROWSER_URL_PREFIX, value: \"/b/%s\" }\\n' \"$1\"\n"
    )
    gen.chmod(0o755)

    kubectl = tmp_path / "kubectl"
    log = tmp_path / "kubectl.log"
    kubectl.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$KUBECTL_LOG\"\nexit 0\n")
    kubectl.chmod(0o755)
    monkeypatch.setenv("KUBECTL_LOG", str(log))

    return {"gen": gen, "kubectl": kubectl, "log": log}


def _hub_with(hub_env, fake_cluster, tmp_path: Path):
    """Repoint the loaded hub at the stand-ins."""
    import dataclasses

    hub_env.settings = dataclasses.replace(
        hub_env.settings,
        add_profile_script=str(fake_cluster["gen"]),
        kubectl_bin=str(fake_cluster["kubectl"]),
    )
    return hub_env


def test_create_passes_the_name_and_job_through_to_the_pod(hub_env, fake_cluster, tmp_path: Path):
    """The roster's name must be the name the *pod* shows.

    add-profile.sh has always read BOT_NAME/BOT_JOB and written them into the
    Deployment, but create_bot ran it with no env, so a bot the operator named
    "LinkedIn outreach" came up calling itself "linkedin" inside its own window.
    The two names have to be one name.
    """
    from fastapi.testclient import TestClient

    hub = _hub_with(hub_env, fake_cluster, tmp_path)
    with TestClient(hub.app) as client:
        res = client.post("/api/bots", headers={"Authorization": "Bearer test-token"},
                          json={"profile": "linkedin", "name": "LinkedIn outreach",
                                "job": "Post and reply as me: never without approval"})
        assert res.status_code == 200, res.text

    manifest = (tmp_path / "profile-linkedin.yaml").read_text()
    assert '- { name: BOT_NAME, value: "LinkedIn outreach" }' in manifest
    assert '- { name: BOT_JOB, value: "Post and reply as me: never without approval" }' in manifest
    # And the registry kept the same name, so the roster and the pod agree.
    assert res.json()["bot"]["display_name"] == "LinkedIn outreach"


def test_delete_drops_the_roster_entry_and_spares_the_pvc(hub_env, fake_cluster, tmp_path: Path):
    """Remove is not "log this bot out".

    The PVC *is* the login, so the default delete must take the pod and leave
    the storage alone — otherwise removing a card silently signs the bot out of
    every account it holds.
    """
    from fastapi.testclient import TestClient

    hub = _hub_with(hub_env, fake_cluster, tmp_path)
    reg = hub.registry.Registry()
    hub.registry.upsert(reg, "x", name="X poster")
    hub.registry.save(hub.settings.registry_path, reg)

    with TestClient(hub.app) as client:
        res = client.delete("/api/bots/x", headers={"Authorization": "Bearer test-token"})
        assert res.status_code == 200, res.text
        assert res.json()["ok"] is True
        assert res.json()["purged"] is False

    assert hub.registry.load(hub.settings.registry_path).get("x") is None

    log = fake_cluster["log"].read_text()
    assert "delete deployment profile-x" in log
    assert "delete service profile-x" in log
    assert "delete pvc" not in log


def test_delete_purge_also_removes_the_storage(hub_env, fake_cluster, tmp_path: Path):
    """Purging is the explicit opt-in that does destroy the login."""
    from fastapi.testclient import TestClient

    hub = _hub_with(hub_env, fake_cluster, tmp_path)
    reg = hub.registry.Registry()
    hub.registry.upsert(reg, "x")
    hub.registry.save(hub.settings.registry_path, reg)

    with TestClient(hub.app) as client:
        res = client.delete("/api/bots/x?purge=true",
                            headers={"Authorization": "Bearer test-token"})
        assert res.status_code == 200, res.text
        assert res.json()["purged"] is True

    assert "delete pvc profile-x" in fake_cluster["log"].read_text()


def test_delete_of_an_unknown_bot_is_404_and_touches_nothing(hub_env, fake_cluster, tmp_path: Path):
    from fastapi.testclient import TestClient

    hub = _hub_with(hub_env, fake_cluster, tmp_path)
    with TestClient(hub.app) as client:
        assert client.delete("/api/bots/nobody").status_code == 401     # token first
        res = client.delete("/api/bots/nobody", headers={"Authorization": "Bearer test-token"})
        assert res.status_code == 404

    assert not fake_cluster["log"].exists()


def test_the_roster_offers_a_remove_control(hub_env):
    """The page has to have the affordance the route backs, or the operator can
    create bots and never get rid of them."""
    roster = Path(__file__).resolve().parents[1] / "src" / "browser_agent" / "ui" / "roster.html"
    html = roster.read_text()
    assert 'data-remove=' in html
    assert 'data-confirm-remove=' in html
    assert 'method: "DELETE"' in html
    # Removing is destructive and irreversible from this page, so it must ask.
    assert "Remove bot" in html


def test_the_schedule_mode_takes_a_clock_per_member(hub_env):
    """The point of a scheduled farm is per-person times — "at 10 this person,
    at 11 that person" — so one shared epoch is not enough: every chosen
    member gets its own input, and the submit sends those as start_times."""
    roster = Path(__file__).resolve().parents[1] / "src" / "browser_agent" / "ui" / "roster.html"
    html = roster.read_text()
    assert "function renderFarmTimes()" in html
    assert "data-time-for=" in html
    # Re-rendered on every member pick and mode change, and reset after a
    # successful create, so the rows always describe the current choice.
    assert html.count("renderFarmTimes()") >= 4
    assert "body.start_times = times" in html
    # A member with neither an override nor the shared time blocks the create.
    assert "pick a start time for" in html


def test_the_bot_page_advertises_the_prefixed_live_view(bot_env):
    """Under a roster the live view is the bot's own, not the roster's.

    origin + "/vnc.html" is right when one domain serves one bot, and wrong
    the moment several share a domain: it would open the roster's viewer. The
    prefix comes from the proxy, and both endpoints the UI reads must carry it.
    """
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        res = client.get("/api/whoami", headers={
            "Authorization": "Bearer test-token",
            "X-Forwarded-Prefix": "/b/linkedin",
        })
        assert res.json()["url_prefix"] == "/b/linkedin"

        state = client.get("/api/state", headers={
            "Authorization": "Bearer test-token",
            "X-Forwarded-Prefix": "/b/linkedin",
        })
        assert state.json()["url_prefix"] == "/b/linkedin"
        assert state.json()["takeover_url"] == "/b/linkedin/vnc.html"

        # And with no prefix — the plain one-domain-per-bot deployment — the
        # old behaviour is untouched.
        plain = client.get("/api/state", headers={"Authorization": "Bearer test-token"})
        assert plain.json()["url_prefix"] == ""
        assert plain.json()["takeover_url"] == "/vnc.html"


def test_the_static_setting_is_the_fallback_when_no_proxy_says_otherwise(bot_env, monkeypatch):
    """BROWSER_URL_PREFIX is what the manifest sets; the proxy header wins.

    A bot reached directly (a local docker run, a debug port-forward) has no
    proxy to send a header, so the configured prefix has to apply on its own.
    """
    from fastapi.testclient import TestClient

    bot_env.settings = bot_env.load_settings.__globals__["load_settings"]()
    monkeypatch.setenv("BROWSER_URL_PREFIX", "/b/from-env")
    bot_env.settings = bot_env.load_settings()
    assert bot_env.settings.url_prefix == "/b/from-env"

    with TestClient(bot_env.app) as client:
        plain = client.get("/api/whoami", headers={"Authorization": "Bearer test-token"})
        assert plain.json()["url_prefix"] == "/b/from-env"


def test_the_served_page_carries_the_prefix(bot_env):
    """The page must know its prefix BEFORE its first request.

    The UI fetches root-absolute paths ("/api/state"), so behind a roster they
    resolve against the origin — the hub — and the bot page renders the roster's
    state while looking healthy. The prefix therefore has to be in the HTML the
    bot served, not fetched afterwards; there is no request it could fetch.
    """
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        page = client.get("/", headers={
            "Authorization": "Bearer test-token",
            "X-Forwarded-Prefix": "/b/linkedin",
        })
        assert page.status_code == 200
        assert '<script>window.BA_PREFIX="/b/linkedin";</script>' in page.text
        # The marker itself must not survive into the response.
        assert "<!--PREFIX-->" not in page.text

        # Standalone — the plain deployment — injects empty, not a stray path.
        plain = client.get("/", headers={"Authorization": "Bearer test-token"})
        assert '<script>window.BA_PREFIX="";</script>' in plain.text

        # And the page really uses it: every api() call is prefixed.
        assert "fetch(PREFIX + path" in page.text


def test_the_live_view_url_keeps_the_prefix_with_a_base_url(bot_env, monkeypatch):
    """A configured BROWSER_BASE_URL must not drop the roster prefix.

    browser_base_url is the deployment's own domain, which under a roster is
    the ROSTER's. Using it raw sent "Open the browser" to the roster's
    /vnc.html — a live view of the wrong window that looks like it worked.
    The prefix is what distinguishes them and it belongs in both branches.
    """
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        state = client.get("/api/state", headers={
            "Authorization": "Bearer test-token",
            "X-Forwarded-Prefix": "/b/linkedin",
        }).json()
        assert state["url_prefix"] == "/b/linkedin"

        page = client.get("/", headers={"Authorization": "Bearer test-token"}).text
        # The page composes base + prefix + "/vnc.html"; with the roster base
        # set this is the exact expression that was wrong.
        assert "base + prefix + \"/vnc.html\"" in page


def test_the_live_view_url_tells_novnc_where_its_socket_is(bot_env):
    """The live view must carry autoconnect and its own websockify path.

    noVNC resolves "path" against the PAGE's directory and defaults it to
    "websockify". Unprefixed that is the origin root, which is right for the
    plain one-bot deployment but wrong under a roster: the page would open the
    ROSTER's socket, or sit at the connect panel when the path does not exist.
    Both failures look like a working page that shows nothing.
    """
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        page = client.get("/", headers={"Authorization": "Bearer test-token"}).text
        # autoconnect, so the operator does not have to find the Connect button
        assert "autoconnect=1" in page
        # and the socket is named relative to the proxy's mount, not the origin
        assert 'const sock = (prefix + "/websockify").replace(/^\\//, "");' in page

    # The roster's own links must carry the same two parameters.
    roster = Path(__file__).resolve().parents[1] / "src" / "browser_agent" / "ui" / "roster.html"
    html = roster.read_text()
    assert "function liveUrl(profile)" in html
    assert "autoconnect=1" in html
    assert "path=b/${esc(profile)}/websockify" in html
    # No bare vnc.html links left on the roster: those are the ones that hang.
    assert 'href="${esc(url)}vnc.html"' not in html


# ---- the recipe library -----------------------------------------------------
#
# The hub is the library's only writer, and every bot reads what it writes, so
# the tests that matter are the refusals: a recipe the pods could not run must
# never reach the ConfigMap. Each of these asserts a 400 with the reason, not
# just a status code.

_STEP_RECIPE = {
    "name": "my-scrape",
    "description": "Search and read the results",
    "entry_url": "https://example.com",
    "steps": [
        {"action": "navigate", "goal": "open the site", "text": "https://example.com"},
        {"action": "click", "goal": "run the search", "selector": "button.go"},
    ],
}


def _hub_token() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def test_the_library_lists_every_recipe_with_its_origin(hub_env):
    from fastapi.testclient import TestClient

    with TestClient(hub_env.app) as client:
        body = client.get("/api/recipes", headers=_hub_token()).json()

    names = {r["name"]: r for r in body["recipes"]}
    assert {"x.post", "plan.task", "agent.task"} <= set(names)
    assert names["x.post"]["origin"] == "builtin"
    assert names["x.post"]["overridable"]
    assert names["agent.task"]["overridable"] == []
    # The editor needs the vocabulary the pods will accept.
    assert body["actions"] == ["navigate", "click", "type", "extract", "wait"]
    assert body["errors"] == []


def test_a_recipe_is_validated_before_it_is_stored(hub_env, tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(hub_env.app) as client:
        assert client.post(
            "/api/recipes/validate", json=_STEP_RECIPE, headers=_hub_token()
        ).json()["ok"] is True

        bad = {**_STEP_RECIPE, "steps": [{"action": "click", "goal": "press it"}]}
        refused = client.post("/api/recipes/validate", json=bad, headers=_hub_token()).json()
        assert refused["ok"] is False
        assert "needs a `selector`" in refused["error"]

        # And the same gate is what a save goes through: nothing reached disk.
        assert client.post("/api/recipes", json=bad, headers=_hub_token()).status_code == 400
        assert not (tmp_path / "recipes" / "my-scrape.json").exists()


def test_a_reserved_name_cannot_be_saved_over(hub_env):
    from fastapi.testclient import TestClient

    with TestClient(hub_env.app) as client:
        res = client.post(
            "/api/recipes", json={**_STEP_RECIPE, "name": "plan.task"}, headers=_hub_token()
        )
        assert res.status_code == 400
        assert "reserved" in res.json()["detail"]


def test_saving_and_deleting_a_recipe_round_trips(hub_env, tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(hub_env.app) as client:
        assert client.post("/api/recipes", json=_STEP_RECIPE, headers=_hub_token()).status_code == 200
        assert (tmp_path / "recipes" / "my-scrape.json").is_file()

        listed = {r["name"]: r for r in client.get("/api/recipes", headers=_hub_token()).json()["recipes"]}
        assert listed["my-scrape"]["origin"] == "stored"
        assert len(listed["my-scrape"]["steps"]) == 2

        removed = client.request(
            "DELETE", "/api/recipes/my-scrape", headers=_hub_token()
        ).json()
        assert removed["ok"] is True
        assert removed["removed"] == "my-scrape"
        assert removed["kind"] == "stored"
        # No cluster here, so the removal is on the hub and nowhere else — and
        # the answer says so instead of implying every bot already agrees.
        assert removed["published"] is False
        assert removed["publish_error"]
        assert not (tmp_path / "recipes" / "my-scrape.json").exists()

    # Deleting something that was never there is a 404, not a silent success.
    with TestClient(hub_env.app) as client:
        assert client.request(
            "DELETE", "/api/recipes/my-scrape", headers=_hub_token()
        ).status_code == 404


def test_an_unknown_config_key_is_refused(hub_env, tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(hub_env.app) as client:
        res = client.put(
            "/api/recipes/x.post/config",
            json={"values": {"no.such.key": 1}},
            headers=_hub_token(),
        )
        assert res.status_code == 400
        assert "not an overridable key" in res.json()["detail"]

        ok = client.put(
            "/api/recipes/x.post/config",
            json={"values": {"selectors.submit": ["button.new"]}},
            headers=_hub_token(),
        ).json()
        assert ok["overrides"] == {"selectors.submit": ["button.new"]}

    # The override lands in the one file every pod mounts.
    stored = json.loads((tmp_path / "recipes" / "_overrides.json").read_text())
    assert stored == {"x.post": {"selectors.submit": ["button.new"]}}


def test_every_overridable_key_has_a_default_to_show(hub_env):
    """The editor renders a field per overridable key, with its default beside it.

    An override only means something against a default, so a key with no default
    is a field the operator cannot reason about — and one the recipe would never
    consult anyway, since cfg() falls back to a literal in Python. This pins the
    two lists together: the closed key list and the values read off the modules
    must not drift apart in either direction.
    """
    from fastapi.testclient import TestClient

    with TestClient(hub_env.app) as client:
        body = client.get("/api/recipes", headers=_hub_token()).json()

    assert body["actions"], "the composer needs the step vocabulary"
    for r in body["recipes"]:
        keys = set(r["overridable"])
        if not keys:
            # agent.task: the freeform path, whose only input is the task text.
            assert not r["defaults"], f"{r['name']} exposes no keys but has defaults"
            continue
        missing = keys - set(r["defaults"])
        assert not missing, f"{r['name']} offers {sorted(missing)} with no default to show"
        extra = set(r["defaults"]) - keys
        assert not extra, f"{r['name']} has defaults for unofferable keys: {sorted(extra)}"
        # A default is what the recipe falls back to, so it is never null.
        assert all(v is not None for v in r["defaults"].values())


def test_a_composed_recipe_is_offered_back_for_editing(hub_env, tmp_path):
    """Composing one and reopening it must show the same steps.

    The composer edits a stored recipe by restating its steps, so the round trip
    has to be lossless — a step that came back different would be a silent edit
    the operator never made.
    """
    from fastapi.testclient import TestClient

    spec = {
        "name": "my-scrape",
        "description": "read the results",
        "entry_url": "https://example.com",
        "steps": [
            {"action": "navigate", "goal": "open it", "text": "https://example.com"},
            {"action": "extract", "goal": "read them", "selector": ".result",
             "done_when": {"selector_visible": ".result"}},
        ],
    }
    with TestClient(hub_env.app) as client:
        assert client.post("/api/recipes", json=spec, headers=_hub_token()).status_code == 200
        listed = {r["name"]: r for r in client.get("/api/recipes", headers=_hub_token()).json()["recipes"]}

    got = listed["my-scrape"]
    assert got["origin"] == "stored"
    steps = got["steps"]
    assert [(s.get("action"), s.get("goal"), s.get("selector"), s.get("text")) for s in steps] == [
        ("navigate", "open it", None, "https://example.com"),
        ("extract", "read them", ".result", None),
    ], "a step came back changed"
    # done_when is carried, not dropped: it is the step's proof of success, and
    # the composer does not offer it, so a round trip is the only thing that can
    # preserve it. The schema fills its absent options with None; the content is
    # what matters.
    assert steps[1]["done_when"]["selector_visible"] == ".result"
    # A composed recipe is edited as steps, not as config, so it exposes no keys.
    assert got["overridable"] == []


def test_the_library_lives_where_the_pods_mount_it(hub_env, tmp_path: Path):
    """A hub writing anywhere else is a library no bot can see.

    The same path is the ConfigMap's mount in every pod, so this pins the hub
    and the manifests to one location.
    """
    assert hub_env.settings.recipes_dir == tmp_path / "recipes"


def test_a_write_is_not_published_unless_the_cluster_is_configured(hub_env, monkeypatch):
    """A hub with no cluster must not reach for one.

    Publishing is off unless RECIPES_PUBLISH is set, which is what keeps a hub
    run from a laptop — or a test — from writing to whatever kubectl happens to
    be pointed at. The edit is still stored; ``published`` says it is not live.
    """
    from fastapi.testclient import TestClient

    assert hub_env.settings.recipes_publish is False
    with TestClient(hub_env.app) as client:
        body = client.post("/api/recipes", json=_STEP_RECIPE, headers=_hub_token()).json()
    assert body["published"] is False
    # And it says why. "Not live" without a reason is the version of this that
    # gets debugged for an hour: the operator cannot tell a misconfigured hub
    # from a missing kubectl from an RBAC gap.
    assert body["publish_error"]


def test_publishing_projects_the_whole_directory(hub_env, tmp_path: Path, monkeypatch):
    """The ConfigMap's keys are exactly the library's files.

    One command builds it from the directory, so adding and deleting a recipe
    both land — a hand-maintained key list would drift the moment a file was
    removed. The hub's own file on disk is what is rendered; nothing is read
    back from the cluster.
    """
    from fastapi.testclient import TestClient

    (tmp_path / "recipes").mkdir(parents=True)
    (tmp_path / "recipes" / "a.json").write_text("{}")
    hub_env.settings = dataclasses.replace(hub_env.settings, recipes_publish=True)

    seen: list[list[str]] = []

    class _Done:
        returncode = 0
        stdout = "apiVersion: v1\nkind: ConfigMap\n"
        stderr = ""

    def _run(cmd, **kwargs):
        seen.append(list(cmd))
        return _Done()

    monkeypatch.setattr(hub_env.subprocess, "run", _run)
    monkeypatch.setattr(hub_env, "_which", lambda _b: "/usr/bin/kubectl")

    published, error = hub_env._publish_library()

    assert published is True
    assert error == ""
    renders = [c for c in seen if "configmap" in c]
    assert len(renders) == 1
    assert f"--from-file={tmp_path / 'recipes'}" in renders[0]
    assert "browser-agent-recipes" in renders[0]
    # And it is applied to this namespace, not the default one.
    applied = [c for c in seen if c[-2:] == ["-f", "-"]]
    assert applied and "browser-agent" in applied[0]


# -- the bot's retry conversation ------------------------------------------
#
# /say is the endpoint that makes a retry a conversation. Two things matter and
# both are easy to get wrong: that "talk to it" does the right thing in *both*
# states, and that the message is recorded even when the task is mid-run (a
# steer that leaves no trace makes the thread disagree with the History).
#
# Which branch /say takes must not depend on whether a real browser happened to
# be fast enough to start the task first, so these tests stop the runner's
# worker and set the state they mean. The behaviour under test is the dispatch,
# not the scheduler.


def _idle(bot_env):
    """Stop the worker so a submitted task stays queued instead of running."""
    worker = bot_env.runner._worker
    if worker is not None:
        worker.cancel()
    bot_env.runner._worker = None


def test_say_queues_the_next_attempt_for_a_finished_task(bot_env):
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        headers = {"Authorization": "Bearer test-token"}
        _idle(bot_env)
        made = client.post("/api/tasks", json={"recipe": "agent.task", "payload": {}},
                           headers=headers).json()
        assert made["status"] == "queued"

        res = client.post(f"/api/tasks/{made['id']}/say",
                          json={"text": "use the other button instead"}, headers=headers)
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["ran"] is True
        assert body["task"]["id"] != made["id"]

        thread = client.get(f"/api/threads/{made['thread_id']}", headers=headers).json()
        assert [m["text"] for m in thread["messages"]] == ["use the other button instead"]
        assert len(thread["attempts"]) == 2
        assert [a["attempt"] for a in thread["attempts"]] == [1, 2]


def test_say_steers_a_running_task_instead_of_queuing(bot_env):
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        headers = {"Authorization": "Bearer test-token"}
        _idle(bot_env)
        made = client.post("/api/tasks", json={"recipe": "agent.task", "payload": {}},
                           headers=headers).json()
        task = bot_env.runner.tasks[made["id"]]
        # Pretend it is mid-run, which is the state the operator is in when they
        # want to correct a task rather than replace it.
        from browser_agent.control import Control
        from browser_agent.tasks import TaskStatus

        bot_env.runner.current = task
        bot_env.runner.control = Control()
        task.status = TaskStatus.RUNNING

        res = client.post(f"/api/tasks/{made['id']}/say",
                          json={"text": "stop, use the other menu"}, headers=headers)
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["ran"] is False
        assert "task" not in body
        assert body["control"]["amendment"] == "stop, use the other menu"
        # Still one attempt: steering does not mint a second one.
        assert len(bot_env.runner.tasks) == 1


def test_say_records_a_mid_run_steer_in_the_thread(bot_env):
    """A steer that leaves no trace makes the thread disagree with the History."""
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        headers = {"Authorization": "Bearer test-token"}
        _idle(bot_env)
        made = client.post("/api/tasks", json={"recipe": "agent.task", "payload": {}},
                           headers=headers).json()
        task = bot_env.runner.tasks[made["id"]]
        from browser_agent.control import Control
        from browser_agent.tasks import TaskStatus

        bot_env.runner.current = task
        bot_env.runner.control = Control()
        task.status = TaskStatus.RUNNING

        client.post(f"/api/tasks/{made['id']}/say", json={"text": "try the search box"},
                    headers=headers)
        thread = client.get(f"/api/threads/{made['thread_id']}", headers=headers).json()
        assert [m["text"] for m in thread["messages"]] == ["try the search box"]
        assert thread["messages"][0]["kind"] == "instruction"


def test_say_stamps_the_instruction_into_goal_too(bot_env):
    """The bug this closes: agent.py reads ``goal`` FIRST, so text-only lands
    nowhere and the operator's new instruction silently does nothing."""
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        headers = {"Authorization": "Bearer test-token"}
        _idle(bot_env)
        made = client.post("/api/tasks", json={"recipe": "agent.task", "payload": {}},
                           headers=headers).json()
        nxt = client.post(f"/api/tasks/{made['id']}/say",
                          json={"text": "do it the other way"}, headers=headers).json()["task"]
        assert nxt["payload"]["goal"] == "do it the other way"
        assert nxt["payload"]["task"] == "do it the other way"


def test_say_refuses_an_empty_message_and_a_bad_kind(bot_env):
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        headers = {"Authorization": "Bearer test-token"}
        _idle(bot_env)
        made = client.post("/api/tasks", json={"recipe": "agent.task", "payload": {}},
                           headers=headers).json()
        assert client.post(f"/api/tasks/{made['id']}/say", json={"text": "   "},
                           headers=headers).status_code == 400
        assert client.post(f"/api/tasks/{made['id']}/say",
                           json={"text": "hi", "kind": "config"}, headers=headers).status_code == 400


def test_say_on_an_unknown_task_is_404(bot_env):
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        res = client.post("/api/tasks/nope/say", json={"text": "hi"},
                          headers={"Authorization": "Bearer test-token"})
        assert res.status_code == 404


def test_a_thread_reports_its_attempts_and_messages(bot_env):
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        headers = {"Authorization": "Bearer test-token"}
        _idle(bot_env)
        made = client.post("/api/tasks", json={"recipe": "agent.task", "payload": {}},
                           headers=headers).json()
        thread = client.get(f"/api/threads/{made['thread_id']}", headers=headers).json()
        assert thread["thread_id"] == made["thread_id"]
        assert [a["id"] for a in thread["attempts"]] == [made["id"]]
        assert thread["messages"] == []
        assert client.get("/api/threads/nope", headers=headers).status_code == 404


def test_activity_can_read_a_finished_attempts_feed(bot_env):
    """Without this the panel could show the live run and nothing else."""
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        headers = {"Authorization": "Bearer test-token"}
        _idle(bot_env)
        made = client.post("/api/tasks", json={"recipe": "agent.task", "payload": {}},
                           headers=headers).json()
        res = client.get(f"/api/activity?task_id={made['id']}", headers=headers)
        assert res.status_code == 200
        assert res.json()["entries"] == []
        assert res.json()["task_id"] == made["id"]
        assert client.get("/api/activity?task_id=nope", headers=headers).status_code == 404


def test_the_retry_endpoint_stamps_goal_as_well(bot_env):
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        headers = {"Authorization": "Bearer test-token"}
        _idle(bot_env)
        made = client.post("/api/tasks", json={"recipe": "agent.task", "payload": {}},
                           headers=headers).json()
        nxt = client.post(f"/api/tasks/{made['id']}/retry",
                          json={"instructions": "try the menu"}, headers=headers).json()
        assert nxt["payload"]["goal"] == "try the menu"
        assert nxt["thread_id"] == made["thread_id"]
        assert nxt["attempt"] == 2


def test_a_retry_records_the_instruction_in_the_thread(bot_env):
    from fastapi.testclient import TestClient

    with TestClient(bot_env.app) as client:
        headers = {"Authorization": "Bearer test-token"}
        _idle(bot_env)
        made = client.post("/api/tasks", json={"recipe": "agent.task", "payload": {}},
                           headers=headers).json()
        client.post(f"/api/tasks/{made['id']}/retry",
                    json={"instructions": "try the menu"}, headers=headers)
        thread = client.get(f"/api/threads/{made['thread_id']}", headers=headers).json()
        assert [m["text"] for m in thread["messages"]] == ["try the menu"]


def test_the_thread_box_cannot_reload_the_page(bot_env):
    """A <button> inside a <form> submits it by default, so the page navigated
    and the operator lost what they typed. Every button in the page must say
    type="button" explicitly — asserted file-wide, because the composer's
    structure is allowed to change while its safety is not."""
    import re

    page = (Path(__file__).resolve().parents[1] / "src" / "browser_agent" / "ui"
            / "index.html").read_text()
    assert "<form" in page, "the page carries forms"
    for tag in re.findall(r"<button[^>]*>", page):
        assert 'type="button"' in tag, f"a button would submit its form: {tag}"


def test_probe_urls_follow_bot_url_template(hub_env, monkeypatch):
    """The reachability probe builds its URL from bot_url_template, the same
    knob the farm runner uses — two URL builders for the same pod is how a
    deploy target that only exists in one of them goes unnoticed."""
    object.__setattr__(hub_env.settings, "bot_url_template",
                       "http://probe-hit:{profile}:{api_port}")
    seen: list[str] = []

    class FakeResp:
        status_code = 200

        def json(self):
            return {"profile": "linkedin", "name": "LinkedIn", "signed_in": False}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, headers=None):
            seen.append(url)
            return FakeResp()

    monkeypatch.setattr(hub_env.httpx, "AsyncClient", FakeClient)

    from fastapi.testclient import TestClient

    reg = hub_env.registry.Registry()
    hub_env.registry.upsert(reg, "linkedin", name="LinkedIn outreach")
    hub_env.registry.save(hub_env.settings.registry_path, reg)

    with TestClient(hub_env.app) as client:
        res = client.get("/api/bots", headers={"Authorization": "Bearer test-token"})
        assert res.status_code == 200

    assert seen, "the probe never asked any pod"
    assert seen == [f"http://probe-hit:linkedin:{hub_env.settings.api_port}/api/whoami"]
