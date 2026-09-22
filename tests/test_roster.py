"""The roster: naming, persistence, and the paths it hands out.

The roster is what turns "a pile of pods" into "my bots", so the things that
break it are quiet ones: a name that is accepted here but rejected by the
manifest generator, a bot whose live view points at the roster instead of
itself, or a create that registers a bot whose pod was never applied. Each of
those looks fine on the page and is wrong in a way only the operator notices.
"""

from __future__ import annotations

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

    import importlib

    from browser_agent import config, hub

    importlib.reload(config)
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
