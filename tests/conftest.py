"""Repository-wide test guards.

The guard here exists because of an incident, so it is worth stating plainly: a
test that exercised the hub's recipe save called the *real* ``kubectl`` and wrote
a ConfigMap into the live ``browser-agent`` namespace. The hub's own guard
(``RECIPES_PUBLISH``, off by default) is the first line of defence, but it is a
*setting* — and a test that turns it on to exercise the publish path would reach
a cluster again. This is the second line, and it is not a setting.

It is deliberately a **denylist, not a blanket ban on subprocesses**: the E2E
tests drive a real browser, and Playwright launches its own driver process. What
must never happen is a test running a command that reaches the cluster or another
machine. A test that wants *those* commands to "run" monkeypatches
``subprocess.run`` and asserts on the argv it was handed — which is what the
publish tests do, and what they should have been doing all along.
"""

from __future__ import annotations

import os
import subprocess

import pytest

#: Programs that act on something outside this machine — a cluster, a container
#: runtime, a remote host. A unit test has no business running any of these, and
#: every reason to run one is a reason to fake the argv instead.
FORBIDDEN = frozenset(
    {"kubectl", "kubeadm", "helm", "oc", "ssh", "scp", "rsync", "docker", "podman"}
)

_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen


def _first(args) -> str:
    """The program from a command, however it was spelled."""
    if isinstance(args, (str, bytes, os.PathLike)):
        text = os.fsdecode(args).strip()
        return text.split()[0] if text else ""
    parts = list(args)
    return os.fsdecode(parts[0]) if parts else ""


def _refuse(args) -> None:
    """Refuse a command that would act on a cluster or another machine.

    Only a **bare program name** is refused. That spelling means "whatever is on
    PATH", which here is the real tool, and running it is the incident this
    guard exists to prevent. A test that passes a *path* has supplied its own
    stand-in — ``fake_cluster`` writes a harmless shell stub named ``kubectl`` —
    and that is precisely the safe pattern, so it is allowed through.
    """
    program = _first(args)
    if os.path.basename(program) not in FORBIDDEN:
        return
    if os.path.dirname(program):
        return
    raise AssertionError(
        f"a test tried to run {program!r} from PATH, which reaches beyond this "
        "machine. Tests must not touch a cluster or a remote host: point the "
        "setting at a stand-in, or monkeypatch subprocess.run/Popen and assert "
        "on the argv. (See tests/test_roster.py's fake_cluster and publish "
        "tests for both patterns.)"
    )


@pytest.fixture(autouse=True)
def _no_cluster_commands(monkeypatch):
    """No test may run kubectl, ssh, docker or the like — anywhere, ever."""

    def guarded_run(*args, **kwargs):
        _refuse(args[0] if args else kwargs.get("args", []))
        return _REAL_RUN(*args, **kwargs)

    def guarded_popen(*args, **kwargs):
        _refuse(args[0] if args else kwargs.get("args", []))
        return _REAL_POPEN(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
    monkeypatch.setattr(subprocess, "Popen", guarded_popen)


#: The inline scripts in the two pages. Checked here because there is no other
#: gate on them: they are strings served to a browser, so nothing in the Python
#: test suite would otherwise notice a syntax error — and a page that fails to
#: parse is a blank screen with the error only in the browser's console, which
#: is the worst possible place for it to surface.
_PAGES = (
    "src/browser_agent/ui/roster.html",
    "src/browser_agent/ui/index.html",
)


@pytest.mark.parametrize("page", _PAGES)
def test_the_page_script_parses(page: str, tmp_path) -> None:
    import re
    import shutil
    from pathlib import Path

    if not shutil.which("node"):
        pytest.skip("node is not installed; cannot parse the page script")
    blocks = re.findall(r"<script>(.*?)</script>", Path(page).read_text(), re.S)
    assert blocks, f"{page} has no inline script to check"
    for i, js in enumerate(blocks):
        # node --check wants a real path, not a pipe — /dev/stdin resolves to an
        # anonymous pipe here and fails ENOENT regardless of the script.
        probe = tmp_path / f"page{i}.js"
        probe.write_text(js)
        proc = _REAL_RUN(["node", "--check", str(probe)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{page} script {i} does not parse:\n{proc.stderr}"
