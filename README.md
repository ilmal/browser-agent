# browser-agent

A browser agent that drives real sites on your behalf, with a human able to
step in when a site throws a captcha, 2FA or a warning. Repurposed from an
earlier single-site Snapchat bot; the container plumbing (Xvfb + x11vnc +
noVNC + real Chrome) is kept and generalised.

## Architecture

Three layers, escalating only as far as needed:

| Layer | Handles | Why |
|---|---|---|
| **Playwright** | the routine path: check logged in, open composer, type, submit | deterministic, fast |
| **browser-use** (LLM) | selector broke, unexpected modal, "find the right button" | no per-site code; reads the DOM |
| **You, over noVNC** | captcha, 2FA, account warnings | real Chrome, your hands |

The LLM layer points at a self-hosted OpenAI-compatible endpoint, so agent
steps cost no per-call API fees.

**Escalation rule: on a captcha or verification challenge, stop — never retry
blindly.** Retrying is what gets accounts flagged. Alert, wait for the human,
then resume; the persistent profile means the next run just works.

## Profiles

Each identity gets its own container and its own persistent Chrome profile
directory (`profiles/<name>/`), bind-mounted from the host. Profiles are never
committed and never shared between identities.

> `profiles/` and `driver_user/` hold **live session cookies**. They are
> gitignored and the pre-commit hook blocks them. This repo is public.

## Taking over a run

Each profile exposes noVNC. Open its URL (Tailscale only — never bind these
ports publicly), solve whatever the site is asking, close the tab. The
profile persists, so the scheduler's next run continues normally.

## Security

The repo is public. Before pushing:

```bash
git config core.hooksPath hooks   # enable the secret/profile guard
```

`hooks/pre-commit` blocks browser profiles, `.env`/key files, and
secret-shaped values from being staged. It is not a substitute for review —
read `git diff --cached` before you push.

## Quick start

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m playwright install chromium

export AGENT_PROFILE=dev PROFILES_ROOT=$PWD/.local/profiles DATA_ROOT=$PWD/.local/data
export MANAGE_DISPLAY=false HEADLESS=false LLM_ENABLED=false CONTROL_TOKEN=devtoken
export PYTHONPATH=src
.venv/bin/python -m uvicorn browser_agent.api:app --port 8899
```

Open <http://localhost:8899/>. Full instructions — local run, the k8s deploy,
adding profiles, scheduling, and what each failure state means — are in
[`docs/OPERATING.md`](docs/OPERATING.md).

## Layout

```
src/browser_agent/
  api.py          control-plane API + serves the admin UI
  tasks.py        task model, recipe registry, run loop (the escalation gate)
  escalation.py   challenge detection + EscalationRequired
  browser.py      persistent Playwright context (headed, so noVNC can take over)
  agent.py        browser-use fallback, same profile and display
  llm.py          OpenAI-compatible client (points at llm-service)
  scheduler.py    SQLite-backed cron store
  notify.py       escalation alerts
  ui/index.html   single-page admin UI
  recipes/        one thin file per site: x, facebook, linkedin
k8s/              one Deployment + PVC + Service per profile
hooks/pre-commit  secret + profile guard (enable via core.hooksPath)
Dockerfile        Chrome + Xvfb + x11vnc + noVNC, tini as PID 1
src/main.py       the original Snapchat bot, kept for reference
```

`src/main.py` / `src/modules/snapchat/` are the original single-site
implementation, kept for reference while recipes are ported.

## Design notes

- **A profile is single-writer.** Pods use `replicas: 1` with
  `strategy: Recreate`, and each profile has its own volume. Two browsers in
  one profile corrupts the session, and a site would never see that from a
  human.
- **The escalation gate is structural.** `EscalationRequired` unwinds the
  task, so a recipe cannot step over a captcha. `tests/test_tasks.py` asserts
  that a blocked task does not invoke the agent and is not retried.
- **The browser is headed on purpose.** A headless context could not be taken
  over by a person, which is the entire point of the noVNC layer.
