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

## Layout

```
hooks/pre-commit        secret + profile guard (enable via core.hooksPath)
Dockerfile-vnc          Chrome + Xvfb + x11vnc + noVNC
docker-compose.yaml     one service per profile
src/main.py             entrypoint
src/modules/            driver + element helpers
src/modules/snapchat/   the original single-site implementation
```

`src/modules/snapchat/` is the original bot, kept for reference while the
per-site recipes are ported to the Playwright + agent structure above.
