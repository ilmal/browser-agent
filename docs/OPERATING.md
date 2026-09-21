# Operating browser-agent

## The one rule

**A challenge stops the run.** Captcha, 2FA prompt, "unusual activity",
rate limit — the task ends `blocked` and a human clears it. It is never
retried automatically. Blind retry is what gets accounts flagged, so the
runner treats a blocker as terminal.

## Run it locally (no k8s)

The fastest loop for developing a recipe. Uses your own Chrome display, not
the container.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m playwright install chromium

export AGENT_PROFILE=dev PROFILES_ROOT=$PWD/.local/profiles DATA_ROOT=$PWD/.local/data
export MANAGE_DISPLAY=false          # use the desktop session, not Xvfb
export HEADLESS=false                # you want to watch it
export LLM_ENABLED=false             # no fallback while iterating
export CONTROL_TOKEN=devtoken
export PYTHONPATH=src
.venv/bin/python -m uvicorn browser_agent.api:app --port 8899
```

Open <http://localhost:8899/>, paste `devtoken` when prompted, then:

1. **Open login page & take over** — signs in once. The profile persists.
2. **Queue task** — pick a recipe, write text, go.

Run the tests with `.venv/bin/python -m pytest tests/ -q`.

### Local container (docker compose)

```bash
CONTROL_TOKEN=devtoken docker compose up -d --build
# admin UI  http://localhost:8901/
# noVNC     http://localhost:6081/vnc.html
```

`docker-compose.yaml` runs the container as **your** uid (`user: "${UID}:${GID}"`),
so the bind-mounted `profiles/` and `data/` directories are writable without a
chown. Without that, the container's own uid 10001 cannot open the schedule
database and it exits with `unable to open database file`. Kubernetes handles
the same problem with `fsGroup`, so nothing needs changing there.

## Run it on k8s (cn1)

### Build and push the image

```bash
cd ~/programing/browser-agent
SHA=$(git rev-parse --short HEAD)
docker build -t browser-agent:$SHA .
docker save browser-agent:$SHA | ssh cn1 'docker load'
ssh cn1 "docker tag browser-agent:$SHA localhost:5000/browser-agent:$SHA && \
         docker push localhost:5000/browser-agent:$SHA"
```

Then set that tag in `k8s/browser-agent.yaml` and:

```bash
ssh cn1 'kubectl apply -f /home/nils/programing/browser-agent/k8s/browser-agent.yaml'
ssh cn1 'kubectl rollout status deployment/profile-x -n browser-agent --timeout=180s'
```

### Adding a profile

```bash
./scripts/add-profile.sh linkedin > k8s/profiles/linkedin.yaml
ssh cn1 'kubectl apply -f -' < k8s/profiles/linkedin.yaml
```

One profile = one Deployment + PVC + Service. Each profile needs its own PVC —
sharing one between pods corrupts the session. Pods run with
`strategy: Recreate` and `replicas: 1` for the same reason: a Chrome profile
is single-writer.

### Several profiles at once (locally)

```bash
CONTROL_TOKEN=devtoken ./scripts/profiles-up.sh x linkedin facebook
```

Writes a compose override with one service per profile and starts them.
Admin UIs land on `:8900, :8901, :8902`, noVNC on `:6080, :6081, :6082`.

### Reaching the UI and noVNC

Both are Tailscale-only. From anywhere on the tailnet:

```bash
ssh -N -L 8899:localhost:8899 -L 6081:localhost:6080 nils@cn1
# or port-forward through kubectl
ssh cn1 'kubectl port-forward -n browser-agent deploy/profile-x 8899:8000 6081:6080'
```

Then `http://localhost:8899/` for the admin UI and `http://localhost:6081/vnc.html`
for the browser itself. **Never expose these ports publicly** — a logged-in
browser session behind a public port is an account takeover.

`CONTROL_TOKEN` is required in the cluster (it is set from the Secret). If it
is empty the API is unauthenticated, which is only acceptable on a local
dev box.

### First run for a new profile

A fresh profile has no session, so the first thing it needs is a human:

1. Open the profile's noVNC.
2. In the admin UI, **Open login page & take over** with that site's login URL.
3. Sign in — including 2FA — in the noVNC window.
4. The profile persists on its PVC. Subsequent runs reuse the session.

## Scheduling

Schedules are 5-field cron, stored in SQLite on the pod, evaluated in-process.
Add them in the admin UI, or:

```bash
curl -X POST http://localhost:8899/api/schedules \
  -H "Authorization: Bearer $CONTROL_TOKEN" -H 'Content-Type: application/json' \
  -d '{"id":"x-weekday-morning","recipe":"x.post","cron":"0 9 * * 1-5","payload":{"text":"…"}}'
```

Keep cadence human. Several profiles firing on the same minute is a pattern a
site can see; stagger them.

## Reading the failure states

| Status | Meaning | What to do |
|---|---|---|
| `done` | Recipe worked, or the agent recovered it (`used_agent: true`) | nothing |
| `blocked` | A human is needed. The agent was **not** invoked | open noVNC, clear it, then **Retry** |
| `failed` | Recipe broke and the fallback also failed, or no fallback | check `detail`; the recipe's selectors probably moved |

`blocked` is the only state that means "a person must act". A `failed` that
mentions selectors is a code fix, not an account problem.

## When a recipe breaks

Recipes are thin and will break when a site redesigns. The order of repair:

1. The agent fallback usually still completes the task, so a broken recipe is
   not an outage — check whether tasks are landing `done` with
   `used_agent: true`.
2. To fix the recipe: run locally with `HEADLESS=false`, watch the browser,
   and update the selector list. Prefer `data-testid` and `aria-label`
   selectors over class names.
3. If a site needs a genuinely different flow, add a new recipe rather than
   growing the existing one.

## Security posture

- Repo is **public**. `hooks/pre-commit` blocks profiles, env files and
  secret-shaped values. Enable it in every clone: `git config core.hooksPath hooks`.
- `profiles/` holds live session cookies. Never commit, never copy between
  machines in a way that ends up in a repo.
- Egress goes through the residential `office-proxy`. Do not remove the proxy
  env vars — a datacenter IP is the single biggest flag.
- The agent prompt forbids creating accounts, changing credentials, accepting
  terms, and DMing people. It is a guardrail, not a sandbox; keep the blast
  radius small by not giving a profile more access than it needs.

## Platform notes

- **X** — most tolerant. Posting from a logged-in browser session is what many
  third-party tools do.
- **Facebook** — Page posting only. Personal-timeline posting is deliberately
  not implemented.
- **LinkedIn** — the most aggressive about automation. Keep it in the watched
  lane: run it while you can pay attention, and do not schedule it unattended
  until it has a long clean history.
