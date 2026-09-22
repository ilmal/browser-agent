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

## plan.task — freeform task, planned and executed

`plan.task` takes `{"task": "<freeform instruction>"}`. A fast text model
(`PLANNER_MODEL`, default `deepseek-v4.1-flash` via llm-service) plans **once**;
the plan runs deterministically:

1. Navigate / click / type / extract / wait, in order, from `entry_url`.
2. A click or type without a CSS selector is resolved by the Laya classifier,
   which picks among the page's visible elements (bounded by
   `LAYA_MAX_CANDIDATES`); the executor binds by index.
3. A step proves itself with its `done_when` predicate; click/type steps
   without one are confirmed by Laya yes/no above `LAYA_MIN_CONFIDENCE`.
4. A step that fails — or a planner that cannot be reached — hands the
   **original task text** to the browser-use agent fallback, which starts from
   the plan's own entry URL.
5. A captcha or account challenge anywhere BLOCKs for a human, as everywhere
   else. The planner is forbidden from planning login, payment or account
   changes; a plan that fails validation FAILs visibly (`plan rejected: …`)
   instead of reaching the agent.

The gate summary in a done task's result shows how much Laya did
(`picks`/`confirms`/`inconclusive`). In k8s the picker is flag-off
(`LAYA_PICK_ENABLED=false`). Re-benched in-pod 2026-09-22 against the prod
`/v1/decide` engine (`scripts/laya_pick_bench.py`): 14/18 correct (78%), and
**100% precision** on every pick that cleared the 0.75 confidence floor — but
only 22% cleared it, short of the bench's 80% coverage bar, so the picker
abstains on most decisions. The reason to keep it off is coverage, not
accuracy: turning it on would not misclick, it would just rarely answer.
Confirmation is on.

## Reading the failure states

| Status | Meaning | What to do |
|---|---|---|
| `done` | Recipe worked, or the agent recovered it (`used_agent: true`) | nothing |
| `blocked` | A human is needed. The agent was **not** invoked | open noVNC, clear it, then **Talk to it** |
| `failed` | Recipe broke and the fallback also failed, or no fallback | check `detail`; the recipe's selectors probably moved |

`blocked` is the only state that means "a person must act". A `failed` that
mentions selectors is a code fix, not an account problem.

### Retrying is a conversation, not a button

A failed or blocked row offers **Talk to it**, which opens the thread: every
attempt at that one piece of work, with what was said between them. Type what to
do differently and the next attempt is queued with it — and told what the
earlier attempts tried, so it does not walk back into the same wall.

Three things that are easy to assume wrongly:

* **A message to a running task steers it; to a finished one it starts the next
  attempt.** One box, and the page says which one you are doing, because needing
  to know the difference before you can speak is the whole problem.
* **The next attempt is briefed, not just re-queued.** Its payload carries a
  `history` block: each earlier attempt's status, its `detail`, and the last few
  feed lines. Both the planner and the agent template read it. An "iterate" that
  did not carry the prior failure forward would just be a second identical roll.
* **A finished attempt's feed is readable.** The runner snapshots the activity
  log when an attempt ends (`GET /api/activity?task_id=`), because the live log
  is reset per task — without the snapshot the thread could show only the
  running attempt and the point of the thread would be lost.

Everything an attempt did is preserved, so a thread is the record of what was
tried and why it stopped, not just a list of outcomes.

## When a recipe breaks

Recipes are thin and will break when a site redesigns. The order of repair:

1. The agent fallback usually still completes the task, so a broken recipe is
   not an outage — check whether tasks are landing `done` with
   `used_agent: true`.
2. **A moved selector is no longer a deploy.** Open the roster, find the recipe,
   press **Edit**, and correct the selector. The library is shared, so the fix
   applies to every bot; the next run anywhere picks it up (the pod's mount
   syncs within about a minute). Prefer `data-testid` and `aria-label`
   selectors over class names.
3. To watch it against the real page while editing, run locally with
   `HEADLESS=false`.
4. If a site needs a genuinely different flow, compose a new recipe from steps
   on the roster rather than growing the existing one.

### The recipe library

Recipes live in one shared directory — a ConfigMap in the cluster, mounted
read-only into every bot — and the roster is its only editor:

* `_overrides.json` — per-recipe config (selectors, URLs, pacing, the planner
  prompt). An override replaces one value; everything else keeps its built-in
  default. **Revert** clears it.
* `<name>.json` — a composed recipe: a saved step list, validated by the same
  schema the planner emits and run by the same executor.

Three things worth knowing before editing:

* **It is library-wide.** One edit changes that recipe for every bot. The
  editor says so; there is no per-bot override.
* **A running task keeps the recipe it started with.** The steps are already
  materialised for that attempt, so an edit lands on the *next* run.
* **Publishing is a deployment fact.** The hub writes the ConfigMap only when
  `RECIPES_PUBLISH` is set (it is, in the cluster). A hub without it stores the
  edit and reports that no pod can see it yet, rather than implying it is live.

The ConfigMap is deliberately **not** declared in `k8s/browser-agent.yaml`: an
`apply` of that file would otherwise reset the operator's recipes to whatever
the manifest says. It is mounted `optional: true`, so a cluster that never
created it behaves exactly as it did before recipes were editable.

### Why the agent attaches over CDP

The browser launches with `--remote-debugging-port=0` and writes the chosen
port into `DevToolsActivePort` inside the profile. The agent reads that file
and attaches to the **running** browser.

Do not "simplify" this back to giving the agent its own
`Browser(user_data_dir=...)`. Chrome allows a second context on the same
profile but starts it **logged out**, so the agent would silently drive a
different browser than the one over noVNC — the human would watch a login page
while the agent worked an empty session.


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
