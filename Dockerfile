# browser-agent runtime: real Chrome on a virtual display, with noVNC for
# human takeover. The browser is headed on purpose — a headless context could
# not be taken over by a person.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:99 \
    PROFILES_ROOT=/profiles \
    DATA_ROOT=/data \
    NOVNC_PORT=6080 \
    MANAGE_DISPLAY=true

# Display stack + runtime libs Chrome needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        x11vnc \
        novnc \
        websockify \
        procps \
        curl \
        ca-certificates \
        fonts-liberation \
        fonts-noto-color-emoji \
        libasound2 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libatspi2.0-0 \
        libcairo2 \
        libcups2 \
        libdbus-1-3 \
        libdrm2 \
        libgbm1 \
        libglib2.0-0 \
        libgtk-3-0 \
        libnspr4 \
        libnss3 \
        libpango-1.0-0 \
        libx11-6 \
        libxcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxkbcommon0 \
        libxrandr2 \
        tini \
    && rm -rf /var/lib/apt/lists/*

# noVNC's app/ui.js fetches ./package.json on startup and fills the version
# badge from it. Debian's novnc package ships the JS but not that file, so the
# fetch 404s and the badge is hidden — the one console error the live view
# produces. Written here rather than hardcoded so it tracks the apt version.
RUN NOVNC_VERSION="$(dpkg-query -W -f='${Version}' novnc | cut -d: -f2 | cut -d- -f1)" \
    && printf '{"name":"novnc","version":"%s"}\n' "$NOVNC_VERSION" > /usr/share/novnc/package.json \
    && chmod a+r /usr/share/novnc/package.json

# kubectl, for the roster's "new bot": the hub runs the same add-profile.sh a
# human would and applies the result. Without it in the image the hub can only
# report can_create:false, so the multi-bot feature the roster exists for is
# unreachable in the one place it matters. Static Go binary, so this adds no
# runtime deps. Version matches the cn1 server (v1.32.13).
COPY --from=registry.k8s.io/kubectl:v1.32.13 /bin/kubectl /usr/local/bin/kubectl

WORKDIR /app

# uv for a reproducible, locked dependency install. The lockfile is the pin:
# `uv sync --frozen` installs exactly the versions (with hashes) recorded in
# uv.lock for this commit, so two builds of the same commit ship the same
# dependencies and a yanked or maliciously retagged release cannot be picked
# up at build time (SEC-BA-009). The uv version here must read the lockfile
# format the repo was locked with.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

# Dependency layer first, project source second: a source change does not
# bust the (large) locked dependency install. The venv lives on a fixed path
# rather than uv's default relative one so runtime and build agree on it.
COPY pyproject.toml README.md uv.lock ./
RUN uv venv --python /usr/local/bin/python3.12 /app/.venv \
    && uv sync --frozen --no-dev --no-cache --extra agent --no-install-project

COPY src ./src
# The roster's "new bot" runs this rather than reimplementing it, so there is
# exactly one definition of what a bot is. It is a generator: it writes YAML to
# stdout and touches nothing, which is why the hub can afford to run it.
COPY scripts/add-profile.sh ./scripts/add-profile.sh

# `agent` extra adds browser-use. Kept in the image because the fallback is the
# whole point of the design; it is inert unless a recipe fails. The Laya gate
# is NOT in the image: in the pod it runs over HTTP against llm-service's
# /v1/decide proxy (LAYA_DECIDE_URL), so no torch and no weights are baked —
# the in-process pip backend is a laptop-development fallback only.
# Second sync pass installs the project itself into the locked environment;
# the dependencies came from uv.lock in the layer above.
RUN uv sync --frozen --no-dev --no-cache --extra agent

# Runtime prefers the locked environment: CMD's `python`, the `playwright`
# CLI below and any console script resolve to the venv first, then fall
# through to the system for the non-Python display stack (Xvnc, x11vnc,
# websockify) which is not in the venv at all.
ENV PATH="/app/.venv/bin:$PATH"

# Browsers live outside any user's home: Playwright's default is $HOME/.cache,
# which differs between build (root) and runtime (agent) and would silently
# disappear. One fixed path, readable by everyone.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN python -m playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright \
    && rm -rf /var/lib/apt/lists/*

# Non-root, with a home so Chrome has somewhere to write.
RUN useradd -m -u 10001 agent && mkdir -p /profiles /data && chown -R agent:agent /profiles /data
USER agent
ENV HOME=/home/agent

EXPOSE 8000 6080

# tini reaps the browser's helper processes; without it Chrome leaves zombies
# when it is PID 1.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "browser_agent"]
