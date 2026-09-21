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

# uv for a reproducible, cached dependency install.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

COPY pyproject.toml README.md ./
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
RUN uv pip install --system --no-cache ".[agent]"

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
