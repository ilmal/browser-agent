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

WORKDIR /app

# uv for a reproducible, cached dependency install.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

COPY pyproject.toml README.md ./
COPY src ./src

# `agent` extra adds browser-use. Kept in the image because the fallback is the
# whole point of the design; it is inert unless a recipe fails. `laya` adds the
# 322M classifier behind the plan.task gate (pulls torch — CPU is fine).
RUN uv pip install --system --no-cache ".[agent,laya]"

# Bake the Laya weights: a first-boot download would make every cold pod start
# depend on huggingface.co, and a blocked download would silently leave the
# gate permanently off. Readable by the runtime user, not writable.
ENV HF_HOME=/models/hf
RUN python -c "from laya import Agent; Agent()" \
    && chmod -R a+rX /models

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
