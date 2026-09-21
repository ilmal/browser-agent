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
# whole point of the design; it is inert unless a recipe fails.
RUN uv pip install --system --no-cache ".[agent]" \
    && python -m playwright install --with-deps chromium \
    && rm -rf /root/.cache

# Non-root, with a home so Chrome has somewhere to write.
RUN useradd -m -u 10001 agent && mkdir -p /profiles /data && chown -R agent:agent /profiles /data
USER agent
ENV HOME=/home/agent

EXPOSE 8000 6080

# tini reaps the browser's helper processes; without it Chrome leaves zombies
# when it is PID 1.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "browser_agent"]
