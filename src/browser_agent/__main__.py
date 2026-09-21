"""Entrypoint: start the control plane (and, in the container, Xvfb + noVNC)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time

from .config import load_settings


def _start_display(settings) -> list[subprocess.Popen]:
    """Start Xvfb + x11vnc + websockify.

    Only in the container: the browser is deliberately headed so that a human
    can take over the same session over noVNC. Outside the container we assume
    an existing display (or a desktop session) and start nothing.
    """
    if os.environ.get("MANAGE_DISPLAY", "true").lower() not in {"1", "true", "yes"}:
        return []

    if not shutil.which("Xvfb"):
        logging.info("Xvfb not found; expecting an existing DISPLAY")
        return []

    display = os.environ.get("DISPLAY", ":99")
    procs: list[subprocess.Popen] = []

    for lock in (f"/tmp/.X{display.lstrip(':')}-lock", f"/tmp/.X11-unix/X{display.lstrip(':')}"):
        if os.path.exists(lock):
            try:
                os.remove(lock)
            except OSError:
                pass

    xvfb = subprocess.Popen(
        [
            "Xvfb",
            display,
            "-screen",
            "0",
            f"{settings.screen_width}x{settings.screen_height}x{settings.screen_depth}",
            "-ac",
            "-nolisten",
            "tcp",
            "-dpi",
            "96",
            "+extension",
            "RANDR",
        ]
    )
    procs.append(xvfb)
    time.sleep(2)

    if shutil.which("x11vnc"):
        procs.append(
            subprocess.Popen(
                [
                    "x11vnc",
                    "-display",
                    display,
                    "-forever",
                    "-shared",
                    "-nopw",
                    "-rfbport",
                    "5900",
                    "-quiet",
                ]
            )
        )
        time.sleep(1)

    if shutil.which("websockify"):
        procs.append(
            subprocess.Popen(
                [
                    "websockify",
                    "--web",
                    "/usr/share/novnc",
                    str(settings.novnc_port),
                    "localhost:5900",
                ]
            )
        )
        time.sleep(1)

    logging.info("display up on %s, noVNC on :%s", display, settings.novnc_port)
    return procs


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = load_settings()

    # Import after display setup so Playwright sees a display if it needs one.
    procs = _start_display(settings)
    try:
        import uvicorn

        uvicorn.run(
            "browser_agent.api:app",
            host="0.0.0.0",
            port=settings.api_port,
            log_level="info",
        )
    finally:
        for p in procs:
            p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
