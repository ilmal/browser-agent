"""What the bot itself learned about a recipe's site, persisted where it runs.

The operator's rule (2026-09-25): "if we have a recipe, like minesweeper.online,
and it stops working, then update the recipe." A recipe whose canonical site
starts refusing it (IP block, moved page) should not just fail loudly — it
should look for a working replacement, verify it, and *record* it, so the next
run starts from what was learned instead of rediscovering the block.

The recipe library (``recipe_store``) is written by the hub only; a bot pod's
copy is a read-only projection. So the bot's own discovery is recorded here —
under ``DATA_ROOT``, which the bot pod owns. Precedence at every reader:

1. the operator's explicit override (recipe library, ``cfg``),
2. what this file records,
3. the recipe's built-in default.

The operator's hand always beats the bot's own conclusion, and both beat the
literal in the code.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

#: Hosts that are never a real play target. The operator excluded the agent's
#: own loopback explicitly ("not on local host — find a real page"), so a
#: record pointing there is refused on read as well as never written.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})


def _path(settings) -> Path:
    return Path(settings.data_root) / "recipe-sites.json"


def load(settings) -> dict[str, dict]:
    """Every recorded replacement, keyed by recipe name. {} when nothing yet."""
    try:
        data = json.loads(_path(settings).read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def record(settings, recipe: str, *, entry_url: str, note: str) -> dict:
    """Record the replacement this bot verified, replacing any earlier one.

    Best-effort like the rest of the bot's own bookkeeping: a store that
    cannot be written must not turn a found site into a failed run — the
    caller still plays on what it found, and the next run merely re-discovers.
    """
    entry = {"entry_url": entry_url, "note": note, "updated_at": time.time()}
    data = load(settings)
    data[recipe] = entry
    try:
        path = _path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(path)
    except Exception:
        log.exception("site-health record for %s could not be written", recipe)
    return entry


def entry_url(settings, recipe: str) -> str | None:
    """The replacement recorded for ``recipe``, or None.

    A record naming an excluded (loopback) host is treated as absent: the
    operator's "not on local host" outranks anything the bot wrote down, so
    such an entry can never be handed back as a place to play.
    """
    url = str(load(settings).get(recipe, {}).get("entry_url") or "")
    if not url:
        return None
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return None
    return None if host in _LOCAL_HOSTS else url
