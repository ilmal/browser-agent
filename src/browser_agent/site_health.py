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
    Obstacle memory recorded for this recipe survives the write: a new
    discovery does not un-block the hosts the bot was forced to route around.
    """
    entry = {"entry_url": entry_url, "note": note, "updated_at": time.time()}
    data = load(settings)
    prev = data.get(recipe)
    if isinstance(prev, dict) and isinstance(prev.get("blocked"), dict):
        entry["blocked"] = prev["blocked"]
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


def _normalize_host(host: str) -> str:
    host = (host or "").lower().strip()
    return host[4:] if host.startswith("www.") else host


def host_of(url: str) -> str:
    """The www-stripped hostname of ``url``, or "" when it has none."""
    try:
        return _normalize_host(urlsplit(url).hostname or "")
    except ValueError:
        return ""


def record_blocked(
    settings, recipe: str, *, host: str, reason: str,
    kind: str = "blocked", url: str = "",
) -> dict | None:
    """Note an obstacle so the next run routes around it instead of into it.

    The operator's rule (2026-09-26): every block is written down, permanently.
    Records are keyed by host and accumulate — each sighting bumps ``count``
    and refreshes ``last_seen`` while ``first_seen`` and the most useful
    ``reason`` are kept. The agent's own loopback is never recorded: the
    operator excluded it explicitly, and it is not a site that blocked us.
    Best-effort like every write here.
    """
    host = _normalize_host(host)
    if not host or host in _LOCAL_HOSTS:
        return None
    data = load(settings)
    entry = data.get(recipe) if isinstance(data.get(recipe), dict) else {}
    blocked = entry.get("blocked") if isinstance(entry, dict) else None
    blocked = dict(blocked) if isinstance(blocked, dict) else {}
    now = time.time()
    prev = blocked.get(host) if isinstance(blocked.get(host), dict) else {}
    record_ = {
        "kind": kind,
        "reason": str(reason or prev.get("reason") or ""),
        "url": str(url or prev.get("url") or ""),
        "first_seen": prev.get("first_seen") or now,
        "last_seen": now,
        "count": int(prev.get("count") or 0) + 1,
    }
    blocked[host] = record_
    base = dict(entry) if isinstance(entry, dict) else {}
    base["blocked"] = blocked
    data[recipe] = base
    try:
        path = _path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(path)
    except Exception:
        log.exception("obstacle record for %s/%s could not be written", recipe, host)
    return record_


def blocked_hosts(settings, recipe: str) -> dict[str, dict]:
    """Every obstacle this bot wrote down for ``recipe``: host -> record."""
    entry = load(settings).get(recipe)
    blocked = entry.get("blocked") if isinstance(entry, dict) else None
    return dict(blocked) if isinstance(blocked, dict) else {}


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
