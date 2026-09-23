"""The roster: which bots exist, and what each one is for.

A bot is one profile — one identity, one browser, one login — reached at its
own URL. The pods are the source of truth for whether a bot is *up*; this file
is the source of truth for what it is *called* and what it is *for*, because
neither of those is discoverable from a running pod (a pod knows its profile
name and nothing else).

Deliberately a plain JSON file, not a database: the roster is a handful of
lines, it is read on every hub request, and a file is trivially inspectable
and repairable when something is wrong. It holds no secrets — a bot's URL is a
path under the roster's own origin, and the tokens live in the Secret.

Written atomically (temp file + rename) because the hub has one worker but a
reader can still arrive mid-write, and a truncated roster is worse than a
stale one.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Bot:
    """One teammate in the roster."""

    profile: str            # identifier; also the k8s Service name and URL slug
    name: str = ""          # display name, defaults to the profile
    job: str = ""           # one line: what this bot is for
    # Where it is reached, as a path on the roster's origin. A roster proxies
    # every bot from one domain, so this is "/b/<profile>/" and not a host.
    url: str = ""
    # Free text the operator can leave for themselves — which login this bot
    # holds, which account, what to do when it hits a wall.
    notes: str = ""
    # Who this identity is, in the operator's words — the persona a farm task
    # is framed for ("You are Kai. Background: …"). Injected into the task text
    # when a farm assigns work to this bot, so each environment runs the same
    # instruction as its own person.
    background: str = ""
    # Account notes for this identity: which site, which login, how the 2FA
    # is solved. Lives on the hub's volume, never in the repo.
    login_notes: str = ""
    created_at: float = 0.0
    # Whether the operator wants it on the roster's active list. Hidden is not
    # stopped: a hidden bot still runs its schedules, it is just out of the way
    # (the same distinction Grok Bot draws between "hide" and "pause").
    hidden: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["display_name"] = self.name or self.profile
        d["url"] = self.url or f"/b/{self.profile}/"
        return d


@dataclass
class Registry:
    bots: list[Bot] = field(default_factory=list)

    def get(self, profile: str) -> Bot | None:
        return next((b for b in self.bots if b.profile == profile), None)


def _clean(value: str, limit: int = 200) -> str:
    """Trim and bound operator text. It is rendered in HTML, but the template
    escapes at render time; this is about keeping the file sane."""
    return " ".join(str(value or "").split())[:limit]


def valid_profile(name: str) -> bool:
    """Same rule as scripts/add-profile.sh, so a name the roster accepts is a
    name the generator and k8s will accept."""
    import re

    return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", str(name or "")))


def load(path: Path) -> Registry:
    if not path.exists():
        return Registry()
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        # A corrupt roster must not take the hub down: an empty one is
        # recoverable by hand, a 500 on every page is not.
        return Registry()
    bots = []
    for item in raw.get("bots", []):
        if not isinstance(item, dict) or not valid_profile(item.get("profile", "")):
            continue
        bots.append(Bot(
            profile=item["profile"],
            name=_clean(item.get("name", "")),
            job=_clean(item.get("job", "")),
            url=_clean(item.get("url", "")),
            notes=_clean(item.get("notes", ""), 500),
            background=_clean(item.get("background", ""), 2000),
            login_notes=_clean(item.get("login_notes", ""), 2000),
            created_at=float(item.get("created_at") or 0.0),
            hidden=bool(item.get("hidden")),
        ))
    return Registry(bots)


def save(path: Path, reg: Registry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"bots": [b.to_dict() for b in reg.bots]}, indent=2) + "\n")
    os.replace(tmp, path)   # atomic: a reader sees the old file or the new one


def remove(reg: Registry, profile: str) -> Bot | None:
    """Drop a bot from the roster, returning the entry that was removed.

    The roster is only the *naming* record — a bot's pod, service and storage
    are the cluster's, and the caller deletes those. Returns None when there
    was nothing to remove, so the caller can answer 404 rather than reporting
    a deletion that never happened.
    """
    bot = reg.get(profile)
    if bot is None:
        return None
    reg.bots = [b for b in reg.bots if b.profile != profile]
    return bot


def upsert(reg: Registry, profile: str, **fields) -> Bot:
    """Add a bot or update the fields given. Never clears a field by omission,
    so recording a new job cannot wipe the notes."""
    bot = reg.get(profile)
    if bot is None:
        bot = Bot(profile=profile, created_at=time.time())
        reg.bots.append(bot)
    for key in ("name", "job", "url", "notes", "background", "login_notes"):
        if key in fields and fields[key] is not None:
            if key in ("background", "login_notes"):
                limit = 2000
            elif key == "notes":
                limit = 500
            else:
                limit = 200
            setattr(bot, key, _clean(fields[key], limit))
    if "hidden" in fields and fields["hidden"] is not None:
        bot.hidden = bool(fields["hidden"])
    reg.bots.sort(key=lambda b: b.created_at)
    return bot
