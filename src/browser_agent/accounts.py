"""Several identities per bot: the accounts one pod can be logged in as.

A bot was one Chrome profile — one identity, one login. That is still the
*concurrency* model, and it has to be: Chrome's user-data-dir is single-writer,
and the whole noVNC takeover story rests on there being exactly one window a
human can be shown. What this module adds is that a bot may hold **several**
profiles and run one of them at a time.

Why not a pod per account, which is what the roster already supports? Because a
bot *is* the agent — "each agent has and can have multiple profiles" is the ask,
and N pods per agent costs N PVCs, N lots of 3 GiB of memory, and N roster lines
for what is one job. The account that is not running is idle storage, and the
thing that makes a second Chrome impossible (a single X display, a single
`x11vnc`) is already the thing that makes one-at-a-time the natural unit.

So the layout is a directory of user-data-dirs inside the bot's own PVC:

    /profiles/<bot>/accounts/<account>/     <- a Chrome user-data-dir

and one of them is ``active``. Everything that used to read
``settings.profile_dir`` reads the active account's directory instead, which is
why the single-account case behaves exactly as before: it is the same code with
a one-element list.

**The migration is the risky part and it is deliberately resumable.** Every bot
that exists today has Chrome's files written *directly* into
``/profiles/<bot>/`` (``Default/``, ``Local State``, ``SingletonLock``, …), and
that directory cannot also hold an ``accounts/`` subdirectory without Chrome
confusing one for the other. On the first boot of this build the legacy contents
are moved into ``accounts/default/``. If that move is interrupted, the next boot
finds a partially-emptied root and simply moves what is left — the loop skips
entries that are already gone, so it converges rather than needing a marker to
know where it was.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: The account a bot has before the operator adds any, and the name the legacy
#: profile directory is migrated to. Not special-cased anywhere in the running
#: code — it is only the default value of ``active``, so a bot the operator has
#: moved on from is not stuck with it.
DEFAULT_ACCOUNT = "default"

#: The directory holding every account's Chrome user-data-dir.
ACCOUNTS_DIR = "accounts"

#: The metadata file, at the profile *root* — beside ``accounts/``, never inside
#: a Chrome directory. Chrome opens the whole user-data-dir and would find a
#: stray JSON sitting in it.
ACCOUNTS_FILE = "accounts.json"


@dataclass
class Account:
    """One identity: a Chrome user-data-dir plus what the operator calls it."""

    name: str               # identifier; also the directory name
    label: str = ""         # what the roster shows; falls back to the name
    # Which login this account holds. Free text on purpose: it is the operator's
    # note to themselves ("nils@u1.se", "the work X account"), and nothing in
    # the code branches on it.
    email: str = ""
    notes: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["display_label"] = self.label or self.name
        return d


def valid_account(name: str) -> bool:
    """Same shape as a profile name, and for the same reason: this becomes a
    directory name, so anything that is not a plain slug is a path traversal
    waiting to happen (``../`` most obviously)."""
    return bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", str(name or "")))


def account_dir(profile_root: Path, account: str) -> Path:
    """The Chrome user-data-dir for one account."""
    return profile_root / ACCOUNTS_DIR / account


def ensure_layout(profile_root: Path) -> list[str]:
    """Bring ``profile_root`` up to the accounts layout, returning what changed.

    Every bot that predates accounts has Chrome's files written directly into
    ``/profiles/<bot>/`` (``Default/``, ``Local State``, ``SingletonLock``, …).
    That directory cannot also hold ``accounts/`` without Chrome confusing one
    for the other, so the legacy contents move into ``accounts/default/``.

    "What is left to move" is computed as *anything at the root that is not
    ours*, on every call — deliberately not "the ``accounts/`` directory is
    absent". Those differ exactly when a previous run was interrupted: the move
    creates ``accounts/default/`` early, so an aborted run has ``accounts/``
    present *and* orphaned Chrome files still at the root, and an
    "is it migrated?" test would declare victory and leave the bot half-moved.
    A half-moved profile reads to Chrome as an empty one — logged out, with the
    login sitting in an account directory nothing points at. So the test is
    idempotent instead: nothing else is ever legitimately at the root, and a
    second pass over an empty list is free.

    An entry that cannot be moved does not abort the pass: leaving the root
    populated is what breaks Chrome, so moving the rest is strictly better, and
    the next boot retries the one that failed.
    """
    profile_root.mkdir(parents=True, exist_ok=True)
    # ACCOUNTS_FILE is what our own store writes here, so it is ours and must
    # not be swept into a Chrome directory, where a reader would find a stray
    # JSON and the store would find no file at all.
    reserved = {ACCOUNTS_DIR, ACCOUNTS_FILE}
    legacy = [p for p in profile_root.iterdir() if p.name not in reserved]

    if not legacy:
        (profile_root / ACCOUNTS_DIR).mkdir(exist_ok=True)
        return []

    dest = account_dir(profile_root, DEFAULT_ACCOUNT)
    dest.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for entry in legacy:
        try:
            # shutil.move, not os.replace: these can be files or directories and
            # may span a mount in a hand-run setup. Within one PVC it is a
            # rename, so it is cheap.
            shutil.move(str(entry), str(dest / entry.name))
            moved.append(entry.name)
        except OSError:
            log.warning("could not move %s into accounts/%s", entry, DEFAULT_ACCOUNT, exc_info=True)
    if moved:
        log.info(
            "migrated profile %s to the accounts layout: moved %d entries into accounts/%s",
            profile_root.name,
            len(moved),
            DEFAULT_ACCOUNT,
        )
    return moved


def signed_in(profile_dir: Path) -> bool:
    """Whether a human has signed in *this* Chrome user-data-dir.

    Chrome writes the profile directory on its very first launch, so "the
    directory has files in it" is true of a bot that has never been touched —
    which is exactly the state the operator needs to be warned about. The honest
    test is whether a login left cookies behind: Chrome creates the empty
    Cookies database on first launch and fills it only when a site sets one.
    Measured on this deployment — a never-used profile has 0, a signed-in one
    has 11. Read over the file's own SQLite, read-only and with a short timeout,
    because the browser holds it open while running.
    """
    import sqlite3

    path = profile_dir
    if not path.is_dir():
        return False
    cookies = path / "Default" / "Cookies"
    if not cookies.is_file():
        # Pre-Chromium-96 layouts kept it under Default/Network. Checked rather
        # than assumed so a profile from an older image still reports right.
        cookies = path / "Default" / "Network" / "Cookies"
    if not cookies.is_file():
        return False
    try:
        # A locked database is not an error worth raising: it means the browser
        # is running, and the count is a roster nicety, not a control path. Fall
        # back to the directory check so a running-but-unknown profile stays
        # visible rather than flipping to "not signed in".
        con = sqlite3.connect(f"file:{cookies}?mode=ro", uri=True, timeout=1.0)
        try:
            n = con.execute("select count(*) from cookies").fetchone()[0]
        finally:
            con.close()
        return bool(n)
    except sqlite3.Error:
        try:
            return any(path.iterdir())
        except OSError:
            return False


def remove_account(profile_root: Path, name: str) -> bool:
    """Delete one account's Chrome directory. Returns whether anything went.

    Deleting this *is* logging out of everything that account holds, so it is
    never done implicitly — the caller asks for it by name, and the API refuses
    to remove the last account because a bot with no account cannot run a task
    at all.
    """
    path = account_dir(profile_root, name)
    if not path.is_dir():
        return False
    shutil.rmtree(path, ignore_errors=True)
    return True


def _clean(value: str, limit: int = 200) -> str:
    return " ".join(str(value or "").split())[:limit]


@dataclass
class AccountStore:
    """The accounts of one bot, and which of them is running.

    A plain JSON file beside the Chrome directories — ``/profiles/<bot>/accounts.json``
    next to ``/profiles/<bot>/accounts/`` — for the same reasons the roster is
    one: a handful of lines, read on every request, trivially inspectable and
    repairable by hand. It sits on the *profile* volume, not the data volume,
    because it describes those directories and would be nonsense without them;
    and it sits *beside* them rather than inside, because Chrome opens a
    user-data-dir wholesale and would find a stray JSON in it.

    It holds no credentials. An account is identified by a slug the operator
    chose; what login it carries is free-text ``email``/``notes`` and nothing
    reads them. The repo is public and this is rendered in HTML.
    """

    root: Path                      # the bot's profile root: /profiles/<bot>
    path: Path                      # the metadata file
    accounts: list[Account] = field(default_factory=list)
    active: str = DEFAULT_ACCOUNT

    # -- reading ----------------------------------------------------------

    def get(self, name: str) -> Account | None:
        return next((a for a in self.accounts if a.name == name), None)

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.accounts]

    def active_account(self) -> str:
        """The account a task runs as when it does not name one.

        Always a real account: an ``active`` pointing at something since removed
        (or at an account a hand-edited file never declared) falls back to the
        first, which is what keeps a bot that has accounts runnable rather than
        failing every task with "no such account"."""
        if any(a.name == self.active for a in self.accounts):
            return self.active
        return self.accounts[0].name if self.accounts else DEFAULT_ACCOUNT

    def summary(self) -> list[dict]:
        """Every account with its live state, for the UI.

        ``signed_in`` and ``browser_running`` are read from disk and from the
        port file respectively, so this is cheap and safe to call on every poll
        — it never touches the browser.
        """
        from .browser import _read_devtools_endpoint  # local: avoid a cycle

        out = []
        active = self.active_account()
        for a in self.accounts:
            directory = account_dir(self.root, a.name)
            out.append({
                **a.to_dict(),
                "active": a.name == active,
                "signed_in": signed_in(directory),
                "dir_exists": directory.is_dir(),
                # Whether *this* account is the one Chrome is currently holding.
                # A stale DevToolsActivePort file is common (Chrome does not
                # remove it), so this is only trusted when something is
                # listening on the port it names.
                "browser_running": _read_devtools_endpoint(directory, timeout_s=0.1) is not None,
            })
        return out

    def to_dict(self) -> dict:
        return {
            "accounts": self.summary(),
            "active": self.active_account(),
            "count": len(self.accounts),
        }

    # -- writing ----------------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "accounts": [asdict(a) for a in self.accounts],
            "active": self.active,
        }, indent=2) + "\n")
        os.replace(tmp, self.path)   # atomic: a reader sees the whole old or new file

    def add(self, name: str, **fields) -> Account:
        existing = self.get(name)
        if existing is None:
            existing = Account(name=name, created_at=time.time())
            self.accounts.append(existing)
        for key in ("label", "email", "notes"):
            if fields.get(key) is not None:
                setattr(existing, key, _clean(fields[key], 500 if key == "notes" else 200))
        # Creating the directory now, not on first launch, so the operator can
        # see the account exists and is simply not signed in yet — which is the
        # state they need to act on. Cheap: Chrome would create it anyway.
        account_dir(self.root, name).mkdir(parents=True, exist_ok=True)
        self.accounts.sort(key=lambda a: a.created_at)
        return existing

    def set_active(self, name: str) -> Account | None:
        account = self.get(name)
        if account is None:
            return None
        self.active = name
        return account

    def remove(self, name: str) -> Account | None:
        """Drop one account from the list. Refuses to remove the last one.

        The directory is deleted by the caller only when it asks (`purge`), and
        the *last* account is never removable: a bot with no accounts has no
        Chrome directory to launch, so every task would fail with a confusing
        error rather than an honest "add an account first".
        """
        account = self.get(name)
        if account is None:
            return None
        if len(self.accounts) <= 1:
            raise ValueError("a bot must keep at least one account")
        self.accounts = [a for a in self.accounts if a.name != name]
        if self.active == name:
            self.active = self.accounts[0].name
        return account


def load_accounts(root: Path, path: Path) -> AccountStore:
    """Read a bot's accounts, migrating and defaulting as needed.

    Never raises: this runs at import of the control plane, so an exception here
    would crashloop the pod. A corrupt file degrades to the single default
    account, which is exactly the pre-accounts behaviour and therefore always
    runnable — and the operator can repair the file by hand.
    """
    store = AccountStore(root=root, path=path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        raw = None
    if isinstance(raw, dict):
        for item in raw.get("accounts") or []:
            if not isinstance(item, dict) or not valid_account(item.get("name", "")):
                continue
            store.accounts.append(Account(
                name=item["name"],
                label=_clean(item.get("label", "")),
                email=_clean(item.get("email", "")),
                notes=_clean(item.get("notes", ""), 500),
                created_at=float(item.get("created_at") or 0.0),
            ))
        if isinstance(raw.get("active"), str):
            store.active = raw["active"]

    try:
        ensure_layout(root)
    except OSError:
        log.warning("could not prepare the accounts layout under %s", root, exc_info=True)

    if not store.accounts:
        # The migrated legacy directory is the first account, so an existing bot
        # keeps its login under a name the operator can actually see.
        store.add(DEFAULT_ACCOUNT)
        store.active = DEFAULT_ACCOUNT
        try:
            store.save()
        except OSError:
            log.warning("could not write %s", path, exc_info=True)
    return store
