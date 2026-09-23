"""The operator's recipe library: built-in config overrides and step recipes.

Until this existed, a recipe's selectors were Python literals and its config
lived in a frozen ``Settings``, so "the button moved" was a code deploy. This
module is the write-free half of the fix: a directory — in the cluster, a
ConfigMap mounted into every pod — holding two kinds of file:

* ``_overrides.json`` — a mapping of ``recipe name -> {key: value}`` for the
  built-in recipes. ``recipes._config.cfg()`` reads it; the Python literal
  stays as the default, so an absent or unreadable file changes nothing.
* ``<name>.json`` — one operator-composed recipe, whose steps are validated by
  ``plan_model`` and executed by ``recipes._plan_exec.run_plan``. No new
  execution machinery: a stored recipe is a saved plan with a name on it.

Two properties are load-bearing and easy to lose:

* **Loading never raises.** ``api.py`` loads settings at import, so an exception
  here is a crashlooping pod with no control plane to fix it from. Every file
  is parsed inside a guard, logged, and skipped.
* **The library is shared.** The hub is the only writer; pods are read-only
  consumers. A pod never writes these files, so there is no last-writer-wins
  problem between profiles.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .plan_model import Plan, PlanRejected, parse_plan

log = logging.getLogger(__name__)

OVERRIDES_FILE = "_overrides.json"

#: Names the runner routes on by string (``tasks.AGENT_RECIPE``/``PLAN_RECIPE``).
#: A stored recipe may not take one: the routing would shadow it and the file
#: would be dead weight that still looks installed.
RESERVED_NAMES = frozenset({"agent.task", "plan.task"})

#: Config keys an operator may override, per recipe. Deliberately a closed list:
#: this is a config surface, not a plugin system. An unknown key is refused on
#: save rather than silently stored and never read.
OVERRIDABLE: dict[str, tuple[str, ...]] = {
    "x.post": (
        "entry_url", "selectors.composer", "selectors.textbox", "selectors.submit",
        "max_chars",
    ),
    "facebook.page_post": (
        "entry_url", "selectors.composer", "selectors.textbox", "selectors.submit",
        "max_chars",
    ),
    "linkedin.page_post": (
        "entry_url", "selectors.composer", "selectors.textbox", "selectors.submit",
        "max_chars",
    ),
    "plan.task": ("entry_url", "planner_prompt"),
    "minesweeper.play": ("entry_url", "pace.min_ms", "pace.max_ms", "max_clicks"),
}

#: The five actions ``plan_model.Step`` already understands. A stored recipe
#: composes these and nothing else — adding one is a change to the step schema
#: and the executor, not a change to a recipe.
STEP_ACTIONS = ("navigate", "click", "type", "extract", "wait")

#: Metadata a *learned* recipe carries that a hand-authored one does not.
#: Preserved through validation on purpose: dropping them would silently promote
#: a recipe the agent wrote from one lucky run to one the router trusts, which
#: is the whole distinction the guardrail exists to hold.
LEARNED_KEYS = ("origin", "unverified", "replays", "min_replays", "created")


@dataclass
class RecipeStore:
    """Read side of the library. Cached; re-read when the directory changes."""

    directory: Path
    _recipes: dict[str, dict[str, Any]] = field(default_factory=dict)
    _overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    _errors: list[str] = field(default_factory=list)
    _stamp: tuple[float, int] | None = None
    _checked_at: float = 0.0
    _ttl_s: float = 10.0

    # -- reading -----------------------------------------------------------

    def refresh(self, *, force: bool = False) -> None:
        """Re-read when the directory changed. Never raises."""
        now = time.monotonic()
        if not force and now - self._checked_at < self._ttl_s:
            return
        self._checked_at = now
        stamp = self._stamp_of()
        if not force and stamp is not None and stamp == self._stamp:
            return
        self._stamp = stamp
        self._read()

    def _stamp_of(self) -> tuple[float, int] | None:
        """(mtime, file count) of the directory — cheap change detection."""
        try:
            entries = [p for p in self.directory.iterdir() if p.suffix == ".json"]
        except OSError:
            return None
        if not entries:
            return (0.0, 0)
        try:
            return (max(p.stat().st_mtime for p in entries), len(entries))
        except OSError:
            return None

    def _read(self) -> None:
        recipes: dict[str, dict[str, Any]] = {}
        overrides: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        for path in sorted(self.directory.glob("*.json")) if self.directory.is_dir() else []:
            try:
                data = json.loads(path.read_text())
            except Exception as exc:  # json, encoding, IO — all "skip this file"
                log.warning("recipe %s is unreadable, skipping: %s", path.name, exc)
                errors.append(f"{path.name}: {exc}")
                continue
            if path.name == OVERRIDES_FILE:
                if not isinstance(data, dict):
                    errors.append(f"{path.name}: expected an object of recipe -> keys")
                    continue
                overrides = {
                    str(k): dict(v) for k, v in data.items() if isinstance(v, dict)
                }
                continue
            try:
                spec = _validate(data, source=path.name)
            except PlanRejected as exc:
                log.warning("recipe %s is invalid, skipping: %s", path.name, exc)
                errors.append(f"{path.name}: {exc}")
                continue
            recipes[spec.name] = spec.spec

        self._recipes = recipes
        self._overrides = overrides
        self._errors = errors

    # -- accessors ---------------------------------------------------------

    @property
    def errors(self) -> list[str]:
        """Files that were skipped, so the editor can show them rather than
        pretending a recipe that failed to load does not exist."""
        self.refresh()
        return list(self._errors)

    def specs(self) -> dict[str, dict[str, Any]]:
        self.refresh()
        return dict(self._recipes)

    def get(self, name: str) -> dict[str, Any] | None:
        self.refresh()
        return self._recipes.get(name)

    def overrides_for(self, name: str) -> dict[str, Any]:
        self.refresh()
        return dict(self._overrides.get(name, {}))

    def all_overrides(self) -> dict[str, dict[str, Any]]:
        self.refresh()
        return {k: dict(v) for k, v in self._overrides.items()}

    def learned_min_replays(self) -> int:
        """How many clean replays a learned recipe needs before it is trusted.

        Read from the environment rather than passed in so a ``RecipeStore``
        built anywhere — the loader, a test, the API — agrees on the number. A
        test that wants the count out of the way sets the env var, the same way
        it sets every other knob.
        """
        import os

        try:
            return max(1, int(os.environ.get("LEARNED_RECIPE_MIN_REPLAYS", "2")))
        except ValueError:
            return 2

    def learned(self) -> list[dict[str, Any]]:
        """Every recipe in this store, filtered to the ones the agent earned.

        A learned directory holds only learned recipes, so this is ``specs()``
        in practice; filtering on ``origin`` keeps that true even if an operator
        copies a spec in by hand, and keeps the router's promotion rule honest.
        """
        return [s for s in self.specs().values() if s.get("origin") == "learned"]

    def trusted_learned(self) -> list[dict[str, Any]]:
        """Learned recipes that have replayed cleanly enough to be routed to."""
        floor = self.learned_min_replays()
        return [
            s for s in self.learned()
            if not s.get("unverified") and int(s.get("replays") or 0) >= floor
        ]

    def cfg(self, recipe: str, key: str, default: Any) -> Any:
        """One overridden value, or the built-in literal.

        Type-checked against the default: a stored string where the recipe
        wants a list of selectors is a bad edit, and returning it would fail
        deep inside Playwright where the cause is unrecognisable. Refusing here
        keeps the built-in working and puts the mistake where it happened.
        """
        value = self.overrides_for(recipe).get(key)
        if value is None:
            return default
        if isinstance(default, list):
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                return value
        elif isinstance(default, bool):
            if isinstance(value, bool):
                return value
        elif isinstance(default, int):
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        elif isinstance(default, float):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        elif isinstance(value, type(default)):
            return value
        log.warning(
            "override %s.%s has the wrong type (%s where %s was expected); using the built-in",
            recipe, key, type(value).__name__, type(default).__name__,
        )
        return default


def _validate(data: Any, *, source: str) -> "StoredSpec":
    """Turn one stored file into an identity plus a validated Plan."""
    if not isinstance(data, dict):
        raise PlanRejected("expected a JSON object")
    name = str(data.get("name") or "").strip()
    if not name:
        raise PlanRejected("`name` is required")
    if name in RESERVED_NAMES:
        raise PlanRejected(f"{name!r} is reserved and cannot be overridden")
    if not _valid_name(name):
        raise PlanRejected(f"{name!r} must be lowercase letters, digits, dashes and dots")

    description = str(data.get("description") or "").strip()
    entry_url = str(data.get("entry_url") or "").strip()
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlanRejected("`steps` must be a non-empty list")

    # parse_plan is the same validator the planner's output goes through, so a
    # stored recipe is held to exactly the standard a model's plan is.
    plan = parse_plan({"entry_url": entry_url, "steps": steps})
    _require_deterministic_targets(plan)

    spec: dict[str, Any] = {
        "name": name,
        "description": description or f"Stored recipe {name}",
        "entry_url": plan.entry_url,
        "steps": [s.model_dump() for s in plan.steps],
    }
    # A learned recipe's provenance rides along, so the router can tell one the
    # agent wrote from one a human did, and hold the former to its replay count.
    for key in LEARNED_KEYS:
        if key in data:
            spec[key] = data[key]

    return StoredSpec(name=name, spec=spec)


def _require_deterministic_targets(plan: Plan) -> None:
    """A stored recipe's click/type must name a selector.

    The planner may leave one out — the pickers resolve it once, on the page in
    front of it. A *saved* recipe has no page: a selectorless step would ask a
    model to guess every time it runs, which is fine for a throwaway plan and a
    poor thing to persist as a reusable recipe.
    """
    for i, step in enumerate(plan.steps):
        if step.action in {"click", "type"} and not step.selector:
            raise PlanRejected(
                f"step {i}: a saved {step.action} step needs a `selector` — "
                "the picker can only resolve a step it is looking at"
            )


def _valid_name(name: str) -> bool:
    import re

    return bool(re.fullmatch(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?", name))


@dataclass
class StoredSpec:
    name: str
    spec: dict[str, Any]


# -- writing ---------------------------------------------------------------
#
# Only the hub calls these: it is the one writer, and it holds the library on
# its own volume. A bot pod's copy is a read-only ConfigMap projection, so it
# never has a reason to write and never has a last-writer-wins race with
# another profile.


class RecipeError(Exception):
    """The operator's edit cannot be stored. The message is shown verbatim."""


def validate_override(recipe: str, key: str, value: Any) -> None:
    """Refuse an override the recipe could not use. Raises RecipeError."""
    allowed = OVERRIDABLE.get(recipe)
    if allowed is None:
        raise RecipeError(f"{recipe!r} has no overridable config")
    if key not in allowed:
        raise RecipeError(
            f"{key!r} is not an overridable key for {recipe}; "
            f"choose one of: {', '.join(allowed)}"
        )
    if key.endswith("entry_url"):
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            raise RecipeError(f"{key} must be an http(s) URL")
    elif key.startswith("selectors."):
        if not isinstance(value, list) or not value or not all(
            isinstance(v, str) and v.strip() for v in value
        ):
            raise RecipeError(f"{key} must be a non-empty list of CSS selectors")
    elif key in {"pace.min_ms", "pace.max_ms"}:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RecipeError(f"{key} must be a non-negative integer (milliseconds)")
    elif key == "max_clicks":
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RecipeError("max_clicks must be a positive integer")
    elif key == "max_chars":
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise RecipeError("max_chars must be a positive integer")
    elif key == "planner_prompt":
        if not isinstance(value, str) or not value.strip():
            raise RecipeError("planner_prompt must be a non-empty string")


def _write_atomic(path: Path, data: Any) -> None:
    """Write via a temp file and rename: a pod may be reading this right now."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def save_step_recipe(store: RecipeStore, spec: dict[str, Any]) -> dict[str, Any]:
    """Create or replace one operator-authored step recipe."""
    # _validate is the same gate the read path uses, so a saved recipe is one
    # the loader is guaranteed to accept — there is no "saved but unreadable".
    validated = _validate(spec, source="edit")
    _write_atomic(store.directory / f"{validated.name}.json", validated.spec)
    store.refresh(force=True)
    return validated.spec


def save_overrides(store: RecipeStore, recipe: str, values: dict[str, Any]) -> dict[str, Any]:
    """Merge config overrides for a built-in recipe."""
    if recipe in RESERVED_NAMES:
        raise RecipeError(f"{recipe!r} is reserved and has no overridable config")
    for key, value in values.items():
        validate_override(recipe, key, value)
    current = store.all_overrides()
    merged = {**current.get(recipe, {}), **values}
    current[recipe] = merged
    _write_atomic(store.directory / OVERRIDES_FILE, current)
    store.refresh(force=True)
    return merged


def delete_recipe(store: RecipeStore, name: str) -> str:
    """Remove a stored recipe, or clear a built-in's overrides.

    Returns "stored" | "override", or raises when there is nothing to remove.
    """
    if name in RESERVED_NAMES:
        raise RecipeError(f"{name!r} is reserved and cannot be removed")
    path = store.directory / f"{name}.json"
    if path.exists():
        try:
            path.unlink()
        except OSError as exc:
            raise RecipeError(f"could not remove {name}: {exc}") from exc
        store.refresh(force=True)
        return "stored"
    overrides = store.all_overrides()
    if name in overrides:
        del overrides[name]
        _write_atomic(store.directory / OVERRIDES_FILE, overrides)
        store.refresh(force=True)
        return "override"
    raise RecipeError(f"no stored recipe or override named {name!r}")


# -- process-wide store -----------------------------------------------------

_STORE: RecipeStore | None = None
_LEARNED: RecipeStore | None = None


def store_for(settings: Settings) -> RecipeStore:
    """The one store for this process (one process = one profile)."""
    global _STORE
    if _STORE is None or _STORE.directory != settings.recipes_dir:
        _STORE = RecipeStore(settings.recipes_dir)
    return _STORE


def learned_store_for(settings: Settings) -> RecipeStore | None:
    """The bot's own store for recipes its agent earned. None when disabled.

    A :class:`RecipeStore` reads whatever ``*.json`` it finds, so the same class
    serves both: the operator's library is one directory of specs, and the
    learned ones are another. They are separate on purpose — a learned recipe
    must never be published to the cluster-wide ConfigMap (see
    ``Settings.learned_recipes_dir``).
    """
    directory = settings.learned_recipes_dir
    if not settings.learn_recipes or not str(directory):
        return None
    global _LEARNED
    if _LEARNED is None or _LEARNED.directory != directory:
        _LEARNED = RecipeStore(directory)
    return _LEARNED


def record_replay(store: RecipeStore, name: str, *, ok: bool) -> dict[str, Any] | None:
    """Count one replay of a learned recipe. Returns the updated spec, or None.

    ``ok`` is the executor's verdict on the whole run, not a step's. A failed
    replay does not increment toward promotion — a recipe that half-worked is
    exactly the one that must keep being offered rather than trusted — and it
    does not reset the count either, so a recipe that has replayed once cleanly
    is not punished for a later page change. Never raises.
    """
    try:
        spec = store.get(name)
        if spec is None:
            return None
        if ok:
            spec["replays"] = int(spec.get("replays") or 0) + 1
        if spec["replays"] >= int(store.learned_min_replays()):
            spec["unverified"] = False
        save_learned_spec(store, spec)
        return spec
    except Exception:
        log.debug("could not record replay for %s", name, exc_info=True)
        return None


def save_learned_spec(store: RecipeStore, spec: dict[str, Any]) -> dict[str, Any]:
    """Write one learned spec, validated like any other. Raises RecipeError.

    ``_validate`` is the same gate the read path uses, so a learned recipe that
    was written is one the loader is guaranteed to accept — the executor's
    contract that a recipe either replays or fails into the agent, never that it
    half-loads.
    """
    validated = _validate(spec, source="learned")
    _write_atomic(store.directory / f"{validated.name}.json", validated.spec)
    store.refresh(force=True)
    return validated.spec
