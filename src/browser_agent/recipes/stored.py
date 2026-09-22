"""A recipe the operator composed out of steps, instead of Python.

The spec is a saved :class:`~browser_agent.plan_model.Plan` plus a name and a
description: the same schema the planner emits, so it is validated by the same
``parse_plan`` and run by the same ``run_plan``. Nothing here executes steps
itself — this class exists only to give a stored spec the ``Recipe`` shape the
registry expects.

The deps are built lazily and injectable, exactly as ``plan_task`` does it:
importing this module must never pay for a model client.
"""

from __future__ import annotations

import logging
from typing import Any

from ..browser import BrowserSession
from ..config import Settings, load_settings
from ..laya_gate import LayaGate
from ..picker import ElementPicker
from ..plan_model import parse_plan
from ..recipe_store import _require_deterministic_targets, store_for
from ._plan_exec import run_plan

log = logging.getLogger(__name__)


class StoredRecipe:
    """One operator-authored recipe, materialised from the library."""

    def __init__(
        self,
        spec: dict[str, Any],
        *,
        settings: Settings | None = None,
        laya: LayaGate | None = None,
        picker: ElementPicker | None = None,
    ) -> None:
        self.name: str = spec["name"]
        self.description: str = spec.get("description") or f"Stored recipe {self.name}"
        self.entry_url: str = str(spec.get("entry_url") or "")
        #: The stored steps, kept as plain dicts so a spec refresh is picked up
        #: on the next run rather than being frozen at registration time.
        self.steps: list[dict[str, Any]] = list(spec.get("steps") or [])
        self._settings = settings
        self._laya = laya
        self._picker = picker

    def replace(self, spec: dict[str, Any]) -> None:
        """Adopt an edited spec in place, so the registry entry stays valid."""
        self.name = spec["name"]
        self.description = spec.get("description") or f"Stored recipe {self.name}"
        self.entry_url = str(spec.get("entry_url") or "")
        self.steps = list(spec.get("steps") or [])

    def _deps(self, settings: Settings) -> None:
        if self._settings is None:
            self._settings = settings
        if self._laya is None:
            self._laya = LayaGate(self._settings)
        if self._picker is None:
            self._picker = ElementPicker(self._settings)

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings or load_settings()
        self._deps(settings)
        # Re-validated on every run: the stored JSON is the source of truth, and
        # an operator may have edited it since this object was built. Both gates
        # — the plan schema and the stored-recipe determinism rule — so a spec
        # that reached this object by any route is held to the same standard a
        # save would have been.
        plan = parse_plan({"entry_url": self.entry_url, "steps": self.steps})
        _require_deterministic_targets(plan)
        task_text = str(payload.get("task") or payload.get("text") or self.description).strip()
        log.info("stored recipe %s: %d step(s), entry %s", self.name, len(plan.steps), plan.entry_url)
        result = await run_plan(session, plan, task_text, settings, self._laya,
                                picker=self._picker)
        result["recipe"] = self.name
        return result


def load_stored_recipes() -> list[str]:
    """Read the library and install what it holds. Never raises.

    Called from ``tasks.get_recipe``/``list_recipes`` (so an edit lands without
    a restart) and once at boot. It is idempotent and cheap when nothing has
    changed: the store caches on the directory's mtime.
    """
    from .. import tasks

    settings = load_settings()
    store = store_for(settings)
    specs = store.specs()
    for name, spec in specs.items():
        try:
            tasks.install_stored(spec)
        except Exception:
            log.exception("could not install stored recipe %s", name)
    # A recipe deleted from the library must stop being runnable; a built-in
    # that was overridden falls back to its Python class, which was never
    # removed from the registry.
    for stale in tasks.stored_recipe_names() - set(specs):
        tasks.forget_recipe(stale)
    return sorted(specs)
