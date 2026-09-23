"""Turn a successful agent run into a candidate recipe.

This is the "save down the steps, the buttons, everything that actually made it
move forward" half of Nils's idea (2026-09-23): the first run of a task pays for
the agent, and what the agent did is written down as a :class:`Plan` the
deterministic executor can replay for free.

Three properties are load-bearing, and they are all about refusing:

* **Complete or nothing.** Every action in the run must map to a step the plan
  executor can perform. A run that scrolled, sent keys, or switched tabs in the
  middle cannot be represented faithfully, so it is not harvested at all. A
  recipe that silently drops a step is worse than no recipe: it would replay a
  different task while claiming to be the same one.
* **A durable target or nothing.** A click/type step is only harvested when the
  element it acted on yields a selector. browser-use names elements by position
  in its own view, which is meaningless to a later run; the element's own id,
  name, or XPath is what survives. No target, no recipe.
* **Never raise.** This runs inside the agent's success path. A harvesting bug
  must cost a lost recipe, never a task that succeeded being reported as failed.

The output is a spec dict in the same shape ``recipe_store`` validates and
``StoredRecipe`` replays — no new execution machinery, the same ``run_plan``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

#: browser-use action name -> the plan step it becomes. The agent's own names
#: (``views.py`` in browser_use/tools) are what ``model_dump`` keys on. Aliases
#: are kept because the package is unpinned and has renamed actions across
#: releases (``go_to_url`` -> ``navigate``, ``click_element`` ->
#: ``click_element_by_index``); a rename must cost a lost harvest, not a wrong
#: mapping, so an unknown name disqualifies the run rather than being guessed.
_STEP_FOR_ACTION: dict[str, str] = {
    "navigate": "navigate",
    "go_to_url": "navigate",
    "click_element_by_index": "click",
    "click_element": "click",
    "click_element_index_only": "click",
    "input_text": "type",
    "type_text": "type",
}

#: Actions that end the run cleanly when they are the last thing it did. A
#: trailing done/extract does not need to be a step; anything else mid-run does.
_TERMINAL_ACTIONS = frozenset({"done", "extract", "extract_content"})

#: An id or name is only turned into a CSS selector when it needs no escaping —
#: a value that requires quoting is a value whose selector would be a quoting
#: bug waiting to happen, and the XPath fallback is right there.
_SIMPLE_IDENT = re.compile(r"^[A-Za-z0-9_-]+$")

#: Cap on a harvested goal string. ``next_goal`` is model output; it is used as
#: a human-readable step description, not a prompt, but keeping it short keeps a
#: learned recipe readable in the operator's library.
_GOAL_MAX = 120


def _dump(action: Any) -> tuple[str, dict[str, Any]] | None:
    """The single (name, params) pair a browser-use ActionModel carries.

    Every action is a model with exactly one field set, so ``model_dump`` with
    ``exclude_none`` is the one honest way to read which action it is — the
    class name is not the action name (``ClickElementAction`` is
    ``click_element_by_index`` to the executor).
    """
    try:
        data = action.model_dump(exclude_none=True)
    except Exception:
        return None
    if not isinstance(data, dict) or len(data) != 1:
        return None
    (name, params), = data.items()
    return str(name), params if isinstance(params, dict) else {}


def _selector_for(elem: Any) -> str | None:
    """A selector for the element an action was dispatched on.

    Preference order is durability: an id survives a re-render, a ``name``
    survives a restyle, and browser-use's XPath survives neither but is what is
    left when the page offers nothing better. The XPath is absolute, so a stale
    one fails visibly at ``locator.first.wait_for`` — a fast failure into the
    agent fallback, which is the contract for every learned recipe.
    """
    if elem is None:
        return None
    attrs = getattr(elem, "attributes", None) or {}
    try:
        ident = str(attrs.get("id") or "").strip()
        if ident and _SIMPLE_IDENT.match(ident):
            return f"#{ident}"
        name = str(attrs.get("name") or "").strip()
        if name and _SIMPLE_IDENT.match(name):
            return f'[name="{name}"]'
    except Exception:
        pass
    xpath = str(getattr(elem, "x_path", "") or "").strip()
    if xpath:
        return f"xpath={xpath}"
    return None


def _label_for(elem: Any) -> str:
    """A human name for the element, for the step's goal when the model gave none."""
    if elem is None:
        return ""
    for attr in ("ax_name", "node_value"):
        value = str(getattr(elem, attr, "") or "").strip()
        if value:
            return value[:80]
    return ""


def _step_from(action: Any, elem: Any, goal: str) -> dict[str, Any] | None:
    """One action + the element it touched -> one plan step, or None."""
    parsed = _dump(action)
    if parsed is None:
        return None
    name, params = parsed
    kind = _STEP_FOR_ACTION.get(name)
    if kind is None:
        return None
    if kind == "navigate":
        url = str(params.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return None
        return {"action": "navigate", "text": url}
    selector = _selector_for(elem)
    if not selector:
        return None
    step_goal = (goal or _label_for(elem) or "the element the agent used")[:_GOAL_MAX]
    if kind == "click":
        return {"action": "click", "goal": step_goal, "selector": selector}
    text = str(params.get("text") or "")
    if not text:
        return None
    return {"action": "type", "goal": step_goal, "text": text, "selector": selector}


def slug_for(goal: str) -> str:
    """A recipe name for a goal: readable, and valid for the store."""
    low = re.sub(r"[^a-z0-9]+", "-", (goal or "").lower()).strip("-")
    low = low[:40].strip("-") or "task"
    digest = hashlib.sha256((goal or "").encode()).hexdigest()[:6]
    return f"learned-{low}-{digest}"


def harvest(
    history: Any, *, entry_url: str, goal: str, task_text: str = ""
) -> dict[str, Any] | None:
    """A storable recipe spec for a successful run, or None if it cannot be one.

    ``history`` is browser-use's ``AgentHistoryList``. Nothing here raises: a
    structure that has moved under the unpinned dependency means "no recipe",
    which is logged and forgotten.
    """
    try:
        return _harvest(history, entry_url=entry_url, goal=goal, task_text=task_text)
    except Exception:
        log.debug("could not harvest a recipe from the run", exc_info=True)
        return None


def _harvest(
    history: Any, *, entry_url: str, goal: str, task_text: str
) -> dict[str, Any] | None:
    url = (entry_url or "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return None

    items = list(getattr(history, "history", None) or [])
    if not items:
        return None

    steps: list[dict[str, Any]] = []
    for item in items:
        output = getattr(item, "model_output", None)
        actions = list(getattr(output, "action", None) or [])
        if not actions:
            return None
        state = getattr(item, "state", None)
        elements = list(getattr(state, "interacted_element", None) or [])
        step_goal = str(getattr(output, "next_goal", "") or "").strip()
        for i, action in enumerate(actions):
            parsed = _dump(action)
            if parsed is None:
                return None
            name, _ = parsed
            if name in _TERMINAL_ACTIONS:
                # A done/extract ends the run. It may only be the last action
                # of the last step; anywhere else the run continued past it and
                # we cannot reproduce that.
                if item is not items[-1] or i != len(actions) - 1:
                    return None
                continue
            elem = elements[i] if i < len(elements) else None
            step = _step_from(action, elem, step_goal)
            if step is None:
                return None
            steps.append(step)

    if not steps:
        return None

    recipe_goal = (task_text or goal or "").strip()
    return {
        "name": slug_for(recipe_goal or url),
        "description": (recipe_goal or "A task the agent completed")[:_GOAL_MAX],
        "entry_url": url,
        "steps": steps,
        "origin": "learned",
        #: Until it has replayed cleanly ``learned_recipe_min_replays`` times it
        #: is offered, never auto-routed. A recipe harvested from one run is a
        #: hypothesis about a second run, and only a replay tests it.
        "unverified": True,
        "replays": 0,
    }
