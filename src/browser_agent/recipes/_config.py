"""Operator overrides for a built-in recipe's config.

A built-in recipe's selector list is a Python literal inside ``run()``, which is
why "the button moved" used to mean a code deploy. Each of those literals is now
the *default* argument to :func:`cfg`, and the operator's override lives in the
library (see ``recipe_store``). With no library — or no entry for that key — the
literal is returned unchanged, so the built-in behaviour is exactly what it was.

Only keys listed in ``recipe_store.OVERRIDABLE`` are ever stored, and the store
refuses a value of the wrong type rather than handing Playwright a string where
it wants a list of selectors.
"""

from __future__ import annotations

from typing import TypeVar

from ..config import load_settings
from ..recipe_store import store_for

T = TypeVar("T")


def cfg(recipe: str, key: str, default: T) -> T:
    """The operator's value for ``recipe.key``, or ``default``.

    A config lookup must never be the reason a recipe fails: the built-in
    literal is always a valid answer, so any problem reading the library
    degrades to exactly the old behaviour.
    """
    try:
        return store_for(load_settings()).cfg(recipe, key, default)
    except Exception:
        return default
