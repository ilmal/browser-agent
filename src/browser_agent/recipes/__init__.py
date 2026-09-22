"""Recipe registry.

Importing this module registers every built-in recipe.
"""

from . import (  # noqa: F401
    agent_task,
    facebook,
    flights,
    linkedin,
    minesweeper,
    plan_task,
    x,
)

__all__ = [
    "agent_task",
    "facebook",
    "flights",
    "linkedin",
    "minesweeper",
    "plan_task",
    "x",
]
