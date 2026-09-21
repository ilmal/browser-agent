"""Recipe registry.

Importing this module registers every built-in recipe.
"""

from . import agent_task, facebook, linkedin, plan_task, x  # noqa: F401

__all__ = ["agent_task", "facebook", "linkedin", "plan_task", "x"]
