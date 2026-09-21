"""Recipe registry.

Importing this module registers every built-in recipe.
"""

from . import agent_task, facebook, linkedin, x  # noqa: F401

__all__ = ["agent_task", "facebook", "linkedin", "x"]
