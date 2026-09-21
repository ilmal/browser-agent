"""Recipe registry.

Importing this module registers every built-in recipe.
"""

from . import facebook, linkedin, x  # noqa: F401

__all__ = ["facebook", "linkedin", "x"]
