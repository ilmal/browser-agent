"""The freeform task recipe.

Not a recipe in the usual sense: it has no Playwright path, so the runner
routes it straight to the agent. It exists so freeform instructions appear in
the registry (and therefore in the admin UI's recipe picker) alongside the
deterministic ones.

Payload:
  goal   — what to do, in plain language (required)
  url    — where to start (defaults to the site's home page)
"""

from __future__ import annotations

from typing import Any

from ..browser import BrowserSession
from ..tasks import AGENT_RECIPE, register


class AgentTask:
    name = AGENT_RECIPE
    description = "Freeform: give the agent an instruction and a start URL."
    # No default: a freeform instruction is meaningless without a page to work
    # on, so the runner requires `url` in the payload rather than silently
    # handing the agent a blank page.
    entry_url = ""

    async def run(self, session: BrowserSession, payload: dict[str, Any]) -> dict[str, Any]:
        # The runner intercepts this recipe before calling run(); reaching here
        # means that interception was bypassed, which is a bug worth surfacing.
        raise RuntimeError("agent.task must be routed by the runner, not run directly")


register(AgentTask())
