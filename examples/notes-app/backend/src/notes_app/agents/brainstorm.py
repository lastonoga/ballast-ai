"""BallastAgents for the brainstorm flow.

Why ``BallastAgent`` instead of building bare ``pydantic_ai.Agent``
instances in the flow file:

1. ``model_settings()`` lives where the model config does — same place
   ``NotesAgent`` sets ``openrouter_reasoning={"effort":"none"}`` and
   ``openrouter_usage={"include": True}``. Brainstorm runs benefit
   from the same OpenRouter knobs (reasoning costs money + adds
   latency; usage tracking is needed for cost dashboards).

2. ``capabilities`` slot — apps can layer ``BudgetGuard`` /
   instrumentation / cost tracking onto these agents without
   touching the flow.

3. ``build_agent`` / ``model_settings`` is the canonical agent-
   construction contract in this codebase; matching it keeps the
   brainstorm path readable next to the other agents.

Output strategy: pydantic-ai's **default** (``ToolOutput`` for
``BaseModel``), with one twist: for ``qwen/qwen3.6*`` model ids we
pass an explicit ``ModelProfile`` (see ``openrouter_profile.py``)
that flips the agent to ``NativeOutput`` mode — provider-side JSON-
schema validation via ``response_format``. Reason: OpenRouter's
qwen3.6 endpoints support native JSON-schema output but reject
``tool_choice="required"`` (which the default ``ToolOutput`` path
sets), so without the profile patch every qwen3.6 call 404s.

For models pydantic-ai's built-in registry already handles correctly
(claude-sonnet, gpt-4o, deepseek-chat, etc.) ``profile_for(...)``
returns ``None`` and the default profile applies — they keep using
``ToolOutput``, which is the most reliable mode where supported.

These agents are NOT exposed via the app's agent dispatch table —
they aren't selected by ``thread.agent``. The ``brainstorm``
workflow function holds them indirectly through the module-level
``DivergentConvergent`` instance and the pattern invokes
``branch.agent.run(task)``. ``BallastAgent.run`` proxies to the
underlying pydantic-ai ``Agent.run``, so a ``BallastAgent`` subclass
satisfies the framework's structural ``DivergentAgent`` protocol
without an adapter. ``build_deps`` returns ``None`` because there is
no per-thread context for one-shot model calls.

No ``from __future__ import annotations``: pydantic-ai introspects
``get_type_hints()`` at decoration time.
"""

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openrouter import OpenRouterModelSettings
from ballast.persistence.thread.domain import Thread
from ballast.runtime import BallastAgent

from notes_app.agents.openrouter import (
    build_openrouter_model,
    default_model_settings,
)
from notes_app.models.todo import TodoIdea, TodoIdeas


class BrainstormDivergentAgent(BallastAgent):
    """One branch of the divergent fan-out.

    Parameterized by constructor args (model / system_prompt /
    temperature) so a single class can back the practical / creative /
    analyst flavours without N subclasses. Returns a pool of
    ``TodoIdeas`` per ``.agent.run(topic)`` call.
    """

    name = "brainstorm-divergent"
    metadata_model = None

    def __init__(
        self,
        *,
        model_name: str,
        system_prompt: str,
        temperature: float = 0.9,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._temperature = temperature

    def build_agent(self) -> Agent[None, Any]:
        return Agent(
            model=build_openrouter_model(self._model_name),
            output_type=TodoIdeas,
            system_prompt=self._system_prompt,
        )

    async def build_deps(
        self,
        *,
        thread: Thread,
        message: ModelMessage | None,
    ) -> None:
        # Brainstorm divergent agents are invoked one-shot from inside
        # BrainstormFlow workflow steps — no per-thread context.
        del thread, message
        return None

    def model_settings(self) -> OpenRouterModelSettings:
        return default_model_settings(temperature=self._temperature)


class BrainstormSynthesizerAgent(BallastAgent):
    """Convergent synthesizer. Picks ONE ``TodoIdea`` from the merged
    candidate pool produced by the divergent agents.

    Lower temperature than divergent agents — synthesis is a selection
    task, not a generation task."""

    name = "brainstorm-synthesizer"
    metadata_model = None

    def __init__(
        self,
        *,
        model_name: str,
        system_prompt: str,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._temperature = temperature

    def build_agent(self) -> Agent[None, Any]:
        return Agent(
            model=build_openrouter_model(self._model_name),
            output_type=TodoIdea,
            system_prompt=self._system_prompt,
        )

    async def build_deps(
        self,
        *,
        thread: Thread,
        message: ModelMessage | None,
    ) -> None:
        del thread, message
        return None

    def model_settings(self) -> OpenRouterModelSettings:
        return default_model_settings(temperature=self._temperature)


__all__ = ["BrainstormDivergentAgent", "BrainstormSynthesizerAgent"]
