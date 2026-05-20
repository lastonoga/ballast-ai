"""OpenRouter-backed ``StateflowAgent`` for the notes app.

``NotesAgent`` is the framework's per-thread agent abstraction
(see ``pydantic_ai_stateflow.runtime.agents.StateflowAgent``). The
registry binds ``Thread.agent == "notes"`` to this class — the
streaming router resolves it per request and uses
``self.agent`` (lazy-cached pydantic-ai ``Agent``), ``build_deps(...)``,
and ``model_settings()`` to drive the run.

Output shape decision (iter 3 round 2, still relevant):
  We use ``output_type=[str, DeferredToolRequests]`` — NOT a structured
  ``BaseModel`` envelope. Reasons:

  - ``ToolOutput`` (pydantic-ai's default for ``BaseModel``) forces
    ``tool_choice="required"`` to drive the synthetic ``final_result``
    tool. OpenRouter's Qwen 3.6 endpoints reject that value.
  - ``NativeOutput`` accepts ``response_format: json_schema`` but the
    model then returns the JSON directly WITHOUT calling real tools.
  - ``PromptedOutput`` works but pollutes the streamed text with JSON.

  Plain ``str`` (plus ``DeferredToolRequests`` for the approval branch)
  sidesteps all three.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_stateflow.grounded import register_grounded_tools
from pydantic_ai_stateflow.runtime import StateflowAgent

from notes_app.notes.tools import NoteToolDeps, register_note_tools

if TYPE_CHECKING:
    from uuid import UUID

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai_stateflow.persistence.thread.domain import Thread

    from notes_app.notes.repository import NoteRepository

DEFAULT_MODEL = "qwen/qwen3.6-plus"
DEFAULT_TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are the assistant inside a personal notes app. "
    "You have tools to create, list, search, edit, and delete notes on "
    "the user's behalf. "
    "When the user asks you to create / find / change / remove a note, "
    "USE THE TOOLS to actually do it — do not just describe what you "
    "would do. After running the tools, briefly confirm what happened "
    "(e.g. 'Saved your note titled \"X\"'). "
    "If the user is chatting and not asking for a note action, just "
    "reply conversationally."
)


class NotesAgent(StateflowAgent):
    """Personal-notes ``StateflowAgent``.

    Carries a ``NoteRepository`` so ``build_deps`` can mint a fresh
    ``NoteToolDeps`` per request, scoped to the requesting tenant.
    """

    name = "notes"
    metadata_model = None  # no per-thread settings yet

    def __init__(
        self,
        *,
        notes_repo: NoteRepository,
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._notes_repo = notes_repo
        self._model_name = model_name
        self._api_key = api_key

    def build_agent(self) -> Agent[NoteToolDeps, Any]:
        resolved_model = self._model_name or os.environ.get(
            "OPENROUTER_MODEL", DEFAULT_MODEL,
        )
        resolved_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY env var is required to build NotesAgent",
            )

        model = OpenRouterModel(
            resolved_model,
            provider=OpenRouterProvider(api_key=resolved_key),
        )

        # ``output_type=[str, DeferredToolRequests]`` opts into pydantic-ai's
        # deferred-tools branch: when the model calls a tool marked
        # ``requires_approval=True`` (e.g. ``delete_note``) the agent
        # pauses and yields a ``DeferredToolRequests`` instead of looping
        # forever over an unresolved tool call. ``VercelAIAdapter`` knows
        # how to serialize the deferred call as a ``tool-approval-request``
        # chunk and to thread approval responses back through
        # ``deferred_tool_results`` on the next round-trip.
        agent: Agent[NoteToolDeps, Any] = Agent(
            model=model,
            output_type=[str, DeferredToolRequests],
            deps_type=NoteToolDeps,
            system_prompt=SYSTEM_PROMPT,
        )
        register_note_tools(agent)
        # Install per-run ``prepare`` hooks on tools whose params are
        # ``Annotated[Ref[T], Selector(...)]``.
        register_grounded_tools(agent)
        return agent

    async def build_deps(
        self,
        *,
        thread: Thread,
        tenant_id: UUID,
        message: ModelMessage | None,
    ) -> NoteToolDeps:
        del thread, message  # unused — notes deps are tenant-scoped only
        return NoteToolDeps(repo=self._notes_repo, tenant_id=tenant_id)

    def model_settings(self) -> OpenRouterModelSettings:
        """Hardcoded OpenRouter settings for the notes-app demo.

        The Alibaba-upstream ``content: null`` rejection (see
        ``KNOWN_BUGS.md`` B9) is fixed at the framework layer via
        ``AssistantMessageNormalizer`` — apps don't need to route
        around it here.
        """
        return OpenRouterModelSettings(
            temperature=DEFAULT_TEMPERATURE,
            openrouter_reasoning={"effort": "none"},
            openrouter_usage={"include": True},
        )
