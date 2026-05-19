"""OpenRouter-backed pydantic-ai Agent for the notes app.

Iteration 3 wires CRUD tools into the agent — the agent now takes a
``NoteToolDeps`` (a repo + a tenant id) and can ``create_note``,
``list_notes``, ``search_notes``, ``edit_note``, and ``delete_note`` on
the user's behalf. Tool calls surface as canonical AG-UI ``TOOL_CALL_*``
events thanks to framework F13 (``make_runner`` v2): the frontend
renders a tool-call card per invocation as the model streams.

Output shape decision (iter 3 round 2):
  We use ``output_type=str`` — NOT a structured ``BaseModel`` envelope.
  Reasons:

  - ``ToolOutput`` (pydantic-ai's default for ``BaseModel``) forces
    ``tool_choice="required"`` to drive the synthetic ``final_result``
    tool. OpenRouter's Qwen 3.6 endpoints reject that value (404 on
    every provider serving ``qwen/qwen3.6-plus``).
  - ``NativeOutput`` accepts ``response_format: json_schema`` but the
    model then returns the JSON directly WITHOUT calling real tools —
    so ``create_note`` never fires and the note is never saved.
  - ``PromptedOutput`` works but currently streams the raw JSON envelope
    through ``make_runner``'s text path, polluting the persisted
    assistant message.

  ``output_type=str`` sidesteps all three: tools fire freely
  (``tool_choice`` stays default ``auto``), the model returns a plain
  string reply, and the streaming text is exactly that reply.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_stateflow.api.streaming import AgentRunner, make_runner

if TYPE_CHECKING:
    from notes_app.notes.repository import NoteRepository
    from notes_app.notes.tools import NoteToolDeps

DEFAULT_MODEL = "qwen/qwen3.6-plus"

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


def build_agent(
    *,
    model_name: str | None = None,
    api_key: str | None = None,
) -> Agent[NoteToolDeps, str]:
    """Build the OpenRouter-backed agent and register the note tools.

    The returned agent has ``deps_type=NoteToolDeps`` — callers supply
    the repo + tenant_id via ``agent.run_stream(..., deps=...)``.
    Resolves ``model_name`` from ``OPENROUTER_MODEL`` env (default
    ``qwen/qwen3.6-plus``) and ``api_key`` from ``OPENROUTER_API_KEY``.
    """
    from notes_app.notes.tools import NoteToolDeps as _Deps
    from notes_app.notes.tools import register_note_tools

    resolved_model = model_name or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY env var is required to build the agent"
        )

    model = OpenRouterModel(
        resolved_model,
        provider=OpenRouterProvider(api_key=resolved_key),
    )
    agent: Agent[_Deps, str] = Agent(
        model=model,
        output_type=str,
        deps_type=_Deps,
        system_prompt=SYSTEM_PROMPT,
    )
    register_note_tools(agent)
    return agent


def build_notes_runner(
    agent: Agent[NoteToolDeps, str],
    note_repo: NoteRepository,
) -> AgentRunner:
    """Wrap ``agent`` as an ``AgentRunner`` that injects per-request deps.

    Uses ``make_runner(deps=factory)`` (F12): mints a fresh
    ``NoteToolDeps(repo, tenant_id)`` per HTTP request from the runner-
    supplied ``tenant_id``. Tool-call SSE events (F13) flow through
    automatically.

    ``text_field`` is the identity — agent output IS the reply string.
    """
    from notes_app.notes.tools import NoteToolDeps

    def deps_factory(*, tenant_id: UUID, **_kwargs: object) -> NoteToolDeps:
        return NoteToolDeps(repo=note_repo, tenant_id=tenant_id)

    return make_runner(
        agent, text_field=lambda out: out or "", deps=deps_factory,
    )
