"""OpenRouter-backed pydantic-ai Agent for the notes app.

Iteration 3 wires CRUD tools into the agent — the agent now takes a
`NoteToolDeps` (a repo + a tenant id) and can `create_note`, `list_notes`,
`search_notes`, `edit_note`, and `delete_note` on the user's behalf.

The agent still streams a `ChatReply` (the prose the assistant says to the
user). Tool calls now also surface as canonical AG-UI ``TOOL_CALL_*``
events thanks to framework F13 (``make_runner`` v2): the frontend renders
a tool-call card per invocation as the model streams.

We keep `build_agent()` separate from `build_notes_runner()` so tests can
build the agent without standing up a runner, and the runner can be
constructed against a fake repo without touching OpenRouter.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.profiles import InlineDefsJsonSchemaTransformer, ModelProfile
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_stateflow.api.streaming import AgentRunner, make_runner

if TYPE_CHECKING:
    from notes_app.notes.repository import NoteRepository
    from notes_app.notes.tools import NoteToolDeps

DEFAULT_MODEL = "qwen/qwen3.6-plus"


# Force native JSON-schema mode for `output_type=BaseModel` agents. Without
# this, pydantic-ai's default is "tool" mode (synthetic final_result tool +
# tool_choice="required"), which OpenRouter's Qwen 3.6 endpoints reject —
# no provider serving qwen/qwen3.6-plus supports tool_choice="required".
# Alibaba's Qwen3.6 endpoints additionally require the literal word "json"
# in the instructions before accepting any response_format; the
# native_output_requires_schema_in_instructions flag inlines the prompted-
# output template (which says "JSON object") into the system prompt to
# satisfy that check.
_QWEN_3_6_NATIVE_PROFILE = ModelProfile(
    json_schema_transformer=InlineDefsJsonSchemaTransformer,
    ignore_streamed_leading_whitespace=True,
    supports_json_schema_output=True,
    supports_json_object_output=True,
    default_structured_output_mode="native",
    native_output_requires_schema_in_instructions=True,
)


def _profile_for(model_id: str) -> ModelProfile | None:
    """Return a profile override for model ids the upstream qwen profile misses.

    Returns None for ids pydantic-ai handles correctly — `OpenRouterModel`
    resolves the profile via its built-in registry in that case.
    """
    if model_id.startswith("qwen/qwen3.6") or model_id.startswith("qwen/qwen-3.6"):
        return _QWEN_3_6_NATIVE_PROFILE
    return None


SYSTEM_PROMPT = (
    "You are the assistant inside a personal notes app. "
    "You have tools to create, list, search, edit, and delete notes on "
    "the user's behalf. "
    "When the user asks you to create / find / change / remove a note, "
    "USE THE TOOLS to actually do it — do not just describe what you "
    "would do. After running the tools, briefly confirm what happened "
    "(e.g. 'Saved your note titled \"X\"'). "
    "If the user is chatting and not asking for a note action, just "
    "reply conversationally. "
    "Always wrap your reply in the ChatReply JSON object."
)


class ChatReply(BaseModel):
    """Structured envelope for the assistant's prose reply.

    Tool calls are dispatched by pydantic-ai before the final reply is
    produced; this object only carries the human-facing text. Iteration 3
    keeps the shape intentionally minimal — richer "result cards" (e.g.
    referencing a saved note's id) can grow on top of this in later
    iterations.
    """

    reply: str = Field(..., description="Plain-text reply to show to the user.")


def build_agent(
    *,
    model_name: str | None = None,
    api_key: str | None = None,
) -> Agent[NoteToolDeps, ChatReply]:
    """Build the OpenRouter-backed agent and register the note tools.

    Uses pydantic-ai's dedicated `OpenRouterModel` + `OpenRouterProvider`
    (NOT the generic `OpenAIModel` + `base_url` shim) so OpenRouter-specific
    behaviour (provider routing, app headers, tool_choice negotiation) is
    handled inside pydantic-ai instead of us re-inventing it.

    The returned agent has `deps_type=NoteToolDeps` — callers supply the
    repo + tenant_id via `agent.run_stream(..., deps=NoteToolDeps(...))`.
    Resolves `model_name` from `OPENROUTER_MODEL` env (default
    `qwen/qwen3.6-plus`) and `api_key` from `OPENROUTER_API_KEY`.
    """
    # Local import: tools.py imports ChatReply from this module, so the
    # symmetric import here would be a cycle at module load.
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
        profile=_profile_for(resolved_model),
    )
    agent: Agent[_Deps, ChatReply] = Agent(
        model=model,
        output_type=ChatReply,
        deps_type=_Deps,
        system_prompt=SYSTEM_PROMPT,
    )
    register_note_tools(agent)
    return agent


def build_notes_runner(
    agent: Agent[NoteToolDeps, ChatReply],
    note_repo: NoteRepository,
) -> AgentRunner:
    """Wrap `agent` as an `AgentRunner` that injects per-request `NoteToolDeps`.

    Uses the framework's ``make_runner(deps=factory)`` (F12): we pass a
    closure that mints a fresh ``NoteToolDeps(repo, tenant_id)`` per HTTP
    request from the runner-supplied ``tenant_id``. Tool-call SSE events
    (F13) flow through ``make_runner`` automatically — no app code needed.
    """
    # Local import to keep the agent module importable without sqlmodel
    # (e.g. pure unit tests of ChatReply schema).
    from notes_app.notes.tools import NoteToolDeps

    def deps_factory(*, tenant_id: UUID, **_kwargs: object) -> NoteToolDeps:
        return NoteToolDeps(repo=note_repo, tenant_id=tenant_id)

    return make_runner(agent, text_field="reply", deps=deps_factory)
