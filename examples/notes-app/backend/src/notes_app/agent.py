"""OpenRouter-backed pydantic-ai Agent for the notes app.

Iteration 3 wires CRUD tools into the agent — the agent now takes a
`NoteToolDeps` (a repo + a tenant id) and can `create_note`, `list_notes`,
`search_notes`, `edit_note`, and `delete_note` on the user's behalf.

The agent still streams a `ChatReply` (the prose the assistant says to the
user). Tool calls happen "underneath" — pydantic-ai dispatches them on the
backend, the model sees the results, then it produces the final prose
reply. The frontend sees the same canonical AG-UI event sequence as
iteration 2.

We keep `build_agent()` separate from `build_notes_runner()` so tests can
build the agent without standing up a runner, and the runner can be
constructed against a fake repo without touching OpenRouter.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_stateflow.api.streaming import AgentRunner, StreamEvent
from pydantic_ai_stateflow.api.streaming.router import (
    _PostMessageBody,
    extract_text,
)

if TYPE_CHECKING:
    from notes_app.notes.repository import NoteRepository
    from notes_app.notes.tools import NoteToolDeps

DEFAULT_MODEL = "qwen/qwen3.6-plus"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
    base_url: str = OPENROUTER_BASE_URL,
) -> Agent[NoteToolDeps, ChatReply]:
    """Build the OpenRouter-backed agent and register the note tools.

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

    provider = OpenAIProvider(base_url=base_url, api_key=resolved_key)
    model = OpenAIModel(resolved_model, provider=provider)
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

    Why not just `make_runner(agent, deps=...)`?

    `make_runner` accepts a static `deps` value at *build* time, but our
    `NoteToolDeps` is per-request (one `tenant_id` per HTTP call). The
    framework helper would need a `deps_factory: Callable[[...], Any]`
    knob to construct deps per stream — recorded as a gap in RETRO.md.

    For iteration 3 we take option (b) from the iteration plan: call
    `agent.run_stream(...)` directly here, mirroring `make_runner`'s diff
    + emit loop, but constructing fresh deps per call. When the framework
    grows a `deps_factory` knob we can switch back to `make_runner`.
    """
    # Local import to keep the agent module importable without sqlmodel
    # (e.g. pure unit tests of ChatReply schema).
    from notes_app.notes.tools import NoteToolDeps

    async def _runner(
        *,
        thread_id: UUID,
        run_id: UUID,
        message: _PostMessageBody,
        tenant_id: UUID,
    ) -> AsyncIterator[StreamEvent]:
        deps = NoteToolDeps(repo=note_repo, tenant_id=tenant_id)
        message_id = uuid4()
        prompt = extract_text(message.parts)

        yield StreamEvent.run_started(thread_id=thread_id, run_id=run_id)
        yield StreamEvent.text_message_start(message_id=message_id)

        last_emitted = ""
        try:
            async with agent.run_stream(prompt, deps=deps) as result:
                async for snapshot in result.stream_output(debounce_by=0.05):
                    current = getattr(snapshot, "reply", "") or ""
                    if not current or current == last_emitted:
                        continue
                    if current.startswith(last_emitted):
                        delta = current[len(last_emitted):]
                    else:
                        # Partial-validation revised the prefix; fall back
                        # to full re-emit (the client treats deltas as
                        # appends — never emit a negative diff).
                        delta = current
                    last_emitted = current
                    if delta:
                        yield StreamEvent.text_message_content(
                            message_id=message_id, delta=delta,
                        )
        except Exception as exc:  # noqa: BLE001 — surface + re-raise
            yield StreamEvent.run_error(message=str(exc))
            raise

        yield StreamEvent.text_message_end(message_id=message_id)
        yield StreamEvent.run_finished(thread_id=thread_id, run_id=run_id)

    return _runner
