"""OpenRouter-backed ``StateflowAgent`` for the notes app.

One file = one agent. ``NotesAgent`` is the framework's per-thread agent
abstraction (see
``pydantic_ai_stateflow.runtime.agents.StateflowAgent``); the registry
binds ``Thread.agent == "notes"`` to this class.

Tools are declared inline with ``@NotesAgent.tool`` decorators at module
load — the framework registers them on the underlying pydantic-ai
``Agent`` automatically when ``self.agent`` is first accessed. Grounded
``Annotated[Ref[Note], Selector(...)]`` parameters get per-run ``prepare``
hooks installed automatically too; no explicit
``register_grounded_tools(...)`` call needed.

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

Note on annotations: we intentionally do NOT use
``from __future__ import annotations`` here. pydantic-ai's tool-
registration introspects parameter types via ``get_type_hints()`` at
decoration time, and postponed-evaluation annotations have bitten this
codebase before (see ``project_pydantic_ai_api_quirks.md``).
"""

import os
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai_stateflow.grounded import Ref, Selector
from pydantic_ai_stateflow.persistence.thread.domain import Thread
from pydantic_ai_stateflow.runtime import StateflowAgent

from notes_app.notes.domain import Note
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


@dataclass
class NoteToolDeps:
    """Per-request dependencies for the note tools."""

    repo: NoteRepository


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
        # pauses and yields a ``DeferredToolRequests`` instead of
        # looping forever over an unresolved tool call.
        return Agent(
            model=model,
            output_type=[str, DeferredToolRequests],
            deps_type=NoteToolDeps,
            system_prompt=SYSTEM_PROMPT,
        )

    async def build_deps(
        self,
        *,
        thread: Thread,
        message: ModelMessage | None,
    ) -> NoteToolDeps:
        del thread, message
        return NoteToolDeps(repo=self._notes_repo)

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


# ── Tools ────────────────────────────────────────────────────────────────────
#
# Declared at module load via ``@NotesAgent.tool``. The framework
# registers them on the underlying pydantic-ai ``Agent`` the first time
# ``NotesAgent.agent`` is accessed, and auto-installs grounded
# ``prepare`` hooks for any ``Annotated[Ref[T], Selector(...)]`` params.


@NotesAgent.tool
async def create_note(
    ctx: RunContext[NoteToolDeps], title: str, body: str,
) -> Note:
    """Create a new note with the given title and body.

    Returns the saved note (including its ``id``, which you should use
    for any follow-up edit/delete in the same turn).
    """
    return await ctx.deps.repo.create(title=title, body=body)


@NotesAgent.tool
async def list_notes(
    ctx: RunContext[NoteToolDeps], limit: int = 20,
) -> list[Note]:
    """List the most recent notes for the current user, newest first.

    Use this when the user asks "show me my notes" or wants an
    overview. Returns at most ``limit`` notes (default 20).
    """
    return await ctx.deps.repo.list_(limit=limit)


@NotesAgent.tool
async def search_notes(
    ctx: RunContext[NoteToolDeps], query: str, limit: int = 20,
) -> list[Note]:
    """Search the user's notes by case-insensitive substring on title+body.

    Returns matching notes newest-first, at most ``limit``. Use this
    when the user references a note by topic or keyword rather than id.
    """
    return await ctx.deps.repo.search(query, limit=limit)


@NotesAgent.tool
async def edit_note(
    ctx: RunContext[NoteToolDeps],
    note_id: Annotated[
        Ref[Note],
        Selector(lambda c: c.deps.repo.list_(limit=1000)),
    ],
    title: str | None = None,
    body: str | None = None,
) -> Note:
    """Edit an existing note. Pass only the fields you want to change.

    ``note_id`` is constrained at the schema level (via Selector) to
    the set of notes that currently exist for this user — you cannot
    fabricate one. Returns the updated note.
    """
    nid = note_id.id if isinstance(note_id, Ref) else note_id
    return await ctx.deps.repo.update(nid, title=title, body=body)


@NotesAgent.tool(requires_approval=True)
async def delete_note(
    ctx: RunContext[NoteToolDeps],
    note_id: Annotated[
        Ref[Note],
        Selector(lambda c: c.deps.repo.list_(limit=1000)),
    ],
) -> str:
    """Delete the note with the given id.

    REQUIRES USER APPROVAL — destructive, irreversible. The model
    proposes the call; pydantic-ai pauses the run and emits a deferred
    ``approval-requested`` part. The frontend renders an approve/cancel
    card; once the user clicks, the response round-trips back through
    ``VercelAIAdapter.deferred_tool_results`` and this body executes
    (or denial is fed back to the model).

    ``note_id`` is constrained at the schema level (via Selector) to
    the set of notes that currently exist for this user. Idempotent —
    safe to call twice. Returns a short confirmation.
    """
    nid = note_id.id if isinstance(note_id, Ref) else note_id
    await ctx.deps.repo.delete(nid)
    return f"deleted {nid}"
