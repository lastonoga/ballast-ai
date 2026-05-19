"""Agent-facing CRUD tools over a `NoteRepository`.

Each function registered via `@agent.tool` becomes a tool the LLM can call.
The docstring is what the model sees in the tool catalog — keep it short,
imperative, and explicit about what the tool returns so the model can
chain follow-up actions.

The `NoteToolDeps` dataclass is the `ctx.deps` shape; it's supplied per
request by the runner in `main.py` (one tenant per HTTP request, one
shared in-memory repo for the process).

Closed-set ``note_id`` constraint: ``edit_note`` / ``delete_note`` declare
their ``note_id`` parameter as
``Annotated[Ref[Note], Selector(lambda c: c.deps.repo.list_(...))]``. The
framework's ``register_grounded_tools`` (in ``agent.build_agent``) reads
that metadata at run-time and installs the per-run ``prepare`` hook that
narrows ``note_id`` to a closed enum of real note IDs (and hides the tool
entirely when the user has zero notes). This replaces the hand-rolled
``_prepare_note_id_closed_set`` we used in iteration 3.

Note on annotations: we intentionally do NOT use
``from __future__ import annotations`` here. pydantic-ai's tool-registration
introspects parameter types via ``get_type_hints()`` at decoration time, and
postponed-evaluation annotations have bitten this codebase before (see
``project_pydantic_ai_api_quirks.md``).
"""

from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from pydantic_ai import Agent, RunContext
from pydantic_ai_stateflow.grounded import Ref, Selector

from notes_app.notes.domain import Note
from notes_app.notes.repository import NoteRepository


@dataclass
class NoteToolDeps:
    """Per-request dependencies for the note tools.

    `repo` is the shared in-memory store (iteration 3); `tenant_id` is
    the requesting tenant (derived from the `X-Tenant-Id` header).
    """

    repo: NoteRepository
    tenant_id: UUID


def register_note_tools(agent: Agent[NoteToolDeps, str]) -> None:
    """Register the CRUD tools on `agent`. Idempotent within one Agent.

    After calling this, the agent owner should also call
    ``pydantic_ai_stateflow.grounded.register_grounded_tools(agent)`` so
    the ``Annotated[Ref[Note], Selector(...)]`` on ``edit_note`` /
    ``delete_note`` becomes a per-run closed-set enum.
    """

    @agent.tool
    async def create_note(
        ctx: RunContext[NoteToolDeps], title: str, body: str,
    ) -> Note:
        """Create a new note with the given title and body.

        Returns the saved note (including its `id`, which you should use
        for any follow-up edit/delete in the same turn).
        """
        return await ctx.deps.repo.create(
            title=title, body=body, tenant_id=ctx.deps.tenant_id,
        )

    @agent.tool
    async def list_notes(
        ctx: RunContext[NoteToolDeps], limit: int = 20,
    ) -> list[Note]:
        """List the most recent notes for the current user, newest first.

        Use this when the user asks "show me my notes" or wants an
        overview. Returns at most `limit` notes (default 20).
        """
        return await ctx.deps.repo.list_(
            tenant_id=ctx.deps.tenant_id, limit=limit,
        )

    @agent.tool
    async def search_notes(
        ctx: RunContext[NoteToolDeps], query: str, limit: int = 20,
    ) -> list[Note]:
        """Search the user's notes by case-insensitive substring on title+body.

        Returns matching notes newest-first, at most `limit`. Use this
        when the user references a note by topic or keyword rather than id.
        """
        return await ctx.deps.repo.search(
            query, tenant_id=ctx.deps.tenant_id, limit=limit,
        )

    @agent.tool
    async def edit_note(
        ctx: RunContext[NoteToolDeps],
        note_id: Annotated[
            Ref[Note],
            Selector(lambda c: c.deps.repo.list_(
                tenant_id=c.deps.tenant_id, limit=1000,
            )),
        ],
        title: str | None = None,
        body: str | None = None,
    ) -> Note:
        """Edit an existing note. Pass only the fields you want to change.

        `note_id` is constrained at the schema level (via Selector) to
        the set of notes that currently exist for this user — you cannot
        fabricate one. Returns the updated note.
        """
        # The framework's Ref-aware tool wrapper passes the UUID through
        # to the function; if a Ref instance slipped in we accept it too.
        nid = note_id.id if isinstance(note_id, Ref) else note_id
        return await ctx.deps.repo.update(
            nid,
            title=title,
            body=body,
            tenant_id=ctx.deps.tenant_id,
        )

    @agent.tool(requires_approval=True)
    async def delete_note(
        ctx: RunContext[NoteToolDeps],
        note_id: Annotated[
            Ref[Note],
            Selector(lambda c: c.deps.repo.list_(
                tenant_id=c.deps.tenant_id, limit=1000,
            )),
        ],
    ) -> str:
        """Delete the note with the given id.

        REQUIRES USER APPROVAL — destructive, irreversible. The model
        proposes the call; pydantic-ai pauses the run and emits a
        deferred ``approval-requested`` part. The frontend renders an
        approve/cancel card; once the user clicks, the response round-
        trips back through ``VercelAIAdapter.deferred_tool_results`` and
        this body executes (or denial is fed back to the model).

        ``note_id`` is constrained at the schema level (via Selector) to
        the set of notes that currently exist for this user. Idempotent —
        safe to call twice. Returns a short confirmation.
        """
        nid = note_id.id if isinstance(note_id, Ref) else note_id
        await ctx.deps.repo.delete(nid, tenant_id=ctx.deps.tenant_id)
        return f"deleted {nid}"
