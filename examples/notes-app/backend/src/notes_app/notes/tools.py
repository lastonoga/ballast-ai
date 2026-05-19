"""Agent-facing CRUD tools over a `NoteRepository`.

Each function registered via `@agent.tool` becomes a tool the LLM can call.
The docstring is what the model sees in the tool catalog — keep it short,
imperative, and explicit about what the tool returns so the model can
chain follow-up actions.

The `NoteToolDeps` dataclass is the `ctx.deps` shape; it's supplied per
request by the runner in `main.py` (one tenant per HTTP request, one
shared in-memory repo for the process).

Note on annotations: we intentionally do NOT use
`from __future__ import annotations` here. pydantic-ai's tool-registration
introspects parameter types via `get_type_hints()` at decoration time, and
postponed-evaluation annotations have bitten this codebase before (see
`project_pydantic_ai_api_quirks.md`). The annotations on these tools are
already concrete (str / UUID / int / Note), so eager evaluation is fine.
"""

from dataclasses import dataclass
from uuid import UUID

from pydantic_ai import Agent, RunContext

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
    """Register the CRUD tools on `agent`. Idempotent within one Agent."""

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
        note_id: UUID,
        title: str | None = None,
        body: str | None = None,
    ) -> Note:
        """Edit an existing note. Pass only the fields you want to change.

        Returns the updated note. Fails if `note_id` does not exist for
        the current user.
        """
        return await ctx.deps.repo.update(
            note_id,
            title=title,
            body=body,
            tenant_id=ctx.deps.tenant_id,
        )

    @agent.tool
    async def delete_note(
        ctx: RunContext[NoteToolDeps], note_id: UUID,
    ) -> str:
        """Delete the note with the given id. Idempotent — safe to call twice.

        Returns a short confirmation string. Use this when the user asks
        to remove or discard a note.
        """
        await ctx.deps.repo.delete(note_id, tenant_id=ctx.deps.tenant_id)
        return f"deleted {note_id}"
