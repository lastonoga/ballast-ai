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
from dataclasses import replace as dataclasses_replace
from uuid import UUID

from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition

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


async def _prepare_note_id_closed_set(
    ctx: RunContext[NoteToolDeps],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    """Per-run `prepare` hook that pins `note_id` to a closed set of UUIDs.

    Fetches the current tenant's notes from the repo and replaces the
    raw-UUID JSON schema for the ``note_id`` parameter with an ``enum``
    of just the IDs that actually exist right now. The model literally
    cannot pass a fabricated UUID — provider-side JSON Schema validation
    (OpenRouter, OpenAI, Anthropic) rejects values outside the enum.

    Side effect: when the user has zero notes, the tool is HIDDEN from
    the catalog (returning ``None``). That forces the model to either
    `list_notes` (and discover there are none) or `create_note` first,
    rather than calling edit/delete against a phantom id.

    This is the iter-3 example-side stand-in for the framework's
    promised ``Ref[Note]`` (spec SP1 GroundedSchema) — same goal (zero
    hallucination on closed-set references) without the full
    hydration / dynamic-schema machinery yet.
    """
    notes = await ctx.deps.repo.list_(
        tenant_id=ctx.deps.tenant_id, limit=1000,
    )
    if not notes:
        return None  # hide tool when there's nothing to reference

    ids = [str(n.id) for n in notes]
    # Build a human-readable enum description so the model can pick the
    # right id from context (title preview) without a second tool call.
    id_to_title = {str(n.id): n.title for n in notes}
    enum_desc = "; ".join(
        f"{nid} = {id_to_title[nid]!r}" for nid in ids[:20]
    )

    props = dict(tool_def.parameters_json_schema.get("properties", {}))
    note_id_schema = dict(props.get("note_id", {}))
    note_id_schema["enum"] = ids
    note_id_schema["description"] = (
        f"ID of an existing note. MUST be one of: {enum_desc}"
    )
    props["note_id"] = note_id_schema

    new_schema = dict(tool_def.parameters_json_schema)
    new_schema["properties"] = props

    return dataclasses_replace(tool_def, parameters_json_schema=new_schema)


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

    @agent.tool(prepare=_prepare_note_id_closed_set)
    async def edit_note(
        ctx: RunContext[NoteToolDeps],
        note_id: UUID,
        title: str | None = None,
        body: str | None = None,
    ) -> Note:
        """Edit an existing note. Pass only the fields you want to change.

        `note_id` is constrained at the schema level to the set of notes
        that currently exist for this user — you cannot fabricate one.
        Returns the updated note.
        """
        return await ctx.deps.repo.update(
            note_id,
            title=title,
            body=body,
            tenant_id=ctx.deps.tenant_id,
        )

    @agent.tool(prepare=_prepare_note_id_closed_set)
    async def delete_note(
        ctx: RunContext[NoteToolDeps], note_id: UUID,
    ) -> str:
        """Delete the note with the given id. Idempotent — safe to call twice.

        `note_id` is constrained at the schema level to the set of notes
        that currently exist for this user. Returns a short confirmation.
        """
        await ctx.deps.repo.delete(note_id, tenant_id=ctx.deps.tenant_id)
        return f"deleted {note_id}"
