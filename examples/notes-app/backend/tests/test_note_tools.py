"""Direct unit tests for the note tools + repo (no LLM involved).

These exercise the `NoteRepository` Protocol's contract through the
in-memory impl, and the tool functions' wiring of `ctx.deps`. They run
without `OPENROUTER_API_KEY`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from notes_app.notes.domain import Note
from notes_app.notes.repository import InMemoryNoteRepository
from notes_app.notes.tools import NoteToolDeps


@dataclass
class _FakeCtx:
    """Minimal RunContext stand-in for direct tool invocation in tests.

    pydantic-ai's `RunContext` is a heavier object, but the only attribute
    our tools read is `.deps`, so a dataclass is enough.
    """

    deps: NoteToolDeps


def _make_deps(repo: InMemoryNoteRepository, tenant_id: UUID) -> NoteToolDeps:
    return NoteToolDeps(repo=repo, tenant_id=tenant_id)


def _bound_tool(agent: Any, name: str) -> Any:
    """Pull a registered tool's wrapped Python function off an Agent."""
    tool = agent._function_toolset.tools[name]  # noqa: SLF001 — test introspection
    return tool.function


async def test_create_note_persists_via_repo(
    repo: InMemoryNoteRepository, tenant_id: UUID,
) -> None:
    """create_note round-trips through the repo and returns the saved Note."""
    deps = _make_deps(repo, tenant_id)
    ctx = _FakeCtx(deps=deps)

    # Call the tool's underlying coroutine directly. We resolve it via the
    # agent's registration to make sure register_note_tools wired it.
    from pydantic_ai import Agent

    from notes_app.notes.tools import register_note_tools

    agent = Agent(
        model="test",
        output_type=str,
        deps_type=NoteToolDeps,
    )
    register_note_tools(agent)
    create_note = _bound_tool(agent, "create_note")

    note = await create_note(ctx, title="grocery", body="milk, eggs")
    assert isinstance(note, Note)
    assert note.tenant_id == tenant_id
    assert note.title == "grocery"
    assert note.body == "milk, eggs"

    listed = await repo.list_(tenant_id=tenant_id)
    assert [n.id for n in listed] == [note.id]


async def test_search_notes_substring_match(
    repo: InMemoryNoteRepository, tenant_id: UUID,
) -> None:
    """search_notes finds case-insensitive substrings across title+body."""
    await repo.create(title="Groceries", body="milk", tenant_id=tenant_id)
    await repo.create(title="Reading", body="Finish the milkman book", tenant_id=tenant_id)
    await repo.create(title="Errands", body="post office", tenant_id=tenant_id)

    hits = await repo.search("MILK", tenant_id=tenant_id)
    titles = {n.title for n in hits}
    assert titles == {"Groceries", "Reading"}, hits

    # Empty query: no spurious matches.
    assert await repo.search("   ", tenant_id=tenant_id) == []


async def test_edit_note_raises_keyerror_for_wrong_tenant(
    repo: InMemoryNoteRepository, tenant_id: UUID,
) -> None:
    """update() refuses to touch another tenant's note (404-equivalent)."""
    note = await repo.create(title="mine", body="x", tenant_id=tenant_id)
    other_tenant = uuid4()

    with pytest.raises(KeyError):
        await repo.update(
            note.id, title="hacked", body=None, tenant_id=other_tenant,
        )

    # Unknown ids also raise even for the right tenant.
    with pytest.raises(KeyError):
        await repo.update(
            uuid4(), title="x", body=None, tenant_id=tenant_id,
        )


async def test_delete_note_idempotent(
    repo: InMemoryNoteRepository, tenant_id: UUID,
) -> None:
    """delete() silently succeeds on unknown / wrong-tenant ids."""
    note = await repo.create(title="t", body="b", tenant_id=tenant_id)

    await repo.delete(note.id, tenant_id=tenant_id)
    assert await repo.get(note.id, tenant_id=tenant_id) is None

    # Second delete on the same id: no-op.
    await repo.delete(note.id, tenant_id=tenant_id)

    # Wrong tenant: no-op.
    other = await repo.create(title="other", body="x", tenant_id=tenant_id)
    await repo.delete(other.id, tenant_id=uuid4())
    assert await repo.get(other.id, tenant_id=tenant_id) is not None


async def test_list_notes_is_tenant_scoped(
    repo: InMemoryNoteRepository,
) -> None:
    """Notes from one tenant never leak into another tenant's list/search."""
    t1, t2 = uuid4(), uuid4()
    await repo.create(title="t1-note", body="x", tenant_id=t1)
    await repo.create(title="t2-note", body="x", tenant_id=t2)

    assert [n.title for n in await repo.list_(tenant_id=t1)] == ["t1-note"]
    assert [n.title for n in await repo.list_(tenant_id=t2)] == ["t2-note"]
    assert await repo.search("note", tenant_id=t1) == await repo.list_(tenant_id=t1)
