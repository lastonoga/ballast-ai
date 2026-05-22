"""Direct unit tests for the note tools + repo (no LLM involved)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from notes_app.agents.notes import NotesAgent, NoteToolDeps
from notes_app.models.note import Note
from notes_app.repositories.note import InMemoryNoteRepository


@dataclass
class _FakeCtx:
    deps: NoteToolDeps


class _TestNotesAgent(NotesAgent):
    """``NotesAgent`` with a TestModel-backed ``build_agent`` — no API key."""

    def build_agent(self) -> Agent[NoteToolDeps, str]:
        return Agent(
            TestModel(custom_output_text="ok"),
            output_type=str,
            deps_type=NoteToolDeps,
        )


def _make_deps(repo: InMemoryNoteRepository) -> NoteToolDeps:
    return NoteToolDeps(repo=repo)


def _agent_with_tools(repo: InMemoryNoteRepository) -> Agent[NoteToolDeps, Any]:
    # Repo plumbing now lives on the per-call ``NoteToolDeps.repo`` —
    # nothing constructor-injected on the agent for the repo. The tests
    # pass ``repo`` through ``_make_deps`` directly.
    del repo
    return _TestNotesAgent().agent


def _bound_tool(agent: Any, name: str) -> Any:
    tool = agent._function_toolset.tools[name]  # noqa: SLF001
    return tool.function


async def test_create_note_persists_via_repo(
    repo: InMemoryNoteRepository,
) -> None:
    deps = _make_deps(repo)
    ctx = _FakeCtx(deps=deps)

    agent = _agent_with_tools(repo)
    create_note = _bound_tool(agent, "create_note")

    note = await create_note(ctx, title="grocery", body="milk, eggs")
    assert isinstance(note, Note)
    assert note.title == "grocery"
    assert note.body == "milk, eggs"

    listed = await repo.list_()
    assert [n.id for n in listed] == [note.id]


async def test_search_notes_substring_match(
    repo: InMemoryNoteRepository,
) -> None:
    await repo.create(title="Groceries", body="milk")
    await repo.create(title="Reading", body="Finish the milkman book")
    await repo.create(title="Errands", body="post office")

    hits = await repo.search("MILK")
    titles = {n.title for n in hits}
    assert titles == {"Groceries", "Reading"}, hits

    # Empty query: no spurious matches.
    assert await repo.search("   ") == []


async def test_edit_note_raises_keyerror_for_unknown_id(
    repo: InMemoryNoteRepository,
) -> None:
    """update() raises on unknown ids."""
    with pytest.raises(KeyError):
        await repo.update(uuid4(), title="x", body=None)


async def test_delete_note_idempotent(
    repo: InMemoryNoteRepository,
) -> None:
    """delete() silently succeeds on unknown ids."""
    note = await repo.create(title="t", body="b")

    await repo.delete(note.id)
    assert await repo.get(note.id) is None

    # Second delete on the same id: no-op.
    await repo.delete(note.id)


async def test_grounded_prepare_narrows_delete_note_to_real_ids(
    repo: InMemoryNoteRepository,
) -> None:
    """``Annotated[Ref[Note], Selector(...)]`` -> JSON-schema enum."""
    agent = _agent_with_tools(repo)

    n = await repo.create(title="g", body="b")
    deps = _make_deps(repo)
    ctx = _FakeCtx(deps=deps)

    delete = agent._function_toolset.tools["delete_note"]  # noqa: SLF001
    assert delete.prepare is not None
    new_def = await delete.prepare(ctx, delete.tool_def)  # type: ignore[arg-type, misc]
    assert new_def is not None
    assert new_def.parameters_json_schema["properties"]["note_id"]["enum"] == [str(n.id)]


async def test_grounded_prepare_hides_delete_note_when_empty(
    repo: InMemoryNoteRepository,
) -> None:
    """Empty repo -> tool hidden so the model can't fabricate an id."""
    agent = _agent_with_tools(repo)

    deps = _make_deps(repo)
    ctx = _FakeCtx(deps=deps)

    delete = agent._function_toolset.tools["delete_note"]  # noqa: SLF001
    assert delete.prepare is not None
    assert (await delete.prepare(ctx, delete.tool_def)) is None  # type: ignore[arg-type, misc]
