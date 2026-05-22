"""Tests for the durable ``propose_todo`` flow.

The new architecture is fire-and-forget from the parent run's POV:

  1. ``propose_todo`` spawns a helper thread + opening message.
  2. Kicks off ``TodoApprovalFlow.run`` via
     ``DBOS.start_workflow_async`` with a pre-allocated workflow_id.
  3. Returns IMMEDIATELY with an "I opened a side conversation" string.
  4. The helper agent's approve / modify / reject tools DBOS.send their
     response to the workflow's id (stored in T2 metadata).
  5. The workflow saves the note (or skips) AND posts a notification
     message back to the parent thread.

Crucially, step (5) doesn't depend on T1's request handler being alive
— that's the whole point of the refactor. So these tests exercise the
durable workflow end-to-end via DBOS.send + handle.get_result.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from pydantic_ai_stateflow.durable import Durable
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_stateflow.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.runtime.event_stream import InProcessEventStream
from pydantic_ai_stateflow.runtime.infra import RunContext

from notes_app.agents.notes import NotesAgent, NoteToolDeps
from notes_app.agents.todo_approval import (
    NotesTodoApprovalAgent,
    TodoApprovalDeps,
)
from notes_app.models.todo_approval import TodoApprovalContext
from notes_app.repositories.note import InMemoryNoteRepository
from notes_app.workflows.todo_approval import TodoApprovalFlow

# ── Shared helpers ───────────────────────────────────────────────────────────


@dataclass
class _FakeCtx:
    deps: Any


class _TestNotesAgent(NotesAgent):
    """``NotesAgent`` with a TestModel-backed ``build_agent`` (no API key)."""

    def build_agent(self) -> Agent[NoteToolDeps, str]:
        return Agent(
            TestModel(custom_output_text="ok"),
            output_type=str,
            deps_type=NoteToolDeps,
        )


class _TestTodoApprovalAgent(NotesTodoApprovalAgent):
    """``NotesTodoApprovalAgent`` with a TestModel-backed ``build_agent``."""

    def build_agent(self) -> Agent[TodoApprovalDeps, str]:
        return Agent(
            TestModel(custom_output_text="ok"),
            output_type=str,
            deps_type=TodoApprovalDeps,
        )


def _bound_tool(agent: Any, name: str) -> Any:
    """Lift a registered tool's bare function off a pydantic-ai Agent."""
    return agent._function_toolset.tools[name].function  # noqa: SLF001


async def _wait_for_helper_thread(
    thread_repo: InMemoryThreadRepository,
    *,
    timeout_s: float = 2.0,
) -> Any:
    """Poll the thread repo until the helper thread appears."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        threads = await thread_repo.list_()
        helpers = [t for t in threads if t.agent == "todo_approval"]
        if helpers:
            return helpers[0]
        await asyncio.sleep(0.02)
    raise AssertionError("Timed out waiting for the helper thread to appear")


async def _trigger_approval(
    *,
    helper_tool: str,
    t2_metadata: dict[str, Any],
    notes_repo: InMemoryNoteRepository,
    **tool_kwargs: Any,
) -> str:
    """Invoke a helper-agent approval tool with the right deps shape.

    Mirrors what ``NotesTodoApprovalAgent.build_deps`` would mint when
    the streaming router runs the helper agent against T2's metadata.
    """
    approval_agent = _TestTodoApprovalAgent()
    fn = _bound_tool(approval_agent.agent, helper_tool)
    deps = TodoApprovalDeps(
        notes_repo=notes_repo,
        request_id=UUID(t2_metadata["request_id"]),
        workflow_id=str(t2_metadata["workflow_id"]),
        metadata=TodoApprovalContext.model_validate(t2_metadata),
    )
    return await fn(_FakeCtx(deps=deps), **tool_kwargs)


async def _spawn_proposal(
    *,
    title: str,
    body: str,
    notes_repo: InMemoryNoteRepository,
    thread_repo: InMemoryThreadRepository,
    todo_flow: TodoApprovalFlow,
) -> tuple[str, Any, dict[str, Any]]:
    """Run ``propose_todo`` and return (return_value, t1, t2_metadata).

    ``todo_flow`` and ``notes_repo`` are reached by the tool via direct
    import of the module-level singletons — tests swap the singleton
    refs via ``monkeypatch.setattr`` in the calling test (the fixture
    receives the per-test repo / flow that way).
    """
    notes_agent = _TestNotesAgent(config_name=f"_TestNotesAgent-{uuid4()}")
    propose_todo = _bound_tool(notes_agent.agent, "propose_todo")

    t1 = await thread_repo.create(agent="notes", metadata={})
    ctx = RunContext(
        thread_repo=thread_repo,
        event_log=InMemoryEventLogRepository(),
        event_stream=InProcessEventStream(),
        parent_thread_id=t1.id,
    )
    deps = NoteToolDeps(
        repo=notes_repo,
        parent_thread_id=t1.id,
        ctx=ctx,
    )
    result = await propose_todo(_FakeCtx(deps=deps), title=title, body=body)

    t2 = await _wait_for_helper_thread(thread_repo)
    return result, t1, dict(t2.metadata_)


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_propose_todo_spawns_helper_thread_and_workflow(
    fresh_dbos_executor: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """propose_todo creates T2 with the right metadata + opening message."""
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    flow = TodoApprovalFlow(
        config_name=f"todo-flow-test-{uuid4()}",
    )
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", notes_repo)
    monkeypatch.setattr("notes_app.workflows.todo_approval.todo_flow", flow)

    result, t1, t2_meta = await _spawn_proposal(
        title="groceries", body="milk eggs",
        notes_repo=notes_repo, thread_repo=thread_repo, todo_flow=flow,
    )

    # Tool returns an "I opened a side conversation" string — NOT a Note.
    assert "opened" in result.lower() or "confirmation" in result.lower()

    # T2 metadata carries the context + framework routing keys.
    assert t2_meta["proposed_title"] == "groceries"
    assert t2_meta["proposed_body"] == "milk eggs"
    assert t2_meta["parent_thread_id"] == str(t1.id)
    UUID(t2_meta["request_id"])  # well-formed UUID
    UUID(t2_meta["workflow_id"])  # well-formed UUID

    # Opening assistant message is seeded.
    threads = await thread_repo.list_()
    t2 = next(t for t in threads if t.agent == "todo_approval")
    history = await thread_repo.history(t2.id)
    assert len(history) == 1
    assert history[0].role == "assistant"
    assert "groceries" in history[0].parts[0]["text"]


@pytest.mark.asyncio
async def test_approve_saves_note_and_notifies_parent(
    fresh_dbos_executor: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: approve → note saved + 'Saved your todo' on T1."""
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    flow = TodoApprovalFlow(
        config_name=f"todo-flow-test-{uuid4()}",
    )
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", notes_repo)
    monkeypatch.setattr("notes_app.workflows.todo_approval.todo_flow", flow)

    _, t1, t2_meta = await _spawn_proposal(
        title="groceries", body="milk eggs",
        notes_repo=notes_repo, thread_repo=thread_repo, todo_flow=flow,
    )
    workflow_id = t2_meta["workflow_id"]

    await _trigger_approval(
        helper_tool="approve",
        t2_metadata=t2_meta,
        notes_repo=notes_repo,
    )

    # Wait for the durable workflow to consume the DBOS message and
    # finish its work (save + notify). ``get_result()`` blocks until the
    # workflow completes (or raises).
    handle = await Durable.retrieve_workflow(workflow_id)
    await handle.get_result()

    listed = await notes_repo.list_()
    assert [n.title for n in listed] == ["groceries"]
    assert listed[0].body == "milk eggs"

    history = await thread_repo.history(t1.id)
    assistant_texts = [
        m.parts[0]["text"] for m in history if m.role == "assistant"
    ]
    assert any("Saved" in t and "groceries" in t for t in assistant_texts)


@pytest.mark.asyncio
async def test_modify_saves_note_with_overrides(
    fresh_dbos_executor: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modify branch: overrides applied, note saved, parent notified."""
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    flow = TodoApprovalFlow(
        config_name=f"todo-flow-test-{uuid4()}",
    )
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", notes_repo)
    monkeypatch.setattr("notes_app.workflows.todo_approval.todo_flow", flow)

    _, t1, t2_meta = await _spawn_proposal(
        title="groceries", body="milk",
        notes_repo=notes_repo, thread_repo=thread_repo, todo_flow=flow,
    )
    workflow_id = t2_meta["workflow_id"]

    await _trigger_approval(
        helper_tool="modify",
        t2_metadata=t2_meta,
        notes_repo=notes_repo,
        new_title="weekly groceries",
        new_body="milk, eggs, bread",
    )

    handle = await Durable.retrieve_workflow(workflow_id)
    await handle.get_result()

    listed = await notes_repo.list_()
    assert [n.title for n in listed] == ["weekly groceries"]
    assert listed[0].body == "milk, eggs, bread"

    history = await thread_repo.history(t1.id)
    assistant_texts = [
        m.parts[0]["text"] for m in history if m.role == "assistant"
    ]
    assert any("weekly groceries" in t for t in assistant_texts)


@pytest.mark.asyncio
async def test_reject_skips_note_and_notifies_parent(
    fresh_dbos_executor: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject branch: no note saved, parent gets a cancellation message."""
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    flow = TodoApprovalFlow(
        config_name=f"todo-flow-test-{uuid4()}",
    )
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", notes_repo)
    monkeypatch.setattr("notes_app.workflows.todo_approval.todo_flow", flow)

    _, t1, t2_meta = await _spawn_proposal(
        title="garbage", body="trash",
        notes_repo=notes_repo, thread_repo=thread_repo, todo_flow=flow,
    )
    workflow_id = t2_meta["workflow_id"]

    await _trigger_approval(
        helper_tool="reject",
        t2_metadata=t2_meta,
        notes_repo=notes_repo,
        reason="too vague",
    )

    handle = await Durable.retrieve_workflow(workflow_id)
    await handle.get_result()

    assert await notes_repo.list_() == []

    history = await thread_repo.history(t1.id)
    assistant_texts = [
        m.parts[0]["text"] for m in history if m.role == "assistant"
    ]
    assert any("cancelled" in t.lower() for t in assistant_texts)
    assert any("too vague" in t.lower() for t in assistant_texts)


@pytest.mark.asyncio
async def test_propose_todo_returns_before_helper_decision(
    fresh_dbos_executor: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: propose_todo must NOT block on the decision.

    If propose_todo blocked, this test would hang (nobody fires the
    helper tool). It must return immediately with the workflow running
    in the background.
    """
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    flow = TodoApprovalFlow(
        config_name=f"todo-flow-test-{uuid4()}",
    )
    monkeypatch.setattr("notes_app.repositories.note.notes_repo", notes_repo)
    monkeypatch.setattr("notes_app.workflows.todo_approval.todo_flow", flow)

    # If this await hung, the test would time out — we explicitly assert
    # it resolves quickly.
    result, _, t2_meta = await asyncio.wait_for(
        _spawn_proposal(
            title="x", body="y",
            notes_repo=notes_repo, thread_repo=thread_repo, todo_flow=flow,
        ),
        timeout=2.0,
    )
    assert result  # non-empty string

    # Clean up: fire reject so the dangling workflow finishes (otherwise
    # the test DBOS fixture has a lingering pending workflow).
    await _trigger_approval(
        helper_tool="reject",
        t2_metadata=t2_meta,
        notes_repo=notes_repo,
    )
    handle = await Durable.retrieve_workflow(t2_meta["workflow_id"])
    await handle.get_result()


@pytest.mark.asyncio
async def test_propose_todo_rejects_when_deps_missing_ctx() -> None:
    """Calling ``propose_todo`` without parent_thread_id / ctx must fail loudly."""
    notes_repo = InMemoryNoteRepository()
    notes_agent = _TestNotesAgent(config_name=f"_TestNotesAgent-{uuid4()}")
    propose_todo = _bound_tool(notes_agent.agent, "propose_todo")

    deps = NoteToolDeps(repo=notes_repo)  # no parent_thread_id / ctx
    with pytest.raises(RuntimeError, match="propose_todo requires"):
        await propose_todo(_FakeCtx(deps=deps), title="x", body="y")
