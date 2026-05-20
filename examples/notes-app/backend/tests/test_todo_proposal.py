"""Tests for the HITL-gated ``propose_todo`` tool + the approval agent.

End-to-end SSE-over-DBOS testing is hard to make reliable — the
streaming router runs the agent loop in a request context AND the
HITLGate.run workflow blocks on ``DBOS.recv``. Driving two concurrent
SSE streams while DBOS sits in the middle is timing-fragile in tests
even though it works in prod.

So we take the pragmatic unit-level path the architecture brief
endorses:

  1. Instantiate the deps that ``propose_todo`` would receive from
     ``NotesAgent.build_deps``.
  2. Run ``propose_todo(ctx, title, body)`` as a coroutine.
  3. In parallel, wait until the HITL repo shows the pending request
     (i.e. the gate has called ``persist_request`` and is now blocked
     on ``DBOS.recv``), then invoke the matching ``approve`` /
     ``reject`` / ``modify`` tool on the approval agent.
  4. ``propose_todo`` should unblock and resolve with the right
     behaviour (note created, RuntimeError, or note created with
     modified payload).

This is the same DBOS.send/recv loop the prod SSE flow uses — just
without the SSE router and frontend.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_stateflow.patterns.hitl import (
    AllowAll,
    HITLGate,
    UIChannel,
)
from pydantic_ai_stateflow.persistence import (
    InMemoryHITLRepository,
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.persistence.hitl.domain import (
    BlockingRequirement,
    BlockingRequirementStatus,
)

from notes_app.agent import NotesAgent, NoteToolDeps
from notes_app.notes.domain import Note
from notes_app.notes.repository import InMemoryNoteRepository
from notes_app.todo_approval_agent import (
    NotesTodoApprovalAgent,
    TodoApprovalDeps,
)

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


async def _wait_for_pending_request(
    hitl_repo: InMemoryHITLRepository,
    *,
    timeout_s: float = 5.0,
) -> BlockingRequirement:
    """Poll until ``hitl_repo.list_pending`` reports the gate request."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        pending = await hitl_repo.list_pending()
        if pending:
            return pending[0]
        await asyncio.sleep(0.02)
    raise AssertionError(
        "Timed out waiting for HITLGate to persist its pending request",
    )


async def _build_propose_deps(
    *,
    notes_repo: InMemoryNoteRepository,
    thread_repo: InMemoryThreadRepository,
    hitl_repo: InMemoryHITLRepository,
) -> tuple[NoteToolDeps, UUID]:
    """Mint the deps + a parent thread id that NotesAgent.build_deps would.

    A NotesAgent thread (T1) is created first so the propose_todo tool
    has a valid parent_thread_id to write into T2's metadata.
    """
    notes_agent = _TestNotesAgent(notes_repo=notes_repo)
    hitl_gate = HITLGate(
        channel=UIChannel(), policy=AllowAll(), repo=hitl_repo,
    )
    # Make sure the tool is registered on the agent so the test agent
    # behaves like prod (not strictly needed since we call the bare
    # function below, but a useful smoke).
    agent_instance = notes_agent.agent
    assert "propose_todo" in agent_instance._function_toolset.tools  # noqa: SLF001

    t1 = await thread_repo.create(agent="notes", metadata={})
    deps = NoteToolDeps(
        repo=notes_repo,
        hitl_gate=hitl_gate,
        thread_repo=thread_repo,
        parent_thread_id=t1.id,
    )
    return deps, t1.id


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_propose_todo_approved_saves_note(
    fresh_dbos_executor: None,
) -> None:
    """Happy path: ``approve`` unblocks the gate and the note is saved."""
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    hitl_repo = InMemoryHITLRepository()

    deps, _t1_id = await _build_propose_deps(
        notes_repo=notes_repo,
        thread_repo=thread_repo,
        hitl_repo=hitl_repo,
    )
    ctx = _FakeCtx(deps=deps)

    notes_agent = _TestNotesAgent(notes_repo=notes_repo)
    propose_todo = _bound_tool(notes_agent.agent, "propose_todo")

    approval_agent = _TestTodoApprovalAgent(
        hitl_repo=hitl_repo, notes_repo=notes_repo,
    )
    approve_fn = _bound_tool(approval_agent.agent, "approve")

    async def _approver() -> None:
        req = await _wait_for_pending_request(hitl_repo)
        # The metadata we wrote on T2 carries the title/body; the
        # approval tool reads them out of TodoApprovalDeps the same
        # way ``NotesTodoApprovalAgent.build_deps`` would.
        approval_deps = TodoApprovalDeps(
            notes_repo=notes_repo,
            hitl_repo=hitl_repo,
            request_id=req.id,
            proposed_title="groceries",
            proposed_body="milk eggs",
        )
        result = await approve_fn(_FakeCtx(deps=approval_deps))
        assert "Approved" in result, result

    approver_task = asyncio.create_task(_approver())
    try:
        note = await propose_todo(
            ctx, title="groceries", body="milk eggs",
        )
    finally:
        await approver_task

    assert isinstance(note, Note)
    assert note.title == "groceries"
    assert note.body == "milk eggs"
    # The note made it into the repo.
    listed = await notes_repo.list_()
    assert [n.id for n in listed] == [note.id]
    # T2 exists with the right metadata.
    threads = await thread_repo.list_()
    approval_threads = [t for t in threads if t.agent == "todo_approval"]
    assert len(approval_threads) == 1
    assert approval_threads[0].metadata_["proposed_title"] == "groceries"
    # The HITL request resolved.
    pending = await hitl_repo.list_pending()
    assert pending == []
    # And a decision row exists.
    assert len(hitl_repo._decisions) == 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_propose_todo_rejected_raises_and_skips_note(
    fresh_dbos_executor: None,
) -> None:
    """Rejection branch: tool raises RuntimeError, no note persisted."""
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    hitl_repo = InMemoryHITLRepository()

    deps, _t1_id = await _build_propose_deps(
        notes_repo=notes_repo,
        thread_repo=thread_repo,
        hitl_repo=hitl_repo,
    )
    ctx = _FakeCtx(deps=deps)

    notes_agent = _TestNotesAgent(notes_repo=notes_repo)
    propose_todo = _bound_tool(notes_agent.agent, "propose_todo")

    approval_agent = _TestTodoApprovalAgent(
        hitl_repo=hitl_repo, notes_repo=notes_repo,
    )
    reject_fn = _bound_tool(approval_agent.agent, "reject")

    async def _rejecter() -> None:
        req = await _wait_for_pending_request(hitl_repo)
        approval_deps = TodoApprovalDeps(
            notes_repo=notes_repo,
            hitl_repo=hitl_repo,
            request_id=req.id,
            proposed_title="garbage",
            proposed_body="trash",
        )
        result = await reject_fn(
            _FakeCtx(deps=approval_deps), reason="too vague",
        )
        assert "Cancelled" in result, result

    rejecter_task = asyncio.create_task(_rejecter())
    try:
        with pytest.raises(RuntimeError, match="rejected by user"):
            await propose_todo(ctx, title="garbage", body="trash")
    finally:
        await rejecter_task

    # Nothing persisted.
    assert await notes_repo.list_() == []
    # HITL request is still resolved (rejection IS a resolution).
    assert (await hitl_repo.list_pending()) == []
    # The single decision row records a "reject" verdict.
    assert len(hitl_repo._decisions) == 1  # noqa: SLF001
    decision = next(iter(hitl_repo._decisions.values()))  # noqa: SLF001
    assert decision.verdict.value == "reject"


@pytest.mark.asyncio
async def test_propose_todo_modified_uses_new_title_and_body(
    fresh_dbos_executor: None,
) -> None:
    """Modify branch: gate returns modified payload, note saved with overrides."""
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    hitl_repo = InMemoryHITLRepository()

    deps, _t1_id = await _build_propose_deps(
        notes_repo=notes_repo,
        thread_repo=thread_repo,
        hitl_repo=hitl_repo,
    )
    ctx = _FakeCtx(deps=deps)

    notes_agent = _TestNotesAgent(notes_repo=notes_repo)
    propose_todo = _bound_tool(notes_agent.agent, "propose_todo")

    approval_agent = _TestTodoApprovalAgent(
        hitl_repo=hitl_repo, notes_repo=notes_repo,
    )
    modify_fn = _bound_tool(approval_agent.agent, "modify")

    async def _modifier() -> None:
        req = await _wait_for_pending_request(hitl_repo)
        approval_deps = TodoApprovalDeps(
            notes_repo=notes_repo,
            hitl_repo=hitl_repo,
            request_id=req.id,
            proposed_title="groceries",
            proposed_body="milk",
        )
        result = await modify_fn(
            _FakeCtx(deps=approval_deps),
            new_title="weekly groceries",
            new_body="milk, eggs, bread",
        )
        assert "Updated" in result, result

    modifier_task = asyncio.create_task(_modifier())
    try:
        note = await propose_todo(ctx, title="groceries", body="milk")
    finally:
        await modifier_task

    assert isinstance(note, Note)
    assert note.title == "weekly groceries"
    assert note.body == "milk, eggs, bread"
    # The modify-branch note is the only thing in the repo.
    listed = await notes_repo.list_()
    assert [n.title for n in listed] == ["weekly groceries"]


@pytest.mark.asyncio
async def test_approve_tool_returns_error_on_unknown_request(
    fresh_dbos_executor: None,
) -> None:
    """Defensive: approve tool with a stale request_id surfaces a tool error."""
    notes_repo = InMemoryNoteRepository()
    hitl_repo = InMemoryHITLRepository()
    approval_agent = _TestTodoApprovalAgent(
        hitl_repo=hitl_repo, notes_repo=notes_repo,
    )
    approve_fn = _bound_tool(approval_agent.agent, "approve")
    reject_fn = _bound_tool(approval_agent.agent, "reject")
    modify_fn = _bound_tool(approval_agent.agent, "modify")

    bogus_deps = TodoApprovalDeps(
        notes_repo=notes_repo,
        hitl_repo=hitl_repo,
        request_id=UUID("00000000-0000-0000-0000-000000000000"),
        proposed_title="t",
        proposed_body="b",
    )
    ctx = _FakeCtx(deps=bogus_deps)
    assert "not found" in (await approve_fn(ctx))
    assert "not found" in (await reject_fn(ctx))
    assert "not found" in (await modify_fn(ctx))


@pytest.mark.asyncio
async def test_propose_todo_creates_side_thread_with_metadata(
    fresh_dbos_executor: None,
) -> None:
    """T2 carries the right ``TodoApprovalMetadata``-shaped JSON."""
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    hitl_repo = InMemoryHITLRepository()

    deps, t1_id = await _build_propose_deps(
        notes_repo=notes_repo,
        thread_repo=thread_repo,
        hitl_repo=hitl_repo,
    )
    ctx = _FakeCtx(deps=deps)

    notes_agent = _TestNotesAgent(notes_repo=notes_repo)
    propose_todo = _bound_tool(notes_agent.agent, "propose_todo")

    approval_agent = _TestTodoApprovalAgent(
        hitl_repo=hitl_repo, notes_repo=notes_repo,
    )
    approve_fn = _bound_tool(approval_agent.agent, "approve")

    captured: dict[str, Any] = {}

    async def _approver() -> None:
        req = await _wait_for_pending_request(hitl_repo)
        # Find the T2 thread.
        threads = await thread_repo.list_()
        t2 = next(t for t in threads if t.agent == "todo_approval")
        captured["t2_metadata"] = dict(t2.metadata_)
        approval_deps = TodoApprovalDeps(
            notes_repo=notes_repo,
            hitl_repo=hitl_repo,
            request_id=req.id,
            proposed_title="x",
            proposed_body="y",
        )
        await approve_fn(_FakeCtx(deps=approval_deps))

    approver_task = asyncio.create_task(_approver())
    try:
        await propose_todo(ctx, title="x", body="y")
    finally:
        await approver_task

    meta = captured["t2_metadata"]
    assert meta["parent_thread_id"] == str(t1_id)
    assert meta["proposed_title"] == "x"
    assert meta["proposed_body"] == "y"
    # request_id is a UUID string.
    UUID(meta["request_id"])


@pytest.mark.asyncio
async def test_propose_todo_seeds_opening_assistant_message_on_t2(
    fresh_dbos_executor: None,
) -> None:
    """The approval thread shows an opening assistant message immediately."""
    notes_repo = InMemoryNoteRepository()
    thread_repo = InMemoryThreadRepository()
    hitl_repo = InMemoryHITLRepository()

    deps, _t1_id = await _build_propose_deps(
        notes_repo=notes_repo,
        thread_repo=thread_repo,
        hitl_repo=hitl_repo,
    )
    ctx = _FakeCtx(deps=deps)

    notes_agent = _TestNotesAgent(notes_repo=notes_repo)
    propose_todo = _bound_tool(notes_agent.agent, "propose_todo")

    approval_agent = _TestTodoApprovalAgent(
        hitl_repo=hitl_repo, notes_repo=notes_repo,
    )
    approve_fn = _bound_tool(approval_agent.agent, "approve")

    captured: dict[str, Any] = {}

    async def _approver() -> None:
        req = await _wait_for_pending_request(hitl_repo)
        threads = await thread_repo.list_()
        t2 = next(t for t in threads if t.agent == "todo_approval")
        history = await thread_repo.history(t2.id)
        captured["t2_history"] = history
        approval_deps = TodoApprovalDeps(
            notes_repo=notes_repo,
            hitl_repo=hitl_repo,
            request_id=req.id,
            proposed_title="alpha",
            proposed_body="beta",
        )
        await approve_fn(_FakeCtx(deps=approval_deps))

    approver_task = asyncio.create_task(_approver())
    try:
        await propose_todo(ctx, title="alpha", body="beta")
    finally:
        await approver_task

    history = captured["t2_history"]
    assert len(history) == 1
    assert history[0].role == "assistant"
    # ``alpha`` and ``beta`` are in the seeded text.
    text_part = history[0].parts[0]
    assert "alpha" in text_part["text"]
    assert "beta" in text_part["text"]


@pytest.mark.asyncio
async def test_propose_todo_rejects_when_deps_missing_hitl_gate() -> None:
    """Calling ``propose_todo`` without HITL plumbing must fail loudly."""
    notes_repo = InMemoryNoteRepository()
    notes_agent = _TestNotesAgent(notes_repo=notes_repo)
    propose_todo = _bound_tool(notes_agent.agent, "propose_todo")

    deps = NoteToolDeps(repo=notes_repo)  # no hitl_gate / thread_repo
    ctx = _FakeCtx(deps=deps)
    with pytest.raises(RuntimeError, match="propose_todo requires"):
        await propose_todo(ctx, title="x", body="y")


# ---------------------------------------------------------------------------
# End-to-end SSE-over-DBOS smoke is intentionally left unimplemented.
#
# The streaming router would need to run TWO concurrent SSE streams
# (T1 + T2) while DBOS sits between them on ``recv``. The orchestration
# is doable but timing-fragile under TestClient, and the brief
# explicitly endorses the unit-level proof above as the load-bearing
# coverage for the commit. The end-to-end flow is verified manually
# against the running app.
# ---------------------------------------------------------------------------


# Suppress "unused" lint flag on imported symbol that we only consume
# via type annotations (BlockingRequirementStatus is referenced via
# the helper's typed return chain).
_ = BlockingRequirementStatus
