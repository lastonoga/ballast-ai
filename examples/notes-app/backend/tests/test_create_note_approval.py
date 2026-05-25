"""End-to-end test for the new ``create_note`` flow:

  1. Tool builds the ProposedNote draft.
  2. (Refiner may run; here it's None — no API key.)
  3. ``create_note_flow`` child workflow calls
     ``UICardChannel(payload_type=ProposedNote).request`` which suspends
     on ``Durable.recv_async``.
  4. From outside, send the approve verdict via ``Durable.send_async``.
  5. Tool returns the persisted note.

This test uses the real UICardChannel + repo + DBOS. It validates the
integration end-to-end without involving an actual LLM run.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ballast.durable import Durable
from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict, register_card_kind,
)
from notes_app.agents.note_refiner import ProposedNote
from notes_app.repositories.note import InMemoryNoteRepository
from notes_app.workflows.create_note import create_note_flow


register_card_kind(ProposedNote)


async def _wait_card_pending(repo, timeout: float = 5.0) -> str:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        rows = await repo.list_pending()
        if rows:
            return rows[0].id
        await asyncio.sleep(0.05)
    raise TimeoutError("no pending card surfaced")


@pytest.mark.asyncio
async def test_create_note_approve_path(
    fresh_dbos_executor: None,
    repo: InMemoryNoteRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ballast.persistence.approval_card import InMemoryApprovalCardRepository
    approvals = InMemoryApprovalCardRepository()
    monkeypatch.setattr(
        "ballast.persistence.approval_card.approval_card_repo", approvals,
    )

    draft = ProposedNote(title="grocery", body="milk, eggs")
    handle = await Durable.start_workflow(create_note_flow, draft)

    card_id = await _wait_card_pending(approvals)
    card = await approvals.get(card_id)
    assert card is not None

    verdict = CardVerdict[ProposedNote](
        decision="approve",
        modified=None,
        answered_at=datetime.now(UTC),
    )
    await Durable.send_async(
        destination_id=card.workflow_id,
        message=verdict.model_dump(mode="json"),
        topic=card.respond_topic,
    )

    note = await handle.get_result()
    assert note is not None
    assert note.title == "grocery"

    listed = await repo.list_()
    assert [n.id for n in listed] == [note.id]


@pytest.mark.asyncio
async def test_create_note_reject_path(
    fresh_dbos_executor: None,
    repo: InMemoryNoteRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ballast.persistence.approval_card import InMemoryApprovalCardRepository
    approvals = InMemoryApprovalCardRepository()
    monkeypatch.setattr(
        "ballast.persistence.approval_card.approval_card_repo", approvals,
    )

    draft = ProposedNote(title="x", body="y")
    handle = await Durable.start_workflow(create_note_flow, draft)
    card_id = await _wait_card_pending(approvals)
    card = await approvals.get(card_id)

    verdict = CardVerdict[ProposedNote](
        decision="reject",
        answered_at=datetime.now(UTC),
    )
    await Durable.send_async(
        destination_id=card.workflow_id,
        message=verdict.model_dump(mode="json"),
        topic=card.respond_topic,
    )

    note = await handle.get_result()
    assert note is None

    listed = await repo.list_()
    assert listed == []
