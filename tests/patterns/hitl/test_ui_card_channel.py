"""``UICardChannel`` — deliver persists a card row + fires the signal;
decode_verdict re-validates dict → CardVerdict[InT]."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from ballast.auth.context import acting_as
from ballast.events.context import progress_to_thread
from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict,
    UICardChannel,
    approval_card_requested,
    register_card_kind,
)
from ballast.persistence.approval_card import (
    InMemoryApprovalCardRepository,
)


class _Note(BaseModel):
    __hitl_kind__ = "note.create"
    title: str
    body: str


@pytest.fixture(autouse=True)
def _register_note_kind() -> None:
    register_card_kind(_Note)


@pytest.fixture
def fresh_repo(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryApprovalCardRepository]:
    fresh = InMemoryApprovalCardRepository()
    monkeypatch.setattr(
        "ballast.persistence.approval_card.approval_card_repo",
        fresh,
    )
    yield fresh


@pytest.mark.asyncio
async def test_deliver_persists_card_and_fires_signal(
    fresh_repo: InMemoryApprovalCardRepository,
) -> None:
    seen: list[Any] = []
    approval_card_requested.connect(
        lambda sender, *, card, **_: seen.append(card),
    )

    chan: UICardChannel[_Note] = UICardChannel(payload_type=_Note)
    with acting_as("user-1"), progress_to_thread("thread-1"):
        await chan.deliver(
            request_id="req-1", workflow_id="wf-1",
            respond_topic="hitl:req-1",
            payload=_Note(title="t", body="b"),
        )

    stored = await fresh_repo.get("req-1")
    assert stored is not None
    assert stored.kind == "note.create"
    assert stored.payload == {"title": "t", "body": "b"}
    assert stored.user_id == "user-1"
    assert stored.parent_thread_id == "thread-1"

    assert len(seen) == 1 and seen[0].id == "req-1"


@pytest.mark.asyncio
async def test_decode_verdict_typed() -> None:
    chan: UICardChannel[_Note] = UICardChannel(payload_type=_Note)
    verdict = await chan.decode_verdict({
        "decision": "approve",
        "modified": {"title": "x", "body": "y"},
        "answered_at": datetime(2026, 5, 25, tzinfo=UTC).isoformat(),
    })
    assert isinstance(verdict, CardVerdict)
    assert verdict.decision == "approve"
    assert verdict.modified is not None
    assert verdict.modified.title == "x"
