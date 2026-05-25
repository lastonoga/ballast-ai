"""``ApprovalCard`` Pydantic model — shape + status transitions."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.persistence.approval_card import ApprovalCard


def _now() -> datetime:
    return datetime(2026, 5, 25, tzinfo=UTC)


def test_pending_card_has_no_resolution() -> None:
    card = ApprovalCard(
        id="req-1", workflow_id="wf-1",
        respond_topic="hitl:req-1", kind="note.create",
        payload={"title": "x", "body": "y"},
        parent_thread_id=None, user_id=None,
        status="pending", created_at=_now(),
    )
    assert card.status == "pending"
    assert card.resolution is None
    assert card.resolved_at is None


def test_status_validates() -> None:
    with pytest.raises(ValueError):
        ApprovalCard(
            id="req-1", workflow_id="wf-1",
            respond_topic="hitl:req-1", kind="note.create",
            payload={}, parent_thread_id=None, user_id=None,
            status="bogus",  # type: ignore[arg-type]
            created_at=_now(),
        )


def test_json_round_trip_preserves_fields() -> None:
    card = ApprovalCard(
        id="req-1", workflow_id="wf-1",
        respond_topic="hitl:req-1", kind="note.create",
        payload={"title": "x", "body": "y"},
        parent_thread_id="t-1", user_id="user-1",
        status="approved", resolution={"decision": "approve"},
        created_at=_now(), resolved_at=_now(),
    )
    dump = card.model_dump_json()
    again = ApprovalCard.model_validate_json(dump)
    assert again == card
