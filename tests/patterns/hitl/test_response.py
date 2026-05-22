from __future__ import annotations

from datetime import UTC, datetime

from pydantic import TypeAdapter

from ballast.patterns.hitl import (
    ApprovedResponse,
    HITLResponse,
    ModifiedResponse,
    RejectedResponse,
    TimeoutResponse,
)


def test_approved_response_has_kind_literal() -> None:
    r = ApprovedResponse(actor_id="alice", answered_at=datetime.now(tz=UTC))
    assert r.kind == "approved"


def test_modified_response_carries_modified_proposal() -> None:
    r = ModifiedResponse(
        actor_id="alice",
        answered_at=datetime.now(tz=UTC),
        modified_proposal={"k": 1},
    )
    assert r.modified_proposal == {"k": 1}


def test_timeout_response_has_no_actor() -> None:
    r = TimeoutResponse(answered_at=datetime.now(tz=UTC))
    assert r.actor_id is None
    assert r.kind == "timeout"


def test_discriminated_union_deserialises_by_kind() -> None:
    """Round-trip through TypeAdapter must dispatch on `kind`."""
    adapter: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)
    payload = {
        "kind": "rejected",
        "actor_id": "bob",
        "answered_at": "2026-05-15T12:00:00Z",
        "feedback": "no",
    }
    resp = adapter.validate_python(payload)
    assert isinstance(resp, RejectedResponse)
    assert resp.actor_id == "bob"
