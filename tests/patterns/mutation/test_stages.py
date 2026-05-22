from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel

from ballast.patterns import HITLGate
from ballast.patterns.hitl import (
    AllowAll,
    ApprovedResponse,
    HITLPrompt,
    InMemoryHITLChannel,
    ModifiedResponse,
    RejectedResponse,
)
from ballast.patterns.mutation import (
    AcceptedResult,
    ApprovalStage,
    Proposal,
    RejectedAt,
)
from ballast.persistence import InMemoryHITLRepository


class _StageRefund(BaseModel):
    amount: int
    note: str = ""


def _gate_with_response(tenant_id, response_kind: str, **kw):
    channel = InMemoryHITLChannel()
    repo = InMemoryHITLRepository()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    orig = repo.persist_request

    async def capture(**rqkw):
        req = await orig(**rqkw)
        if response_kind == "approved":
            channel.set_response(req.id, ApprovedResponse(
                actor_id="alice", answered_at=datetime.now(tz=UTC), **kw,
            ))
        elif response_kind == "rejected":
            channel.set_response(req.id, RejectedResponse(
                actor_id="alice", answered_at=datetime.now(tz=UTC), **kw,
            ))
        elif response_kind == "modified":
            channel.set_response(req.id, ModifiedResponse(
                actor_id="alice", answered_at=datetime.now(tz=UTC), **kw,
            ))
        return req

    repo.persist_request = capture  # type: ignore[method-assign]
    return gate


@pytest.mark.asyncio
async def test_approval_stage_passes_through_on_approved(
    fresh_dbos_executor: None,
):
    tid = uuid4()
    gate = _gate_with_response(tid, "approved")
    stage = ApprovalStage[_StageRefund](
        hitl=gate,
        prompt_builder=lambda p: HITLPrompt(
            title="x", context=str(p.payload),
            decision_kinds={"approved", "rejected"},
        ),
    )
    proposal = Proposal[_StageRefund](
        proposal_id=uuid4(), payload=_StageRefund(amount=10),
    )
    result = await stage.process(proposal)
    assert isinstance(result, AcceptedResult)
    assert result.proposal is proposal


@pytest.mark.asyncio
async def test_approval_stage_returns_rejected_at_on_rejected(
    fresh_dbos_executor: None,
):
    tid = uuid4()
    gate = _gate_with_response(tid, "rejected", feedback="too big")
    stage = ApprovalStage[_StageRefund](
        hitl=gate,
        prompt_builder=lambda p: HITLPrompt(
            title="x", context="y",
            decision_kinds={"approved", "rejected"},
        ),
    )
    proposal = Proposal[_StageRefund](
        proposal_id=uuid4(), payload=_StageRefund(amount=10),
    )
    result = await stage.process(proposal)
    assert isinstance(result, RejectedAt)
    assert result.reason == "too big"
    assert result.actor_id == "alice"


@pytest.mark.asyncio
async def test_approval_stage_modify_requires_editable_paths(
    fresh_dbos_executor: None,
):
    """Spec: allow_modify=True without editable_paths is a ConfigError."""
    tid = uuid4()
    gate = _gate_with_response(tid, "approved")
    with pytest.raises(ValueError, match="editable_paths"):
        ApprovalStage[_StageRefund](
            hitl=gate,
            prompt_builder=lambda p: HITLPrompt(
                title="x", context="y", decision_kinds={"approved"},
            ),
            allow_modify=True,
        )


@pytest.mark.asyncio
async def test_approval_stage_applies_modification_inside_whitelist(
    fresh_dbos_executor: None,
):
    tid = uuid4()
    gate = _gate_with_response(
        tid, "modified", modified_proposal={"amount": 5, "note": "reduced"},
    )
    stage = ApprovalStage[_StageRefund](
        hitl=gate,
        prompt_builder=lambda p: HITLPrompt(
            title="x", context="y",
            decision_kinds={"approved", "rejected", "modified"},
        ),
        allow_modify=True,
        editable_paths={"amount", "note"},
    )
    proposal = Proposal[_StageRefund](
        proposal_id=uuid4(), payload=_StageRefund(amount=10),
    )
    result = await stage.process(proposal)
    assert isinstance(result, AcceptedResult)
    assert result.proposal.payload.amount == 5
    assert result.proposal.payload.note == "reduced"


@pytest.mark.asyncio
async def test_approval_stage_rejects_modification_outside_whitelist(
    fresh_dbos_executor: None,
):
    tid = uuid4()
    gate = _gate_with_response(
        tid, "modified",
        modified_proposal={"amount": 5, "note": "x", "evil_field": True},
    )
    stage = ApprovalStage[_StageRefund](
        hitl=gate,
        prompt_builder=lambda p: HITLPrompt(
            title="x", context="y",
            decision_kinds={"approved", "modified"},
        ),
        allow_modify=True,
        editable_paths={"amount"},
    )
    proposal = Proposal[_StageRefund](
        proposal_id=uuid4(), payload=_StageRefund(amount=10, note="orig"),
    )
    result = await stage.process(proposal)
    assert isinstance(result, RejectedAt)
    assert "whitelist" in result.reason.lower() or "outside" in result.reason.lower()
