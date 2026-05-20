from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.patterns import MutationPipeline, MutationRejected
from pydantic_ai_stateflow.patterns.mutation import (
    AcceptedResult,
    Proposal,
    RaiseOnReject,
    RejectedAt,
)
from pydantic_ai_stateflow.persistence import InMemoryOutboxRepository


class _PipeRefund(BaseModel):
    amount: int


_RefundProposal = Proposal[_PipeRefund]


class _AcceptStage:
    name = "accept"

    async def process(self, proposal):
        return AcceptedResult(proposal=proposal, entity_id=uuid4())


class _RejectStage:
    name = "validation"

    async def process(self, proposal):
        return RejectedAt(stage=self.name, reason="too big")


class _NoopUoW:
    """Test UoW that satisfies the Protocol without a real SQLAlchemy session."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def commit(self):
        return None

    async def rollback(self):
        return None


class _CountingApply:
    def __init__(self):
        self.calls = 0
        self.last_proposal = None

    async def apply(self, proposal, *, uow):
        self.calls += 1
        self.last_proposal = proposal
        return uuid4()


@pytest.mark.asyncio
async def test_pipeline_runs_all_stages_then_applies(fresh_dbos_executor: None):
    apply = _CountingApply()
    outbox = InMemoryOutboxRepository()
    pipeline = MutationPipeline[_PipeRefund](
        stages=[_AcceptStage(), _AcceptStage()],
        apply=apply,
        uow_factory=lambda: _NoopUoW(),
        outbox=outbox,
        event_type="refund.applied",
    )
    proposal = Proposal[_PipeRefund](
        proposal_id=uuid4(), payload=_PipeRefund(amount=10),
    )
    result = await pipeline.run(proposal)
    assert isinstance(result, AcceptedResult)
    assert apply.calls == 1
    pending = await outbox.list_undelivered()
    assert len(pending) == 1
    assert len(outbox._rows) == 1
    assert outbox._rows[0].event_type == "refund.applied"


@pytest.mark.asyncio
async def test_pipeline_halts_on_first_rejected_and_drops_by_default(
    fresh_dbos_executor: None,
):
    apply = _CountingApply()
    pipeline = MutationPipeline[_PipeRefund](
        stages=[_AcceptStage(), _RejectStage(), _AcceptStage()],
        apply=apply,
        uow_factory=lambda: _NoopUoW(),
        outbox=InMemoryOutboxRepository(),
    )
    proposal = Proposal[_PipeRefund](
        proposal_id=uuid4(), payload=_PipeRefund(amount=10),
    )
    result = await pipeline.run(proposal)
    assert isinstance(result, RejectedAt)
    assert result.stage == "validation"
    assert apply.calls == 0


@pytest.mark.asyncio
async def test_pipeline_raises_when_raise_on_reject_policy_used(
    fresh_dbos_executor: None,
):
    pipeline = MutationPipeline[_PipeRefund](
        stages=[_RejectStage()],
        apply=_CountingApply(),
        uow_factory=lambda: _NoopUoW(),
        outbox=InMemoryOutboxRepository(),
        reject_policy=RaiseOnReject(),
    )
    with pytest.raises(MutationRejected):
        await pipeline.run(
            Proposal[_PipeRefund](proposal_id=uuid4(), payload=_PipeRefund(amount=10)),
        )


@pytest.mark.asyncio
async def test_pipeline_deterministic_workflow_id(fresh_dbos_executor: None):
    """Same (pipeline_name, proposal_id) -> same workflow_id."""
    from pydantic_ai_stateflow.runtime.det import Det
    from pydantic_ai_stateflow.runtime.idempotency import IdempotencyInput

    pipeline = MutationPipeline[_PipeRefund](
        stages=[_AcceptStage()],
        apply=_CountingApply(),
        uow_factory=lambda: _NoopUoW(),
        outbox=InMemoryOutboxRepository(),
        pipeline_name="refund_pipeline",
    )
    pid = uuid4()
    wf_id_1 = await pipeline.derive_workflow_id(pid)
    wf_id_2 = await pipeline.derive_workflow_id(pid)
    assert wf_id_1 == wf_id_2
    expected = await Det.uuid_for(IdempotencyInput(
        namespace="mutation_pipeline",
        parts={"pipeline_name": "refund_pipeline", "proposal_id": pid},
    ))
    assert wf_id_1 == expected


@pytest.mark.asyncio
async def test_pipeline_skips_outbox_when_event_type_not_set(
    fresh_dbos_executor: None,
):
    outbox = InMemoryOutboxRepository()
    pipeline = MutationPipeline[_PipeRefund](
        stages=[_AcceptStage()],
        apply=_CountingApply(),
        uow_factory=lambda: _NoopUoW(),
        outbox=outbox,
        event_type=None,
    )
    await pipeline.run(
        Proposal[_PipeRefund](proposal_id=uuid4(), payload=_PipeRefund(amount=10)),
    )
    assert outbox._rows == []


def test_pipeline_default_name():
    pipeline = MutationPipeline[_PipeRefund](
        stages=[_AcceptStage()],
        apply=_CountingApply(),
        uow_factory=lambda: _NoopUoW(),
        outbox=InMemoryOutboxRepository(),
    )
    assert pipeline.name == "mutation_pipeline"
