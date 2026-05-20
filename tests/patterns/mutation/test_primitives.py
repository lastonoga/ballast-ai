from uuid import uuid4

from pydantic import BaseModel

from pydantic_ai_stateflow.patterns.mutation import (
    AcceptedResult,
    ApplyTransaction,
    Proposal,
    RejectedAt,
    Stage,
)


class _Refund(BaseModel):
    amount: int
    reason: str


def test_proposal_carries_proposal_id_payload_and_actor():
    p = Proposal[_Refund](
        proposal_id=uuid4(),
        payload=_Refund(amount=50, reason="late"),
        actor_id="alice",
    )
    assert p.payload.amount == 50
    assert p.actor_id == "alice"


def test_accepted_result_wraps_proposal():
    p = Proposal[_Refund](
        proposal_id=uuid4(),
        payload=_Refund(amount=50, reason="late"),
    )
    accepted = AcceptedResult[_Refund](proposal=p, entity_id=uuid4())
    assert accepted.proposal is p


def test_rejected_at_records_stage_reason_actor():
    r = RejectedAt(stage="validation", reason="invalid", actor_id="alice")
    assert r.stage == "validation"
    assert r.reason == "invalid"
    assert r.actor_id == "alice"


def test_stage_protocol_satisfied_structurally():
    class MyStage:
        name = "noop"

        async def process(self, proposal):
            return AcceptedResult(proposal=proposal, entity_id=uuid4())

    assert isinstance(MyStage(), Stage)


def test_apply_transaction_protocol_satisfied_structurally():
    class MyApply:
        async def apply(self, proposal, *, uow):
            return uuid4()

    assert isinstance(MyApply(), ApplyTransaction)
