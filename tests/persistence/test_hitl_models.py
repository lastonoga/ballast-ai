from uuid import uuid4

from sqlmodel import SQLModel

from pydantic_ai_stateflow.persistence.hitl import (
    AuthzDenialRow,
    BlockingRequirement,
    BlockingRequirementRow,
    BlockingRequirementStatus,
    Decision,
    DecisionRow,
    DecisionVerdict,
    HITLPurpose,
)


def test_hitl_tables_registered():
    for name in ("hitl_blocking_requirements", "hitl_decisions", "hitl_authz_denials"):
        assert name in SQLModel.metadata.tables


def test_blocking_requirement_row_minimal():
    row = BlockingRequirementRow(
        tenant_id=uuid4(),
        gate_kind="strategy_review",
        workflow_id=uuid4(),
        payload={"prompt": "approve?"},
        purpose=HITLPurpose.APPROVAL.value,
        status=BlockingRequirementStatus.PENDING.value,
    )
    assert row.gate_kind == "strategy_review"
    assert row.payload == {"prompt": "approve?"}


def test_decision_row_minimal():
    row = DecisionRow(
        tenant_id=uuid4(),
        blocking_requirement_id=uuid4(),
        actor_id="founder-1",
        verdict=DecisionVerdict.APPROVE.value,
        payload={"feedback": "ok"},
    )
    assert row.verdict == "approve"
    assert row.helper_verdict_payload is None
    assert row.helper_verdict_context_type is None


def test_authz_denial_row_minimal():
    row = AuthzDenialRow(
        tenant_id=uuid4(),
        request_id=uuid4(),
        actor_id="intruder",
        voter_votes={"voter1": "DENY"},
    )
    assert row.actor_id == "intruder"


def test_domain_models_from_rows():
    req_row = BlockingRequirementRow(
        tenant_id=uuid4(), gate_kind="x", workflow_id=uuid4(),
        payload={}, purpose=HITLPurpose.APPROVAL.value,
        status=BlockingRequirementStatus.PENDING.value,
    )
    domain_req = BlockingRequirement.from_row(req_row)
    assert domain_req.status == BlockingRequirementStatus.PENDING
    assert domain_req.purpose == HITLPurpose.APPROVAL

    dec_row = DecisionRow(
        tenant_id=uuid4(), blocking_requirement_id=uuid4(),
        actor_id="a", verdict=DecisionVerdict.REJECT.value, payload={"reason": "no"},
    )
    domain_dec = Decision.from_row(dec_row)
    assert domain_dec.verdict == DecisionVerdict.REJECT
