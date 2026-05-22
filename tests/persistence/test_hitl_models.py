from uuid import uuid4

from sqlmodel import SQLModel

from ballast.persistence.hitl import (
    AuthzDenial,
    BlockingRequirement,
    BlockingRequirementStatus,
    Decision,
    DecisionVerdict,
    HITLPurpose,
)


def test_hitl_tables_registered() -> None:
    for name in ("hitl_blocking_requirements", "hitl_decisions", "hitl_authz_denials"):
        assert name in SQLModel.metadata.tables


def test_blocking_requirement_minimal() -> None:
    req = BlockingRequirement(
        gate_kind="strategy_review",
        workflow_id=uuid4(),
        payload={"prompt": "approve?"},
        purpose=HITLPurpose.APPROVAL.value,
        status=BlockingRequirementStatus.PENDING,
    )
    assert req.gate_kind == "strategy_review"
    assert req.payload == {"prompt": "approve?"}
    assert req.status == BlockingRequirementStatus.PENDING


def test_decision_minimal() -> None:
    dec = Decision(
        blocking_requirement_id=uuid4(),
        actor_id="founder-1",
        verdict=DecisionVerdict.APPROVE,
        payload={"feedback": "ok"},
    )
    assert dec.verdict == DecisionVerdict.APPROVE
    assert dec.helper_verdict_payload is None
    assert dec.helper_verdict_context_type is None


def test_authz_denial_minimal() -> None:
    denial = AuthzDenial(
        request_id=uuid4(),
        actor_id="intruder",
        voter_votes={"voter1": "DENY"},
    )
    assert denial.actor_id == "intruder"
