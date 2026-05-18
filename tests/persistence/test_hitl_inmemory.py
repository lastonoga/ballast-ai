from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence.hitl import (
    BlockingRequirementStatus,
    DecisionVerdict,
    HITLPurpose,
    InMemoryHITLRepository,
)


@pytest.fixture
def repo() -> InMemoryHITLRepository:
    return InMemoryHITLRepository()


@pytest.fixture
def tenant_id():
    return uuid4()


@pytest.mark.asyncio
async def test_persist_request_creates_pending_record(repo, tenant_id):
    workflow_id = uuid4()
    req = await repo.persist_request(
        prompt={"title": "approve?"},
        workflow_id=workflow_id, gate_kind="strategy_review",
        purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id,
    )
    assert req.status == BlockingRequirementStatus.PENDING
    assert req.gate_kind == "strategy_review"


@pytest.mark.asyncio
async def test_persist_response_resolves_request(repo, tenant_id):
    req = await repo.persist_request(
        prompt={}, workflow_id=uuid4(), gate_kind="g",
        purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id,
    )
    dec = await repo.persist_response(
        request_id=req.id, actor_id="founder-1",
        verdict=DecisionVerdict.APPROVE.value, payload={},
        tenant_id=tenant_id,
    )
    assert dec.verdict == DecisionVerdict.APPROVE
    # Request should now be resolved
    loaded = await repo.load_request(req.id, tenant_id=tenant_id)
    assert loaded.status == BlockingRequirementStatus.RESOLVED


@pytest.mark.asyncio
async def test_persist_timeout_marks_status(repo, tenant_id):
    req = await repo.persist_request(
        prompt={}, workflow_id=uuid4(), gate_kind="g",
        purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id,
    )
    await repo.persist_timeout(req.id, tenant_id=tenant_id)
    loaded = await repo.load_request(req.id, tenant_id=tenant_id)
    assert loaded.status == BlockingRequirementStatus.TIMED_OUT


@pytest.mark.asyncio
async def test_persist_authz_denied_records_attempt(repo, tenant_id):
    req = await repo.persist_request(
        prompt={}, workflow_id=uuid4(), gate_kind="g",
        purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id,
    )
    await repo.persist_authz_denied(
        request_id=req.id, actor_id="intruder",
        voter_votes={"v1": "DENY"}, tenant_id=tenant_id,
    )
    # Request stays pending
    loaded = await repo.load_request(req.id, tenant_id=tenant_id)
    assert loaded.status == BlockingRequirementStatus.PENDING


@pytest.mark.asyncio
async def test_list_pending_for_tenant(repo, tenant_id):
    other = uuid4()
    await repo.persist_request(prompt={}, workflow_id=uuid4(), gate_kind="g",
                                purpose=HITLPurpose.APPROVAL.value, tenant_id=tenant_id)
    await repo.persist_request(prompt={}, workflow_id=uuid4(), gate_kind="g",
                                purpose=HITLPurpose.APPROVAL.value, tenant_id=other)
    pending = await repo.list_pending(tenant_id=tenant_id)
    assert all(p.tenant_id == tenant_id for p in pending)
    assert len(pending) == 1


@pytest.mark.asyncio
async def test_purpose_accepts_custom_string_for_app_extension(repo, tenant_id):
    """Apps can pass their own purpose strings (not in HITLPurpose enum) —
    they round-trip as raw strings on the domain model (extensibility point)."""
    req = await repo.persist_request(
        prompt={}, workflow_id=uuid4(), gate_kind="g",
        purpose="compliance_review",  # ← not in HITLPurpose enum
        tenant_id=tenant_id,
    )
    assert req.purpose == "compliance_review"
    loaded = await repo.load_request(req.id, tenant_id=tenant_id)
    assert loaded is not None
    assert loaded.purpose == "compliance_review"


@pytest.mark.asyncio
async def test_purpose_known_value_coerces_to_enum(repo, tenant_id):
    """Framework-known purposes round-trip as the HITLPurpose enum."""
    req = await repo.persist_request(
        prompt={}, workflow_id=uuid4(), gate_kind="g",
        purpose=HITLPurpose.APPROVAL.value,
        tenant_id=tenant_id,
    )
    assert req.purpose == HITLPurpose.APPROVAL
    assert isinstance(req.purpose, HITLPurpose)
