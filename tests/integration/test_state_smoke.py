"""End-to-end smoke test exercising tenant + thread + message + outbox + HITL in one PG session."""

from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence import (
    PostgresHITLRepository,
    PostgresOutboxRepository,
    PostgresThreadRepository,
    SqlAlchemyUnitOfWork,
)
from pydantic_ai_stateflow.persistence.hitl import (
    BlockingRequirementStatus,
    DecisionVerdict,
    HITLPurpose,
)
from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow


@pytest.mark.asyncio
async def test_full_state_round_trip(session_factory):  # type: ignore[no-untyped-def]
    """Single tenant, single thread, two messages, one outbox event, one HITL approval — all in PG."""
    tenant_id = uuid4()
    workflow_id = uuid4()

    # 1. Create tenant
    async with session_factory() as s:
        s.add(TenantRow(id=tenant_id, name="smoke-tenant"))
        await s.commit()

    # 2. Create thread + add two messages
    uow1 = SqlAlchemyUnitOfWork(session_factory)
    async with uow1:
        threads = PostgresThreadRepository(uow1.session)
        thread = await threads.create(
            agent="hitl",
            metadata={"gate_kind": "strategy_review"},
            actor_id="founder-1",
            tenant_id=tenant_id,
        )
        await threads.add_message(
            thread.id,
            role="user",
            parts=[{"kind": "text", "content": "approve please"}],
            tenant_id=tenant_id,
        )
        await threads.add_message(
            thread.id,
            role="assistant",
            parts=[{"kind": "text", "content": "context follows..."}],
            tenant_id=tenant_id,
        )

    # 3. Enqueue an outbox event + persist a HITL request in the SAME tx (transactional outbox)
    uow2 = SqlAlchemyUnitOfWork(session_factory)
    async with uow2:
        outbox = PostgresOutboxRepository(uow2.session)
        hitl = PostgresHITLRepository(uow2.session)

        req = await hitl.persist_request(
            prompt={"title": "approve order?", "amount": 100},
            workflow_id=workflow_id,
            gate_kind="strategy_review",
            purpose=HITLPurpose.APPROVAL.value,
            tenant_id=tenant_id,
        )
        await outbox.enqueue(
            event_type="HITLRequested",
            payload={"request_id": str(req.id), "gate_kind": "strategy_review"},
            tenant_id=tenant_id,
            workflow_id=workflow_id,
        )

    # 4. Founder responds with approve
    uow3 = SqlAlchemyUnitOfWork(session_factory)
    async with uow3:
        hitl = PostgresHITLRepository(uow3.session)
        await hitl.persist_response(
            request_id=req.id,
            actor_id="founder-1",
            verdict=DecisionVerdict.APPROVE.value,
            payload={"feedback": "looks good"},
            tenant_id=tenant_id,
            helper_verdict_payload={"rationale": "all checks passed", "confidence": 0.92},
            helper_verdict_context_type="waves_app.hitl.contexts.StrategyReviewContext",
            helper_thread_id=thread.id,
        )

    # 5. Verify final state
    async with session_factory() as s:
        threads_repo = PostgresThreadRepository(s)
        history = await threads_repo.history(thread.id, tenant_id=tenant_id, limit=10)
        assert [m.role for m in history] == ["user", "assistant"]

        hitl_repo = PostgresHITLRepository(s)
        resolved = await hitl_repo.load_request(req.id, tenant_id=tenant_id)
        assert resolved is not None
        assert resolved.status == BlockingRequirementStatus.RESOLVED

        outbox_repo = PostgresOutboxRepository(s)
        events = await outbox_repo.list_undelivered(tenant_id=tenant_id, limit=10)
        assert any(e.event_type == "HITLRequested" for e in events)
