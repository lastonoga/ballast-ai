from uuid import uuid4

import pytest

from ballast.persistence import SqlAlchemyUnitOfWork
from ballast.persistence.hitl import (
    BlockingRequirementStatus,
    DecisionVerdict,
    HITLPurpose,
    PostgresHITLRepository,
)


@pytest.mark.asyncio
async def test_request_response_postgres(session_factory):
    uow = SqlAlchemyUnitOfWork(session_factory)
    async with uow:
        repo = PostgresHITLRepository(uow.session)
        req = await repo.persist_request(
            prompt={"title": "go?"}, workflow_id=uuid4(), gate_kind="g",
            purpose=HITLPurpose.APPROVAL.value,
        )

    async with session_factory() as s:
        repo2 = PostgresHITLRepository(s)
        loaded = await repo2.load_request(req.id)
        assert loaded.status == BlockingRequirementStatus.PENDING

    uow2 = SqlAlchemyUnitOfWork(session_factory)
    async with uow2:
        repo3 = PostgresHITLRepository(uow2.session)
        await repo3.persist_response(
            request_id=req.id, actor_id="founder",
            verdict=DecisionVerdict.APPROVE.value, payload={},
        )

    async with session_factory() as s:
        repo4 = PostgresHITLRepository(s)
        final = await repo4.load_request(req.id)
        assert final.status == BlockingRequirementStatus.RESOLVED
