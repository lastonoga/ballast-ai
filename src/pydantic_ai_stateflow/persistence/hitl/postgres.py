from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from pydantic_ai_stateflow.persistence.hitl.domain import (
    AuthzDenial,
    BlockingRequirement,
    BlockingRequirementStatus,
    Decision,
)


class PostgresHITLRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def persist_request(
        self, *, prompt: dict[str, Any], workflow_id: UUID,
        gate_kind: str, purpose: str,
        timeout_at: datetime | None = None,
        request_id: UUID | None = None,
    ) -> BlockingRequirement:
        kwargs: dict[str, Any] = {
            "gate_kind": gate_kind,
            "workflow_id": workflow_id,
            "payload": dict(prompt),
            "purpose": str(purpose),
            "status": BlockingRequirementStatus.PENDING,
            "timeout_at": timeout_at,
        }
        if request_id is not None:
            kwargs["id"] = request_id
        req = BlockingRequirement(**kwargs)
        self._s.add(req)
        await self._s.flush()
        await self._s.refresh(req)
        return req

    async def load_request(
        self, request_id: UUID,
    ) -> BlockingRequirement | None:
        stmt = select(BlockingRequirement).where(
            col(BlockingRequirement.id) == request_id,
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def persist_response(
        self, *, request_id: UUID, actor_id: str, verdict: str,
        payload: dict[str, Any],
        helper_verdict_payload: dict[str, Any] | None = None,
        helper_verdict_context_type: str | None = None,
        helper_thread_id: UUID | None = None,
    ) -> Decision:
        from pydantic_ai_stateflow.persistence.hitl.domain import DecisionVerdict
        dec = Decision(
            blocking_requirement_id=request_id,
            actor_id=actor_id,
            verdict=DecisionVerdict(verdict),
            payload=dict(payload),
            helper_verdict_payload=helper_verdict_payload,
            helper_verdict_context_type=helper_verdict_context_type,
            helper_thread_id=helper_thread_id,
        )
        self._s.add(dec)
        now = datetime.now(tz=UTC)
        await self._s.execute(
            update(BlockingRequirement)
            .where(col(BlockingRequirement.id) == request_id)
            .values(
                status=BlockingRequirementStatus.RESOLVED,
                resolved_at=now,
            ),
        )
        await self._s.flush()
        await self._s.refresh(dec)
        return dec

    async def persist_timeout(self, request_id: UUID) -> None:
        now = datetime.now(tz=UTC)
        await self._s.execute(
            update(BlockingRequirement)
            .where(col(BlockingRequirement.id) == request_id)
            .values(
                status=BlockingRequirementStatus.TIMED_OUT,
                resolved_at=now,
            ),
        )

    async def persist_authz_denied(
        self, *, request_id: UUID, actor_id: str,
        voter_votes: dict[str, Any],
    ) -> AuthzDenial:
        denial = AuthzDenial(
            request_id=request_id,
            actor_id=actor_id,
            voter_votes=dict(voter_votes),
        )
        self._s.add(denial)
        await self._s.flush()
        await self._s.refresh(denial)
        return denial

    async def list_pending(
        self, *, limit: int = 100,
    ) -> list[BlockingRequirement]:
        stmt = (
            select(BlockingRequirement)
            .where(col(BlockingRequirement.status) == BlockingRequirementStatus.PENDING)
            .order_by(col(BlockingRequirement.created_at).asc())
            .limit(limit)
        )
        return list((await self._s.execute(stmt)).scalars().all())
