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
from pydantic_ai_stateflow.persistence.hitl.persistence import (
    AuthzDenialRow,
    BlockingRequirementRow,
    DecisionRow,
)


class PostgresHITLRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def persist_request(
        self, *, prompt: dict[str, Any], workflow_id: UUID,
        gate_kind: str, purpose: str, tenant_id: UUID,
        timeout_at: datetime | None = None,
    ) -> BlockingRequirement:
        row = BlockingRequirementRow(
            tenant_id=tenant_id, gate_kind=gate_kind,
            workflow_id=workflow_id, payload=dict(prompt),
            purpose=purpose, status=BlockingRequirementStatus.PENDING.value,
            timeout_at=timeout_at,
        )
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row)
        return BlockingRequirement.from_row(row)

    async def load_request(self, request_id: UUID, *, tenant_id: UUID) -> BlockingRequirement | None:
        stmt = select(BlockingRequirementRow).where(
            col(BlockingRequirementRow.id) == request_id,
            col(BlockingRequirementRow.tenant_id) == tenant_id,
        )
        row = (await self._s.execute(stmt)).scalar_one_or_none()
        return BlockingRequirement.from_row(row) if row is not None else None

    async def persist_response(
        self, *, request_id: UUID, actor_id: str, verdict: str,
        payload: dict[str, Any], tenant_id: UUID,
        helper_verdict_payload: dict[str, Any] | None = None,
        helper_verdict_context_type: str | None = None,
        helper_thread_id: UUID | None = None,
    ) -> Decision:
        row = DecisionRow(
            tenant_id=tenant_id, blocking_requirement_id=request_id,
            actor_id=actor_id, verdict=verdict, payload=dict(payload),
            helper_verdict_payload=helper_verdict_payload,
            helper_verdict_context_type=helper_verdict_context_type,
            helper_thread_id=helper_thread_id,
        )
        self._s.add(row)
        now = datetime.now(tz=UTC)
        await self._s.execute(
            update(BlockingRequirementRow)
            .where(
                col(BlockingRequirementRow.id) == request_id,
                col(BlockingRequirementRow.tenant_id) == tenant_id,
            )
            .values(status=BlockingRequirementStatus.RESOLVED.value, resolved_at=now)
        )
        await self._s.flush()
        await self._s.refresh(row)
        return Decision.from_row(row)

    async def persist_timeout(self, request_id: UUID, *, tenant_id: UUID) -> None:
        now = datetime.now(tz=UTC)
        await self._s.execute(
            update(BlockingRequirementRow)
            .where(
                col(BlockingRequirementRow.id) == request_id,
                col(BlockingRequirementRow.tenant_id) == tenant_id,
            )
            .values(status=BlockingRequirementStatus.TIMED_OUT.value, resolved_at=now)
        )

    async def persist_authz_denied(
        self, *, request_id: UUID, actor_id: str,
        voter_votes: dict[str, Any], tenant_id: UUID,
    ) -> AuthzDenial:
        row = AuthzDenialRow(
            tenant_id=tenant_id, request_id=request_id,
            actor_id=actor_id, voter_votes=dict(voter_votes),
        )
        self._s.add(row)
        await self._s.flush()
        await self._s.refresh(row)
        return AuthzDenial.from_row(row)

    async def list_pending(
        self, *, tenant_id: UUID, limit: int = 100
    ) -> list[BlockingRequirement]:
        stmt = (
            select(BlockingRequirementRow)
            .where(
                col(BlockingRequirementRow.tenant_id) == tenant_id,
                col(BlockingRequirementRow.status) == BlockingRequirementStatus.PENDING.value,
            )
            .order_by(col(BlockingRequirementRow.created_at).asc())
            .limit(limit)
        )
        rows = (await self._s.execute(stmt)).scalars().all()
        return [BlockingRequirement.from_row(r) for r in rows]
