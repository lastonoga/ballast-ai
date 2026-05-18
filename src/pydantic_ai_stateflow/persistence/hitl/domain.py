from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.persistence.hitl.persistence import (
    AuthzDenialRow,
    BlockingRequirementRow,
    DecisionRow,
)


class HITLPurpose(StrEnum):
    APPROVAL = "approval"
    REJECT_RECOVERY = "reject_recovery"
    AMBIGUITY = "ambiguity"
    POLICY_CONFLICT = "policy_conflict"


class BlockingRequirementStatus(StrEnum):
    PENDING = "pending"
    RESOLVED = "resolved"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class DecisionVerdict(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"
    OVERRIDE = "override"


class BlockingRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    gate_kind: str
    workflow_id: UUID
    payload: dict[str, Any]
    purpose: HITLPurpose
    status: BlockingRequirementStatus
    timeout_at: datetime | None
    created_at: datetime
    resolved_at: datetime | None

    @classmethod
    def from_row(cls, row: BlockingRequirementRow) -> BlockingRequirement:
        return cls(
            id=row.id, tenant_id=row.tenant_id, gate_kind=row.gate_kind,
            workflow_id=row.workflow_id, payload=row.payload,
            purpose=HITLPurpose(row.purpose),
            status=BlockingRequirementStatus(row.status),
            timeout_at=row.timeout_at, created_at=row.created_at,
            resolved_at=row.resolved_at,
        )


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    blocking_requirement_id: UUID
    actor_id: str
    verdict: DecisionVerdict
    payload: dict[str, Any]
    helper_verdict_payload: dict[str, Any] | None
    helper_verdict_context_type: str | None
    helper_thread_id: UUID | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: DecisionRow) -> Decision:
        return cls(
            id=row.id, tenant_id=row.tenant_id,
            blocking_requirement_id=row.blocking_requirement_id,
            actor_id=row.actor_id, verdict=DecisionVerdict(row.verdict),
            payload=row.payload,
            helper_verdict_payload=row.helper_verdict_payload,
            helper_verdict_context_type=row.helper_verdict_context_type,
            helper_thread_id=row.helper_thread_id,
            created_at=row.created_at,
        )


class AuthzDenial(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    request_id: UUID
    actor_id: str
    voter_votes: dict[str, Any]
    attempted_at: datetime

    @classmethod
    def from_row(cls, row: AuthzDenialRow) -> AuthzDenial:
        return cls(
            id=row.id, tenant_id=row.tenant_id, request_id=row.request_id,
            actor_id=row.actor_id, voter_votes=row.voter_votes,
            attempted_at=row.attempted_at,
        )
