"""HITL persistence — single SQLModel per entity (no Row/Domain split).

Three persisted entities — ``BlockingRequirement``, ``Decision``,
``AuthzDenial`` — each is a ``table=True`` SQLModel used as both row
and API/domain payload.

**No tenant_id.** Apps that need multi-tenancy filter at their own
layer (custom HITL router / repo wrapper / RLS policy). Audit fields
that are part of the HITL pattern itself (``Decision.actor_id``,
``AuthzDenial.actor_id``, ``voter_votes``) STAY — they're not
identity-presumption, they're audit semantics of the approval flow.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class HITLPurpose(StrEnum):
    """Framework-suggested values for *why* a HITL gate was raised.

    EXTENSIBLE: apps pass any string to ``persist_request(purpose=...)``.
    The DB stores plain ``str``; this enum is a hint for known values.
    """

    APPROVAL = "approval"
    REJECT_RECOVERY = "reject_recovery"
    AMBIGUITY = "ambiguity"
    POLICY_CONFLICT = "policy_conflict"


class BlockingRequirementStatus(StrEnum):
    """CLOSED — finite lifecycle."""

    PENDING = "pending"
    RESOLVED = "resolved"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class DecisionVerdict(StrEnum):
    """CLOSED — HITLGate Pattern logic branches on these specific verdicts."""

    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"
    OVERRIDE = "override"


class BlockingRequirement(SQLModel, table=True):
    """A blocked workflow step awaiting human decision."""

    __tablename__ = "hitl_blocking_requirements"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    gate_kind: str
    workflow_id: UUID = Field(index=True)
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    purpose: str  # HITLPurpose value or custom string
    status: BlockingRequirementStatus = Field(
        default=BlockingRequirementStatus.PENDING,
    )
    timeout_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    created_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    resolved_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


class Decision(SQLModel, table=True):
    """A human decision (approve/reject/…) against a BlockingRequirement."""

    __tablename__ = "hitl_decisions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    blocking_requirement_id: UUID = Field(
        foreign_key="hitl_blocking_requirements.id", index=True,
    )
    actor_id: str  # audit: who made the call
    verdict: DecisionVerdict
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    helper_verdict_payload: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )
    helper_verdict_context_type: str | None = None
    helper_thread_id: UUID | None = Field(default=None, foreign_key="threads.id")
    created_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )


class AuthzDenial(SQLModel, table=True):
    """An attempted decision rejected by HITL policy (voter votes)."""

    __tablename__ = "hitl_authz_denials"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    request_id: UUID = Field(
        foreign_key="hitl_blocking_requirements.id", index=True,
    )
    actor_id: str
    voter_votes: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    attempted_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
