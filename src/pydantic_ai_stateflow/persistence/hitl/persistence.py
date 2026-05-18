from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


class BlockingRequirementRow(SQLModel, table=True):
    __tablename__ = "hitl_blocking_requirements"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    gate_kind: str
    workflow_id: UUID = Field(index=True)
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    purpose: str                                               # HITLPurpose value
    status: str                                                # BlockingRequirementStatus value
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


class DecisionRow(SQLModel, table=True):
    __tablename__ = "hitl_decisions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    blocking_requirement_id: UUID = Field(
        foreign_key="hitl_blocking_requirements.id", index=True
    )
    actor_id: str
    verdict: str                                               # DecisionVerdict value
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    helper_verdict_payload: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )
    helper_verdict_context_type: str | None = Field(default=None)
    helper_thread_id: UUID | None = Field(default=None, foreign_key="threads.id")
    created_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )


class AuthzDenialRow(SQLModel, table=True):
    __tablename__ = "hitl_authz_denials"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    request_id: UUID = Field(foreign_key="hitl_blocking_requirements.id", index=True)
    actor_id: str
    voter_votes: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
    )
    attempted_at: datetime = Field(
        default_factory=_now_utc,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
