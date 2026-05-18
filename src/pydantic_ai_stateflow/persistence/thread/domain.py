from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.persistence.thread.persistence import MessageRow, ThreadRow


class ThreadPurpose(StrEnum):
    ONBOARDING = "onboarding"
    CONVERSATION = "conversation"
    HITL = "hitl"


class ThreadStatus(StrEnum):
    """Lifecycle state of a thread.

    A thread is `OPEN` from creation until closed. Closing is the terminal
    state — once closed, no further messages can be appended. Reason for
    closing is application-specific (HITL resolved, onboarding done,
    conversation archived, etc.).
    """

    OPEN = "open"
    CLOSED = "closed"


class Thread(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    purpose: ThreadPurpose | str  # may be domain-specific str
    purpose_metadata: dict[str, Any]
    workflow_id: UUID | None
    actor_id: str
    status: ThreadStatus
    created_at: datetime
    closed_at: datetime | None

    @classmethod
    def from_row(cls, row: ThreadRow) -> Thread:
        # Coerce known purposes to enum; unknown stays as str
        try:
            purpose: ThreadPurpose | str = ThreadPurpose(row.purpose)
        except ValueError:
            purpose = row.purpose
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            purpose=purpose,
            purpose_metadata=row.purpose_metadata,
            workflow_id=row.workflow_id,
            actor_id=row.actor_id,
            status=ThreadStatus(row.status),
            created_at=row.created_at,
            closed_at=row.closed_at,
        )


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    tenant_id: UUID
    thread_id: UUID
    role: str
    parts: list[dict[str, Any]]
    created_at: datetime

    @classmethod
    def from_row(cls, row: MessageRow) -> Message:
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            thread_id=row.thread_id,
            role=row.role,
            parts=row.parts,
            created_at=row.created_at,
        )
