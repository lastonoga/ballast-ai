from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.persistence.outbox.persistence import OutboxRow


class OutboxEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID
    tenant_id: UUID
    event_type: str
    payload: dict[str, Any]
    workflow_id: UUID | None
    delivered_at: datetime | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: OutboxRow) -> OutboxEvent:
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            event_type=row.event_type,
            payload=row.payload,
            workflow_id=row.workflow_id,
            delivered_at=row.delivered_at,
            created_at=row.created_at,
        )
