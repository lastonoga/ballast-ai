from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from pydantic_ai_stateflow.persistence.tenant.persistence import TenantRow


class Tenant(BaseModel):
    """Pydantic domain representation of a tenant."""
    model_config = ConfigDict(frozen=True)

    id: UUID
    name: str
    created_at: datetime

    @classmethod
    def from_row(cls, row: TenantRow) -> Tenant:
        return cls(id=row.id, name=row.name, created_at=row.created_at)
