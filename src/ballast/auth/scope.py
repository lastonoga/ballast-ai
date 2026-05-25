"""``Scope`` — base scope for memory queries / repository filters.

Apps subclass to add domain-specific dimensions (project_id, org_id,
team_id, …). ``extra="allow"`` so consumers can graceful-read
app-custom fields via getattr without requiring a subclass.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Scope(BaseModel):
    """Base scope. Subclass to add app-specific dimensions."""

    model_config = ConfigDict(extra="allow")

    user_id:   str | None = None
    tenant_id: str | None = None
    thread_id: str | None = None


__all__ = ["Scope"]
