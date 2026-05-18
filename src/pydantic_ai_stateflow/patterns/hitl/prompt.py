from __future__ import annotations

from datetime import timedelta
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class HITLOption(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    label: str


class HITLPrompt(BaseModel):
    """Canonical HITL prompt (spec 4A.0.2).

    `tenant_id` is REQUIRED on the prompt — readers (channel, repo, gate)
    pull it from here instead of taking a separate kwarg. This removes a
    class of cross-tenant bugs.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: UUID
    title: str
    context: str
    decision_kinds: set[str]
    options: list[HITLOption] = []
    timeout: timedelta | None = None
