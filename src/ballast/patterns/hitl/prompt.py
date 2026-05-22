from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel, ConfigDict


class HITLOption(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    label: str


class HITLPrompt(BaseModel):
    """Canonical HITL prompt.

    No ``tenant_id`` — the framework doesn't presume multi-tenancy.
    Apps that need scoping put it into ``HITLPrompt`` subclass fields
    or carry it through ``Policy.can(...)`` context.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    context: str
    decision_kinds: set[str]
    options: list[HITLOption] = []
    timeout: timedelta | None = None
