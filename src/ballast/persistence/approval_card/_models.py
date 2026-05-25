"""``ApprovalCard`` — one human approval request awaiting a decision."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

CardStatus = Literal["pending", "approved", "rejected", "timeout"]

# JSONB on Postgres, JSON on every other dialect (sqlite, …).
_JSON_PORTABLE = JSONB().with_variant(JSON(), "sqlite")


class ApprovalCard(SQLModel, table=True):
    """One pending / resolved approval request displayed in the inbox.

    ``id`` doubles as the HITL ``request_id`` so the wire topic
    (``f"hitl:{id}"``) is stable across the channel ↔ workflow ↔ router
    hops. ``payload`` is the channel's input model as JSON;
    ``resolution`` is the verdict dump once decided.

    .. note::
       SQLModel 0.0.38 with ``table=True`` + ``sa_column`` does not enforce
       Pydantic ``Literal`` constraints at construction time. Status integrity
       is enforced by the SQL column type (``String(16)``) and by the
       application-level state machine in each repository's ``resolve``
       method. ``@field_validator`` / ``@model_validator`` decorators have no
       effect on ``sa_column``-mapped fields in this version of SQLModel.
    """

    __tablename__ = "approval_cards"

    id: str = Field(primary_key=True)
    workflow_id: str
    respond_topic: str
    kind: str = Field(index=True)
    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(_JSON_PORTABLE, nullable=False),
    )
    parent_thread_id: str | None = Field(default=None, index=True)
    user_id: str | None = Field(default=None, index=True)
    status: CardStatus = Field(
        default="pending",
        sa_column=Column(String(16), nullable=False, index=True),
    )
    resolution: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(_JSON_PORTABLE, nullable=True),
    )
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    resolved_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


__all__ = ["ApprovalCard", "CardStatus"]
