"""``ApprovalCard`` — one human approval request awaiting a decision."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

CardStatus = Literal["pending", "approved", "rejected", "timeout"]

# JSONB on Postgres, JSON on every other dialect (sqlite, …). Only the
# JSON-typed columns need an explicit sa_column variant — str fields use
# SQLModel's native column inference.
#
# SQLModel 0.0.38 with ``table=True`` uses ``sqlmodel_table_construct``
# which bypasses ``__pydantic_validator__``, so field/model validators and
# Literal type annotations are NOT enforced at construction time.  We
# restore those invariants via ``model_post_init``:
#   • status is validated against ``CardStatus`` literals
#   • datetime fields are coerced from ISO-8601 strings on JSON round-trips
#
# ``sa_column`` is kept for:
#   • ``payload`` / ``resolution`` — need JSONB-on-PG / JSON-on-SQLite
#   • ``status`` — keeps String(16) DDL type matching the Alembic migration
#   • ``created_at`` / ``resolved_at`` — DateTime(timezone=True) for PG TZ
_JSON_PORTABLE = JSONB().with_variant(JSON(), "sqlite")

_ALLOWED_STATUS: frozenset[str] = frozenset(
    {"pending", "approved", "rejected", "timeout"}
)


class ApprovalCard(SQLModel, table=True):
    """One pending / resolved approval request displayed in the inbox.

    ``id`` doubles as the HITL ``request_id`` so the wire topic
    (``f"hitl:{id}"``) is stable across the channel ↔ workflow ↔ router
    hops. ``payload`` is the channel's input model as JSON;
    ``resolution`` is the verdict dump once decided.
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

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Restore Pydantic invariants bypassed by ``sqlmodel_table_construct``.

        SQLModel 0.0.38 table models skip ``__pydantic_validator__`` on
        construction.  This hook runs after every ``__init__`` and after
        ``model_validate_json``, so it is the correct place to:
        * validate ``status`` against its ``Literal`` domain, and
        * coerce ISO-8601 strings → ``datetime`` objects (JSON round-trips).
        """
        # Validate status
        if self.status not in _ALLOWED_STATUS:
            raise ValueError(
                f"status must be one of {sorted(_ALLOWED_STATUS)!r},"
                f" got {self.status!r}"
            )
        # Coerce datetime strings produced by model_dump_json / model_validate_json
        if isinstance(self.created_at, str):
            object.__setattr__(
                self,
                "created_at",
                datetime.fromisoformat(self.created_at.replace("Z", "+00:00")),
            )
        if isinstance(self.resolved_at, str):
            object.__setattr__(
                self,
                "resolved_at",
                datetime.fromisoformat(self.resolved_at.replace("Z", "+00:00")),
            )


__all__ = ["ApprovalCard", "CardStatus"]
