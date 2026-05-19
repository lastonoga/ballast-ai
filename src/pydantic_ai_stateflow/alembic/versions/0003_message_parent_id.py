"""messages.parent_id self-FK for conversation branching

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-19 00:00:00.000000

Threads become trees rather than lists. ``parent_id`` is the id of the
message this one replies to:

  - first user turn of a thread:           parent_id = NULL
  - assistant reply to a user turn:        parent_id = <that user's id>
  - follow-up user turn after assistant:   parent_id = <that assistant's id>

Multiple children of the same parent are *branches* (produced by
regenerate-message or user-message edits). Active branch = the path that
picks ``max(created_at)`` at every fork — see
``pydantic_ai_stateflow.persistence.thread.repository._walk_active_branch``.

Backfill strategy for legacy linear rows: leave them with NULL parent_id.
The walker has a fallback path that returns NULL-only-parent rows in
created_at order, so old conversations keep rendering correctly.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "parent_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("messages.id"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_messages_parent_id",
        "messages",
        ["parent_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_messages_parent_id", table_name="messages")
    op.drop_column("messages", "parent_id")
