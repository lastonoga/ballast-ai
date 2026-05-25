"""approval_cards table

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-25 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "approval_cards",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column("respond_topic", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("parent_thread_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column(
            "status",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column(
            "resolution",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_approval_cards_kind", "approval_cards", ["kind"])
    op.create_index(
        "ix_approval_cards_parent_thread_id", "approval_cards", ["parent_thread_id"],
    )
    op.create_index("ix_approval_cards_user_id", "approval_cards", ["user_id"])
    op.create_index("ix_approval_cards_status", "approval_cards", ["status"])
    op.create_index("ix_approval_cards_created_at", "approval_cards", ["created_at"])


def downgrade() -> None:
    op.drop_table("approval_cards")
