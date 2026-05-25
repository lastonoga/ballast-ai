"""create episodes table

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-25
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "episodes",
        sa.Column("id", sa.String(), nullable=False, primary_key=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("tenant_id", sa.String(), nullable=True),
        sa.Column("thread_id", sa.String(), nullable=True),
        sa.Column("preview", sa.String(), nullable=False),
        sa.Column("summary", sa.String(), nullable=True),
        sa.Column("full", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column(
            "references_json",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "metadata_json",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_episodes_source", "episodes", ["source"])
    op.create_index("ix_episodes_user_id", "episodes", ["user_id"])
    op.create_index("ix_episodes_tenant_id", "episodes", ["tenant_id"])
    op.create_index("ix_episodes_thread_id", "episodes", ["thread_id"])
    op.create_index("ix_episodes_occurred_at", "episodes", ["occurred_at"])
    op.execute(
        "CREATE INDEX ix_episodes_embedding_cos ON episodes "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
    )


def downgrade() -> None:
    op.drop_index("ix_episodes_embedding_cos", table_name="episodes")
    op.drop_index("ix_episodes_occurred_at", table_name="episodes")
    op.drop_index("ix_episodes_thread_id", table_name="episodes")
    op.drop_index("ix_episodes_tenant_id", table_name="episodes")
    op.drop_index("ix_episodes_user_id", table_name="episodes")
    op.drop_index("ix_episodes_source", table_name="episodes")
    op.drop_table("episodes")
