"""framework tables: threads, messages, thread_events

The framework intentionally does NOT model tenants, actors, or
identity. Apps that need multi-tenancy / per-user scoping store that
data in ``threads.metadata`` (free-form JSONB) and filter at their
own layer.

Revision ID: 0001
Revises:
Create Date: 2026-05-15 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. threads
    op.create_table(
        "threads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("agent", sa.String(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default=sa.text("'open'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_threads_workflow_id", "threads", ["workflow_id"])
    op.create_index("ix_threads_status", "threads", ["status"])

    # 2. messages (FK → threads)
    #
    # Flat linear list — no ``parent_id``. ``id`` is a free-form string
    # (not UUID) so assistant-ui's short client ids like
    # ``"MbPSd9jddGfC6UAV"`` round-trip 1:1. Backend-issued ids default
    # to ``str(uuid4())``. The UI runtime renders a flat array so we
    # don't model branches — edit / regenerate collapse to "truncate
    # then append" in ``POST /threads/{id}/messages``.
    op.create_table(
        "messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "thread_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("threads.id"),
            nullable=False,
        ),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column(
            "parts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_messages_thread_id", "messages", ["thread_id"])

    # 3. thread_events
    op.create_table(
        "thread_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column(
            "thread_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("threads.id"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "thread_id", "seq", name="uq_thread_events_thread_seq",
        ),
    )
    op.create_index(
        "ix_thread_events_thread_id", "thread_events", ["thread_id"],
    )


def downgrade() -> None:
    op.drop_table("thread_events")
    op.drop_table("messages")
    op.drop_table("threads")
