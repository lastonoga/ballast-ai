"""notes-app initial: notes + threads + messages + thread_events

Single migration creating every persistent table the app boots with.
Uses ONLY generic SQLAlchemy types (``String``, ``JSON``, ``DateTime``,
``CHAR(36)`` for UUIDs) so the same migration runs on SQLite (default)
and Postgres (prod) without dialect branching.

Note: DBOS workflow state lives on its OWN sqlite file managed by DBOS
itself (see ``main.py:_dbos_db_url``) — not covered here.

Revision ID: 0001
Revises:
Create Date: 2026-05-23 00:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# CHAR(36) is the dialect-portable way to store a UUID's string form.
# On Postgres the future migration can ALTER to ``uuid`` if needed;
# SQLAlchemy's SQLModel UUID type round-trips strings transparently.
_UUID = sa.CHAR(36)


def upgrade() -> None:
    # 1. notes — app-owned
    op.create_table(
        "notes",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("body", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # 2. threads — framework-owned
    op.create_table(
        "threads",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("agent", sa.String(), nullable=False),
        sa.Column(
            "metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("workflow_id", _UUID, nullable=True),
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

    # 3. messages — framework-owned (FK → threads)
    op.create_table(
        "messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "thread_id",
            _UUID,
            sa.ForeignKey("threads.id"),
            nullable=False,
        ),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column(
            "parts",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_messages_thread_id", "messages", ["thread_id"])

    # 4. thread_events — framework-owned (event log for SSE replay)
    op.create_table(
        "thread_events",
        sa.Column("id", _UUID, primary_key=True),
        sa.Column("thread_id", _UUID, nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column(
            "payload",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_thread_events_thread_id", "thread_events", ["thread_id"])
    op.create_index("ix_thread_events_created_at", "thread_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_thread_events_created_at", table_name="thread_events")
    op.drop_index("ix_thread_events_thread_id", table_name="thread_events")
    op.drop_table("thread_events")
    op.drop_index("ix_messages_thread_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_threads_status", table_name="threads")
    op.drop_index("ix_threads_workflow_id", table_name="threads")
    op.drop_table("threads")
    op.drop_table("notes")
