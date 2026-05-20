"""framework tables: tenants, threads, messages, outbox, hitl_*

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
    # 1. tenants (no FKs)
    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    # 2. threads (FK → tenants)
    op.create_table(
        "threads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("agent", sa.String(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default=sa.text("'open'"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_threads_tenant_id", "threads", ["tenant_id"])
    op.create_index("ix_threads_workflow_id", "threads", ["workflow_id"])
    op.create_index("ix_threads_status", "threads", ["status"])

    # 3. messages (FK → tenants, threads)
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
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
    op.create_index("ix_messages_tenant_id", "messages", ["tenant_id"])
    op.create_index("ix_messages_thread_id", "messages", ["thread_id"])

    # 4. outbox (FK → tenants)
    op.create_table(
        "outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_outbox_tenant_id", "outbox", ["tenant_id"])
    op.create_index("ix_outbox_delivered_at", "outbox", ["delivered_at"])
    op.create_index("ix_outbox_created_at", "outbox", ["created_at"])

    # 5. hitl_blocking_requirements (FK → tenants)
    op.create_table(
        "hitl_blocking_requirements",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column("gate_kind", sa.String(), nullable=False),
        sa.Column("workflow_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("purpose", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("timeout_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_hitl_blocking_requirements_tenant_id",
        "hitl_blocking_requirements",
        ["tenant_id"],
    )
    op.create_index(
        "ix_hitl_blocking_requirements_workflow_id",
        "hitl_blocking_requirements",
        ["workflow_id"],
    )
    op.create_index(
        "ix_hitl_blocking_requirements_created_at",
        "hitl_blocking_requirements",
        ["created_at"],
    )

    # 6. hitl_decisions (FK → tenants, hitl_blocking_requirements, threads (nullable))
    op.create_table(
        "hitl_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "blocking_requirement_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hitl_blocking_requirements.id"),
            nullable=False,
        ),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("verdict", sa.String(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "helper_verdict_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("helper_verdict_context_type", sa.String(), nullable=True),
        sa.Column(
            "helper_thread_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("threads.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_hitl_decisions_tenant_id", "hitl_decisions", ["tenant_id"])
    op.create_index(
        "ix_hitl_decisions_blocking_requirement_id",
        "hitl_decisions",
        ["blocking_requirement_id"],
    )
    op.create_index("ix_hitl_decisions_created_at", "hitl_decisions", ["created_at"])

    # 7. hitl_authz_denials (FK → tenants, hitl_blocking_requirements)
    op.create_table(
        "hitl_authz_denials",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("hitl_blocking_requirements.id"),
            nullable=False,
        ),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column(
            "voter_votes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_hitl_authz_denials_tenant_id", "hitl_authz_denials", ["tenant_id"]
    )
    op.create_index(
        "ix_hitl_authz_denials_request_id", "hitl_authz_denials", ["request_id"]
    )


def downgrade() -> None:
    op.drop_table("hitl_authz_denials")
    op.drop_table("hitl_decisions")
    op.drop_table("hitl_blocking_requirements")
    op.drop_table("outbox")
    op.drop_table("messages")
    op.drop_table("threads")
    op.drop_table("tenants")
