"""thread.title column + threadstatus 'archived' value

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-19 00:00:00.000000

ThreadStatus is stored as a plain VARCHAR (no native enum type was created in
0001), so the 'archived' value is implicitly accepted — no DDL needed for it.
This migration just adds the nullable ``threads.title`` column.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "threads",
        sa.Column("title", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("threads", "title")
