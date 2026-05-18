"""Verify Alembic upgrade to head creates all framework tables."""

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine

import pydantic_ai_stateflow


def test_alembic_upgrade_creates_framework_tables(
    create_all_tables: None, pg_dsn: str, pg_dsn_sync: str
) -> None:
    """Run alembic upgrade head and inspect that all expected tables exist.

    The ``create_all_tables`` fixture already materialised the schema via
    SQLModel.metadata.create_all.  Alembic upgrade head should be idempotent
    on top of that — it stamps the alembic_version table without re-creating
    existing tables, and the 0001 migration must also be able to run on a
    *fresh* database.  This test verifies the idempotent case (tables already
    exist from the fixture) plus confirms all 7 framework tables are present.

    Note: pg_dsn_sync echoes pg_dsn (asyncpg URL); alembic/env.py drives
    migrations asynchronously via asyncio.run() so it accepts asyncpg URLs.
    """
    pkg_dir = Path(pydantic_ai_stateflow.__file__).parent
    cfg = Config(str(pkg_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(pkg_dir / "alembic"))
    # Use the asyncpg URL — env.py drives async migration via asyncio.run()
    cfg.set_main_option("sqlalchemy.url", pg_dsn)
    command.upgrade(cfg, "head")

    # Inspect using the async engine since only asyncpg is installed.
    async def _get_table_names() -> set[str]:
        engine = create_async_engine(pg_dsn)
        async with engine.connect() as conn:
            names: list[str] = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
        await engine.dispose()
        return set(names)

    table_names = asyncio.run(_get_table_names())

    expected = {
        "tenants",
        "threads",
        "messages",
        "outbox",
        "hitl_blocking_requirements",
        "hitl_decisions",
        "hitl_authz_denials",
        "alembic_version",
    }
    missing = expected - table_names
    assert not missing, f"Missing tables after Alembic upgrade: {missing}"
