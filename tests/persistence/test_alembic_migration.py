"""Verify Alembic upgrade to head creates all framework tables on a fresh DB."""

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

import ballast


def test_alembic_upgrade_creates_framework_tables(pg_dsn: str) -> None:
    """Drop all framework tables, then verify Alembic upgrade head recreates them.

    Tests the migration in isolation (not idempotency on top of an existing
    schema). The migration must work on a fresh database — that's the
    production contract.
    """
    pkg_dir = Path(ballast.__file__).parent

    # Drop all framework tables (and any leftover alembic_version) for a clean slate.
    async def _reset_schema() -> None:
        engine = create_async_engine(pg_dsn)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
            # Also drop alembic_version if it exists from a prior run
            await conn.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
        await engine.dispose()

    asyncio.run(_reset_schema())

    # Run the migration against the now-empty database
    cfg = Config(str(pkg_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(pkg_dir / "alembic"))
    cfg.set_main_option("sqlalchemy.url", pg_dsn)
    command.upgrade(cfg, "head")

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

    # Restore schema for downstream tests in the session
    async def _restore_schema() -> None:
        engine = create_async_engine(pg_dsn)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        await engine.dispose()

    # The migration already created tables; just stamp metadata as up-to-date
    # by ensuring create_all is a no-op (which it is — tables already exist).
    asyncio.run(_restore_schema())
