"""Alembic environment for pydantic-ai-stateflow framework tables.

Importing this module imports every persistence module under
`ballast.persistence.*` so SQLModel.metadata is populated.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlmodel import SQLModel

# Import all framework persistence modules so their tables register with metadata.
import ballast.persistence.events.domain  # noqa: F401
import ballast.persistence.thread.domain  # noqa: F401

config = context.config

# SP3: SP2 settings override — when BALLAST_DBOS__DATABASE_URL (or any
# of the legacy aliases) is set, prefer it over the alembic.ini
# placeholder. Guarded so env.py stays standalone-usable when settings
# are not available in the process (e.g. CI smoke `alembic check`).
try:
    from ballast.settings import get_settings

    _settings = get_settings()
    if _settings.dbos.database_url:
        config.set_main_option("sqlalchemy.url", _settings.dbos.database_url)
except Exception:
    # Settings unavailable — keep ini fallback.
    pass

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online_async() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online_sync() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        do_run_migrations(connection)
    connectable.dispose()


def _is_async_url(url: str | None) -> bool:
    """Branch on URL scheme: only `+asyncpg` / `+aiosqlite` need the async path."""
    if not url:
        return False
    return "+asyncpg" in url or "+aiosqlite" in url or "+asyncmy" in url


if context.is_offline_mode():
    run_migrations_offline()
elif _is_async_url(config.get_main_option("sqlalchemy.url")):
    asyncio.run(run_migrations_online_async())
else:
    run_migrations_online_sync()
