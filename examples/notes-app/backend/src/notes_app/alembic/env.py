"""Alembic environment for notes-app's persistent state.

Imports the SQLModel domain modules from BOTH the framework
(``ballast.persistence.thread.domain`` + ``ballast.persistence.events.domain``)
AND the notes-app (``notes_app.models.note``) so ``SQLModel.metadata``
contains every table the app cares about: ``notes`` + ``threads`` +
``messages`` + ``thread_events``.

The DBOS workflow store lives on its own sqlite file (see
``main.py:_dbos_db_url``) and is NOT managed by this alembic config.

URL precedence:
1. ``-x dburl=...`` (alembic CLI flag) — highest
2. ``NOTES_APP_DATABASE_URL`` env var
3. ``sqlalchemy.url`` from alembic.ini — lowest (default sqlite file)
"""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlmodel import SQLModel

# Import all SQLModel table classes so they register with SQLModel.metadata.
import ballast.persistence.events.domain  # noqa: F401
import ballast.persistence.thread.domain  # noqa: F401

import notes_app.models.note  # noqa: F401

config = context.config

# URL resolution: CLI -x flag → env var → ini default.
_x_args = context.get_x_argument(as_dictionary=True)
_url = (
    _x_args.get("dburl")
    or os.environ.get("NOTES_APP_DATABASE_URL")
    or config.get_main_option("sqlalchemy.url")
)
if _url:
    config.set_main_option("sqlalchemy.url", _url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # batch mode lets sqlite handle ALTER-TABLE operations cleanly.
        render_as_batch=connection.dialect.name == "sqlite",
    )
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
    if not url:
        return False
    return "+asyncpg" in url or "+aiosqlite" in url or "+asyncmy" in url


if context.is_offline_mode():
    run_migrations_offline()
elif _is_async_url(config.get_main_option("sqlalchemy.url")):
    asyncio.run(run_migrations_online_async())
else:
    run_migrations_online_sync()
