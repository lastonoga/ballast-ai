from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DBOSConfig:
    """Framework wrapper around DBOS configuration.

    DBOS itself uses a TypedDict / dict-shape config — we keep a frozen
    dataclass at our boundary so the framework API is type-stable even if
    DBOS internals shift.
    """

    database_url: str
    app_name: str


def build_dbos_config(
    pg_dsn: str,
    *,
    app_name: str = "pydantic-ai-stateflow",
) -> DBOSConfig:
    """Translate a Sub-project #2 asyncpg DSN into a DBOS-friendly URL.

    DBOS uses synchronous psycopg under the hood for its own internal
    workflow_status table, so an asyncpg-flavored URL is stripped.
    """
    sync_url = re.sub(r"^postgresql\+asyncpg://", "postgresql://", pg_dsn)
    return DBOSConfig(database_url=sync_url, app_name=app_name)
