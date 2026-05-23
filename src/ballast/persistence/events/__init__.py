"""Thread event log — durable append-only stream per thread.

``DurableAgent`` writes every event from its ``run_stream`` here (via
``@DBOS.step`` so each persistence is idempotent on workflow replay).
The streaming router's SSE endpoint reads from this log on reconnect
to replay events the client missed while disconnected — together with
the live ``EventStream`` signal channel, this gives "no gaps" SSE
across browser tab close / network blip / process restart.
"""

from ballast.persistence.events.domain import ThreadEvent
from ballast.persistence.events.repository import (
    EventLogRepository,
    InMemoryEventLogRepository,
    PostgresEventLogRepository,
)

__all__ = [
    "EventLogRepository",
    "InMemoryEventLogRepository",
    "PostgresEventLogRepository",
    "ThreadEvent",
]
