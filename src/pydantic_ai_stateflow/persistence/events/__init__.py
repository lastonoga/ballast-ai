"""Thread event log — durable append-only stream per thread.

``StateflowDurableAgent`` writes every event from its ``run_stream`` here (via
``@DBOS.step`` so each persistence is idempotent on workflow replay).
The streaming router's SSE endpoint reads from this log on reconnect
to replay events the client missed while disconnected — together with
the live ``EventStream`` signal channel, this gives "no gaps" SSE
across browser tab close / network blip / process restart.
"""

from pydantic_ai_stateflow.persistence.events.domain import ThreadEvent
from pydantic_ai_stateflow.persistence.events.repository import (
    EventLogRepository,
    InMemoryEventLogRepository,
)

__all__ = [
    "EventLogRepository",
    "InMemoryEventLogRepository",
    "ThreadEvent",
]
