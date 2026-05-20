"""``ServiceProvider`` that binds an ``EventStream`` + ``EventLogRepository``.

Wires the two-layer "durable log + live signal" infrastructure into
the Engine's ``Container`` so the streaming router (and any other
component that needs to publish or subscribe to thread events) can
resolve them via DI.

The provider takes both as constructor args — apps that swap to
postgres / redis / etc. just inject the appropriate implementation.
The framework ships ``InProcessEventStream`` + ``InMemoryEventLogRepository``
as the dev / single-worker default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai_stateflow.persistence.events.repository import (
    EventLogRepository,
    InMemoryEventLogRepository,
)
from pydantic_ai_stateflow.runtime.event_stream import (
    EventStream,
    InProcessEventStream,
)

if TYPE_CHECKING:
    from pydantic_ai_stateflow.runtime.container import Container


class EventStreamProvider:
    """Bind ``EventStream`` + ``EventLogRepository`` onto the container.

    Defaults to in-process / in-memory — apps override either by
    passing concrete instances::

        EventStreamProvider(
            stream=PostgresEventStream(dsn=...),
            log=PostgresEventLogRepository(session=...),
        )
    """

    def __init__(
        self,
        *,
        stream: EventStream | None = None,
        log: EventLogRepository | None = None,
    ) -> None:
        self._stream: EventStream = stream or InProcessEventStream()
        self._log: EventLogRepository = log or InMemoryEventLogRepository()

    async def register(self, container: Container) -> None:
        container.bind(EventStream, self._stream)
        container.bind(EventLogRepository, self._log)
