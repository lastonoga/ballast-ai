"""EventsProvider — wires the app's :class:`EventLogRepository` and
:class:`EventStream` onto Ballast."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ballast.app import Ballast
    from ballast.persistence.events.repository import EventLogRepository
    from ballast.runtime.event_stream import EventStream


class EventsProvider:
    """Set the event-log repository + in-process event stream on the
    :class:`Engine`."""

    def __init__(
        self,
        event_log: "EventLogRepository",
        event_stream: "EventStream",
    ) -> None:
        self._event_log = event_log
        self._event_stream = event_stream

    def register(self, ballast: "Ballast") -> None:
        ballast._set_event_log(self._event_log)
        ballast._set_event_stream(self._event_stream)


__all__ = ["EventsProvider"]
