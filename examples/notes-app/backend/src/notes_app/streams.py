"""Notes-app event stream — extends framework's InProcess impl.

Apps own the stream instance for the same reason they own the repos:
keeps the wiring point in one place when a process-local stream is
swapped for a Redis-backed one without a refactor.
"""
from __future__ import annotations

from ballast.runtime.event_stream import InProcessEventStream


class NotesEventStream(InProcessEventStream):
    """Notes-app's event stream. Inherits InProcess impl as-is."""


event_stream: NotesEventStream = NotesEventStream()


__all__ = ["NotesEventStream", "event_stream"]
