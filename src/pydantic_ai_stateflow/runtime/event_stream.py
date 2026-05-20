"""Pluggable signal transport for live thread-event broadcasts.

The framework persists every ``DurableAgent`` event to the event log
(``EventLogRepository.append``) — that's the durable source of truth.
The ``EventStream`` adds a thin, fire-and-forget **notification**
channel on top: it lets live SSE consumers wake up the moment a new
event lands instead of polling the log.

Two layers, one purpose:

  - ``EventLogRepository`` (DURABLE)  → "what happened, ever"
  - ``EventStream`` (BEST-EFFORT)     → "wake up, look at the log"

Implementations swap freely — apps pick the transport that matches
their deployment shape:

  - ``InProcessEventStream`` — asyncio.Queue per channel. Single
    process (dev / single-worker uvicorn). Zero infra.
  - ``PostgresEventStream``  — postgres ``LISTEN/NOTIFY``. Multi-
    process. Reuses existing DB connection. (TODO — not yet shipped.)
  - ``RedisStreamsEventStream`` — high throughput + retention. Adds
    Redis dependency. (TODO.)
  - ``SnsSqsEventStream`` — AWS-native pub/sub. (TODO.)

Contract for adapter authors (Protocol below): notifications MAY be
lost, MAY be duplicated, MAY arrive out of order. The consumer pulls
the actual events from the log by ``seq`` after each notification —
that gives us a single, durable replay path and absorbs whatever
quirks the transport has.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager


@dataclass(frozen=True)
class EventNotification:
    """Skinny pointer published when a new event is appended to the log.

    Consumers read the full event body from
    ``EventLogRepository.read_since(thread_id, after_seq=last_seen)``
    after waking up — this keeps the notification payload small
    (important for ``LISTEN/NOTIFY``'s 8KB limit and for cheap
    fan-out across many subscribers) and gives a uniform replay path
    that works for both live-tail AND reconnect-resume.
    """

    thread_id: UUID
    seq: int


@runtime_checkable
class EventStream(Protocol):
    """Pluggable signal transport for SSE consumers.

    Contract:
      - ``publish(channel, notification)`` — fire-and-forget. SHOULD
        NOT block the publisher even when consumers are slow / absent;
        backpressure policy is up to the implementor (drop, queue,
        slow-publisher are all valid trade-offs).
      - ``subscribe(channel)`` — async context manager yielding an
        async iterator of notifications. ``__aexit__`` MUST remove
        the subscription (no leaks).
      - Notifications MAY be lost / duplicated / reordered. Consumers
        deduplicate + reorder by reading from ``EventLogRepository``
        using the ``seq`` carried in each notification.

    Channel namespace is opaque strings — the framework uses
    ``"thread:{uuid}"`` but implementations MUST accept any string.
    """

    async def publish(
        self, channel: str, notification: EventNotification,
    ) -> None: ...

    def subscribe(
        self, channel: str,
    ) -> AbstractAsyncContextManager[AsyncIterator[EventNotification]]: ...


class InProcessEventStream:
    """Single-process pub/sub backed by ``asyncio.Queue``.

    Zero infra — the default for dev, tests, and single-worker
    uvicorn deployments. For scale-out / multi-process apps, swap
    for ``PostgresEventStream`` (LISTEN/NOTIFY) or a real broker.
    """

    def __init__(self, *, queue_maxsize: int = 0) -> None:
        # Per-channel list of subscriber queues — fan-out delivers each
        # notification to every active subscriber on the channel.
        self._subscribers: defaultdict[
            str, list[asyncio.Queue[EventNotification]]
        ] = defaultdict(list)
        # 0 → unbounded. Apps with high event rates + risk of slow
        # consumers should set a finite bound + drop policy in a
        # wrapper around this class.
        self._maxsize = queue_maxsize

    async def publish(
        self, channel: str, notification: EventNotification,
    ) -> None:
        # Copy the list so a concurrent subscribe/unsubscribe during
        # iteration doesn't mutate it under us.
        for q in list(self._subscribers.get(channel, ())):
            try:
                q.put_nowait(notification)
            except asyncio.QueueFull:
                # Drop on full — caller's choice if they care, the
                # event is still in the durable log for replay.
                pass

    @asynccontextmanager
    async def subscribe(
        self, channel: str,
    ) -> AsyncIterator[AsyncIterator[EventNotification]]:
        q: asyncio.Queue[EventNotification] = asyncio.Queue(
            maxsize=self._maxsize,
        )
        self._subscribers[channel].append(q)
        try:
            async def gen() -> AsyncIterator[EventNotification]:
                while True:
                    yield await q.get()

            yield gen()
        finally:
            self._subscribers[channel].remove(q)
            if not self._subscribers[channel]:
                del self._subscribers[channel]


def thread_channel(thread_id: UUID) -> str:
    """Canonical channel name for a thread's event stream.

    Centralizing the format here keeps publishers and subscribers in
    sync — apps that want to subscribe externally just call this
    helper with the thread id.
    """
    return f"thread:{thread_id}"


__all__: list[str] = [
    "EventNotification",
    "EventStream",
    "InProcessEventStream",
    "thread_channel",
]


# ``Any`` re-export above sees unused otherwise; keep mypy happy.
_ = Any
