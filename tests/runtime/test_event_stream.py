"""Tests for ``EventStream`` Protocol + ``InProcessEventStream`` impl.

The contract is intentionally loose (notifications may be lost /
duplicated / reordered) so consumers always read full events from the
``EventLogRepository`` after waking up. The in-process implementation
preserves order + delivery within a single process, which is the most
common dev / single-worker case.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from pydantic_ai_stateflow.persistence import (
    EventLogRepository,
    InMemoryEventLogRepository,
)
from pydantic_ai_stateflow.runtime import (
    EventNotification,
    EventStream,
    InProcessEventStream,
    thread_channel,
)


@pytest.mark.asyncio
async def test_in_process_publish_then_subscribe_delivers_notification() -> None:
    """Single publisher + single subscriber: notification arrives in order."""
    stream = InProcessEventStream()
    thread_id = uuid4()
    channel = thread_channel(thread_id)

    received: list[EventNotification] = []

    async def consume() -> None:
        async with stream.subscribe(channel) as events:
            async for ev in events:
                received.append(ev)
                if ev.seq == 2:
                    return

    consumer = asyncio.create_task(consume())

    # Give the subscriber a tick to register before we publish.
    await asyncio.sleep(0)
    await stream.publish(channel, EventNotification(thread_id=thread_id, seq=1))
    await stream.publish(channel, EventNotification(thread_id=thread_id, seq=2))

    await asyncio.wait_for(consumer, timeout=1.0)

    assert [n.seq for n in received] == [1, 2]


@pytest.mark.asyncio
async def test_subscribe_cleanup_removes_subscription() -> None:
    """After ``__aexit__`` the subscriber list is empty (no leaks)."""
    stream = InProcessEventStream()
    thread_id = uuid4()
    channel = thread_channel(thread_id)

    async with stream.subscribe(channel) as _:
        assert channel in stream._subscribers  # noqa: SLF001
        assert len(stream._subscribers[channel]) == 1  # noqa: SLF001

    # Exit removed the queue and (since it was the only one) the bucket.
    assert channel not in stream._subscribers  # noqa: SLF001


@pytest.mark.asyncio
async def test_fanout_two_subscribers_both_receive() -> None:
    """Both subscribers on the same channel get a copy of each notification."""
    stream = InProcessEventStream()
    thread_id = uuid4()
    channel = thread_channel(thread_id)

    received_a: list[int] = []
    received_b: list[int] = []

    async def consume(received: list[int]) -> None:
        async with stream.subscribe(channel) as events:
            async for ev in events:
                received.append(ev.seq)
                if ev.seq == 3:
                    return

    a = asyncio.create_task(consume(received_a))
    b = asyncio.create_task(consume(received_b))
    await asyncio.sleep(0)  # let both subscribers register

    for s in (1, 2, 3):
        await stream.publish(channel, EventNotification(thread_id=thread_id, seq=s))

    await asyncio.wait_for(asyncio.gather(a, b), timeout=1.0)
    assert received_a == [1, 2, 3]
    assert received_b == [1, 2, 3]


@pytest.mark.asyncio
async def test_event_log_append_and_read_since() -> None:
    """``EventLogRepository`` round-trip + monotonic ``seq`` per thread."""
    log: EventLogRepository = InMemoryEventLogRepository()
    thread_id = uuid4()

    ev1 = await log.append(
        thread_id=thread_id, kind="text-delta", payload={"text": "hi"},
    )
    ev2 = await log.append(
        thread_id=thread_id, kind="text-delta", payload={"text": "there"},
    )
    ev3 = await log.append(
        thread_id=thread_id, kind="done", payload={},
    )

    assert ev1.seq == 1
    assert ev2.seq == 2
    assert ev3.seq == 3
    assert await log.latest_seq(thread_id) == 3

    since_zero = await log.read_since(thread_id, after_seq=0)
    assert [e.seq for e in since_zero] == [1, 2, 3]

    since_one = await log.read_since(thread_id, after_seq=1)
    assert [e.seq for e in since_one] == [2, 3]

    since_last = await log.read_since(thread_id, after_seq=3)
    assert since_last == []


@pytest.mark.asyncio
async def test_event_log_isolates_threads() -> None:
    """``seq`` is per-thread (not global) and ``read_since`` is scoped."""
    log = InMemoryEventLogRepository()
    a, b = uuid4(), uuid4()

    await log.append(thread_id=a, kind="x", payload={})
    await log.append(thread_id=b, kind="x", payload={})
    await log.append(thread_id=a, kind="x", payload={})

    assert await log.latest_seq(a) == 2
    assert await log.latest_seq(b) == 1

    events_a = await log.read_since(a)
    events_b = await log.read_since(b)
    assert [e.seq for e in events_a] == [1, 2]
    assert [e.seq for e in events_b] == [1]


