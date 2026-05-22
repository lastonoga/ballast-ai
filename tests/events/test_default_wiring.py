"""End-to-end signal wiring smoke test.

Builds a Ballast with :class:`EventsProvider`, appends a message via
``thread_repo.add_message`` and asserts that the framework's default
:data:`message_added` handler writes the ``message-added`` row into the
event log AND publishes onto the matching event-stream channel — i.e.
the repo + signal + provider chain matches what callers used to
hand-write inline.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from ballast.persistence.events.repository import (
    InMemoryEventLogRepository,
)
from ballast.persistence.thread.repository import (
    InMemoryThreadRepository,
)
from ballast.providers.events import EventsProvider
from ballast.runtime.engine import (
    Engine,
    _reset_ballast_for_tests,
    _set_ballast,
)
from ballast.runtime.event_stream import (
    EventNotification,
    InProcessEventStream,
    thread_channel,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def wired_engine() -> "Iterator[Engine]":
    """Construct an in-memory Engine + connect the framework defaults.

    Avoids the full ``Ballast.fastapi`` boot — we don't need FastAPI to
    test the signal-dispatch chain, just the singletons and the
    provider's ``register`` side effect.
    """
    _reset_ballast_for_tests()
    thread_repo = InMemoryThreadRepository()
    event_log = InMemoryEventLogRepository()
    event_stream = InProcessEventStream()
    engine = Engine(
        thread_repo=thread_repo,
        event_log=event_log,
        event_stream=event_stream,
    )
    _set_ballast(engine)

    # ``register`` needs a ``Ballast``-shaped object only for the
    # ``_set_event_log`` / ``_set_event_stream`` hooks; we've installed
    # the engine ourselves above, so call ``connect`` on the signals
    # directly via the provider's wiring path.
    class _Stub:
        def _set_event_log(self, _log: object) -> None: ...
        def _set_event_stream(self, _stream: object) -> None: ...

    EventsProvider(event_log=event_log, event_stream=event_stream).register(
        _Stub(),  # type: ignore[arg-type]
    )

    try:
        yield engine
    finally:
        _reset_ballast_for_tests()


@pytest.mark.asyncio
async def test_add_message_fires_default_handler(
    wired_engine: Engine,
) -> None:
    """``thread_repo.add_message`` → ``message_added`` → event log + publish."""
    thread = await wired_engine.thread_repo.create(agent="dummy")
    channel = thread_channel(thread.id)
    received: list[EventNotification] = []

    async def _consume() -> None:
        async with wired_engine.event_stream.subscribe(channel) as stream:
            async for n in stream:
                received.append(n)
                break  # one notification is enough for the assertion

    consumer = asyncio.create_task(_consume())
    # Yield to let the subscription register before we publish.
    await asyncio.sleep(0)

    msg = await wired_engine.thread_repo.add_message(
        thread.id,
        role="user",
        parts=[{"type": "text", "text": "hi"}],
    )

    await asyncio.wait_for(consumer, timeout=1.0)

    # 1) Event log got a ``message-added`` row.
    events = await wired_engine.event_log.read_since(thread.id, after_seq=0)
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "message-added"
    assert ev.payload["id"] == msg.id
    assert ev.payload["role"] == "user"
    assert ev.payload["parts"] == [{"type": "text", "text": "hi"}]

    # 2) Event stream published a matching notification.
    assert len(received) == 1
    assert received[0].thread_id == thread.id
    assert received[0].seq == ev.seq


@pytest.mark.asyncio
async def test_silent_skips_default_handler(wired_engine: Engine) -> None:
    """``silent=True`` short-circuits the signal so no event log row lands."""
    thread = await wired_engine.thread_repo.create(agent="dummy")
    await wired_engine.thread_repo.add_message(
        thread.id,
        role="user",
        parts=[{"type": "text", "text": "shh"}],
        silent=True,
    )
    events = await wired_engine.event_log.read_since(thread.id, after_seq=0)
    assert events == []


@pytest.mark.asyncio
async def test_upsert_message_also_fires(wired_engine: Engine) -> None:
    """``upsert_message`` emits the same signal as ``add_message``."""
    thread = await wired_engine.thread_repo.create(agent="dummy")
    await wired_engine.thread_repo.upsert_message(
        thread.id,
        id="m1",
        role="assistant",
        parts=[{"type": "text", "text": "first"}],
    )
    await wired_engine.thread_repo.upsert_message(
        thread.id,
        id="m1",
        role="assistant",
        parts=[{"type": "text", "text": "second"}],
    )
    events = await wired_engine.event_log.read_since(thread.id, after_seq=0)
    # Two upserts (insert + in-place replace) → two signal fires →
    # two event-log rows. Frontend treats both as "look at the log",
    # and the replay path collapses to one final message by id.
    assert len(events) == 2
    assert events[0].payload["parts"] == [{"type": "text", "text": "first"}]
    assert events[1].payload["parts"] == [{"type": "text", "text": "second"}]
