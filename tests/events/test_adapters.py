"""Tests for ``ballast.events.adapters`` (signal → thread routing)."""
from __future__ import annotations

import pytest
from pydantic import BaseModel
from typing import Literal

from ballast.events import (
    Signal,
    route_to_thread_as_data,
    route_to_thread_as_text,
)
from ballast.persistence import InMemoryEventLogRepository, InMemoryThreadRepository
from ballast.providers.events import EventsProvider
from ballast.runtime.engine import Engine, _reset_ballast_for_tests, _set_ballast
from ballast.runtime.event_stream import InProcessEventStream
from ballast.app import Ballast
from ballast.settings import BallastSettings


class _Event(BaseModel):
    type: Literal["demo-event"] = "demo-event"
    value: int


@pytest.fixture
def _ballast_with_events() -> tuple[InMemoryThreadRepository, Ballast]:
    """Real Ballast wired with InMemory infra so signal defaults fire."""
    _reset_ballast_for_tests()
    thread_repo = InMemoryThreadRepository()
    event_log = InMemoryEventLogRepository()
    event_stream = InProcessEventStream()
    ballast = Ballast(BallastSettings())
    EventsProvider(event_log, event_stream).register(ballast)
    _set_ballast(Engine(
        thread_repo=thread_repo, event_log=event_log, event_stream=event_stream,
    ))
    yield thread_repo, ballast
    _reset_ballast_for_tests()


@pytest.mark.asyncio
async def test_route_to_thread_as_text_default_formatter(
    _ballast_with_events: tuple[InMemoryThreadRepository, Ballast],
) -> None:
    """No format_fn → default `<type>: k=v` debug-ish formatting."""
    thread_repo, _ = _ballast_with_events
    thread = await thread_repo.create(agent="x", metadata={})
    sig = Signal("test_default_text")

    disconnect = route_to_thread_as_text(sig, thread_id=thread.id)
    try:
        await sig.send(sender=object(), event=_Event(value=42))
    finally:
        disconnect()

    history = await thread_repo.history(thread.id)
    assert len(history) == 1
    text = history[0].parts[0]["text"]
    assert "demo-event" in text
    assert "value=42" in text


@pytest.mark.asyncio
async def test_route_to_thread_as_text_custom_formatter(
    _ballast_with_events: tuple[InMemoryThreadRepository, Ballast],
) -> None:
    thread_repo, _ = _ballast_with_events
    thread = await thread_repo.create(agent="x", metadata={})
    sig = Signal("test_custom_text")

    disconnect = route_to_thread_as_text(
        sig, thread_id=thread.id,
        format_fn=lambda ev: f"Got {ev.value}",
    )
    try:
        await sig.send(sender=object(), event=_Event(value=7))
    finally:
        disconnect()

    history = await thread_repo.history(thread.id)
    assert history[0].parts[0]["text"] == "Got 7"


@pytest.mark.asyncio
async def test_route_to_thread_as_text_skips_empty_format(
    _ballast_with_events: tuple[InMemoryThreadRepository, Ballast],
) -> None:
    """Empty string from format_fn → no message posted."""
    thread_repo, _ = _ballast_with_events
    thread = await thread_repo.create(agent="x", metadata={})
    sig = Signal("test_skip_text")

    disconnect = route_to_thread_as_text(
        sig, thread_id=thread.id, format_fn=lambda _: "",
    )
    try:
        await sig.send(sender=object(), event=_Event(value=1))
    finally:
        disconnect()

    assert await thread_repo.history(thread.id) == []


@pytest.mark.asyncio
async def test_route_to_thread_as_data_emits_typed_part(
    _ballast_with_events: tuple[InMemoryThreadRepository, Ballast],
) -> None:
    thread_repo, _ = _ballast_with_events
    thread = await thread_repo.create(agent="x", metadata={})
    sig = Signal("test_data")

    disconnect = route_to_thread_as_data(sig, thread_id=thread.id)
    try:
        await sig.send(sender=object(), event=_Event(value=99))
    finally:
        disconnect()

    history = await thread_repo.history(thread.id)
    assert len(history) == 1
    part = history[0].parts[0]
    assert part["type"] == "data-demo-event"
    assert part["data"] == {"type": "demo-event", "value": 99}
    assert part["state"] == "done"


@pytest.mark.asyncio
async def test_disconnect_unregisters_handler(
    _ballast_with_events: tuple[InMemoryThreadRepository, Ballast],
) -> None:
    thread_repo, _ = _ballast_with_events
    thread = await thread_repo.create(agent="x", metadata={})
    sig = Signal("test_disconnect")

    disconnect = route_to_thread_as_text(sig, thread_id=thread.id)
    await sig.send(sender=object(), event=_Event(value=1))
    disconnect()
    await sig.send(sender=object(), event=_Event(value=2))

    history = await thread_repo.history(thread.id)
    assert len(history) == 1  # only the pre-disconnect emit landed
