"""Tests for ``ThreadEventType`` / ``ThreadEventBroadcaster`` /
``ThreadEventStream`` + the underlying ``upsert_message`` repo method.

Covers:
  - one-shot ``emit`` with persistent=True → message in repo + signal
  - one-shot ``emit`` with persistent=False → no repo write, signal only
  - streaming ``stream()`` → message_id stable across updates, repo row
    replaced in place
  - data schema validation rejects wrong-shape inputs
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from ballast.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from ballast.runtime import (
    InProcessEventStream,
    ThreadEventBroadcaster,
    ThreadEventType,
)


class ProgressData(BaseModel):
    step: str
    status: Literal["running", "ok", "failed"]


PROGRESS = ThreadEventType("brainstorm-progress", ProgressData)


async def _setup() -> tuple[
    ThreadEventBroadcaster,
    InMemoryThreadRepository,
    InMemoryEventLogRepository,
    InProcessEventStream,
]:
    thread_repo = InMemoryThreadRepository()
    event_log = InMemoryEventLogRepository()
    event_stream = InProcessEventStream()
    broadcaster = ThreadEventBroadcaster(
        thread_repo=thread_repo,
        event_log=event_log,
        event_stream=event_stream,
    )
    return broadcaster, thread_repo, event_log, event_stream


@pytest.mark.asyncio
async def test_emit_persistent_writes_message_and_signals() -> None:
    broadcaster, thread_repo, event_log, _ = await _setup()
    thread = await thread_repo.create(agent="test")

    msg_id = await PROGRESS.emit(
        broadcaster, thread.id,
        ProgressData(step="diverge", status="ok"),
    )

    history = await thread_repo.history(thread.id)
    assert len(history) == 1
    msg = history[0]
    assert msg.id == msg_id
    assert msg.role == "assistant"
    assert msg.parts == [{
        "type": "data-brainstorm-progress",
        "data": {"step": "diverge", "status": "ok"},
    }]

    events = await event_log.read_since(thread.id, after_seq=0)
    assert len(events) == 1
    assert events[0].kind == "message-added"
    assert events[0].payload["id"] == msg_id


@pytest.mark.asyncio
async def test_emit_transient_skips_repo_but_signals() -> None:
    broadcaster, thread_repo, event_log, _ = await _setup()
    thread = await thread_repo.create(agent="test")

    msg_id = await PROGRESS.emit(
        broadcaster, thread.id,
        ProgressData(step="hint", status="running"),
        persistent=False,
    )

    # Repo is empty — transient events never hit thread_repo.
    history = await thread_repo.history(thread.id)
    assert history == []

    # Event log still records the signal so SSE consumers see it.
    events = await event_log.read_since(thread.id, after_seq=0)
    assert len(events) == 1
    assert events[0].payload["id"] == msg_id
    assert events[0].payload["parts"][0]["type"] == "data-brainstorm-progress"


@pytest.mark.asyncio
async def test_stream_overwrites_same_message() -> None:
    broadcaster, thread_repo, event_log, _ = await _setup()
    thread = await thread_repo.create(agent="test")

    async with PROGRESS.stream(broadcaster, thread.id) as stream:
        await stream.update(ProgressData(step="diverge", status="running"))
        await stream.update(ProgressData(step="diverge", status="ok"))
        await stream.update(ProgressData(step="converge", status="ok"))

    # Exactly one message in history — the last snapshot wins.
    history = await thread_repo.history(thread.id)
    assert len(history) == 1, [m.parts for m in history]
    final_part = history[0].parts[0]
    assert final_part["data"] == {"step": "converge", "status": "ok"}

    # Three signals fired (one per update).
    events = await event_log.read_since(thread.id, after_seq=0)
    assert len(events) == 3
    assert all(e.kind == "message-added" for e in events)
    # All three carry the same message_id — the SSE consumer keys by
    # id and replaces, hence one visible UI event that mutates.
    msg_ids = {e.payload["id"] for e in events}
    assert len(msg_ids) == 1


@pytest.mark.asyncio
async def test_stream_transient_skips_repo_entirely() -> None:
    broadcaster, thread_repo, event_log, _ = await _setup()
    thread = await thread_repo.create(agent="test")

    async with PROGRESS.stream(
        broadcaster, thread.id, persistent=False,
    ) as stream:
        await stream.update(ProgressData(step="a", status="running"))
        await stream.update(ProgressData(step="a", status="ok"))

    assert await thread_repo.history(thread.id) == []
    events = await event_log.read_since(thread.id, after_seq=0)
    assert len(events) == 2
    assert all(e.kind == "message-added" for e in events)


@pytest.mark.asyncio
async def test_emit_rejects_wrong_data_shape() -> None:
    broadcaster, thread_repo, _, _ = await _setup()
    thread = await thread_repo.create(agent="test")

    class OtherShape(BaseModel):
        whatever: int

    with pytest.raises(TypeError, match="brainstorm-progress"):
        await PROGRESS.emit(
            broadcaster, thread.id,
            OtherShape(whatever=1),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_part_type_matches_wire_convention() -> None:
    # Frontend ``makeAssistantMessagePart({type: "data-xxx"})`` MUST
    # use the same string this property returns.
    assert PROGRESS.part_type == "data-brainstorm-progress"


def test_event_type_rejects_illegal_names() -> None:
    with pytest.raises(ValueError):
        ThreadEventType("", ProgressData)
    with pytest.raises(ValueError):
        ThreadEventType("bad name", ProgressData)
    with pytest.raises(ValueError):
        ThreadEventType("bad/name", ProgressData)


@pytest.mark.asyncio
async def test_upsert_replaces_parts_preserves_created_at() -> None:
    thread_repo = InMemoryThreadRepository()
    thread = await thread_repo.create(agent="test")

    first = await thread_repo.upsert_message(
        thread.id, id="msg-1", role="assistant",
        parts=[{"type": "text", "text": "hi", "state": "done"}],
    )
    second = await thread_repo.upsert_message(
        thread.id, id="msg-1", role="assistant",
        parts=[{"type": "text", "text": "bye", "state": "done"}],
    )
    assert first.id == second.id == "msg-1"
    assert first.created_at == second.created_at
    history = await thread_repo.history(thread.id)
    assert len(history) == 1
    assert history[0].parts[0]["text"] == "bye"
