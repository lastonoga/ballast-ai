"""Unit tests for ``Infra`` + ``RunContext``."""
from __future__ import annotations

from uuid import uuid4

import pytest

from pydantic_ai_stateflow.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.runtime.event_stream import InProcessEventStream
from pydantic_ai_stateflow.runtime.infra import Infra, RunContext
from pydantic_ai_stateflow.runtime.thread_events import ThreadEventBroadcaster


def _build_infra() -> Infra:
    return Infra(
        thread_repo=InMemoryThreadRepository(),
        event_log=InMemoryEventLogRepository(),
        event_stream=InProcessEventStream(),
    )


def test_infra_broadcaster_cached() -> None:
    infra = _build_infra()
    assert infra.broadcaster is infra.broadcaster
    assert isinstance(infra.broadcaster, ThreadEventBroadcaster)


def test_context_inherits_infra() -> None:
    infra = _build_infra()
    ctx = infra.context()
    assert ctx.thread_repo is infra.thread_repo
    assert ctx.event_log is infra.event_log
    assert ctx.event_stream is infra.event_stream
    assert ctx.parent_thread_id is None
    assert ctx.workflow_id is None


def test_context_with_per_call_fields() -> None:
    infra = _build_infra()
    parent = uuid4()
    ctx = infra.context(parent_thread_id=parent, workflow_id="wf-1")
    assert ctx.parent_thread_id == parent
    assert ctx.workflow_id == "wf-1"


def test_context_broadcaster_cached() -> None:
    infra = _build_infra()
    ctx = infra.context()
    assert ctx.broadcaster is ctx.broadcaster


def test_context_with_() -> None:
    infra = _build_infra()
    ctx = infra.context()
    ctx2 = ctx.with_(workflow_id="wf-2")
    assert ctx.workflow_id is None
    assert ctx2.workflow_id == "wf-2"
    assert ctx2.thread_repo is ctx.thread_repo


def test_infra_is_frozen() -> None:
    infra = _build_infra()
    with pytest.raises(Exception):
        infra.thread_repo = InMemoryThreadRepository()  # type: ignore[misc]


def test_runcontext_is_frozen() -> None:
    infra = _build_infra()
    ctx = infra.context()
    with pytest.raises(Exception):
        ctx.parent_thread_id = uuid4()  # type: ignore[misc]
