"""Tests for ``DurableAgent`` — the durable-by-default BallastAgent.

The agent's ``__init__`` no longer takes the infra triplet — the
process-wide ``Engine`` (built by ``ballast.create_app``) supplies repos +
event log + stream via ``get_ballast()`` whenever the DBOS workflow
body needs them.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any
from uuid import UUID, uuid4

import pytest
from dbos import DBOS, SetWorkflowID
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel

from ballast.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from ballast.runtime import (
    EventNotification,
    InProcessEventStream,
    DurableAgent,
    thread_channel,
)
from ballast.runtime.engine import (
    Engine,
    _reset_ballast_for_tests,
    _set_ballast,
)

_counter = itertools.count()


class _NotesDurableAgent(DurableAgent):
    """Minimal ``DurableAgent`` subclass for tests — TestModel-backed."""

    name = "notes-durable-test"
    metadata_model = None

    def build_agent(self) -> Agent[Any, str]:
        return Agent(
            TestModel(custom_output_text="ok"),
            output_type=str,
        )

    async def build_deps(
        self, *, thread: Any, message: ModelMessage | None,
    ) -> None:
        del thread, message
        return None


def _build(thread_repo, log, stream) -> _NotesDurableAgent:
    """Return a durable-agent instance for tests.

    Also installs a fresh process-wide ``Engine`` so the workflow body's
    ``get_ballast()`` resolves to the test repos.
    """
    durable = _NotesDurableAgent(
        config_name=f"durable-test-{next(_counter)}",
    )
    _reset_ballast_for_tests()
    _set_ballast(Engine(
        thread_repo=thread_repo, event_log=log, event_stream=stream,
    ))
    return durable


@pytest.mark.asyncio
async def test_run_persists_streaming_event_taxonomy(
    fresh_dbos_executor: None,
) -> None:
    """One run → log gets ``start`` → text-part lifecycle → ``done`` in order."""
    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()
    durable = _build(thread_repo, log, stream)

    thread = await thread_repo.create(agent="notes-durable-test", metadata={})

    with SetWorkflowID(str(uuid4())):
        await DBOS.start_workflow_async(
            durable._run_with_tracking,
            thread_id_str=str(thread.id),
            prompt="hi",
            history_dump=[],
        )

    for _ in range(200):
        events = await log.read_since(thread.id)
        if events and events[-1].kind == "done":
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("Workflow did not produce a 'done' event in time")

    kinds = [e.kind for e in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert events[0].payload["prompt"] == "hi"
    text_kinds = kinds[1:-1]
    assert "text-start" in text_kinds
    assert "text-end" in text_kinds
    assert any(k == "text-delta" for k in text_kinds)
    deltas = "".join(
        e.payload["text"] for e in events if e.kind == "text-delta"
    )
    assert deltas == "ok"


@pytest.mark.asyncio
async def test_run_publishes_notifications_for_each_event(
    fresh_dbos_executor: None,
) -> None:
    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()
    durable = _build(thread_repo, log, stream)

    thread = await thread_repo.create(agent="notes-durable-test", metadata={})
    channel = thread_channel(thread.id)

    received: list[EventNotification] = []

    async def consume() -> None:
        async with stream.subscribe(channel) as events:
            async for n in events:
                received.append(n)
                if n.seq == 3:
                    return

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)

    with SetWorkflowID(str(uuid4())):
        await DBOS.start_workflow_async(
            durable._run_with_tracking,
            thread_id_str=str(thread.id),
            prompt="hi",
            history_dump=[],
        )

    await asyncio.wait_for(consumer, timeout=2.0)
    assert [n.seq for n in received] == [1, 2, 3]
    assert all(n.thread_id == thread.id for n in received)


@pytest.mark.asyncio
async def test_run_emits_error_event_when_thread_missing(
    fresh_dbos_executor: None,
) -> None:
    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()
    durable = _build(thread_repo, log, stream)

    bogus = uuid4()

    with SetWorkflowID(str(uuid4())):
        handle = await DBOS.start_workflow_async(
            durable._run_with_tracking,
            thread_id_str=str(bogus),
            prompt="hi",
            history_dump=[],
        )
    await handle.get_result()

    events = await log.read_since(bogus)
    assert [e.kind for e in events] == ["error"]
    assert "not found" in events[0].payload["message"].lower()


@pytest.mark.asyncio
async def test_cancel_thread_runs_emits_cancelled_event(
    fresh_dbos_executor: None,
) -> None:
    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()
    durable = _build(thread_repo, log, stream)
    thread = await thread_repo.create(agent="notes-durable-test", metadata={})

    cancelled = await durable.cancel_thread_runs(thread.id)
    assert cancelled == 0

    events = await log.read_since(thread.id)
    assert [e.kind for e in events] == ["cancelled"]
    assert events[0].payload["workflows_cancelled"] == 0


@pytest.mark.asyncio
async def test_enqueue_run_deterministic_workflow_id(
    fresh_dbos_executor: None,
) -> None:
    from ballast.runtime.durable_agent import (
        agent_run_workflow_id,
    )

    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()
    durable = _build(thread_repo, log, stream)
    thread = await thread_repo.create(agent="notes-durable-test", metadata={})
    user_msg_id = str(uuid4())

    handle = await durable.enqueue_run(
        thread_id=thread.id, user_message_id=user_msg_id,
        prompt="hi", history_dump=[],
    )
    expected = agent_run_workflow_id(thread.id, user_msg_id)
    assert handle.workflow_id == expected
    await handle.get_result()


@pytest.mark.asyncio
async def test_subclass_inherits_stateflow_agent_machinery() -> None:
    """``DurableAgent`` subclasses retain ``name`` / ``metadata_model`` / tools."""
    from ballast.runtime.agents import BallastAgent

    assert issubclass(_NotesDurableAgent, BallastAgent)
    assert _NotesDurableAgent.name == "notes-durable-test"
    assert _NotesDurableAgent.metadata_model is None
    assert hasattr(_NotesDurableAgent, "tool")
    assert hasattr(_NotesDurableAgent, "system_prompt")
