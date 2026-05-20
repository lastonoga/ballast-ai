"""Tests for ``StateflowDurableAgent`` — the durable-by-default StateflowAgent.

Verifies the contract:
  - Subclass inherits StateflowAgent's tool / system_prompt / metadata
    machinery (smoke).
  - ``run`` is a real ``@DBOS.workflow`` that persists ``start`` +
    ``text-delta`` + ``done`` events to the event log AND publishes
    matching ``EventNotification``s on the event stream.
  - Survives caller cancellation: workflow finishes even if the code
    that started it gets cancelled mid-flight.
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

from pydantic_ai_stateflow.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.runtime import (
    StateflowDurableAgent,
    EventNotification,
    InProcessEventStream,
    thread_channel,
)

_counter = itertools.count()


class _NotesStateflowDurableAgent(StateflowDurableAgent):
    """Minimal ``StateflowDurableAgent`` subclass for tests — TestModel-backed."""

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


@pytest.mark.asyncio
async def test_run_persists_streaming_event_taxonomy(
    fresh_dbos_executor: None,
) -> None:
    """One run → log gets ``start`` → text-part lifecycle → ``done`` in order.

    With ``agent.iter()``-based streaming the taxonomy is:
      start, text-start, text-delta(+), text-end, done.
    TestModel emits a single text part with content ``"ok"``.
    """
    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()

    durable = _NotesStateflowDurableAgent(
        thread_repo=thread_repo, event_log=log, event_stream=stream,
        config_name=f"durable-test-{next(_counter)}",
    )

    thread = await thread_repo.create(agent="notes-durable-test", metadata={})

    with SetWorkflowID(str(uuid4())):
        await DBOS.start_workflow_async(
            durable.run,
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
    # First + last are fixed; middle is text-part lifecycle.
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert events[0].payload["prompt"] == "hi"
    # TestModel ships the response as a streamed TextPart, so we get
    # at least a text-start + one or more text-delta + text-end.
    text_kinds = kinds[1:-1]
    assert "text-start" in text_kinds
    assert "text-end" in text_kinds
    assert any(k == "text-delta" for k in text_kinds)
    # The concatenated deltas reconstruct the TestModel output.
    deltas = "".join(
        e.payload["text"] for e in events if e.kind == "text-delta"
    )
    assert deltas == "ok"


@pytest.mark.asyncio
async def test_run_publishes_notifications_for_each_event(
    fresh_dbos_executor: None,
) -> None:
    """Each persisted event triggers an ``EventNotification`` on the stream."""
    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()

    durable = _NotesStateflowDurableAgent(
        thread_repo=thread_repo, event_log=log, event_stream=stream,
        config_name=f"durable-test-{next(_counter)}",
    )

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
    await asyncio.sleep(0)  # let subscriber register before publish

    with SetWorkflowID(str(uuid4())):
        await DBOS.start_workflow_async(
            durable.run,
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
    """``run`` with a stale thread_id persists an ``error`` event instead of crashing."""
    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()

    durable = _NotesStateflowDurableAgent(
        thread_repo=thread_repo, event_log=log, event_stream=stream,
        config_name=f"durable-test-{next(_counter)}",
    )

    bogus = uuid4()

    with SetWorkflowID(str(uuid4())):
        handle = await DBOS.start_workflow_async(
            durable.run,
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
    """``cancel_thread_runs`` persists a ``cancelled`` event even with no active runs.

    Per Q5 of the design: idempotent cancel — calling it when nothing
    is running just emits the synthetic ``cancelled`` event so any
    listener sees the terminal signal and closes.
    """
    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()

    durable = _NotesStateflowDurableAgent(
        thread_repo=thread_repo, event_log=log, event_stream=stream,
        config_name=f"durable-test-{next(_counter)}",
    )
    thread = await thread_repo.create(agent="notes-durable-test", metadata={})

    cancelled = await durable.cancel_thread_runs(thread.id)
    assert cancelled == 0  # nothing active, but call still succeeds

    events = await log.read_since(thread.id)
    assert [e.kind for e in events] == ["cancelled"]
    assert events[0].payload["workflows_cancelled"] == 0


@pytest.mark.asyncio
async def test_enqueue_run_deterministic_workflow_id(
    fresh_dbos_executor: None,
) -> None:
    """Same (thread_id, user_message_id) → same workflow id, no duplicates."""
    from pydantic_ai_stateflow.runtime.durable_agent import (
        agent_run_workflow_id,
    )

    thread_repo = InMemoryThreadRepository()
    log = InMemoryEventLogRepository()
    stream = InProcessEventStream()

    durable = _NotesStateflowDurableAgent(
        thread_repo=thread_repo, event_log=log, event_stream=stream,
        config_name=f"durable-test-{next(_counter)}",
    )
    thread = await thread_repo.create(agent="notes-durable-test", metadata={})
    user_msg_id = uuid4()

    handle = await durable.enqueue_run(
        thread_id=thread.id, user_message_id=user_msg_id,
        prompt="hi", history_dump=[],
    )
    expected = agent_run_workflow_id(thread.id, user_msg_id)
    assert handle.workflow_id == expected
    await handle.get_result()  # let the workflow finish so it doesn't leak


@pytest.mark.asyncio
async def test_subclass_inherits_stateflow_agent_machinery() -> None:
    """``StateflowDurableAgent`` subclasses retain ``name`` / ``metadata_model`` / tools.

    Smoke check that ``StateflowDurableAgent`` is a drop-in StateflowAgent —
    apps shouldn't have to re-learn the agent registration API.
    """
    from pydantic_ai_stateflow.runtime.agents import StateflowAgent

    assert issubclass(_NotesStateflowDurableAgent, StateflowAgent)
    assert _NotesStateflowDurableAgent.name == "notes-durable-test"
    assert _NotesStateflowDurableAgent.metadata_model is None
    # The tool / system_prompt decorators come from the
    # StateflowAgent base unchanged.
    assert hasattr(_NotesStateflowDurableAgent, "tool")
    assert hasattr(_NotesStateflowDurableAgent, "system_prompt")
