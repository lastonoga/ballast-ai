"""``StateflowDurableAgent`` — ``StateflowAgent`` variant with durable run loop.

The motivating problem (from the design discussion):

  ``StateflowAgent.agent.run_stream(...)`` runs inside the FastAPI
  request handler's asyncio task. When the SSE consumer dies (browser
  tab closed, network blip, request timeout), the task is cancelled,
  ``CancelledError`` cascades down to every ``await`` in tool bodies,
  and any side effects depending on the model's response are lost.
  ``DurableHITLWorkflow`` works around this for HITL specifically by
  spawning a separate ``@DBOS.workflow``; ``StateflowDurableAgent`` solves it
  at the source — the WHOLE agent run lives inside a workflow.

What this buys:

  - Caller cancellation only kills the HTTP response, not the agent
    run. The workflow continues; events keep landing in the durable
    log; on reconnect the SSE handler replays missed events via
    Last-Event-ID.
  - Process crash recovers via DBOS's standard workflow recovery —
    the agent run resumes from the last persisted step.
  - HITL becomes a regular ``await hitl_gate.ask_helper(...)`` from
    inside a tool — the workflow boundary protects it. No more
    ``DurableHITLWorkflow`` + ``on_decision`` boilerplate for typical
    HITL flows.

What this costs:

  - Tool side effects MUST be idempotent (DBOS replays workflow
    steps on recovery). The framework's ``Det.now / uuid_for / random_*``
    helpers handle non-determinism, but app-side tools that hit
    external systems (DB writes, HTTP POSTs, payment processors)
    must be wrapped in ``@DBOS.step`` with idempotency keys —
    capability ``IdempotentTools`` (separate task #128) automates
    this opt-in.
  - Performance: every persisted event is a row write + signal
    publish. For very high-throughput agent runs swap the in-memory
    log + in-process stream for postgres / Redis.
  - Per-thread serialization: only one ``StateflowDurableAgent.run`` can be
    in-flight per thread at a time (DBOS queue policy, task #127).

Apps adopt ``StateflowDurableAgent`` by subclassing it instead of
``StateflowAgent`` — the rest of the contract (``build_agent``,
``build_deps``, ``model_settings``, ``@SomeAgent.tool``,
``@SomeAgent.system_prompt``, ``metadata_model``) is unchanged.
"""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING, Any
from uuid import UUID

from dbos import (
    DBOS,
    DBOSConfiguredInstance,
    Queue,
    SetEnqueueOptions,
    SetWorkflowID,
)

from pydantic_ai_stateflow.persistence.events.repository import (
    EventLogRepository,
)
from pydantic_ai_stateflow.runtime.agents import StateflowAgent, _ToolEntry
from pydantic_ai_stateflow.runtime.event_stream import (
    EventNotification,
    EventStream,
    thread_channel,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dbos._dbos import WorkflowHandleAsync

    from pydantic_ai_stateflow.persistence.thread.repository import (
        ThreadRepository,
    )

_instance_counter = itertools.count()


# Per-thread serialization queue.
#
# Single global DBOS queue with ``partition_queue=True`` + ``concurrency=1``.
# Each ``thread_id`` acts as a partition key — at most ONE workflow runs at
# a time per thread, but different threads run concurrently. This prevents
# the race where two user messages in the same thread spawn parallel agent
# runs that interleave writes / step on each other's message history /
# duplicate side effects.
#
# Module-level so DBOS sees the registration BEFORE ``DBOS.launch()`` runs
# in apps that import this module at boot.
AGENT_RUN_QUEUE: Queue = Queue(
    name="stateflow-agent-runs",
    concurrency=1,
    partition_queue=True,
)


def agent_run_workflow_id(thread_id: UUID, user_message_id: UUID) -> str:
    """Deterministic workflow id for one (thread, user message) pair.

    Re-using the same id idempotently attaches a request retry to the
    in-flight workflow instead of spawning a duplicate. The prefix
    ``"agent-run:"`` is used by ``cancel_thread_runs`` to find all
    workflows for a thread via ``list_workflows_async(workflow_id_prefix=...)``.
    """
    return f"agent-run:{thread_id}:{user_message_id}"


def _agent_run_prefix(thread_id: UUID) -> str:
    """All workflow ids for ``thread_id`` start with this prefix."""
    return f"agent-run:{thread_id}:"


@DBOS.dbos_class()
class StateflowDurableAgent(StateflowAgent, DBOSConfiguredInstance):
    """``StateflowAgent`` whose ``run`` is a durable DBOS workflow.

    Inherits everything from ``StateflowAgent`` — tool decorators,
    system-prompt decorators, ``metadata_model`` validation, the
    lazy-cached pydantic-ai ``Agent`` build. Apps subclass exactly
    the same way; the difference is internal (the run loop is now a
    ``@DBOS.workflow``).

    Three transport dependencies wired at construction:

      - ``thread_repo``: load thread + (separately) persist messages.
      - ``event_log``:  append every emitted event for durable replay.
      - ``event_stream``: publish ``EventNotification(seq)`` so live
        SSE consumers wake up without polling the log.

    The streaming router checks for ``isinstance(instance,
    StateflowDurableAgent)`` and routes through the durable path; plain
    ``StateflowAgent`` subclasses keep the current direct streaming
    path. Apps opt in by choosing which base class to extend.
    """

    def __init__(
        self,
        *,
        thread_repo: ThreadRepository,
        event_log: EventLogRepository,
        event_stream: EventStream,
        config_name: str | None = None,
    ) -> None:
        # DBOSConfiguredInstance needs a unique config_name per process
        # so DBOS can rebind the instance to in-flight workflows after a
        # restart. Default to ``durable-agent:<AgentClassName>`` — apps
        # with multiple instances of the same class (rare) override
        # with their own stable string.
        cls_name = type(self).__name__
        super().__init__(
            config_name=config_name
            or f"durable-agent:{cls_name}-{next(_instance_counter)}",
        )
        self._thread_repo = thread_repo
        self._event_log = event_log
        self._event_stream = event_stream

    def _wrap_tool_fn(
        self,
        fn: Callable[..., Any],
        entry: _ToolEntry,
    ) -> Callable[..., Any]:
        """Wrap tools with ``persistent=True`` in ``@DBOS.step`` for replay safety.

        Why: when the agent loop runs inside ``@DBOS.workflow`` and the
        process crashes mid-run, DBOS recovery REPLAYS the workflow
        from the start — every tool call re-executes. Read-only tools
        (``list_notes``, ``search_notes``) are fine; tools that hit
        external systems (DB writes, API POSTs, payments) would
        double-fire.

        Wrapping the tool function in ``@DBOS.step`` memoizes its
        return value in DBOS's system database — replay sees a
        recorded step and returns the cached result without
        re-executing the body.

        Default policy for ``StateflowDurableAgent``: ``persistent=None``
        (the unset default) is treated as ``True`` — apps get safety
        by default and have to explicitly opt out for read-only tools
        with ``@SomeAgent.tool(persistent=False)``. Trading per-call
        DBOS-sys-db write overhead for "won't silently double-create
        a row on crash recovery" is the right default for an agent
        framework targeting production reliability.
        """
        # ``None`` (unset) → DurableAgent default = persist.
        # ``False`` → explicit opt-out (read-only tool).
        # ``True`` → explicit opt-in.
        if entry.persistent is False:
            return fn

        # ``@DBOS.step`` wants a unique name per registered step. The
        # tool's qualified name is stable across replays and unique
        # within an agent run, which is what DBOS's memoization key
        # needs.
        step_name = f"tool:{fn.__qualname__}"
        wrapped = DBOS.step(name=step_name)(fn)

        # Preserve introspection metadata that pydantic-ai's tool
        # registration reads (it uses ``get_type_hints`` + the function
        # signature to derive the JSON schema). ``functools.wraps`` on
        # the DBOS-step return value would clobber the step's behavior,
        # so re-mirror just the attributes pydantic-ai actually reads.
        functools.update_wrapper(
            wrapped, fn,
            assigned=("__module__", "__name__", "__qualname__",
                      "__doc__", "__annotations__"),
            updated=(),
        )
        return wrapped

    @DBOS.step()
    async def _persist_and_publish(
        self,
        *,
        thread_id: UUID,
        kind: str,
        payload: dict[str, Any],
    ) -> int:
        """Append one event to the durable log + publish a wake-up signal.

        Wrapped as ``@DBOS.step`` so DBOS records the operation in its
        execution log — recovery skips the step if it already ran,
        which is the safety net for non-idempotent log appends across
        workflow replays.

        Returns the assigned ``seq`` so callers can correlate the
        durable row with the signal that announced it.
        """
        ev = await self._event_log.append(
            thread_id=thread_id, kind=kind, payload=dict(payload),
        )
        await self._event_stream.publish(
            thread_channel(thread_id),
            EventNotification(thread_id=thread_id, seq=ev.seq),
        )
        return ev.seq

    async def enqueue_run(
        self,
        *,
        thread_id: UUID,
        user_message_id: UUID,
        prompt: str,
        history_dump: list[dict[str, Any]],
    ) -> WorkflowHandleAsync[None]:
        """Enqueue ``self.run`` into the per-thread serialization queue.

        Returns the DBOS handle for the enqueued workflow. The workflow
        id is deterministic per (thread_id, user_message_id) so a
        retried request attaches to the existing run instead of
        spawning a duplicate.

        The partition key is the stringified ``thread_id`` —
        ``AGENT_RUN_QUEUE`` is configured with ``concurrency=1``, so at
        most one workflow runs per thread at a time. Other messages
        for the SAME thread wait in the queue; other threads run
        concurrently.
        """
        workflow_id = agent_run_workflow_id(thread_id, user_message_id)
        # ``SetWorkflowID`` pre-allocates the id; ``SetEnqueueOptions``
        # routes the enqueue into the right partition. Both are
        # context-managers that stack on the per-task DBOS context.
        # ``enqueue_async`` (not ``enqueue``) is required in async code —
        # the sync variant returns a ``WorkflowHandle`` that can't be
        # awaited.
        with SetWorkflowID(workflow_id), SetEnqueueOptions(
            queue_partition_key=str(thread_id),
        ):
            return await AGENT_RUN_QUEUE.enqueue_async(
                self.run,
                thread_id_str=str(thread_id),
                prompt=prompt,
                history_dump=history_dump,
            )

    async def cancel_thread_runs(self, thread_id: UUID) -> int:
        """Cancel every active workflow for ``thread_id`` + emit a ``cancelled`` event.

        Active = ``ENQUEUED`` or ``PENDING`` (i.e. waiting in queue
        or already running). ``SUCCESS`` / ``ERROR`` / ``CANCELLED``
        workflows are left alone — calling cancel on a finished
        workflow is a no-op (we follow Q5: idempotent cancel).

        Returns the number of workflows that were actually cancelled
        so callers can surface "nothing to cancel" vs "1 cancelled"
        in their HTTP response.
        """
        # Both ENQUEUED (queued, not yet running) and PENDING (running)
        # are cancellable. DELAYED is a future-timer state we don't
        # emit but check defensively.
        active_statuses = ["ENQUEUED", "PENDING", "DELAYED"]
        prefix = _agent_run_prefix(thread_id)
        workflows = await DBOS.list_workflows_async(
            workflow_id_prefix=prefix,
            status=active_statuses,
            limit=100,
        )
        cancelled = 0
        for wf in workflows:
            await DBOS.cancel_workflow_async(wf.workflow_id)
            cancelled += 1

        # Synthetic terminal event so the SSE consumer sees something
        # in the log and closes — the cancelled workflow itself may
        # not get a chance to emit anything (DBOS cancellation just
        # marks the row + interrupts the task).
        await self._event_log.append(
            thread_id=thread_id,
            kind="cancelled",
            payload={"workflows_cancelled": cancelled},
        )
        await self._event_stream.publish(
            thread_channel(thread_id),
            EventNotification(thread_id=thread_id, seq=0),
        )
        return cancelled

    @DBOS.workflow()
    async def run(
        self,
        *,
        thread_id_str: str,
        prompt: str,
        history_dump: list[dict[str, Any]],
    ) -> None:
        """Durable agent run — replaces ``agent.run_stream`` in the request handler.

        Args are JSON-friendly primitives so DBOS workflow
        serialization is robust across pydantic / pickle version
        changes:

          - ``thread_id_str``: stringified UUID of the target thread.
          - ``prompt``: extracted user-message text (the streaming
            router pulls this out of the Vercel-AI request body).
          - ``history_dump``: ``[m.model_dump(mode="json") for m in
            messages_to_model_history(...)]`` — replay-safe.

        The model output is currently persisted as a single
        ``text-delta`` event followed by a ``done`` marker. Token-
        level streaming + tool-call events arrive in task #127 when
        the streaming router is rewired to drive this workflow and
        consume the live ``agent.iter()`` graph.
        """
        from pydantic_ai.messages import (  # noqa: PLC0415
            ModelMessage,
            ModelMessagesTypeAdapter,
        )
        from pydantic_ai.usage import UsageLimits  # noqa: PLC0415
        del UsageLimits  # placeholder import — wired in #127

        thread_id = UUID(thread_id_str)
        thread = await self._thread_repo.load(thread_id)
        if thread is None:
            await self._persist_and_publish(
                thread_id=thread_id,
                kind="error",
                payload={"message": f"Thread {thread_id} not found"},
            )
            return

        # Rehydrate ModelMessage history. ``ModelMessagesTypeAdapter`` is
        # the canonical pydantic-ai way to round-trip a list of
        # ModelMessage dicts back into typed objects.
        history: list[ModelMessage] = (
            ModelMessagesTypeAdapter.validate_python(history_dump)
            if history_dump else []
        )

        deps = await self.build_deps(thread=thread, message=None)
        model_settings = self.model_settings()

        await self._persist_and_publish(
            thread_id=thread_id, kind="start", payload={"prompt": prompt},
        )

        try:
            result = await self.agent.run(
                prompt,
                message_history=history,
                deps=deps,
                model_settings=model_settings,
            )
        except Exception as exc:
            await self._persist_and_publish(
                thread_id=thread_id,
                kind="error",
                payload={"message": str(exc), "type": type(exc).__name__},
            )
            raise

        output = result.output
        output_text = output if isinstance(output, str) else str(output)
        await self._persist_and_publish(
            thread_id=thread_id,
            kind="text-delta",
            payload={"text": output_text},
        )
        await self._persist_and_publish(
            thread_id=thread_id, kind="done", payload={},
        )


__all__ = ["StateflowDurableAgent"]
