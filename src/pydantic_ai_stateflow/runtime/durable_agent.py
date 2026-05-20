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

from dbos import DBOS, DBOSConfiguredInstance

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

    from pydantic_ai_stateflow.persistence.thread.repository import (
        ThreadRepository,
    )

_instance_counter = itertools.count()


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
