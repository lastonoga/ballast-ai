"""``StateflowDurableAgent`` — ``StateflowAgent`` durably executed via DBOSAgent.

Built on top of pydantic-ai's ``DBOSAgent`` wrapper, which:

  - Wraps the underlying ``Agent`` so that every **model request** and
    **MCP toolset op** is automatically a ``@DBOS.step``. Regular
    ``@SomeAgent.tool`` functions are NOT step-wrapped — they run
    inline in workflow context. On replay their side effects re-fire,
    and they don't appear in the DBOS step log. Apps that need
    idempotency for tools must implement it themselves (e.g.
    ``INSERT ... ON CONFLICT DO NOTHING`` or manual ``@DBOS.step``).
  - Exposes ``run()`` as a ``@DBOS.workflow`` — the whole agent loop
    is durable end-to-end.
  - Supports ``event_stream_handler`` for streaming AgentStreamEvents
    out of the workflow without having to drive ``agent.iter()``
    manually.

What our subclass adds on top:

  - An outer per-turn workflow (``_run_with_tracking``) that:
    1. Emits a ``start`` ThreadEvent to the durable log.
    2. Calls ``DBOSAgent.run(...)`` (a child workflow).
    3. Routes every AgentStreamEvent through the event-stream handler →
       ``ThreadEvent`` rows + ``EventNotification`` signals so the SSE
       consumer can replay/tail.
    4. Persists the assistant turn to ``thread_repo``.
    5. Emits ``approval-request`` ThreadEvents per ``DeferredToolRequests``
       approval entry (HITL: tool needs user approve/reject).
    6. Emits ``done`` (or ``error``).

  - A per-thread serialization queue (``AGENT_RUN_QUEUE`` with
    ``partition_queue=True`` + ``concurrency=1``) so a thread can't
    have two agent turns interleaving.

  - ``enqueue_run`` (fresh user prompt) and ``enqueue_approval_resume``
    (resume after the user supplied ``DeferredToolResults`` via the
    ``POST /threads/{id}/approve`` endpoint).

  - ``cancel_thread_runs`` — best-effort cancel of every active workflow
    for a thread + a synthetic ``cancelled`` ThreadEvent so the SSE
    consumer closes.

Apps still subclass ``StateflowDurableAgent`` exactly like
``StateflowAgent``; ``build_agent`` / ``build_deps`` / ``model_settings``
/ ``@SomeAgent.tool`` are unchanged. The DBOSAgent wrapping is internal.
"""

from __future__ import annotations

import itertools
from contextvars import ContextVar
from functools import cached_property
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from dbos import (
    DBOS,
    DBOSConfiguredInstance,
    Queue,
    SetEnqueueOptions,
    SetWorkflowID,
)

from pydantic_ai_stateflow.durable import Durable

from pydantic_ai_stateflow.runtime.agents import StateflowAgent
from pydantic_ai_stateflow.runtime.event_stream import (
    EventNotification,
    thread_channel,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterable

    from dbos._dbos import WorkflowHandleAsync
    from pydantic_ai.durable_exec.dbos import DBOSAgent
    from pydantic_ai.messages import AgentStreamEvent
    from pydantic_ai.tools import RunContext as PydanticAIRunContext

    from pydantic_ai_stateflow.persistence.events.repository import (
        EventLogRepository,
    )
    from pydantic_ai_stateflow.persistence.thread.repository import (
        ThreadRepository,
    )
    from pydantic_ai_stateflow.runtime.event_stream import EventStream
    from pydantic_ai_stateflow.runtime.infra import RunContext

_instance_counter = itertools.count()


# Per-thread serialization queue. ``partition_queue=True`` + ``concurrency=1``
# means at most ONE workflow runs at a time per (thread_id) partition,
# but different threads run concurrently. This serializes turns within
# a thread (no two parallel agent runs writing assistant turns over
# each other) without blocking the whole app.
#
# Module-level so DBOS sees the registration BEFORE ``Durable.launch()``.
AGENT_RUN_QUEUE: Queue = Queue(
    name="stateflow-agent-runs",
    concurrency=1,
    partition_queue=True,
)


def agent_run_workflow_id(thread_id: UUID, user_message_id: str) -> str:
    """Deterministic workflow id for one (thread, user message) pair.

    Same id idempotently attaches a request retry to the in-flight
    workflow instead of spawning a duplicate. The prefix
    ``"agent-run:"`` is used by ``cancel_thread_runs`` to find all
    workflows for a thread via
    ``list_workflows_async(workflow_id_prefix=...)``.
    """
    return f"agent-run:{thread_id}:{user_message_id}"


def _agent_resume_workflow_id(thread_id: UUID, suffix: str) -> str:
    """Workflow id for an approval-resume run. Distinct from the
    initial run's id so the queue treats it as a fresh turn but the
    prefix still matches for cancellation."""
    return f"agent-run:{thread_id}:resume:{suffix}"


def _agent_run_prefix(thread_id: UUID) -> str:
    """All workflow ids for ``thread_id`` start with this prefix."""
    return f"agent-run:{thread_id}:"


# Thread-id carried through the call stack so the event-stream handler
# (set on DBOSAgent at construction; doesn't get explicit thread context
# in its callback) knows which thread to write events to. ContextVar
# propagates across asyncio await boundaries within a task.
_current_thread_id: ContextVar[UUID | None] = ContextVar(
    "_stateflow_current_thread_id", default=None,
)


@Durable.dbos_class()
class StateflowDurableAgent(StateflowAgent, DBOSConfiguredInstance):
    """``StateflowAgent`` whose agent loop runs as a DBOS workflow.

    Subclasses provide the usual ``build_agent`` / ``build_deps`` /
    ``model_settings`` / ``@SomeAgent.tool`` machinery from
    ``StateflowAgent``. This subclass wraps the resulting pydantic-ai
    ``Agent`` in ``DBOSAgent``, so model requests and tool calls
    become durable steps automatically.

    Three transport dependencies wired at construction:

      - ``thread_repo``: load thread + persist messages.
      - ``event_log``:   append every emitted event for SSE replay.
      - ``event_stream``: publish ``EventNotification(seq)`` so live
        SSE consumers wake without polling the log.

    The streaming router checks ``isinstance(instance,
    StateflowDurableAgent)`` and routes through the durable path
    (``enqueue_run`` + SSE-tail-the-event-log). Plain ``StateflowAgent``
    subclasses keep the non-durable inline streaming path.
    """

    def __init__(
        self,
        *,
        config_name: str | None = None,
    ) -> None:
        # DBOSConfiguredInstance requires a stable name so DBOS can
        # rebind the instance to in-flight workflows after restart.
        # Default to ``cls.__qualname__`` — the common case is one
        # instance per class in a process, and the class name is the
        # most stable identifier across restarts (survives module-path
        # changes, doesn't depend on instantiation order).
        # Apps with multiple instances of the same class (multi-tenant,
        # per-test isolation) override explicitly.
        super().__init__(
            config_name=config_name or type(self).__qualname__,
        )
        # Per-call infra triplet. Populated by ``enqueue_run`` /
        # ``enqueue_approval_resume`` / ``cancel_thread_runs`` from
        # the supplied ``RunContext``. The DBOS workflow body reads
        # them off ``self`` (workflow args are pickled — repo objects
        # are not picklable, so the indirection is required).
        self._thread_repo: ThreadRepository | None = None
        self._event_log: EventLogRepository | None = None
        self._event_stream: EventStream | None = None

    def _bind_infra(self, ctx: "RunContext") -> None:
        """Stash the per-call infra triplet on the instance.

        ``StateflowDurableAgent`` is built once at app startup and reused
        across requests; the DBOS workflow body needs to read repos off
        the instance (workflow args are pickled, repos are not picklable).
        Each ``enqueue_*`` call rebinds the triplet from the supplied
        ``RunContext`` so the workflow sees the right repos for this
        request.
        """
        self._thread_repo = ctx.thread_repo
        self._event_log = ctx.event_log
        self._event_stream = ctx.event_stream

    @cached_property
    def dbos_agent(self) -> DBOSAgent[Any, Any]:
        """Lazy-cached ``DBOSAgent`` wrapping the base pydantic-ai ``Agent``.

        Constructed once on first access. The wrapped agent inherits
        all tools / system prompts / metadata model registered via
        ``@SomeAgent.tool`` / ``@SomeAgent.system_prompt`` on the
        subclass — those are baked into ``self.agent`` by the base
        class machinery.

        ``event_stream_handler`` is set on the ``DBOSAgent`` instance
        here (not passed per-call to ``run``) so it doesn't end up in
        the workflow's arg tuple — DBOS pickles workflow args, and a
        bound method on this instance pulls in non-picklable state
        (asyncio primitives on the event_stream, ContextVar refs, …).
        Setting it on the instance makes the handler a property the
        workflow reads at runtime instead of an arg it serializes.
        """
        from pydantic_ai.durable_exec.dbos import DBOSAgent  # noqa: PLC0415

        # ``name`` is the DBOS configured-instance key + workflow-name
        # prefix. Must be unique across DBOS instances in this process,
        # so we use ``self.config_name`` (set in __init__ from the user
        # override or ``durable-agent:<ClassName>-<counter>``) rather
        # than ``self.name`` (the registry key, shared across
        # construction). For prod recovery, override config_name to a
        # stable string at construction time.
        return DBOSAgent(
            self.agent,
            name=self.config_name,
            event_stream_handler=self._handle_stream_event,
        )

    async def _handle_stream_event(
        self,
        ctx: "PydanticAIRunContext[Any]",
        stream: "AsyncIterable[AgentStreamEvent]",
    ) -> None:
        """``event_stream_handler`` callback — runs INSIDE the DBOS workflow.

        Translates each pydantic-ai ``AgentStreamEvent`` into a
        ``ThreadEvent`` row and publishes an ``EventNotification`` so
        the SSE consumer wakes. ``DBOSAgent`` wraps this callback so it
        receives a single-event stream per invocation; we still iterate
        defensively to support both wrapped and direct calling.

        The thread_id comes from the ``_current_thread_id`` ContextVar
        set by ``_run_with_tracking`` — it propagates through the
        workflow → child workflow → handler call chain because ContextVar
        is preserved across asyncio await boundaries in the same task.
        """
        del ctx
        thread_id = _current_thread_id.get()
        if thread_id is None:
            return
        async for event in stream:
            await self._translate_event_step(
                thread_id=thread_id, event=event,
            )

    @Durable.step()
    async def _translate_event_step(
        self,
        *,
        thread_id: UUID,
        event: Any,
    ) -> None:
        """Persist one ``AgentStreamEvent`` as a ``ThreadEvent`` row.

        Wrapped as ``@DBOS.step`` so persisted events are memoised on
        replay (we don't double-write on workflow recovery). Unknown
        event types are silently skipped (forward-compat for new
        pydantic-ai event kinds).
        """
        from pydantic_ai.messages import (  # noqa: PLC0415
            FunctionToolResultEvent,
            PartDeltaEvent,
            PartEndEvent,
            PartStartEvent,
            TextPart,
            TextPartDelta,
            ToolCallPart,
            ToolCallPartDelta,
        )

        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, TextPart):
                await self._persist_and_publish(
                    thread_id=thread_id,
                    kind="text-start",
                    payload={"part_index": event.index},
                )
                if part.content:
                    # Some providers ship the full text in PartStart
                    # (no streaming deltas). Surface it as a delta so
                    # downstream encoders don't lose it.
                    await self._persist_and_publish(
                        thread_id=thread_id,
                        kind="text-delta",
                        payload={
                            "part_index": event.index,
                            "text": part.content,
                        },
                    )
            elif isinstance(part, ToolCallPart):
                await self._persist_and_publish(
                    thread_id=thread_id,
                    kind="tool-call-start",
                    payload={
                        "part_index": event.index,
                        "tool_call_id": part.tool_call_id,
                        "tool_name": part.tool_name,
                        "args": part.args_as_dict() if part.args else {},
                    },
                )
            return

        if isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, TextPartDelta) and delta.content_delta:
                await self._persist_and_publish(
                    thread_id=thread_id,
                    kind="text-delta",
                    payload={
                        "part_index": event.index,
                        "text": delta.content_delta,
                    },
                )
            elif isinstance(delta, ToolCallPartDelta):
                payload: dict[str, Any] = {"part_index": event.index}
                if delta.args_delta is not None:
                    payload["args_delta"] = delta.args_delta
                if delta.tool_name_delta is not None:
                    payload["tool_name_delta"] = delta.tool_name_delta
                if delta.tool_call_id is not None:
                    payload["tool_call_id"] = delta.tool_call_id
                if len(payload) > 1:
                    await self._persist_and_publish(
                        thread_id=thread_id,
                        kind="tool-call-delta",
                        payload=payload,
                    )
            return

        if isinstance(event, PartEndEvent):
            part = event.part
            if isinstance(part, TextPart):
                await self._persist_and_publish(
                    thread_id=thread_id,
                    kind="text-end",
                    payload={"part_index": event.index},
                )
            elif isinstance(part, ToolCallPart):
                await self._persist_and_publish(
                    thread_id=thread_id,
                    kind="tool-call-end",
                    payload={
                        "part_index": event.index,
                        "tool_call_id": part.tool_call_id,
                        "tool_name": part.tool_name,
                        "args": part.args_as_dict() if part.args else {},
                    },
                )
            return

        if isinstance(event, FunctionToolResultEvent):
            result = event.part
            await self._persist_and_publish(
                thread_id=thread_id,
                kind="tool-result",
                payload={
                    "tool_call_id": result.tool_call_id,
                    "tool_name": result.tool_name,
                    "output": _safe_jsonify(
                        getattr(result, "content", None),
                    ),
                },
            )
            return

    async def _persist_and_publish(
        self,
        *,
        thread_id: UUID,
        kind: str,
        payload: dict[str, Any],
    ) -> int:
        """Append one event to the durable log + publish a wake-up signal.

        NOT decorated as ``@DBOS.step`` directly because this helper is
        also called from ``_translate_event_step`` (which IS a step) —
        nesting steps is fine but adds overhead. Callers that need step
        idempotency wrap their call site.
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
        ctx: "RunContext",
        *,
        thread_id: UUID,
        user_message_id: str,
        prompt: str,
        history_dump: list[dict[str, Any]],
    ) -> WorkflowHandleAsync[None]:
        """Enqueue a fresh-prompt agent turn into the per-thread queue.

        Returns the DBOS handle for the enqueued workflow. The id is
        deterministic per (thread_id, user_message_id) so a request
        retry attaches to the existing workflow instead of spawning a
        duplicate.
        """
        self._bind_infra(ctx)
        workflow_id = agent_run_workflow_id(thread_id, user_message_id)
        with SetWorkflowID(workflow_id), SetEnqueueOptions(
            queue_partition_key=str(thread_id),
        ):
            return await Durable.enqueue(
                AGENT_RUN_QUEUE,
                self._run_with_tracking,
                thread_id_str=str(thread_id),
                prompt=prompt,
                history_dump=history_dump,
                deferred_tool_results_dump=None,
            )

    async def enqueue_approval_resume(
        self,
        ctx: "RunContext",
        *,
        thread_id: UUID,
        history_dump: list[dict[str, Any]],
        approvals: dict[str, dict[str, Any]],
    ) -> WorkflowHandleAsync[None]:
        """Enqueue an approval-resume turn — agent re-runs with the
        user's ``DeferredToolResults`` to actually execute (or deny)
        the previously-paused tool call.

        ``approvals`` is a plain dict so DBOS workflow arg serialization
        survives library version changes; reconstructed into
        ``DeferredToolResults`` inside the workflow body.
        """
        self._bind_infra(ctx)
        suffix = uuid4().hex
        workflow_id = _agent_resume_workflow_id(thread_id, suffix)
        with SetWorkflowID(workflow_id), SetEnqueueOptions(
            queue_partition_key=str(thread_id),
        ):
            return await Durable.enqueue(
                AGENT_RUN_QUEUE,
                self._run_with_tracking,
                thread_id_str=str(thread_id),
                prompt="",
                history_dump=history_dump,
                deferred_tool_results_dump={"approvals": approvals},
            )

    async def cancel_thread_runs(
        self, ctx: "RunContext", thread_id: UUID,
    ) -> int:
        """Cancel every active workflow for ``thread_id`` + emit ``cancelled``."""
        self._bind_infra(ctx)
        active_statuses = ["ENQUEUED", "PENDING", "DELAYED"]
        prefix = _agent_run_prefix(thread_id)
        workflows = await Durable.list_workflows(
            workflow_id_prefix=prefix,
            status=active_statuses,
            limit=100,
        )
        cancelled = 0
        for wf in workflows:
            await Durable.cancel_workflow(wf.workflow_id)
            cancelled += 1

        # Synthetic terminal event so the SSE consumer closes — the
        # cancelled workflow itself may not get a chance to emit
        # anything (DBOS cancellation just marks the row + interrupts).
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

    @Durable.workflow()
    async def _run_with_tracking(
        self,
        *,
        thread_id_str: str,
        prompt: str,
        history_dump: list[dict[str, Any]],
        deferred_tool_results_dump: dict[str, Any] | None = None,
    ) -> None:
        """Outer per-turn workflow that brackets ``DBOSAgent.run`` with
        our ``start`` / ``done`` / ``approval-request`` / ``error``
        ThreadEvents and the assistant-turn persistence.

        Args are JSON-friendly primitives so DBOS workflow
        serialization survives pydantic / pickle / library version
        changes:

          - ``thread_id_str``                — stringified UUID.
          - ``prompt``                       — user prompt text
            (empty for approval-resume).
          - ``history_dump``                 — ``ModelMessagesTypeAdapter``
            JSON dump of the full conversation history.
          - ``deferred_tool_results_dump``   — optional; for
            approval-resume, dict of ``{tool_call_id: bool | {...}}``.
        """
        from pydantic_ai import (  # noqa: PLC0415
            DeferredToolRequests,
            DeferredToolResults,
            ToolDenied,
        )
        from pydantic_ai.messages import (  # noqa: PLC0415
            ModelMessagesTypeAdapter,
        )

        thread_id = UUID(thread_id_str)
        thread = await self._thread_repo.load(thread_id)
        if thread is None:
            await self._persist_and_publish(
                thread_id=thread_id,
                kind="error",
                payload={"message": f"Thread {thread_id} not found"},
            )
            return

        # Rehydrate types from JSON-safe wire shapes.
        history = (
            ModelMessagesTypeAdapter.validate_python(history_dump)
            if history_dump else []
        )
        deferred_results: DeferredToolResults | None = None
        if deferred_tool_results_dump:
            approvals_raw = deferred_tool_results_dump.get("approvals") or {}
            approvals: dict[str, bool | ToolDenied] = {}
            for tcid, decision in approvals_raw.items():
                if isinstance(decision, bool):
                    approvals[tcid] = decision
                elif isinstance(decision, dict) and "message" in decision:
                    approvals[tcid] = ToolDenied(message=str(decision["message"]))
                else:
                    approvals[tcid] = False
            deferred_results = DeferredToolResults(approvals=approvals)

        deps = await self.build_deps(thread=thread, message=None)
        model_settings = self.model_settings()

        await self._persist_and_publish(
            thread_id=thread_id,
            kind="start",
            payload={
                "prompt": prompt,
                "conversation_id": str(thread_id),
            },
        )

        token = _current_thread_id.set(thread_id)
        try:
            result = await self.dbos_agent.run(
                prompt if prompt else None,
                message_history=history,
                deferred_tool_results=deferred_results,
                deps=deps,
                model_settings=model_settings,
                # ``conversation_id == thread.id`` so every run in the
                # same thread shares a logical conversation grouping for
                # pydantic-ai's history-derivation + provider replay
                # (e.g. Anthropic prompt caching keys off it).
                conversation_id=str(thread_id),
            )
        except Exception as exc:
            await self._persist_and_publish(
                thread_id=thread_id,
                kind="error",
                payload={"message": str(exc), "type": type(exc).__name__},
            )
            raise
        finally:
            _current_thread_id.reset(token)

        # Persist the assistant turn so subsequent runs see it in the
        # repo-driven history. Skipped when output is DeferredToolRequests
        # (the resumption run will produce the real assistant turn).
        await self._persist_assistant_turn(
            thread_id=thread_id, result=result,
        )

        # HITL: emit one ``approval-request`` ThreadEvent per deferred
        # call so the wire encoder can ship Vercel v6
        # ``tool-approval-request`` chunks and the frontend renders an
        # approval card.
        if isinstance(result.output, DeferredToolRequests):
            for tc in result.output.approvals:
                await self._persist_and_publish(
                    thread_id=thread_id,
                    kind="approval-request",
                    payload={
                        "tool_call_id": tc.tool_call_id,
                        "tool_name": tc.tool_name,
                    },
                )

        await self._persist_and_publish(
            thread_id=thread_id, kind="done", payload={},
        )

    @Durable.step()
    async def _persist_assistant_turn(
        self,
        *,
        thread_id: UUID,
        result: Any,
    ) -> None:
        """Dump the assistant's Vercel-AI UI parts and persist as one row.

        ``@DBOS.step`` keeps the write idempotent across workflow
        replays. Skipped when the output is ``DeferredToolRequests``
        (paused mid-run waiting for approval — the resumption run
        emits the real assistant).
        """
        from pydantic_ai import DeferredToolRequests  # noqa: PLC0415
        from pydantic_ai.ui.vercel_ai import VercelAIAdapter  # noqa: PLC0415

        output = result.output
        if isinstance(output, DeferredToolRequests):
            return

        all_msgs = result.all_messages()
        ui_msgs = VercelAIAdapter.dump_messages(all_msgs, sdk_version=6)

        last_user_idx = -1
        for i, m in enumerate(ui_msgs):
            if m.role == "user":
                last_user_idx = i

        asst_parts: list[dict[str, Any]] = []
        for m in ui_msgs[last_user_idx + 1:]:
            if m.role != "assistant":
                continue
            for p in m.parts:
                asst_parts.append(
                    p.model_dump(
                        mode="json", by_alias=True, exclude_none=True,
                    ),
                )

        if not asst_parts:
            return

        await self._thread_repo.add_message(
            thread_id,
            role="assistant",
            parts=asst_parts,
        )

    async def find_active_workflow_id(self, thread_id: UUID) -> str | None:
        """Return the workflow_id of the most-recent active run for
        ``thread_id``, or None. Used by ``POST /threads/{id}/approve``
        to locate the workflow waiting on a deferred tool — even though
        with stateless-resume the workflow is already terminated, the
        endpoint still needs the prefix lookup to confirm a recent run
        existed.
        """
        del thread_id
        # Not used in stateless-resume mode but kept for future
        # DBOS.recv-based HITL.
        return None


def _safe_jsonify(value: Any) -> Any:
    """Best-effort JSON-friendly representation of a tool return value."""
    try:
        from pydantic import TypeAdapter  # noqa: PLC0415

        return TypeAdapter(type(value)).dump_python(value, mode="json")
    except Exception:
        return str(value)


__all__ = [
    "AGENT_RUN_QUEUE",
    "StateflowDurableAgent",
    "agent_run_workflow_id",
]
