"""Durable, fire-and-forget HITL helper-thread pattern.

The blocking ``HITLGate.ask_helper`` couples the caller's await with the
helper's decision — fine for short decisions where the request handler
stays alive, fatal for long ones (user closes tab → request cancelled →
``await`` raises ``CancelledError`` → post-decision logic never runs).

``DurableHITLWorkflow`` decouples the two via DBOS:

  1. Caller invokes ``open(helper_agent, context, ...)`` from any
     async context — pydantic-ai tool, FastAPI handler, background
     job. ``open`` spawns the helper thread (with ``context.model_dump()``
     as metadata, plus framework routing keys), kicks off the durable
     workflow via ``DBOS.start_workflow_async``, and returns the
     helper thread IMMEDIATELY. The caller is now off the hook.

  2. The durable workflow blocks on ``DBOS.recv_async`` for the
     helper's decision. Helper agent's tools route their
     ``HITLResponse`` to the workflow via
     ``DBOS.send_async(destination_id=workflow_id, ...)`` — the
     framework writes both ``request_id`` and ``workflow_id`` onto the
     helper thread's metadata so the helper's tool body can read them.

  3. Once the decision arrives, the workflow rehydrates ``context``
     against ``helper_agent.metadata_model`` (so ``on_decision``
     receives a typed object, not a raw dict) and calls the subclass's
     ``on_decision(response=..., context=...)``. App owns save / notify /
     audit logic there.

Because the whole post-decision path lives inside the durable workflow,
it survives the caller's death — the user can close the browser, the
parent SSE stream can time out, the process can restart (DBOS recovers
the workflow from persisted state) — and the post-decision work still
runs to completion.

Apps subclass ``DurableHITLWorkflow``, override ``on_decision``, and
register one instance per workflow kind at startup. They invoke
``open(...)`` from wherever they need to spawn an approval flow.
"""

from __future__ import annotations

import importlib
import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from dbos import DBOS, DBOSConfiguredInstance, SetEnqueueOptions, SetWorkflowID
from pydantic import BaseModel, TypeAdapter

from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.logging import get_logger
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.runtime.event_stream import (
    EventNotification,
    thread_channel,
)

if TYPE_CHECKING:
    from datetime import timedelta

    from pydantic_ai_stateflow.persistence.events.repository import (
        EventLogRepository,
    )
    from pydantic_ai_stateflow.persistence.thread.domain import Thread
    from pydantic_ai_stateflow.persistence.thread.repository import (
        ThreadRepository,
    )
    from pydantic_ai_stateflow.runtime.agents import StateflowAgent
    from pydantic_ai_stateflow.runtime.event_stream import EventStream

_log = get_logger(__name__)
_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)
_instance_counter = itertools.count()

# Effectively "wait forever" without violating ``DBOS.recv_async``'s
# arithmetic (it adds ``timeout_seconds`` to ``time.time()`` for sleep
# accounting and rejects ``None``). ≈ 1 year.
_NO_TIMEOUT_SECONDS: float = 365 * 24 * 60 * 60.0


@Durable.dbos_class()
class DurableHITLWorkflow(DBOSConfiguredInstance):
    """Abstract base for fire-and-forget durable HITL flows.

    Subclasses MUST override ``on_decision``. The base provides:
      - ``open(helper_agent, context)`` — spawn helper thread + start
        workflow + return helper thread.
      - ``run(...)`` — ``@DBOS.workflow`` body that blocks on the
        helper's response and dispatches to ``on_decision``.

    The ``DBOSConfiguredInstance`` machinery lets DBOS rehydrate the
    instance after a crash by ``config_name`` lookup, so pass a stable
    name into ``super().__init__(config_name=...)`` if you want
    recovery to bind the same Python object to the same workflow on
    restart.
    """

    def __init__(
        self,
        *,
        thread_repo: ThreadRepository,
        event_log: EventLogRepository | None = None,
        event_stream: EventStream | None = None,
        config_name: str | None = None,
    ) -> None:
        super().__init__(
            config_name=config_name
            or f"durable-hitl-{next(_instance_counter)}",
        )
        self.thread_repo = thread_repo
        # Optional. When wired, ``open`` emits a ``thread-created`` event
        # into the ``notify_parent_thread_id`` event log so any open
        # ``GET /threads/{id}/events`` SSE consumer can refresh the
        # thread list without polling. ``_notify`` (callable by
        # subclasses) similarly emits ``message-added``.
        self._event_log = event_log
        self._event_stream = event_stream

    @abstractmethod
    async def on_decision(
        self,
        *,
        response: HITLResponse,
        context: BaseModel,
    ) -> None:
        """App-specific post-decision logic.

        Runs inside the durable workflow with a fully-rehydrated typed
        ``context`` (the ``helper_agent.metadata_model`` instance the
        caller passed to ``open``) and the validated ``response``.
        Implementations should persist whatever they need, notify
        whoever needs it, and return — any exception aborts the
        workflow (and triggers DBOS's retry/dead-letter behaviour
        according to the configured policy).
        """
        raise NotImplementedError

    @traced(TraceName.PATTERN_HITL_GATE, attrs=lambda self, *, helper_agent, **__: {
        "pattern": "durable_hitl",
        "helper_agent": helper_agent.name,
    })
    async def open(
        self,
        *,
        helper_agent: type[StateflowAgent],
        context: BaseModel,
        opening_message: str | None = None,
        timeout: timedelta | None = None,
        notify_parent_thread_id: UUID | None = None,
    ) -> Thread:
        """Spawn the helper thread + start the durable workflow.

        Returns the new helper ``Thread``. The caller MAY embed its id
        in their own response (e.g. so a UI can deep-link to the side
        thread); they MUST NOT await the decision — that happens inside
        the workflow which this method has already detached from the
        caller's lifetime.

        ``context`` must be an instance of
        ``helper_agent.metadata_model``. ``opening_message`` (optional)
        seeds an assistant message on the new thread so the user sees
        something the moment they open it.

        ``timeout`` (optional ``timedelta``) bounds how long the
        workflow waits for the helper's response. On timeout
        ``on_decision`` is called with a ``TimeoutResponse``.

        ``notify_parent_thread_id`` (optional): when supplied AND the
        instance has ``event_log`` / ``event_stream`` wired, emits a
        ``thread-created`` event into that parent thread's event log
        so a frontend listening on ``GET /threads/{parent}/events``
        can refresh its thread list immediately (no F5).
        """
        metadata_model = helper_agent.metadata_model
        if metadata_model is None:
            raise ValueError(
                f"{helper_agent.__name__}.metadata_model is None — cannot "
                "use it as a HITL helper agent. Set a metadata_model that "
                "validates the context shape.",
            )
        if not isinstance(context, metadata_model):
            raise TypeError(
                f"context must be an instance of "
                f"{helper_agent.__name__}.metadata_model "
                f"({metadata_model.__name__}), got {type(context).__name__}",
            )

        request_id = uuid4()
        workflow_id = str(uuid4())

        # Helper thread metadata is the union of (user-facing context
        # fields) + (framework routing keys). The helper agent's tools
        # read all three from raw ``thread.metadata_``; its
        # ``metadata_model`` validation ignores the extras (pydantic
        # default ``extra="ignore"``).
        thread_metadata: dict[str, Any] = context.model_dump(mode="json")
        thread_metadata["request_id"] = str(request_id)
        thread_metadata["workflow_id"] = workflow_id

        thread = await self.thread_repo.create(
            agent=helper_agent.name,
            metadata=thread_metadata,
        )
        if opening_message:
            await self.thread_repo.add_message(
                thread.id,
                role="assistant",
                parts=[{
                    "type": "text",
                    "text": opening_message,
                    "state": "done",
                }],
            )

        # Pre-allocate the workflow id via ``SetWorkflowID`` so it
        # matches what we wrote into thread metadata BEFORE starting —
        # the helper's tools read that field to address ``DBOS.send``.
        timeout_seconds = (
            timeout.total_seconds() if timeout is not None
            else _NO_TIMEOUT_SECONDS
        )
        # FQN of the ``metadata_model`` class so the workflow can
        # rehydrate ``context`` to a typed object on the other side
        # WITHOUT depending on ``StateflowAgent`` registry state (which
        # may not be populated in tests or worker processes that don't
        # register agents at boot).
        context_class_fqn = (
            f"{metadata_model.__module__}.{metadata_model.__qualname__}"
        )
        # **Clear inherited queue partition** before spawning the HITL
        # workflow. ``DBOS.start_workflow_async`` (``_core.py:991``)
        # inherits ``queue_partition_key`` from the parent workflow's
        # local context. If the caller (e.g. a tool inside an agent
        # workflow that's running in a partitioned queue like
        # ``AGENT_RUN_QUEUE`` with concurrency=1) has a partition_key
        # set, the HITL workflow inherits it AND ENQUEUED state, but
        # without a matching queue worker it sits forever (status
        # ``ENQUEUED``, never ``PENDING``). Force partition_key=None so
        # the child runs on the default executor regardless of caller.
        with SetWorkflowID(workflow_id), SetEnqueueOptions(
            queue_partition_key=None,
        ):
            await Durable.start_workflow(
                self.run,
                context_dict=context.model_dump(mode="json"),
                request_id=str(request_id),
                context_class_fqn=context_class_fqn,
                timeout_seconds=timeout_seconds,
            )

        # Emit ``thread-created`` on the parent thread's event log so
        # a frontend listening on its long-lived SSE can refresh the
        # thread list immediately. No-op when notify_parent_thread_id
        # or event_log isn't wired.
        _log.info(
            "DurableHITLWorkflow.open notify_parent=%s event_log_wired=%s "
            "event_stream_wired=%s helper_thread=%s",
            notify_parent_thread_id,
            self._event_log is not None,
            self._event_stream is not None,
            thread.id,
        )
        if notify_parent_thread_id is not None and self._event_log is not None:
            ev = await self._event_log.append(
                thread_id=notify_parent_thread_id,
                kind="thread-created",
                payload={
                    "thread_id": str(thread.id),
                    "agent": helper_agent.name,
                    "metadata": thread_metadata,
                },
            )
            if self._event_stream is not None:
                await self._event_stream.publish(
                    thread_channel(notify_parent_thread_id),
                    EventNotification(
                        thread_id=notify_parent_thread_id, seq=ev.seq,
                    ),
                )

        return thread

    @Durable.workflow()
    async def run(
        self,
        *,
        context_dict: dict[str, Any],
        request_id: str,
        context_class_fqn: str,
        timeout_seconds: float = _NO_TIMEOUT_SECONDS,
    ) -> None:
        """Block on the helper's response, then dispatch to ``on_decision``.

        Args are kept as JSON-friendly primitives (dict / str / float)
        so DBOS workflow serialization is robust across pydantic /
        pickle version changes.

        ``context_class_fqn`` is the fully-qualified name of the
        ``metadata_model`` class — resolved via ``importlib`` so the
        workflow can rehydrate ``context_dict`` into a typed
        ``BaseModel`` instance without depending on any process-global
        registry state.
        """
        topic = _hitl_topic(UUID(request_id))
        payload = await DBOS.recv_async(topic, timeout_seconds=timeout_seconds)

        if payload is None:
            from datetime import UTC, datetime  # noqa: PLC0415

            response: HITLResponse = TimeoutResponse(
                answered_at=datetime.now(tz=UTC),
            )
        else:
            response = _RESPONSE_ADAPTER.validate_python(payload)

        context_cls = _resolve_class(context_class_fqn)
        context = context_cls.model_validate(context_dict)

        await self.on_decision(response=response, context=context)


def _resolve_class(fqn: str) -> Any:
    """Resolve ``module.path.ClassName`` → class object.

    Used by the durable workflow to rehydrate the ``metadata_model``
    class from a string FQN — avoids needing the
    ``StateflowAgent`` registry to be populated in the recovery
    process.
    """
    module_path, _, name = fqn.rpartition(".")
    if not module_path:
        raise ValueError(f"FQN must be fully qualified, got {fqn!r}")
    mod = importlib.import_module(module_path)
    return getattr(mod, name)
