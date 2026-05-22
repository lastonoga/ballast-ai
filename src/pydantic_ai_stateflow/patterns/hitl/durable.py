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
     ``Durable.send_async(destination_id=workflow_id, ...)`` — the
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

    from pydantic_ai_stateflow.persistence.thread.domain import Thread
    from pydantic_ai_stateflow.runtime.agents import StateflowAgent

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
        config_name: str | None = None,
    ) -> None:
        # Default to ``cls.__qualname__`` for the same reason as
        # ``StateflowDurableAgent`` — singleton-per-class is the
        # common case; multiple instances of the same class (per-test
        # isolation) override explicitly.
        super().__init__(
            config_name=config_name or type(self).__qualname__,
        )

    @abstractmethod
    async def on_decision(
        self,
        *,
        response: HITLResponse,
        context: BaseModel,
    ) -> None:
        """App-specific post-decision logic.

        Runs with a fully-rehydrated typed ``context`` (the
        ``helper_agent.metadata_model`` instance the caller passed to
        ``open``) and the validated ``response``. Implementations
        should persist whatever they need, notify whoever needs it,
        and return — any exception aborts the workflow (and triggers
        DBOS's retry/dead-letter behaviour according to the configured
        policy).

        **Durability semantics**: the framework invokes ``on_decision``
        from inside a ``@Durable.step``, so its return value (``None``)
        is memoised on first invocation. On workflow replay (recovery
        after a crash) the step is skipped and the body does NOT
        re-execute — side effects ARE durable / exactly-once provided
        the body's effects are themselves atomic at the I/O layer
        (database INSERTs, etc). Implementations therefore do NOT need
        manual idempotency guards.
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
        """Spawn the helper thread + start the durable decision workflow.

        Returns the new helper ``Thread``. The caller MAY embed its id
        in their own response (e.g. so a UI can deep-link to the side
        thread); they MUST NOT await the decision — that happens
        inside the durable workflow this method spawns.

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

        ----------------------------------------------------------------
        **Atomicity / crash recovery**:
        ``open`` is a thin validating shim that immediately delegates
        to ``_open_workflow`` — a ``@Durable.workflow`` whose body
        does **all** the side effects (thread.create, opening message,
        start of the decision workflow, thread-created event). DBOS
        records ``_open_workflow`` in its workflow log, so a crash
        between ``thread_repo.create`` and ``Durable.start_workflow``
        is recovered: on restart DBOS replays the same workflow and
        the step-level memoisation skips already-completed effects.

        We allocate uuids HERE (in the caller fiber) and pass them
        IN as workflow inputs — generating uuids inside the durable
        body would be non-deterministic across replays.
        ----------------------------------------------------------------
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

        # Pre-allocate routing ids in the CALLER fiber so the workflow
        # body uses stable values across replay (uuid4 inside the
        # workflow body would diverge on every replay).
        request_id = uuid4()
        decision_workflow_id = str(uuid4())

        # FQN of the ``metadata_model`` class so the durable workflow
        # can rehydrate ``context`` to a typed object on the recovery
        # side WITHOUT depending on ``StateflowAgent`` registry state.
        context_class_fqn = (
            f"{metadata_model.__module__}.{metadata_model.__qualname__}"
        )

        timeout_seconds = (
            timeout.total_seconds() if timeout is not None
            else _NO_TIMEOUT_SECONDS
        )

        return await self._open_workflow(
            helper_agent_name=helper_agent.name,
            context_dict=context.model_dump(mode="json"),
            context_class_fqn=context_class_fqn,
            opening_message=opening_message,
            request_id=str(request_id),
            decision_workflow_id=decision_workflow_id,
            timeout_seconds=timeout_seconds,
            notify_parent_thread_id=(
                str(notify_parent_thread_id)
                if notify_parent_thread_id is not None else None
            ),
        )

    @Durable.workflow()
    async def _open_workflow(
        self,
        *,
        helper_agent_name: str,
        context_dict: dict[str, Any],
        context_class_fqn: str,
        opening_message: str | None,
        request_id: str,
        decision_workflow_id: str,
        timeout_seconds: float,
        notify_parent_thread_id: str | None,
    ) -> Thread:
        """Durable body of :meth:`open` — every side effect is a step.

        DBOS records this workflow + all its constituent steps. A
        crash anywhere in the body recovers correctly on restart: the
        recorded steps are skipped (memoised) and only the unfinished
        tail re-runs. The decision workflow started below has a
        deterministic id (passed in by the caller) so re-issuing the
        ``Durable.start_workflow`` call is idempotent — DBOS dedupes
        by ``workflow_id``."""
        # Helper thread metadata = user-facing context fields + framework
        # routing keys. The helper agent's tools read all three from raw
        # ``thread.metadata_``; ``metadata_model`` validation ignores
        # extras (pydantic ``extra="ignore"`` default).
        thread_metadata: dict[str, Any] = dict(context_dict)
        thread_metadata["request_id"] = request_id
        thread_metadata["workflow_id"] = decision_workflow_id

        thread = await self._create_helper_thread(
            agent_name=helper_agent_name,
            metadata=thread_metadata,
        )

        if opening_message:
            await self._seed_opening_message(thread.id, opening_message)

        # **Clear inherited queue partition** before spawning the
        # decision workflow. ``Durable.start_workflow`` inherits
        # ``queue_partition_key`` from the local DBOS context. If
        # we're already running inside a partitioned queue (e.g.
        # ``AGENT_RUN_QUEUE`` with concurrency=1 from a tool call),
        # the decision workflow inherits it AND stays ENQUEUED
        # forever without a matching worker. Force partition_key=None
        # so the child runs on the default executor.
        with SetWorkflowID(decision_workflow_id), SetEnqueueOptions(
            queue_partition_key=None,
        ):
            await Durable.start_workflow(
                self.run,
                context_dict=context_dict,
                request_id=request_id,
                context_class_fqn=context_class_fqn,
                timeout_seconds=timeout_seconds,
            )

        if notify_parent_thread_id is not None:
            await self._emit_thread_created(
                parent_thread_id_str=notify_parent_thread_id,
                helper_thread_id=thread.id,
                helper_agent_name=helper_agent_name,
                thread_metadata=thread_metadata,
            )

        return thread

    # ── per-step writes (memoised) ───────────────────────────────────

    @Durable.step()
    async def _create_helper_thread(
        self, *, agent_name: str, metadata: dict[str, Any],
    ) -> Thread:
        from pydantic_ai_stateflow.runtime.engine import get_engine  # noqa: PLC0415
        return await get_engine().thread_repo.create(
            agent=agent_name, metadata=metadata,
        )

    @Durable.step()
    async def _seed_opening_message(
        self, thread_id: UUID, opening_message: str,
    ) -> None:
        from pydantic_ai_stateflow.runtime.engine import get_engine  # noqa: PLC0415
        await get_engine().thread_repo.add_message(
            thread_id,
            role="assistant",
            parts=[{
                "type": "text",
                "text": opening_message,
                "state": "done",
            }],
        )

    @Durable.step()
    async def _emit_thread_created(
        self,
        *,
        parent_thread_id_str: str,
        helper_thread_id: UUID,
        helper_agent_name: str,
        thread_metadata: dict[str, Any],
    ) -> None:
        """Emit ``thread-created`` into the parent thread's event log."""
        from pydantic_ai_stateflow.runtime.engine import get_engine  # noqa: PLC0415
        engine = get_engine()
        parent_id = UUID(parent_thread_id_str)
        _log.info(
            "DurableHITLWorkflow.open notify_parent=%s helper_thread=%s",
            parent_id, helper_thread_id,
        )
        ev = await engine.event_log.append(
            thread_id=parent_id,
            kind="thread-created",
            payload={
                "thread_id": str(helper_thread_id),
                "agent": helper_agent_name,
                "metadata": thread_metadata,
            },
        )
        await engine.event_stream.publish(
            thread_channel(parent_id),
            EventNotification(thread_id=parent_id, seq=ev.seq),
        )

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
        payload = await Durable.recv_async(topic, timeout_seconds=timeout_seconds)

        if payload is None:
            from datetime import UTC, datetime  # noqa: PLC0415

            response: HITLResponse = TimeoutResponse(
                answered_at=datetime.now(tz=UTC),
            )
        else:
            response = _RESPONSE_ADAPTER.validate_python(payload)

        context_cls = _resolve_class(context_class_fqn)
        context = context_cls.model_validate(context_dict)

        # Route through ``_dispatch_on_decision`` (a step) so the
        # body's side effects are memoised across workflow replay.
        # Without the step wrapper, an in-flight workflow that gets
        # recovered after a crash would re-execute ``on_decision``
        # from scratch — and the typical implementations (notify
        # parent thread, persist domain entity) would double-fire.
        await self._dispatch_on_decision(response=response, context=context)

    @Durable.step()
    async def _dispatch_on_decision(
        self,
        *,
        response: HITLResponse,
        context: BaseModel,
    ) -> None:
        """Step wrapper around ``on_decision`` for replay idempotency.

        DBOS memoises step return values (``None`` here) by step name
        + args. On workflow replay the step is skipped without
        re-invoking the body, which means ``on_decision`` runs
        exactly once across the workflow's lifetime — including
        across crashes and restarts."""
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
