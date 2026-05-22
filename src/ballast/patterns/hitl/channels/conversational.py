from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID

from dbos import DBOS, SetWorkflowID
from pydantic import TypeAdapter

from ballast.durable import Durable
from ballast.observability.spans import traced
from ballast.observability.trace_names import TraceName
from ballast.patterns.hitl.helper.session import (
    DefaultHelperSessionRunner,
    HelperSessionInput,
)
from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence.thread.repository import ThreadRepository
from ballast.runtime.det import Det
from ballast.runtime.idempotency import IdempotencyInput

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)


async def start_workflow_async(
    workflow_fn: Callable[..., Any],
    input: Any,
    *,
    idempotency_key: str,
) -> None:
    """Thin wrapper around ``Durable.start_workflow`` + ``SetWorkflowID``.

    Uses ``Durable.start_workflow`` so the OTel trace context the caller
    is in (e.g. ``pattern.hitl_gate``) becomes the parent of every span
    emitted by the spawned helper-session workflow — visible as one
    tree in Logfire instead of a detached root.
    """
    with SetWorkflowID(idempotency_key):
        await Durable.start_workflow(workflow_fn, input)


class ConversationalChannel:
    """HITL channel backed by a helper pydantic-ai Agent in its own workflow."""

    name: ClassVar[str] = "conversational"

    def __init__(
        self,
        *,
        helper_session_runner: DefaultHelperSessionRunner,
        thread_repo: ThreadRepository,
        base_agent_module: str,
        base_agent_attr: str | None,
        context_type: type[Any] | None,
        gate_workflow_id_resolver: Callable[[], UUID],
        actor_id: str = "founder",
    ) -> None:
        self.helper_session_runner = helper_session_runner
        self.thread_repo = thread_repo
        self.base_agent_module = base_agent_module
        self.base_agent_attr = base_agent_attr
        self.context_type = context_type
        self.gate_workflow_id_resolver = gate_workflow_id_resolver
        self.actor_id = actor_id

    @traced(TraceName.CHANNEL_CONVERSATIONAL, attrs=lambda self, prompt, *, request_id: {
        "request_id": str(request_id),
    })
    async def ask(
        self, prompt: HITLPrompt, *, request_id: UUID,
    ) -> HITLResponse:
        idempotency_key = await self._idempotency_key(request_id)
        gate_wf = self.gate_workflow_id_resolver()
        input = HelperSessionInput(
            prompt_payload=prompt.model_dump(mode="json"),
            request_id=request_id,
            gate_workflow_id=gate_wf,
            base_agent_module=self.base_agent_module,
            base_agent_attr=self.base_agent_attr,
            context_type_fqn=(
                f"{self.context_type.__module__}."
                f"{self.context_type.__qualname__}"
                if self.context_type is not None else None
            ),
            actor_id=self.actor_id,
        )
        await start_workflow_async(
            self.helper_session_runner.run, input,
            idempotency_key=idempotency_key,
        )

        topic = _hitl_topic(request_id)
        if prompt.timeout is not None:
            payload = await Durable.recv(
                topic, timeout_seconds=prompt.timeout.total_seconds(),
            )
        else:
            payload = await Durable.recv(topic)
        if payload is None:
            return TimeoutResponse(answered_at=datetime.now(tz=UTC))
        return _RESPONSE_ADAPTER.validate_python(payload)

    @staticmethod
    async def _idempotency_key(request_id: UUID) -> str:
        derived = await Det.uuid_for(
            IdempotencyInput(
                namespace="helper_session",
                parts={"request_id": request_id},
            ),
        )
        return f"helper:{derived}"
