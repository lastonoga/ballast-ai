from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID

from dbos import DBOS, SetWorkflowID
from pydantic import TypeAdapter

from pydantic_ai_stateflow.patterns.hitl.helper.session import (
    DefaultHelperSessionRunner,
    HelperSessionInput,
)
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository
from pydantic_ai_stateflow.runtime.det import Det
from pydantic_ai_stateflow.runtime.idempotency import IdempotencyInput

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)


async def start_workflow_async(
    workflow_fn: Callable[..., Any],
    input: Any,
    *,
    idempotency_key: str,
) -> None:
    """Thin wrapper around `DBOS.start_workflow_async` + `SetWorkflowID`.

    Extracted so tests can patch a single symbol. Production: this kicks off
    the helper-session workflow under a deterministic workflow_id so retries
    of the parent gate workflow re-attach to the same session instead of
    spawning new ones.
    """
    with SetWorkflowID(idempotency_key):
        await DBOS.start_workflow_async(workflow_fn, input)


class ConversationalChannel:
    """HITL channel backed by a helper pydantic-ai Agent in its own workflow.

    Lifecycle (spec 3J.1):
      1. `ask()` computes a deterministic idempotency key.
      2. Starts `helper_session_runner.run` as an INDEPENDENT workflow under
         that key — so gate-workflow replay reuses the same session instead
         of spawning duplicates.
      3. `DBOS.recv`s on the gate's tenant-scoped topic until the helper
         agent invokes an approval tool (which DBOS.sends to that topic).
      4. Returns the resulting `HITLResponse` (or `TimeoutResponse`).
    """

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

    async def ask(
        self, prompt: HITLPrompt, *, request_id: UUID,
    ) -> HITLResponse:
        idempotency_key = await self._idempotency_key(
            prompt.tenant_id, request_id,
        )
        gate_wf = self.gate_workflow_id_resolver()
        input = HelperSessionInput(
            prompt_payload=prompt.model_dump(mode="json"),
            request_id=request_id,
            tenant_id=prompt.tenant_id,
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

        topic = _hitl_topic(prompt.tenant_id, request_id)
        if prompt.timeout is not None:
            payload = await DBOS.recv(
                topic, timeout_seconds=prompt.timeout.total_seconds(),
            )
        else:
            payload = await DBOS.recv(topic)
        if payload is None:
            return TimeoutResponse(answered_at=datetime.now(tz=UTC))
        return _RESPONSE_ADAPTER.validate_python(payload)

    @staticmethod
    async def _idempotency_key(tenant_id: UUID, request_id: UUID) -> str:
        derived = await Det.uuid_for(
            IdempotencyInput(
                namespace="helper_session",
                parts={
                    "tenant_id": tenant_id,
                    "request_id": request_id,
                },
            ),
        )
        return f"helper:{derived}"
