from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, cast
from uuid import UUID

from dbos import DBOS
from pydantic import TypeAdapter

from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)


class UIChannel:
    """HITL channel backed by a FastAPI inbound endpoint.

    The endpoint (built via `build_hitl_router`) does endpoint-side
    authz + `DBOS.send` to the gate's tenant-scoped topic. This
    channel simply blocks on `DBOS.recv` and returns the response.

    Defense-in-depth re-check happens in `HITLGate.run` (SP5) — UIChannel
    intentionally does NOT re-check policy.
    """

    name: ClassVar[str] = "ui"

    async def ask(self, prompt: HITLPrompt, *, request_id: UUID) -> HITLResponse:
        topic = _hitl_topic(prompt.tenant_id, request_id)
        timeout_seconds = (
            prompt.timeout.total_seconds() if prompt.timeout is not None else None
        )
        payload = await DBOS.recv(topic, timeout_seconds=cast(Any, timeout_seconds))
        if payload is None:
            return TimeoutResponse(answered_at=datetime.now(tz=UTC))
        return _RESPONSE_ADAPTER.validate_python(payload)
