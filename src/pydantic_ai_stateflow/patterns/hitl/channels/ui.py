from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, cast
from uuid import UUID

from dbos import DBOS
from pydantic import TypeAdapter

from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName
from pydantic_ai_stateflow.patterns.hitl.prompt import HITLPrompt
from pydantic_ai_stateflow.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)


class UIChannel:
    """HITL channel backed by a FastAPI inbound endpoint.

    The endpoint (built via ``build_hitl_router``) does endpoint-side
    authz + ``DBOS.send`` to the gate's per-request topic. This
    channel simply blocks on ``DBOS.recv`` and returns the response.
    """

    name: ClassVar[str] = "ui"

    # No-timeout HITL prompts still need a finite ``timeout_seconds`` for
    # ``DBOS.recv_async`` because dbos passes it straight into
    # ``time.time() + seconds`` for sleep accounting. We use a very large
    # ceiling (≈ 1 year) — effectively "wait forever" but won't trip the
    # arithmetic.
    _NO_TIMEOUT_SECONDS: ClassVar[float] = 365 * 24 * 60 * 60.0

    @traced(TraceName.CHANNEL_UI, attrs=lambda self, prompt, *, request_id: {
        "request_id": str(request_id),
    })
    async def ask(self, prompt: HITLPrompt, *, request_id: UUID) -> HITLResponse:
        topic = _hitl_topic(request_id)
        timeout_seconds = (
            prompt.timeout.total_seconds()
            if prompt.timeout is not None
            else self._NO_TIMEOUT_SECONDS
        )
        # ``recv_async`` is the only correct call here: ``HITLGate.run`` is
        # ``@DBOS.workflow()`` so the body executes on DBOS's background
        # event loop. The sync ``DBOS.recv`` aborts with "called while
        # an event loop is running" inside that loop (dbos 2.22+).
        recv_async = getattr(DBOS, "recv_async", None)
        if recv_async is not None:
            payload = await recv_async(
                topic, timeout_seconds=cast(Any, timeout_seconds),
            )
        else:  # pragma: no cover — older dbos releases
            payload = await DBOS.recv(
                topic, timeout_seconds=cast(Any, timeout_seconds),
            )
        if payload is None:
            return TimeoutResponse(answered_at=datetime.now(tz=UTC))
        return _RESPONSE_ADAPTER.validate_python(payload)
