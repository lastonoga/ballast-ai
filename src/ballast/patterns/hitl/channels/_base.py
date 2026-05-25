"""``DBOSHITLChannel`` ABC — shared suspend boilerplate for any channel
that delivers verdicts via DBOS topics (the common case)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Generic
from uuid import uuid4

from ballast.durable import Durable
from ballast.patterns.hitl.channels._protocol import InT, VerdictT


class DBOSHITLChannel(Generic[InT, VerdictT], ABC):
    """Channels that use DBOS topics for verdict delivery.

    Subclasses fill in ``deliver`` (surface the request to the human)
    and ``decode_verdict`` (re-hydrate the dict that arrived on the
    DBOS topic into a typed VerdictT). ``request`` orchestrates them
    with ``Durable.recv_async`` so the calling workflow is recoverable
    across crashes.
    """

    async def request(
        self,
        payload: InT,
        *,
        timeout: timedelta | None = None,
    ) -> VerdictT:
        request_id  = str(uuid4())
        workflow_id = Durable.current_workflow_id()
        topic       = f"hitl:{request_id}"
        await self.deliver(
            request_id=request_id, workflow_id=workflow_id,
            respond_topic=topic, payload=payload,
        )
        timeout_seconds = (
            timeout.total_seconds() if timeout is not None else None
        )
        raw = await Durable.recv_async(
            topic=topic, timeout_seconds=timeout_seconds,
        )
        return await self.decode_verdict(raw)

    @abstractmethod
    async def deliver(
        self, *,
        request_id: str, workflow_id: str, respond_topic: str,
        payload: InT,
    ) -> None: ...

    @abstractmethod
    async def decode_verdict(self, raw: Any) -> VerdictT: ...


__all__ = ["DBOSHITLChannel"]
