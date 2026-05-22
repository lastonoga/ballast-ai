from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.response import (
    ApprovedResponse,
    HITLResponse,
    ModifiedResponse,
    RejectedResponse,
    TimeoutResponse,
)


@runtime_checkable
class HITLChannel(Protocol):
    """Strategy port — request a response from a human (or fake in tests).

    The full receive-and-validate loop (spec 2C.4) belongs in HITLGate,
    not the channel — channels just produce a response (or raise).
    """

    async def ask(self, prompt: HITLPrompt, *, request_id: UUID) -> HITLResponse: ...


class InMemoryHITLChannel:
    """Test/dev channel: caller preloads canned responses keyed by request_id.

    Used by HITLGate tests and end-to-end smoke tests. Real channels
    (Slack, FastAPI, webhook) ship in SP6.
    """

    def __init__(self) -> None:
        self._responses: dict[
            UUID,
            ApprovedResponse | RejectedResponse | ModifiedResponse | TimeoutResponse,
        ] = {}

    def set_response(
        self,
        request_id: UUID,
        response: ApprovedResponse | RejectedResponse | ModifiedResponse | TimeoutResponse,
    ) -> None:
        self._responses[request_id] = response

    async def ask(self, prompt: HITLPrompt, *, request_id: UUID) -> HITLResponse:
        if request_id not in self._responses:
            raise KeyError(
                f"InMemoryHITLChannel: no preloaded response for {request_id}"
            )
        return self._responses[request_id]
