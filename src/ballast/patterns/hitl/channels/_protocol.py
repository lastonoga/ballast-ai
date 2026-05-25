"""``HITLChannel`` Protocol — one method, full lifecycle."""
from __future__ import annotations

from datetime import timedelta
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

InT      = TypeVar("InT",      bound=BaseModel)
VerdictT = TypeVar("VerdictT", bound=BaseModel)


class HITLChannel(Protocol, Generic[InT, VerdictT]):
    """Owns the full request lifecycle for one human decision.

    A channel knows what payload type it accepts, how to surface the
    request (UI card, chat thread, Slack, …), how to wait for the
    verdict, and how to decode the response into a typed model. The
    framework knows nothing about the medium.
    """

    async def request(
        self,
        payload: InT,
        *,
        timeout: timedelta | None = None,
    ) -> VerdictT: ...


__all__ = ["HITLChannel", "InT", "VerdictT"]
