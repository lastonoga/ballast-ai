"""``CoALABase`` ABC — ergonomic base with default observe + learn."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InT      = TypeVar("InT")
ObsT     = TypeVar("ObsT")
ContextT = TypeVar("ContextT")
OutT     = TypeVar("OutT")


class CoALABase(Generic[InT, ObsT, ContextT, OutT], ABC):
    """Minimal-friction base. Apps override only the phases they need.

    ``observe`` defaults to identity (input passes through). Override
    when you need to extract intent / entities / signals before retrieve.

    ``learn`` defaults to no-op. Override to write episodes, facts,
    learned skills, etc. — anything the app wants to persist.

    ``retrieve`` and ``act`` are abstract — every meaningful unit has a
    retrieval step (even if it returns an empty Context) and an act
    step (the actual work).
    """

    async def observe(self, input: InT) -> ObsT:
        return input  # type: ignore[return-value]

    @abstractmethod
    async def retrieve(self, observation: ObsT) -> ContextT: ...

    @abstractmethod
    async def act(self, observation: ObsT, context: ContextT) -> OutT: ...

    async def learn(
        self, observation: ObsT, context: ContextT, output: OutT,
    ) -> None:
        return None


__all__ = ["CoALABase"]
