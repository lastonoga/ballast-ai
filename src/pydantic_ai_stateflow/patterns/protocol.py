from __future__ import annotations

from typing import ClassVar, Protocol, TypeVar, runtime_checkable
from uuid import UUID

InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT", covariant=True)


@runtime_checkable
class Pattern(Protocol[InT, OutT]):
    """Structural type — Patterns are plain classes implementing this contract.

    NOT a base class (post code-review). Removes incentive to add hidden
    base behavior. Concrete patterns (`Reflection`, `MapReduce`,
    `MutationPipeline`, etc.) are regular classes that satisfy the protocol.

    Tenant_id is always a kwarg of `run` (canonical carrier per 4A.0.6).
    """

    name: ClassVar[str]

    async def run(self, input: InT, *, tenant_id: UUID) -> OutT: ...
