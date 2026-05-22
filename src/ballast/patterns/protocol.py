from __future__ import annotations

from typing import ClassVar, Protocol, TypeVar, runtime_checkable

InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT", covariant=True)


@runtime_checkable
class Pattern(Protocol[InT, OutT]):
    """Structural type — Patterns are plain classes implementing this contract.

    Apps that need per-run scoping (tenant, workspace) pass it inside
    the ``input`` value or via captured constructor state — the
    framework's Pattern contract is identity-agnostic.
    """

    name: ClassVar[str]

    async def run(self, input: InT) -> OutT: ...
