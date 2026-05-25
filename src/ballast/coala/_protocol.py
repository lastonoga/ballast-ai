"""``CoALAUnit`` Protocol — single contract for memory-aware computation."""
from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

InT      = TypeVar("InT")
ObsT     = TypeVar("ObsT")
ContextT = TypeVar("ContextT")
OutT     = TypeVar("OutT")


@runtime_checkable
class CoALAUnit(Protocol[InT, ObsT, ContextT, OutT]):
    """Unit of memory-aware computation following CoALA's 4-phase
    decision procedure.

    Same contract regardless of runtime — a workflow, an agent tool, an
    agent capability — any can be wrapped via the corresponding adapter
    (``as_workflow``, ``as_tool``, ``as_capability``).

    Phase semantics (from Sumers et al., "Cognitive Architectures for
    Language Agents"):
      observe  — parse raw input into structured working-memory state
      retrieve — pull relevant long-term memory based on observation
      act      — reason + ground + execute; produces output
      learn    — persist insights back into long-term memory
    """

    async def observe(self, input: InT) -> ObsT: ...
    async def retrieve(self, observation: ObsT) -> ContextT: ...
    async def act(self, observation: ObsT, context: ContextT) -> OutT: ...
    async def learn(self, observation: ObsT, context: ContextT, output: OutT) -> None: ...


__all__ = ["CoALAUnit"]
