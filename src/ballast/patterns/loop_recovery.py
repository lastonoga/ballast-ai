from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from ballast.capabilities.helpers import Critique, SemanticLoopDetected

OutT = TypeVar("OutT")


@runtime_checkable
class LoopRecoveryPolicy(Protocol, Generic[OutT]):
    """Strategy for handling a SemanticLoopDetected inside Reflection.

    Called AFTER the critic has rejected a draft AND TypedLoopGuard has
    detected repetition. Implementations may abort, accept-last, or
    escalate-to-HITL (the last belongs in app code — it depends on an
    Asker which Patterns themselves don't own).
    """

    async def handle(
        self, ctx: Any, draft: OutT, feedback: list[Critique]
    ) -> OutT: ...


class AbortOnLoop:
    """Default LoopRecoveryPolicy — surface the SemanticLoopDetected to caller."""

    async def handle(
        self, ctx: Any, draft: Any, feedback: list[Critique]
    ) -> Any:
        raise SemanticLoopDetected(
            snapshot=str(draft)[:200],
            similarities=[],
        )
