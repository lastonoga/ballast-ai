from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from ballast.patterns.errors import MutationRejected
from ballast.patterns.mutation.primitives import RejectedAt


class RejectAction(StrEnum):
    """Spec 4A.0.10: a RejectPolicy MUST return a terminal action.

    No recursive HITL ping-pong; max_escalations is enforced inside the policy.
    """

    DROP = "drop"
    RETRY = "retry"
    ACCEPT = "accept"


@runtime_checkable
class RejectPolicy(Protocol):
    async def handle(
        self, rejected: RejectedAt, retries_so_far: int,
    ) -> RejectAction: ...


class DropOnReject:
    """Default — drop the proposal silently; pipeline returns the RejectedAt."""

    async def handle(
        self, rejected: RejectedAt, retries_so_far: int,
    ) -> RejectAction:
        return RejectAction.DROP


class RaiseOnReject:
    """Surface the rejection as MutationRejected. Useful when callers expect
    pipeline success and want failure to be a hard exception (cron jobs etc.)."""

    async def handle(
        self, rejected: RejectedAt, retries_so_far: int,
    ) -> RejectAction:
        raise MutationRejected(
            stage=rejected.stage,
            reason=rejected.reason,
            actor_id=rejected.actor_id,
        )
