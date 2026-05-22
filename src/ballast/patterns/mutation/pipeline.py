from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance

from ballast.durable import Durable

from ballast.observability.spans import traced
from ballast.observability.trace_names import TraceName
from ballast.patterns.mutation.primitives import (
    AcceptedResult,
    ApplyTransaction,
    Proposal,
    RejectedAt,
    Stage,
    UnitOfWork,
)
from ballast.patterns.mutation.reject_policy import (
    DropOnReject,
    RejectAction,
    RejectPolicy,
)
from ballast.persistence import OutboxRepository
from ballast.runtime.det import Det
from ballast.runtime.idempotency import IdempotencyInput

T = TypeVar("T")

_instance_counter = itertools.count()


@Durable.dbos_class()
class MutationPipeline(DBOSConfiguredInstance, Generic[T]):
    """Stages sequentially → first reject halts → apply in one UoW transaction.

    Idempotency: ``workflow_id`` is derived from
    ``(pipeline_name, proposal_id)``. Retries of the same proposal by a
    flaky parent share workflow_id; DBOS dedupes and returns the cached
    ``AcceptedResult``.

    Apps that need per-run scoping carry it inside the ``proposal``
    payload — the framework's pipeline is identity-agnostic.
    """

    name: ClassVar[str] = "mutation_pipeline"

    def __init__(
        self,
        stages: list[Stage],
        *,
        apply: ApplyTransaction[T],
        uow_factory: Callable[[], UnitOfWork],
        outbox: OutboxRepository,
        event_type: str | None = None,
        reject_policy: RejectPolicy | None = None,
        pipeline_name: str = "mutation_pipeline",
    ) -> None:
        super().__init__(config_name=f"mutation-pipeline-{next(_instance_counter)}")
        self.stages = list(stages)
        self.apply = apply
        self.uow_factory = uow_factory
        self.outbox = outbox
        self.event_type = event_type
        self.reject_policy: RejectPolicy = reject_policy or DropOnReject()
        self.pipeline_name = pipeline_name
        self.name = pipeline_name  # type: ignore[misc]

    async def derive_workflow_id(self, proposal_id: UUID) -> UUID:
        """Public so callers can pre-compute the id (e.g. for DBOS SetWorkflowID)."""
        return await Det.uuid_for(IdempotencyInput(
            namespace="mutation_pipeline",
            parts={
                "pipeline_name": self.pipeline_name,
                "proposal_id": proposal_id,
            },
        ))

    @Durable.workflow()
    @traced(TraceName.PATTERN_MUTATION_PIPELINE, attrs=lambda self, proposal: {
        "pattern": self.name,
    })
    async def run(
        self, proposal: Proposal[T],
    ) -> AcceptedResult[T] | RejectedAt:
        retries_so_far = 0
        current = proposal
        for stage in self.stages:
            result: AcceptedResult[Any] | RejectedAt = await stage.process(current)
            if isinstance(result, RejectedAt):
                action = await self.reject_policy.handle(result, retries_so_far)
                if action == RejectAction.DROP:
                    return result
                if action == RejectAction.ACCEPT:
                    break
                if action == RejectAction.RETRY:
                    retries_so_far += 1
                    continue
            else:
                current = result.proposal
        return await self._apply_and_emit(current)

    @Durable.step()
    async def _apply_and_emit(
        self, proposal: Proposal[T],
    ) -> AcceptedResult[T]:
        async with self.uow_factory() as uow:
            entity_id = await self.apply.apply(proposal, uow=uow)
            if self.event_type is not None:
                await self.outbox.enqueue(
                    event_type=self.event_type,
                    payload={
                        "proposal_id": str(proposal.proposal_id),
                        "entity_id": str(entity_id),
                    },
                )
            return AcceptedResult(proposal=proposal, entity_id=entity_id)
