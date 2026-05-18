from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance

from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.patterns.mutation.primitives import (
    AcceptedResult,
    ApplyTransaction,
    Proposal,
    RejectedAt,
    Stage,
)
from pydantic_ai_stateflow.patterns.mutation.reject_policy import (
    DropOnReject,
    RejectAction,
    RejectPolicy,
)
from pydantic_ai_stateflow.persistence import OutboxRepository, UnitOfWork
from pydantic_ai_stateflow.runtime.det import Det
from pydantic_ai_stateflow.runtime.idempotency import IdempotencyInput

T = TypeVar("T")

_instance_counter = itertools.count()


@DBOS.dbos_class()
class MutationPipeline(DBOSConfiguredInstance, Generic[T]):
    """Stages sequentially → first reject halts → apply in one UoW transaction.

    Idempotency (spec 2C.3 post-review): workflow_id is derived from
    (pipeline_name, proposal_id, tenant_id). Retries of the same proposal
    by a flaky parent share workflow_id; DBOS dedupes and returns the
    cached AcceptedResult.

    The pipeline is `Pattern[Proposal[T], AcceptedResult[T] | RejectedAt]`
    structurally.
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

    async def derive_workflow_id(self, proposal_id: UUID, tenant_id: UUID) -> UUID:
        """Public so callers can pre-compute the id (e.g. for DBOS SetWorkflowID)."""
        return await Det.uuid_for(IdempotencyInput(
            namespace="mutation_pipeline",
            parts={
                "pipeline_name": self.pipeline_name,
                "proposal_id": proposal_id,
                "tenant_id": tenant_id,
            },
        ))

    @DBOS.workflow()
    @traced("pattern.mutation_pipeline", attrs=lambda self, proposal, *, tenant_id: {
        "tenant_id": str(tenant_id), "pattern": self.name,
    })
    async def run(
        self, proposal: Proposal[T], *, tenant_id: UUID,
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
        return await self._apply_and_emit(current, tenant_id=tenant_id)

    @DBOS.step()
    async def _apply_and_emit(
        self, proposal: Proposal[T], *, tenant_id: UUID,
    ) -> AcceptedResult[T]:
        async with self.uow_factory() as uow:
            entity_id = await self.apply.apply(
                proposal, uow=uow, tenant_id=tenant_id,
            )
            if self.event_type is not None:
                await self.outbox.enqueue(
                    event_type=self.event_type,
                    payload={
                        "proposal_id": str(proposal.proposal_id),
                        "entity_id": str(entity_id),
                    },
                    tenant_id=tenant_id,
                )
            return AcceptedResult(proposal=proposal, entity_id=entity_id)
