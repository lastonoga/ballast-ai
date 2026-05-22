from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


@runtime_checkable
class UnitOfWork(Protocol):
    """Async context manager that commits on clean exit, rolls back on exception."""

    async def __aenter__(self) -> UnitOfWork: ...
    async def __aexit__(self, *exc_info: object) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


class Proposal(BaseModel, Generic[T]):
    """A pending mutation. Drives idempotency in MutationPipeline via proposal_id."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    proposal_id: UUID
    payload: T
    actor_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AcceptedResult(BaseModel, Generic[T]):
    """Successful pipeline outcome: the (possibly modified) proposal was applied."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    proposal: Proposal[T]
    entity_id: UUID


class RejectedAt(BaseModel):
    """Stage X said no. Pipeline halts at first RejectedAt."""

    model_config = ConfigDict(frozen=True)

    stage: str
    reason: str
    actor_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Stage(Protocol):
    """A pipeline step. Returns AcceptedResult (continue) or RejectedAt (halt)."""

    name: str

    async def process(
        self, proposal: Proposal[Any],
    ) -> AcceptedResult[Any] | RejectedAt: ...


@runtime_checkable
class ApplyTransaction(Protocol, Generic[T]):
    """Transactional terminal step: applies an accepted proposal to the system of record.

    Implementations should call repos through `uow.session` so the write
    and the outbox event live in one transaction (transactional outbox).
    """

    async def apply(
        self,
        proposal: Proposal[T],
        *,
        uow: UnitOfWork,
    ) -> UUID: ...
