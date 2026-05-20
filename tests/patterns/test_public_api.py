from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow import (
    AcceptedResult,
    AllowAll,
    ApprovalStage,
    ApprovedResponse,
    Chunker,
    HITLGate,
    HITLPrompt,
    InMemoryHITLChannel,
    MapReduce,
    MutationPipeline,
    Pattern,
    Proposal,
    Reducer,
    Reflection,
    ReflectionExhausted,
    Stage,
)
from pydantic_ai_stateflow.capabilities.helpers import Critique
from pydantic_ai_stateflow.persistence import (
    InMemoryHITLRepository,
    InMemoryOutboxRepository,
)


def test_all_pattern_symbols_visible_at_top_level():
    assert Reflection is not None
    assert MapReduce is not None
    assert MutationPipeline is not None
    assert HITLGate is not None
    assert ApprovalStage is not None
    assert Chunker is not None
    assert Reducer is not None
    assert Stage is not None
    assert ReflectionExhausted is not None


def test_reflection_satisfies_pattern_protocol_structurally():
    pattern = Reflection[str, str](
        writer=lambda t: "ok",
        critic=lambda p: Critique(passed=True),
    )
    assert isinstance(pattern, Pattern)


class _SmokeDoc(BaseModel):
    text: str


class _SmokeRefund(BaseModel):
    amount: int


_SmokeRefundProposal = Proposal[_SmokeRefund]


class _SmokeWordChunker:
    def chunk(self, doc: _SmokeDoc) -> list[str]:
        return doc.text.split()


class _SmokeUniqueReducer:
    async def reduce(self, items: list[str]) -> list[str]:
        return sorted(set(items))


class _SmokeNoopUoW:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *e):
        return None

    async def commit(self):
        return None

    async def rollback(self):
        return None


class _SmokeNoopApply:
    async def apply(self, proposal, *, uow):
        return uuid4()


@pytest.mark.asyncio
async def test_end_to_end_reflection_mapreduce_mutation_hitl_compose(
    fresh_dbos_executor: None,
):
    """Smoke: Reflection emits a doc -> MapReduce dedupes words ->
    each word becomes a Proposal that runs through MutationPipeline whose
    only stage is ApprovalStage (auto-approved by InMemoryHITLChannel)."""

    async def writer(task):
        return _SmokeDoc(text="alpha beta alpha gamma")

    async def critic(payload):
        return Critique(passed=True)

    refl: Pattern[str, _SmokeDoc] = Reflection[str, _SmokeDoc](
        writer=writer, critic=critic,
    )
    doc = await refl.run("seed")
    assert doc.text == "alpha beta alpha gamma"

    async def extractor(chunk: str) -> str | None:
        return chunk

    mr: Pattern[_SmokeDoc, list[str]] = MapReduce[_SmokeDoc, str, str](
        chunker=_SmokeWordChunker(),
        extractor=extractor,
        reducer=_SmokeUniqueReducer(),
    )
    words = await mr.run(doc)
    assert words == ["alpha", "beta", "gamma"]

    channel = InMemoryHITLChannel()
    repo = InMemoryHITLRepository()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    orig_persist = repo.persist_request

    async def auto_approve(**kw):
        req = await orig_persist(**kw)
        channel.set_response(
            req.id,
            ApprovedResponse(actor_id="alice", answered_at=datetime.now(tz=UTC)),
        )
        return req

    repo.persist_request = auto_approve  # type: ignore[method-assign]

    approval = ApprovalStage[_SmokeRefund](
        hitl=gate,
        prompt_builder=lambda p: HITLPrompt(
            title=f"refund {p.payload.amount}",
            context=str(p.payload),
            decision_kinds={"approved", "rejected"},
        ),
    )
    pipeline = MutationPipeline[_SmokeRefund](
        stages=[approval],
        apply=_SmokeNoopApply(),
        uow_factory=lambda: _SmokeNoopUoW(),
        outbox=InMemoryOutboxRepository(),
    )

    accepted_count = 0
    for word in words:
        proposal = Proposal[_SmokeRefund](
            proposal_id=uuid4(),
            payload=_SmokeRefund(amount=len(word)),
        )
        result = await pipeline.run(proposal)
        if isinstance(result, AcceptedResult):
            accepted_count += 1
    assert accepted_count == len(words)
