"""End-to-end smoke test exercising every Sub-project #1 component."""

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow import (
    Det,
    GroundedAgent,
    IdempotencyInput,
    Ref,
)


class Candidate(BaseModel):
    id: UUID
    label: str
    score: float


class Customer(BaseModel):
    id: UUID
    email: str


class Context(BaseModel):
    customer: Customer
    candidates: list[Candidate]


class Decision(BaseModel):
    chosen_customer: Ref[Customer]
    chosen_candidate: Ref[Candidate]
    rationale: str


class FakeRepo:
    def __init__(self, mapping: dict[UUID, object]) -> None:
        self._mapping = mapping

    async def load(self, id: UUID):
        return self._mapping[id]


@pytest.mark.asyncio
async def test_full_grounded_flow_with_hydration_and_idempotency():
    customer_id = uuid4()
    candidate_ids = [uuid4(), uuid4(), uuid4()]
    customer = Customer(id=customer_id, email="who@where.com")
    candidates = [
        Candidate(id=i, label=f"c{idx}", score=idx * 0.1)
        for idx, i in enumerate(candidate_ids)
    ]

    ctx = Context(customer=customer, candidates=candidates)

    # FunctionModel that returns the second candidate
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result",
            args={
                "chosen_customer": str(customer_id),
                "chosen_candidate": str(candidate_ids[1]),
                "rationale": "highest available score",
            },
        )])

    base_agent: Agent[None, Decision] = Agent(
        model=FunctionModel(fn), output_type=Decision,
    )
    grounded = GroundedAgent(base_agent, output_type=Decision)
    result = await grounded.run(ctx, instructions="pick best candidate")

    # Typed Ref instances
    assert isinstance(result.value.chosen_customer, Ref)
    assert isinstance(result.value.chosen_candidate, Ref)
    assert result.value.chosen_customer.id == customer_id
    assert result.value.chosen_candidate.id == candidate_ids[1]

    # Hydrate by entity class name
    cust_repo = FakeRepo({customer_id: customer})
    cand_repo = FakeRepo({c.id: c for c in candidates})
    hydrated = await result.hydrate(Customer=cust_repo, Candidate=cand_repo)

    assert isinstance(hydrated["chosen_customer"], Customer)
    assert isinstance(hydrated["chosen_candidate"], Candidate)
    assert hydrated["chosen_candidate"].label == "c1"

    # Det.uuid_for produces a stable idempotency key for this run
    key = await Det.uuid_for(IdempotencyInput(
        namespace="smoke",
        parts={
            "customer_id": customer_id,
            "chosen_candidate_id": result.value.chosen_candidate.id,
        },
    ))
    assert isinstance(key, UUID)
    assert key.version == 5
