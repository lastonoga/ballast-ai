import json

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow.capabilities import SemanticLoopDetector


class _IdentityEmbedder:
    """Returns deterministic vector derived from input text."""

    async def embed(self, text: str) -> list[float]:
        h = abs(hash(text)) % 1_000_000
        return [float(h % 100), float((h // 100) % 100), float((h // 10000) % 100)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.mark.asyncio
async def test_default_selector_extracts_text_and_toolcalls() -> None:
    """Default selector serialises TextPart + ToolCallPart args stably."""
    from pydantic_ai_stateflow.capabilities.semantic_loop import _default_response_text

    resp = ModelResponse(
        parts=[
            TextPart(content="hello"),
            ToolCallPart(tool_name="do", args={"k": 1}),
        ]
    )
    snap = _default_response_text(resp)
    assert "hello" in snap
    assert "do" in snap
    assert json.dumps({"k": 1}, sort_keys=True) in snap


@pytest.mark.asyncio
async def test_loop_detector_allows_diverse_responses() -> None:
    """A single non-looping run completes cleanly."""
    counter = {"i": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        counter["i"] += 1
        return ModelResponse(parts=[TextPart(content=f"answer_{counter['i']}")])

    detector = SemanticLoopDetector(
        embedder=_IdentityEmbedder(), threshold=0.99, window=2,
    )
    agent = Agent(model=FunctionModel(fn), capabilities=[detector])
    await agent.run("hi")
