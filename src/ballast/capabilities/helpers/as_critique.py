from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


class Critique(BaseModel):
    """Canonical critique shape used by Reflection (SP5)."""

    passed: bool
    issues: list[str] = []
    suggestions: list[str] = []
    confidence: float = 1.0


def _coerce_to_critique(value: Any) -> Critique:
    if isinstance(value, Critique):
        return value
    if isinstance(value, bool):
        return Critique(passed=value, confidence=1.0)
    passed = getattr(value, "passed", None)
    if isinstance(passed, bool):
        return Critique(
            passed=passed,
            issues=list(getattr(value, "issues", []) or []),
            suggestions=list(getattr(value, "suggestions", []) or []),
            confidence=float(getattr(value, "confidence", 1.0)),
        )
    raise TypeError(f"Cannot coerce {type(value).__name__} to Critique")


def _extract_payload(messages: list[ModelMessage]) -> Any:
    # Walk from the end for the first part exposing `.content` — robust to
    # pydantic-ai message shapes that mix system / tool / user parts.
    for message in reversed(messages):
        for part in getattr(message, "parts", ()):
            content = getattr(part, "content", None)
            if content is not None:
                return content
    return None


def as_critique(fn: Callable[[Any], Awaitable[Any]] | Any) -> Agent[Any, Critique]:
    """Wrap a non-LLM critic (callable or object with .check) as a pydantic-ai Agent.

    Lets Reflection (SP5) accept any critic - LLM agent, plain Python
    function, or stateful object with a `check()` method - through a single
    uniform interface (`Agent.run(...)`). Internally uses FunctionModel so no
    real LLM is invoked.
    """
    callable_fn: Callable[[Any], Awaitable[Any]] = (
        fn.check if hasattr(fn, "check") and callable(fn.check) else fn
    )

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        payload = _extract_payload(messages)
        verdict = await callable_fn(payload)
        critique = _coerce_to_critique(verdict)
        return ModelResponse(
            parts=[ToolCallPart(tool_name="final_result", args=critique.model_dump())]
        )

    return Agent(model=FunctionModel(model_fn), output_type=Critique)
