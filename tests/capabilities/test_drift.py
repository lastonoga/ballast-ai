"""GoalDriftDetector capability surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart, RequestUsage

from ballast.capabilities.drift import GoalDriftDetector
from ballast.drift._core import DriftEngine
from ballast.drift._protocols import DriftCheckSignal, DriftContext


class _RecordingEngine:
    """DriftEngine-like fake — captures every maybe_check invocation."""
    def __init__(self):
        self.calls: list[tuple[DriftCheckSignal, DriftContext]] = []

    async def maybe_check(self, signal, ctx):
        self.calls.append((signal, ctx))
        return None


def _req_ctx(messages):
    """Minimal ModelRequestContext stand-in."""
    @dataclass
    class _C:
        messages: list
        model_settings: Any = None
        model_request_parameters: Any = None
    return _C(messages=messages)


def _resp_with_tool_calls(n: int, in_tokens=10, out_tokens=20) -> ModelResponse:
    parts = [ToolCallPart(tool_name=f"t{i}", args={}, tool_call_id=f"id-{i}") for i in range(n)]
    parts.append(TextPart(content="ok"))
    usage = RequestUsage(input_tokens=in_tokens, output_tokens=out_tokens)
    return ModelResponse(parts=parts, usage=usage)


@pytest.mark.asyncio
async def test_after_model_request_increments_counters_and_invokes_engine() -> None:
    engine = _RecordingEngine()
    cap = GoalDriftDetector(engine=engine)  # type: ignore[arg-type]
    per_run = await cap.for_run(ctx=None)   # type: ignore[arg-type]
    assert per_run is not cap     # fresh instance

    messages = [ModelRequest(parts=[UserPromptPart(content="hi")])]
    rc = _req_ctx(messages)

    response = _resp_with_tool_calls(2, in_tokens=10, out_tokens=20)
    await per_run.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    await per_run.after_model_request(ctx=None, request_context=rc, response=response)  # type: ignore[arg-type]

    assert len(engine.calls) == 1
    sig, drift_ctx = engine.calls[0]
    assert sig.step_index == 1
    assert sig.tool_calls == 2
    assert sig.tokens_used == 30
    assert sig.seconds_elapsed >= 0
    assert drift_ctx.messages == messages
    assert drift_ctx.workflow_input is None


@pytest.mark.asyncio
async def test_for_run_isolates_counters_between_runs() -> None:
    engine = _RecordingEngine()
    cap = GoalDriftDetector(engine=engine)  # type: ignore[arg-type]
    run_a = await cap.for_run(ctx=None)  # type: ignore[arg-type]
    run_b = await cap.for_run(ctx=None)  # type: ignore[arg-type]

    rc = _req_ctx([])
    resp = _resp_with_tool_calls(1, in_tokens=5, out_tokens=5)
    await run_a.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    await run_a.after_model_request(ctx=None, request_context=rc, response=resp)  # type: ignore[arg-type]

    await run_b.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    await run_b.after_model_request(ctx=None, request_context=rc, response=resp)  # type: ignore[arg-type]

    # Each run independently sees step_index=1.
    assert engine.calls[0][0].step_index == 1
    assert engine.calls[1][0].step_index == 1


@pytest.mark.asyncio
async def test_metadata_provider_populates_drift_context_metadata() -> None:
    engine = _RecordingEngine()

    def mp(ctx, request_context):
        return {"budget": {"input_tokens": 99, "max_input_tokens": 100}}

    cap = GoalDriftDetector(engine=engine, metadata_provider=mp)  # type: ignore[arg-type]
    per_run = await cap.for_run(ctx=None)  # type: ignore[arg-type]

    rc = _req_ctx([])
    resp = _resp_with_tool_calls(0)
    await per_run.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    await per_run.after_model_request(ctx=None, request_context=rc, response=resp)  # type: ignore[arg-type]

    assert engine.calls[0][1].metadata == {"budget": {"input_tokens": 99, "max_input_tokens": 100}}


@pytest.mark.asyncio
async def test_engine_exception_is_swallowed() -> None:
    class _Boom:
        async def maybe_check(self, sig, ctx):
            raise RuntimeError("engine down")

    cap = GoalDriftDetector(engine=_Boom())  # type: ignore[arg-type]
    per_run = await cap.for_run(ctx=None)  # type: ignore[arg-type]

    rc = _req_ctx([])
    resp = _resp_with_tool_calls(0)
    await per_run.before_model_request(ctx=None, request_context=rc)  # type: ignore[arg-type]
    # Should NOT raise — exception from engine swallowed.
    await per_run.after_model_request(ctx=None, request_context=rc, response=resp)  # type: ignore[arg-type]
