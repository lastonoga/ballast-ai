"""Tests for `make_runner`'s tool-call streaming + deps-factory paths.

Drives a *real* ``pydantic_ai.Agent`` against a ``FunctionModel`` so the
adapter exercises the actual ``agent.iter`` event taxonomy from
pydantic-ai 1.97.0 (``PartStartEvent`` / ``PartDeltaEvent`` /
``FunctionToolCallEvent`` / ``FinalResultEvent``).

The ``test_pydantic_ai_adapter.py`` file covers the FakeAgent text-only
path; this file covers everything that requires the real graph
iterator's tool-call event shape.
"""

# NOTE: NO `from __future__ import annotations` here. pydantic-ai's tool
# registration uses ``get_type_hints()`` at decoration time, which fails
# on locally-scoped types (e.g. ``_ToolDeps`` defined inside a test fn)
# under postponed evaluation. The annotations on these tests' tools are
# concrete (str / UUID / dataclass), so eager evaluation is fine.

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from pydantic_ai_stateflow.api.streaming import StreamEvent, make_runner
from pydantic_ai_stateflow.api.streaming.kinds import StreamEventKind
from pydantic_ai_stateflow.api.streaming.router import _PostMessageBody


class Reply(BaseModel):
    reply: str = ""


@dataclass
class _Deps:
    """Simple per-request deps for the factory tests."""

    label: str
    seen_kwargs: dict[str, Any]


async def _collect_runner(runner: Any) -> list[StreamEvent]:
    body = _PostMessageBody.model_validate(
        {"role": "user", "parts": [{"type": "text", "text": "do it"}]},
    )
    out: list[StreamEvent] = []
    async for ev in runner(
        thread_id=uuid4(), run_id=uuid4(), message=body, tenant_id=uuid4(),
    ):
        out.append(ev)
    return out


def _kinds(events: list[StreamEvent]) -> list[str]:
    return [e.kind for e in events]


# ---------------------------------------------------------------------------
# F12 — deps factory
# ---------------------------------------------------------------------------


def _make_trivial_agent() -> Agent[_Deps | None, Reply]:
    """An agent that immediately returns the final structured output —
    no real tool calls. Used to assert deps-factory invocation without
    exercising the tool-call path.
    """

    async def stream_fn(
        _messages: list[ModelMessage], _info: AgentInfo,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        yield {
            0: DeltaToolCall(
                name="final_result",
                json_args='{"reply": "ok"}',
                tool_call_id="call_final",
            ),
        }

    return Agent(
        FunctionModel(stream_function=stream_fn),
        output_type=Reply,
        deps_type=_Deps,
    )


@pytest.mark.asyncio
async def test_make_runner_calls_deps_factory_per_request() -> None:
    """A callable ``deps`` is invoked once per stream with the runner kwargs."""
    seen: dict[str, Any] = {}

    def factory(**kwargs: Any) -> _Deps:
        seen.update(kwargs)
        return _Deps(label="from-factory", seen_kwargs=kwargs)

    agent = _make_trivial_agent()
    runner = make_runner(agent, text_field="reply", deps=factory)

    tid, rid, tenant = uuid4(), uuid4(), uuid4()
    body = _PostMessageBody.model_validate(
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
    )
    async for _ev in runner(
        thread_id=tid, run_id=rid, message=body, tenant_id=tenant,
    ):
        pass

    assert seen["thread_id"] == tid
    assert seen["run_id"] == rid
    assert seen["tenant_id"] == tenant
    assert seen["message"] is body


@pytest.mark.asyncio
async def test_make_runner_supports_static_deps_value() -> None:
    """Non-callable ``deps`` is passed through unchanged on every call."""

    @dataclass(frozen=True)
    class _Static:
        token: str = "static"

    static = _Static()
    agent = _make_trivial_agent()
    captured: list[Any] = []

    # Wrap the agent's iter to capture what deps got passed.
    real_iter = agent.iter

    def _spying_iter(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs.get("deps"))
        return real_iter(*args, **kwargs)

    agent.iter = _spying_iter  # type: ignore[method-assign]

    runner = make_runner(agent, text_field="reply", deps=static)
    await _collect_runner(runner)
    assert captured == [static]


@pytest.mark.asyncio
async def test_make_runner_supports_async_deps_factory() -> None:
    """A coroutine-function ``deps`` is awaited and its result forwarded."""
    agent = _make_trivial_agent()
    captured: list[Any] = []

    real_iter = agent.iter

    def _spying_iter(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs.get("deps"))
        return real_iter(*args, **kwargs)

    agent.iter = _spying_iter  # type: ignore[method-assign]

    async def factory(**_kwargs: Any) -> _Deps:
        return _Deps(label="async", seen_kwargs=_kwargs)

    runner = make_runner(agent, text_field="reply", deps=factory)
    await _collect_runner(runner)
    assert len(captured) == 1
    assert isinstance(captured[0], _Deps)
    assert captured[0].label == "async"


# ---------------------------------------------------------------------------
# F13 — tool-call SSE events
# ---------------------------------------------------------------------------


def _make_tool_agent(
    *, calls: list[dict[int, DeltaToolCall]],
) -> Agent[None, Reply]:
    """Build an agent that streams the given per-turn tool calls + final output.

    The ``calls`` list is one entry per turn: each entry is a
    ``DeltaToolCall`` map (the FunctionModel stream protocol). After all
    custom turns are exhausted the model emits the synthetic
    ``final_result`` call for ``Reply``.
    """
    turn = {"i": 0}

    async def stream_fn(
        _messages: list[ModelMessage], _info: AgentInfo,
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        idx = turn["i"]
        turn["i"] += 1
        if idx < len(calls):
            yield "thinking..."
            yield calls[idx]
        else:
            yield {
                0: DeltaToolCall(
                    name="final_result",
                    json_args='{"reply": "done"}',
                    tool_call_id="call_final",
                ),
            }

    agent: Agent[None, Reply] = Agent(
        FunctionModel(stream_function=stream_fn),
        output_type=Reply,
    )

    @agent.tool_plain
    def echo(msg: str) -> str:
        return msg

    @agent.tool_plain
    def shout(msg: str) -> str:
        return msg.upper()

    return agent


@pytest.mark.asyncio
async def test_make_runner_emits_tool_call_events_for_each_tool_call() -> None:
    agent = _make_tool_agent(
        calls=[
            {
                0: DeltaToolCall(
                    name="echo",
                    json_args='{"msg": "hi"}',
                    tool_call_id="call_1",
                ),
            },
        ],
    )
    runner = make_runner(agent, text_field="reply")
    events = await _collect_runner(runner)
    kinds = _kinds(events)

    # Canonical envelope.
    assert kinds[0] == "RUN_STARTED"
    assert kinds[1] == "TEXT_MESSAGE_START"
    assert kinds[-1] == "RUN_FINISHED"
    assert kinds[-2] == "TEXT_MESSAGE_END"

    # Tool-call trio is present and for `echo` (NOT for `final_result`).
    tool_events = [
        e for e in events
        if e.kind in {
            StreamEventKind.TOOL_CALL_START.value,
            StreamEventKind.TOOL_CALL_ARGS.value,
            StreamEventKind.TOOL_CALL_END.value,
        }
    ]
    starts = [e for e in tool_events if e.kind == "TOOL_CALL_START"]
    ends = [e for e in tool_events if e.kind == "TOOL_CALL_END"]
    args = [e for e in tool_events if e.kind == "TOOL_CALL_ARGS"]
    assert len(starts) == 1
    assert len(ends) == 1
    assert len(args) >= 1
    assert starts[0].data["toolCallName"] == "echo"
    assert starts[0].data["toolCallId"] == "call_1"
    assert ends[0].data["toolCallId"] == "call_1"
    # All args events carry the same tool_call_id as their start.
    assert all(a.data["toolCallId"] == "call_1" for a in args)
    # Args payload contains the echo argument.
    combined = "".join(a.data["delta"] for a in args)
    assert "hi" in combined

    # The synthetic final_result tool MUST NOT leak as a tool_call.
    final_ids = [
        e.data.get("toolCallId") for e in tool_events
    ]
    assert "call_final" not in final_ids

    # Text content includes the model's prose AND the final reply text
    # (whichever path the adapter used to surface "done").
    text_deltas = [
        e.data["delta"]
        for e in events
        if e.kind == "TEXT_MESSAGE_CONTENT"
    ]
    combined_text = "".join(text_deltas)
    assert "thinking" in combined_text or "done" in combined_text


@pytest.mark.asyncio
async def test_make_runner_handles_multiple_tool_calls_in_one_turn() -> None:
    """Two tool calls in a single model turn each get a distinct trio."""
    agent = _make_tool_agent(
        calls=[
            {
                0: DeltaToolCall(
                    name="echo",
                    json_args='{"msg": "a"}',
                    tool_call_id="call_a",
                ),
                1: DeltaToolCall(
                    name="shout",
                    json_args='{"msg": "b"}',
                    tool_call_id="call_b",
                ),
            },
        ],
    )
    runner = make_runner(agent, text_field="reply")
    events = await _collect_runner(runner)
    starts = [e for e in events if e.kind == "TOOL_CALL_START"]
    ends = [e for e in events if e.kind == "TOOL_CALL_END"]
    start_ids = {e.data["toolCallId"] for e in starts}
    end_ids = {e.data["toolCallId"] for e in ends}
    assert {"call_a", "call_b"} <= start_ids
    assert {"call_a", "call_b"} <= end_ids
    # Tool names matched correctly.
    by_id_name = {
        e.data["toolCallId"]: e.data["toolCallName"] for e in starts
    }
    assert by_id_name["call_a"] == "echo"
    assert by_id_name["call_b"] == "shout"


@pytest.mark.asyncio
async def test_make_runner_threads_per_request_deps_into_tools() -> None:
    """End-to-end: the deps factory's value reaches the tool via RunContext."""

    @dataclass
    class _ToolDeps:
        marker: str

    agent: Agent[_ToolDeps, Reply] = Agent(
        FunctionModel(stream_function=_stream_marker_then_final),
        output_type=Reply,
        deps_type=_ToolDeps,
    )

    seen_markers: list[str] = []

    @agent.tool
    def grab_marker(ctx: RunContext[_ToolDeps]) -> str:
        seen_markers.append(ctx.deps.marker)
        return ctx.deps.marker

    def factory(*, tenant_id: UUID, **_: Any) -> _ToolDeps:
        return _ToolDeps(marker=f"tenant:{tenant_id}")

    runner = make_runner(agent, text_field="reply", deps=factory)
    tenant = uuid4()
    body = _PostMessageBody.model_validate(
        {"role": "user", "parts": [{"type": "text", "text": "go"}]},
    )
    async for _ev in runner(
        thread_id=uuid4(), run_id=uuid4(), message=body, tenant_id=tenant,
    ):
        pass

    assert seen_markers == [f"tenant:{tenant}"]


async def _stream_marker_then_final(
    _messages: list[ModelMessage], _info: AgentInfo,
) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
    """Helper shared by the per-request-deps test: call the tool once,
    then emit the structured final_result on the next turn.
    """
    if len(_messages) <= 2:
        yield {
            0: DeltaToolCall(
                name="grab_marker",
                json_args="{}",
                tool_call_id="call_grab",
            ),
        }
    else:
        yield {
            0: DeltaToolCall(
                name="final_result",
                json_args='{"reply": "done"}',
                tool_call_id="call_final",
            ),
        }
