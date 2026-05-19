"""Tests for ``make_runner(thread_repo=...)`` message_history wiring.

The runner reconstructs ``message_history`` from the supplied
``ThreadRepository`` and passes a ``list[ModelMessage]`` to
``agent.iter(...)``. This is the server-stateful pillar: clients ship
only the current user turn and the backend rehydrates context.

The fake agent records the ``message_history`` it received so we can
assert the conversion + dedup logic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.api.streaming import StreamEvent, make_runner
from pydantic_ai_stateflow.api.streaming.router import _PostMessageBody
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)


@dataclass
class _FakeRunResult:
    output: Any = None


@dataclass
class _FakeCtx:
    pass


@dataclass
class _FakeModelStream:
    events: list[Any]

    async def __aenter__(self) -> _FakeModelStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        for ev in self.events:
            yield ev


@dataclass
class _FakeModelRequestNode:
    events: list[Any] = field(default_factory=list)

    def stream(self, _ctx: _FakeCtx) -> _FakeModelStream:
        return _FakeModelStream(self.events)


@dataclass
class _FakeAgentRun:
    nodes: list[Any]
    ctx: _FakeCtx = field(default_factory=_FakeCtx)
    result: _FakeRunResult = field(default_factory=_FakeRunResult)

    async def __aenter__(self) -> _FakeAgentRun:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self) -> AsyncIterator[Any]:
        for n in self.nodes:
            yield n


class _RecordingAgent:
    """Records ``iter()``'s ``message_history`` kwarg per call."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_message_history: Any = "UNSET"  # sentinel
        self.last_deps: Any = None

    def iter(
        self,
        prompt: str,
        *,
        deps: Any = None,
        message_history: Any = None,
    ) -> _FakeAgentRun:
        self.last_prompt = prompt
        self.last_deps = deps
        self.last_message_history = message_history
        return _FakeAgentRun(nodes=[_FakeModelRequestNode(events=[])])


@pytest.fixture(autouse=True)
def _patch_agent_node_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai import Agent

    def _is_mr(node: Any) -> bool:
        return isinstance(node, _FakeModelRequestNode)

    def _is_ct(_node: Any) -> bool:
        return False

    def _is_end(_node: Any) -> bool:
        return False

    monkeypatch.setattr(Agent, "is_model_request_node", staticmethod(_is_mr))
    monkeypatch.setattr(Agent, "is_call_tools_node", staticmethod(_is_ct))
    monkeypatch.setattr(Agent, "is_end_node", staticmethod(_is_end))


async def _drive(runner: Any, *, thread_id: Any, tenant_id: Any, prompt: str) -> None:
    body = _PostMessageBody.model_validate(
        {"role": "user", "parts": [{"type": "text", "text": prompt}]},
    )
    async for _ev in runner(
        thread_id=thread_id, run_id=uuid4(), message=body, tenant_id=tenant_id,
    ):
        # Drain — the fake agent emits nothing besides framing events.
        assert isinstance(_ev, StreamEvent)


@pytest.mark.asyncio
async def test_make_runner_passes_repo_history_as_message_history() -> None:
    """Three prior turns in the repo become a 3-message ModelMessage list."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    repo = InMemoryThreadRepository()
    tenant_id = uuid4()
    thread = await repo.create(
        purpose="conversation",
        purpose_metadata={},
        actor_id="user",
        tenant_id=tenant_id,
    )
    # Seed three prior turns.
    await repo.add_message(
        thread.id, role="user",
        parts=[{"type": "text", "text": "first user"}], tenant_id=tenant_id,
    )
    await repo.add_message(
        thread.id, role="assistant",
        parts=[{"type": "text", "text": "first assistant"}], tenant_id=tenant_id,
    )
    await repo.add_message(
        thread.id, role="user",
        parts=[{"type": "text", "text": "second user"}], tenant_id=tenant_id,
    )
    await repo.add_message(
        thread.id, role="assistant",
        parts=[{"type": "text", "text": "second assistant"}], tenant_id=tenant_id,
    )

    agent = _RecordingAgent()
    runner = make_runner(
        agent,  # type: ignore[arg-type]
        text_field=lambda out: out or "",
        thread_repo=repo,
    )
    await _drive(runner, thread_id=thread.id, tenant_id=tenant_id, prompt="now")

    history = agent.last_message_history
    assert isinstance(history, list)
    assert len(history) == 4
    assert isinstance(history[0], ModelRequest)
    p0 = history[0].parts[0]
    assert isinstance(p0, UserPromptPart)
    assert p0.content == "first user"
    assert isinstance(history[1], ModelResponse)
    p1 = history[1].parts[0]
    assert isinstance(p1, TextPart)
    assert p1.content == "first assistant"
    assert isinstance(history[2], ModelRequest)
    p2 = history[2].parts[0]
    assert isinstance(p2, UserPromptPart)
    assert p2.content == "second user"
    assert isinstance(history[3], ModelResponse)
    p3 = history[3].parts[0]
    assert isinstance(p3, TextPart)
    assert p3.content == "second assistant"


@pytest.mark.asyncio
async def test_make_runner_excludes_current_prompt_from_history() -> None:
    """The just-persisted user turn (router persists pre-runner) is filtered."""
    repo = InMemoryThreadRepository()
    tenant_id = uuid4()
    thread = await repo.create(
        purpose="conversation", purpose_metadata={},
        actor_id="user", tenant_id=tenant_id,
    )
    # Prior turn.
    await repo.add_message(
        thread.id, role="user",
        parts=[{"type": "text", "text": "hello"}], tenant_id=tenant_id,
    )
    await repo.add_message(
        thread.id, role="assistant",
        parts=[{"type": "text", "text": "hi back"}], tenant_id=tenant_id,
    )
    # Simulate the router persisting the current user turn before the runner.
    current_prompt = "what's up?"
    await repo.add_message(
        thread.id, role="user",
        parts=[{"type": "text", "text": current_prompt}], tenant_id=tenant_id,
    )

    agent = _RecordingAgent()
    runner = make_runner(
        agent,  # type: ignore[arg-type]
        text_field=lambda out: out or "",
        thread_repo=repo,
    )
    await _drive(
        runner, thread_id=thread.id, tenant_id=tenant_id, prompt=current_prompt,
    )

    from pydantic_ai.messages import TextPart, UserPromptPart

    history = agent.last_message_history
    assert isinstance(history, list)
    assert len(history) == 2  # prior user + prior assistant (current excluded)
    p0 = history[0].parts[0]
    assert isinstance(p0, UserPromptPart)
    assert p0.content == "hello"
    p1 = history[1].parts[0]
    assert isinstance(p1, TextPart)
    assert p1.content == "hi back"


@pytest.mark.asyncio
async def test_make_runner_without_thread_repo_passes_no_history() -> None:
    """Backward compat: omitting ``thread_repo`` => message_history=None."""
    agent = _RecordingAgent()
    runner = make_runner(
        agent,  # type: ignore[arg-type]
        text_field=lambda out: out or "",
    )
    await _drive(runner, thread_id=uuid4(), tenant_id=uuid4(), prompt="hi")

    assert agent.last_message_history is None


@pytest.mark.asyncio
async def test_make_runner_skips_empty_text_rows() -> None:
    """Rows with empty/no text are dropped from the rehydrated history."""
    repo = InMemoryThreadRepository()
    tenant_id = uuid4()
    thread = await repo.create(
        purpose="conversation", purpose_metadata={},
        actor_id="user", tenant_id=tenant_id,
    )
    await repo.add_message(
        thread.id, role="user",
        parts=[{"type": "text", "text": "real text"}], tenant_id=tenant_id,
    )
    # Empty assistant row (e.g. a tool-only turn captured as no text).
    await repo.add_message(
        thread.id, role="assistant",
        parts=[{"type": "text", "text": ""}], tenant_id=tenant_id,
    )

    agent = _RecordingAgent()
    runner = make_runner(
        agent,  # type: ignore[arg-type]
        text_field=lambda out: out or "",
        thread_repo=repo,
    )
    await _drive(runner, thread_id=thread.id, tenant_id=tenant_id, prompt="now")

    from pydantic_ai.messages import UserPromptPart

    history = agent.last_message_history
    assert isinstance(history, list)
    assert len(history) == 1
    p0 = history[0].parts[0]
    assert isinstance(p0, UserPromptPart)
    assert p0.content == "real text"
