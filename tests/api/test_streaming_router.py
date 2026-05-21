"""Tests for the server-stateful Vercel-AI streaming endpoint built on
``pydantic_ai.ui.vercel_ai.VercelAIAdapter``.

The framework owns:
  - thread + message persistence (404 on missing thread, no lazy-create)
  - user-message persistence before the stream starts
  - ``message_history`` reconstruction from the repo
  - assistant-reply persistence via the ``on_complete`` callback
  - ``deps_factory`` invocation per request

The wire format, body parsing, and event taxonomy are delegated to
``VercelAIAdapter`` — we DON'T re-test those.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_stateflow.api.streaming.router import streaming_router
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.runtime import StateflowAgent


class _TestStateflowAgent(StateflowAgent):
    """Test seam: wraps a pre-built pydantic-ai ``Agent`` + optional
    ``deps_factory`` + ``model_settings`` into a ``StateflowAgent``
    instance the registry can resolve."""

    name = "conversation"

    def __init__(
        self,
        agent: Agent[Any, Any],
        *,
        deps_factory: Any = None,
        model_settings: Any = None,
    ) -> None:
        self._agent = agent
        self._deps_factory = deps_factory
        self._model_settings = model_settings

    def build_agent(self) -> Agent[Any, Any]:
        return self._agent

    async def build_deps(self, *, thread: Any, message: Any) -> Any:
        if self._deps_factory is None:
            return None
        result = self._deps_factory(thread_id=thread.id, message=message)
        import inspect as _inspect
        if _inspect.isawaitable(result):
            return await result
        return result

    def model_settings(self) -> Any:
        return self._model_settings


def _ag_ui_body(*, thread_id: UUID, user_text: str) -> dict[str, Any]:
    """Build a minimal Vercel-AI ``SubmitMessage`` body."""
    return {
        "trigger": "submit-message",
        "id": str(thread_id),
        "messages": [
            {
                "id": str(uuid4()),
                "role": "user",
                "parts": [
                    {"type": "text", "text": user_text, "state": "done"},
                ],
            },
        ],
    }


def _build_app(
    repo: InMemoryThreadRepository,
    agent: Agent[Any, Any],
    *,
    deps_factory: Any = None,
    model_settings: Any = None,
) -> FastAPI:
    """Wire ``agent`` as the ``"conversation"`` StateflowAgent and
    build a streaming-router-only FastAPI app over ``repo``.

    Uses the module-level ``streaming_router`` + ``app.state``-based
    agent resolution (no process-global registry).
    """
    from pydantic_ai_stateflow.api.error_middleware import install_error_handlers
    from pydantic_ai_stateflow.persistence import (
        InMemoryEventLogRepository,
    )
    from pydantic_ai_stateflow.runtime.event_stream import InProcessEventStream

    app = FastAPI()
    app.state.thread_repo = repo
    app.state.event_log = InMemoryEventLogRepository()
    app.state.event_stream = InProcessEventStream()
    app.state.agents = {
        "conversation": _TestStateflowAgent(
            agent,
            deps_factory=deps_factory,
            model_settings=model_settings,
        ),
    }
    app.include_router(streaming_router)
    install_error_handlers(app)
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_returns_404_when_thread_missing() -> None:
    repo = InMemoryThreadRepository()
    agent: Agent[None, str] = Agent(TestModel(), output_type=str)
    app = _build_app(repo, agent)

    missing = uuid4()
    async with _client(app) as c:
        r = await c.post(
            f"/threads/{missing}/messages",
            json=_ag_ui_body(thread_id=missing, user_text="hi"),
            headers={"Accept": "text/event-stream"},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_persists_user_message_before_stream() -> None:
    repo = InMemoryThreadRepository()
    thread = await repo.create(agent="conversation", metadata={})
    agent: Agent[None, str] = Agent(TestModel(custom_output_text="ok"), output_type=str)
    app = _build_app(repo, agent)

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="hello world"),
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200
        _ = r.text

    msgs = await repo.history(thread.id)
    roles = [m.role for m in msgs]
    assert "user" in roles
    user_row = next(m for m in msgs if m.role == "user")
    assert user_row.parts == [
        {"type": "text", "text": "hello world", "state": "done"},
    ]


@pytest.mark.asyncio
async def test_assistant_reply_persisted_via_on_complete() -> None:
    repo = InMemoryThreadRepository()
    thread = await repo.create(agent="conversation", metadata={})
    agent: Agent[None, str] = Agent(
        TestModel(custom_output_text="hi there"), output_type=str,
    )
    app = _build_app(repo, agent)

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="hi"),
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200
        _ = r.text

    msgs = await repo.history(thread.id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    text_parts = [
        p for p in msgs[1].parts if p.get("type") == "text"
    ]
    assert text_parts, msgs[1].parts
    assert any(p.get("text") == "hi there" for p in text_parts), text_parts


@pytest.mark.asyncio
async def test_message_history_reconstructed_from_repo() -> None:
    """Body carries full history (matching useChat behavior); body-vs-DB
    sync keeps the prior 3 rows + appends the new turn-3 user msg.
    Agent then sees all 3 user turns in ``message_history``.
    """
    repo = InMemoryThreadRepository()
    thread = await repo.create(agent="conversation", metadata={})
    t1u = await repo.add_message(
        thread.id, role="user", id="t1u",
        parts=[{"type": "text", "text": "turn1-user", "state": "done"}],
    )
    t1a = await repo.add_message(
        thread.id, role="assistant", id="t1a",
        parts=[{"type": "text", "text": "turn1-assistant", "state": "done"}],
    )
    t2u = await repo.add_message(
        thread.id, role="user", id="t2u",
        parts=[{"type": "text", "text": "turn2-user", "state": "done"}],
    )

    seen_messages: list[list[Any]] = []

    async def capture(messages: list[Any], _info: AgentInfo) -> ModelResponse:
        seen_messages.append(messages)
        return ModelResponse(parts=[TextPart(content="ack")])

    async def capture_stream(
        messages: list[Any], _info: AgentInfo,
    ) -> AsyncIterator[str]:
        seen_messages.append(messages)
        yield "ack"

    agent: Agent[None, str] = Agent(
        FunctionModel(capture, stream_function=capture_stream), output_type=str,
    )
    app = _build_app(repo, agent)

    body = {
        "trigger": "submit-message",
        "id": str(thread.id),
        "messages": [
            {
                "id": t1u.id, "role": "user",
                "parts": [{"type": "text", "text": "turn1-user", "state": "done"}],
            },
            {
                "id": t1a.id, "role": "assistant",
                "parts": [{"type": "text", "text": "turn1-assistant", "state": "done"}],
            },
            {
                "id": t2u.id, "role": "user",
                "parts": [{"type": "text", "text": "turn2-user", "state": "done"}],
            },
            {
                "id": str(uuid4()), "role": "user",
                "parts": [{"type": "text", "text": "turn3-user", "state": "done"}],
            },
        ],
    }

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=body,
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200
        _ = r.text

    assert len(seen_messages) == 1
    user_texts: list[str] = []
    for m in seen_messages[0]:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    user_texts.append(p.content)
    assert user_texts.count("turn1-user") == 1
    assert user_texts.count("turn2-user") == 1
    assert user_texts.count("turn3-user") == 1


@pytest.mark.asyncio
async def test_stream_emits_canonical_vercel_ai_events() -> None:
    repo = InMemoryThreadRepository()
    thread = await repo.create(agent="conversation", metadata={})
    agent: Agent[None, str] = Agent(
        TestModel(custom_output_text="hello"), output_type=str,
    )
    app = _build_app(repo, agent)

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="hi"),
            headers={"Accept": "text/event-stream"},
        )
        body = r.text

    types_seen: list[str] = []
    for line in body.splitlines():
        if line.startswith("data:"):
            raw = line[len("data:"):].strip()
            if raw == "[DONE]":
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict) and "type" in payload:
                types_seen.append(payload["type"])
    assert "start" in types_seen
    assert "finish" in types_seen


@pytest.mark.asyncio
async def test_deps_factory_invoked_per_request() -> None:
    from dataclasses import dataclass

    @dataclass
    class MyDeps:
        label: str

    repo = InMemoryThreadRepository()
    thread = await repo.create(agent="conversation", metadata={})
    agent: Agent[MyDeps, str] = Agent(
        TestModel(custom_output_text="ok"), output_type=str, deps_type=MyDeps,
    )
    received: list[MyDeps] = []

    @agent.tool_plain
    def stash() -> str:
        return "noop"

    async def deps_factory(*, thread_id: UUID, **_kw: Any) -> MyDeps:
        deps = MyDeps(label=f"req:{thread_id}")
        received.append(deps)
        return deps

    app = _build_app(repo, agent, deps_factory=deps_factory)
    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="hi"),
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200
        _ = r.text

    assert len(received) == 1
    assert received[0].label == f"req:{thread.id}"


@pytest.mark.asyncio
async def test_model_settings_flow_through() -> None:
    """``model_settings`` reach the agent run — inspected via FunctionModel."""
    from pydantic_ai.settings import ModelSettings

    repo = InMemoryThreadRepository()
    thread = await repo.create(agent="conversation", metadata={})
    settings = ModelSettings(temperature=0.42)
    seen_settings: list[ModelSettings | None] = []

    async def capture(
        _messages: list[Any], info: AgentInfo,
    ) -> ModelResponse:
        seen_settings.append(info.model_settings)
        return ModelResponse(parts=[TextPart(content="ok")])

    async def capture_stream(
        _messages: list[Any], info: AgentInfo,
    ) -> AsyncIterator[str]:
        seen_settings.append(info.model_settings)
        yield "ok"

    agent: Agent[None, str] = Agent(
        FunctionModel(capture, stream_function=capture_stream), output_type=str,
    )
    app = _build_app(repo, agent, model_settings=settings)

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="hi"),
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200
        _ = r.text

    assert len(seen_settings) == 1
    observed = seen_settings[0]
    assert observed is not None
    assert observed.get("temperature") == 0.42


@pytest.mark.asyncio
async def test_regenerate_truncates_old_assistant_then_emits_new() -> None:
    """Regenerate = body has user but no trailing assistant. The body-vs-DB
    sync drops the stale assistant row; the agent run then persists a
    fresh one. Flat list: edit / regenerate collapse to truncate-then-append.
    """
    repo = InMemoryThreadRepository()
    thread = await repo.create(agent="conversation", metadata={})
    user_msg = await repo.add_message(
        thread.id, role="user", id="u1",
        parts=[{"type": "text", "text": "hi"}],
    )
    asst_v1 = await repo.add_message(
        thread.id, role="assistant", id="a1",
        parts=[{"type": "text", "text": "v1"}],
    )

    agent: Agent[None, str] = Agent(
        TestModel(custom_output_text="v2"), output_type=str,
    )
    app = _build_app(repo, agent)

    body: dict[str, Any] = {
        "trigger": "regenerate-message",
        "id": str(thread.id),
        "messageId": str(asst_v1.id),
        "messages": [
            {
                "id": user_msg.id, "role": "user",
                "parts": [{"type": "text", "text": "hi", "state": "done"}],
            },
        ],
    }
    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=body,
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200, r.text
        _ = r.text

    history = await repo.history(thread.id)
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[0].id == user_msg.id  # user preserved
    assert history[1].id != asst_v1.id  # stale assistant dropped, new minted


@pytest.mark.asyncio
async def test_approval_response_keeps_tool_call_in_adapter_messages() -> None:
    """Approval responses (Vercel SDK v6 ``tool-*`` parts with approval
    decision) require the originating assistant turn — with its
    ``tool-call`` part — to survive the message-trim step."""
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter

    from pydantic_ai_stateflow.api.streaming.router import (
        _trim_adapter_messages_to_last_user_prompt,
    )

    repo = InMemoryThreadRepository()
    thread = await repo.create(agent="conversation", metadata={})
    agent: Agent[None, str] = Agent(TestModel(), output_type=str)

    body = {
        "trigger": "submit-message",
        "id": str(thread.id),
        "messages": [
            {
                "id": str(uuid4()), "role": "user",
                "parts": [{"type": "text", "text": "delete x", "state": "done"}],
            },
            {
                "id": str(uuid4()), "role": "assistant",
                "parts": [
                    {
                        "type": "tool-delete_note", "toolCallId": "call_1",
                        "state": "approval-responded",
                        "input": {"note_id": "abc"},
                        "approval": {
                            "id": "appr_1", "approved": False, "reason": "no",
                        },
                    },
                ],
            },
        ],
    }

    run_input = VercelAIAdapter.build_run_input(json.dumps(body).encode())
    adapter = VercelAIAdapter(agent=agent, run_input=run_input, sdk_version=6)

    assert adapter.deferred_tool_results is not None

    msgs_before = list(adapter.messages)
    if adapter.deferred_tool_results is None:
        _trim_adapter_messages_to_last_user_prompt(adapter)
    msgs_after = list(adapter.messages)

    assert msgs_before == msgs_after, (
        "trim must not run when deferred_tool_results is present"
    )


@pytest.mark.asyncio
async def test_pii_guard_redacts_email_in_live_sse_stream() -> None:
    """End-to-end: an Agent wired with ``PIIGuard`` must scrub PII from
    the SSE body BEFORE the client sees it, not just from the persisted
    assistant message.

    This pins the user-facing bug: the assistant-ui frontend renders raw
    bytes off the live stream and never re-syncs from persistence on
    stream end, so leaking PII via SSE deltas is a real user-visible
    leak even when the persisted row is clean.
    """
    import re as _re

    from pydantic_ai_stateflow.capabilities import PIIGuard, RegexDetector

    email_re = _re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    leak = "Contact alice@example.com to follow up."

    def fn(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=leak)])

    async def fn_stream(
        _messages: list[Any], _info: AgentInfo,
    ) -> AsyncIterator[str]:
        # Split deliberately so the "@" arrives in a later chunk — the
        # classic split-across-deltas case PIIGuard.wrap_run_event_stream
        # must cover.
        yield "Contact alice"
        yield "@example.com to follow up."

    repo = InMemoryThreadRepository()
    thread = await repo.create(agent="conversation", metadata={})
    agent: Agent[None, str] = Agent(
        FunctionModel(fn, stream_function=fn_stream),
        output_type=str,
        capabilities=[PIIGuard(detector=RegexDetector(patterns=[email_re]))],
    )
    app = _build_app(repo, agent)

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="who?"),
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200, r.text
        body = r.text

    # The user-facing assertion: the raw email must NOT have reached
    # the SSE consumer.
    assert "alice@example.com" not in body, body
    assert "[REDACTED]" in body, body

    # Persistence path should also be clean (after_model_request handles
    # the non-streaming reconstruction the on_complete callback uses).
    msgs = await repo.history(thread.id)
    asst = next(m for m in msgs if m.role == "assistant")
    persisted_text = json.dumps(asst.parts)
    assert "alice@example.com" not in persisted_text, persisted_text
