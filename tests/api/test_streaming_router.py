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

from pydantic_ai_stateflow.api.streaming import build_streaming_router
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)


def _ag_ui_body(*, thread_id: UUID, user_text: str) -> dict[str, Any]:
    """Build a minimal Vercel-AI ``SubmitMessage`` body.

    Name kept for diff churn; format is Vercel AI SDK v6.
    """
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
    **kwargs: Any,
) -> FastAPI:
    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent=agent, **kwargs),
    )
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_returns_404_when_thread_missing() -> None:
    repo = InMemoryThreadRepository()
    agent: Agent[None, str] = Agent(TestModel(), output_type=str)
    app = _build_app(repo, agent)

    tenant_id = uuid4()
    missing = uuid4()
    async with _client(app) as c:
        r = await c.post(
            f"/threads/{missing}/messages",
            json=_ag_ui_body(thread_id=missing, user_text="hi"),
            headers={
                "X-Tenant-Id": str(tenant_id),
                "Accept": "text/event-stream",
            },
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_persists_user_message_before_stream() -> None:
    repo = InMemoryThreadRepository()
    tenant_id = uuid4()
    thread = await repo.create(
        purpose="conversation",
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    agent: Agent[None, str] = Agent(TestModel(custom_output_text="ok"), output_type=str)
    app = _build_app(repo, agent)

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="hello world"),
            headers={
                "X-Tenant-Id": str(tenant_id),
                "Accept": "text/event-stream",
            },
        )
        assert r.status_code == 200
        _ = r.text

    msgs = await repo.history(thread.id, tenant_id=tenant_id)
    roles = [m.role for m in msgs]
    assert "user" in roles
    user_row = next(m for m in msgs if m.role == "user")
    assert user_row.parts == [{"type": "text", "text": "hello world"}]


@pytest.mark.asyncio
async def test_assistant_reply_persisted_via_on_complete() -> None:
    repo = InMemoryThreadRepository()
    tenant_id = uuid4()
    thread = await repo.create(
        purpose="conversation",
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    agent: Agent[None, str] = Agent(
        TestModel(custom_output_text="hi there"), output_type=str,
    )
    app = _build_app(repo, agent)

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="hi"),
            headers={
                "X-Tenant-Id": str(tenant_id),
                "Accept": "text/event-stream",
            },
        )
        assert r.status_code == 200
        _ = r.text

    msgs = await repo.history(thread.id, tenant_id=tenant_id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].parts == [{"type": "text", "text": "hi there"}]


@pytest.mark.asyncio
async def test_message_history_reconstructed_from_repo() -> None:
    """Seed 3 prior turns; verify the agent sees them via
    ``last_model_request_parameters.messages``."""
    repo = InMemoryThreadRepository()
    tenant_id = uuid4()
    thread = await repo.create(
        purpose="conversation",
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    # Seed three prior turns directly into the repo.
    await repo.add_message(
        thread.id,
        role="user",
        parts=[{"type": "text", "text": "turn1-user"}],
        tenant_id=tenant_id,
    )
    await repo.add_message(
        thread.id,
        role="assistant",
        parts=[{"type": "text", "text": "turn1-assistant"}],
        tenant_id=tenant_id,
    )
    await repo.add_message(
        thread.id,
        role="user",
        parts=[{"type": "text", "text": "turn2-user"}],
        tenant_id=tenant_id,
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

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="turn3-user"),
            headers={
                "X-Tenant-Id": str(tenant_id),
                "Accept": "text/event-stream",
            },
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
    # All three prior user turns + the current "turn3-user" should be visible
    # exactly once each (no duplicate of the current user turn from repo).
    assert user_texts.count("turn1-user") == 1
    assert user_texts.count("turn2-user") == 1
    assert user_texts.count("turn3-user") == 1


@pytest.mark.asyncio
async def test_stream_emits_canonical_vercel_ai_events() -> None:
    repo = InMemoryThreadRepository()
    tenant_id = uuid4()
    thread = await repo.create(
        purpose="conversation",
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    agent: Agent[None, str] = Agent(
        TestModel(custom_output_text="hello"), output_type=str,
    )
    app = _build_app(repo, agent)

    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="hi"),
            headers={
                "X-Tenant-Id": str(tenant_id),
                "Accept": "text/event-stream",
            },
        )
        body = r.text

    # Vercel AI SDK SSE: lines like `data: {"type":"start",...}`.
    types_seen: list[str] = []
    for line in body.splitlines():
        if line.startswith("data:"):
            raw = line[len("data:"):].strip()
            if raw == "[DONE]":
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict) and "type" in payload:
                types_seen.append(payload["type"])
    # A successful Vercel AI stream brackets the response with `start`
    # and `finish` chunks (lifecycle events). Don't pin specific text-*
    # event names — they depend on TestModel internals.
    assert "start" in types_seen
    assert "finish" in types_seen


@pytest.mark.asyncio
async def test_deps_factory_invoked_per_request() -> None:
    from dataclasses import dataclass

    @dataclass
    class MyDeps:
        tenant_id: UUID
        label: str

    repo = InMemoryThreadRepository()
    tenant_id = uuid4()
    thread = await repo.create(
        purpose="conversation",
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
    agent: Agent[MyDeps, str] = Agent(
        TestModel(custom_output_text="ok"), output_type=str, deps_type=MyDeps,
    )
    received: list[MyDeps] = []

    @agent.tool_plain
    def stash() -> str:
        return "noop"

    async def deps_factory(*, thread_id: UUID, tenant_id: UUID, **_kw: Any) -> MyDeps:
        deps = MyDeps(tenant_id=tenant_id, label=f"req:{thread_id}")
        received.append(deps)
        return deps

    app = _build_app(repo, agent, deps_factory=deps_factory)
    async with _client(app) as c:
        r = await c.post(
            f"/threads/{thread.id}/messages",
            json=_ag_ui_body(thread_id=thread.id, user_text="hi"),
            headers={
                "X-Tenant-Id": str(tenant_id),
                "Accept": "text/event-stream",
            },
        )
        assert r.status_code == 200
        _ = r.text

    assert len(received) == 1
    assert received[0].tenant_id == tenant_id
    assert received[0].label == f"req:{thread.id}"


@pytest.mark.asyncio
async def test_model_settings_flow_through() -> None:
    """``model_settings`` reach the agent run — inspected via FunctionModel."""
    from pydantic_ai.settings import ModelSettings

    repo = InMemoryThreadRepository()
    tenant_id = uuid4()
    thread = await repo.create(
        purpose="conversation",
        purpose_metadata={},
        actor_id="a",
        tenant_id=tenant_id,
    )
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
            headers={
                "X-Tenant-Id": str(tenant_id),
                "Accept": "text/event-stream",
            },
        )
        assert r.status_code == 200
        _ = r.text

    assert len(seen_settings) == 1
    observed = seen_settings[0]
    assert observed is not None
    assert observed.get("temperature") == 0.42
