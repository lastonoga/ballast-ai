"""Smoke test for the iteration-2 backend (post 2.1 framework migration).

We exercise the full surface via FastAPI's `TestClient`:

  1. `POST /threads` → 201
  2. `GET  /threads/{id}` → 200
  3. `POST /threads/{id}/messages` → SSE with canonical AG-UI events:
     `RUN_STARTED → TEXT_MESSAGE_CONTENT × N → RUN_FINISHED`.

The streaming test uses a deterministic fake `AgentRunner` so we don't hit
OpenRouter in CI. A second test exercises the real OpenRouter path but is
skipped when `OPENROUTER_API_KEY` is absent.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic_ai_stateflow.api.streaming import StreamEvent
from pydantic_ai_stateflow.api.streaming.router import _PostMessageBody

from notes_app.main import build_app


async def _fake_runner(
    *,
    thread_id: UUID,
    run_id: UUID,
    message: _PostMessageBody,
    tenant_id: UUID,
) -> AsyncIterator[StreamEvent]:
    del message, tenant_id
    msg_id = uuid4()
    yield StreamEvent.run_started(thread_id=thread_id, run_id=run_id)
    yield StreamEvent.text_message_start(message_id=msg_id)
    yield StreamEvent.text_message_content(message_id=msg_id, delta="Hello")
    yield StreamEvent.text_message_content(message_id=msg_id, delta=", world!")
    yield StreamEvent.text_message_end(message_id=msg_id)
    yield StreamEvent.run_finished(thread_id=thread_id, run_id=run_id)


def _parse_sse(body: str) -> list[tuple[str, str]]:
    """Crude SSE parser: returns ordered (event, data) pairs."""
    out: list[tuple[str, str]] = []
    event: str | None = None
    data: str | None = None
    for line in body.splitlines():
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data = line[len("data: "):]
        elif line == "" and event is not None and data is not None:
            out.append((event, data))
            event, data = None, None
    return out


def test_threads_crud_and_streaming_fake() -> None:
    """End-to-end with a fake runner — no network."""
    app = build_app(agent_runner=_fake_runner)
    tenant_id = str(uuid4())

    with TestClient(app) as client:
        # 1) Create
        r = client.post(
            "/threads",
            headers={"X-Tenant-Id": tenant_id},
            json={"purpose": "chat", "actor_id": "alice"},
        )
        assert r.status_code == 201, r.text
        thread = r.json()
        thread_id = thread["id"]

        # 2) Get
        r = client.get(
            f"/threads/{thread_id}", headers={"X-Tenant-Id": tenant_id},
        )
        assert r.status_code == 200, r.text
        assert r.json()["id"] == thread_id

        # 3) Stream
        r = client.post(
            f"/threads/{thread_id}/messages",
            headers={"X-Tenant-Id": tenant_id},
            json={
                "role": "user",
                "parts": [{"type": "text", "text": "hi"}],
            },
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(r.text)
        kinds = [k for k, _ in events]
        assert "RUN_STARTED" in kinds, f"missing RUN_STARTED in {kinds}"
        assert "TEXT_MESSAGE_CONTENT" in kinds, (
            f"missing TEXT_MESSAGE_CONTENT in {kinds}"
        )
        assert kinds[-1] == "RUN_FINISHED", (
            f"last event must be RUN_FINISHED; got {kinds}"
        )


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="no OPENROUTER_API_KEY — skipping live OpenRouter smoke",
)
def test_streaming_live_openrouter() -> None:  # pragma: no cover — network
    """Live smoke against OpenRouter — only runs when key is present."""
    app = build_app()  # default = lazy real runner
    tenant_id = str(uuid4())

    with TestClient(app) as client:
        r = client.post(
            "/threads",
            headers={"X-Tenant-Id": tenant_id},
            json={"purpose": "chat", "actor_id": "alice"},
        )
        assert r.status_code == 201
        thread_id = r.json()["id"]

        r = client.post(
            f"/threads/{thread_id}/messages",
            headers={"X-Tenant-Id": tenant_id},
            json={
                "role": "user",
                "parts": [{"type": "text", "text": "Reply with the single word: pong"}],
            },
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)
        kinds = [k for k, _ in events]
        assert kinds, "expected at least one SSE event"
        assert kinds[-1] in {"RUN_FINISHED", "RUN_ERROR"}, f"got {kinds}"
