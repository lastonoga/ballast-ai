"""Smoke test for the notes-app backend (using sf.testing.TestEngine).

These tests share the module-level ``app`` from ``notes_app.main`` —
isolated-state per-test would require constructing a fresh ``app``
(the singletons are module-level), but for these smoke tests sharing
is OK; we either:

  - inspect already-attached app.state (read-only test), or
  - swap the agent registered as "notes" with a MockAgent and assert
    on the response (mutates app.state.agents but restores after).
"""
from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

import pydantic_ai_stateflow as sf
import pytest
from fastapi.testclient import TestClient

from notes_app.main import app, notes_repo


def _ag_ui_body(thread_id: str, user_text: str) -> dict[str, Any]:
    return {
        "trigger": "submit-message",
        "id": thread_id,
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


def _parse_sse_types(body: str) -> list[str]:
    types: list[str] = []
    for line in body.splitlines():
        if line.startswith("data:"):
            raw = line[len("data:"):].strip()
            if raw == "[DONE]":
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict) and "type" in payload:
                types.append(payload["type"])
    return types


def test_notes_repo_attached_to_app_state() -> None:
    """The module-level ``notes_repo`` is published on ``app.state``."""
    from notes_app.notes.repository import InMemoryNoteRepository

    with TestClient(app):
        assert isinstance(app.state.notes_repo, InMemoryNoteRepository)
        assert app.state.notes_repo is notes_repo


def test_threads_crud_and_streaming_fake() -> None:
    """End-to-end with a TestModel-backed agent override (no network).

    Swaps the registered ``"notes"`` agent with a ``MockAgent`` for the
    duration of the test, then restores it. This exercises the
    non-durable streaming path (MockAgent is not a StateflowDurableAgent).
    """
    from notes_app.agent import NotesAgent

    mock_agent = sf.testing.MockAgent.with_output("Hello, world!")

    saved_agent = app.state.agents[NotesAgent.name]
    app.state.agents[NotesAgent.name] = mock_agent
    try:
        with TestClient(app) as client:
            r = client.post("/threads", json={})
            assert r.status_code == 201, r.text
            thread = r.json()
            thread_id = thread["id"]
            assert thread["agent"] == NotesAgent.name

            r = client.get(f"/threads/{thread_id}")
            assert r.status_code == 200, r.text
            assert r.json()["id"] == thread_id

            r = client.post(
                f"/threads/{thread_id}/messages",
                headers={"Accept": "text/event-stream"},
                json=_ag_ui_body(thread_id, "hi"),
            )
            assert r.status_code == 200, r.text
            assert r.headers["content-type"].startswith("text/event-stream")

            kinds = _parse_sse_types(r.text)
            assert "start" in kinds, kinds
            assert "finish" in kinds, kinds
    finally:
        app.state.agents[NotesAgent.name] = saved_agent


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="no OPENROUTER_API_KEY — skipping live OpenRouter smoke",
)
def test_streaming_live_openrouter() -> None:  # pragma: no cover — network
    with TestClient(app) as client:
        r = client.post("/threads", json={})
        assert r.status_code == 201
        thread_id = r.json()["id"]

        r = client.post(
            f"/threads/{thread_id}/messages",
            headers={"Accept": "text/event-stream"},
            json=_ag_ui_body(thread_id, "Reply with the single word: pong"),
        )
        assert r.status_code == 200
        kinds = _parse_sse_types(r.text)
        assert kinds, "expected at least one SSE event"
        assert "finish" in kinds or "error" in kinds, kinds


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="no OPENROUTER_API_KEY — skipping live notes-tool smoke",
)
def test_live_create_note_tool_call() -> None:  # pragma: no cover — network
    with TestClient(app) as client:
        r = client.post("/threads", json={})
        assert r.status_code == 201
        thread_id = r.json()["id"]

        r = client.post(
            f"/threads/{thread_id}/messages",
            headers={"Accept": "text/event-stream"},
            json=_ag_ui_body(
                thread_id,
                "Please create a note titled 'Grocery list' "
                "with body 'milk, eggs, bread'.",
            ),
        )
        assert r.status_code == 200
        kinds = _parse_sse_types(r.text)
        assert "finish" in kinds or "error" in kinds, kinds

        import asyncio

        notes = asyncio.get_event_loop().run_until_complete(
            notes_repo.list_(),
        )
        assert notes, f"expected at least one note saved; got events={kinds}"
        assert any(
            "grocery" in n.title.lower() or "grocery" in n.body.lower()
            for n in notes
        ), f"no grocery note in {[n.title for n in notes]}"
