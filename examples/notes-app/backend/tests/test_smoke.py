"""Smoke test for the notes-app backend."""

from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from notes_app.agent import NotesAgent, NoteToolDeps
from notes_app.main import build_app
from notes_app.notes.repository import InMemoryNoteRepository


class _FakeNotesAgent(NotesAgent):
    """``NotesAgent`` variant whose ``build_agent`` returns a TestModel agent."""

    def __init__(
        self,
        *,
        notes_repo: InMemoryNoteRepository,
        with_tools: bool = False,
    ) -> None:
        super().__init__(notes_repo=notes_repo)
        self._with_tools = with_tools

    def build_agent(self) -> Agent[NoteToolDeps, str]:
        return Agent(
            TestModel(custom_output_text="Hello, world!"),
            output_type=str,
            deps_type=NoteToolDeps,
        )

    @property  # type: ignore[override]
    def agent(self) -> Agent[NoteToolDeps, str]:
        cache_key = "_test_agent"
        cached = self.__dict__.get(cache_key)
        if cached is not None:
            return cached
        if self._with_tools:
            built = super().agent
        else:
            built = self.build_agent()
        self.__dict__[cache_key] = built
        return built

    def model_settings(self) -> None:
        return None


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


def test_note_repository_is_bound_in_container() -> None:
    from notes_app.notes.repository import NoteRepository

    notes_repo = InMemoryNoteRepository()
    app = build_app(
        notes_repo=notes_repo,
        notes_agent=_FakeNotesAgent(notes_repo=notes_repo),
    )
    with TestClient(app):
        assert app.state.container.has(NoteRepository)
        assert app.state.container.get(NoteRepository) is notes_repo


def test_threads_crud_and_streaming_fake() -> None:
    """End-to-end with a TestModel-backed agent — no network."""
    notes_repo = InMemoryNoteRepository()
    app = build_app(
        notes_repo=notes_repo,
        notes_agent=_FakeNotesAgent(notes_repo=notes_repo),
    )

    with TestClient(app) as client:
        r = client.post("/threads", json={})
        assert r.status_code == 201, r.text
        thread = r.json()
        thread_id = thread["id"]
        assert thread["agent"] == "notes"

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


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="no OPENROUTER_API_KEY — skipping live OpenRouter smoke",
)
def test_streaming_live_openrouter() -> None:  # pragma: no cover — network
    app = build_app()

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
    notes_repo = InMemoryNoteRepository()
    app = build_app(notes_repo=notes_repo)

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
