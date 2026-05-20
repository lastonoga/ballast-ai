"""Smoke test for the notes-app backend.

We exercise the full surface via FastAPI's ``TestClient``:

  1. ``POST /threads`` → 201
  2. ``GET  /threads/{id}`` → 200
  3. ``POST /threads/{id}/messages`` with an AG-UI ``RunAgentInput``
     body → SSE with canonical AG-UI events
     (``RUN_STARTED → ... → RUN_FINISHED``).

The fast test wires the app with a ``TestModel``-backed pydantic-ai
``Agent`` so we don't hit OpenRouter in CI. Two further tests exercise
the real OpenRouter path but are skipped when ``OPENROUTER_API_KEY`` is
absent.
"""

from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from notes_app.main import build_app
from notes_app.notes.repository import InMemoryNoteRepository
from notes_app.notes.tools import NoteToolDeps, register_note_tools


def _fake_agent(
    notes_repo: InMemoryNoteRepository | None = None,
    *,
    with_tools: bool = False,
) -> Agent[NoteToolDeps, str]:
    """Build an in-memory agent over ``TestModel`` for CI smoke.

    ``with_tools=False`` (default) skips tool registration so ``TestModel``
    just emits a plain text reply (it would otherwise auto-call every
    registered tool, which buries the lifecycle ``finish`` chunk behind a
    fanout of tool-call events the basic smoke check doesn't need).
    """
    del notes_repo  # repo is bound via the deps factory in build_app
    agent: Agent[NoteToolDeps, str] = Agent(
        TestModel(custom_output_text="Hello, world!"),
        output_type=str,
        deps_type=NoteToolDeps,
    )
    if with_tools:
        register_note_tools(agent)
    return agent


def _ag_ui_body(thread_id: str, user_text: str) -> dict[str, Any]:
    """Build a Vercel AI ``SubmitMessage`` body.

    Name kept for diff churn; format is Vercel AI SDK v6.
    """
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
    """Parse Vercel AI SSE ``data: {"type": "..."}`` frames into type tokens."""
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
    """The notes repo must be reachable via ``app.state.container``
    so the deps factory (and any future router) can resolve it without
    a module-level singleton.
    """
    from notes_app.notes.repository import NoteRepository

    notes_repo = InMemoryNoteRepository()
    app = build_app(notes_repo=notes_repo, agent=_fake_agent(notes_repo))
    with TestClient(app):
        assert app.state.container.has(NoteRepository)
        assert app.state.container.get(NoteRepository) is notes_repo


def test_threads_crud_and_streaming_fake() -> None:
    """End-to-end with a TestModel-backed agent — no network."""
    notes_repo = InMemoryNoteRepository()
    app = build_app(notes_repo=notes_repo, agent=_fake_agent(notes_repo))
    tenant_id = str(uuid4())

    with TestClient(app) as client:
        # 1) Create via the notes-app's own POST /threads endpoint.
        r = client.post(
            "/threads",
            headers={"X-Tenant-Id": tenant_id},
            json={"actor_id": "alice"},
        )
        assert r.status_code == 201, r.text
        thread = r.json()
        thread_id = thread["id"]
        assert thread["agent"] == "notes"

        # 2) Get
        r = client.get(
            f"/threads/{thread_id}", headers={"X-Tenant-Id": tenant_id},
        )
        assert r.status_code == 200, r.text
        assert r.json()["id"] == thread_id

        # 3) Stream — native Vercel AI ``SubmitMessage`` body
        r = client.post(
            f"/threads/{thread_id}/messages",
            headers={
                "X-Tenant-Id": tenant_id,
                "Accept": "text/event-stream",
            },
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
    """Live smoke against OpenRouter — only runs when key is present."""
    app = build_app()  # default = lazy real agent
    tenant_id = str(uuid4())

    with TestClient(app) as client:
        r = client.post(
            "/threads",
            headers={"X-Tenant-Id": tenant_id},
            json={"actor_id": "alice"},
        )
        assert r.status_code == 201
        thread_id = r.json()["id"]

        r = client.post(
            f"/threads/{thread_id}/messages",
            headers={
                "X-Tenant-Id": tenant_id,
                "Accept": "text/event-stream",
            },
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
    """Ask the LLM to create a note; assert the in-memory repo has it."""
    notes_repo = InMemoryNoteRepository()
    app = build_app(notes_repo=notes_repo)
    tenant_id = uuid4()

    with TestClient(app) as client:
        r = client.post(
            "/threads",
            headers={"X-Tenant-Id": str(tenant_id)},
            json={"actor_id": "alice"},
        )
        assert r.status_code == 201
        thread_id = r.json()["id"]

        r = client.post(
            f"/threads/{thread_id}/messages",
            headers={
                "X-Tenant-Id": str(tenant_id),
                "Accept": "text/event-stream",
            },
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
            notes_repo.list_(tenant_id=tenant_id),
        )
        assert notes, f"expected at least one note saved; got events={kinds}"
        assert any(
            "grocery" in n.title.lower() or "grocery" in n.body.lower()
            for n in notes
        ), f"no grocery note in {[n.title for n in notes]}"
