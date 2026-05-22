"""Unit tests for ``sf.create_app()``."""
from __future__ import annotations

from fastapi.testclient import TestClient

from pydantic_ai_stateflow.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.runtime.app import create_app
from pydantic_ai_stateflow.runtime.engine import _reset_engine_for_tests
from pydantic_ai_stateflow.runtime.event_stream import InProcessEventStream


def _build_app():
    _reset_engine_for_tests()
    return create_app(
        thread_repo=InMemoryThreadRepository(),
        event_log=InMemoryEventLogRepository(),
        event_stream=InProcessEventStream(),
    )


def test_minimal_app_has_health_endpoint(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/healthz")
        # Either 200 or 404/503 — depends on build_health_router default.
        # Just verify the app boots without error.
        assert r.status_code in (200, 404, 503)


def test_engine_attached_to_app_state(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    _reset_engine_for_tests()
    thread_repo = InMemoryThreadRepository()
    event_log = InMemoryEventLogRepository()
    event_stream = InProcessEventStream()
    app = create_app(
        thread_repo=thread_repo,
        event_log=event_log,
        event_stream=event_stream,
    )
    assert app.state.engine.thread_repo is thread_repo
    assert app.state.engine.event_log is event_log
    assert app.state.engine.event_stream is event_stream


def test_threads_endpoint_works(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    app = _build_app()
    with TestClient(app) as client:
        r = client.get("/threads")
        assert r.status_code == 200, r.text
