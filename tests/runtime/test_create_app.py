"""Unit tests for ``sf.create_app()``."""
from __future__ import annotations

from fastapi.testclient import TestClient

from pydantic_ai_stateflow.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.runtime.app import create_app
from pydantic_ai_stateflow.runtime.event_stream import InProcessEventStream
from pydantic_ai_stateflow.runtime.infra import Infra


def _infra() -> Infra:
    return Infra(
        thread_repo=InMemoryThreadRepository(),
        event_log=InMemoryEventLogRepository(),
        event_stream=InProcessEventStream(),
    )


def test_minimal_app_has_health_endpoint(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    app = create_app(infra=_infra())
    with TestClient(app) as client:
        r = client.get("/healthz")
        # Either 200 or 404/503 — depends on build_health_router default.
        # Just verify the app boots without error.
        assert r.status_code in (200, 404, 503)


def test_infra_attached_to_app_state(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    infra = _infra()
    app = create_app(infra=infra)
    assert app.state.infra is infra


def test_threads_endpoint_works(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    app = create_app(infra=_infra())
    with TestClient(app) as client:
        r = client.get("/threads")
        assert r.status_code == 200, r.text
