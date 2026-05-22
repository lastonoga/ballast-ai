"""Smoke tests for ``Ballast(settings).use(*providers).fastapi(...)``."""
from __future__ import annotations

from fastapi.testclient import TestClient

import ballast
from ballast.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from ballast.providers import EventsProvider, ThreadsProvider
from ballast.runtime.engine import _reset_engine_for_tests
from ballast.runtime.event_stream import InProcessEventStream
from ballast.settings import BallastSettings


def _build_ballast_app(fresh_dbos_executor: None) -> ballast.Ballast:
    del fresh_dbos_executor
    _reset_engine_for_tests()
    thread_repo = InMemoryThreadRepository()
    event_log = InMemoryEventLogRepository()
    event_stream = InProcessEventStream()
    return (
        ballast.Ballast(BallastSettings())
        .use(
            ThreadsProvider(thread_repo),
            EventsProvider(event_log, event_stream),
        )
    )


def test_ballast_fastapi_app_boots(fresh_dbos_executor: None) -> None:
    app = _build_ballast_app(fresh_dbos_executor).fastapi()
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code in (200, 404, 503)


def test_ballast_providers_propagate_to_engine(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    _reset_engine_for_tests()
    thread_repo = InMemoryThreadRepository()
    event_log = InMemoryEventLogRepository()
    event_stream = InProcessEventStream()
    app = (
        ballast.Ballast(BallastSettings())
        .use(
            ThreadsProvider(thread_repo),
            EventsProvider(event_log, event_stream),
        )
        .fastapi()
    )
    assert app.state.engine.thread_repo is thread_repo
    assert app.state.engine.event_log is event_log
    assert app.state.engine.event_stream is event_stream


def test_ballast_threads_endpoint_works(fresh_dbos_executor: None) -> None:
    app = _build_ballast_app(fresh_dbos_executor).fastapi()
    with TestClient(app) as client:
        r = client.get("/threads")
        assert r.status_code == 200, r.text


def test_ballast_cors_dev_shortcut(fresh_dbos_executor: None) -> None:
    app = _build_ballast_app(fresh_dbos_executor).fastapi(cors="dev")
    # Issue a preflight to verify CORS middleware mounted.
    with TestClient(app) as client:
        r = client.options(
            "/threads",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in {k.lower() for k in r.headers}


def test_ballast_cors_unknown_shortcut_raises(fresh_dbos_executor: None) -> None:
    import pytest

    app_builder = _build_ballast_app(fresh_dbos_executor)
    with pytest.raises(ValueError, match="Unknown cors shortcut"):
        app_builder.fastapi(cors="bogus")
