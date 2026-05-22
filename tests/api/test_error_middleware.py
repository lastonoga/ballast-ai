"""Unit tests for ``install_error_handlers`` + middleware behavior."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.api.error_middleware import (
    PROBLEM_JSON,
    install_error_handlers,
)
from ballast.errors import BallastError, ThreadNotFound
from ballast.settings import reset_settings


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    # Reset settings to defaults between tests so env-vars don't leak.
    reset_settings()
    yield
    reset_settings()


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/raise")
    def _raise() -> None:
        raise ThreadNotFound(
            "thread 7f3b not found",
            hint="Confirm the id",
            context={"thread_id": "7f3b"},
        )

    @app.get("/stateflow_generic")
    def _generic() -> None:
        raise BallastError("oops", hint="fix it")

    install_error_handlers(app)
    return app


def test_handler_renders_problem_json() -> None:
    app = _make_app()
    with TestClient(app) as client:
        r = client.get("/raise")
        assert r.status_code == 404
        assert r.headers["content-type"].startswith(PROBLEM_JSON)
        body = r.json()
        assert body["error"]["code"] == "BALLAST_PERSISTENCE_THREAD_NOT_FOUND"
        assert body["error"]["detail"] == "thread 7f3b not found"
        assert body["error"]["hint"] == "Confirm the id"
        assert body["error"]["context"] == {"thread_id": "7f3b"}


def test_generic_stateflow_error_500() -> None:
    app = _make_app()
    with TestClient(app) as client:
        r = client.get("/stateflow_generic")
        assert r.status_code == 500


def test_traceback_hidden_by_default(monkeypatch) -> None:
    # Default environment is "dev" which auto-exposes tracebacks; use "production"
    # to verify that non-dev environments suppress tracebacks.
    monkeypatch.setenv("BALLAST_OBSERVABILITY__ENVIRONMENT", "production")
    reset_settings()
    app = _make_app()
    with TestClient(app) as client:
        r = client.get("/raise")
        assert "traceback" not in r.json()["error"]


def test_traceback_when_dev(monkeypatch) -> None:
    monkeypatch.setenv("BALLAST_OBSERVABILITY__ENVIRONMENT", "dev")
    reset_settings()
    app = _make_app()
    with TestClient(app) as client:
        r = client.get("/raise")
        # In dev env, tri-state default → True.
        assert "traceback" in r.json()["error"]


def test_traceback_explicit_off_in_dev(monkeypatch) -> None:
    monkeypatch.setenv("BALLAST_OBSERVABILITY__ENVIRONMENT", "dev")
    monkeypatch.setenv("BALLAST_API__EXPOSE_TRACEBACKS", "false")
    reset_settings()
    app = _make_app()
    with TestClient(app) as client:
        r = client.get("/raise")
        assert "traceback" not in r.json()["error"]


def test_install_idempotent() -> None:
    app = _make_app()
    install_error_handlers(app)
    install_error_handlers(app)
    # No double-install crash. Verified by the fact that the test completes.


def test_install_skipped_when_setting_false(monkeypatch) -> None:
    monkeypatch.setenv("BALLAST_API__INSTALL_ERROR_MIDDLEWARE", "false")
    reset_settings()
    app = FastAPI()
    install_error_handlers(app)
    # The flag should be unset because install_error_handlers short-circuited.
    assert not getattr(app.state, "_stateflow_error_handlers_installed", False)
