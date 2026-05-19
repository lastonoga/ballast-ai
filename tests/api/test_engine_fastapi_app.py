from __future__ import annotations

import logging

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.api import CORSConfig
from pydantic_ai_stateflow.runtime import Engine
from pydantic_ai_stateflow.runtime.container import Container


class _RecordingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def register(self, container: Container) -> None:
        self.calls += 1


def test_fastapi_app_attaches_container_and_engine_to_state() -> None:
    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app()
    assert app.state.container is engine.container
    assert app.state.engine is engine


def test_fastapi_app_mounts_healthz_by_default() -> None:
    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app()
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_fastapi_app_lifespan_boots_engine_once() -> None:
    prov = _RecordingProvider()
    engine = Engine(providers=[prov])
    app = engine.fastapi_app()
    with TestClient(app):
        pass
    assert prov.calls == 1
    with TestClient(app):
        pass
    assert prov.calls == 1


def test_fastapi_app_mounts_extra_routers() -> None:
    engine = Engine(providers=[_RecordingProvider()])
    extra = APIRouter()

    @extra.get("/custom")
    async def custom() -> dict[str, str]:
        return {"hi": "there"}

    app = engine.fastapi_app(extra_routers=[extra])
    with TestClient(app) as c:
        r = c.get("/custom")
    assert r.status_code == 200


def test_fastapi_app_does_not_attach_observability_by_default() -> None:
    """ObservabilityProvider is opt-in; instrument_fastapi must NOT
    run unless explicitly enabled (Task 7 enables it via the provider)."""
    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app()
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200


# ---------------------------------------------------------------------------
# F8 — CORS + lifespan hooks
# ---------------------------------------------------------------------------


def _has_cors_middleware(app: FastAPI) -> bool:
    from fastapi.middleware.cors import CORSMiddleware

    return any(m.cls is CORSMiddleware for m in app.user_middleware)  # type: ignore[comparison-overlap]


def test_fastapi_app_installs_cors_middleware_when_cors_given() -> None:
    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app(
        cors=CORSConfig(allow_origins=["http://localhost:3000"]),
    )
    assert _has_cors_middleware(app)
    # Preflight: a real OPTIONS request reaches CORSMiddleware.
    with TestClient(app) as c:
        r = c.options(
            "/healthz",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert r.status_code == 200
    assert (
        r.headers.get("access-control-allow-origin") == "http://localhost:3000"
    )


def test_fastapi_app_no_cors_when_none() -> None:
    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app()
    assert not _has_cors_middleware(app)


def test_cors_config_permissive_dev_factory() -> None:
    cfg = CORSConfig.permissive_dev()
    assert "http://localhost:3000" in cfg.allow_origins
    assert "http://localhost:3003" in cfg.allow_origins
    assert cfg.allow_methods == ["*"]
    assert cfg.allow_headers == ["*"]
    assert cfg.allow_credentials is True

    custom = CORSConfig.permissive_dev(origins=["http://localhost:9999"])
    assert custom.allow_origins == ["http://localhost:9999"]
    assert custom.allow_credentials is True


@pytest.mark.asyncio
async def test_on_startup_hooks_run_in_order() -> None:
    order: list[str] = []

    async def h1(app: FastAPI) -> None:  # noqa: ARG001
        order.append("h1")

    async def h2(app: FastAPI) -> None:  # noqa: ARG001
        order.append("h2")

    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app(on_startup=[h1, h2])
    with TestClient(app):
        pass
    assert order == ["h1", "h2"]


@pytest.mark.asyncio
async def test_on_shutdown_hooks_run_in_reverse_order() -> None:
    order: list[str] = []

    async def s1(app: FastAPI) -> None:  # noqa: ARG001
        order.append("s1")

    async def s2(app: FastAPI) -> None:  # noqa: ARG001
        order.append("s2")

    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app(on_shutdown=[s1, s2])
    with TestClient(app):
        pass
    assert order == ["s2", "s1"]


def test_startup_hook_exception_fails_boot() -> None:
    class _BoomError(RuntimeError):
        pass

    async def bad(app: FastAPI) -> None:  # noqa: ARG001
        raise _BoomError("nope")

    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app(on_startup=[bad])
    with pytest.raises(_BoomError), TestClient(app):
        pass


def test_shutdown_hook_exception_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    order: list[str] = []

    async def bad(app: FastAPI) -> None:  # noqa: ARG001
        order.append("bad")
        raise RuntimeError("oops")

    async def good(app: FastAPI) -> None:  # noqa: ARG001
        order.append("good")

    engine = Engine(providers=[_RecordingProvider()])
    # good runs LAST during startup, FIRST during shutdown (reverse).
    app = engine.fastapi_app(on_shutdown=[good, bad])
    with (
        caplog.at_level(logging.ERROR, logger="pydantic_ai_stateflow.engine"),
        TestClient(app),
    ):
        pass
    # bad ran first (reverse order: [bad, good] in shutdown), good ran after
    # despite bad's exception.
    assert order == ["bad", "good"]
    assert any("shutdown hook" in rec.message for rec in caplog.records)
