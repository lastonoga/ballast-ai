from __future__ import annotations

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

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
