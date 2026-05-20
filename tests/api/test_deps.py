from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.api.deps import get_container, get_engine
from pydantic_ai_stateflow.runtime import Engine

_ContainerDep = Depends(get_container)
_EngineDep = Depends(get_engine)


class _NoopProvider:
    async def register(self, container) -> None:
        return None


def _app_with_engine() -> tuple[FastAPI, Engine]:
    engine = Engine(providers=[_NoopProvider()])
    app = FastAPI()
    app.state.container = engine.container
    app.state.engine = engine

    @app.get("/echo-container")
    async def echo_container(container=_ContainerDep) -> dict[str, str]:
        return {"container_class": type(container).__name__}

    @app.get("/echo-engine")
    async def echo_engine(engine=_EngineDep) -> dict[str, str]:
        return {"engine_class": type(engine).__name__}

    return app, engine


def test_get_container_pulls_from_app_state() -> None:
    app, _ = _app_with_engine()
    with TestClient(app) as c:
        r = c.get("/echo-container")
    assert r.status_code == 200
    assert r.json()["container_class"] == "DefaultContainer"


def test_get_container_raises_when_unset() -> None:
    app = FastAPI()

    @app.get("/x")
    async def x(container=_ContainerDep) -> dict[str, str]:
        return {"ok": "1"}

    with TestClient(app) as c:
        r = c.get("/x")
    assert r.status_code == 500
    assert "container" in r.text.lower()


def test_get_engine_pulls_from_app_state() -> None:
    app, engine = _app_with_engine()
    with TestClient(app) as c:
        r = c.get("/echo-engine")
    assert r.status_code == 200
    assert r.json()["engine_class"] == "Engine"
