from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.api.health import build_health_router


def test_healthz_returns_200_ok():
    app = FastAPI()
    app.include_router(build_health_router())
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz_respects_prefix():
    app = FastAPI()
    app.include_router(build_health_router(prefix="/api"))
    with TestClient(app) as c:
        r = c.get("/api/healthz")
    assert r.status_code == 200


def test_healthz_passes_optional_checks():
    """Optional checks fold into response when provided."""
    calls: list[str] = []

    async def db_ok() -> bool:
        calls.append("db")
        return True

    app = FastAPI()
    app.include_router(build_health_router(checks={"db": db_ok}))
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "checks": {"db": "ok"}}
    assert calls == ["db"]
