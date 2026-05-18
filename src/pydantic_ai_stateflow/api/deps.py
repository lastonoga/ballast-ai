from __future__ import annotations

from collections.abc import Callable
from typing import cast
from uuid import UUID

from fastapi import HTTPException, Request

from pydantic_ai_stateflow.runtime import Engine
from pydantic_ai_stateflow.runtime.container import Container

TenantResolver = Callable[[Request], UUID]


def get_container(request: Request) -> Container:
    """Resolve the framework Container from `app.state.container`.

    Spec 4A.0.7 forbids globals — the Container is attached to the FastAPI
    application by `Engine.fastapi_app(...)` and read here.
    """
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=500,
            detail="Container not attached to app.state — call Engine.fastapi_app()",
        )
    return cast(Container, container)


def get_engine(request: Request) -> Engine:
    """Resolve the Engine from `app.state.engine`."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=500,
            detail="Engine not attached to app.state — call Engine.fastapi_app()",
        )
    return cast(Engine, engine)


def get_tenant_id(request: Request) -> UUID:
    """Resolve the tenant for this request.

    Order:
      1. If `app.state.tenant_resolver` is set, call it (app-defined auth wins).
      2. Otherwise read `X-Tenant-Id` header and parse as UUID.

    Raises 400 if neither path yields a valid UUID.
    """
    resolver: TenantResolver | None = getattr(
        request.app.state, "tenant_resolver", None,
    )
    if resolver is not None:
        return resolver(request)
    raw = request.headers.get("X-Tenant-Id")
    if not raw:
        raise HTTPException(status_code=400, detail="X-Tenant-Id header required")
    try:
        return UUID(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="X-Tenant-Id must be a UUID",
        ) from exc
