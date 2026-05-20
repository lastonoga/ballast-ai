from __future__ import annotations

from typing import cast

from fastapi import HTTPException, Request

from pydantic_ai_stateflow.runtime import Engine
from pydantic_ai_stateflow.runtime.container import Container


def get_container(request: Request) -> Container:
    """Resolve the framework Container from ``app.state.container``."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=500,
            detail="Container not attached to app.state — call Engine.fastapi_app()",
        )
    return cast(Container, container)


def get_engine(request: Request) -> Engine:
    """Resolve the Engine from ``app.state.engine``."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=500,
            detail="Engine not attached to app.state — call Engine.fastapi_app()",
        )
    return cast(Engine, engine)
