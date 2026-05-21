"""StateflowError → application/problem+json middleware + handler.

Auto-installed by ``sf.create_app()`` when
``settings.api.install_error_middleware`` is True (the default).
"""
from __future__ import annotations

import traceback as _tb
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from pydantic_ai_stateflow.errors import StateflowError
from pydantic_ai_stateflow.logging import get_logger

_logger = get_logger(__name__)
PROBLEM_JSON = "application/problem+json"


def _resolve_expose_tracebacks() -> bool:
    """Tri-state resolution from settings.

    None → auto-detect (True iff observability.environment == "dev").
    Explicit True/False → wins.
    """
    from pydantic_ai_stateflow.settings import get_settings

    settings = get_settings()
    explicit = settings.api.expose_tracebacks
    if explicit is not None:
        return explicit
    return settings.observability.environment == "dev"


def _render(exc: StateflowError, *, include_trace: bool) -> dict[str, Any]:
    body: dict[str, Any] = {"error": exc.to_dict()}
    if include_trace:
        body["error"]["traceback"] = _tb.format_exc()
    return body


def _emit_log(exc: StateflowError) -> None:
    _logger.warning(
        "StateflowError raised: code=%s detail=%s context=%s",
        exc.code, exc.detail, exc.context,
    )


def _emit_span_event(exc: StateflowError) -> None:
    """Attach a span event to the current logfire span (if logfire is present)."""
    try:
        from pydantic_ai_stateflow.observability import has_logfire
    except ImportError:
        return
    if not has_logfire():
        return
    try:
        import logfire
        logfire.span("stateflow_error", _level="error").__enter__().set_attributes({
            "stateflow.error.code": exc.code,
            "stateflow.error.detail": exc.detail,
            "stateflow.error.hint": exc.hint or "",
        })
    except Exception:
        # never crash on observability failures
        _logger.exception("logfire span emit failed")


async def stateflow_error_handler(request: Request, exc: StateflowError) -> JSONResponse:
    del request
    include_trace = _resolve_expose_tracebacks()
    _emit_log(exc)
    _emit_span_event(exc)
    return JSONResponse(
        content=_render(exc, include_trace=include_trace),
        status_code=exc.status_code,
        media_type=PROBLEM_JSON,
    )


class StateflowErrorMiddleware(BaseHTTPMiddleware):
    """Catches StateflowError that escapes streaming / background paths."""

    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except StateflowError as exc:
            return await stateflow_error_handler(request, exc)


def install_error_handlers(app: FastAPI) -> None:
    """Mount the handler + middleware on ``app``. Idempotent.

    Gated by ``settings.api.install_error_middleware``. Apps that
    handle StateflowError themselves set the flag False and skip this.
    """
    from pydantic_ai_stateflow.settings import get_settings

    if not get_settings().api.install_error_middleware:
        return
    if getattr(app.state, "_stateflow_error_handlers_installed", False):
        return
    app.add_exception_handler(StateflowError, stateflow_error_handler)
    app.add_middleware(StateflowErrorMiddleware)
    app.state._stateflow_error_handlers_installed = True


__all__ = [
    "PROBLEM_JSON",
    "StateflowErrorMiddleware",
    "install_error_handlers",
    "stateflow_error_handler",
]
