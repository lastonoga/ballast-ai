"""BallastError → application/problem+json middleware + handler.

Auto-installed by ``ballast.create_app()`` when
``settings.api.install_error_middleware`` is True (the default).
"""
from __future__ import annotations

import traceback as _tb
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ballast.errors import BallastError
from ballast.logging import get_logger

_logger = get_logger(__name__)
PROBLEM_JSON = "application/problem+json"


def _resolve_expose_tracebacks() -> bool:
    """Tri-state resolution from settings.

    None → auto-detect (True iff observability.environment == "dev").
    Explicit True/False → wins.
    """
    from ballast.settings import get_settings

    settings = get_settings()
    explicit = settings.api.expose_tracebacks
    if explicit is not None:
        return explicit
    return settings.observability.environment == "dev"


def _render(exc: BallastError, *, include_trace: bool) -> dict[str, Any]:
    body: dict[str, Any] = {"error": exc.to_dict()}
    if include_trace:
        body["error"]["traceback"] = _tb.format_exc()
    return body


def _emit_log(exc: BallastError) -> None:
    _logger.warning(
        "BallastError raised: code=%s detail=%s context=%s",
        exc.code, exc.detail, exc.context,
    )


def _emit_span_event(exc: BallastError) -> None:
    """Attach a span event to the current logfire span (if logfire is present).

    Gated on ``observability.config.is_configured()`` — emitting a
    logfire span before ``configure()`` triggers
    ``LogfireNotConfiguredWarning`` in tests where logfire is importable
    but no observability has been installed.
    """
    try:
        from ballast.observability import has_logfire
        from ballast.observability.config import is_configured
    except ImportError:
        return
    if not has_logfire() or not is_configured():
        return
    try:
        import logfire
        with logfire.span("stateflow_error", _level="error") as span:
            span.set_attributes({
                "stateflow.error.code": exc.code,
                "stateflow.error.detail": exc.detail,
                "stateflow.error.hint": exc.hint or "",
            })
            span.record_exception(exc)
    except Exception:
        # never crash on observability failures
        _logger.exception("logfire span emit failed")


async def stateflow_error_handler(request: Request, exc: BallastError) -> JSONResponse:
    del request
    include_trace = _resolve_expose_tracebacks()
    _emit_log(exc)
    _emit_span_event(exc)
    return JSONResponse(
        content=_render(exc, include_trace=include_trace),
        status_code=exc.status_code,
        media_type=PROBLEM_JSON,
    )


class BallastErrorMiddleware(BaseHTTPMiddleware):
    """Catches BallastError that escapes streaming / background paths.

    Also logs unexpected (non-Ballast) exceptions with full traceback before
    re-raising — otherwise FastAPI's default 500 handler swallows them
    silently and there is no way to tell what blew up.
    """

    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except BallastError as exc:
            return await stateflow_error_handler(request, exc)
        except Exception:
            _logger.exception(
                "Unhandled exception during %s %s",
                request.method, request.url.path,
            )
            raise


def install_error_handlers(app: FastAPI) -> None:
    """Mount the handler + middleware on ``app``. Idempotent.

    Gated by ``settings.api.install_error_middleware``. Apps that
    handle BallastError themselves set the flag False and skip this.
    """
    from ballast.settings import get_settings

    if not get_settings().api.install_error_middleware:
        return
    if getattr(app.state, "_stateflow_error_handlers_installed", False):
        return
    app.add_exception_handler(BallastError, stateflow_error_handler)
    app.add_middleware(BallastErrorMiddleware)
    app.state._stateflow_error_handlers_installed = True


__all__ = [
    "PROBLEM_JSON",
    "BallastErrorMiddleware",
    "install_error_handlers",
    "stateflow_error_handler",
]
