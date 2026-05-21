"""OTel trace-context propagation across DBOS queue / workflow boundaries.

## Why this exists

When ``DBOS.Queue.enqueue_async(fn, ...)`` schedules ``fn`` onto a
worker, the worker runs in a **different asyncio fiber** from the
enqueueing code. Python's OpenTelemetry SDK keeps the active span /
trace context in a ``contextvars.ContextVar`` — that ContextVar is
fiber-local, so the worker starts with no active span.

Practical consequence: every span emitted from inside an enqueued
function (e.g. an ``@traced`` decorator on a pattern step, or the
pydantic-ai instrumentation that wraps each model request) becomes a
new **root** trace in Logfire instead of a child of the workflow
that enqueued it. The user sees the brainstorm fan-out as N
disconnected traces rather than one tree.

## Pattern

The W3C ``traceparent`` propagator is purpose-built for exactly this
situation (microservice boundaries, message queues). We treat the
DBOS queue as a "message queue" and use the same trick:

  enqueueing code:
      carrier = inject_otel_carrier()  # captures current trace
      await queue.enqueue_async(worker_fn, carrier, ...other_args)

  worker:
      token = attach_otel_carrier(carrier)
      try:
          ... # spans here are children of the caller's active span
      finally:
          detach_otel_carrier(token)

The carrier is a tiny ``dict[str, str]`` (``traceparent`` + optional
``tracestate`` headers); DBOS pickle-serializes it without issue and
the worker round-trips it back into an OTel ``Context``.

When OpenTelemetry isn't installed at all (e.g. tests that don't
import logfire), both functions degrade to no-ops — ``inject_*``
returns ``None`` and ``attach_*(None)`` returns a sentinel that
``detach_*`` ignores. Code that wraps a worker body can therefore
unconditionally pass / attach without feature-detecting OTel.

Refs:
- W3C Trace Context: https://www.w3.org/TR/trace-context/
- OTel Python propagation API:
  https://opentelemetry-python.readthedocs.io/en/latest/api/propagate.html
"""

from __future__ import annotations

from typing import Any

# Sentinel returned by ``attach_otel_carrier`` when propagation is a
# no-op (OTel missing OR carrier is None/empty). ``detach_*`` checks
# for this so callers don't have to special-case it.
_NOOP_TOKEN: object = object()


def _otel_modules() -> tuple[Any, Any] | None:
    """Resolve ``opentelemetry.propagate`` + ``opentelemetry.context``
    if installed; ``None`` otherwise.

    Lazy import + cached at module level by Python's import system on
    first successful call.
    """
    try:
        from opentelemetry import context as otel_context  # noqa: PLC0415
        from opentelemetry import propagate as otel_propagate  # noqa: PLC0415
    except Exception:
        return None
    return otel_propagate, otel_context


def inject_otel_carrier() -> dict[str, str] | None:
    """Capture the current OTel context into a serializable carrier.

    Returns ``None`` when OpenTelemetry isn't installed (no spans to
    propagate anyway). When OTel is present but no span is currently
    active, returns an empty dict — pass it through anyway; the
    receiving side will treat empty/None identically and not attach
    a context.

    The result is a plain ``dict[str, str]`` (W3C ``traceparent`` /
    ``tracestate`` headers). DBOS pickles workflow args, so a pure-
    string dict survives the queue round-trip cleanly.
    """
    mods = _otel_modules()
    if mods is None:
        return None
    propagate, _ = mods
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def attach_otel_carrier(carrier: dict[str, str] | None) -> Any:
    """Activate the trace context from ``carrier`` on the current
    fiber. Returns an opaque token to pass back to
    ``detach_otel_carrier`` for cleanup.

    No-op (returns a sentinel) when OTel is missing OR carrier is
    falsy. Always call ``detach_otel_carrier(token)`` in a ``finally``
    block to avoid leaking the activation into subsequent code on
    the same fiber.
    """
    if not carrier:
        return _NOOP_TOKEN
    mods = _otel_modules()
    if mods is None:
        return _NOOP_TOKEN
    propagate, context = mods
    ctx = propagate.extract(carrier)
    return context.attach(ctx)


def detach_otel_carrier(token: Any) -> None:
    """Pop the context activated by ``attach_otel_carrier``."""
    if token is _NOOP_TOKEN:
        return
    mods = _otel_modules()
    if mods is None:
        return
    _, context = mods
    try:
        context.detach(token)
    except Exception:
        # Detaching a token from a different context is harmless in
        # most OTel SDKs but spammy — swallow defensively.
        pass


class otel_context_from:  # noqa: N801 — context-manager naming
    """Sugar over ``attach_otel_carrier`` / ``detach_otel_carrier``.

    Usage::

        async def _worker(carrier, ...):
            with otel_context_from(carrier):
                ...  # spans emitted here nest under the caller
    """

    def __init__(self, carrier: dict[str, str] | None) -> None:
        self._carrier = carrier
        self._token: Any = _NOOP_TOKEN

    def __enter__(self) -> None:
        self._token = attach_otel_carrier(self._carrier)
        return None

    def __exit__(self, *_: Any) -> None:
        detach_otel_carrier(self._token)


__all__ = [
    "attach_otel_carrier",
    "detach_otel_carrier",
    "inject_otel_carrier",
    "otel_context_from",
]
