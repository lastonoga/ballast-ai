"""Unified OTel trace propagation across every DBOS workflow / queue
boundary in the framework. One mechanism — three call sites:

* ``traced_enqueue(queue, fn, *args, **kwargs)`` — wraps
  ``Queue.enqueue_async``; captures the current trace context and
  ships it through as a magic kwarg.

* ``traced_start_workflow(fn, *args, **kwargs)`` — wraps
  ``DBOS.start_workflow_async``; same mechanism. Works for both
  child workflows enqueued from inside a parent workflow body and
  one-shot workflows kicked off from a request handler.

* ``@traced_workflow_step`` — applied to the worker function
  (``@DBOS.workflow`` body or ``@DBOS.step``). Extracts the carrier
  from the magic kwarg before invoking the body, attaches it as the
  active OTel context, then detaches in ``finally``. Every span
  emitted inside the body (pydantic-ai instrumentation, nested
  ``@traced`` blocks, manually opened ``logfire.span(...)``) becomes
  a child of the caller's active span.

## Why a "magic" kwarg and not a positional argument

The convention here is to inject the carrier under
``__otel_carrier__`` rather than prepending it to ``*args``. Two
benefits:

1. Pattern authors don't have to rearrange existing call signatures.
   A worker declared ``async def _run(self, *, topic, parent_id)``
   stays exactly the same — ``@traced_workflow_step`` pops the
   carrier from kwargs before the body sees it.

2. Mixing the carrier into positional args makes it visible to
   ``DBOS.list_workflow_steps`` and the inspector UI, which is
   noisy. Hiding it in a kwarg keeps step args readable.

DBOS records the magic kwarg in its step / workflow log so
workflow REPLAY uses the SAME carrier — which means resumed runs
re-attach the original trace context. That's the right behaviour:
the recovery span tree visually matches the original execution.

## Why not just call ``inject_otel_carrier`` / ``attach_otel_carrier``
## directly at every call site?

We did, briefly. Pattern authors forgot to wire it in
``DurableHITLWorkflow`` and the brainstorm endpoint, and every
addition needed a 4-line ritual. Centralising the inject/attach in
two helpers (caller-side) and one decorator (worker-side) makes the
pattern uniform and impossible to half-implement: every async DBOS
boundary in the framework should be using one of these three names.

## Failure modes / no-op behaviour

* OpenTelemetry not installed: ``inject_otel_carrier`` returns
  ``None`` → the magic kwarg's value is ``None`` →
  ``attach_otel_carrier(None)`` is a no-op. Zero overhead.

* Carrier present but stale (e.g. recovered workflow whose original
  trace context's exporter is gone): OTel happily attaches anyway
  and emits spans under the recovered trace id. No crash.

* Worker not decorated with ``@traced_workflow_step``: the magic
  kwarg is forwarded as a real kwarg to the worker's body, which
  will likely TypeError on the unexpected argument. This is
  intentional — surfaces the missing decorator at first call rather
  than silently dropping propagation.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from pydantic_ai_stateflow.observability.otel_carrier import (
    inject_otel_carrier,
    otel_context_from,
)

if TYPE_CHECKING:
    from dbos import Queue
    from dbos._core import WorkflowHandleAsync

P = ParamSpec("P")
R = TypeVar("R")

# Public name of the magic kwarg. Pattern authors should treat this as
# a reserved keyword; never accept ``__otel_carrier__`` in their own
# worker signatures.
_CARRIER_KWARG = "__otel_carrier__"


def traced_workflow_step(
    fn: Callable[P, Awaitable[R]],
) -> Callable[..., Awaitable[R]]:
    """Decorate a worker function so it picks up the caller's OTel
    context.

    Apply OUTSIDE-IN — i.e. directly on the body, BEFORE ``@DBOS.step()``
    / ``@DBOS.workflow()`` so DBOS records the carrier kwarg::

        @DBOS.step()
        @traced_workflow_step
        async def _diverge_one(self, label: str, sample: int, task: T):
            ...

    The worker's own signature stays unchanged; the carrier travels in
    a kwarg that ``traced_enqueue`` / ``traced_start_workflow`` inject
    and this decorator pops before calling the body.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        carrier = kwargs.pop(_CARRIER_KWARG, None)
        with otel_context_from(carrier):
            return await fn(*args, **kwargs)

    return wrapper


async def traced_enqueue(
    queue: Queue,
    fn: Callable[P, Awaitable[R]],
    *args: Any,
    **kwargs: Any,
) -> WorkflowHandleAsync[R]:
    """``Queue.enqueue_async`` + automatic OTel carrier injection.

    Replaces direct calls to ``queue.enqueue_async(fn, ...)`` at every
    fan-out site. The worker function must be decorated with
    ``@traced_workflow_step`` (or it'll raise on the unexpected
    ``__otel_carrier__`` kwarg).
    """
    kwargs[_CARRIER_KWARG] = inject_otel_carrier()
    return await queue.enqueue_async(fn, *args, **kwargs)


async def traced_start_workflow(
    fn: Callable[P, Awaitable[R]],
    *args: Any,
    **kwargs: Any,
) -> WorkflowHandleAsync[R]:
    """``DBOS.start_workflow_async`` + automatic OTel carrier injection.

    Use anywhere a workflow is detached from the calling fiber: child
    workflows spawned from a parent workflow body, one-shot workflows
    kicked off from a FastAPI handler, helper workflows opened by
    ``DurableHITLWorkflow.open``.

    Combine freely with ``SetWorkflowID`` / ``SetEnqueueOptions``::

        with SetWorkflowID(wid), SetEnqueueOptions(...):
            await traced_start_workflow(MyFlow.run, *args)
    """
    from dbos import DBOS  # noqa: PLC0415 — lazy so framework tests don't pin dbos

    kwargs[_CARRIER_KWARG] = inject_otel_carrier()
    return await DBOS.start_workflow_async(fn, *args, **kwargs)


__all__ = [
    "traced_enqueue",
    "traced_start_workflow",
    "traced_workflow_step",
]
