"""``Durable`` — single facade over DBOS + OTel propagation.

Pattern authors use ``Durable.workflow / step / dbos_class`` on
worker bodies and ``Durable.enqueue / start_workflow`` at call sites
INSTEAD of importing ``DBOS`` / ``Queue`` directly. Behaviour:

* The decorators stack ``@DBOS.workflow()`` / ``@DBOS.step()`` with
  automatic extraction of the W3C ``traceparent`` carrier that the
  call-side helpers inject. Every span emitted inside the worker
  body (pydantic-ai ``chat`` spans, ``@traced`` blocks, manual
  ``logfire.span``) becomes a child of the caller's active span in
  Logfire — no fragmented root traces across DBOS queue / workflow
  boundaries.

* The call-side helpers (``Durable.enqueue``, ``Durable.start_workflow``)
  inject the current OTel context into a magic kwarg the worker-side
  decorator pops back out. Together they guarantee the propagation is
  ``either both or neither``: forgetting the decorator on the worker
  surfaces as a noisy ``TypeError`` on the magic kwarg, not silent
  trace fragmentation.

See ``observability/otel_carrier.py`` for the low-level primitives
(``inject_otel_carrier`` / ``otel_context_from``) and the rationale
behind the W3C-traceparent-through-a-queue trick.

## Why this lives at top level, not under ``observability/``

``observability/`` is for "tools that observe code" (logfire spans,
cost extractors, structured loggers). ``Durable`` is in the
opposite direction — it's a **dependency-inversion shim** that
pattern authors take as a hard dep instead of taking ``DBOS``. That
makes it core framework surface (``from pydantic_ai_stateflow import
Durable``), not observability glue.

## Migration recipe for existing patterns

Recipe::

    from dbos import DBOSConfiguredInstance, Queue
    from pydantic_ai_stateflow import Durable

    @Durable.dbos_class()
    class MyPattern(DBOSConfiguredInstance):
        @Durable.workflow()
        async def run(self, task):
            await Durable.enqueue(self._q, self._worker, task)

        @Durable.step()
        async def _worker(self, task):
            ...

``DBOSConfiguredInstance`` and ``Queue`` are runtime types (not
behaviour we want to wrap), so they keep their direct import from
``dbos``. Everything that crosses a fiber boundary
(workflow / step / enqueue / start_workflow) goes through
``Durable``.
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

# Name of the kwarg used to ship the OTel carrier through DBOS. Pattern
# authors must NOT accept this kwarg in their own worker signatures —
# the decorators (``Durable.workflow`` / ``Durable.step``) pop it
# before invoking the body. Treat as reserved.
_CARRIER_KWARG = "__otel_carrier__"


def _wrap_with_carrier_attach(
    fn: Callable[P, Awaitable[R]],
) -> Callable[..., Awaitable[R]]:
    """Pop the carrier kwarg, attach OTel context, then call ``fn``.

    Internal — sits between DBOS's own decorator (``@DBOS.workflow`` /
    ``@DBOS.step``) and the user's body. DBOS records the carrier as
    part of the step / workflow input log so replays restore the same
    trace context.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> R:
        carrier = kwargs.pop(_CARRIER_KWARG, None)
        with otel_context_from(carrier):
            return await fn(*args, **kwargs)

    return wrapper


class Durable:
    """Facade bundling DBOS workflow lifecycle + OTel propagation.

    Use exclusively in patterns / app code that crosses a DBOS
    fiber boundary. Direct ``DBOS.workflow`` / ``DBOS.step`` /
    ``Queue.enqueue_async`` calls still work but lose trace nesting
    in Logfire.
    """

    # ── worker-side decorators ──────────────────────────────────────

    @staticmethod
    def workflow(
        **dbos_kwargs: Any,
    ) -> Callable[[Callable[P, Awaitable[R]]], Callable[..., Awaitable[R]]]:
        """``@DBOS.workflow()`` + automatic OTel attach on entry.

        Forward ``**dbos_kwargs`` (e.g. ``name=``) verbatim to
        ``DBOS.workflow``. Decorator order is fixed internally — apply
        as the **only** workflow decorator on the body.
        """
        from dbos import DBOS  # noqa: PLC0415 — lazy so tests w/o dbos still import

        dbos_decorator = DBOS.workflow(**dbos_kwargs)

        def decorator(
            fn: Callable[P, Awaitable[R]],
        ) -> Callable[..., Awaitable[R]]:
            return dbos_decorator(_wrap_with_carrier_attach(fn))

        return decorator

    @staticmethod
    def step(
        **dbos_kwargs: Any,
    ) -> Callable[[Callable[P, Awaitable[R]]], Callable[..., Awaitable[R]]]:
        """``@DBOS.step()`` + automatic OTel attach on entry."""
        from dbos import DBOS  # noqa: PLC0415

        dbos_decorator = DBOS.step(**dbos_kwargs)

        def decorator(
            fn: Callable[P, Awaitable[R]],
        ) -> Callable[..., Awaitable[R]]:
            return dbos_decorator(_wrap_with_carrier_attach(fn))

        return decorator

    @staticmethod
    def dbos_class(**dbos_kwargs: Any) -> Callable[[type], type]:
        """Pass-through to ``@DBOS.dbos_class()``.

        Re-exported here for one-import-source convenience. Has no
        OTel side effects — the class-level decorator is just DBOS's
        own configured-instance registration helper, and there's no
        boundary crossing at class declaration time.
        """
        from dbos import DBOS  # noqa: PLC0415

        return DBOS.dbos_class(**dbos_kwargs)

    # ── caller-side helpers ─────────────────────────────────────────

    @staticmethod
    async def enqueue(
        queue: Queue,
        fn: Callable[P, Awaitable[R]],
        *args: Any,
        **kwargs: Any,
    ) -> WorkflowHandleAsync[R]:
        """``queue.enqueue_async(fn, *args, **kwargs)`` + carrier inject.

        ``fn`` MUST be decorated with ``@Durable.step`` (or
        ``@Durable.workflow``) — the magic kwarg this helper adds will
        otherwise hit the body as an unexpected argument.
        """
        kwargs[_CARRIER_KWARG] = inject_otel_carrier()
        return await queue.enqueue_async(fn, *args, **kwargs)

    @staticmethod
    async def start_workflow(
        fn: Callable[P, Awaitable[R]],
        *args: Any,
        **kwargs: Any,
    ) -> WorkflowHandleAsync[R]:
        """``DBOS.start_workflow_async(fn, *args, **kwargs)`` + carrier
        inject.

        Use with ``SetWorkflowID`` / ``SetEnqueueOptions`` freely::

            with SetWorkflowID(wid), SetEnqueueOptions(...):
                await Durable.start_workflow(MyFlow.run, ...)
        """
        from dbos import DBOS  # noqa: PLC0415

        kwargs[_CARRIER_KWARG] = inject_otel_carrier()
        return await DBOS.start_workflow_async(fn, *args, **kwargs)


__all__ = ["Durable"]
