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
makes it core framework surface (``from ballast import
Durable``), not observability glue.

## Migration recipe for existing patterns

Recipe::

    from dbos import DBOSConfiguredInstance, Queue
    from ballast import Durable

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

# OTel-carrier helpers are imported LAZILY at call time. Importing them
# eagerly at module load triggers ``observability/__init__.py``, which
# transitively pulls in ``runtime`` modules that themselves want
# ``Durable`` (e.g. ``DurableAgent``) — a circular import.
# Lazy import inside the wrapper functions sidesteps the cycle and has
# negligible per-call cost (Python caches module imports).

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
        from ballast.observability.otel_carrier import (  # noqa: PLC0415
            otel_context_from,
        )

        carrier = kwargs.pop(_CARRIER_KWARG, None)
        with otel_context_from(carrier):
            return await fn(*args, **kwargs)

    return wrapper


def _inject_carrier() -> dict[str, str] | None:
    """Local indirection so the OTel import stays lazy."""
    from ballast.observability.otel_carrier import (  # noqa: PLC0415
        inject_otel_carrier,
    )
    return inject_otel_carrier()


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
        kwargs[_CARRIER_KWARG] = _inject_carrier()
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

        kwargs[_CARRIER_KWARG] = _inject_carrier()
        return await DBOS.start_workflow_async(fn, *args, **kwargs)

    # ── inter-workflow messaging ────────────────────────────────────
    #
    # ``send``/``recv`` happen INSIDE a workflow's own fiber — they
    # don't spawn anything, so they don't need OTel carrier
    # propagation. They're re-exposed on ``Durable`` purely so pattern
    # authors never have to import ``dbos`` directly (the framework
    # owns its dependency on DBOS through this one facade).

    @staticmethod
    async def send_async(
        destination_id: str, message: Any, topic: str | None = None,
    ) -> None:
        """``DBOS.send_async`` — message to another workflow's recv channel."""
        from dbos import DBOS  # noqa: PLC0415
        await DBOS.send_async(destination_id, message, topic)

    @staticmethod
    def send(
        destination_id: str, message: Any, topic: str | None = None,
    ) -> None:
        """``DBOS.send`` — sync sibling of :meth:`send_async`.

        Aborts with "called sync from async" inside a workflow body —
        use :meth:`send_async` from inside ``@Durable.workflow``."""
        from dbos import DBOS  # noqa: PLC0415
        DBOS.send(destination_id, message, topic)

    @staticmethod
    async def recv_async(
        topic: str | None = None, timeout_seconds: float | None = None,
    ) -> Any:
        """``DBOS.recv_async`` — block until a matching message arrives."""
        from dbos import DBOS  # noqa: PLC0415
        if timeout_seconds is None:
            return await DBOS.recv_async(topic)
        return await DBOS.recv_async(topic, timeout_seconds=timeout_seconds)

    @staticmethod
    async def recv(
        topic: str | None = None, timeout_seconds: float | None = None,
    ) -> Any:
        """``DBOS.recv`` — sync sibling of :meth:`recv_async`.

        Pydantic-ai's DBOS recv is implemented via ``recv`` (not
        ``recv_async``) inside some channel implementations; that's
        intentional and fine because those channels run on a sync
        fiber. Most new code should prefer :meth:`recv_async`."""
        from dbos import DBOS  # noqa: PLC0415
        if timeout_seconds is None:
            return await DBOS.recv(topic)
        return await DBOS.recv(topic, timeout_seconds=timeout_seconds)

    # ── control plane / introspection ───────────────────────────────

    @staticmethod
    async def list_workflows(**kwargs: Any) -> Any:
        """``DBOS.list_workflows_async`` — query the workflow log."""
        from dbos import DBOS  # noqa: PLC0415
        return await DBOS.list_workflows_async(**kwargs)

    @staticmethod
    async def list_workflow_steps(
        workflow_id: str, **kwargs: Any,
    ) -> Any:
        """``DBOS.list_workflow_steps_async``."""
        from dbos import DBOS  # noqa: PLC0415
        return await DBOS.list_workflow_steps_async(workflow_id, **kwargs)

    @staticmethod
    async def cancel_workflow(workflow_id: str) -> None:
        """``DBOS.cancel_workflow_async``."""
        from dbos import DBOS  # noqa: PLC0415
        await DBOS.cancel_workflow_async(workflow_id)

    @staticmethod
    async def resume_workflow(workflow_id: str) -> Any:
        """``DBOS.resume_workflow_async`` — returns a workflow handle."""
        from dbos import DBOS  # noqa: PLC0415
        return await DBOS.resume_workflow_async(workflow_id)

    @staticmethod
    async def fork_workflow(
        workflow_id: str, start_step: int, **kwargs: Any,
    ) -> Any:
        """``DBOS.fork_workflow_async`` — returns the forked handle."""
        from dbos import DBOS  # noqa: PLC0415
        return await DBOS.fork_workflow_async(workflow_id, start_step, **kwargs)

    @staticmethod
    async def retrieve_workflow(workflow_id: str) -> Any:
        """``DBOS.retrieve_workflow_async`` — handle for an existing
        workflow id (used by tests / callers waiting on results)."""
        from dbos import DBOS  # noqa: PLC0415
        return await DBOS.retrieve_workflow_async(workflow_id)

    # ── context accessor ────────────────────────────────────────────

    @staticmethod
    def current_workflow_id() -> str:
        """Workflow id of the currently-executing ``@Durable.workflow``
        body. Equivalent to reading ``DBOS.workflow_id``.

        Use inside a workflow body to mint child-workflow ids that
        encode the parent's id, or to address ``send_async`` calls
        back to the current workflow."""
        from typing import cast  # noqa: PLC0415

        from dbos import DBOS  # noqa: PLC0415
        return cast(str, DBOS.workflow_id)

    # ── lifecycle ───────────────────────────────────────────────────

    @staticmethod
    def init(config: Any) -> None:
        """``DBOS(config=config)`` — register the singleton.

        Call at app boot, BEFORE :meth:`launch`. ``config`` is a
        ``DBOSConfig`` instance (re-imported from ``dbos`` by the app;
        not wrapped here to avoid pinning the type to a specific DBOS
        version)."""
        from dbos import DBOS  # noqa: PLC0415
        DBOS(config=config)

    @staticmethod
    def launch() -> None:
        """``DBOS.launch()`` — start the workflow runtime + queue
        workers. Call after :meth:`init` and after every workflow /
        step / configured-instance has been declared (DBOS warns
        otherwise)."""
        from dbos import DBOS  # noqa: PLC0415
        DBOS.launch()

    @staticmethod
    def destroy(*, destroy_registry: bool = False) -> None:
        """``DBOS.destroy(...)`` — tear down the runtime.

        ``destroy_registry=False`` (default) leaves the workflow /
        step / queue registrations in place so the same process can
        re-launch DBOS later (used by tests that re-create the
        FastAPI app per case)."""
        from dbos import DBOS  # noqa: PLC0415
        DBOS.destroy(destroy_registry=destroy_registry)


__all__ = ["Durable"]
