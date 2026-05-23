"""Context-manager API for pattern-progress routing without arg threading.

Patterns emit typed events on their own signals (see e.g.
:data:`ballast.patterns.divergent_convergent.events.divergent_convergent_progress`)
and also publish a human-readable narration via
:data:`ballast.events.chat_message_requested` when a destination is
configured. Apps pick the destination by wrapping the call site in a
context manager:

    with progress_to_thread(thread_id=parent_thread_id):
        chosen = await _divergent.run(topic)

The pattern body reads :data:`progress_thread_var` to discover the
active destination — no kwarg threading through every pattern method,
no closure connected to a Signal inside the workflow body.

## Why a ContextVar (and not a plain kwarg)

* Patterns are framework primitives — they shouldn't know about
  "thread destinations" as first-class arguments. ContextVar pushes
  destination configuration to the call site without baking it into
  the pattern signature.
* Apps can stack contexts (``with progress_to_thread(t1): ...
  with progress_to_thread(t2): ...``) and inner scopes win.
* Empirically — :class:`asyncio.Task` propagates ContextVars across
  task boundaries, and DBOS workflows / steps inherit the caller's
  context. (Queue workers via ``Durable.enqueue`` do NOT — but
  patterns emit progress from the workflow body, not from queue
  workers, so this isn't a problem in practice.)

## Adding new destinations

Today only ``progress_to_thread`` ships. Other destinations follow
the same pattern: a context manager that sets its own ContextVar,
plus a small block inside pattern bodies that reads + acts on it.
Apps that need fully custom routing connect directly to the typed
pattern signal (``divergent_convergent_progress`` and friends) —
that channel stays open regardless of which contexts are active.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID


progress_thread_var: ContextVar["UUID | None"] = ContextVar(
    "ballast.progress_thread", default=None,
)
"""Active destination for pattern progress narration.

When set (via :func:`progress_to_thread`), patterns post one
``chat_message_requested`` per observable boundary into this thread.
When ``None``, patterns still fire their typed signals so observers
listening on those see the events, but no chat narration is
emitted."""


@contextmanager
def progress_to_thread(thread_id: "UUID") -> "Iterator[None]":
    """Route pattern progress narration into ``thread_id`` for the scope.

    Usage::

        @Durable.workflow()
        async def my_flow(task):
            with progress_to_thread(task.parent_thread_id):
                chosen = await _divergent.run(task.topic)
            ...

    Nested scopes inherit + override the inner-most binding (standard
    ContextVar semantics — ``reset(token)`` restores the previous
    value on exit). On exception the binding is still cleaned up.
    """
    token = progress_thread_var.set(thread_id)
    try:
        yield
    finally:
        progress_thread_var.reset(token)


__all__ = ["progress_thread_var", "progress_to_thread"]
