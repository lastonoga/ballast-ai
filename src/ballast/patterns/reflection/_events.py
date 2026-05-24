"""Reflection progress events — typed signal + default chat router.

Matches the wire shape used by ``DivergentConvergent`` / brainstorm:
the pattern's loop publishes ``ReflectionEvent`` instances on
``reflection_progress``; the default handler (auto-connected at module
import) renders each one as a ``data-reflection-{type}`` UI card via
:class:`ThreadEventBroadcaster` when a ``progress_to_thread(...)``
scope is active.

Apps that want a different routing — e.g. only stream certain types,
or send to Slack — disconnect :func:`default_chat_router` and connect
their own ``@receiver(reflection_progress)`` handler.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ballast.events import Signal, progress_thread_var


class ReflectionEvent(BaseModel):
    """One observable boundary inside a ``Reflection.run`` loop.

    ``type``:
      - ``"draft"``     — writer produced iteration N's draft
      - ``"critique"``  — critic assessed iteration N's draft
      - ``"refine"``    — non-passing critique, loop continues
      - ``"passed"``    — terminal: critic approved, run returns
      - ``"exhausted"`` — terminal: hit max_iter, run raises

    ``payload`` is free-form per event type:
      - ``draft``     → ``{"draft": stringified output}``
      - ``critique``  → ``{"passed": bool, "issues": [...],
                           "suggestions": [...], "confidence": float}``
      - ``refine``    → ``{}``  (loop continuation marker)
      - ``passed``    → ``{}``
      - ``exhausted`` → ``{"last_critique": critique.model_dump()}``
    """

    type: Literal["draft", "critique", "refine", "passed", "exhausted"]
    iter: int
    payload: dict[str, Any]


reflection_progress: Signal = Signal("reflection.progress")
"""Module-level signal carrying each :data:`ReflectionEvent` the pattern
emits. Handlers receive ``(sender=Reflection_instance, event=...)``."""


async def default_chat_router(
    sender: Any,  # noqa: ARG001
    *,
    event: ReflectionEvent,
    **_: Any,
) -> None:
    """Bundled :data:`reflection_progress` handler.

    Reads :data:`progress_thread_var` — if the workflow body didn't
    open a ``progress_to_thread(...)`` scope this is a no-op. Otherwise
    writes a ``data-reflection-{type}`` part to the thread via the
    engine's broadcaster (one round-trip: persist + event log +
    publish).

    Auto-connected at module import. Disconnect + connect your own
    handler if you want different routing.
    """
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415

    thread_id = progress_thread_var.get()
    if thread_id is None:
        return
    await get_ballast().broadcaster.emit_raw(
        thread_id,
        part={
            "type": f"data-reflection-{event.type}",
            "data": event.model_dump(mode="json"),
        },
        persistent=True,
    )


reflection_progress.connect(default_chat_router)


__all__ = [
    "ReflectionEvent",
    "default_chat_router",
    "reflection_progress",
]
