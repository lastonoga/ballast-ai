"""Module-level ``@Durable.step`` helpers for ``ThreadChannel``.

These are fresh registrations — NOT shared with the copies in
``ask.py`` / ``durable.py`` (which are still live and must not be
disturbed). T4 will delete those legacy copies; until then, the names
here are prefixed with ``_tc_`` to avoid any ``@Durable.step`` name
collisions.

All three functions must be defined at module top-level so DBOS can
register them at import time, before ``DBOS.launch()``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from ballast.durable import Durable

if TYPE_CHECKING:
    pass


@Durable.step()
async def _create_helper_thread(
    *, agent_name: str, metadata: dict[str, Any],
) -> UUID:
    """Create a helper thread in the thread repository (memoised step)."""
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415

    thread = await get_ballast().thread_repo.create(
        agent=agent_name, metadata=metadata,
    )
    return thread.id


@Durable.step()
async def _seed_opening_message(thread_id: UUID, opening_message: str) -> None:
    """Append an opening assistant message to the helper thread (memoised step)."""
    from ballast.runtime.engine import get_ballast  # noqa: PLC0415

    await get_ballast().thread_repo.add_message(
        thread_id,
        role="assistant",
        parts=[{
            "type": "text",
            "text": opening_message,
            "state": "done",
        }],
    )


async def _notify_parent_thread(
    *,
    parent_thread_id: str,
    helper_thread_id: UUID,
    helper_agent_name: str,
    helper_metadata: dict[str, Any],
) -> None:
    """Emit ``helper_thread_created`` so the parent thread's UI refreshes.

    Not a ``@Durable.step`` because ``Signal.send`` is not a pure
    side-effect that needs memoisation — it's in-process fan-out that
    happens on every (re)play anyway.  Callers that need step-level
    memoisation should wrap the whole ``deliver`` path in a workflow.
    """
    import sys  # noqa: PLC0415

    from ballast.events import helper_thread_created  # noqa: PLC0415

    await helper_thread_created.send(
        sys.modules[__name__],
        parent_thread_id=UUID(parent_thread_id),
        helper_thread_id=helper_thread_id,
        helper_agent_name=helper_agent_name,
        helper_metadata=helper_metadata,
    )


__all__ = [
    "_create_helper_thread",
    "_notify_parent_thread",
    "_seed_opening_message",
]
