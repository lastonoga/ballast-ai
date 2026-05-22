"""Framework cross-cutting infrastructure bundle + per-call run context.

``Infra`` holds the singletons (repos, event log, stream, broadcaster) an
app declares ONCE at startup. ``RunContext`` is the per-call envelope
that flows / agents receive as their first method argument — it carries
the same triplet plus per-call fields (parent thread id, workflow id).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from pydantic_ai_stateflow.persistence.events.repository import (
        EventLogRepository,
    )
    from pydantic_ai_stateflow.persistence.thread.repository import (
        ThreadRepository,
    )
    from pydantic_ai_stateflow.runtime.event_stream import EventStream
    from pydantic_ai_stateflow.runtime.thread_events import ThreadEventBroadcaster


@dataclass(frozen=True)
class Infra:
    """Framework-owned cross-cutting singletons.

    Construct once at app startup; pass to ``sf.create_app(infra=...)``
    and use ``infra.context(...)`` to mint per-call ``RunContext`` for
    flow/agent invocations.
    """

    thread_repo: "ThreadRepository"
    event_log: "EventLogRepository"
    event_stream: "EventStream"

    @cached_property
    def broadcaster(self) -> "ThreadEventBroadcaster":
        """Derived: a ``ThreadEventBroadcaster`` over the same triplet."""
        from pydantic_ai_stateflow.runtime.thread_events import (
            ThreadEventBroadcaster,
        )
        return ThreadEventBroadcaster(
            thread_repo=self.thread_repo,
            event_log=self.event_log,
            event_stream=self.event_stream,
        )

    def context(
        self,
        *,
        parent_thread_id: UUID | None = None,
        workflow_id: str | None = None,
    ) -> "RunContext":
        """Build a per-call ``RunContext`` from this Infra."""
        return RunContext(
            thread_repo=self.thread_repo,
            event_log=self.event_log,
            event_stream=self.event_stream,
            parent_thread_id=parent_thread_id,
            workflow_id=workflow_id,
        )


@dataclass(frozen=True)
class RunContext:
    """Per-call envelope passed as first argument to flow/agent methods."""

    thread_repo: "ThreadRepository"
    event_log: "EventLogRepository"
    event_stream: "EventStream"
    parent_thread_id: UUID | None = None
    workflow_id: str | None = None

    @cached_property
    def broadcaster(self) -> "ThreadEventBroadcaster":
        from pydantic_ai_stateflow.runtime.thread_events import (
            ThreadEventBroadcaster,
        )
        return ThreadEventBroadcaster(
            thread_repo=self.thread_repo,
            event_log=self.event_log,
            event_stream=self.event_stream,
        )

    def with_(self, **changes: object) -> "RunContext":
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)


__all__ = ["Infra", "RunContext"]
