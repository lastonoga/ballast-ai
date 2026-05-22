"""Process-wide Engine config holder.

Constructed once by ``ballast.create_app(thread_repo=, event_log=, event_stream=, ...)``.
Framework code reads it via ``ballast.get_engine()`` (lazy lookup, raises
``ConfigurationError`` if create_app hasn't been called).

NOT a DI container — just a typed dataclass holding the singletons
the framework needs at runtime. Apps own the actual repo/stream
instances; they pass them into ``create_app`` once.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from ballast.errors import ConfigurationError

if TYPE_CHECKING:
    from ballast.persistence.events.repository import (
        EventLogRepository,
    )
    from ballast.persistence.thread.repository import (
        ThreadRepository,
    )
    from ballast.runtime.event_stream import EventStream
    from ballast.runtime.thread_events import ThreadEventBroadcaster


@dataclass(frozen=True)
class Engine:
    """Configured runtime — repos + stream + derived broadcaster."""

    thread_repo: "ThreadRepository"
    event_log: "EventLogRepository"
    event_stream: "EventStream"

    @cached_property
    def broadcaster(self) -> "ThreadEventBroadcaster":
        from ballast.runtime.thread_events import (
            ThreadEventBroadcaster,
        )
        return ThreadEventBroadcaster(
            thread_repo=self.thread_repo,
            event_log=self.event_log,
            event_stream=self.event_stream,
        )


_engine: Engine | None = None


def get_engine() -> Engine:
    """Return the process-wide engine. Raises if create_app not called."""
    if _engine is None:
        raise ConfigurationError(
            "Engine not initialized — call ballast.create_app(...) first",
            hint="ballast.create_app builds the Engine; framework code reads it via get_engine()",
        )
    return _engine


def _set_engine(engine: Engine) -> None:
    """Set the process-wide engine. Called ONLY by create_app.

    Idempotent if same engine; raises if reassigning to a different one
    (apps that need to swap must explicitly reset for testing).
    """
    global _engine
    if _engine is not None and _engine is not engine:
        raise ConfigurationError(
            "ballast.create_app() called twice with different configs",
            hint="One Engine per process. For tests, use _reset_engine_for_tests() between cases.",
        )
    _engine = engine


def _reset_engine_for_tests() -> None:
    """Test-only: clear the singleton so the next create_app builds fresh."""
    global _engine
    _engine = None


__all__ = ["Engine", "_reset_engine_for_tests", "_set_engine", "get_engine"]
