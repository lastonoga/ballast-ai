"""``TestEngine`` — pre-wired Stateflow app for tests.

API:

    engine = TestEngine.default()
    engine.override(ThreadRepository, custom_repo)
    with engine.test_client() as client:
        r = client.post("/threads", json={})

What it does:
- Builds a fresh in-memory app with InMemory repos / event-log / stream
  by default; ``ballast.create_app`` constructs the framework ``Engine``
  from them and installs the process-wide singleton.
- ``override(target, replacement)`` swaps a repo / event-log / event-stream
  via ``app.dependency_overrides[get_X]`` (FastAPI native).
- ``test_client()`` returns a ``TestClient`` context manager that
  runs FastAPI lifespan (and DBOS lifecycle if configured).
- ``_reset_ballast_for_tests()`` is called on every entry so the
  process-wide ``Engine`` singleton doesn't leak across cases.
- Unique DBOS ``name`` per TestEngine instance avoids in-process collisions.

The framework no longer manages a workflow/agent registry — apps own
their own routes and instance lookup. Tests that need to swap an
agent implementation construct the app themselves with the test
agent, or override repos and let the app's own resolution path see
the right doubles.
"""
from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from ballast.api.deps import (
    get_event_log,
    get_event_stream,
    get_thread_repo,
)
from ballast.observability.config import (
    _reset_observability_for_tests,
)
from ballast.persistence import (
    EventLogRepository,
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from ballast.persistence.thread.repository import ThreadRepository
from ballast.runtime.app import create_app
from ballast.runtime.engine import _reset_ballast_for_tests
from ballast.runtime.event_stream import (
    EventStream,
    InProcessEventStream,
)


class TestEngine:
    """Pre-wired Stateflow app for tests."""

    # Prevent pytest from trying to collect this as a test class.
    __test__ = False

    def __init__(
        self,
        *,
        thread_repo: ThreadRepository | None = None,
        event_log: EventLogRepository | None = None,
        event_stream: EventStream | None = None,
        dbos: Any | None = None,
        manage_dbos_lifecycle: bool = False,
        extra_routers: Sequence[APIRouter] = (),
    ) -> None:
        self._thread_repo = thread_repo or InMemoryThreadRepository()
        self._event_log = event_log or InMemoryEventLogRepository()
        self._event_stream = event_stream or InProcessEventStream()
        self._dbos = dbos
        self._manage_dbos = manage_dbos_lifecycle
        self._extra_routers: list[APIRouter] = list(extra_routers)
        # Per-instance overrides applied to FastAPI dependency_overrides.
        self._overrides: dict[Any, Any] = {}
        self._app: FastAPI | None = None
        self._client: TestClient | None = None
        self._tempdir: Path | None = None

    # -- Construction --

    @classmethod
    def default(cls) -> TestEngine:
        """Construct a TestEngine with InMemory repos and no extra routers.

        DBOS is NOT launched by default — tests that need durable
        workflows should construct with ``dbos=DBOSConfig(...)`` and
        ``manage_dbos_lifecycle=True``.
        """
        return cls()

    # -- Override semantics --

    def override(self, target: type, replacement: Any) -> TestEngine:
        """Override a repo / event-log / event-stream with a test double.

        Supported targets (resolved via ``app.dependency_overrides``):
        - ``ThreadRepository`` subclass → replaces the value used by
          ``Depends(get_thread_repo)``.
        - ``EventLogRepository`` subclass → ``Depends(get_event_log)``.
        - ``EventStream`` subclass → ``Depends(get_event_stream)``.

        Returns ``self`` for chaining.
        """
        from ballast.persistence.events.repository import (
            EventLogRepository as _EvLog,
        )

        if isinstance(replacement, ThreadRepository) and issubclass(target, ThreadRepository):
            self._thread_repo = replacement
            self._overrides[get_thread_repo] = lambda: replacement
        elif isinstance(replacement, _EvLog) and issubclass(target, _EvLog):
            self._event_log = replacement
            self._overrides[get_event_log] = lambda: replacement
        elif isinstance(replacement, EventStream) and issubclass(target, EventStream):
            self._event_stream = replacement
            self._overrides[get_event_stream] = lambda: replacement
        else:
            raise TypeError(
                f"Cannot override {target.__name__}: only ThreadRepository / "
                f"EventLogRepository / EventStream are supported. The "
                f"framework no longer manages a workflow / agent registry — "
                f"tests that need to swap an agent should construct the app "
                f"themselves with the test double.",
            )
        return self

    # -- Lifecycle --

    def test_client(self) -> TestEngine:
        """Return self — TestEngine IS its own context manager.

        Use:
            with engine.test_client() as client:
                client.post(...)
        """
        return self

    def __enter__(self) -> TestClient:
        # Build a per-instance DBOS config if requested but not provided.
        dbos_config = self._dbos
        if dbos_config is None and self._manage_dbos:
            from dbos import DBOSConfig
            tempdir = Path(tempfile.mkdtemp(prefix="stateflow-test-"))
            self._tempdir = tempdir
            dbos_config = DBOSConfig(
                name=f"stateflow-test-{uuid4().hex[:8]}",
                system_database_url=f"sqlite:///{tempdir / 'dbos.sqlite'}",
            )

        _reset_observability_for_tests()
        _reset_ballast_for_tests()
        self._app = create_app(
            thread_repo=self._thread_repo,
            event_log=self._event_log,
            event_stream=self._event_stream,
            dbos=dbos_config,
            manage_dbos_lifecycle=self._manage_dbos,
            extra_routers=self._extra_routers,
        )
        # Apply overrides AFTER create_app so they win over the defaults.
        for key, value in self._overrides.items():
            self._app.dependency_overrides[key] = value
        self._client = TestClient(self._app)
        self._client.__enter__()
        return self._client

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            self._client.__exit__(exc_type, exc, tb)
            self._client = None
        self._app = None
        _reset_ballast_for_tests()
        if self._tempdir is not None:
            import shutil
            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None


__all__ = ["TestEngine"]
