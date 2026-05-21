"""``TestEngine`` — pre-wired Stateflow app for tests.

API:

    engine = TestEngine.default()
    engine.override(NotesAgent, MockAgent.with_output("hi"))
    engine.override(BrainstormFlow, MockFlow.returning(BrainstormOutcome(...)))
    with engine.test_client() as client:
        r = client.post("/threads", json={})

What it does:
- Builds a fresh in-memory app with no workflows/agents by default.
- ``override(target, replacement)`` swaps an instance via
  ``app.dependency_overrides[get_X]`` (FastAPI native).
- ``test_client()`` returns a ``TestClient`` context manager that
  runs FastAPI lifespan (and DBOS lifecycle if configured).
- Unique DBOS ``name`` per TestEngine instance avoids in-process collisions.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.api.deps import (
    get_event_log,
    get_event_stream,
    get_thread_repo,
)
from pydantic_ai_stateflow.observability.config import (
    _reset_observability_for_tests,
)
from pydantic_ai_stateflow.persistence import (
    EventLogRepository,
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository
from pydantic_ai_stateflow.runtime.agents import StateflowAgent
from pydantic_ai_stateflow.runtime.app import create_app
from pydantic_ai_stateflow.runtime.event_stream import (
    EventStream,
    InProcessEventStream,
)
from pydantic_ai_stateflow.runtime.workflows import workflow_metadata


class TestEngine:
    """Pre-wired Stateflow app for tests.

    Lifecycle owned by ``test_client()``: enter the context to launch
    DBOS + run lifespan, exit to tear down. Overrides applied before
    entry take effect; overrides applied after entry are visible to
    subsequent requests on the same client.
    """

    # Prevent pytest from trying to collect this as a test class
    # (the leading "Test" prefix would otherwise trigger collection).
    __test__ = False

    def __init__(
        self,
        *,
        workflows: list[object] | None = None,
        agents: list[StateflowAgent] | None = None,
        thread_repo: ThreadRepository | None = None,
        event_log: EventLogRepository | None = None,
        event_stream: EventStream | None = None,
        dbos: Any | None = None,
        manage_dbos_lifecycle: bool = False,
    ) -> None:
        self._workflows: list[object] = list(workflows or [])
        self._agents: list[StateflowAgent] = list(agents or [])
        self._thread_repo = thread_repo or InMemoryThreadRepository()
        self._event_log = event_log or InMemoryEventLogRepository()
        self._event_stream = event_stream or InProcessEventStream()
        self._dbos = dbos
        self._manage_dbos = manage_dbos_lifecycle
        # Per-instance overrides applied to FastAPI dependency_overrides.
        self._overrides: dict[Any, Any] = {}
        # Workflow / agent overrides keyed by kebab-name. Applied by
        # mutating ``app.state.{workflows,agents}[name]`` after build —
        # the auto-generated route's ``Depends(get_workflow_instance(...))``
        # closes over a freshly-minted resolver, so ``dependency_overrides``
        # can't reach it.
        self._workflow_overrides: dict[str, object] = {}
        self._agent_overrides: dict[str, StateflowAgent] = {}
        # Built once at __enter__ to keep dependency_overrides assignments
        # consistent across the same lifespan window.
        self._app: FastAPI | None = None
        self._client: TestClient | None = None
        self._tempdir: Path | None = None

    # -- Construction --

    @classmethod
    def default(cls) -> TestEngine:
        """Construct a TestEngine with no workflows/agents and InMemory repos.

        DBOS is NOT launched by default — tests that need durable
        workflows should construct with ``dbos=DBOSConfig(...)`` and
        ``manage_dbos_lifecycle=True``.
        """
        return cls()

    # -- Override semantics --

    def override(self, target: type, replacement: Any) -> TestEngine:
        """Override a class registration with a replacement instance.

        Supported targets:
        - ``ThreadRepository`` subclass / ``EventLogRepository`` /
          ``EventStream`` → replaces the value used by ``Depends(get_X)``.
        - A ``StateflowAgent`` subclass → replaces the agent instance in
          the registry under its kebab-name.
        - A ``@sf.workflow`` class → replaces the workflow instance
          under its kebab-name.

        Returns ``self`` for chaining.
        """
        from pydantic_ai_stateflow.persistence.events.repository import (
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
        elif isinstance(target, type) and issubclass(target, StateflowAgent):
            name = getattr(target, "name", None)
            if name is None:
                raise TypeError(
                    f"Cannot override agent class {target.__name__}: "
                    f"no ``name`` ClassVar — was @sf.stateflow_agent applied?",
                )
            self._agent_overrides[name] = replacement
        else:
            # Workflow override: assume @sf.workflow-decorated class.
            try:
                name, _in, _out, _blocking = workflow_metadata(target)
            except TypeError as exc:
                raise TypeError(
                    f"Cannot override {target.__name__}: not a recognized "
                    f"repo / EventStream / StateflowAgent / @sf.workflow class",
                ) from exc
            # Don't mutate ``self._workflows`` — ``create_app`` reads it to
            # mount the auto-generated route, which requires real
            # ``workflow_metadata``. Stash the override and apply by
            # rewriting ``app.state.workflows[name]`` post-build.
            self._workflow_overrides[name] = replacement
        return self

    # -- Lifecycle --

    def test_client(self) -> TestEngine:
        """Return self — TestEngine IS its own context manager.

        Entering: builds the app, applies overrides, launches DBOS
        (if configured), starts the FastAPI lifespan via TestClient.

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
        self._app = create_app(
            workflows=self._workflows,
            agents=self._agents,
            thread_repo=self._thread_repo,
            event_log=self._event_log,
            event_stream=self._event_stream,
            dbos=dbos_config,
            manage_dbos_lifecycle=self._manage_dbos,
        )
        # Apply overrides AFTER create_app so they win over the defaults.
        for key, value in self._overrides.items():
            self._app.dependency_overrides[key] = value
        # Apply workflow / agent overrides by rewriting app.state — the
        # auto-generated route closes over a fresh ``get_workflow_instance``
        # resolver that ``dependency_overrides`` can't reach.
        for name, wf_replacement in self._workflow_overrides.items():
            self._app.state.workflows[name] = wf_replacement
        for name, ag_replacement in self._agent_overrides.items():
            self._app.state.agents[name] = ag_replacement
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
        if self._tempdir is not None:
            import shutil
            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None


__all__ = ["TestEngine"]
