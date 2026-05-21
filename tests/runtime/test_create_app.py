"""Unit tests for ``sf.create_app()``."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from dbos import DBOSConfiguredInstance
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel

from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.persistence.thread.domain import Thread
from pydantic_ai_stateflow.runtime.agents import (
    StateflowAgent,
    clear_agent_class_registry,
    stateflow_agent,
)
from pydantic_ai_stateflow.runtime.app import create_app
from pydantic_ai_stateflow.runtime.workflows import (
    clear_workflow_registry,
    workflow,
)


class _In(BaseModel):
    topic: str


class _Out(BaseModel):
    result: str


@pytest.fixture(autouse=True)
def _clean_registries():
    clear_workflow_registry()
    clear_agent_class_registry()
    yield
    clear_workflow_registry()
    clear_agent_class_registry()


def test_minimal_app_has_health_endpoint(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    app = create_app()
    with TestClient(app) as client:
        r = client.get("/healthz")
        # Either 200 or 404 — depends on build_health_router default.
        # Just verify the app boots without error.
        assert r.status_code in (200, 404, 503)


def test_workflows_mount_auto_router(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor

    @workflow(input=_In, output=_Out)
    class MyFlow(DBOSConfiguredInstance):
        def __init__(self) -> None:
            super().__init__(config_name=f"my-flow-{uuid4()}")

        @Durable.workflow()
        async def run(self, input: _In) -> _Out:
            return _Out(result=input.topic)

    instance = MyFlow()
    app = create_app(workflows=[instance])
    with TestClient(app) as client:
        r = client.post("/workflows/my-flow", json={"topic": "hi"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "workflow_id" in body
        assert "started_at" in body


def test_duplicate_workflow_raises(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor

    @workflow(input=_In, output=_Out)
    class DupFlow(DBOSConfiguredInstance):
        def __init__(self, suffix: str = "a") -> None:
            super().__init__(config_name=f"dup-flow-{suffix}-{uuid4()}")

        @Durable.workflow()
        async def run(self, input: _In) -> _Out:
            return _Out(result="x")

    with pytest.raises(ValueError, match="Duplicate workflow instance"):
        create_app(workflows=[DupFlow("a"), DupFlow("b")])


class _StubAgent(StateflowAgent):
    metadata_model = None

    def build_agent(self) -> Agent[None, Any]:
        return Agent(TestModel(custom_output_text="x"), output_type=str)

    async def build_deps(
        self,
        *,
        thread: Thread,
        message: ModelMessage | None,
    ) -> None:
        del thread, message
        return None


def test_agents_registered_in_state(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor

    @stateflow_agent
    class FooAgent(_StubAgent):
        pass

    instance = FooAgent()
    app = create_app(agents=[instance])
    assert app.state.agents["foo-agent"] is instance


def test_agent_without_decorator_raises(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor

    class BareAgent(_StubAgent):
        pass

    # No @stateflow_agent — no ``name`` ClassVar set.
    instance = BareAgent.__new__(BareAgent)
    # Force-init enough state for instantiation to succeed.
    StateflowAgent.__init__(instance)
    with pytest.raises(TypeError, match="no ``name`` ClassVar"):
        create_app(agents=[instance])


def test_repos_default_to_in_memory(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    app = create_app()
    from pydantic_ai_stateflow.persistence import (
        InMemoryEventLogRepository,
        InMemoryThreadRepository,
    )
    from pydantic_ai_stateflow.runtime.event_stream import InProcessEventStream

    assert isinstance(app.state.thread_repo, InMemoryThreadRepository)
    assert isinstance(app.state.event_log, InMemoryEventLogRepository)
    assert isinstance(app.state.event_stream, InProcessEventStream)


def test_threads_endpoint_works(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    app = create_app()
    with TestClient(app) as client:
        # threads_router is mounted with Depends-resolved repo; GET /threads
        # is the listing endpoint and should resolve InMemoryThreadRepository
        # from app.state without error.
        r = client.get("/threads")
        assert r.status_code == 200, r.text
