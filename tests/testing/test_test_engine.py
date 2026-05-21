"""Unit tests for ``TestEngine``."""
from __future__ import annotations

from uuid import uuid4

import pytest
from dbos import DBOSConfiguredInstance
from pydantic import BaseModel

from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.observability.config import (
    _reset_observability_for_tests,
)
from pydantic_ai_stateflow.persistence import InMemoryThreadRepository
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository
from pydantic_ai_stateflow.runtime.agents import clear_agent_class_registry
from pydantic_ai_stateflow.runtime.workflows import (
    clear_workflow_registry,
    workflow,
)
from pydantic_ai_stateflow.testing import MockFlow, TestEngine


class _In(BaseModel):
    topic: str


class _Out(BaseModel):
    result: str


@pytest.fixture(autouse=True)
def _clean_registries():
    clear_workflow_registry()
    clear_agent_class_registry()
    _reset_observability_for_tests()
    yield
    clear_workflow_registry()
    clear_agent_class_registry()
    _reset_observability_for_tests()


def test_default_engine_boots(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    with TestEngine.default().test_client() as client:
        r = client.get("/threads")
        assert r.status_code == 200, r.text


def test_override_thread_repo(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    engine = TestEngine.default()
    custom = InMemoryThreadRepository()
    engine.override(ThreadRepository, custom)
    with engine.test_client() as client:
        r = client.get("/threads")
        assert r.status_code == 200, r.text


def test_override_workflow_with_mock(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor

    @workflow(input=_In, output=_Out)
    class MyFlow(DBOSConfiguredInstance):
        def __init__(self) -> None:
            super().__init__(config_name=f"my-flow-{uuid4()}")

        @Durable.workflow()
        async def run(self, input: _In) -> _Out:
            return _Out(result=f"real:{input.topic}")

    real_instance = MyFlow()
    # Provide the real instance so HTTP autogen mounts, then override.
    engine = TestEngine(workflows=[real_instance])
    mock = MockFlow.returning(_Out(result="mocked"))
    # We're overriding by class, so the override resolver maps the
    # workflow's kebab-name to the mock.
    engine.override(MyFlow, mock)
    with engine.test_client() as client:
        r = client.post("/workflows/my-flow", json={"topic": "x"})
        assert r.status_code == 200, r.text
        # The mock's run returns the _Out directly; auto-route is fire-
        # and-forget by default so we get WorkflowStartResponse, NOT
        # the mock's return value. So we just check the call landed.
        # If the test wanted the body, the workflow should be blocking.
        # For now: assert mock was called.
    assert mock.calls, "mock workflow.run should have been called"


def test_engine_cleans_up_on_exit(fresh_dbos_executor: None) -> None:
    """No DBOS state leaks across TestEngine context boundaries in same process."""
    del fresh_dbos_executor
    e1 = TestEngine.default()
    with e1.test_client() as c1:
        assert c1.get("/threads").status_code == 200
    e2 = TestEngine.default()
    with e2.test_client() as c2:
        assert c2.get("/threads").status_code == 200


def test_override_chains(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    engine = TestEngine.default()
    repo = InMemoryThreadRepository()
    result = engine.override(ThreadRepository, repo)
    assert result is engine
