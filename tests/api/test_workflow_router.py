"""Unit tests for ``build_workflow_router``."""
from __future__ import annotations

from uuid import uuid4

import pytest
from dbos import DBOSConfiguredInstance
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from pydantic_ai_stateflow.api.workflow_router import (
    WorkflowStartResponse,
    build_workflow_router,
)
from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.runtime.workflows import (
    clear_workflow_registry,
    workflow,
)


class _In(BaseModel):
    topic: str


class _Out(BaseModel):
    result: str


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_workflow_registry()
    yield
    clear_workflow_registry()


def _make_app_with_instance(instance: object) -> FastAPI:
    """Wrap an app+state setup that mimics what create_app will do."""
    app = FastAPI()
    name = type(instance)._sf_workflow_name  # type: ignore[attr-defined]
    app.state.workflows = {name: instance}
    router = build_workflow_router(instance)
    app.include_router(router)
    return app


def test_undecorated_class_raises(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor

    class Bare(DBOSConfiguredInstance):
        def __init__(self) -> None:
            super().__init__(config_name=f"bare-{uuid4()}")

    instance = Bare()
    with pytest.raises(TypeError, match="not @sf.workflow-decorated"):
        build_workflow_router(instance)


def test_default_workflow_id_is_deterministic_per_input(
    fresh_dbos_executor: None,
) -> None:
    """Same input → same workflow id; different input → different id."""
    del fresh_dbos_executor
    from pydantic_ai_stateflow.api.workflow_router import _default_workflow_id

    a = _default_workflow_id("flow", _In(topic="x"))
    b = _default_workflow_id("flow", _In(topic="x"))
    c = _default_workflow_id("flow", _In(topic="y"))
    assert a == b
    assert a != c
    assert a.startswith("flow:")


def test_route_path_is_kebab_name(fresh_dbos_executor: None) -> None:
    """``MyFlow`` → ``POST /workflows/my-flow``."""
    del fresh_dbos_executor

    @workflow(input=_In, output=_Out)
    class MyFlow(DBOSConfiguredInstance):
        def __init__(self) -> None:
            super().__init__(config_name=f"my-flow-{uuid4()}")

        @Durable.workflow()
        async def run(self, input: _In) -> _Out:
            return _Out(result=f"done:{input.topic}")

    instance = MyFlow()
    app = _make_app_with_instance(instance)
    with TestClient(app) as client:
        r = client.post("/workflows/my-flow", json={"topic": "hi"})
        assert r.status_code == 200, r.text
        body = r.json()
        # Fire-and-forget: response is WorkflowStartResponse shape.
        assert "workflow_id" in body
        assert "started_at" in body
        assert body["workflow_id"].startswith("my-flow:")


def test_blocking_mode_returns_output_model(
    fresh_dbos_executor: None,
) -> None:
    del fresh_dbos_executor

    @workflow(input=_In, output=_Out, blocking=True)
    class BlockingFlow(DBOSConfiguredInstance):
        def __init__(self) -> None:
            super().__init__(config_name=f"blocking-flow-{uuid4()}")

        @Durable.workflow()
        async def run(self, input: _In) -> _Out:
            return _Out(result=f"blocking:{input.topic}")

    instance = BlockingFlow()
    app = _make_app_with_instance(instance)
    with TestClient(app) as client:
        r = client.post("/workflows/blocking-flow", json={"topic": "x"})
        assert r.status_code == 200, r.text
        assert r.json() == {"result": "blocking:x"}


def test_custom_workflow_id_classmethod_honored(
    fresh_dbos_executor: None,
) -> None:
    del fresh_dbos_executor

    @workflow(input=_In, output=_Out)
    class CustomIdFlow(DBOSConfiguredInstance):
        def __init__(self) -> None:
            super().__init__(config_name=f"custom-id-flow-{uuid4()}")

        @staticmethod
        def workflow_id(input: _In) -> str:
            return f"custom-{input.topic}"

        @Durable.workflow()
        async def run(self, input: _In) -> _Out:
            return _Out(result=input.topic)

    instance = CustomIdFlow()
    app = _make_app_with_instance(instance)
    with TestClient(app) as client:
        r = client.post("/workflows/custom-id-flow", json={"topic": "abc"})
        assert r.status_code == 200, r.text
        assert r.json()["workflow_id"] == "custom-abc"


def test_invalid_body_returns_422(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor

    @workflow(input=_In, output=_Out)
    class StrictFlow(DBOSConfiguredInstance):
        def __init__(self) -> None:
            super().__init__(config_name=f"strict-flow-{uuid4()}")

        @Durable.workflow()
        async def run(self, input: _In) -> _Out:
            return _Out(result="x")

    instance = StrictFlow()
    app = _make_app_with_instance(instance)
    with TestClient(app) as client:
        r = client.post("/workflows/strict-flow", json={"wrong_field": "x"})
        assert r.status_code == 422
