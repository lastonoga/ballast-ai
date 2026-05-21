"""Unit tests for ``@sf.workflow`` decorator + workflow registry."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.runtime.workflows import (
    clear_workflow_registry,
    get_workflow_class,
    list_workflow_classes,
    workflow,
    workflow_metadata,
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


def test_kebab_name_auto_derived() -> None:
    @workflow(input=_In, output=_Out)
    class BrainstormFlow:
        async def run(self, input: _In) -> _Out:
            return _Out(result=input.topic)

    name, in_t, out_t, blocking = workflow_metadata(BrainstormFlow)
    assert name == "brainstorm-flow"
    assert in_t is _In
    assert out_t is _Out
    assert blocking is False


def test_explicit_name_override() -> None:
    @workflow(name="custom-name", input=_In, output=_Out)
    class MyFlow:
        async def run(self, input: _In) -> _Out:
            return _Out(result="x")

    assert workflow_metadata(MyFlow)[0] == "custom-name"


def test_xml_acronym_kebab() -> None:
    @workflow(input=_In, output=_Out)
    class MyXMLFlow:
        async def run(self, input: _In) -> _Out:
            return _Out(result="x")

    assert workflow_metadata(MyXMLFlow)[0] == "my-xml-flow"


def test_blocking_flag() -> None:
    @workflow(input=_In, output=_Out, blocking=True)
    class BlockingFlow:
        async def run(self, input: _In) -> _Out:
            return _Out(result="x")

    assert workflow_metadata(BlockingFlow)[3] is True


def test_missing_input_output_raises() -> None:
    with pytest.raises(TypeError, match="input= and output= are required"):
        @workflow(input=_In)  # type: ignore[call-overload]
        class _Bad:
            async def run(self, input: _In) -> _Out: ...


def test_missing_run_method_raises() -> None:
    with pytest.raises(TypeError, match="must define ``async def run"):
        @workflow(input=_In, output=_Out)
        class _NoRun:
            pass


def test_duplicate_name_raises() -> None:
    @workflow(input=_In, output=_Out)
    class DupFlow:
        async def run(self, input: _In) -> _Out:
            return _Out(result="a")

    with pytest.raises(ValueError, match="Duplicate @sf.workflow name"):
        @workflow(name="dup-flow", input=_In, output=_Out)
        class _Other:
            async def run(self, input: _In) -> _Out:
                return _Out(result="b")


def test_instance_metadata_lookup() -> None:
    @workflow(input=_In, output=_Out)
    class FetchFlow:
        async def run(self, input: _In) -> _Out:
            return _Out(result="x")

    instance = FetchFlow()
    assert workflow_metadata(instance)[0] == "fetch-flow"


def test_workflow_metadata_on_undecorated_class_raises() -> None:
    class Bare:
        pass

    with pytest.raises(TypeError, match="not @sf.workflow-decorated"):
        workflow_metadata(Bare)


def test_registry_listing() -> None:
    @workflow(input=_In, output=_Out)
    class A:
        async def run(self, input: _In) -> _Out:
            return _Out(result="a")

    @workflow(input=_In, output=_Out)
    class B:
        async def run(self, input: _In) -> _Out:
            return _Out(result="b")

    cls_map = list_workflow_classes()
    assert "a" in cls_map and "b" in cls_map
    assert get_workflow_class("a") is A
