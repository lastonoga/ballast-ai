"""Unit tests for ``@sf.stateflow_agent`` decorator."""
from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel

from pydantic_ai_stateflow.persistence.thread.domain import Thread
from pydantic_ai_stateflow.runtime.agents import (
    StateflowAgent,
    clear_agent_class_registry,
    get_agent_class,
    list_agent_classes,
    stateflow_agent,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_agent_class_registry()
    yield
    clear_agent_class_registry()


class _StubAgent(StateflowAgent):
    """Common test fixture so subclasses are concrete."""
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


def test_kebab_name_auto_derived() -> None:
    @stateflow_agent
    class NotesAgent(_StubAgent):
        pass

    assert NotesAgent.name == "notes-agent"
    assert get_agent_class("notes-agent") is NotesAgent


def test_explicit_class_name_wins_when_set() -> None:
    @stateflow_agent
    class MyAgent(_StubAgent):
        name = "explicit"

    assert MyAgent.name == "explicit"
    assert get_agent_class("explicit") is MyAgent


def test_duplicate_name_raises() -> None:
    @stateflow_agent
    class DupAgent(_StubAgent):
        pass

    with pytest.raises(ValueError, match="Duplicate @sf.stateflow_agent"):
        @stateflow_agent
        class DupAgent(_StubAgent):  # noqa: F811 — intentional duplicate
            pass


def test_xml_acronym_kebab() -> None:
    @stateflow_agent
    class XMLParserAgent(_StubAgent):
        pass

    assert XMLParserAgent.name == "xml-parser-agent"


def test_registry_listing() -> None:
    @stateflow_agent
    class FooAgent(_StubAgent):
        pass

    @stateflow_agent
    class BarAgent(_StubAgent):
        pass

    cls_map = list_agent_classes()
    assert cls_map["foo-agent"] is FooAgent
    assert cls_map["bar-agent"] is BarAgent
