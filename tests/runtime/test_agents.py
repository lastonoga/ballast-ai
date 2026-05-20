"""``StateflowAgent`` ABC + registry + ``validate_thread_metadata``."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_stateflow.runtime import (
    StateflowAgent,
    clear_agent_registry,
    get_agent,
    list_agents,
    register_agent,
    validate_thread_metadata,
)
from pydantic_ai_stateflow.runtime.agents import _resolve_agent_name


class _NoMetaAgent(StateflowAgent):
    name = "no_meta"

    def build_agent(self) -> Agent[None, str]:
        return Agent(TestModel(custom_output_text="ok"), output_type=str)

    async def build_deps(self, *, thread: Any, tenant_id: UUID, message: Any) -> None:
        del thread, tenant_id, message
        return None


class _NotesMetadata(BaseModel):
    relations: dict[str, str] = {}
    context: dict[str, Any] = {}


class _WithMetaAgent(StateflowAgent):
    name = "with_meta"
    metadata_model = _NotesMetadata

    def build_agent(self) -> Agent[None, str]:
        return Agent(TestModel(custom_output_text="ok"), output_type=str)

    async def build_deps(self, *, thread: Any, tenant_id: UUID, message: Any) -> None:
        del thread, tenant_id, message
        return None


@pytest.fixture(autouse=True)
def _isolated_registry() -> Any:
    clear_agent_registry()
    yield
    clear_agent_registry()


def test_register_and_get() -> None:
    instance = _NoMetaAgent()
    register_agent(instance)
    assert get_agent("no_meta") is instance


def test_register_is_idempotent_overwrites() -> None:
    first = _NoMetaAgent()
    second = _NoMetaAgent()
    register_agent(first)
    register_agent(second)
    assert get_agent("no_meta") is second


def test_get_agent_unknown_raises_with_helpful_message() -> None:
    register_agent(_NoMetaAgent())
    with pytest.raises(KeyError) as exc_info:
        get_agent("mystery")
    msg = str(exc_info.value)
    assert "mystery" in msg
    assert "no_meta" in msg
    assert "register_agent" in msg


def test_list_agents_returns_snapshot_sorted() -> None:
    register_agent(_WithMetaAgent())
    register_agent(_NoMetaAgent())
    names = [type(a).name for a in list_agents()]
    assert names == ["no_meta", "with_meta"]


def test_register_rejects_empty_name() -> None:
    class _Bad(StateflowAgent):
        name = ""

        def build_agent(self) -> Agent[None, str]:
            return Agent(TestModel(), output_type=str)

        async def build_deps(self, **_kw: Any) -> None:
            return None

    with pytest.raises(ValueError, match="non-empty ClassVar"):
        register_agent(_Bad())


def test_resolve_agent_name_accepts_str_class_and_instance() -> None:
    instance = _NoMetaAgent()
    assert _resolve_agent_name("no_meta") == "no_meta"
    assert _resolve_agent_name(_NoMetaAgent) == "no_meta"
    assert _resolve_agent_name(instance) == "no_meta"


def test_resolve_agent_name_rejects_garbage() -> None:
    with pytest.raises(TypeError, match="AgentRef"):
        _resolve_agent_name(42)


def test_validate_metadata_passthrough_when_no_model() -> None:
    register_agent(_NoMetaAgent())
    out = validate_thread_metadata("no_meta", {"anything": "goes", "n": 1})
    assert out == {"anything": "goes", "n": 1}


def test_validate_metadata_round_trips_through_model() -> None:
    register_agent(_WithMetaAgent())
    out = validate_thread_metadata(
        _WithMetaAgent,
        {"relations": {"workflow": "W-1"}, "context": {"lang": "en"}},
    )
    assert out == {"relations": {"workflow": "W-1"}, "context": {"lang": "en"}}


def test_validate_metadata_rejects_bad_shape() -> None:
    register_agent(_WithMetaAgent())
    with pytest.raises(ValidationError):
        validate_thread_metadata("with_meta", {"relations": "not-a-dict"})


def test_validate_metadata_normalizes_none_to_empty() -> None:
    register_agent(_NoMetaAgent())
    assert validate_thread_metadata("no_meta", None) == {}


def test_lazy_agent_property_caches() -> None:
    instance = _NoMetaAgent()
    register_agent(instance)
    a1 = instance.agent
    a2 = instance.agent
    assert a1 is a2  # cached_property
    # Sanity: it's a real pydantic-ai Agent
    assert hasattr(a1, "run")


def test_clear_agent_registry() -> None:
    register_agent(_NoMetaAgent())
    register_agent(_WithMetaAgent())
    assert len(list_agents()) == 2
    clear_agent_registry()
    assert list_agents() == []
    with pytest.raises(KeyError):
        get_agent("no_meta")


def test_default_model_settings_is_none() -> None:
    instance = _NoMetaAgent()
    assert instance.model_settings() is None


def test_validate_metadata_unknown_agent_raises() -> None:
    with pytest.raises(KeyError):
        validate_thread_metadata("nobody", {})


def test_metadata_unknown_field_raises_when_model_is_strict() -> None:
    """Default pydantic config allows extras; this just documents that
    extra-field strictness is the metadata_model's responsibility."""
    register_agent(_WithMetaAgent())
    # No assertion error — extras pass through silently with default config.
    out = validate_thread_metadata("with_meta", {"extra": "x"})
    assert "extra" not in out  # dropped during round-trip dump


def test_metadata_does_not_break_when_subclass_has_no_extra_tools() -> None:
    """Smoke: an agent with zero ``@tool`` decorations still builds."""
    instance = _NoMetaAgent()
    a = instance.agent
    assert a._function_toolset.tools == {}  # noqa: SLF001
