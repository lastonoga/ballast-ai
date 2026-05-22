"""``StateflowAgent`` ABC + ``validate_thread_metadata``.

The framework no longer maintains an agent class registry —
``Thread.agent`` is an opaque app-owned string. Apps that want
metadata validation pass the class (or an instance) to
``validate_thread_metadata`` directly.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_stateflow.runtime import StateflowAgent, validate_thread_metadata


class _NotesMetadata(BaseModel):
    relations: dict[str, str] = {}
    context: dict[str, Any] = {}


def _build_agents() -> tuple[type[StateflowAgent], type[StateflowAgent]]:
    class _NoMetaAgent(StateflowAgent):
        name = "no-meta"

        def build_agent(self) -> Agent[None, str]:
            return Agent(TestModel(custom_output_text="ok"), output_type=str)

        async def build_deps(
            self, *, thread: Any, tenant_id: UUID, message: Any,
        ) -> None:
            del thread, tenant_id, message
            return None

    class _WithMetaAgent(StateflowAgent):
        name = "with-meta"
        metadata_model = _NotesMetadata

        def build_agent(self) -> Agent[None, str]:
            return Agent(TestModel(custom_output_text="ok"), output_type=str)

        async def build_deps(
            self, *, thread: Any, tenant_id: UUID, message: Any,
        ) -> None:
            del thread, tenant_id, message
            return None

    return _NoMetaAgent, _WithMetaAgent


def test_validate_metadata_passthrough_when_no_model() -> None:
    no_meta, _ = _build_agents()
    out = validate_thread_metadata(no_meta, {"anything": "goes", "n": 1})
    assert out == {"anything": "goes", "n": 1}


def test_validate_metadata_round_trips_through_model() -> None:
    _, with_meta = _build_agents()
    out = validate_thread_metadata(
        with_meta,
        {"relations": {"workflow": "W-1"}, "context": {"lang": "en"}},
    )
    assert out == {"relations": {"workflow": "W-1"}, "context": {"lang": "en"}}


def test_validate_metadata_rejects_bad_shape() -> None:
    _, with_meta = _build_agents()
    with pytest.raises(ValidationError):
        validate_thread_metadata(with_meta, {"relations": "not-a-dict"})


def test_validate_metadata_normalizes_none_to_empty() -> None:
    no_meta, _ = _build_agents()
    assert validate_thread_metadata(no_meta, None) == {}


def test_validate_metadata_accepts_class_or_instance() -> None:
    no_meta, _ = _build_agents()
    payload = {"k": "v"}
    assert validate_thread_metadata(no_meta, payload) == payload
    assert validate_thread_metadata(no_meta(), payload) == payload


def test_validate_metadata_rejects_garbage_ref() -> None:
    with pytest.raises(TypeError, match="AgentRef"):
        validate_thread_metadata(42, {})  # type: ignore[arg-type]


def test_metadata_unknown_field_dropped_during_round_trip() -> None:
    _, with_meta = _build_agents()
    out = validate_thread_metadata(with_meta, {"extra": "x"})
    assert "extra" not in out  # extras dropped by default pydantic config


def test_lazy_agent_property_caches() -> None:
    no_meta, _ = _build_agents()
    instance = no_meta()
    a1 = instance.agent
    a2 = instance.agent
    assert a1 is a2  # cached_property
    assert hasattr(a1, "run")


def test_default_model_settings_is_none() -> None:
    no_meta, _ = _build_agents()
    assert no_meta().model_settings() is None


def test_no_tools_subclass_still_builds() -> None:
    no_meta, _ = _build_agents()
    a = no_meta().agent
    assert a._function_toolset.tools == {}  # noqa: SLF001
