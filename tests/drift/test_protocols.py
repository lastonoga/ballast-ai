"""DriftCheckSignal + DriftContext vehicle dataclasses."""
from __future__ import annotations

from ballast.drift._protocols import DriftCheckSignal, DriftContext


def test_signal_holds_per_step_counters() -> None:
    s = DriftCheckSignal(
        step_index=3, tool_calls=5, tokens_used=1200, seconds_elapsed=42.5,
    )
    assert s.step_index == 3
    assert s.tool_calls == 5
    assert s.tokens_used == 1200
    assert s.seconds_elapsed == 42.5


def test_context_defaults_metadata_to_empty_dict() -> None:
    c = DriftContext(messages=[], run_ctx=None, workflow_input=None)
    assert c.messages == []
    assert c.run_ctx is None
    assert c.workflow_input is None
    assert c.metadata == {}


def test_context_preserves_explicit_metadata() -> None:
    c = DriftContext(
        messages=[], run_ctx=None, workflow_input={"x": 1},
        metadata={"budget": {"input_tokens": 100}},
    )
    assert c.workflow_input == {"x": 1}
    assert c.metadata["budget"]["input_tokens"] == 100
