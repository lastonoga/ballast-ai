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


def test_drift_check_strategy_runtime_checkable() -> None:
    from ballast.drift._protocols import DriftCheckStrategy

    class _Stub:
        def should_check(self, signal):
            return True
    assert isinstance(_Stub(), DriftCheckStrategy)

    class _Missing:
        pass
    assert not isinstance(_Missing(), DriftCheckStrategy)


def test_trace_window_runtime_checkable() -> None:
    from ballast.drift._protocols import TraceWindow

    class _Stub:
        async def slice(self, ctx):
            return []
    assert isinstance(_Stub(), TraceWindow)


def test_goal_source_runtime_checkable() -> None:
    from ballast.drift._protocols import GoalSource

    class _Stub:
        async def goal(self, ctx):
            return ""
    assert isinstance(_Stub(), GoalSource)


def test_prompt_builder_runtime_checkable() -> None:
    from ballast.drift._protocols import PromptBuilder

    class _Stub:
        def build(self, goal, trace):
            return ""
    assert isinstance(_Stub(), PromptBuilder)


def test_drift_handler_runtime_checkable() -> None:
    from ballast.drift._protocols import DriftHandler

    class _Stub:
        async def handle(self, verdict, ctx):
            return None
    assert isinstance(_Stub(), DriftHandler)
