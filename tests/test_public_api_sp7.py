"""SP7 exports smoke (post SP1 T11 cleanup)."""
from __future__ import annotations

import ballast


def test_sp7_exports_present():
    assert hasattr(ballast, "ObservabilityConfig")
    assert hasattr(ballast, "has_logfire")
    assert hasattr(ballast, "traced")
    assert hasattr(ballast, "build_a2a_router")
    assert hasattr(ballast, "build_health_router")
    assert hasattr(ballast, "create_app")
    assert hasattr(ballast, "A2AAgentAdapter")
    assert hasattr(ballast, "AgentCard")
    assert hasattr(ballast, "DepsFactory")
    assert hasattr(ballast, "extract_text")
    assert hasattr(ballast, "messages_to_model_history")
    assert hasattr(ballast, "Dataset")
    assert hasattr(ballast, "EvalCase")
    assert hasattr(ballast, "EvalReport")
    assert hasattr(ballast, "EvalRunOutput")
    assert hasattr(ballast, "SchemaAdherenceScorer")
    assert hasattr(ballast, "ScoreResult")
    assert hasattr(ballast, "Scorer")
