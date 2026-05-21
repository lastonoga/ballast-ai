"""SP7 exports smoke (post SP1 T11 cleanup)."""
from __future__ import annotations

import pydantic_ai_stateflow as sf


def test_sp7_exports_present():
    assert hasattr(sf, "ObservabilityConfig")
    assert hasattr(sf, "has_logfire")
    assert hasattr(sf, "traced")
    assert hasattr(sf, "build_a2a_router")
    assert hasattr(sf, "build_health_router")
    assert hasattr(sf, "create_app")
    assert hasattr(sf, "A2AAgentAdapter")
    assert hasattr(sf, "AgentCard")
    assert hasattr(sf, "DepsFactory")
    assert hasattr(sf, "extract_text")
    assert hasattr(sf, "messages_to_model_history")
    assert hasattr(sf, "Dataset")
    assert hasattr(sf, "EvalCase")
    assert hasattr(sf, "EvalReport")
    assert hasattr(sf, "EvalRunOutput")
    assert hasattr(sf, "SchemaAdherenceScorer")
    assert hasattr(sf, "ScoreResult")
    assert hasattr(sf, "Scorer")
