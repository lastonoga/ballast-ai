"""SP7 exports smoke."""
from __future__ import annotations

import pydantic_ai_stateflow as sf


def test_sp7_exports_present():
    assert hasattr(sf, "ObservabilityProvider")
    assert hasattr(sf, "has_logfire")
    assert hasattr(sf, "traced")
    assert hasattr(sf, "build_threads_router")
    assert hasattr(sf, "build_a2a_router")
    assert hasattr(sf, "build_streaming_router")
    assert hasattr(sf, "build_health_router")
    assert hasattr(sf, "get_container")
    assert hasattr(sf, "get_engine")
    assert hasattr(sf, "get_tenant_id")
    assert hasattr(sf, "A2AAgentAdapter")
    assert hasattr(sf, "AgentCard")
    assert hasattr(sf, "AGUIEncoder")
    assert hasattr(sf, "VercelEncoder")
    assert hasattr(sf, "StreamEvent")
    assert hasattr(sf, "Dataset")
    assert hasattr(sf, "EvalCase")
    assert hasattr(sf, "EvalReport")
    assert hasattr(sf, "EvalRunOutput")
    assert hasattr(sf, "SchemaAdherenceScorer")
    assert hasattr(sf, "ScoreResult")
    assert hasattr(sf, "Scorer")
