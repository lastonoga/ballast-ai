"""Observability — ObservabilityProvider with a soft logfire dependency.

logfire is an optional dependency. When absent, every observability shim
degrades to a no-op so the test suite (and applications that don't want
telemetry) keep working. Spec 4D, 4H.
"""

from pydantic_ai_stateflow.observability.provider import (
    ObservabilityProvider,
    has_logfire,
)
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName

__all__ = ["ObservabilityProvider", "TraceName", "has_logfire", "traced"]
