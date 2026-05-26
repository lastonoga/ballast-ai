"""Circuit Breaker — resilience primitive for protecting async function invocations."""
from ballast.resilience.circuit_breaker._adapters.capability import as_capability
from ballast.resilience.circuit_breaker._adapters.step import BreakerStep, as_step
from ballast.resilience.circuit_breaker._adapters.workflow import as_workflow_decorator
from ballast.resilience.circuit_breaker._breaker import CircuitBreaker
from ballast.resilience.circuit_breaker._fallbacks import (
    CallFallback, Chain, EscalateToHITL, RaiseError, ReturnValue,
)
from ballast.resilience.circuit_breaker._protocols import (
    FallbackPolicy, ScopeKey, ThresholdFactory, ThresholdPolicy,
)
from ballast.resilience.circuit_breaker._scope import (
    global_scope, per_step_scope, per_tool_scope,
)
from ballast.resilience.circuit_breaker._state import (
    BreakerState, BreakerStats, CircuitOpenError,
)
from ballast.resilience.circuit_breaker._thresholds import (
    Consecutive, WindowedCount, WindowedRate,
)

__all__ = [
    "BreakerState", "BreakerStats", "BreakerStep",
    "CallFallback", "Chain", "CircuitBreaker", "CircuitOpenError",
    "Consecutive", "EscalateToHITL", "FallbackPolicy",
    "RaiseError", "ReturnValue", "ScopeKey",
    "ThresholdFactory", "ThresholdPolicy",
    "WindowedCount", "WindowedRate",
    "as_capability", "as_step", "as_workflow_decorator",
    "global_scope", "per_step_scope", "per_tool_scope",
]
