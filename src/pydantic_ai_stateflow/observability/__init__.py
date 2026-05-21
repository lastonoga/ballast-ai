"""Observability — ObservabilityProvider with a soft logfire dependency.

logfire is an optional dependency. When absent, every observability shim
degrades to a no-op so the test suite (and applications that don't want
telemetry) keep working. Spec 4D, 4H.
"""

from pydantic_ai_stateflow.observability.cost import (
    CostExtractor,
    OpenRouterCostExtractor,
    OpenRouterUpstreamCostExtractor,
    ProviderDetailsCostExtractor,
    configure_cost_extractors,
    install_cost_fallback_patch,
    register_cost_extractor,
)
from pydantic_ai_stateflow.observability.provider import has_logfire
from pydantic_ai_stateflow.observability.otel_carrier import (
    attach_otel_carrier,
    detach_otel_carrier,
    inject_otel_carrier,
    otel_context_from,
)
from pydantic_ai_stateflow.observability.spans import traced
from pydantic_ai_stateflow.observability.trace_names import TraceName

__all__ = [
    "CostExtractor",
    "OpenRouterCostExtractor",
    "OpenRouterUpstreamCostExtractor",
    "ProviderDetailsCostExtractor",
    "TraceName",
    "attach_otel_carrier",
    "configure_cost_extractors",
    "detach_otel_carrier",
    "has_logfire",
    "inject_otel_carrier",
    "install_cost_fallback_patch",
    "otel_context_from",
    "register_cost_extractor",
    "traced",
]


# Auto-install the cost-fallback patch at framework import. The patch
# is purely additive — without any extractors registered it re-raises
# the original ``LookupError``, so behaviour is identical to
# unpatched pydantic-ai when nothing else opts in. Apps register
# extractors via ``ObservabilityProvider(cost_extractors=...)`` or
# ``register_cost_extractor(...)`` and the same patch handles them.
install_cost_fallback_patch()
