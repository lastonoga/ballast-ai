"""pydantic-ai-stateflow — Sub-project #1 (Foundation) public API.

Layer 0 (GroundedSchema):
    Ref, GroundedAgent, GroundedResult, GroundedResolver
    GroundedError, GroundedBuildError, GroundedHydrationError

Runtime helpers:
    Det, IdempotencyInput, IdempotencyValue

Patterns:
    Pattern (Protocol)

Sub-project #3 (Runtime):
    Container, DefaultContainer, Engine, EngineInvariantViolation
    ServiceProvider, CoreProvider, PersistenceProvider
    DBOSConfig, build_dbos_config

Sub-project #4 (Capabilities):
    BudgetExhausted, BudgetGuard, GroundedRetry, PIIGuard,
    SemanticLoopDetector, StateflowCapability
    Critique, Embedder, SemanticDeduper, SemanticLoopDetected,
    TypedLoopGuard, as_critique
"""

from pydantic_ai_stateflow.capabilities import (
    BudgetExhausted,
    BudgetGuard,
    GroundedRetry,
    PIIGuard,
    SemanticLoopDetector,
    StateflowCapability,
)
from pydantic_ai_stateflow.capabilities.helpers import (
    Critique,
    Embedder,
    SemanticDeduper,
    SemanticLoopDetected,
    TypedLoopGuard,
    as_critique,
)
from pydantic_ai_stateflow.grounded import (
    GroundedAgent,
    GroundedBuildError,
    GroundedError,
    GroundedHydrationError,
    GroundedResolver,
    GroundedResult,
    Ref,
)
from pydantic_ai_stateflow.patterns import Pattern
from pydantic_ai_stateflow.providers import CoreProvider, PersistenceProvider
from pydantic_ai_stateflow.runtime import (
    Container,
    DBOSConfig,
    DefaultContainer,
    Det,
    Engine,
    EngineInvariantViolation,
    IdempotencyInput,
    IdempotencyValue,
    ServiceProvider,
    build_dbos_config,
)

__all__ = [
    "BudgetExhausted",
    "BudgetGuard",
    "Container",
    "CoreProvider",
    "Critique",
    "DBOSConfig",
    "DefaultContainer",
    "Det",
    "Embedder",
    "Engine",
    "EngineInvariantViolation",
    "GroundedAgent",
    "GroundedBuildError",
    "GroundedError",
    "GroundedHydrationError",
    "GroundedResolver",
    "GroundedResult",
    "GroundedRetry",
    "IdempotencyInput",
    "IdempotencyValue",
    "PIIGuard",
    "Pattern",
    "PersistenceProvider",
    "Ref",
    "SemanticDeduper",
    "SemanticLoopDetected",
    "SemanticLoopDetector",
    "ServiceProvider",
    "StateflowCapability",
    "TypedLoopGuard",
    "as_critique",
    "build_dbos_config",
]
