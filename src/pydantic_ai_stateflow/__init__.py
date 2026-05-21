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

Sub-project #5 (Patterns):
    Reflection + LoopRecoveryPolicy / AbortOnLoop, MapReduce (+ Chunker /
    Reducer), MutationPipeline (+ Proposal, Stage, ApprovalStage,
    PartialApprovalStage, ApplyTransaction, AcceptedResult, RejectedAt,
    RejectAction, RejectPolicy, DropOnReject, RaiseOnReject),
    HITLGate (+ HITLChannel, InMemoryHITLChannel, HITLPrompt, HITLOption,
    HITLResponse, ApprovedResponse, RejectedResponse, ModifiedResponse,
    TimeoutResponse, Policy, Voter, AllowAll, DenyAll, AccessDecision),
    pattern errors (PatternError, ReflectionExhausted, HITLDenied,
    HITLTimedOut, MutationRejected).

Sub-project #6 (HITL channels):
    UIChannel, WebhookChannel, WebhookConfig, ConversationalChannel,
    HelperVerdict, HelperAgentFactory, HelperDeps, HelperToolBox,
    HelperSessionInput, HelperSessionRunner, DefaultHelperSessionRunner,
    make_helper_agent_with_approval_tools, build_hitl_router.

Sub-project #7 (API + Observability + Evals):
    build_a2a_router, build_health_router, build_streaming_router,
    build_threads_router, get_container, get_engine,
    A2AAgentAdapter, AgentCard, DepsFactory, extract_text,
    messages_to_model_history,
    Dataset, EvalCase, EvalReport, EvalRunOutput, SchemaAdherenceScorer,
    ScoreResult, Scorer, ObservabilityProvider, has_logfire, traced.

    Note: wire encoding, body parsing, event taxonomy, and the tool-
    approval round-trip are delegated to
    ``pydantic_ai.ui.vercel_ai.VercelAIAdapter`` — the framework no
    longer ships its own ``AGUIEncoder`` / ``VercelEncoder`` /
    ``StreamEvent`` / ``StreamEventKind`` / ``AgentRunner`` / ``make_runner``.
"""

# Side-effect import: attaches NullHandler to the framework root logger
# AND auto-configures a StreamHandler when ``STATEFLOW_LOG_LEVEL`` is
# set. Import this BEFORE the shim so any shim diagnostics route through
# the framework logger.
from pydantic_ai_stateflow import logging as _logging  # noqa: F401

# Side-effect import: applies upstream pydantic-ai compatibility shims
# (e.g. normalize OpenAI assistant ``content: null`` → ``""`` for tool-
# call turns so Alibaba/strict Qwen endpoints accept the request). See
# ``_compat/`` for the full rationale.
from pydantic_ai_stateflow import _compat as _compat  # noqa: F401

from pydantic_ai_stateflow.api import (
    A2AAgentAdapter,
    AgentCard,
    CORSConfig,
    DepsFactory,
    build_a2a_router,
    build_dbos_router,
    build_health_router,
    build_streaming_router,
    build_threads_router,
    extract_text,
    get_container,
    get_engine,
    messages_to_model_history,
)
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
from pydantic_ai_stateflow.evals import (
    Dataset,
    EvalCase,
    EvalReport,
    EvalRunOutput,
    SchemaAdherenceScorer,
    Scorer,
    ScoreResult,
)
from pydantic_ai_stateflow.logging import (
    configure as configure_logging,
    get_logger,
)
from pydantic_ai_stateflow.grounded import (
    GroundedAgent,
    GroundedBuildError,
    GroundedError,
    GroundedHydrationError,
    GroundedResolver,
    GroundedResult,
    Ref,
    Selector,
    SelectorRegistry,
    register_grounded_tools,
)
from pydantic_ai_stateflow.observability import (
    CostExtractor,
    ObservabilityProvider,
    OpenRouterCostExtractor,
    OpenRouterUpstreamCostExtractor,
    ProviderDetailsCostExtractor,
    TraceName,
    configure_cost_extractors,
    has_logfire,
    register_cost_extractor,
    traced,
)
from pydantic_ai_stateflow.patterns import (
    AbortOnLoop,
    ApprovalStage,
    Chunker,
    DivergentAgent,
    DivergentBranch,
    DivergentConvergent,
    HITLDenied,
    HITLGate,
    HITLTimedOut,
    InsufficientDivergence,
    LoopRecoveryPolicy,
    MapReduce,
    MutationPipeline,
    MutationRejected,
    PartialApprovalStage,
    Pattern,
    PatternError,
    Projector,
    Proposal,
    Reducer,
    Reflection,
    ReflectionExhausted,
    SemanticDedup,
    SemanticDedupConfig,
    Synthesizer,
    Verifier,
)
from pydantic_ai_stateflow.patterns.hitl import (
    AccessDecision,
    AllowAll,
    ApprovedResponse,
    ConversationalChannel,
    DefaultHelperSessionRunner,
    DenyAll,
    HelperAgentFactory,
    HelperDeps,
    HelperSessionInput,
    HelperSessionRunner,
    HelperToolBox,
    HelperVerdict,
    HITLChannel,
    HITLOption,
    HITLPrompt,
    HITLResponse,
    InMemoryHITLChannel,
    ModifiedResponse,
    Policy,
    RejectedResponse,
    TimeoutResponse,
    UIChannel,
    Voter,
    WebhookChannel,
    WebhookConfig,
    build_hitl_router,
    make_helper_agent_with_approval_tools,
)
from pydantic_ai_stateflow.patterns.mutation import (
    AcceptedResult,
    ApplyTransaction,
    DropOnReject,
    RaiseOnReject,
    RejectAction,
    RejectedAt,
    RejectPolicy,
    Stage,
)
from pydantic_ai_stateflow.providers import CoreProvider, PersistenceProvider
from pydantic_ai_stateflow.runtime import (
    AgentRef,
    Container,
    DBOSConfig,
    DefaultContainer,
    Det,
    Engine,
    EngineInvariantViolation,
    IdempotencyInput,
    IdempotencyValue,
    ServiceProvider,
    StateflowAgent,
    build_dbos_config,
    clear_agent_registry,
    get_agent,
    list_agents,
    register_agent,
    validate_thread_metadata,
)

__all__ = [
    "A2AAgentAdapter",
    "AbortOnLoop",
    "AcceptedResult",
    "AccessDecision",
    "AgentCard",
    "AllowAll",
    "ApplyTransaction",
    "ApprovalStage",
    "ApprovedResponse",
    "BudgetExhausted",
    "BudgetGuard",
    "CORSConfig",
    "Chunker",
    "Container",
    "ConversationalChannel",
    "CoreProvider",
    "Critique",
    "DBOSConfig",
    "Dataset",
    "DefaultContainer",
    "DefaultHelperSessionRunner",
    "DenyAll",
    "DepsFactory",
    "Det",
    "DivergentAgent",
    "DivergentBranch",
    "DivergentConvergent",
    "DropOnReject",
    "Embedder",
    "Engine",
    "EngineInvariantViolation",
    "EvalCase",
    "EvalReport",
    "EvalRunOutput",
    "GroundedAgent",
    "GroundedBuildError",
    "GroundedError",
    "GroundedHydrationError",
    "GroundedResolver",
    "GroundedResult",
    "GroundedRetry",
    "HITLChannel",
    "HITLDenied",
    "HITLGate",
    "HITLOption",
    "HITLPrompt",
    "HITLResponse",
    "HITLTimedOut",
    "HelperAgentFactory",
    "HelperDeps",
    "HelperSessionInput",
    "HelperSessionRunner",
    "HelperToolBox",
    "HelperVerdict",
    "IdempotencyInput",
    "IdempotencyValue",
    "InMemoryHITLChannel",
    "InsufficientDivergence",
    "LoopRecoveryPolicy",
    "MapReduce",
    "ModifiedResponse",
    "MutationPipeline",
    "MutationRejected",
    "CostExtractor",
    "ObservabilityProvider",
    "OpenRouterCostExtractor",
    "OpenRouterUpstreamCostExtractor",
    "PIIGuard",
    "ProviderDetailsCostExtractor",
    "PartialApprovalStage",
    "Pattern",
    "PatternError",
    "PersistenceProvider",
    "Policy",
    "Projector",
    "Proposal",
    "RaiseOnReject",
    "Reducer",
    "Ref",
    "Reflection",
    "ReflectionExhausted",
    "RejectAction",
    "RejectPolicy",
    "RejectedAt",
    "RejectedResponse",
    "SchemaAdherenceScorer",
    "ScoreResult",
    "Scorer",
    "Selector",
    "SelectorRegistry",
    "SemanticDedup",
    "SemanticDedupConfig",
    "SemanticDeduper",
    "SemanticLoopDetected",
    "SemanticLoopDetector",
    "AgentRef",
    "ServiceProvider",
    "Stage",
    "StateflowAgent",
    "StateflowCapability",
    "Synthesizer",
    "TimeoutResponse",
    "TraceName",
    "TypedLoopGuard",
    "UIChannel",
    "Verifier",
    "Voter",
    "WebhookChannel",
    "WebhookConfig",
    "as_critique",
    "build_a2a_router",
    "build_dbos_config",
    "build_dbos_router",
    "build_health_router",
    "build_hitl_router",
    "build_streaming_router",
    "build_threads_router",
    "clear_agent_registry",
    "configure_cost_extractors",
    "configure_logging",
    "extract_text",
    "get_agent",
    "get_container",
    "get_engine",
    "get_logger",
    "has_logfire",
    "list_agents",
    "make_helper_agent_with_approval_tools",
    "messages_to_model_history",
    "register_agent",
    "register_cost_extractor",
    "register_grounded_tools",
    "traced",
    "validate_thread_metadata",
]
