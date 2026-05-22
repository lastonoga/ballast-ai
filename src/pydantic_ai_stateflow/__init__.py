"""pydantic-ai-stateflow — public framework surface.

App entry point:
    ``create_app(...)`` builds a FastAPI app with thread CRUD, streaming
    chat, A2A endpoints, the DBOS router, and an auto-generated
    ``POST /workflows/{kebab-name}`` per registered ``@sf.workflow``.

Authoring primitives:
    ``stateflow_agent`` — class decorator to register a ``StateflowAgent``
      subclass (resolves under its kebab-name).
    ``workflow`` — class decorator to register a durable workflow whose
      ``async def run(self, input) -> output`` is wrapped with
      ``@Durable.workflow()``.
    ``Durable`` — DBOS facade that bundles workflow / step / queue
      decoration with OTel context propagation.

Errors:
    ``StateflowError`` (base) plus structured subclasses (``ThreadNotFound``,
    ``AgentNotRegistered``, ``WorkflowNotFound``, ``EmptyMessageBody``,
    ``CancelNotSupported``, ``ConfigurationInvariantViolation``, …).
    Auto-rendered as ``application/problem+json`` by the error
    middleware that ``create_app`` installs.

Configuration:
    ``StateflowSettings`` — pydantic-settings hierarchy
      (``api``, ``observability``, ``dbos``, …) with env-var loading.
    ``ObservabilityConfig`` — call ``.install()`` once to enable
      logfire + auto-instrumentation; ``.instrument_app(app)`` attaches
      the FastAPI integration.

Testing:
    ``testing.TestEngine`` — boots an in-memory framework for tests,
    drops DBOS state on teardown.

Other primitives:
    Grounded schema (``Ref``, ``GroundedAgent``, ``Selector``, …),
    patterns (``Reflection``, ``MapReduce``, ``HITLGate``, ``MutationPipeline``,
    ``SemanticDedup``, ``DivergentConvergent``, …), capabilities
    (``BudgetGuard``, ``GroundedRetry``, ``SemanticLoopDetector``, …) and
    evals (``Dataset``, ``Scorer``, ``SchemaAdherenceScorer``).

    Wire encoding, body parsing, and the tool-approval round-trip are
    delegated to ``pydantic_ai.ui.vercel_ai.VercelAIAdapter`` — the
    framework no longer ships its own ``AGUIEncoder`` / ``VercelEncoder``
    / ``StreamEvent`` / ``AgentRunner`` / ``make_runner``.
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
    build_health_router,
    extract_text,
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
    OpenRouterCostExtractor,
    OpenRouterUpstreamCostExtractor,
    ProviderDetailsCostExtractor,
    TraceName,
    configure_cost_extractors,
    has_logfire,
    register_cost_extractor,
    traced,
)
from pydantic_ai_stateflow.errors import (
    AgentNotRegistered,
    AuthError,
    AuthorizationDenied,
    CancelNotSupported,
    ConfigurationError,
    ConfigurationInvariantViolation,
    EmptyMessageBody,
    MissingDependencyError,
    PersistenceError,
    SettingsValidationError,
    StateflowError,
    ThreadMetadataInvalid,
    ThreadNotFound,
    WorkflowNotFound,
    format_error,
)
from pydantic_ai_stateflow.observability.config import ObservabilityConfig
from pydantic_ai_stateflow.runtime.app import create_app
from pydantic_ai_stateflow.settings import (
    StateflowSettings,
    get_settings,
    reset_settings,
    settings,
)
from pydantic_ai_stateflow.runtime.workflows import (
    clear_workflow_registry,
    get_workflow_class,
    list_workflow_classes,
    workflow,
    workflow_metadata,
)
from pydantic_ai_stateflow.runtime.agents import (
    clear_agent_class_registry,
    get_agent_class,
    list_agent_classes,
    stateflow_agent,
)
from pydantic_ai_stateflow import testing  # noqa: F401 — submodule namespace
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
from pydantic_ai_stateflow.durable import Durable
from pydantic_ai_stateflow.runtime import (  # noqa: I001
    ThreadEventBroadcaster,
    ThreadEventStream,
    ThreadEventType,
)
from pydantic_ai_stateflow.runtime import (
    AgentRef,
    DBOSConfig,
    Det,
    IdempotencyInput,
    IdempotencyValue,
    StateflowAgent,
    build_dbos_config,
    validate_thread_metadata,
)

__all__ = [
    "A2AAgentAdapter",
    "AbortOnLoop",
    "AcceptedResult",
    "AccessDecision",
    "AgentCard",
    "AgentNotRegistered",
    "AllowAll",
    "ApplyTransaction",
    "ApprovalStage",
    "ApprovedResponse",
    "AuthError",
    "AuthorizationDenied",
    "BudgetExhausted",
    "BudgetGuard",
    "CORSConfig",
    "CancelNotSupported",
    "Chunker",
    "ConfigurationError",
    "ConfigurationInvariantViolation",
    "ConversationalChannel",
    "Critique",
    "DBOSConfig",
    "Dataset",
    "DefaultHelperSessionRunner",
    "DenyAll",
    "DepsFactory",
    "Det",
    "Durable",
    "DivergentAgent",
    "DivergentBranch",
    "DivergentConvergent",
    "DropOnReject",
    "Embedder",
    "EmptyMessageBody",
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
    "MissingDependencyError",
    "ModifiedResponse",
    "MutationPipeline",
    "MutationRejected",
    "CostExtractor",
    "ObservabilityConfig",
    "OpenRouterCostExtractor",
    "OpenRouterUpstreamCostExtractor",
    "PIIGuard",
    "PersistenceError",
    "ProviderDetailsCostExtractor",
    "PartialApprovalStage",
    "Pattern",
    "PatternError",
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
    "SettingsValidationError",
    "AgentRef",
    "Stage",
    "StateflowAgent",
    "StateflowCapability",
    "StateflowError",
    "StateflowSettings",
    "Synthesizer",
    "ThreadEventBroadcaster",
    "ThreadEventStream",
    "ThreadEventType",
    "ThreadMetadataInvalid",
    "ThreadNotFound",
    "TimeoutResponse",
    "TraceName",
    "TypedLoopGuard",
    "UIChannel",
    "Verifier",
    "Voter",
    "WebhookChannel",
    "WebhookConfig",
    "WorkflowNotFound",
    "as_critique",
    "build_a2a_router",
    "build_dbos_config",
    "build_health_router",
    "build_hitl_router",
    "clear_agent_class_registry",
    "clear_workflow_registry",
    "configure_cost_extractors",
    "configure_logging",
    "create_app",
    "extract_text",
    "format_error",
    "get_settings",
    "get_agent_class",
    "get_logger",
    "get_workflow_class",
    "has_logfire",
    "list_agent_classes",
    "list_workflow_classes",
    "make_helper_agent_with_approval_tools",
    "messages_to_model_history",
    "register_cost_extractor",
    "register_grounded_tools",
    "reset_settings",
    "settings",
    "stateflow_agent",
    "testing",
    "traced",
    "validate_thread_metadata",
    "workflow",
    "workflow_metadata",
]
