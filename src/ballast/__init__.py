"""pydantic-ai-stateflow — public framework surface.

App entry point:
    ``create_app(*, thread_repo=, event_log=, event_stream=, ...)`` builds
    a FastAPI app with thread CRUD, the DBOS router, and a health
    endpoint. Internally constructs an ``Engine`` from the supplied
    repos + stream and stashes it as the process-wide singleton.
    Apps mount their own streaming / cancel / workflow routes via
    ``extra_routers=[...]``.

Authoring primitives:
    ``Engine`` — frozen dataclass bundling repos + event log + stream;
      built once by ``create_app`` and exposed via ``get_engine()``
      for framework code that needs lazy access.
    ``get_engine`` — process-wide accessor; raises ``ConfigurationError``
      if ``create_app`` hasn't been called yet.
    ``stream_response`` — primitive for ``POST /threads/{id}/messages``
      style routes: body-vs-DB sync, durable / inline dispatch,
      Vercel-AI streaming, assistant-turn persistence.
    ``cancel_thread_workflows`` — primitive cancelling every active
      workflow for a thread.
    ``Durable`` — DBOS facade that bundles workflow / step / queue
      decoration with OTel context propagation.

Errors:
    ``BallastError`` (base) plus structured subclasses (``ThreadNotFound``,
    ``AgentNotRegistered``, ``WorkflowNotFound``, ``EmptyMessageBody``,
    ``CancelNotSupported``, ``ConfigurationInvariantViolation``, …).
    Auto-rendered as ``application/problem+json`` by the error
    middleware that ``create_app`` installs.

Configuration:
    ``BallastSettings`` — pydantic-settings hierarchy
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
# AND auto-configures a StreamHandler when ``BALLAST_LOG_LEVEL`` is
# set. Import this BEFORE the shim so any shim diagnostics route through
# the framework logger.
from ballast import logging as _logging  # noqa: F401

# Side-effect import: applies upstream pydantic-ai compatibility shims
# (e.g. normalize OpenAI assistant ``content: null`` → ``""`` for tool-
# call turns so Alibaba/strict Qwen endpoints accept the request). See
# ``_compat/`` for the full rationale.
from ballast import _compat as _compat  # noqa: F401

from ballast.api import (
    A2AAgentAdapter,
    AgentCard,
    CORSConfig,
    DepsFactory,
    build_a2a_router,
    build_health_router,
    cancel_thread_workflows,
    extract_text,
    messages_to_model_history,
    stream_response,
)
from ballast.api.deps import (
    get_engine_dep,
    get_event_log,
    get_event_stream,
    get_thread_repo,
)
from ballast.capabilities import (
    BudgetExhausted,
    BudgetGuard,
    GroundedRetry,
    PIIGuard,
    SemanticLoopDetector,
    BallastCapability,
)
from ballast.capabilities.helpers import (
    Critique,
    Embedder,
    SemanticDeduper,
    SemanticLoopDetected,
    TypedLoopGuard,
    as_critique,
)
from ballast.evals import (
    Dataset,
    EvalCase,
    EvalReport,
    EvalRunOutput,
    SchemaAdherenceScorer,
    Scorer,
    ScoreResult,
)
from ballast.logging import (
    configure as configure_logging,
    get_logger,
)
from ballast.grounded import (
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
from ballast.observability import (
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
from ballast.errors import (
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
    BallastError,
    ThreadMetadataInvalid,
    ThreadNotFound,
    WorkflowNotFound,
    format_error,
)
from ballast.app import Ballast, LifespanHook, Provider
from ballast.observability.config import ObservabilityConfig
from ballast.runtime.app import create_app
from ballast.runtime.engine import Engine, get_engine
from ballast.runtime.registry import Named, Registry
from ballast.settings import (
    BallastSettings,
    get_settings,
    reset_settings,
    settings,
)
from ballast import testing  # noqa: F401 — submodule namespace
from ballast.patterns import (
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
from ballast.patterns.hitl import (
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
from ballast.patterns.mutation import (
    AcceptedResult,
    ApplyTransaction,
    DropOnReject,
    RaiseOnReject,
    RejectAction,
    RejectedAt,
    RejectPolicy,
    Stage,
)
from ballast.durable import Durable
from ballast.runtime import (  # noqa: I001
    ThreadEventBroadcaster,
    ThreadEventStream,
    ThreadEventType,
)
from ballast.runtime import (
    AgentRef,
    DBOSConfig,
    Det,
    IdempotencyInput,
    IdempotencyValue,
    BallastAgent,
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
    "Engine",
    "AllowAll",
    "ApplyTransaction",
    "ApprovalStage",
    "ApprovedResponse",
    "AuthError",
    "AuthorizationDenied",
    "Ballast",
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
    "LifespanHook",
    "LoopRecoveryPolicy",
    "MapReduce",
    "MissingDependencyError",
    "ModifiedResponse",
    "MutationPipeline",
    "MutationRejected",
    "Named",
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
    "Provider",
    "RaiseOnReject",
    "Reducer",
    "Registry",
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
    "BallastAgent",
    "BallastCapability",
    "BallastError",
    "BallastSettings",
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
    "cancel_thread_workflows",
    "configure_cost_extractors",
    "configure_logging",
    "create_app",
    "extract_text",
    "format_error",
    "get_engine",
    "get_event_log",
    "get_event_stream",
    "get_logger",
    "get_settings",
    "get_thread_repo",
    "has_logfire",
    "make_helper_agent_with_approval_tools",
    "messages_to_model_history",
    "register_cost_extractor",
    "register_grounded_tools",
    "reset_settings",
    "settings",
    "stream_response",
    "testing",
    "traced",
    "validate_thread_metadata",
]
