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
      built once by ``create_app`` and exposed via ``get_ballast()``
      for framework code that needs lazy access.
    ``get_ballast`` — process-wide accessor; raises ``ConfigurationError``
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
    patterns (``DivergentConvergent``, ``HITLChannel`` / ``ThreadChannel`` /
    ``UICardChannel``), capabilities (``BudgetGuard``, ``GroundedRetry``,
    ``SemanticLoopDetector``, …) and evals (``Dataset``, ``Scorer``,
    ``SchemaAdherenceScorer``).

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
    BallastCapability,
    BudgetExhausted,
    BudgetGuard,
    GroundedRetry,
    JudgeAfterRun,
    JudgeFailed,
    JudgeUnavailable,
    JudgeVerdict,
    LLMJudge,
    PairwiseVerdict,
    PIIGuard,
    SemanticLoopDetector,
    persist_verdict_as_thread_event,
    set_default_judge_model,
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
from ballast.events import (
    Signal,
    helper_thread_created,
    message_added,
    receiver,
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
from ballast.runtime.engine import Engine, get_ballast
from ballast.runtime.registry import Named, Registry
from ballast.settings import (
    BallastSettings,
    get_settings,
    reset_settings,
    settings,
)
from ballast import testing  # noqa: F401 — submodule namespace
from ballast.patterns import (
    DivergentAgent,
    DivergentBranch,
    DivergentConvergent,
    HITLDenied,
    HITLTimedOut,
    InsufficientDivergence,
    Pattern,
    PatternError,
    Reflection,
    ReflectionExhausted,
    Synthesizer,
    Verifier,
)
from ballast.patterns.hitl import (
    CardVerdict,
    DBOSHITLChannel,
    HITLChannel,
    ThreadChannel,
    UICardChannel,
    register_card_kind,
)
# Memory subsystem (CoALA Phase 1)
from ballast.memory import Scope
from ballast.memory.episodic import (
    DetailLevel,
    Episode,
    EpisodicMemory,
    EpisodicSource,
    RecallResult,
    RememberTurn,
    ScoredEpisode,
)
from ballast.memory.episodic.sources._thread import ThreadEpisodicSource

try:
    from ballast.memory.episodic.sources._vector import (  # noqa: PLC0415
        EpisodeRow,
        VectorEpisodicSource,
    )
except ImportError:
    EpisodeRow = None  # type: ignore[assignment,misc]
    VectorEpisodicSource = None  # type: ignore[assignment,misc]
from ballast.memory.episodic.strategies import (
    AllRelevant,
    Cluster,
    MapReduce as MapReduceStrategy,
    Recency,
    RecallStrategy,
    TopK,
)
from ballast.patterns.map_reduce import MapReduce
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
    "A2AAgentAdapter",
    "AgentCard",
    "AllRelevant",
    "AgentNotRegistered",
    "AgentRef",
    "AuthError",
    "AuthorizationDenied",
    "Ballast",
    "BallastAgent",
    "BallastCapability",
    "BallastError",
    "BallastSettings",
    "BudgetExhausted",
    "BudgetGuard",
    "CORSConfig",
    "CancelNotSupported",
    "CardVerdict",
    "ConfigurationError",
    "ConfigurationInvariantViolation",
    "CostExtractor",
    "Cluster",
    "Critique",
    "DBOSConfig",
    "DBOSHITLChannel",
    "Dataset",
    "DetailLevel",
    "DepsFactory",
    "Det",
    "DivergentAgent",
    "DivergentBranch",
    "DivergentConvergent",
    "Durable",
    "Embedder",
    "EmptyMessageBody",
    "Engine",
    "Episode",
    "EpisodeRow",
    "EpisodicMemory",
    "EpisodicSource",
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
    "HITLTimedOut",
    "IdempotencyInput",
    "IdempotencyValue",
    "InsufficientDivergence",
    "JudgeAfterRun",
    "JudgeFailed",
    "JudgeUnavailable",
    "JudgeVerdict",
    "LLMJudge",
    "LifespanHook",
    "MapReduceStrategy",
    "MissingDependencyError",
    "Named",
    "ObservabilityConfig",
    "OpenRouterCostExtractor",
    "OpenRouterUpstreamCostExtractor",
    "PIIGuard",
    "PairwiseVerdict",
    "Pattern",
    "PatternError",
    "PersistenceError",
    "Provider",
    "ProviderDetailsCostExtractor",
    "RecallResult",
    "RecallStrategy",
    "Recency",
    "Ref",
    "Reflection",
    "ReflectionExhausted",
    "Registry",
    "RememberTurn",
    "SchemaAdherenceScorer",
    "Scope",
    "ScoreResult",
    "ScoredEpisode",
    "Scorer",
    "Selector",
    "SelectorRegistry",
    "SemanticDeduper",
    "SemanticLoopDetected",
    "SemanticLoopDetector",
    "SettingsValidationError",
    "Signal",
    "Synthesizer",
    "ThreadChannel",
    "ThreadEpisodicSource",
    "ThreadEventBroadcaster",
    "ThreadEventStream",
    "ThreadEventType",
    "ThreadMetadataInvalid",
    "ThreadNotFound",
    "TopK",
    "TraceName",
    "TypedLoopGuard",
    "UICardChannel",
    "Verifier",
    "VectorEpisodicSource",
    "WorkflowNotFound",
    "as_critique",
    "MapReduce",
    "build_a2a_router",
    "build_dbos_config",
    "build_health_router",
    "cancel_thread_workflows",
    "configure_cost_extractors",
    "configure_logging",
    "create_app",
    "extract_text",
    "format_error",
    "get_ballast",
    "get_event_log",
    "get_event_stream",
    "get_logger",
    "get_settings",
    "get_thread_repo",
    "has_logfire",
    "helper_thread_created",
    "message_added",
    "messages_to_model_history",
    "receiver",
    "register_card_kind",
    "register_cost_extractor",
    "register_grounded_tools",
    "reset_settings",
    "set_default_judge_model",
    "settings",
    "stream_response",
    "testing",
    "traced",
    "validate_thread_metadata",
]
