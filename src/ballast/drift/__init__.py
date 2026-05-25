"""Goal Drift Detection — pluggable LLM-judge sidecar for agent runs."""
from ballast.drift._protocols import (
    DriftCheckSignal,
    DriftContext,
    DriftCheckStrategy,
    DriftHandler,
    GoalSource,
    PromptBuilder,
    TraceWindow,
)
from ballast.drift._strategies import (
    AfterEveryStep,
    Compose as ComposeStrategy,
    EveryNSteps,
    EveryNToolCalls,
    OnBudgetThreshold,
    Periodic,
)
from ballast.drift._goal_sources import (
    ExplicitGoal, FirstUserMessage, LastUserMessage, WorkflowInput,
)
from ballast.drift._verdict import DefaultDriftVerdict, DriftVerdictBase
from ballast.drift._judge import DefaultPromptBuilder, make_default_judge
from ballast.drift._handlers import (
    Compose as ComposeHandler,
    EmitDriftEvent,
    EscalateToHITL,
    GoalDriftError,
    LogOnly,
    RaiseDriftError,
)
from ballast.drift._windows import (
    FullTrace,
    LastNMessages,
    SinceLastUserMessage,
    TokenBudgetWindow,
)
from ballast.drift._core import DriftEngine
from ballast.drift._coala import goal_drift_as_unit

__all__ = [
    "AfterEveryStep",
    "ComposeHandler",
    "ComposeStrategy",
    "DefaultDriftVerdict",
    "DefaultPromptBuilder",
    "DriftCheckSignal",
    "DriftCheckStrategy",
    "DriftContext",
    "DriftEngine",
    "DriftHandler",
    "DriftVerdictBase",
    "EmitDriftEvent",
    "EscalateToHITL",
    "EveryNSteps",
    "EveryNToolCalls",
    "ExplicitGoal",
    "FirstUserMessage",
    "FullTrace",
    "goal_drift_as_unit",
    "GoalDriftError",
    "GoalSource",
    "LastUserMessage",
    "LogOnly",
    "LastNMessages",
    "make_default_judge",
    "OnBudgetThreshold",
    "Periodic",
    "PromptBuilder",
    "RaiseDriftError",
    "SinceLastUserMessage",
    "TokenBudgetWindow",
    "TraceWindow",
    "WorkflowInput",
]
