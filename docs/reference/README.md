# Reference

Information-oriented. Every public symbol, every config option, every Protocol.

> **You're here because:** you know what to look up. Use the index, find the page, copy the signature.

For *learning* what these things mean and why, read [explanation/](../explanation/). For *recipes*, see [how-to](../howto/).

## Index

### Core runtime
- `reference/core/ballast-app.md` — `Ballast` builder + fluent setters
- `reference/core/engine.md` — `Engine` + lifecycle hooks
- `reference/core/durable.md` — `Durable` facade (workflow / step / dbos_class / recv_async / send_async)
- `reference/core/settings.md` — `BallastSettings`, environment variables

### Agents
- `reference/agents/ballast-agent.md` — `BallastAgent` ABC + `@tool` decorator
- `reference/agents/durable-agent.md` — `DurableAgent` (durable-mode subclass)
- `reference/agents/registry.md` — `Registry[BallastAgent]` (app-owned)

### Capabilities (`BallastCapability` subclasses)
- `reference/capabilities/base.md` — `BallastCapability` ABC + lifecycle hooks
- `reference/capabilities/budget-guard.md`
- `reference/capabilities/semantic-loop-detector.md`
- `reference/capabilities/typed-loop-guard.md`
- `reference/capabilities/pii-guard.md`
- `reference/capabilities/grounded-retry.md`
- `reference/capabilities/goal-drift-detector.md` — `DriftEngine` + 5 Protocols + built-in impls
- `reference/capabilities/approval-capability.md` — `tool_card_map` + verdict mapping
- `reference/capabilities/llm-judge.md` — `JudgeAfterRun`, `LLMJudge`, verdicts

### Patterns (composable workflows)
- `reference/patterns/pattern-protocol.md`
- `reference/patterns/reflection.md`
- `reference/patterns/mapreduce.md`
- `reference/patterns/divergent-convergent.md` — incl. `on_progress` callback
- `reference/patterns/plan-and-execute.md` — `Plan`, `PlannedStep`, `StepRegistry`, `Step` Protocol, `RePlanPolicy`
- `reference/patterns/hitl-gate.md`
- `reference/patterns/mutation-pipeline.md` — `MutationPipeline`, `ApprovalStage`, `PartialApprovalStage`

### Resilience
- `reference/resilience/circuit-breaker.md` — `CircuitBreaker`, `BreakerState`, `BreakerStats`, adapters
- `reference/resilience/threshold-policies.md` — `Consecutive`, `WindowedCount`, `WindowedRate`
- `reference/resilience/fallback-policies.md` — `RaiseError`, `ReturnValue`, `CallFallback`, `EscalateToHITL`, `Chain`
- `reference/resilience/scope-helpers.md` — `global_scope`, `per_tool_scope`, `per_step_scope`

### Cognitive (CoALA)
- `reference/coala/coala-unit-protocol.md`
- `reference/coala/coala-base.md`
- `reference/coala/adapters.md` — `as_workflow`, `as_tool`, `as_capability`

### Quality
- `reference/quality/scored.md` — `Scored[T, ConfidenceT]`, `Confidence`, helpers

### Grounded
- `reference/grounded/ref-t.md` — `Ref[T]`, `hydrate`, pydantic schema integration
- `reference/grounded/selector.md` — narrowing tool-input JSON Schema
- `reference/grounded/scan-output.md` — output-schema walker + role detection
- `reference/grounded/grounded-agent.md` — `GroundedAgent`, `GroundedResult`, `HydrationMap`

### HITL
- `reference/hitl/channels.md` — `HITLChannel` Protocol, `DBOSHITLChannel` ABC
- `reference/hitl/ui-card-channel.md` — `UICardChannel`, `ApprovalCard`, `CardVerdict`, `register_card_kind`
- `reference/hitl/thread-channel.md` — in-chat marker flow
- `reference/hitl/helper-agent.md` — `HelperAgent`, `ConversationalChannel`, `HelperVerdict[T]`

### Persistence
- `reference/persistence/thread-repository.md`
- `reference/persistence/approval-card-repository.md`
- `reference/persistence/sql-repositories.md`
- `reference/persistence/alembic-migrations.md`

### API surfaces
- `reference/api/streaming-router.md` — SSE streaming endpoint + Last-Event-ID resume
- `reference/api/approvals-router.md`
- `reference/api/a2a-router.md`
- `reference/api/health-router.md`

### Observability
- `reference/observability/traced.md`
- `reference/observability/cost-extractors.md`
- `reference/observability/events.md` — `Signal`, `ThreadEventBroadcaster`, `ThreadEventStream`

### Evals
- `reference/evals/dataset.md` — `Dataset`, `EvalCase`, `EvalReport`
- `reference/evals/scorers.md` — `SchemaAdherenceScorer`, custom `Scorer` Protocol

### Errors
- `reference/errors.md` — all `BallastError` subclasses with `code` / `status_code` / `hint` / `context`

---

**Auto-generation note:** Most reference pages can/should be generated from docstrings + type signatures. A future task will set up `mkdocstrings` or similar tooling. For now, this index serves as the topology.
