# How-to guides

Task-oriented recipes. Each guide answers a specific "How do I…?" question with the minimal code needed.

> **You're here because:** you know what you want; you need the recipe. Skim the index, jump straight to the code.

## Catalog

### Patterns
- `howto/use-mapreduce-for-rag.md` — process documents bigger than the context window
- `howto/use-reflection-for-quality.md` — writer-critic-refiner loop with iteration cap
- `howto/use-divergent-convergent-for-brainstorm.md` — exploration → synthesis pipeline
- `howto/build-plan-execute-pipeline.md` — planner-emitted DAG + framework dispatch

### Capabilities
- `howto/add-budget-guard.md` — token + iteration caps
- `howto/add-semantic-loop-detector.md` — embedding-based loop detection
- `howto/add-goal-drift-detector.md` — async LLM-judge against original goal
- `howto/add-typed-loop-guard.md` — typed-output convergence detector

### Resilience
- `howto/add-circuit-breaker-to-tool.md` — per-tool CB with `as_workflow_decorator`
- `howto/protect-plan-execute-steps.md` — wrap DAG nodes with `as_step(cb, step)`
- `howto/configure-fallback-chain.md` — `Chain(CallFallback, EscalateToHITL, RaiseError)`
- `howto/define-custom-threshold.md` — write your own `ThresholdPolicy`

### Quality
- `howto/add-confidence-to-tool-outputs.md` — `Scored[T]` for Map-phase quality signal
- `howto/filter-and-rank-by-confidence.md` — built-in helpers usage

### CoALA
- `howto/build-coala-unit.md` — observe / retrieve / act / learn skeleton
- `howto/deploy-coala-unit-as-tool.md` — `as_tool(unit)` for agent
- `howto/deploy-coala-unit-as-workflow.md` — `as_workflow(unit)` for durable pipeline
- `howto/compose-coala-units-in-plan.md` — `UnitStep` inside `PlanAndExecute`

### HITL
- `howto/add-approval-card-flow.md` — `UICardChannel` + `ApprovalCard` + frontend panel
- `howto/auto-bridge-requires-approval.md` — `ApprovalCapability` for pydantic-ai's `requires_approval=True`
- `howto/build-helper-agent-clarification.md` — `HelperAgent` + `ConversationalChannel`
- `howto/customize-hitl-channel.md` — write a Slack/email/Telegram channel

### Grounded
- `howto/use-ref-to-prevent-id-hallucination.md` — `Ref[T]` + `scan_output` schema narrowing
- `howto/hydrate-refs-in-tool-output.md` — `result.hydrate(**repos)`
- `howto/define-app-domain-models.md` — pydantic models that play nice with `Ref`

### Persistence
- `howto/wire-postgres-thread-repo.md` — SqlThreadRepository + Alembic
- `howto/persist-approval-cards.md` — SqlApprovalCardRepository + migration 0002
- `howto/swap-thread-repo-for-mongo.md` — custom `ThreadRepository` impl

### Observability
- `howto/add-logfire-tracing.md` — `logfire.configure()` + `@traced`
- `howto/run-llm-judge-evaluation.md` — `LLMJudge` + `JudgeAfterRun`
- `howto/build-eval-dataset-from-traces.md` — `dataset-from-traces` CLI

### Testing
- `howto/test-agents-with-testmodel.md` — pydantic-ai `TestModel` patterns
- `howto/test-coala-units.md` — direct phase testing
- `howto/test-workflows-with-dbos-fixture.md` — module-scoped DBOS bootstrap

### Operations
- `howto/handle-workflow-cancellation.md` — `cancel_thread_workflows`
- `howto/migrate-purpose-to-agent.md` — pre-1.0 schema rename (historical)

---

**Don't see your task?** Check [reference/](../reference/) for the underlying API; chances are the recipe is straightforward composition. If you find yourself building substantial helper code from scratch — open an issue, it might be a candidate for the framework.
