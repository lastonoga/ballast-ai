# How-to guides

Find the pain you're having. Click the recipe. Copy the code.

> **You're here because:** you have a specific symptom and need the fix. Skip the explanation — read it later in [explanation/](../explanation/) if you want to understand why.

---

## "My agent is burning my budget"

You hit OpenAI rate limits, your token spend tripled overnight, or your agent loops on the same tool until your CI job dies.

| Pain | Recipe |
|---|---|
| Agent loops forever, ignoring step counts | [cost-control/cap-tokens-and-iterations.md](cost-control/cap-tokens-and-iterations.md) |
| Agent stays in scope but reasoning slowly drifts off-task | [cost-control/detect-goal-drift.md](cost-control/detect-goal-drift.md) |

→ Underlying primitives: `BudgetGuard`, `SemanticLoopDetector`, `TypedLoopGuard`, `GoalDriftDetector`.

---

## "Things break in production; I need confidence in my changes"

External APIs go down, deployments break agent behaviour in subtle ways, you can't run tests without paying OpenAI.

| Pain | Recipe |
|---|---|
| Flaky external API → cascading retries → budget gone | [reliability/handle-flaky-external-api.md](reliability/handle-flaky-external-api.md) |
| Unit tests need real LLM calls; slow + expensive | [reliability/test-without-real-llm.md](reliability/test-without-real-llm.md) |
| Workflow uses `@Durable.workflow`; can't test locally | [reliability/test-durable-workflows.md](reliability/test-durable-workflows.md) |
| Need to test CoALA unit phases without DBOS/LLM | [reliability/test-coala-units.md](reliability/test-coala-units.md) |

→ Underlying primitives: `CircuitBreaker`, pydantic-ai `TestModel`, DBOS SQLite fixture.

---

## "I need a human in the loop, not just YOLO automation"

The agent is about to do something irreversible (publish, send email, transfer funds). Or it needs information only the user knows.

| Pain | Recipe |
|---|---|
| Tool is dangerous; should require approval before running | [trust-and-safety/require-approval-for-dangerous-tools.md](trust-and-safety/require-approval-for-dangerous-tools.md) |
| Agent needs to ask user a structured clarifying question mid-conversation | [trust-and-safety/ask-user-clarifying-questions.md](trust-and-safety/ask-user-clarifying-questions.md) |
| Approvals get lost on process restart; need audit trail | [trust-and-safety/audit-trail-of-approvals.md](trust-and-safety/audit-trail-of-approvals.md) |

→ Underlying primitives: `ApprovalCapability` (auto-bridges `requires_approval=True`), `UICardChannel`, `HelperAgent`, `SqlApprovalCardRepository`.

---

## "My agent makes up things"

Agent invents UUIDs, hallucinates dates, returns confident-sounding garbage. You need typed, validated, grounded outputs.

| Pain | Recipe |
|---|---|
| LLM hallucinates entity IDs (Note.id, User.id, ...) | [data-quality/prevent-id-hallucination.md](data-quality/prevent-id-hallucination.md) |
| Tool / agent outputs lack a quality signal — can't filter / rank | [data-quality/add-confidence-to-outputs.md](data-quality/add-confidence-to-outputs.md) |

→ Underlying primitives: `Ref[T]` + `GroundedAgent`, `Scored[T]`.

---

## "My input is too big, OR my output is too monotonous"

Documents exceed context, agent generates predictable safe answers, drafts are mediocre and need refinement.

| Pain | Recipe |
|---|---|
| Document doesn't fit in context window ("Lost in the Middle") | [scaling-context/process-large-documents.md](scaling-context/process-large-documents.md) |
| Agent produces homogeneous outputs — need broad exploration first | [scaling-context/explore-then-synthesize.md](scaling-context/explore-then-synthesize.md) |
| First drafts are OK but not great — need writer/critic/refiner loop | [scaling-context/iterate-with-self-critique.md](scaling-context/iterate-with-self-critique.md) |

→ Underlying primitives: `MapReduce`, `DivergentConvergent`, `Reflection`.

---

## "I have a multi-step task, not one shot"

The task needs planning, dependencies, parallel branches, sub-steps. ReAct doesn't scale.

| Pain | Recipe |
|---|---|
| Need a planner-first → executor approach with typed DAG | [multi-step-orchestration/plan-then-execute.md](multi-step-orchestration/plan-then-execute.md) |
| Want one piece of memory-aware logic deployed as a tool / workflow / capability without rewriting | [multi-step-orchestration/build-cognitive-units.md](multi-step-orchestration/build-cognitive-units.md) |

→ Underlying primitives: `PlanAndExecute`, `CoALAUnit` + adapters.

---

## "I have no idea what my agent is actually doing"

Production goes silent. Outputs look fine but you can't tell which step is slow, what cost what, or whether quality is degrading.

| Pain | Recipe |
|---|---|
| Need end-to-end traces: workflow → agent → tools → tokens → cost | [observability-and-evals/add-tracing.md](observability-and-evals/add-tracing.md) |
| Need automatic quality grading on every response | [observability-and-evals/grade-outputs-continuously.md](observability-and-evals/grade-outputs-continuously.md) |
| Need to A/B test a new agent against last week's real user inputs | [observability-and-evals/replay-traces-for-regression.md](observability-and-evals/replay-traces-for-regression.md) |

→ Underlying primitives: `logfire`, `@traced`, `LLMJudge` + `JudgeAfterRun`, `Dataset` + `Scorer`.

---

## "State disappears on restart"

Thread history is in memory. HITL cards get lost. You restart the API and conversations vanish.

| Pain | Recipe |
|---|---|
| Threads + messages need to survive restarts | [state-persistence/persist-conversations.md](state-persistence/persist-conversations.md) |
| Approval cards (pending decisions) need to survive restarts | [trust-and-safety/audit-trail-of-approvals.md](trust-and-safety/audit-trail-of-approvals.md) |

→ Underlying primitives: `SqlThreadRepository`, `SqlApprovalCardRepository`, included Alembic migrations.

---

## Don't see your pain?

- Look at [explanation/article-pain-points.md](../explanation/article-pain-points.md) — every pain from the production-failures article mapped to a solution.
- Check [reference/](../reference/) — the API surface.
- Open an issue if the pain is real and the solution isn't here yet.

## Recipe shape (so you know what to expect)

Every recipe is:
1. **Pain** — the symptom, one paragraph
2. **Minimum** — copy-pasteable code that solves it
3. **Variations** — common follow-up needs (custom thresholds, fallback chains, etc.)
4. **Bridges** — how this combines with other primitives
5. **Caveats** — gotchas, what NOT to do
6. **Related** — adjacent recipes
