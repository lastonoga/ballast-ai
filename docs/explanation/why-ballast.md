# Why Ballast

## Mission

**Ballast exists to make production-grade agentic LLM applications a composition task, not an infrastructure project.**

You shouldn't have to hand-roll workflow durability, circuit breakers, goal-drift detection, HITL approval flows, and typed handoffs every time you ship an agent. You should write the *interesting* part — the agent's tools, its system prompt, the business logic — and let the framework do the rest.

That's it. That's the whole pitch.

## Why this matters now

The industry has converged on a sobering observation: **agentic systems fail in production not because foundation models are insufficiently smart, but because the infrastructure scaffolding around them is missing or under-built.**

From the article *"Архитектура и надёжность агентных LLM-систем в Production"*:

> "К 2027 году более 40% проектов в области агентного ИИ будут свернуты. Причиной этих отказов выступает не недостаток интеллектуальных способностей самих LLM, а отсутствие строгой инженерной дисциплины в проектировании процессов, интеграции внешних инструментов и оркестрации многошаговых рабочих процессов."

> "Лишь 11% организаций успешно внедрили агентов в полноценную production-среду, в то время как 38% застряли на стадии пилотирования, не имея возможности масштабировать решения из-за их непредсказуемости."

The **compounding error problem** is the simplest way to grasp this. If each LLM step succeeds 85% of the time, a 10-step workflow succeeds **20%** of the time. To climb out of that hole, you need *more than one* mechanism per concern: budget guards, loop detectors, goal-drift judges, circuit breakers, HITL escalation, structured contracts, durable replay. Each addresses a different failure mode. Each is non-trivial to build correctly.

Ballast supplies them, opinionated and ready, but assembled from Protocols so apps can swap any of them.

## Three core beliefs

### 1. The LLM is a *probabilistic coprocessor* inside a strict caracass

> "Сдвиг парадигмы: от отношения к LLM как к всемогущему оракулу к отношению к LLM как к ненадежному сопроцессору, который должен быть заключен в жесткий каркас верифицируемого кода."

The carcass is:
- **Typed data contracts** (`pydantic-ai` structured outputs, `Scored[T]`, `Ref[T]`)
- **Durable workflows** (`DBOS` — crash-safe by default)
- **Capabilities** that observe + correct (`BudgetGuard`, `GoalDriftDetector`, `SemanticLoopDetector`)
- **Resilience primitives** (`CircuitBreaker`, retries with backoff)
- **HITL channels** for the cases where automation isn't enough (`UICardChannel`, `ThreadChannel`)

### 2. Patterns over monoliths

> "Никогда не поручайте одной модели всю задачу целиком. Монолитные промпты мертвы. Сложные процессы должны быть разбиты на узлы направленного ациклического графа (DAG)."

Ballast ships first-class composable patterns: `MapReduce` (sharded extraction), `Reflection` (writer-critic-refiner), `DivergentConvergent` (CREATIVEDC for variety), `PlanAndExecute` (planner-driven DAG). Each is `@Durable.workflow`-wrapped, observable, and composable.

### 3. Customization everywhere via Protocols

Every framework decision is a `Protocol` or a callable hook. Strategy for "when to fire the drift judge"? `DriftCheckStrategy`. Threshold for circuit breaker? `ThresholdPolicy`. Where to find episodes? `EpisodicSource`. The framework ships built-ins; apps swap any of them.

## What we don't believe

- ❌ **Heavyweight per-feature facades.** We deleted our own Episodic/Semantic Memory facades (Phase 1+2 of CoALA) in favor of a single `CoALAUnit` Protocol + adapters when we realized apps wanted to own storage themselves.
- ❌ **"Just trust the LLM."** Every output passes through some validator (typed schema, confidence threshold, judge verdict, HITL gate) before it affects state outside the agent's transient context.
- ❌ **Magic registration.** No metaclass tricks, no global registries. Apps wire dependencies explicitly through fluent setters or constructor injection. The single exception is `__hitl_kind__` registry — and even that's explicit.
- ❌ **Framework-owned schema.** Apps own their domain models. Framework only knows about `Thread`, `Message`, `ApprovalCard` because those are the universal contract surfaces.

## How to know if Ballast is for you

**Good fit:**
- Multi-step agentic workflows (not just one-shot prompts)
- Need durability across crashes
- Need HITL gates for high-stakes actions
- Need observable tracing through agent steps
- Comfortable with pydantic-ai / DBOS / FastAPI ecosystem

**Probably not a fit:**
- Pure RAG / chat (overkill — use vanilla pydantic-ai)
- Single-LLM-call apps with no tools (overkill)
- You need a different model SDK (Ballast leans on pydantic-ai's provider abstraction)
- You need a different workflow engine (Ballast leans on DBOS)

## Next

- [article-pain-points.md](article-pain-points.md) — concrete mapping from production pain to framework solutions
- [architecture-overview.md](architecture-overview.md) — stack diagram + layer responsibilities
- [customization-everywhere.md](customization-everywhere.md) — how Protocol-first design lets you swap any piece
