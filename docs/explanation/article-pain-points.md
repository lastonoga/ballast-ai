# Article Pain Points → Ballast Solutions

This document maps every concrete production pain identified in *"Архитектура и надёжность агентных LLM-систем в Production"* to the Ballast primitive that addresses it. Pain on the left, solution on the right, working code snippets at the bottom of each row.

> If you've read the article and want to know "OK but how do I actually solve X with this framework?" — this is the page.

---

## Fundamental failure modes

### 1. Compounding error problem

**Pain (article):** "Если точность выполнения каждого отдельного шага агентом составляет 85%, то вероятность успешного завершения 10-шагового алгоритма падает до 20%."

**Solution:** Stack **multiple** orthogonal capabilities + patterns + resilience primitives. Each addresses a different failure mode.

```python
agent = Agent(
    model="openai:gpt-4o",
    capabilities=[
        BudgetGuard(max_iterations=10, max_input_tokens=20_000),
        SemanticLoopDetector(embedder=my_embedder),
        TypedLoopGuard(output_type=ResearchSummary),
        GoalDriftDetector(DriftEngine(...)),
        ApprovalCapability(tool_card_map={...}),
    ],
)
```

Five capabilities, five independent guards. The math: if each catches 80% of its class of failures, your composite reliability climbs to ~99.97% per step.

### 2. Brittle connectors / schema drift

**Pain:** "Генерируемые схемы инструментов стандарта MCP теряли ключи типизации. Массивы данных внезапно теряли поле items."

**Solution:** Typed contracts everywhere via `pydantic-ai` structured outputs + framework-side wrappers like `Scored[T]` and `Ref[T]`.

```python
class ResearchOutput(BaseModel):
    summary: str
    citations: list[Ref[Source]]

agent = Agent(model=..., output_type=Scored[ResearchOutput])
# Pydantic validates on receipt; any drift fails fast with a typed error.
```

### 3. Polling trap

**Pain:** "Агент тратит весь свой лимит токенов на постоянный опрос внешнего сервиса в ожидании изменения состояния."

**Solution:** Event-driven via `Durable.recv_async` + Signal channels. HITL waits ≠ polling.

```python
# Instead of polling for approval:
verdict = await ui_card_channel.request(payload, timeout=timedelta(hours=4))
# Suspends workflow durably; resumes when human clicks approve/reject in UI.
# Zero polling. DBOS handles the durable wait.
```

---

## Convergence + completeness failures

### 4. Artificial Hivemind / premature convergence

**Pain:** "LLM фокусируется на первом найденном локальном оптимуме, отсекая глубокое семантическое исследование."

**Solution:** **`DivergentConvergent` pattern** (CREATIVEDC methodology) — separates wide divergent exploration from narrow convergent synthesis.

```python
pattern = DivergentConvergent(
    divergent_agent=hypothesis_generator,    # explores broad space
    convergent_agent=synthesizer,            # filters + refines
    branch_count=8,
    dedup_threshold=0.92,
)
result = await pattern.run(brief)
```

Optional `on_progress` callback streams typed events (`BranchEnqueued`, `DedupCompleted`, `ConvergeStarted`, …) to UI so users see what's happening.

### 5. Context truncation / "Lost in the Middle"

**Pain:** "Окно контекста 40K-160K токенов — попытка передать весь текст приводит к усечению + потере информации в середине."

**Solution:** **`MapReduce` pattern** — sharded extraction with bounded chunks + global reduce.

```python
mr = MapReduce(
    map_agent=extractor_per_chunk,    # processes one chunk, returns Scored[Fact]
    reduce_agent=synthesizer,         # aggregates filtered facts
    map_concurrency=8,
    collapse_threshold=20,            # collapses to intermediate digests if too many
)
summary = await mr.run(document_chunks)
```

Combined with `Scored[T]` (next), Map workers return `{value, rationale, confidence}` — reduce filters by confidence before synthesizing.

### 6. Confidence scoring (Map-phase quality signal)

**Pain:** "Воркер на этапе Map должен извлечь информацию + сформировать Rationale + присвоить Confidence Score от 1 до 5."

**Solution:** **`Scored[T, ConfidenceT]`** — typed wrapper with `value + rationale + confidence` (default labels `low / medium / high` to avoid mean-reversion).

```python
from ballast.quality.scored import Scored, filter_by_min_confidence, rank_by_confidence

async def map_extract(chunk) -> Scored[Fact]:
    return await extractor.run(chunk, output_type=Scored[Fact])

async def reduce_synthesize(items: list[Scored[Fact]]) -> Summary:
    kept = filter_by_min_confidence(items, "medium")
    ranked = rank_by_confidence(kept)
    return await summarizer.run(prompt_with(ranked))
```

---

## Loop / runaway protection

### 7. Loop-happiness / token-burning

**Pain:** "Агент застрял в рекурсивном тупике, бесконечно повторяя один и тот же вызов инструмента."

**Solution:** Three orthogonal guards.

```python
agent = Agent(model=..., capabilities=[
    BudgetGuard(max_iterations=20, max_input_tokens=50_000),
    SemanticLoopDetector(embedder=embedder, threshold=0.95, window=3),
    TypedLoopGuard(output_type=MyOutput),
])
```

- `BudgetGuard` — hard cap on iterations + tokens
- `SemanticLoopDetector` — embedding-based detection of repeated outputs within a run (cosine sim ≥ 0.95)
- `TypedLoopGuard` — detects when typed outputs converge between Pattern iterations (e.g. Reflection critic-refiner cycle)

### 8. Anti-recursion in reflection loops

**Pain:** "Генератор и критик попеременно отвергают выводы друг друга, сжигая тысячи долларов за минуты."

**Solution:** `Reflection` pattern bakes in iteration cap + uses `TypedLoopGuard` automatically.

```python
reflection = Reflection(
    writer=draft_agent,
    critic=critic_agent,
    refiner=refiner_agent,
    max_iterations=3,                     # hard cap
    output_type=ArticleDraft,
)
```

### 9. Circuit breakers + mandatory final states

**Pain:** "Превышение лимита повторных ошибок → деградация: детерминированный fallback / context refresh / эскалация человеку."

**Solution:** **`CircuitBreaker`** — classic 3-state machine (Closed / Open / Half-Open) with pluggable threshold + fallback.

```python
cb = CircuitBreaker(
    threshold_factory=lambda: WindowedRate(rate=0.5, window=timedelta(minutes=2), min_samples=10),
    fallback=Chain(
        CallFallback(cheap_model_fallback),
        EscalateToHITL(channel=ui_card_channel, card_factory=ServiceDownCard),
    ),
    scope_key=per_tool_scope,
    recovery_after=timedelta(seconds=30),
)
# Wrap any async call:
result = await cb.call(flaky_external_api, ctx={"tool_name": "search"})
# Or adapter:
@as_workflow_decorator(cb, scope_ctx={"tool_name": "publish"})
async def publish_post(input): ...
```

---

## Structured output discipline

### 10. JSON generation via "ask nicely" anti-pattern

**Pain:** "Генерация JSON через простой промпт недопустима в production."

**Solution:** Native `pydantic-ai` `output_type` for every agent. Framework wraps with `Scored[T]` / `Ref[T]` for richer contracts.

```python
agent = Agent(
    model=...,
    output_type=Scored[ResearchSummary],   # typed, validated, JSON-Schema-enforced
)
```

Pydantic-ai handles vendor-specific structured-output APIs (OpenAI response_format, Anthropic tool_use, etc.) — Ballast doesn't reinvent this.

### 11. Grounded references (anti-hallucinated IDs)

**Pain:** When agent must reference an entity (Note, User, Project), it tends to hallucinate UUIDs.

**Solution:** **`Ref[T]`** + `scan_output` schema narrowing — JSON Schema is dynamically narrowed to a `Literal` enum of valid IDs.

```python
class ResearchOutput(BaseModel):
    summary: str
    project: Ref[Project]   # framework narrows JSON Schema to enum of real project IDs

result = await grounded_agent.run(brief)
hydrated = await result.hydrate(project_repo=project_repo)
print(hydrated.project.name)
```

---

## Self-reflection / hallucination protection

### 12. Reflective loop (Writer-Critic-Refiner)

**Pain:** "Снижение токсичности 75.8%, рост качества 18.5 п.п. при правильной реализации саморефлексии."

**Solution:** **`Reflection` pattern**, built-in.

```python
reflection = Reflection(
    writer=draft_agent,
    critic=Agent(model=..., output_type=Critique),
    refiner=refiner_agent,
    max_iterations=3,
    output_type=ArticleDraft,
)
final = await reflection.run(brief)
```

### 13. Self-RAG / CRAG (corrective retrieval)

**Status:** Identified as a pain in the article ("Self-RAG / Corrective RAG"). Not yet implemented as a first-class primitive. Apps wire it manually inside `CoALAUnit.retrieve()` + `act()` today; framework will add an explicit `RetrievalQualityGuard` in a future iteration.

---

## Multi-agent / DAG orchestration

### 14. ReAct doesn't scale — Plan-then-Execute does

**Pain:** "ReAct вызывает дорогую LLM на каждый мелкий шаг и планирует на одну итерацию вперёд."

**Solution:** **`PlanAndExecute`** pattern. Planner agent emits a typed DAG (`Plan`); framework dispatches each step via `StepRegistry` (LLM / callable / CoALA unit / sub-workflow / custom).

```python
registry = StepRegistry.with_defaults()
registry.register_agent("researcher", researcher_agent)
registry.register_unit("summarize", SummarizeUnit())
registry.register_workflow("publish", publish_workflow)

pattern = PlanAndExecute(planner=planner_agent, registry=registry)
outputs = await pattern.run({"topic": "ML safety"})
```

Wave-by-wave DAG execution with `asyncio.gather` + semaphore. `@Durable.step`-memoised steps replay correctly across crashes.

### 15. Broken handoffs

**Pain:** "Агент A передаёт агенту B неструктурированный текст; B теряет контекст."

**Solution:** Typed `Step.execute(plan_input, dep_outputs, ctx)` contract in `PlanAndExecute`. Sub-step outputs are typed dictionaries keyed by step.id; downstream steps reference upstream values explicitly.

```python
PlannedStep(
    id="summary",
    kind="llm",
    depends_on=["research"],
    params={"prompt_template": "Summarize: {research.findings}"},
)
```

Framework also exposes `ThreadEvent` / `DBOSSignal` typed channels for between-agent communication.

### 16. Semantic routing (cost-based dispatch)

**Status:** Not yet implemented. Article recommends embedding-based intent classification routing cheap queries to RAG / cheap models, escalating only ambiguous ones to a tier-1 LLM. Ballast has `Embedder` Protocol ready; pattern is a thin wrapper away. Future iteration.

---

## Observability + drift control

### 17. Distributed tracing

**Pain:** "Сбои агентов протекают бесшумно — через плавное снижение качества рассуждений, раздувание памяти."

**Solution:** `logfire` + OTel integration baked in. `@traced` decorator on workflows + per-step DBOS spans give end-to-end visibility.

```python
from ballast import traced

@traced(name="my-step")
async def custom_logic(input): ...
```

### 18. LLM-as-a-Judge

**Pain:** "Для непрерывной автоматизированной валидации применяются специально дообученные LLM-судьи."

**Solution:** `LLMJudge` + `JudgeAfterRun` capability + `SchemaAdherenceScorer` for evals.

```python
judge = LLMJudge(
    rubric="Output must cite at least one source and avoid speculation.",
    threshold=0.7,
    sync=False,    # async — doesn't block user reply
)
agent = Agent(model=..., capabilities=[JudgeAfterRun(judge=judge, on_verdict=persist)])
```

### 19. Goal drift detection

**Pain:** "Логика принятия решений на последних шагах перестаёт опираться на изначальные инструкции."

**Solution:** **`GoalDriftDetector`** capability + `with_drift_monitor` decorator.

```python
detector = GoalDriftDetector(DriftEngine(
    strategy=EveryNToolCalls(5),
    window=LastNMessages(10),
    goal_source=FirstUserMessage(),
    judge=make_default_judge(),
    handlers=[EmitDriftEvent(sink=thread_event_publisher), EscalateToHITL(channel=ui)],
))

agent = Agent(model=..., capabilities=[detector])
```

`DriftCheckStrategy` / `TraceWindow` / `GoalSource` / `DriftHandler` are all Protocols — swap any of them.

### 20. Statistical data drift

**Status:** Not yet implemented as primitive. Framework exposes `Embedder` Protocol; apps build baseline embeddings + measure incoming traffic. Future addition.

---

## Human-in-the-Loop (HITL)

### 21. Exception escalation, not routine review

**Pain:** "Эскалация к человеку только по программному триггеру: confidence < threshold, policy conflict, financial limit exceeded."

**Solution:** **`UICardChannel`** + **`ThreadChannel`** + **`ApprovalCard`** persistence + **`ApprovalCapability`** (auto-bridges pydantic-ai `requires_approval=True` to cards).

```python
@agent.tool(requires_approval=True)
async def publish_post(title: str, body: str) -> str:
    # Tool body — only runs if human approves
    return await blog.publish(title, body)

approval_cap = ApprovalCapability(tool_card_map={
    "publish_post": (PublishCard, lambda tc, deps: PublishCard(**tc.args), ui_channel),
})
agent = Agent(model=..., capabilities=[approval_cap])
```

Verdict shapes (`approve` / `reject` / `modified`) map cleanly to pydantic-ai's `ToolApproved(override_args=...)` / `ToolDenied(message=...)`. The `modified` field lets the human edit args before approval.

### 22. Guidance, not approval

**Pain:** "Вместо примитивной верификации готового ответа, агент идентифицирует точку принятия стратегического решения и запрашивает у человека направление."

**Solution:** `HelperAgent` + `HelperSessionRunner` + `ConversationalChannel`. Agent opens a side-thread, asks human a question, gets a typed `HelperVerdict[ContextT]`, continues.

```python
verdict = await hitl_gate.ask_helper(
    helper_agent=clarifier_agent,
    context=current_state,
)
# Human chats with helper in a separate panel; verdict comes back typed.
```

### 23. Accountability gateway

**Status:** Documented in article (cryptographic confirmation for high-risk finance/medical use cases). Out of scope for current framework — apps build on top of `ApprovalCard` if needed (e.g., add signature field + verify in repository before commit).

### 24. Instrumented feedback

**Pain:** "Каждое вмешательство человека → reusable evaluator."

**Solution:** `ApprovalCard` rows persist with full verdict + reasoning. Eval framework (`Dataset`, `EvalCase`) can replay against captured verdicts. Future: automatic clustering of `ApprovalCard.feedback` for prompt-tuning hints.

---

## Cognitive architecture

### 25. CoALA-style memory + decision procedure

**Pain (CoALA paper, indirectly referenced by article's general "memory" concerns):** Agents need observe/retrieve/act/learn phases that integrate with workflows AND tools AND capabilities.

**Solution:** **`CoALAUnit`** Protocol + `CoALABase` ABC + 3 adapters (`as_workflow` / `as_tool` / `as_capability`). One contract, three runtime surfaces. Apps own all storage.

```python
class ResearchUnit(CoALABase[Query, Observation, Context, Summary]):
    async def observe(self, q): return Observation(intent=q.text)
    async def retrieve(self, obs): return await my_kb.search(obs.intent)
    async def act(self, obs, ctx): return await summarizer.run(prompt_with(ctx))
    async def learn(self, obs, ctx, out): await my_episodes.write(...)

# Same unit, three deployments:
agent.tools = [as_tool(ResearchUnit())]
flow = as_workflow(ResearchUnit())
agent.capabilities = [as_capability(ResearchUnit())]
```

---

## Out of scope (deliberately deferred)

Article-mentioned items NOT in current framework:
- **Schema-Aligned Parsing (BAML)** — alternative to pydantic for ultra-token-efficient parsing. We use pydantic-ai which covers 90% of cases; BAML integration is a 3rd-party concern apps can wire if they want.
- **Constrained Decoding (Outlines/XGrammar)** — requires local model + low-level access. Apps using local vLLM/llama.cpp can wire it; not a framework primitive.
- **Prometheus 2 judge models** — apps may use any judge model via `LLMJudge(model=...)`.
- **Persistent CB state (Redis/Postgres)** — CB state is in-memory per process. Cluster-aware CB is a future addition.
- **GraphBit-style Rust execution engine** — Python performance limits exist; if/when needed apps shard via DBOS queues.
- **MARA (Microsoft Multi-Agent Reference Architecture) Onion layers** — Ballast's package structure (`capabilities/` / `patterns/` / `resilience/` / `coala/` / `runtime/` / `persistence/`) follows the same separation-of-concerns spirit but doesn't replicate MARA verbatim.
