# pydantic-ai-stateflow-engine — Design Spec

- **Date:** 2026-05-15
- **Status:** Section 1 (Foundation) approved. Sections 2–3 TBD.
- **Authors:** Kir + Claude (brainstorming session)
- **Source material:** `.context/attachments/pasted_text_2026-05-15_15-09-12.txt` (architectural analysis of LLM production failures, RU)

---

## Section 1 — Foundation (approved)

### 1.1 Vision

`pydantic-ai-stateflow-engine` — production-grade orchestration framework для агентов, превращающий LLM из "ненадёжного оракула" в типизированное, durable, аудируемое когнитивное ядро. Опирается на готовые примитивы pydantic-ai (Capabilities, Hooks, Embedder, DeferredToolRequests, pydantic-evals, DBOS plugin) и добавляет:

1. **GroundedSchema (L0)** — type-driven input-bound output schemas, физически предотвращающие галлюцинации UUID/refs через `Ref[Entity]`-типы
2. **Patterns (L2)** — durable multi-step workflow-классы (MapReduce, Reflection, MutationPipeline, HITLGate, SelfRAG)
3. **MutationPipeline** — обязательный конвейер write-flow, гарантирующий что LLM никогда не пишет в state напрямую

Цель — устранить наиболее частые production-failures, описанные в исходном документе: schema drift, hallucinated refs, infinite loops, token burning, goal drift, broken handoffs, недетерминированную мутацию state.

### 1.2 Architecture Layers

```
┌──────────────────────────────────────────────────────────────┐
│ L7  API & Streaming │ FastAPI (REST + SSE)                   │
│                     │   • REST: /threads, /chats,            │
│                     │           /messages, /runs, /proposals,│
│                     │           /hitl                        │
│                     │   • Streaming: AG-UI protocol          │
│                     │   • Streaming: Vercel AI SDK DataStream│
│                     │   • Auth: Depends → actor              │
│                     │   • Authz: Voter / Policy gate         │
│                     │   • DBOS-FastAPI integration           │
├──────────────────────────────────────────────────────────────┤
│ L6  Observability   │ logfire spans, drift dashboards,       │
│                     │   eval-from-trace tooling              │
├──────────────────────────────────────────────────────────────┤
│ L5  Evals           │ pydantic-evals + custom Scorers        │
│                     │   (SchemaAdherence, MutationAcceptance,│
│                     │    IterationBudget, GroundedReference) │
├──────────────────────────────────────────────────────────────┤
│ L4  State           │ SQLModel (Persistence) + Pydantic       │
│   (Postgres)        │   (Domain) + Repository (port)         │
│                     │   • infra:  threads, chats, messages,  │
│                     │             checkpoints, outbox,       │
│                     │             drift_metrics, eval_runs,  │
│                     │             hitl_requests,             │
│                     │             hitl_responses             │
│                     │   • domain: <user-defined entities>    │
│                     │   • Alembic для миграций               │
├──────────────────────────────────────────────────────────────┤
│ L3  Durable Runtime │ DBOS workflow/step boundary,           │
│                     │   queues, DBOS.recv() для HITL,        │
│                     │   recovery                             │
├──────────────────────────────────────────────────────────────┤
│ L2  Patterns        │ Composable durable workflows:          │
│                     │   MapReduce, Reflection,               │
│                     │   MutationPipeline, SelfRAG,           │
│                     │   CorrectiveRAG, DivergentConvergent,  │
│                     │   PlanAndExecute, SemanticRouter,      │
│                     │   HITLGate ──→ uses HITLChannel port   │
│                     │                                         │
│                     │ HITLChannel port (Strategy pattern):    │
│                     │   • ThreadToolChannel (DeferredTool)    │
│                     │   • UIChannel (FastAPI + SSE)           │
│                     │   • SlackChannel (slack-sdk + webhook)  │
│                     │   • WebhookChannel (generic HTTP)       │
│                     │   • EscalationChannel (compound)        │
│                     │   • FakeChannel (tests)                 │
├──────────────────────────────────────────────────────────────┤
│ L1  Capabilities    │ Plug-style per-run middleware:         │
│                     │   SemanticLoopDetector, BudgetGuard,   │
│                     │   GoalDriftDetector, LLMJudgeHook,     │
│                     │   PIIGuard, GroundedRetry              │
├──────────────────────────────────────────────────────────────┤
│ L0  GroundedSchema  │ Type-driven binding via Ref[EntityT].  │
│                     │ Resolver сканирует output_type,        │
│                     │ собирает T-instances из context →      │
│                     │ создаёт DynamicOutput через create_model│
│                     │ с Literal[*ids] вместо UUID.           │
│                     │ Escape hatch: constraints={...} в run()│
│                     │ Hydration: ref.hydrate(repo) → Entity  │
└──────────────────────────────────────────────────────────────┘
```

**Вертикальная ось:**
- L0 ↔ один `agent.run()`
- L1 — внутри `agent.run()` (Plug-chain через pydantic-ai Capabilities)
- L2 — multi-step workflow (Pipeline of `@DBOS.step` calls)
- L3 — runtime для L2 (DBOS), обеспечивает durability
- L4–L7 — пересекают все слои

### 1.3 Vocabulary

| Термин | Источник | Что значит у нас |
|---|---|---|
| **Capability** | pydantic-ai (нативно) | reusable bundle: tools + hooks + instructions + model settings |
| **Plug** | Phoenix (mental model) | способ думать про Capability: `RunState → RunState` typed transform |
| **Pipeline** | Laravel | первичный примитив multi-step (Chain of Responsibility): `handle(passable, next)` |
| **Pattern** | (наш термин) | high-level Pipeline-based workflow, обёрнутый в `@DBOS.workflow` |
| **Workflow / Step** | Temporal / DBOS | граница детерминизма: workflow = pure orchestration, step = side effect |
| **Proposal[T]** | (наш термин) | непримененная LLM-мутация, проходящая через MutationPipeline |
| **Policy** | Laravel | object-level authz rule: `Policy.can(actor, action, resource) → bool` |
| **Voter** | Symfony | fine-grained authz decision: `vote(...) → GRANT | DENY | ABSTAIN` |
| **Service Provider** | Laravel | bootstrap-класс: `register()` (binds в DI) + `boot()` (init) |
| **Event Dispatcher** | Symfony | pub-sub для lifecycle событий, не для control flow |
| **Depends** | FastAPI | function-signature DI для Pattern-зависимостей (agents, validators) |
| **GroundedSchema** | (наш термин) | type-driven output binding через `Ref[EntityT]` |
| **Ref[T]** | (наш термин) | typed UUID reference на Entity типа T; LLM видит как string, Python — как объект |
| **HITLChannel** | (наш термин) | Strategy port для запроса решения у человека (Slack/UI/inline tool/webhook) |
| **Bounded Context** | DDD / Phoenix | пакетная единица домена: `agents/<context>/{patterns, capabilities, models}` |

### 1.4 Core Principles (17 final)

1. **Explicit DI > Global State** — никаких Laravel-Facades, Django settings, `get_llm_client()` globals. Зависимости — через function signatures (`Depends`) или конструкторы.
2. **Composition > Inheritance** — Capabilities/Patterns собираются списком. Никаких CBV mixins.
3. **Explicit Events > Implicit Signals** — `EventDispatcher` для observability; control flow всегда видим в коде Pattern.
4. **Pipelines as Primitive** — Pattern (MutationPipeline, Reflection) — это Pipeline. Каждый stage явный, заменяемый, тестируемый.
5. **Policy + Voter Authorization** — Policy = высокоуровневое правило; Voter = композируемое решение. Используется в MutationPipeline.PolicyCheck.
6. **State has Two Sides: Persistence rows vs Domain models.** Persistence — SQLModel `table=True` в `persistence.py`. Domain — Pydantic в `domain.py` (могут совпадать для простого CRUD). **Бизнес-логика никогда на persistence-классах.** Repository — единственный мост; импорт persistence-классов вне `repositories/` — code smell, ловится линтером.
7. **Workflow/Step Determinism Boundary** (Temporal rule) — все side effects (`agent.run()`, DB-вызовы, tool-calls) в `@DBOS.step`. `@DBOS.workflow` — только orchestration.
8. **DAG composition, no cycles** — capabilities composing topologically; фреймворк ошибается рано на cycles.
9. **12-factor config** — `pydantic_settings.BaseSettings`, env-driven, никаких хардкодов.
10. **Bounded contexts as package structure** — `agents/<domain>/{patterns.py, capabilities.py, models.py, evals.py}` — публичный API через `__all__`.
11. **Repository Pattern as the only port** — все Patterns/Capabilities зависят от Repository-протоколов, не от SQLAlchemy. Даёт тестируемость, доменную чистоту, заменяемость, транзакционность.
12. **Thin API layer** — FastAPI endpoints только парсинг + Pattern.run() / Repository.query() + streaming. Никакой бизнес-логики.
13. **Streaming-first для chat UX** — все chat-эндпоинты через `agent.run_stream_events()`. AG-UI и Vercel — два адаптера формата.
14. **Thread = бизнес-объект, не транспорт** — Thread имеет идентичность, привязан к актору, владеет историей и текущим `workflow_id`. Возобновление = reattach к DBOS-workflow.
15. **Authentication via Depends, Authorization via Voter/Policy** — Auth (кто?) в FastAPI Depends; authz (что можно?) в Voter/Policy. Одна Policy переиспользуется в HTTP-layer и в MutationPipeline.PolicyCheck.
16. **HITL is a Channel, not a Mechanism** — Pattern не знает где ждёт человек, только то что нужно `HITLDecision`. Канал инжектится.
17. **Type-driven closed sets** — Closed-set гарантия выражается через тип `Ref[EntityT]` в output template. Фреймворк собирает все T-instances из context и формирует Literal автоматически. Никаких path-strings, глобальных имён, Annotated-маркеров. Escape hatch `constraints={"path": values}` остаётся для редких case override / ad-hoc подмножеств.

### 1.5 No-Buy List (что мы осознанно НЕ делаем)

**Из Django / DRF:**
- Signals (`post_save` и т.п.) — implicit control flow
- Class-Based Views + Mixins — inheritance-based composition не нужна
- ViewSets / REST routing — агенты не HTTP handlers
- Renderers / Parsers (content negotiation) — Pydantic нативно
- Django settings module — global state, mock-hostile
- ContentType generic FK — over-abstraction
- DRF Throttling per-endpoint — DBOS concurrency model

**Из Rails:**
- ActiveRecord (domain + persistence в одном) — SRP violation
- Observers — implicit
- Engines — overkill
- Heavy Generators — boilerplate > code
- ActiveJob — DBOS закрывает

**Из Laravel:**
- Facades — global service locator
- Macros (monkey-patch) — implicit
- Eloquent Observers — implicit

**Из Symfony:**
- Form Component (CSRF) — web-specific

**Из Spring / NestJS:**
- AOP proxies / runtime reflection — Capabilities делают то же явно
- Heavy decorator metadata (XML, `@Module` стеки) — избыточно
- Implicit DI scopes — DBOS владеет scope-границей

**Общие:**
- Convention-over-configuration "магия" — safety-critical flows должны быть явными
- Global service locators — Dependencies в function signatures
- Cyclic capability deps — DAG-only

**ORM-специфичные:**
- Импорт `*Row` (SQLModel `table=True`) в Pattern / Capability / endpoint — нарушает Repository-port; ловится линтером
- Бизнес-методы на SQLModel-row классах — ActiveRecord
- Сырые SQL-запросы из Pattern-кода — через Repository (`repo.query(spec)`)

**FastAPI / streaming:**
- Бизнес-логика в endpoint — endpoint тонкий
- Своя реализация AG-UI / Vercel сериализации — есть нативные `to_ag_ui()` / `to_vercel_ai_sdk()`
- Прямой `Agent.run()` из endpoint без Pattern-обёртки — теряем durability/observability
- WebSocket для streaming — SSE проще, нативно для AG-UI/Vercel
- Polling endpoint вместо SSE — выгорание токенов

**HITL:**
- Hard-coded `if slack_enabled: ...` в Pattern — Channel port решает
- Channel со state в process memory — state живёт в Postgres
- Polling `/hitl/responses` из Pattern — resume только через `DBOS.recv`
- Прямой `agent.run()` с tool approval без `HITLGate` Pattern — теряем audit trail

**GroundedSchema:**
- Path-strings как primary API — не рефакторятся, нет статической проверки
- Annotated со строковыми именами/путями — имена в строках хрупкие
- Авто-сканирование по convention имён — implicit
- Глобальные binding tokens — не локальны, не рефакторятся
- Свободный `UUID` в output как ссылка без `Ref[T]` — теряем guarantee
- Пост-валидация output по списку IDs — поздно; должна быть в момент генерации через Literal
- Передача в context сырых dict вместо Pydantic-моделей — теряем type-driven resolver

### 1.6 Tech Stack (final)

| Слой | Технологии |
|---|---|
| Agent runtime | `pydantic-ai` |
| Type system & domain | `pydantic` 2.x |
| ORM + Pydantic мост | `sqlmodel` (Pydantic API + SQLAlchemy под капотом) |
| Pure SQLAlchemy | `sqlalchemy[asyncio]` для advanced cases (polymorphic, custom types) |
| Migrations | `alembic` |
| Durable execution | `dbos-transact` (через `PydanticAIPlugin` от pydantic-ai) |
| Database | PostgreSQL |
| Embeddings | `pydantic_ai.Embedder` |
| Evals | `pydantic_evals` |
| Observability | `logfire` |
| Config | `pydantic-settings` |
| HTTP API | `fastapi`, `sse-starlette`, `uvicorn` |
| UI protocols | `pydantic-ai` нативные `to_ag_ui()` / `to_vercel_ai_sdk()` |
| Auth (caller identity) | FastAPI `Depends` (Bearer/OAuth) |
| Slack | `slack-sdk` (`SlackChannel`) |
| Webhook delivery | `httpx` (`WebhookChannel`) |
| CLI | `typer` |
| Linting | `ruff` + custom rule "no `*Row` outside `repositories/`" |
| Tests | `pytest`, `pytest-asyncio`, `testcontainers` (PG для интеграции) |

### 1.7 L0 GroundedSchema — финальный API

```python
# 1. Доменная модель — обычная Pydantic / SQLModel
class Candidate(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    label: str
    score: float

# 2. Context — обычный Pydantic
class Context(BaseModel):
    candidates: list[Candidate]
    customer: Customer
    user_query: str

# 3. Output template объявляет ссылки типом Ref[T]
from pydantic_ai_stateflow import Ref

class DecisionItem(BaseModel):
    candidate:   Ref[Candidate]      # ← typed reference
    new_status:  OrderStatus          # ← Literal/Enum — closed set из типа
    rationale:   str
    confidence:  float

class Decision(BaseModel):
    items: list[DecisionItem]
    overall_note: str

# 4. Запуск
result = await GroundedAgent(agent, output_type=Decision).run(
    context=ctx,
    instructions="Pick best candidates",
)

# 5. Под капотом:
#    a) scan output_type → найти все Ref[T] поля
#    b) для каждого Ref[T]: собрать T-instances из context
#    c) create_model: Ref[T] → Literal[*ids], рекурсивно для nested/list
#    d) agent.run(output_type=DynamicOutput)
#    e) Pydantic валидирует Literal → 0% галлюцинаций
```

**Ref[T] как Pydantic-тип:**

- На уровне JSON / LLM: строка UUID (без обёртки `{"id": ...}`)
- На уровне Python: типизированный объект `Ref[T]` с `.id: UUID`
- Hydration через `await ref.hydrate(repo)` → `EntityT`

**Resolver rules:**

| Контекст | Поведение |
|---|---|
| `Ref[Candidate]` + `list[Candidate]` в context | `Literal[*ids]` |
| `Ref[Candidate]` + одиночный `Candidate` | `Literal[id]` |
| `Ref[T]` + несколько источников T | union allowed-sets |
| `Ref[T]` + 0 источников | construction-time error |
| `list[Ref[T]]` | `list[Literal[*ids]]` (подмножество) |
| `Optional[Ref[T]]` | `Optional[Literal[*ids]]` |
| `Ref[A] \| Ref[B]` | union allowed-sets обоих |
| Literal/Enum поле в output, тот же enum в context | пересечение допустимых значений |
| Новая сущность (не ссылка) | `UUID = Field(default_factory=uuid4)` — не Ref |

**Escape hatch (опционально):**

```python
result = await GroundedAgent(agent, output_type=Decision).run(
    context=ctx,
    constraints={
        "items[*].candidate": [c.id for c in ctx.candidates[:3]],  # override
    },
)
```

Path-style для редких случаев override / ad-hoc подмножеств — не primary API.

### 1.8 HITLChannel — API skeleton

```python
class HITLPrompt(BaseModel):
    title: str
    context: str                                  # rich agent context
    decision_type: Literal[
        "approve_reject",
        "choose_option",                          # Guidance, Not Approval
        "free_text",
    ]
    options: list[HITLOption] = []
    actor_filter: ActorFilter                     # кто вправе ответить
    timeout: timedelta | None = None

class HITLResponse(BaseModel):
    decision: Literal["approved", "rejected", "option", "feedback", "timeout"]
    chosen_option_id: str | None = None
    feedback: str | None = None
    actor_id: str | None = None
    answered_at: datetime

class HITLChannel(Protocol):
    name: str
    async def ask(self, prompt: HITLPrompt, *, request_id: UUID) -> HITLResponse: ...

class HITLGate(Pattern[InT, OutT]):
    def __init__(self, *, channel: HITLChannel, voter: Voter, policy: HITLPolicy): ...

    @DBOS.workflow()
    async def run(self, prompt: HITLPrompt) -> HITLResponse:
        request_id = await self._persist_request(prompt)
        response = await self.channel.ask(prompt, request_id=request_id)
        if not self.voter.vote(response.actor_id, "decide", prompt).is_grant:
            raise HITLAuthzDenied(actor=response.actor_id)
        await self._persist_response(request_id, response)
        return response
```

**Built-in channels:**

| Channel | Механизм | DBOS-resume |
|---|---|---|
| `ThreadToolChannel` | `DeferredToolRequests` → tool call в текущем thread | `DBOS.recv(request_id)` |
| `UIChannel` | INSERT в `hitl_requests` + SSE event → frontend; POST `/hitl/{id}/respond` | `DBOS.send(request_id, response)` |
| `SlackChannel` | `chat.postMessage` со кнопками; webhook `/hitl/slack/callback` | `DBOS.send(...)` |
| `WebhookChannel` | POST на configured URL; callback на `/hitl/webhook/{request_id}` | `DBOS.send(...)` |
| `EscalationChannel` | composite `[Ch1 with timeout, Ch2 with timeout, ...]` | внутренний loop |
| `FakeChannel(answer=...)` | для тестов — мгновенно возвращает | синхронный |

### 1.9 MutationPipeline — концепт

LLM никогда не пишет в state напрямую. Любая мутация = `Proposal[T]` → конвейер:

```
LLM output ──► Proposal[T]
                 │
                 ▼
 ┌────────── @DBOS.workflow ──────────┐
 │ 1. Validate     (schema + types)    │  @DBOS.step
 │ 2. ResolveRefs  (FK → live entity)  │  @DBOS.step (Repository)
 │ 3. Dedup        (idempotency keys)  │  @DBOS.step
 │ 4. QualityGate  (confidence, biz)   │  @DBOS.step
 │ 5. PolicyCheck  (Voter / Policy)    │  @DBOS.step
 │ 6. ToCommand    (Proposal→Command)  │  pure
 │ 7. Apply        (state mutation)    │  @DBOS.transaction
 │ 8. EmitEvent    (outbox row)        │  same @DBOS.transaction
 └─────────────────────────────────────┘
                 │
                 ▼
   AcceptedResult | RejectedAt(stage, reason)
                 │
                 ▼ (опц.)
   feedback в LLM через ModelRetry для самокоррекции
```

Стадии — Protocols (`Validator`, `RefResolver`, `Deduper`, ...). Композиция через `&` / `|`. Apply + Emit в одной транзакции = transactional outbox pattern. Идемпотентность через `proposal.proposal_id = hash(input + LLM-output)`.

### 1.10 Patterns как DBOS-aware classes — концепт

```python
class Reflection(Pattern[InT, OutT]):
    def __init__(self, writer, critic, *, max_iterations=5, budget=None): ...

    @DBOS.workflow()
    async def run(self, task: InT) -> OutT:
        for i in range(self.max_iterations):
            draft = await self._write(task)          # @DBOS.step
            critique = await self._critique(draft)   # @DBOS.step
            if critique.passed: return draft
            task = task.with_feedback(critique)
        raise BudgetExhausted()
```

**Композиция Patterns:**
- Pattern может вызвать Pattern — child workflow в DBOS со своим recovery
- `EscalationChannel`-стиль каскад через композицию
- Token budget пробрасывается parent → child
- Cycle detection через `max_depth` + `SemanticLoopDetector` на child-spawn
- Idempotency каскадом через workflow_id = hash(parent + pattern + input)

### 1.11 Дата-флоу (общая диаграмма)

```
                            ┌───────────────┐
                            │  Client / UI  │ (Next.js, CopilotKit, etc.)
                            └───────┬───────┘
                                    │ HTTP + SSE
                            ┌───────▼───────────────┐
                            │  L7  FastAPI          │
                            │  • thin endpoints     │
                            │  • Depends → actor    │
                            │  • Voter/Policy gate  │
                            └───────┬───────────────┘
                                    │ Pattern.run() / Repository
                            ┌───────▼───────────────┐
                            │  L2  Patterns         │
                            │  (DBOS workflows)     │
                            └───┬────────────┬──────┘
                  Agent.run()   │            │ Repository.* (через port)
                                │            │
                  ┌─────────────▼──┐    ┌────▼────────────┐
                  │ L1 Capabilities│    │ L4 Repository   │
                  │ (in agent run) │    │ → SQLModel/     │
                  │                │    │   SQLAlchemy    │
                  └─────────────┬──┘    └────┬────────────┘
                                │            │ @DBOS.transaction
                  ┌─────────────▼──┐    ┌────▼────────────┐
                  │   LLM provider │    │   Postgres      │
                  └────────────────┘    └─────────────────┘
```

---

## Section 2 — Detailed Architecture (TBD)

Будет покрывать:

- **L0 implementation:**
  - `Ref[T]` как Pydantic-тип через `__get_pydantic_core_schema__`
  - Resolver: scan output_type → collect entity types → match в context → build dynamic model
  - `create_model` рекурсивно для nested / list
  - Hydration через Repository
  - Edge cases: union refs, partial entities, multiple sources, escape hatch path-grammar

- **L1 Capability catalog (со скелетами кода):**
  - `SemanticLoopDetector` — embedder + косинусное сходство выходов
  - `BudgetGuard` — token / iteration budget tracking
  - `GoalDriftDetector` — async LLM-judge на goal alignment
  - `LLMJudgeHook` — async output scoring → eval store
  - `PIIGuard` — `after_model_request` redaction
  - `GroundedRetry` — умный feedback в `ModelRetry`

- **L2 Pattern catalog (со скелетами кода):**
  - `MapReduce` — fanout через `DBOS.queue`, reduce с confidence-merge
  - `Reflection` — Writer/Critic/Refiner loop
  - `MutationPipeline` — все 8 стадий, Pipeline-композиция
  - `HITLGate` — обёртка вокруг HITLChannel
  - `SelfRAG`, `CorrectiveRAG` — RAG с self-evaluation
  - `DivergentConvergent` — CREATIVEDC pattern
  - `PlanAndExecute` — планировщик + исполнители
  - `SemanticRouter` — embedding-based intent classification

- **L3 Workflow/Step Boundary детально:**
  - Что обязательно в `@DBOS.step` vs `@DBOS.workflow`
  - Lint-rule для проверки
  - Idempotency и replay-safety
  - Child workflows, fanout через queues
  - HITL pause через `DBOS.recv()`

- **Bootstrap / Service Provider pattern:**
  - `Engine.register(ServiceProvider)` API
  - `register()` / `boot()` lifecycle
  - DI через FastAPI `Depends` + наш `Container`
  - Entry-points discovery (опционально)

- **Authorization (Policy + Voter):**
  - `Policy.can(actor, action, resource) → bool`
  - `Voter.vote(actor, action, resource) → GRANT|DENY|ABSTAIN`
  - `AccessDecisionManager` (Symfony-style tally)
  - Использование в `MutationPipeline.PolicyCheck` и в endpoints

- **Event Dispatcher:**
  - `Event` base + типизированные события (`AgentRunStarted`, `PatternStepCompleted`, `MutationApplied`, `HITLRequested`, ...)
  - `EventDispatcher.dispatch(event)` — sync для observability, async через outbox для side effects

---

## Section 3 — Infrastructure & Process (TBD)

Будет покрывать:

- **L4 State schema:**
  - SQLModel-классы для всех infra-таблиц (threads, chats, messages, checkpoints, outbox, drift_metrics, eval_runs, hitl_requests, hitl_responses)
  - Repository protocols + базовые реализации (`PostgresRepository`, `InMemoryRepository`)
  - Alembic setup
  - Indexes, partitioning гипотезы

- **L5 Evals:**
  - Кастомные Scorers (`SchemaAdherenceScorer`, `MutationAcceptanceScorer`, `IterationBudgetScorer`, `GroundedReferenceScorer`)
  - `Dataset` builders из production traces
  - CI integration

- **L6 Observability:**
  - Logfire instrumentation setup
  - Span conventions (Pattern, Step, Capability, Channel)
  - Pre-configured dashboards (drift, HITL latency, budget burn, schema-adherence)
  - Drift detection pipeline (embedding-based + statistical)

- **L7 FastAPI API surface:**
  - REST endpoints полный список (threads, chats, messages, runs, proposals, hitl, evals)
  - SSE streaming через AG-UI и Vercel formats
  - Auth setup (Bearer, OAuth examples)
  - DBOS-FastAPI integration

- **Project structure (DDD bounded contexts):**
  - Recommended layout: `agents/<context>/{patterns.py, capabilities.py, models/persistence.py, models/domain.py, repositories.py, evals.py}`
  - Public API через `__all__`
  - Custom ruff rule для `*Row`-imports

- **Testing strategy:**
  - Unit: `TestModel` / `FunctionModel` + `InMemoryRepository` + `FakeChannel`
  - Integration: реальный PG через testcontainers + DBOS test mode
  - End-to-end: один golden scenario через все слои
  - Eval: pydantic-evals в CI

- **MVP scope (v1) — тонкий вертикальный срез:**
  - L0: `Ref[T]` + resolver + escape hatch
  - L1: `SemanticLoopDetector` + `BudgetGuard`
  - L2: `Reflection` + `MapReduce` + `MutationPipeline` + `HITLGate` с `ThreadToolChannel` и `UIChannel`
  - L3: полная DBOS интеграция
  - L4: infra-таблицы (без advanced indexing)
  - L5: `SchemaAdherenceScorer` + один Dataset
  - L6: базовый logfire
  - L7: FastAPI surface с AG-UI и Vercel adapters

- **Golden scenario** — конкретный бизнес-сценарий, через который прогоняется весь MVP (TBD — нужен от пользователя).

- **Examples:**
  - End-to-end Pattern + Capability + MutationPipeline + HITLGate
  - Test suite для каждого слоя

---

## Open Questions

- [ ] **Golden scenario для MVP** — нужен конкретный бизнес-кейс (домен, entities, что агент решает). Без него Section 3 будет в абстракции.
- [ ] **Eval-from-trace формат** — нужен ли инструмент конвертации DBOS workflow traces → pydantic-evals Dataset? (вероятно да на v2)
- [ ] **Multi-tenant** — нужен ли scope thread/proposals по tenant_id с самого начала? (если да — добавить в L4 schema)
- [ ] **A2A vs AG-UI** — какой протокол primary для inter-agent коммуникации (если она вообще нужна в v1)?
