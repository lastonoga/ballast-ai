# pydantic-ai-stateflow-engine — Design Spec

- **Date:** 2026-05-15
- **Status:** Sections 1 + 2 (A–E) approved. Section 3 (infrastructure & process) TBD.
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

## Section 2 — Detailed Architecture (approved)

### 2A — L0 GroundedSchema Implementation

#### 2A.1 Public API

```python
# pydantic_ai_stateflow/grounded/__init__.py
EntityT = TypeVar("EntityT", bound=BaseModel)

class Ref(Generic[EntityT]):
    """Typed UUID reference to an Entity of type EntityT.

    - LLM/JSON layer: plain UUID string (no wrapper)
    - Python layer:   typed reference; .id, .hydrate(repo)
    """
    __slots__ = ("id", "_entity_type")

    def __init__(self, id: UUID, *, entity_type: type[EntityT]): ...

    @classmethod
    def __class_getitem__(cls, item: type[EntityT]) -> type["Ref[EntityT]"]:
        # Возвращает специализированный subclass с привязкой к T
        ...

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        # JSON ↔ UUID string; Python ↔ Ref(id, entity_type=T)
        ...

    async def hydrate(self, repo: "Repository[EntityT]") -> EntityT:
        return await repo.load(self.id)


class GroundedAgent(Generic[CtxT, OutT]):
    def __init__(self, agent: Agent, *, output_type: type[OutT]):
        self.agent = agent
        self.output_type = output_type
        self._resolver = GroundedResolver(output_type)

    async def run(
        self,
        context: BaseModel,
        *,
        instructions: str | None = None,
        constraints: dict[str, list[Any] | Any] | None = None,  # escape hatch
        **agent_kwargs,
    ) -> GroundedResult[OutT]: ...


class GroundedResult(BaseModel, Generic[OutT]):
    value: OutT
    hydration_map: HydrationMap
    raw: AgentRunResult

    async def hydrate(self, **repos: Repository[Any]) -> OutT: ...
```

#### 2A.2 Resolver Algorithm

```python
class FieldRole(StrEnum):
    REF, LIST_REF, ENUM, NESTED, LIST_NESTED, FREE = ...

@dataclass
class FieldSpec:
    path: str
    role: FieldRole
    target_type: type | None
    nested_spec: "OutputSpec | None"

class GroundedResolver:
    def __init__(self, output_type: type[BaseModel]):
        self._spec = self._scan_output(output_type)

    def build(self, context: BaseModel, constraints: dict | None) -> tuple[type[BaseModel], HydrationMap]:
        sources = self._scan_context(context, self._spec)
        if constraints:
            sources = self._apply_constraints_override(sources, constraints)
        return self._build_dynamic(self.output_type, sources, path=""), HydrationMap(self._spec, sources)
```

`_build_dynamic` рекурсивно создаёт через `create_model`:
- `Ref[T]` → `Literal[*ids]`
- `list[Ref[T]]` → `list[Literal[*ids]]`
- Enum/Literal с совпадающим типом из context → пересечение
- Nested BaseModel → рекурсия с новым `DynamicNested...` классом
- `list[BaseModel]` → рекурсия + broadcast

#### 2A.3 Resolver Rules (full table)

| Случай | Поведение |
|---|---|
| `Ref[Candidate]` + `list[Candidate]` в context | `Literal[*ids]` |
| `Ref[Candidate]` + одиночный `Candidate` | `Literal[id]` |
| `Ref[T]` + несколько источников T | union allowed-sets |
| `Ref[T]` + 0 источников | `GroundedBuildError` (construction-time) |
| `list[Ref[T]]` | `list[Literal[*ids]]` (подмножество) |
| `Optional[Ref[T]]` + 0 instances | `Optional[None]` с warning |
| `Ref[A] | Ref[B]` | union allowed-sets обоих |
| Literal/Enum поле в output, тот же enum в context | пересечение допустимых значений |
| Recursive entity (Tree) | scan ограничен по depth (default 5); circular → ошибка |
| Очень большие allowed-sets (>1000 ids) | warning: `SemanticRouter` рекомендуется |
| Partial-Entity DTO (subset полей) | scanner ищет `.id`; иначе игнор |
| Constraints override path не существует | `GroundedBuildError` (construction-time) |
| Новая сущность (не ссылка) | обычный `UUID = Field(default_factory=uuid4)` — не `Ref` |

#### 2A.4 Hydration via Repository

```python
class HydrationMap:
    spec: OutputSpec

    async def hydrate(self, value: OutT, repos: dict[type, Repository[Any]]) -> OutT:
        # Walk value, для каждого Ref-path заменяет .id на await repo.load(id)
        ...

# Использование:
hydrated = await result.hydrate(Candidate=candidate_repo, Customer=customer_repo)
```

Repos индексируются **типом**, не именем. Соответствует type-driven духу.

#### 2A.5 Tests

L0 не зависит от DBOS / БД / агентов. Тесты — pure pydantic:

```python
def test_grounded_resolver_basic():
    class Cand(BaseModel): id: UUID; label: str
    class Ctx(BaseModel):  candidates: list[Cand]
    class Out(BaseModel):  chosen: Ref[Cand]; rationale: str

    ctx = Ctx(candidates=[Cand(id=u1, label="a"), Cand(id=u2, label="b")])
    Dynamic, _ = GroundedResolver(Out).build(ctx, constraints=None)

    assert Dynamic.model_fields["chosen"].annotation == Literal[u1, u2]
    Dynamic.model_validate({"chosen": str(u1), "rationale": "..."})  # ok
    with pytest.raises(ValidationError):
        Dynamic.model_validate({"chosen": str(uuid4()), "rationale": "..."})
```

---

### 2B — L1 Capabilities Catalog

Все наследуют от `StateflowCapability(AbstractCapability)` с авто-Logfire-span и стандартной конфигурацией через `pydantic-settings`.

#### 2B.1 BudgetGuard (outermost)

Token + iteration limit. State хранится в `ctx.state["budget"]` → переживает DBOS replay.

```python
class BudgetGuard(StateflowCapability):
    name = "budget_guard"
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_iterations: int = 20

    def get_ordering(self): return CapabilityOrdering(position="outermost")

    async def before_model_request(self, ctx, *, request_context):
        if ctx.run_step >= self.max_iterations:
            raise BudgetExhausted(reason="max_iterations", at_step=ctx.run_step)
        return request_context

    async def after_model_request(self, ctx, *, request_context, response):
        if not self._budget_for_ctx(ctx).consume(response.usage):
            raise BudgetExhausted(reason="tokens", usage=response.usage)
        return response
```

#### 2B.2 SemanticLoopDetector (L1, raw response)

Детектит повторяющиеся model responses внутри одного `agent.run()`. Для typed-output detection между Pattern iterations — `TypedLoopGuard` в L2.

```python
class SemanticLoopDetector(StateflowCapability):
    name = "semantic_loop_detector"
    embedder: Embedder
    threshold: float = 0.95
    window: int = 3
    selector: Callable[[ModelResponse], str] = _default_response_text

    async def after_model_request(self, ctx, *, request_context, response):
        snapshot = self.selector(response)
        await self._deduper(ctx).add_and_check(snapshot, threshold=self.threshold, window=self.window)
        return response
```

`_default_response_text` сериализует TextPart + ToolCallPart (с stable JSON-args).

#### 2B.3 TypedLoopGuard[OutT] (L2 helper для Pattern-iterations)

```python
@dataclass
class TypedLoopGuard(Generic[OutT]):
    embedder: Embedder
    selector: Callable[[OutT], str | list[str]]
    threshold: float = 0.95
    window: int = 3
    _deduper: SemanticDeduper = field(default_factory=SemanticDeduper)

    async def check(self, output: OutT) -> None:
        snapshots = self.selector(output)
        if isinstance(snapshots, str): snapshots = [snapshots]
        for snap in snapshots:
            await self._deduper.add_and_check(snap, threshold=self.threshold, window=self.window)
```

Используется в Pattern (Reflection) между iterations. Selector — callable, type-safe (refactor-safe).

#### 2B.4 SemanticDeduper (shared helper)

```python
class SemanticDeduper:
    """Скользящее окно эмбеддингов + cosine-check. Shared между 2B.2 и 2B.3."""
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._history: deque = deque()

    async def add_and_check(self, snapshot: str, *, threshold: float, window: int) -> None:
        emb = await self.embedder.embed(snapshot)
        if len(self._history) == window: self._history.popleft()
        if len(self._history) >= 2 and all(cosine(emb, h) >= threshold for h in self._history):
            raise SemanticLoopDetected(snapshot=snapshot[:200])
        self._history.append(emb)
```

#### 2B.5 GoalDriftDetector

Async LLM-judge между шагами агента.

```python
class GoalDriftDetector(StateflowCapability):
    name = "goal_drift"
    judge: Agent[None, DriftVerdict]
    check_every: int = 3
    threshold: float = 0.7
    on_drift: DriftPolicy = WarnOnly()   # Strategy: WarnOnly / RaiseOnDrift / EscalateHITL(hitl=...)

    async def after_node_run(self, ctx, *, node, result):
        if ctx.run_step == 0:
            ctx.state[f"{self.name}.initial_goal"] = _extract_user_prompt(ctx)
            return result
        if ctx.run_step % self.check_every != 0:
            return result

        verdict = await self.judge.run(DriftCheckInput(
            initial_goal=ctx.state[f"{self.name}.initial_goal"],
            current_trajectory=_summarize_trajectory(ctx),
        ))
        if verdict.output.confidence < self.threshold:
            await self.on_drift.handle(ctx, verdict.output)
        return result
```

#### 2B.6 LLMJudgeHook

Async (fire-and-forget) оценка финального output → eval store. Мост между production runtime и L5 evals.

```python
class LLMJudgeHook(StateflowCapability):
    name = "llm_judge"
    judge: Agent[None, JudgeVerdict]
    eval_store: EvalStore
    criteria: list[Criterion]
    sample_rate: float = 1.0

    async def after_run(self, ctx, *, result):
        if random.random() > self.sample_rate: return result
        DBOS.start_workflow(
            self._judge_and_store,
            run_id=ctx.run_id, output=result.output, criteria=self.criteria,
            tenant_id=ctx.tenant_id,
        )
        return result

    @DBOS.workflow()
    async def _judge_and_store(self, run_id, output, criteria, tenant_id):
        verdict = await self.judge.run(JudgeInput(output=output, criteria=criteria))
        await self.eval_store.persist(run_id, verdict.output, tenant_id=tenant_id)
```

#### 2B.7 PIIGuard (innermost)

```python
class PIIGuard(StateflowCapability):
    name = "pii_guard"
    patterns: list[re.Pattern]
    replacement: str = "[REDACTED]"

    def get_ordering(self): return CapabilityOrdering(position="innermost")

    async def after_model_request(self, ctx, *, request_context, response):
        for part in response.parts:
            if isinstance(part, TextPart):
                for pat in self.patterns:
                    part.content = pat.sub(self.replacement, part.content)
        return response
```

#### 2B.8 GroundedRetry

Превращает ValidationError → структурированный feedback через `ModelRetry`. Бьёт в "boundary-condition failures" из исходного документа.

```python
class GroundedRetry(StateflowCapability):
    name = "grounded_retry"
    max_retries: int = 3

    async def on_output_validate_error(self, ctx, *, raw_output, error):
        attempts = ctx.state.get(f"{self.name}.attempts", 0)
        if attempts >= self.max_retries: raise error
        ctx.state[f"{self.name}.attempts"] = attempts + 1
        feedback = self._build_feedback(error, raw_output)
        raise ModelRetry(feedback)

    def _build_feedback(self, error: ValidationError, raw_output) -> str:
        # literal_error → "Field X must be one of: [..]. You returned: Y"
        # missing → "Required field X is missing"
        ...
```

#### 2B.9 Composition example

```python
agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        BudgetGuard(max_iterations=10),                                 # outermost
        GoalDriftDetector(judge=cheap_judge, check_every=3),
        SemanticLoopDetector(embedder=embedder, threshold=0.95),
        GroundedRetry(max_retries=3),
        LLMJudgeHook(judge=quality_judge, eval_store=store, sample_rate=0.2),
        PIIGuard(patterns=[EMAIL_RE, PHONE_RE]),                        # innermost
    ],
)
```

Wrap-порядок (middleware semantics): `before_*` top-to-bottom; `after_*` reverse; `wrap_*` nests outermost-first.

---

### 2C — L2 Patterns: Core Four

#### 2C.0 Pattern base

```python
class Pattern(Generic[InT, OutT]):
    name: ClassVar[str]
    # tenant_id обязателен в .run(), не в __init__ (singleton pattern instances)
```

`tenant_id` уходит в `.run(input, *, tenant_id)` — kwarg-параметр каждого вызова. Pattern instance — переиспользуемый.

#### 2C.1 Reflection

```python
class Critique(BaseModel):
    passed: bool
    issues: list[str] = []
    suggestions: list[str] = []
    confidence: float

class LoopRecoveryPolicy(Protocol[OutT]):
    async def handle(self, ctx, draft: OutT, feedback: list[Critique]) -> OutT: ...

class AbortOnLoop:       async def handle(self, ctx, draft, fb): raise SemanticLoopDetected(...)
class AcceptLast:        async def handle(self, ctx, draft, fb): return draft
class EscalateToHITL:
    def __init__(self, hitl: HITLGate): self.hitl = hitl
    async def handle(self, ctx, draft, fb):
        resp = await self.hitl.run(HITLPrompt(...), tenant_id=ctx.tenant_id, purpose="ambiguity")
        if resp.decision == "approved": return draft
        raise ReflectionAborted(by_actor=resp.actor_id)

class Reflection(Pattern[InT, OutT]):
    name = "reflection"
    def __init__(
        self,
        writer: Agent[Any, OutT],
        critic: Agent[Any, Critique],
        *,
        max_iterations: int = 5,
        loop_guard: TypedLoopGuard[OutT] | None = None,
        loop_recovery: LoopRecoveryPolicy[OutT] = AbortOnLoop(),
        feedback_renderer: Callable[[InT, list[Critique]], InT] = default_feedback_renderer,
    ): ...

    @DBOS.workflow()
    async def run(self, task: InT, *, tenant_id: UUID) -> OutT:
        feedback: list[Critique] = []
        for i in range(self.max_iterations):
            draft = await self._write(task, feedback=feedback)
            if self.loop_guard:
                try:    await self.loop_guard.check(draft)
                except SemanticLoopDetected:
                    return await self.loop_recovery.handle(self._ctx, draft, feedback)
            critique = await self._critique(draft, task)
            if critique.passed: return draft
            feedback.append(critique)
        raise ReflectionExhausted(iterations=self.max_iterations, last_feedback=feedback)

    @DBOS.step()
    async def _write(self, task, feedback): ...
    @DBOS.step()
    async def _critique(self, draft, task): ...
```

#### 2C.2 MapReduce

```python
class Chunker(Protocol[Doc, Chunk]):
    def chunk(self, doc: Doc) -> list[Chunk]: ...

class Reducer(Protocol[Item]):
    async def reduce(self, items: list[Item]) -> list[Item]: ...

class MapReduce(Pattern[Doc, list[Item]], Generic[Doc, Chunk, Item]):
    name = "map_reduce"
    def __init__(
        self,
        chunker: Chunker[Doc, Chunk],
        extractor: Agent[Chunk, Item | None],     # None = empty chunk
        reducer: Reducer[Item],
        *,
        concurrency: int = 8,
    ): ...

    @DBOS.workflow()
    async def run(self, doc: Doc, *, tenant_id: UUID) -> list[Item]:
        chunks = await self._chunk(doc)
        queue = DBOS.Queue(f"mapreduce-{self.workflow_id}", concurrency_limit=self.concurrency)
        handles = [queue.enqueue(self._extract_one, chunk, tenant_id) for chunk in chunks]
        results = await asyncio.gather(*[h.get_result() for h in handles])
        return await self._reduce([r for r in results if r is not None])

    @DBOS.step()
    async def _chunk(self, doc): return self.chunker.chunk(doc)

    @DBOS.workflow()
    async def _extract_one(self, chunk, tenant_id):
        return (await self.extractor.run(ExtractInput(chunk=chunk))).output

    @DBOS.step()
    async def _reduce(self, items):
        return await self.reducer.reduce(items)
```

#### 2C.3 MutationPipeline

Stages — параметризованный list. Apply — отдельный required параметр (инвариант: всегда последний, transactional).

```python
class Stage(Protocol[T]):
    name: str
    async def process(self, proposal: Proposal[T]) -> StageResult[T]: ...

class StageResult(Generic[T]): ...
class Accept(StageResult[T]):       proposal: Proposal[T]
class RejectedAt(StageResult[T]):   stage: str; reason: str; actor_id: str | None; metadata: dict

class RejectPolicy(Protocol[T]):
    async def handle(self, rejected: RejectedAt[T]) -> RejectAction: ...

class DropOnReject:               # просто отбросить
class RetryModelOnReject:         # feedback для LLM → ModelRetry
class EscalateToHITLOnReject:     # вторая попытка через approval

class MutationPipeline(Pattern[Proposal[T], AcceptedResult[T] | RejectedAt]):
    name = "mutation_pipeline"
    def __init__(
        self,
        stages: list[Stage[T]],                          # mutable, в любом порядке
        *,
        apply: ApplyTransaction[T, EntityT],             # required, last, @DBOS.transaction
        emit_event: type[DomainEvent] | None = None,
        reject_policy: RejectPolicy[T] = DropOnReject(),
        repo: ProposalRepository,
    ): ...

    @DBOS.workflow()
    async def run(self, proposal, *, tenant_id):
        for stage in self.stages:
            result = await stage.process(proposal)
            if isinstance(result, RejectedAt):
                return await self.reject_policy.handle(result)
            proposal = result.proposal  # stage может модифицировать (см. ApprovalStage)
        return await self._apply_and_emit(proposal)

    @DBOS.transaction()
    async def _apply_and_emit(self, proposal, session: AsyncSession):
        # Apply + Emit в одной транзакции = transactional outbox
        ...
```

**Approval как Stage** (proactive HITL, Role A — встраивается в stages list):

```python
class ApprovalStage(Stage[T]):
    """Single-actor approval + optional modify-flow.

    Modify требует: allow_modify=True, editable_paths whitelist, revalidate_stages.
    """
    name: str
    def __init__(
        self,
        hitl: HITLGate,
        *,
        when: Callable = lambda _: True,
        prompt_builder: Callable[[Proposal[T]], HITLPrompt],
        stage_name: str = "approval",
        allow_modify: bool = False,
        editable_paths: set[str] | None = None,
        revalidate_stages: list[Stage[T]] = (),
        modify_policy: ModifyPolicy[T] = StrictWhitelist(),
    ):
        if allow_modify and editable_paths is None:
            raise ConfigError("allow_modify=True requires explicit editable_paths")

    async def process(self, proposal):
        if not self.when(proposal): return Accept(proposal)
        prompt = self.prompt_builder(proposal)
        if self.allow_modify:
            prompt = prompt.model_copy(update={"decision_type": "approve_modify_reject", ...})
        resp = await self.hitl.run(prompt, purpose="approval")
        match resp.decision:
            case "approved": return Accept(proposal)
            case "rejected" | "timeout":
                return RejectedAt(stage=self.name, reason=resp.feedback, actor_id=resp.actor_id)
            case "modified":
                return await self._handle_modify(proposal, resp)

    async def _handle_modify(self, proposal, resp):
        modified = self.modify_policy.apply(proposal, resp.modified_proposal)
        diff = json_diff_paths(proposal, modified)
        if self.editable_paths and not diff.issubset(self.editable_paths):
            return RejectedAt(stage=self.name, reason=f"modifications outside whitelist: {diff - self.editable_paths}")
        for s in self.revalidate_stages:
            r = await s.process(modified)
            if isinstance(r, RejectedAt):
                return RejectedAt(stage=f"{self.name}.revalidate.{r.stage}", reason=r.reason)
        await self._emit_modification_event(proposal, modified, resp.actor_id)
        return Accept(modified)


class ModifyPolicy(Protocol[T]):
    def apply(self, original: Proposal[T], modifications: dict) -> Proposal[T]: ...

class StrictWhitelist: ...     # default: только разрешённые поля, остальное отбрасывается
class FullReplace:     ...     # принимает полный объект (опасно)
class JsonPatchPolicy: ...     # RFC 6902 JSON Patch
```

#### 2C.4 HITLGate

```python
class HITLPrompt(BaseModel):
    title: str
    context: str
    decision_type: Literal["approve_reject", "approve_modify_reject", "choose_option", "free_text"]
    options: list[HITLOption] = []
    timeout: timedelta | None = None
    # actor_filter удалён — authz через Voter

class HITLResponse(BaseModel):
    decision: Literal["approved", "modified", "rejected", "timeout"]
    modified_proposal: dict | None = None
    feedback: str | None = None
    actor_id: str | None = None
    answered_at: datetime

class HITLGate(Pattern[HITLPrompt, HITLResponse]):
    name = "hitl_gate"
    def __init__(self, channel: HITLChannel, *, policy: Policy, repo: HITLRepository): ...

    @DBOS.workflow()
    async def run(
        self,
        prompt: HITLPrompt,
        *,
        tenant_id: UUID,
        purpose: Literal["approval", "reject_recovery", "ambiguity", "policy_conflict"] = "approval",
    ) -> HITLResponse:
        request_id = await self.repo.persist_request(prompt, tenant_id=tenant_id, purpose=purpose)
        try:
            response = await self.channel.ask(prompt, request_id=request_id)
        except HITLTimeout:
            await self.repo.persist_timeout(request_id)
            return HITLResponse(decision="timeout", answered_at=datetime.now(UTC))

        decision = await self.policy.can(actor=response.actor_id, action="decide", resource=prompt, tenant_id=tenant_id)
        if not decision.is_grant: raise HITLAuthzDenied(actor=response.actor_id, votes=decision.votes)

        await self.repo.persist_response(request_id, response)
        return response
```

#### 2C.5 SOLID review summary

| Pattern | S | O | L | I | D | KISS | DRY | YAGNI | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Reflection | ✓ | ✓ Strategy | ✓ | ✓ | ✓ Policies | ✓ | ✓ | ✓ | ok |
| MapReduce | ✓ | ✓ Protocols | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ no hierarchical | ok |
| MutationPipeline | ✓ orchestrator | ✓ stages list | ✓ | ✓ | ✓ | ✓ | ✓ RejectPolicy | ✓ | ok |
| HITLGate | ✓ | ✓ Channel port | ✓ | ✓ | ✓ channel+policy+repo | ✓ | ✓ single authz | ✓ | ok |

---

### 2D — L2 Patterns: Extended (5 more)

#### 2D.1 SelfRAG

```python
class Retriever(Protocol):
    async def retrieve(self, query: RetrievalQuery, tenant_id: UUID, *, top_k: int) -> list[Document]: ...

class QueryRewriter(Protocol):
    async def rewrite(self, original, missing, prior_docs) -> RetrievalQuery: ...

class SelfRAG(Pattern[RetrievalQuery, OutT]):
    name = "self_rag"
    def __init__(
        self, retriever: Retriever, generator: Agent, grounding_judge: Agent,
        *, max_retrieval_rounds: int = 3, top_k: int = 5,
        query_rewriter: QueryRewriter = LLMQueryRewriter(),
    ): ...

    @DBOS.workflow()
    async def run(self, query, *, tenant_id):
        seen, current = [], query
        for _ in range(self.max_retrieval_rounds):
            new = await self._retrieve(current, tenant_id)
            seen.extend(new)
            output = await self._generate(seen, query)
            verdict = await self._verify_grounding(output, seen)
            if verdict.is_grounded: return output
            current = await self.query_rewriter.rewrite(query, verdict.missing_facts, seen)
        raise GroundingExhausted(...)
```

#### 2D.2 CorrectiveRAG

```python
class RetrievalQuality(StrEnum): CORRECT = "correct"; AMBIGUOUS = "ambiguous"; INCORRECT = "incorrect"
class QualityClassifier(Protocol): ...
class FallbackRetriever(Protocol): ...

class CorrectiveRAG(Pattern):
    def __init__(
        self, primary_retriever, classifier: QualityClassifier, generator,
        *, on_ambiguous: list[FallbackRetriever] = (), on_incorrect: list[FallbackRetriever] = (),
        top_k: int = 5,
    ): ...

    @DBOS.workflow()
    async def run(self, query, *, tenant_id):
        docs = await self._retrieve_primary(query, tenant_id)
        quality = await self._classify(query, docs)
        match quality:
            case RetrievalQuality.CORRECT:    pass
            case RetrievalQuality.AMBIGUOUS:  docs.extend(await self._fallbacks(self.on_ambiguous, query, tenant_id))
            case RetrievalQuality.INCORRECT:  docs = await self._fallbacks(self.on_incorrect, query, tenant_id)
        return (await self._generate(docs, query)).output
```

#### 2D.3 DivergentConvergent

```python
class ConvergenceCriteria(Protocol[Hypothesis, OutT]):
    async def synthesize(self, hypotheses: list[Hypothesis], task: Any) -> OutT: ...

class DivergentConvergent(Pattern, Generic[InT, Hypothesis, OutT]):
    def __init__(
        self,
        divergent_agents: list[Agent[Any, list[Hypothesis]]],     # parallel frontier models
        convergent_synthesizer: Agent[Any, OutT],
        criteria: ConvergenceCriteria = LLMSynthesis(),
        *, divergent_concurrency: int = 4, min_hypotheses: int = 5,
        dedup_threshold: float = 0.92, embedder: Embedder | None = None,
    ): ...

    @DBOS.workflow()
    async def run(self, task, *, tenant_id):
        # Divergent phase — параллельно через DBOS.queue
        queue = DBOS.Queue(...)
        handles = [queue.enqueue(self._diverge_one, i, task) for i in range(len(self.divergent_agents))]
        pools = await asyncio.gather(*[h.get_result() for h in handles])
        merged = await self._merge_and_dedup(pools)
        if len(merged) < self.min_hypotheses: raise InsufficientDivergence(...)
        # Convergent phase
        return await self._converge(merged, task)
```

#### 2D.4 PlanAndExecute

```python
class PlanStep(BaseModel):
    step_id: str
    description: str
    depends_on: list[str] = []
    executor_name: str
    inputs: dict = {}

class Plan(BaseModel):
    steps: list[PlanStep]
    success_criteria: str

class Replanner(Protocol):
    async def revise(self, plan, failed_step, error) -> Plan | None: ...

class PlanAndExecute(Pattern):
    def __init__(
        self, planner: Agent[Any, Plan], executors: dict[str, Agent], synthesizer: Agent,
        *, replanner: Replanner | None = None, max_replan_attempts: int = 2, concurrency: int = 4,
    ): ...

    @DBOS.workflow()
    async def run(self, task, *, tenant_id):
        plan = await self._plan(task)
        results, attempts = {}, 0
        while True:
            try:
                results = await self._execute_dag(plan, results)
                return (await self._synthesize(plan, results)).output
            except StepFailed as e:
                if not self.replanner or attempts >= self.max_replan_attempts:
                    raise PlanFailed(plan=plan, partial_results=results) from e
                plan = await self.replanner.revise(plan, e.step, e.error) or None
                if plan is None: raise PlanFailed(...)
                attempts += 1
```

DAG execution через топологические волны + `DBOS.queue` concurrency limit.

#### 2D.5 SemanticRouter

```python
@dataclass
class Route(Generic[OutT]):
    name: str
    utterances: list[str]
    handler: Agent[Any, OutT] | Pattern[Any, OutT]
    min_confidence: float = 0.7

class SemanticRouter(Pattern[str, OutT]):
    def __init__(
        self, routes: list[Route[OutT]], embedder: Embedder,
        *, fallback_agent: Agent | None = None,
        on_no_match: Literal["fallback", "raise"] = "fallback",
    ):
        if on_no_match == "fallback" and fallback_agent is None:
            raise ConfigError("fallback_agent required")

    @DBOS.workflow()
    async def run(self, query, *, tenant_id):
        query_emb = await self._embed(query)
        best, score = await self._best_route(query_emb)
        if best is None or score < best.min_confidence:
            if self.fallback_agent is None: raise NoRouteMatched(...)
            return (await self.fallback_agent.run(query)).output
        return await self._dispatch(best, query, tenant_id)
```

Route.handler — `Agent` или `Pattern`; единая точка dispatch. Embeddings prewarmed на boot Pattern.

---

### 2E — L3 Infrastructure

#### 2E.1 Workflow/Step Determinism Boundary

| В `@DBOS.workflow` (orchestration) | В `@DBOS.step` (side effect) |
|---|---|
| Pure Python control flow (if, for, while) | `agent.run(...)` |
| Композиция child workflows / steps через `await` | `repo.load(...)`, `repo.persist(...)` |
| `DBOS.queue`, `DBOS.start_workflow`, `DBOS.recv` | HTTP calls (`httpx`, A2A) |
| Чтение `ctx.state` (накопленные results) | Side-effects на внешние системы (Slack, email, webhook) |
| | Embedder вызовы |
| | Random / time (получают результат из step и возвращают) |
| | File I/O |
| ❌ `time.time()`, `datetime.now()` | ✅ `Det.now()`, `Det.uuid4()` |
| ❌ `random.choice()` | ✅ `Det.random_choice(...)` |
| ❌ `os.environ[...]` (через DI) | |
| ❌ Direct httpx/requests | |
| ❌ `await asyncio.sleep(...)` | ✅ `DBOS.sleep(...)` |

**Lint rules** (mandatory CI gate, custom ruff):
- `STATEFLOW001` — Forbidden side-effect inside @DBOS.workflow body
- `STATEFLOW002` — `datetime.now()` / `time.time()` outside @DBOS.step
- `STATEFLOW003` — Direct httpx/requests call outside @DBOS.step
- `STATEFLOW004` — `random.*` outside @DBOS.step
- `STATEFLOW005` — `asyncio.sleep` inside workflow (use `DBOS.sleep`)
- `STATEFLOW006` — Repository call outside @DBOS.step
- `STATEFLOW007` — `agent.run(...)` outside @DBOS.step
- `STATEFLOW008` — `_*Row` SQLModel import outside `repositories/`
- `STATEFLOW009` — Repository protocol method without `tenant_id` parameter

**`Det` helpers:**

```python
class Det:
    @staticmethod
    @DBOS.step()
    async def now() -> datetime: return datetime.now(tz=UTC)
    @staticmethod
    @DBOS.step()
    async def uuid4() -> UUID: return uuid4()
    @staticmethod
    @DBOS.step()
    async def random_choice(seq: list[T]) -> T: return random.choice(seq)
```

#### 2E.2 Bootstrap / Service Provider

```python
class ServiceProvider(Protocol):
    def register(self, container: Container) -> None: ...
    async def boot(self, container: Container) -> None: ...

class Container:
    def bind(self, protocol: type[T], factory: Callable[[Container], T], *, singleton: bool = True) -> None: ...
    def get(self, protocol: type[T]) -> T: ...
    def fastapi_dependency(self, protocol: type[T]) -> Callable[[], T]:
        return lambda: self.get(protocol)

class Engine:
    def __init__(self, providers: list[ServiceProvider], settings: AppSettings):
        self.container = Container()
        self.providers = providers
        self.settings = settings

    async def boot(self):
        for p in self.providers: p.register(self.container)
        for p in self.providers: await p.boot(self.container)

    def fastapi_app(self) -> FastAPI: ...
```

**Принципы:**
- No autodiscovery — providers перечисляются явно (KISS, refactor-safe)
- `register` → `boot` двухфазно
- Container type-driven (`Container.get(Protocol)`, не string key)
- Singleton по умолчанию (pattern instances переиспользуются)

#### 2E.3 Policy + Voter Authorization

```python
class VoterDecision(Enum): GRANT = "grant"; DENY = "deny"; ABSTAIN = "abstain"

@dataclass
class VoterVote: decision: VoterDecision; voter_name: str; reason: str = ""

class Voter(Protocol[T_Resource]):
    name: str
    def supports(self, action: str, resource: T_Resource) -> bool: ...
    async def vote(self, *, actor, action, resource, tenant_id) -> VoterVote: ...

class AccessDecisionStrategy(StrEnum):
    AFFIRMATIVE = "affirmative"
    CONSENSUS   = "consensus"
    UNANIMOUS   = "unanimous"

class AccessDecisionManager:
    def __init__(self, voters: list[Voter], *, strategy: AccessDecisionStrategy = UNANIMOUS): ...
    async def decide(self, *, actor, action, resource, tenant_id) -> AccessDecision: ...

class Policy(Protocol[T_Resource]):
    async def can(self, *, actor, action, resource, tenant_id) -> AccessDecision: ...

class VoterPolicy(Policy):
    def __init__(self, manager: AccessDecisionManager): ...
    async def can(self, **kw): return await self.manager.decide(**kw)
```

**Применение в трёх точках (DRY):**
1. `MutationPipeline.PolicyStage` — write-flow gating
2. `HITLGate.run()` — кто может отвечать
3. FastAPI endpoint Depends — кто может звать

#### 2E.4 Event Dispatcher

```python
class DomainEvent(BaseModel):
    event_id: UUID = Field(default_factory=uuid4)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tenant_id: UUID                  # обязательно для multi-tenancy
    workflow_id: UUID | None = None
    run_id: UUID | None = None
    actor_id: str | None = None

class EventListener(Protocol[E]):
    event_type: type[E]
    async def handle(self, event: E) -> None: ...

class EventDispatcher:
    def subscribe(self, event_type: type[E], listener: EventListener[E]) -> None: ...

    @DBOS.step()                       # dispatch — side effect
    async def dispatch(self, event: DomainEvent) -> None:
        for listener in self._listeners.get(type(event), []):
            try: await listener.handle(event)
            except Exception: logfire.exception(...)
```

**Стандартные события:**

| Lifecycle | Mutation | HITL | Quality |
|---|---|---|---|
| `AgentRunStarted` | `ProposalAccepted` | `HITLRequested` | `SemanticLoopDetected` |
| `AgentRunCompleted` | `ProposalRejected` | `HITLResponded` | `GoalDriftDetected` |
| `AgentRunFailed` | `ProposalModified` | `HITLTimedOut` | `BudgetExhausted` |
| `PatternStarted` | | | |
| `PatternCompleted` | | | |
| `StepCompleted` | | | |

**Стандартные listeners (passive only):**
- `LogfireListener` — все события → logfire spans
- `OutboxListener` — audit events → outbox table
- `DriftMetricsListener` — метрики в `drift_metrics` для dashboards
- `EvalCaseListener` — runs в `eval_traces` для eval-from-trace (1.14)

**Принцип:** listener — пассивный. Только observability/audit/persistence. Контроль flow остаётся в Pattern-коде.

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

## Section 1 — Addenda (approved 2026-05-15)

### 1.12 Multi-tenant as first-class concept

Multi-tenancy входит во **все** слои с v1:

- **L4 State:** каждая infra-таблица имеет `tenant_id: UUID` + FK на `tenants(id)` + индекс. Row-Level Security (Postgres RLS) — hardening на v1.1.
- **L7 API:** `tenant_id` извлекается из auth token (FastAPI `Depends`) → `RunContext.tenant_id`. Endpoint без tenant context → 401.
- **L3 DBOS:** `tenant_id` входит в `workflow_id = hash(tenant_id, pattern, input)` — идемпотентность строго per-tenant. Никаких cross-tenant конфликтов.
- **L2 Patterns:** `RunContext.tenant_id` пробрасывается во все child workflows. Pattern не запускается без context-а.
- **L1 Capabilities:** Capability видит tenant через `ctx.deps.tenant_id`. Eval store пишет per-tenant. Budget'ы могут быть per-tenant.
- **L0 GroundedSchema:** не затрагивается (type-level concern).
- **L5 Evals:** `Dataset` фильтруется по `tenant_id`; кросс-tenant eval запрещён.
- **Repository:** все методы — `repo.load(id, tenant_id)`. Без `tenant_id` — ошибка типа (Repository.Protocol).
- **Ruff rule:** "Repository method без `tenant_id` argument" — fails CI.

### 1.13 A2A as primary inter-agent protocol

Inter-agent коммуникация (Pattern → Pattern across services / external agents) — через **A2A** (`agent.to_a2a()` нативно от pydantic-ai). AG-UI и Vercel AI SDK — **только** для UI streaming.

- **L7** дополняется `/.well-known/agent.json` (A2A discovery) + `/a2a/{agent_name}` endpoints
- **L2 Patterns** вызывают удалённых агентов через `RemoteAgent("https://service/a2a/agent")` — он реализует тот же `Agent`-протокол, замены не требуется
- **Authentication for A2A:** signed tokens (issuer per-tenant); Voter проверяет grant на cross-service call
- **Outbound A2A call** — это `@DBOS.step` (side effect), не workflow-level код

### 1.14 Eval-from-trace tooling (L5)

Каждый production run автоматически становится reusable eval-кейсом.

- **CLI:** `stateflow evals dataset-from-traces --since <date> --pattern <name> --filter <expr> --out <path>`
- Под капотом: запрос в DBOS state (`workflow_runs`, `step_results`, `outbox`) → группировка по run → `Case(inputs=..., expected=..., metadata={run_id, tenant_id, outcome})`
- `run_id` сохраняется как metadata → eval failure прослеживается обратно к production incident
- Закрывает "Automated Feedback Loops (CLHF)" из исходного документа
- Cross-tenant export запрещён (см. 1.12)

## Open Questions (remaining)

### Pending for v1
- [ ] **Golden scenario для MVP** — нужен конкретный бизнес-кейс (домен, entities, что агент решает). Без него Section 3 будет в абстракции.

### Deferred to v2

- **Multi-actor approval (Quorum)** — `QuorumApprovalStage` с weighted quorum, parallel HITL workflows. Влечёт `parent_quorum_id` в `hitl_requests` и view `quorum_responses` в L4.
- **Modify + Quorum combined** — UX-сложно (какая версия побеждает если двое modify).
- **Early-termination для Quorum** — через `DBOS.recv` polling вместо `gather` (экономит latency).
- **PartialApprovalStrategy** — `partial_strategy: PartialApprovalStrategy = DropOnPartial()` для unreached quorum.
- **Hierarchical reduce в MapReduce** — когда items > prompt budget.
- **Streaming partial results** — `progress_publisher: ProgressPublisher | None` в SelfRAG / PlanAndExecute для UI прогресса.
- **`agent.iter()` checkpointing** — мост между L1 capabilities и L3 durability для возобновления mid-run.
- **Custom `Det.now/uuid4` overrides** для testing time travel.
