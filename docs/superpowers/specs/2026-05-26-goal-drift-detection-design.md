# Goal Drift Detection — Design Spec

**Date:** 2026-05-26
**Status:** Draft (awaiting user review)
**Source motivation:** "Архитектура и надёжность агентных LLM-систем в Production" — section on Goal & Reasoning Drift / Drift Detection.

## Problem

Длительно работающие LLM-агенты постепенно отклоняются от первоначальной цели пользователя (Goal drift). По мере раздувания контекста логика принятия решений на последних шагах перестаёт опираться на изначальные инструкции, и агент начинает решать совершенно иную задачу, игнорируя первоначальные требования. В отличие от классического сбоя приложения, который сопровождается trace stack'ом и остановкой процесса, дрейф ИИ-агента происходит постепенно, скрытно и без явных сообщений об ошибках.

Существующие capabilities в framework'е (`BudgetGuard`, `SemanticLoopDetector`, `TypedLoopGuard`) ловят синтаксические / ресурсные аномалии, но не семантический дрейф цели. Нужен механизм, который **периодически смотрит на trace и спрашивает у LLM-судьи: "мы ещё идём к изначальной цели или уже куда-то не туда?"** — с возможностью эскалировать (warn / HITL / hard-fail) при обнаружении.

## Goals

- Single contract для drift detection — апп декларирует когда / на что смотреть / как реагировать; framework выполняет.
- Composable через Protocol-based DI: апп подсовывает любую часть pipeline'а.
- Two runtime surfaces: agent (`BallastCapability`) + workflow (`@with_drift_monitor` decorator). Один shared engine, два wrapper'а.
- Fail-safe: drift judge или handler errors никогда не ломают основной flow (за исключением намеренного `RaiseDriftError`).
- Минимум новых зависимостей — только pydantic-ai `Agent` + `BallastCapability` + `Durable` + `RunContext` (всё уже есть).

## Non-goals

- Statistical Data Drift (embedding-based отклонение трафика от эталона) — отдельный паттерн.
- Persistence verdict'ов в БД для analytics — апп сам через свой handler.
- Multi-judge consensus / adaptive threshold — follow-up specs.
- Замена существующих capabilities (`BudgetGuard`, `SemanticLoopDetector` остаются и работают параллельно).

## Architecture

### File structure

```
src/ballast/drift/                 # shared core (Protocols + impls + engine)
  __init__.py                      # public re-exports
  _protocols.py                    # 5 Protocols + DriftVerdictBase + DriftCheckSignal/DriftContext
  _verdict.py                      # DefaultDriftVerdict (rich BaseModel)
  _strategies.py                   # DriftCheckStrategy impls
  _windows.py                      # TraceWindow impls
  _goal_sources.py                 # GoalSource impls
  _handlers.py                     # DriftHandler impls + GoalDriftError
  _judge.py                        # DefaultPromptBuilder + make_default_judge factory
  _core.py                         # DriftEngine + _run_drift_check

src/ballast/capabilities/
  drift.py                         # GoalDriftDetector(BallastCapability) — thin wrapper

src/ballast/patterns/
  drift_monitor.py                 # with_drift_monitor(...) decorator — thin wrapper

tests/drift/                       # core tests
  test_protocols.py
  test_strategies.py
  test_windows.py
  test_goal_sources.py
  test_handlers.py
  test_judge.py
  test_core.py
tests/capabilities/
  test_drift.py                    # capability surface
tests/patterns/
  test_drift_monitor.py            # workflow surface
```

**Принцип размещения:** logic в `ballast.drift` (cross-cutting concern). Тонкие runtime-обёртки — в `capabilities/` (agent surface) и `patterns/` (workflow surface). `_core` ничего не знает ни о `BallastCapability`, ни о `@Durable.workflow`.

### Public API

`from ballast.drift import ...`:
- `GoalDriftDetector` (capability) — re-exported from `ballast.capabilities.drift`
- `with_drift_monitor` (workflow decorator) — re-exported from `ballast.patterns.drift_monitor`
- `DriftEngine` (composition root)
- `DriftVerdictBase`, `DefaultDriftVerdict`
- 5 Protocols: `DriftCheckStrategy`, `TraceWindow`, `GoalSource`, `DriftHandler`, `PromptBuilder`
- Built-in impls (см. ниже)
- `DriftCheckSignal`, `DriftContext` (для апп-кастомных Protocols)
- `GoalDriftError` (raised by `RaiseDriftError` handler)
- `make_default_judge` (factory)

Не выносится в top-level `from ballast import ...` — клиенты пишут `from ballast.drift import ...`. Top-level re-export только для `GoalDriftDetector` + `with_drift_monitor` (основные runtime сюрфейсы), чтобы выровняться с другими capability/pattern экспортами.

## Components

### Protocols + verdict

```python
class DriftVerdictBase(BaseModel):
    """Minimum contract — framework reads these two fields."""
    should_interrupt: bool
    reason: str            # CoT обоснование (для логов / HITL контекста)

class DefaultDriftVerdict(DriftVerdictBase):
    score: float           # 0.0=полный дрейф ... 1.0=на цели
    category: Literal["on_track", "loose", "drifted"]
    suggested_action: str | None = None


@runtime_checkable
class DriftCheckStrategy(Protocol):
    """When to fire the judge. Stateful — instances may track counters."""
    def should_check(self, signal: "DriftCheckSignal") -> bool: ...

@runtime_checkable
class TraceWindow(Protocol):
    """What slice of history the judge sees."""
    async def slice(self, ctx: "DriftContext") -> list[ModelMessage]: ...

@runtime_checkable
class GoalSource(Protocol):
    """Where the original objective comes from."""
    async def goal(self, ctx: "DriftContext") -> str: ...

@runtime_checkable
class PromptBuilder(Protocol):
    """How to ask the judge. Returns user prompt for judge agent."""
    def build(self, goal: str, trace: list[ModelMessage]) -> str: ...

@runtime_checkable
class DriftHandler(Protocol):
    """What to do on drift. Multiple handlers run in declared order."""
    async def handle(self, verdict: DriftVerdictBase, ctx: "DriftContext") -> None: ...
```

### Vehicle types

```python
@dataclass
class DriftCheckSignal:
    """Lightweight ping for DriftCheckStrategy.should_check.
    Passed on every step (cheap to construct, no I/O)."""
    step_index: int           # сколько LLM-шагов прошло
    tool_calls: int           # сколько tool-call'ов
    tokens_used: int          # суммарно
    seconds_elapsed: float    # с начала run/workflow

@dataclass
class DriftContext:
    """Полный контекст для window/goal/handler. Read-only снаружи.
    Собирается ТОЛЬКО когда strategy.should_check() == True."""
    messages: list[ModelMessage]        # вся история на момент check'а
    run_ctx: RunContext | None          # agent surface only
    workflow_input: Any | None          # workflow surface only
    metadata: dict[str, Any]            # apps stash whatever (e.g. budget state)
```

### Core engine

```python
@dataclass
class DriftEngine:
    strategy:      DriftCheckStrategy
    window:        TraceWindow
    goal_source:   GoalSource
    prompt:        PromptBuilder
    judge:         Agent[None, DriftVerdictBase]
    handlers:      list[DriftHandler]
    verdict_model: type[DriftVerdictBase] = DefaultDriftVerdict

    async def maybe_check(
        self, signal: DriftCheckSignal, ctx: DriftContext,
    ) -> DriftVerdictBase | None:
        """Single entry point. Returns verdict if check fired, else None.

        Surface wrappers (capability / workflow) build signal+context and call this.
        """
        if not self.strategy.should_check(signal):
            return None

        goal  = await self.goal_source.goal(ctx)
        trace = await self.window.slice(ctx)
        if not trace:                              # nothing to judge
            return None

        prompt = self.prompt.build(goal, trace)
        try:
            judge_result = await self.judge.run(
                prompt, output_type=self.verdict_model,
            )
            verdict = judge_result.output
        except Exception:
            _log.exception("drift judge failed (swallowed)")
            return None

        if verdict.should_interrupt:
            for handler in self.handlers:
                try:
                    await handler.handle(verdict, ctx)
                except GoalDriftError:
                    raise        # RaiseDriftError is the one handler whose
                                 # exception is intentional — bubble up
                except Exception:
                    _log.exception("drift handler %r failed (swallowed)",
                                   type(handler).__name__)
        return verdict
```

**Свойства:**
- Single entry point. Capability + workflow surface оба зовут `maybe_check`.
- Fail-safe: judge / non-Raise handler exceptions swallowed.
- `RaiseDriftError` — единственный handler, чьи exceptions пробрасываются (это его смысл).
- Idempotent on empty trace — пустое окно пропускается.
- Pure function от signal+context (модульно тестируется).

### Built-in implementations

#### `DriftCheckStrategy` (когда судить)

- `AfterEveryStep()` — каждый шаг (точно, дорого).
- `EveryNToolCalls(n=5)` — каждые N tool-call'ов.
- `EveryNSteps(n=3)` — каждые N LLM-шагов.
- `Periodic(seconds=30.0)` — по `seconds_elapsed`.
- `OnBudgetThreshold(fraction=0.5)` — когда `tokens_used / budget` пересекает порог. Читает `ctx.metadata["budget"]` (convention с `BudgetGuard`).
- `Compose(*strategies)` — OR-комбинация.

#### `TraceWindow` (что показать судье)

- `FullTrace()` — все сообщения (дорого).
- `LastNMessages(n=10)` — хвост.
- `SinceLastUserMessage()` — от последнего user-сообщения.
- `TokenBudgetWindow(max_tokens=4000)` — обрезка с конца по токенам.

#### `GoalSource` (откуда брать цель)

- `FirstUserMessage()` — первый user-msg в trace.
- `LastUserMessage()` — последний user-msg (per-turn).
- `WorkflowInput()` — `ctx.workflow_input` (для workflow surface'а).
- `ExplicitGoal(goal: str)` — статически прибит при wire-up.
- `Summarized(agent: Agent[None, str], every_n: int = 20)` — пересжимает первые N msg в одно goal-предложение раз в N сообщений.

#### `DriftHandler` (что делать на drift)

- `LogOnly()` — `_log.warning`, ничего не блокирует.
- `EmitDriftEvent(event_name="goal_drift")` — шлёт thread event с `verdict.dict()`. Не блокирует.
- `RaiseDriftError()` — бросает `GoalDriftError(verdict)`. Workflow падает, DBOS ловит, retry/escalate по своим правилам.
- `EscalateToHITL(channel, *, card_kind="goal_drift")` — открывает `ApprovalCard` через канал, БЛОКИРУЕТ пока не decision. Апп должен зарегистрировать `CardVerdict` subclass под `goal_drift` (стандартный HITL flow).
- `Compose(*handlers)` — цепочка, exceptions глотаются индивидуально.

#### `PromptBuilder` + judge

- `DefaultPromptBuilder()` — CoT-style prompt: "Below is the original goal and recent trace. Step by step: does the agent's current trajectory still serve the goal? Output a structured verdict."
- `make_default_judge(model="openai:gpt-4o-mini") -> Agent[None, DriftVerdictBase]` — factory; апп подсовывает свой Agent с custom моделью.

### Capability surface (`src/ballast/capabilities/drift.py`)

```python
class GoalDriftDetector(BallastCapability):
    """Agent-side drift monitor. Fires on after_run hooks per agent step."""
    name = "goal_drift_detector"

    def __init__(self, engine: DriftEngine) -> None:
        self._engine = engine
        self._step_index = 0
        self._tool_calls = 0
        self._started_at: float | None = None

    async def before_model_request(self, ctx, request):
        if self._started_at is None:
            self._started_at = time.monotonic()
        return request

    async def after_run(self, ctx, *, result):
        self._step_index += 1
        self._tool_calls += _count_tool_calls(result)
        signal = DriftCheckSignal(
            step_index=self._step_index,
            tool_calls=self._tool_calls,
            tokens_used=_extract_tokens(result),
            seconds_elapsed=time.monotonic() - self._started_at,
        )
        drift_ctx = DriftContext(
            messages=list(result.all_messages()),
            run_ctx=ctx,
            workflow_input=None,
            metadata={},
        )
        await self._engine.maybe_check(signal, drift_ctx)
        return result
```

### Workflow surface (`src/ballast/patterns/drift_monitor.py`)

```python
def with_drift_monitor(engine: DriftEngine):
    """Decorator: wraps an async function, runs drift check in a background
    coroutine on the configured strategy. Suitable for @Durable.workflow
    bodies that don't have an agent-loop attached."""
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            monitor_task = asyncio.create_task(
                _background_monitor(
                    engine,
                    fn_input=args[0] if args else next(iter(kwargs.values()), None),
                ),
            )
            try:
                return await fn(*args, **kwargs)
            finally:
                monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor_task
        return wrapper
    return deco
```

**Background monitor:**
- Раз в 1.0s tick'ает, формирует `DriftCheckSignal` с накопленным `seconds_elapsed`.
- `DriftContext.messages = []`; `workflow_input = fn_input`.
- ⇒ default `TraceWindow` impl'ы вернут `[]`, `_core.maybe_check` пропустит.
- ⇒ workflow surface полезен ТОЛЬКО когда апп подсовывает custom `TraceWindow` который читает state из БД / `ctx.metadata` / т.п., или когда workflow внутри гоняет агентов и эти агенты дописывают что-то в shared state.

Это известное ограничение, документируется в docstring `with_drift_monitor` и в README.

### Composition с CoALAUnit (optional)

Опциональная factory `goal_drift_as_unit(engine) -> CoALABase`:

```python
class _GoalDriftUnit(CoALABase[
    DriftContext,                # InT  — захват состояния
    tuple[str, list[ModelMessage]],  # ObsT — goal + trace
    DriftVerdictBase,            # ContextT — verdict от судьи
    DriftVerdictBase,            # OutT
]):
    def __init__(self, engine: DriftEngine): ...
    async def observe(self, ctx): ...
    async def retrieve(self, obs):    # вызывает engine.judge
    async def act(self, obs, verdict): # вызывает handlers, returns verdict
```

Затем апп оборачивает через `as_capability(unit)` / `as_workflow(unit)` — получает те же runtime-сюрфейсы через CoALA infrastructure. **Опциональный sugar** для аппов уже использующих CoALA; основной surface остаётся `GoalDriftDetector` + `@with_drift_monitor`.

## Data flow

### Agent surface (`GoalDriftDetector` capability)

```
[agent step N completes]
    → BallastCapability.after_run(ctx, result=AgentRunResult)
    → counters++ (step_index, tool_calls)
    → DriftCheckSignal(step, tool_calls, tokens, seconds)
    → DriftContext(messages=result.all_messages(), run_ctx=ctx, ...)
    → DriftEngine.maybe_check(signal, ctx)
        → strategy.should_check(signal)?
            no  → return None (cheap path)
            yes → goal_source.goal(ctx)
                → window.slice(ctx)
                → prompt.build(goal, trace)
                → judge.run(prompt, output_type=DefaultDriftVerdict)
                → if verdict.should_interrupt:
                    for handler in handlers:
                        await handler.handle(verdict, ctx)
                → return verdict
    → result returned to agent loop unchanged
```

### Workflow surface (`@with_drift_monitor`)

```
[workflow starts]
    → wrapper creates background_monitor asyncio.Task
    → workflow body runs to completion
    → finally: monitor_task.cancel()

[background_monitor loop, every 1.0s]
    → tick++, seconds_elapsed = now - start
    → DriftCheckSignal(step_index=tick, tool_calls=0, tokens_used=0, seconds)
    → DriftContext(messages=[], workflow_input=fn_input, ...)
    → DriftEngine.maybe_check(...)
        → если custom window читает state из БД — есть что судить
        → если default window — пустой trace, пропуск
```

## Error handling

| Layer | Behavior |
|---|---|
| `DriftEngine.maybe_check` — judge exception | Swallowed → `_log.exception("drift judge failed (swallowed)")` → return `None` |
| `DriftEngine.maybe_check` — non-Raise handler exception | Swallowed indivudually → `_log.exception("drift handler %r failed (swallowed)")` → продолжает цепочку |
| `RaiseDriftError` handler | `GoalDriftError(verdict)` пробрасывается наверх (это его контракт) |
| `GoalDriftDetector.after_run` — own exception | Swallowed silently (pydantic-ai capability hooks must not crash agent) |
| `with_drift_monitor` background monitor — exception в tick'е | `_log.exception` в loop'е, продолжает |
| `with_drift_monitor` body — exception | Пробрасывается; `finally` отменяет monitor task |

## Testing strategy

```
tests/drift/
  test_protocols.py       # 5 isinstance(stub, Protocol) checks (runtime_checkable)
  test_strategies.py      # каждая built-in strategy: счётчики, edge cases,
                          # OnBudgetThreshold reads ctx.metadata correctly
  test_windows.py         # slicing correctness: FullTrace, LastNMessages с
                          # пустым trace, SinceLastUserMessage, TokenBudgetWindow
  test_goal_sources.py    # extraction из разных message-shapes
                          # (system/user/assistant, multi-turn, empty)
  test_handlers.py        # LogOnly, EmitDriftEvent (mock ctx),
                          # RaiseDriftError (verify GoalDriftError),
                          # EscalateToHITL (mock channel),
                          # Compose (ordering, exception isolation)
  test_judge.py           # DefaultPromptBuilder output shape;
                          # make_default_judge constructs valid Agent (no LLM call)
  test_core.py            # DriftEngine.maybe_check:
                          # — should_check=False → None
                          # — empty trace → None
                          # — judge raises → None + log
                          # — should_interrupt=False → no handlers
                          # — should_interrupt=True + handler raises → other handlers still run
                          # — RaiseDriftError → bubbles up
tests/capabilities/
  test_drift.py           # GoalDriftDetector через TestModel + fake DriftEngine:
                          # — счётчики растут (step_index, tool_calls)
                          # — signal формируется корректно
                          # — exception в engine не ломает agent
tests/patterns/
  test_drift_monitor.py   # decorator wrap:
                          # — body run completes; monitor cancelled in finally
                          # — body exception → finally still cancels monitor
                          # — monitor tick exception → logged, loop continues
```

## Integration с существующими capabilities

- **BudgetGuard** — пишет `tokens_used / budget_cap` в `ctx.metadata["budget"]`. `OnBudgetThreshold` strategy читает. Convention документируется в обоих capability docstrings.
- **SemanticLoopDetector** / **TypedLoopGuard** — ортогональны. Несколько capability могут стоять в одном списке. Loop detection = синтаксические циклы (cosine sim), drift detection = семантическое смещение цели.
- **HITLChannel** — `EscalateToHITL` принимает существующий канал (`UICardChannel` / `ThreadChannel`). Card `kind="goal_drift"` — апп регистрирует `CardVerdict` subclass под этот kind (стандартный HITL flow). Если апп не зарегистрировал — `EscalateToHITL.__init__` валидирует и падает с понятной ошибкой.
- **OTel / logfire** — `_run_drift_check` оборачивается в `@traced("drift_check")` span; verdict идёт в span attributes. Drift events видны в logfire alongside agent run.
- **CoALA** — опциональный `goal_drift_as_unit(engine)` factory для аппов, желающих ленить через CoALA adapter'ы. Основной surface остаётся прямой.

## Demo (notes-app, optional follow-up)

Notes-app может получить `GoalDriftDetector` в `NotesAgent` для демо:

```python
class NotesAgent(DurableAgent):
    def build_agent(self):
        return Agent(
            ...,
            capabilities=[
                GoalDriftDetector(DriftEngine(
                    strategy=EveryNToolCalls(n=3),
                    window=LastNMessages(n=10),
                    goal_source=FirstUserMessage(),
                    prompt=DefaultPromptBuilder(),
                    judge=make_default_judge(),
                    handlers=[EmitDriftEvent()],
                )),
            ],
        )
```

UI получает `goal_drift` thread event'ы → показывает уведомление "agent might be drifting". Это **отдельная задача**, вне scope этого spec'а — implementation plan для core может ограничиться framework + framework tests.

## Out of scope

- Statistical Data Drift (embedding-distance baseline).
- Persistence verdict'ов в БД (через handler в апп-коде).
- Multi-judge consensus / weighted voting.
- Adaptive threshold (по historical drift rate).
- Auto-summarized goal через background workflow (только naive `Summarized` strategy в plain Python).
- Per-tool drift specialization (отдельные strategy per tool).
- Frontend / UI components для drift notifications (notes-app может сделать как demo).

## Open questions for review

1. **OnBudgetThreshold convention** — `BudgetGuard` сейчас НЕ пишет в `ctx.metadata`. Чтобы strategy работала, либо (a) добавить в `BudgetGuard` запись `ctx.metadata["budget"]` (мелкая правка существующего capability), либо (b) `OnBudgetThreshold` принимает callable `budget_fn(ctx) -> tuple[int, int]` (более гибко, но менее out-of-the-box). Дефолт plan'а: (a) — простой shared convention.

2. **`EscalateToHITL` blocking semantics** — handler ждёт user decision (через `channel.request_decision`). Если другие handlers в `Compose(...)` стоят ПОСЛЕ него — они побегут только после ответа. Альтернатива: `EscalateToHITL` fire-and-forget (создаёт карту, не ждёт). Дефолт: blocking (соответствует семантике "handler" — последовательно), но это поведение может удивить. Стоит ли делать fire-and-forget дефолтом? Plan'у уточнить.

3. **Workflow surface ограничение** — описано выше. Принимаем как known limitation в этом spec'е; если в практике окажется болью, отдельный follow-up spec про "workflow drift с state-from-DB window" решит.
