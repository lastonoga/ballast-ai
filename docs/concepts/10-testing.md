# 10. Testing

**Prerequisites:** [01-agents.md](01-agents.md), [07-capabilities.md](07-capabilities.md), [08-running-an-app.md](08-running-an-app.md).

## Introduction

Testing LLM-driven code has a reputation for being slow, flaky, and expensive. Hit the real model and tests take seconds-to-minutes per case, cost money, and produce different answers run-to-run. Mock the model and you're testing your mock, not your code. Neither extreme is workable; the question is where to draw the line so the tests actually catch things.

The framework's stance: use pydantic-ai's `TestModel` (or `FunctionModel`) for unit tests of agents — scripted, deterministic, free, fast. Use the in-memory repos for unit tests of routes. Use a small DBOS SQLite fixture for testing durable workflows. Save real LLM calls for an integration suite you run on demand (or nightly), not on every PR.

This chapter walks through each layer: what to test at the unit level, how to assert on tool calls and capability state, how to test DBOS workflows without spinning up Postgres, and how the framework's `ballast.testing` package provides the fixtures that make all this one line of setup.

## The mental model: a pyramid

```
       ┌─────────────────────┐
       │ Integration: real   │   ← Few. Nightly. Cost money.
       │ LLM + real DB       │
       ├─────────────────────┤
       │ Workflow: TestModel │   ← Some. DBOS SQLite fixture.
       │ + DBOS SQLite       │
       ├─────────────────────┤
       │ Route: in-memory    │   ← More. FastAPI TestClient + fakes.
       │ repos + TestModel   │
       ├─────────────────────┤
       │ Unit: TestModel,    │   ← Many. Sub-millisecond.
       │ no I/O              │
       └─────────────────────┘
```

The pyramid isn't novel; the framework just keeps each layer easy enough that you'll actually use it. The bottom layer should be free and fast enough that you don't think twice about adding more tests.

## Unit tests: `TestModel` for scripted runs

pydantic-ai ships `TestModel` (in `pydantic_ai.models.test`). The framework doesn't wrap or replace it — use it directly:

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

class Fact(BaseModel):
    text: str

@pytest.mark.asyncio
async def test_extractor_returns_typed_fact():
    agent = Agent(
        model=TestModel(custom_output_text='{"text": "hello"}'),
        output_type=Fact,
    )
    result = await agent.run("Extract a fact")
    assert result.output == Fact(text="hello")
```

`TestModel` doesn't call any LLM. It returns whatever you configured. Two common configurations:

- **`custom_output_text="..."`** — the model returns this as a single text part.
- **`call_tools=[ToolCallPart(...)]`** — the model calls these tools. Useful for testing tool dispatch.

For more elaborate scripted behavior (different responses across turns), use `FunctionModel(my_fn)` where `my_fn(messages) -> ModelResponse`.

### Asserting on tool calls

When the model under test calls tools, you usually want to assert *which tools* were called *with what arguments*. Two patterns:

**Pattern 1: spy on the tool.** Replace the tool function with a `MagicMock` (or a recording wrapper) before the run:

```python
recorded = []

async def search_notes(ctx, query: str) -> str:
    recorded.append({"query": query})
    return "fake result"

agent = Agent(
    model=FunctionModel(call_search_then_finish),
    tools=[search_notes],
)
await agent.run("find ML notes")
assert recorded == [{"query": "ML notes"}]
```

**Pattern 2: read `result.all_messages()` after the run.** Every tool call and every tool result is in the message history:

```python
result = await agent.run("...")
tool_calls = [
    part for msg in result.all_messages()
    for part in msg.parts if isinstance(part, ToolCallPart)
]
assert len(tool_calls) == 1
assert tool_calls[0].tool_name == "search_notes"
```

The first pattern is more direct; the second is necessary when you're testing through a wrapper (a Pattern, a workflow) that hides the agent.

### Testing typed output

Validation happens inside `agent.run`. If `TestModel` returns malformed JSON for your schema, the agent retries. To assert that a *real* invalid response would be rejected:

```python
agent = Agent(
    model=TestModel(custom_output_text='{"text": 123}'),  # wrong type
    output_type=Fact,
    retries=0,  # don't retry
)
with pytest.raises(UnexpectedModelBehavior):
    await agent.run("...")
```

For `Scored[T]` outputs (chapter 6), the wrapper validates rationale + confidence the same way. A `TestModel` returning `{"value": {...}, "rationale": "x", "confidence": "high"}` will pydantic-validate; anything missing the rationale will fail.

### Testing capabilities

Capabilities are independently testable. Construct one, call the hook directly, assert on internal state:

```python
@pytest.mark.asyncio
async def test_budget_guard_trips_on_iterations():
    guard = BudgetGuard(max_iterations=2)
    for_run = await guard.for_run(fake_run_context)
    # Simulate 3 model requests
    for _ in range(3):
        await for_run.after_model_request(
            fake_run_context,
            request_context=fake_request_context,
            response=fake_response_with_usage(input=10, output=10),
        )
    # The 3rd should have raised — assert via pytest.raises in real code
```

For full integration (capability inside an agent), construct the agent with `capabilities=[guard]` and run with `TestModel` that produces many turns.

## Route tests: `TestClient` + in-memory repos

For route-level tests, use the framework's `client` fixture:

```python
# conftest.py
pytest_plugins = ["ballast.testing.pytest_plugin"]

# test_chat.py
def test_post_thread(client):
    response = client.post("/threads", json={"agent": "notes"})
    assert response.status_code == 200
    assert "id" in response.json()
```

The fixture builds a `TestEngine.default()` (in-memory repos, no DBOS, no observability), constructs the FastAPI app, runs lifespan, and yields a `TestClient`. Two related fixtures:

- **`engine`** (function-scoped) — the `TestEngine` itself if you need to interact with repos directly.
- **`client`** (function-scoped) — the `TestClient`; depends on `engine`.

### Swapping a repo for a test

```python
from ballast.persistence.thread import get_thread_repo

def test_with_custom_repo(client):
    fake_repo = MyFakeThreadRepository()
    client.app.dependency_overrides[get_thread_repo] = lambda: fake_repo
    response = client.post("/threads", json={"agent": "notes"})
    assert response.status_code == 200
    # assert fake_repo received the create call
```

Use `dependency_overrides` rather than constructing a fresh `Ballast` per test — it's faster and the override is per-test.

### Swapping the agent

The framework doesn't own your agents (chapter 8), so swapping them for tests means importing the module and replacing the global:

```python
def test_agent_under_test_model(client, monkeypatch):
    from app import agents
    test_agent = Agent(model=TestModel(custom_output_text="scripted"), ...)
    monkeypatch.setattr(agents, "notes_agent", test_agent)

    response = client.post("/chat", json={"text": "hi"})
    assert response.json()["output"] == "scripted"
```

This is just standard pytest monkeypatching. The framework doesn't add anything on top.

## Workflow tests: DBOS SQLite fixture

`@Durable.workflow` decorated functions need DBOS launched, which means a database. For unit tests, you want this fast and isolated. The framework's testing helper sets up an in-memory SQLite-backed DBOS instance per test module:

```python
# conftest.py
pytest_plugins = ["ballast.testing.pytest_plugin"]

# test_workflows.py
@pytest.mark.asyncio
async def test_my_workflow(engine):
    # engine has DBOS launched, in-memory SQLite, in-memory thread repo
    result = await my_durable_workflow(input_data)
    assert result.status == "ok"
```

The fixture is module-scoped so launching DBOS once per module amortizes the cost. Function-scoped isolation comes from each test using fresh inputs / workflow IDs.

For tests that need a *fresh* DBOS state per test (rare), construct your own `TestEngine` with a unique SQLite path:

```python
@pytest.fixture
async def fresh_engine(tmp_path):
    engine = TestEngine(dbos_db=f"sqlite:///{tmp_path}/dbos.db")
    async with engine.test_client():
        yield engine
```

## Testing HITL flows with fake channels

Approval flows go through an `ApprovalCardRepository` + a `HITLChannel`. For unit tests, use the in-memory repo and a fake channel that synchronously returns a verdict:

```python
class AutoApproveChannel:
    def __init__(self, verdict_factory):
        self._factory = verdict_factory

    async def request(self, *, card, timeout=None):
        return self._factory(card)

@pytest.mark.asyncio
async def test_approval_path(engine):
    channel = AutoApproveChannel(lambda card: ApprovedVerdict(approved=True))
    # Inject channel into your workflow, run, assert
```

For tests of the queue itself (cards getting added, listed, resolved), interact with `engine.approval_repo` directly:

```python
async def test_card_lifecycle(engine):
    card = ApprovalCard(id="x", ...)
    await engine.approval_repo.add(card)
    pending = await engine.approval_repo.list_pending()
    assert card in pending
    await engine.approval_repo.resolve(card.id, verdict=ApprovedVerdict(...))
    assert card not in await engine.approval_repo.list_pending()
```

## Integration tests: marked separately

Real LLM + real DB tests live behind a pytest marker:

```python
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "integration: real LLM calls and real DB; slow and expensive",
]

# tests/integration/test_real_extraction.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_against_real_gpt4():
    agent = Agent(model="openai:gpt-4o-mini", ...)
    result = await agent.run("...")
    assert "expected substring" in result.output
```

In CI, run `pytest -m "not integration"` on every push and `pytest -m integration` nightly (or before release). Keep the integration suite small — it's there to catch *integration* bugs (provider API changes, schema regressions), not to verify your business logic. Business logic belongs in unit tests with `TestModel`.

## What to test and what not to test

A few priorities, in order:

1. **Tool argument shapes and behavior.** If your tool takes `query: str` and you typo'd it to `queries: list[str]`, the model will call it wrong. A unit test catches this.
2. **Typed output validation.** Your `Scored[Fact]` schema needs to actually accept the shape your model produces. One integration test per output type per quarter is fine; many unit tests with `TestModel` keep the validation logic exercised.
3. **Capability state transitions.** `BudgetGuard` should trip at the right iteration; `CircuitBreaker` should open after N failures; etc. These are pure state-machine tests — fast, deterministic, important.
4. **Route happy paths.** A `client.post("/chat")` smoke test for each endpoint, asserting status code + response shape.
5. **Workflow durability.** One test that proves your workflow resumes after a simulated crash. Don't write fifty.

What *not* to test, because it's not worth the effort:

- **Exact LLM output strings.** Models drift, providers update, your assertions go stale. Test that the *structure* is right (it validates), not that the text matches a magic string.
- **Prompt wording.** The prompt is content, not contract. Snapshot it if you want; don't gate CI on it.

## Common mistakes

- **Sharing a `TestModel` across tests.** `TestModel` has internal state (call count). Construct a fresh one per test.
- **Not using `pytest.mark.asyncio`.** Async tests silently pass without it (the coroutine isn't awaited). Use the marker (or `asyncio_mode = "auto"` in `pyproject.toml`).
- **Mocking too deep.** If you're mocking `pydantic_ai.Agent.run` itself, you've gone too far — use `TestModel` instead. Mocks of internals break when the framework updates.
- **Hitting real APIs in unit tests.** Anyone running tests offline (planes, trains, code review) will be blocked. Unit suite must be hermetic.

## What this chapter did NOT cover

- The exact pydantic-ai `TestModel` / `FunctionModel` API — see pydantic-ai's docs for the full surface.
- Property-based testing with Hypothesis — generally a good fit for `Scored[T]` filters and CircuitBreaker state machines, not covered here.
- Load testing — different concern; use `locust` or similar against a real-LLM staging environment.
- Evaluation runs (LLMJudge over fixed datasets) — chapter 23.

## Where to go next

→ [11-budget-and-loops.md](11-budget-and-loops.md) — production hardening against runaway behavior.
