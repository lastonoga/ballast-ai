# How to test agents with `TestModel`

**Problem:** Your agent has tools, capabilities, structured output. You want unit tests that run in milliseconds — no real LLM calls, deterministic, free.

**Solution:** Use pydantic-ai's `TestModel`. The framework's agents accept it as a regular `model=` argument. `TestModel` returns scripted responses; you assert on tool-call arguments and final output.

## Minimum

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


@pytest.mark.asyncio
async def test_agent_returns_summary() -> None:
    agent = Agent(
        model=TestModel(custom_output_text="The summary is X."),
        system_prompt="Summarize the user's text.",
    )
    result = await agent.run("some long text")
    assert result.output == "The summary is X."
```

No network calls, runs in <100ms. Use as the default for any agent unit test.

## Test typed output

```python
from pydantic import BaseModel


class Summary(BaseModel):
    title: str
    body: str


@pytest.mark.asyncio
async def test_typed_output() -> None:
    agent = Agent(
        model=TestModel(custom_output_args={"title": "T", "body": "B"}),
        output_type=Summary,
    )
    result = await agent.run("text")
    assert result.output == Summary(title="T", body="B")
```

`custom_output_args` is the dict that pydantic-ai validates against your `output_type`.

## Test tool calls

```python
@pytest.mark.asyncio
async def test_tool_is_called_with_correct_args() -> None:
    captured = []

    agent = Agent(model=TestModel(call_tools=["search"]))

    @agent.tool
    async def search(ctx, query: str) -> str:
        captured.append(query)
        return f"results for {query}"

    await agent.run("find something")
    assert captured == ["find something"]
```

`call_tools=[...]` tells `TestModel` to call the named tools (with auto-generated args based on parameter types).

For finer control over the args:

```python
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ToolCallPart

agent = Agent(
    model=TestModel(
        custom_output_text="done",
        call_tools_with_args={"search": {"query": "exact thing"}},
    ),
)
```

## Test capabilities

```python
from ballast import BudgetGuard, BudgetExhausted


@pytest.mark.asyncio
async def test_budget_exhausts_after_iterations() -> None:
    agent = Agent(
        model=TestModel(call_tools=["loop_tool"]),
        capabilities=[BudgetGuard(max_iterations=3)],
    )

    @agent.tool
    async def loop_tool(ctx) -> str:
        return "keep going"

    with pytest.raises(BudgetExhausted):
        await agent.run("loop forever")
```

Or assert on capability state directly:

```python
@pytest.mark.asyncio
async def test_drift_detector_counts_steps() -> None:
    engine = _FakeDriftEngine()
    cap = GoalDriftDetector(engine=engine)

    agent = Agent(model=TestModel(custom_output_text="done"), capabilities=[cap])
    await agent.run("test")
    assert engine.calls == 1     # one after_model_request
```

## Test patterns

```python
from ballast import MapReduce


@pytest.mark.asyncio
async def test_mapreduce_aggregates() -> None:
    async def map_fn(x): return x.upper()
    async def reduce_fn(items): return ",".join(items)

    mr = MapReduce(map_step=map_fn, reduce_step=reduce_fn, map_concurrency=2)
    out = await mr.run(["a", "b", "c"])
    assert out == "A,B,C"
```

Pattern unit tests don't need an LLM at all when you use callable steps. For agent-based patterns:

```python
@pytest.mark.asyncio
async def test_reflection_completes_within_iterations() -> None:
    writer = Agent(model=TestModel(custom_output_args={"title": "draft", "body": "..."}), output_type=Article)
    critic = Agent(model=TestModel(custom_output_args={"issues": [], "severity": 0}), output_type=Critique)
    refiner = Agent(model=TestModel(custom_output_args={"title": "final", "body": "..."}), output_type=Article)

    reflection = Reflection(
        writer=writer, critic=critic, refiner=refiner,
        max_iterations=3, accept_if=lambda c: c.severity == 0,
        output_type=Article,
    )
    result = await reflection.run("topic")
    assert result.title == "draft"     # accepted immediately, no refine call
```

## Test Scored[T] outputs

```python
@pytest.mark.asyncio
async def test_scored_output() -> None:
    agent = Agent(
        model=TestModel(custom_output_args={
            "value": {"text": "fact"},
            "rationale": "stated explicitly",
            "confidence": "high",
        }),
        output_type=Scored[Fact],
    )
    result = await agent.run("doc")
    assert result.output.confidence == "high"
```

## Snapshot the model history

`TestModel` records the exact prompts it received — useful for asserting on system prompt construction:

```python
@pytest.mark.asyncio
async def test_system_prompt_includes_context() -> None:
    test_model = TestModel(custom_output_text="ok")
    agent = Agent(model=test_model, system_prompt="You are a helpful assistant.")

    await agent.run("hi")

    messages = test_model.last_model_request_parameters.messages
    assert any("helpful assistant" in str(m) for m in messages)
```

## Mock parametric variations

```python
@pytest.mark.parametrize("query,expected", [
    ("ML", "ML summary"),
    ("crypto", "crypto summary"),
])
@pytest.mark.asyncio
async def test_per_query(query, expected) -> None:
    agent = Agent(model=TestModel(custom_output_text=expected))
    result = await agent.run(query)
    assert expected in result.output
```

## When you DO need a real LLM

Mark integration tests separately:

```python
@pytest.mark.integration                # exclude from default runs
@pytest.mark.asyncio
async def test_real_openai() -> None:
    agent = Agent(model="openai:gpt-4o-mini", ...)
    result = await agent.run("...")
    # Assert on shape, not exact content (LLM is non-deterministic).
```

Run with `pytest -m integration` only when you need it.

## Caveats

- **`TestModel` doesn't simulate streaming.** For streaming tests, use `FunctionModel` from pydantic-ai with a custom generator.
- **Tool argument auto-generation is shallow.** For complex pydantic models, supply `call_tools_with_args={tool_name: {...}}` explicitly.
- **Capability hook tests need real `RunContext`.** Most capability tests use `AsyncMock(spec=RunContext)`; some need a small `_FakeCtx` dataclass — see `tests/capabilities/test_approval.py` for the pattern.
- **DBOS workflow tests need a fixture.** See [test-durable-workflows.md](test-durable-workflows.md).

## Related

- [test-coala-units.md](test-coala-units.md) — direct phase testing without adapters
- [test-durable-workflows.md](test-durable-workflows.md) — DBOS bootstrap for workflow tests
- Reference: pydantic-ai `TestModel` docs
