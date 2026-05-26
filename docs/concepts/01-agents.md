# 1. Agents

**Prerequisites:** none — this is the first chapter.

## Introduction

An **agent** is the unit of work in Ballast. Everything else — tools, capabilities, patterns, workflows — exists to make agents more useful, more reliable, or more composable. Before any of that makes sense, you need to understand what an agent actually *is* and what running one looks like end to end.

If you've used pydantic-ai before, you already know most of this — Ballast keeps the `Agent` class exactly as pydantic-ai defines it, with no wrapping or magic. What Ballast adds is layered on top: a base class with sensible defaults (`BallastAgent`), a durable subclass for crash-safe workflows (`DurableAgent`), a builder for assembling agents into a runnable app (`Ballast`), and a set of cross-cutting capabilities (`BudgetGuard`, `GoalDriftDetector`, and so on). All of those are introduced in later chapters; this chapter is about the agent itself.

If you haven't used pydantic-ai before, don't worry — this chapter starts from zero. The pydantic-ai documentation is the canonical reference for the underlying class, but you don't need to read it first.

## The mental model

An agent is a stateful conversation between *your code* and *an LLM*. You configure it once with:

- a **model** to call (`openai:gpt-4o`, `anthropic:claude-3-5-sonnet`, etc.)
- a **system prompt** that sets the agent's role and rules
- optionally, an **output type** — a pydantic model the final answer must validate against
- optionally, **tools** — typed Python functions the agent is allowed to call mid-conversation
- optionally, **capabilities** — cross-cutting concerns that hook into the lifecycle (budget caps, drift detection, approvals)

Then you invoke it with `await agent.run(user_input)`. Internally, the agent loops:

1. Send the current message history + system prompt + tool definitions to the model.
2. Receive a response. If the model called a tool, run the tool, append the result to history, go to step 1.
3. If the model returned a final answer (matching `output_type` if set), return.

That loop is the agent's *run loop*. Everything in Ballast — capabilities, patterns, resilience primitives — either hooks into this loop or wraps it from outside.

## Constructing an agent

The simplest possible agent:

```python
from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
)

result = await agent.run("What's the capital of France?")
print(result.output)   # 'Paris'
```

That's pure pydantic-ai. No Ballast involved. The agent has no tools, no structured output, no capabilities — just a model and a prompt. This works for chat-style apps but doesn't scale to anything interesting.

Add a structured output type and you get validated typed replies:

```python
from pydantic import BaseModel

class CountryInfo(BaseModel):
    capital: str
    population: int
    currency: str

agent = Agent(
    model="openai:gpt-4o-mini",
    system_prompt="Answer factually about countries.",
    output_type=CountryInfo,
)

result = await agent.run("Tell me about France.")
print(result.output.capital)     # 'Paris'
print(result.output.population)  # 67000000 (or whatever the model returns)
```

Now `result.output` is a real `CountryInfo` instance, validated by pydantic. If the model returns malformed JSON, validation fails fast with a clear error instead of giving you a string you'd have to parse yourself. The next chapter ([structured-output](03-structured-output.md)) goes deep on this.

Add tools and the agent can take action:

```python
agent = Agent(model="openai:gpt-4o-mini")

@agent.tool_plain
async def get_weather(city: str) -> dict:
    """Look up the current weather for a city."""
    return await weather_api.fetch(city)

result = await agent.run("Should I bring a jacket to London tomorrow?")
```

The model now sees `get_weather` as a callable function with a typed `city: str` parameter and a return type. It will decide on its own whether to call it. The next chapter ([tools](02-tools.md)) covers this in detail.

Add capabilities and the agent gets cross-cutting protections without you wiring each hook by hand:

```python
from ballast import BudgetGuard, SemanticLoopDetector

agent = Agent(
    model="openai:gpt-4o",
    system_prompt="Help the user accomplish their task.",
    capabilities=[
        BudgetGuard(max_iterations=10, max_input_tokens=20_000),
        SemanticLoopDetector(embedder=my_embedder),
    ],
)
```

Now every call to `agent.run(...)` enforces those limits automatically. Chapter [07-capabilities](07-capabilities.md) covers this layer in depth.

## `BallastAgent` — the base class

The framework defines `BallastAgent` as a small ABC over the plain pydantic-ai `Agent`. You don't strictly need it — apps can use bare `Agent` instances anywhere — but `BallastAgent` gives you three conveniences:

1. **A `name: ClassVar[str]` field** — used as a registry key when the framework dispatches threads to the right agent.
2. **A `metadata_model: ClassVar[type[BaseModel] | None]`** — a typed validator for per-thread metadata that the app stores in the `Thread.metadata` JSON field.
3. **Class-level tool registration** via `@MyAgent.tool` and `@MyAgent.system_prompt` decorators — keeps related agent definitions co-located.

The shape is:

```python
from typing import Any, ClassVar
from pydantic import BaseModel
from ballast import BallastAgent
from pydantic_ai import Agent

class _NotesMetadata(BaseModel):
    project_id: str | None = None

class NotesAgent(BallastAgent):
    name: ClassVar[str] = "notes"
    metadata_model: ClassVar[type[BaseModel] | None] = _NotesMetadata

    def build_agent(self) -> Agent[Any, Any]:
        return Agent(
            model="openai:gpt-4o-mini",
            system_prompt="Help the user manage their notes.",
            tools=[...],
            capabilities=[...],
        )
```

`build_agent` returns the underlying pydantic-ai `Agent`. The framework calls this lazily — you can construct the `NotesAgent()` instance freely without forcing the agent loop to initialize.

Why `name`? Because in a multi-agent app, each thread is associated with one agent by string name (`Thread.agent`). The streaming router resolves `Thread.agent` → instance via an app-owned `Registry[BallastAgent]`. The name is your contract; pick something stable and obvious.

Why `metadata_model`? Because threads carry app-specific metadata, and you want it validated. If `NotesAgent` is going to read `thread.metadata["project_id"]`, declaring `_NotesMetadata` makes that contract explicit, validated on write, and visible to the rest of your code.

## `DurableAgent` — for workflow-managed agents

`DurableAgent` is a subclass of `BallastAgent` that wraps every `agent.run` call inside a DBOS `@workflow`. This means:

- The agent run is **crash-safe**. If your process dies mid-run, DBOS resumes the workflow when it comes back up.
- Tool calls + capability hooks are **replay-aware**. Already-completed steps are skipped on replay; only the unfinished tail re-executes.
- HITL waits (via `Durable.recv_async`) are **durable**. A workflow can wait days for a human approval; you can deploy + restart without losing the wait.

The trade-off: `DurableAgent` only makes sense if you're running in a context that has DBOS configured (which `Ballast.fastapi_app()` sets up for you). For ephemeral scripts or unit tests, use plain `BallastAgent` and skip the durability overhead.

In most real apps, you'll have a mix: chat-style agents that handle one user message at a time can be plain `BallastAgent`; agents that orchestrate multi-step workflows (publish a post with approval, run a long research task, etc.) are `DurableAgent`. The shape is identical from the app's perspective:

```python
class TodoApprovalAgent(DurableAgent):
    name = "todo_approval"
    metadata_model = TodoApprovalContext

    def build_agent(self) -> Agent[Any, Any]:
        return Agent(
            model="openai:gpt-4o",
            system_prompt="Propose todos for the user to approve.",
            tools=[...],
        )
```

Same constructor signature, same `build_agent` pattern. The "durable" part is invisible to your code — it's how the framework wraps the agent's execution under the hood.

## Running an agent

There are three ways to invoke an agent's run loop:

### `agent.run(input)` — the simple case

Returns an `AgentRunResult` when the loop completes:

```python
result = await agent.run("Find me ML papers from 2024")
print(result.output)            # final answer (validated against output_type if set)
print(result.all_messages())    # full message history including tool calls
```

This is what 90% of code uses. If you don't care about per-token streaming or step-by-step inspection, use this.

### `agent.run_stream(input)` — for SSE / live UIs

Returns an async iterator that yields partial response chunks as the model generates them:

```python
async with agent.run_stream("Tell me a story") as response:
    async for chunk in response.stream_text():
        print(chunk, end="", flush=True)
    final = await response.get_output()
```

Use this when you want to surface partial output to a UI in real time. The framework's streaming router (`build_streaming_router`) uses this internally.

### `agent.iter(input)` — for fine-grained control

Returns an async iterator of *graph nodes* — the agent's internal step-by-step execution. You can inspect or modify each step:

```python
async with agent.iter("Plan a trip") as run:
    async for node in run:
        if isinstance(node, ModelRequestNode):
            print("about to call model with:", node.request)
        elif isinstance(node, CallToolsNode):
            print("about to call tools:", [tc.tool_name for tc in node.tool_calls])
    final = run.result
```

Use this when you're building patterns or debugging. Most app code doesn't need this level of control.

## The single-step lifecycle

When you call `agent.run(...)`, the framework walks through this sequence for *each iteration* of the agent's loop:

1. **`for_run(ctx)`** — capabilities clone themselves per run. (Chapter 7 covers this.)
2. **`before_model_request(ctx, request_context)`** — capabilities see the prompt before the LLM call.
3. **The model call itself** — pydantic-ai sends the request, awaits the response.
4. **`after_model_request(ctx, request_context, response)`** — capabilities see the response. This is where token counting, semantic loop detection, and drift checks happen.
5. **Tool dispatch** — if the model called any tools, the framework executes them. Tools that have `requires_approval=True` end up in `DeferredToolRequests` (covered in chapter 21).
6. **Loop or finalize** — if the model returned a final answer, the loop exits.

Once the loop exits:

7. **`after_run(ctx, result=result)`** — capabilities see the final `AgentRunResult`. This is where judges (`JudgeAfterRun`) and the framework's HITL bridge run.
8. **`wrap_run` hooks** — any capability that overrides `wrap_run` got to see the whole thing from outside; it returns the final result to the caller.

The lifecycle is what makes capabilities composable. You don't have to coordinate between them — each one hooks where it makes sense and the framework orchestrates the order.

## What this chapter did NOT cover

A lot. Specifically:

- How to *define* tools the agent can call (chapter 2)
- How to *force* the output into a typed shape (chapter 3)
- How to pass *per-run state* (database connections, user context) into tools (chapter 4)
- What `Ref[T]` and `Scored[T]` add to your typed outputs (chapters 5, 6)
- What capabilities actually *do* and how to stack them (chapter 7)
- How to actually *deploy* an agent into a FastAPI app (chapter 8)

Each of those is its own chapter. The reason: each one is a distinct mental model you can hold separately. If we tried to cover everything at once, the right time to learn each piece would be lost in the noise.

## Where to go next

→ [02-tools.md](02-tools.md) — defining tools so the agent can take action.
