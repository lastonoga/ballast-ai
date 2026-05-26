# 3. Structured output

**Prerequisites:** [01-agents.md](01-agents.md), [02-tools.md](02-tools.md).

## Introduction

Production agents need to return data your code can use, not free-form prose your code has to parse. The naive approach — "I'll write a system prompt asking the LLM to return valid JSON" — fails the moment the model produces a missing comma, a wrong quote, an extra trailing character. You end up writing brittle regex-and-retry parsers, and your downstream code still occasionally crashes when the model produces output you didn't anticipate.

The correct approach is to make typed output a *contract*, not a request. You declare a pydantic model; pydantic-ai converts it to a JSON Schema; the framework sends it to the model with provider-specific structured-output directives (OpenAI's `response_format`, Anthropic's tool-use schema, etc.); the response is validated on receipt; your code receives a real, typed Python object.

This chapter walks through how `output_type` works, what happens on validation failure, how unions enable the deferred-tool / HITL pattern, and the discipline of keeping output types small and focused.

## The mental model

When you pass `output_type=MyModel` to `Agent()`, three things happen:

1. **Schema extraction.** pydantic-ai builds a JSON Schema from `MyModel`. The schema becomes the model's "function-call schema" for its final answer.
2. **Provider wiring.** Depending on which model you're using, the schema is sent via the provider's native structured-output mechanism. OpenAI uses `response_format`; Anthropic uses an injected tool definition; some providers fall back to prompt-level instructions.
3. **Response validation.** When the model returns, pydantic-ai parses the response against `MyModel`. If it validates, you get `result.output` as a typed instance. If it doesn't, pydantic-ai gives the model an error and lets it retry (within the agent's iteration budget).

The contract is end-to-end. The schema goes out; validated objects come back. Your code never sees raw JSON.

## The simplest case

```python
from pydantic import BaseModel
from pydantic_ai import Agent

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
print(type(result.output))           # <class '__main__.CountryInfo'>
print(result.output.capital)         # 'Paris'
print(result.output.population)      # an int
```

`result.output` is a real `CountryInfo` instance — typed, validated, ready to use. No parsing, no type assertions, no `isinstance` checks.

The output type can be any pydantic model. Nested models, lists, optionals, unions, enums, custom validators — all work because pydantic-ai converts everything to JSON Schema and back through pydantic.

```python
from typing import Literal
from pydantic import BaseModel, Field

class Hotel(BaseModel):
    name: str
    star_rating: int = Field(ge=1, le=5)
    nightly_rate_usd: int

class TripPlan(BaseModel):
    destination: str
    travel_dates: tuple[str, str]      # ISO date strings
    hotels: list[Hotel]                # at least one element
    budget_status: Literal["under", "on", "over"]
    notes: str

agent = Agent(model=..., output_type=TripPlan)
result = await agent.run("Plan a long weekend in Lisbon for two, budget $1500.")
plan = result.output
for h in plan.hotels:
    print(f"{h.name}: ${h.nightly_rate_usd}/night, {h.star_rating}★")
```

Nested validation works. If the model returns a hotel with `star_rating=7`, pydantic catches it, returns the error to the model, the model retries. You never see the invalid response in your code.

## What happens on validation failure

Three scenarios:

### The model returns malformed JSON

Mostly impossible with modern providers that support structured outputs — the JSON is generated under syntactic constraints, so it's parseable by construction. With older providers that fall back to prompt-level instructions, you may occasionally see this; pydantic-ai gives the model the parse error and asks it to retry. After `max_iterations` retries (or whatever your `BudgetGuard` allows), the agent gives up and raises.

### The JSON is valid but doesn't match the schema

Common case. The model omits a field, returns the wrong type for a property, includes an unknown variant of an enum. pydantic-ai gives the model the validation error in the next turn:

> "Validation failed: nightly_rate_usd was a string but expected an integer."

The model corrects on retry. Most of the time, one retry is enough. Occasionally the schema is genuinely ambiguous and the model keeps failing — that's a signal to refine the schema's descriptions (use pydantic `Field(description=...)` to clarify intent) rather than to fight the validation layer.

### The model never produces valid output

If the schema is unsatisfiable (e.g., you specified `Literal["spanish"]` but the user's question is about French history), the model will keep retrying until iteration / token budget runs out. At that point you get an error from `BudgetGuard` (if configured) or pydantic-ai's max-retry exception. Wrap in `try` / `except` and surface a friendly degradation:

```python
from ballast import BudgetExhausted
from pydantic_ai.exceptions import UnexpectedModelBehavior

try:
    result = await agent.run("...")
    return result.output
except (BudgetExhausted, UnexpectedModelBehavior):
    return TripPlan(destination="unknown", ..., notes="Could not produce a plan; please rephrase.")
```

## Unions for deferred / multi-modal output

`output_type` accepts a *union* of types. This is how the framework's HITL pattern works:

```python
from pydantic_ai.tools import DeferredToolRequests

agent = Agent(
    model="openai:gpt-4o",
    output_type=[str, DeferredToolRequests],   # union
    tools=[my_tools],
)
```

Now `result.output` is either a `str` (the agent finished normally) *or* a `DeferredToolRequests` (the agent called a `requires_approval=True` tool that needs human approval before it can run). Your code checks which:

```python
result = await agent.run(query)
if isinstance(result.output, DeferredToolRequests):
    # Open approval cards, await verdicts, resume the agent.
    # ApprovalCapability does this automatically (chapter 21).
    ...
else:
    return result.output    # plain string answer
```

Unions extend to your own types too:

```python
class WeatherAnswer(BaseModel):
    city: str
    forecast: str

class TodoAnswer(BaseModel):
    items: list[str]

class GenericChat(BaseModel):
    text: str

agent = Agent(
    model=...,
    output_type=[WeatherAnswer, TodoAnswer, GenericChat],
)

result = await agent.run(user_input)
if isinstance(result.output, WeatherAnswer):
    render_weather_card(result.output)
elif isinstance(result.output, TodoAnswer):
    render_todo_list(result.output)
else:
    render_text(result.output.text)
```

The model picks which variant fits the user's question best. Useful for agents that handle multiple distinct request types and want to render different UIs per type.

## When to keep `output_type` simple — and when not to

The temptation is to design huge output schemas with deep nesting and many optional fields, trying to capture every possible response. Resist it. Large output schemas cause three problems:

1. **Models hallucinate fields they don't have data for.** A schema with 50 optional fields invites the model to fabricate plausible-sounding values for fields it has no information about.
2. **Schema descriptions get ignored.** When a schema is large, the model uses less and less of the per-field description; you end up relying on field names alone.
3. **Token cost.** Large schemas are large prompts. Every call costs more.

Heuristic: an output schema for a single turn should fit in your head. If you can't summarize it in five fields, it's too big.

When you genuinely need rich structured output, split it across multiple agents or use a pattern. For example: agent #1 extracts entities; agent #2 generates summaries about those entities; the orchestration is a `MapReduce` or `PlanAndExecute` (chapters 16, 18) rather than one mega-agent with a 30-field output.

## Composing `output_type` with framework wrappers

Ballast adds two generic wrappers you'll see often: `Scored[T]` (chapter 6) and `Ref[T]` (chapter 5). They compose naturally with `output_type`:

```python
from ballast import Scored, Ref

class ResearchSummary(BaseModel):
    summary: str
    project: Ref[Project]      # typed entity reference

agent = Agent(
    model=...,
    output_type=Scored[ResearchSummary],     # wrap with confidence + rationale
)

result = await agent.run("Research the ML deployment project")
print(result.output.value.summary)                       # the text
print(result.output.value.project)                       # Ref[Project] (just an ID)
print(result.output.rationale)                           # why the LLM is confident
print(result.output.confidence)                          # "low" | "medium" | "high"
```

You're building up typed contracts: `Scored` adds quality signal, `Ref` adds grounded entity references, the inner `ResearchSummary` carries the actual content. Each layer addresses one concern. Chapters 5 and 6 cover the wrappers; for now, know that they slot in cleanly.

## Streaming structured output

When using `agent.run_stream(...)` with an `output_type`, pydantic-ai supports partial structured output — you get incrementally more complete instances as the model generates:

```python
async with agent.run_stream(query) as response:
    async for partial in response.stream(debounce_by=None):
        # partial is a (possibly incomplete) TripPlan
        ...
    final = await response.get_output()
```

Useful for surfacing partial results to a UI. The framework's streaming router emits structured-output events to the assistant-ui frontend automatically; chapter 22 covers the observability pieces.

## Discipline checklist

A small set of habits that pay off:

- **Always set `output_type` unless the agent is pure chat.** Strings are not a contract.
- **Make required fields actually required.** Don't make `id: str | None = None` to "be flexible" — if the model has to produce an id, force it.
- **Use `Field(description=...)`** for fields whose name doesn't fully convey meaning. The model reads these descriptions when picking values.
- **Reject "junk" values explicitly with validators.** A `Literal["small", "medium", "large"]` is better than a free string.
- **Prefer enums or `Literal[...]` over strings** when the value space is small and finite. The model is much better at constrained choice than at free-form generation.
- **Don't catch `ValidationError` to silently coerce.** If the model is consistently producing invalid output, fix the schema or the prompt — don't paper over it.

## What this chapter did NOT cover

- How to pass per-run state (DB connections, user identity) into tools — that's chapter 4.
- `Ref[T]` for grounded entity references — chapter 5.
- `Scored[T]` for confidence + rationale wrappers — chapter 6.
- What happens when the model needs to recover from a bad output (`Reflection` pattern) — chapter 15.

## Where to go next

→ [04-dependencies-and-state.md](04-dependencies-and-state.md) — how to give tools and capabilities the per-run state they need.
