# 2. Tools

**Prerequisites:** [01-agents.md](01-agents.md).

## Introduction

An agent without tools is a chatbot — it can talk about things but it can't *do* things. The moment your agent needs to look something up, change state, call an external API, or interact with the world in any way, you need tools.

A tool, in pydantic-ai (and therefore in Ballast), is just a typed Python function the agent is allowed to call. The framework turns your function signature into a JSON Schema that goes to the model alongside the system prompt. The model then decides — call by call — whether and when to invoke the tool, and with what arguments. When it calls, you get to run real Python: hit a database, call an API, transform data, return a result. The result flows back to the model as part of the conversation history, and the agent continues.

This chapter covers how to define tools, the difference between `@agent.tool` and `@agent.tool_plain`, what `requires_approval=True` does (and the bridge that turns it into HITL cards, covered in chapter 21), and the practical patterns for organizing tools across a real codebase.

## The mental model

Three things are happening when you add a tool to an agent:

1. **Schema extraction** — pydantic-ai inspects your function's type hints, builds a pydantic model of its arguments, and converts that to JSON Schema. The schema is sent to the LLM as part of the tool definition.
2. **Dispatch** — when the model emits a tool call, pydantic-ai parses the call's arguments against your schema. If the schema doesn't match, the model gets a validation error in the next turn and gets to retry.
3. **Execution** — your function runs with validated, typed arguments. Whatever you return is serialized (JSON, mostly) and sent back to the model as the tool's result.

The agent's loop runs your tool synchronously from its perspective — it doesn't proceed to the next model call until your tool returns. Slow tool = slow agent. Plan accordingly.

## The simplest tool

```python
from pydantic_ai import Agent

agent = Agent(model="openai:gpt-4o-mini")

@agent.tool_plain
async def get_weather(city: str) -> dict:
    """Look up the current weather for a city."""
    return await weather_api.fetch(city)

result = await agent.run("Should I bring an umbrella to London today?")
```

That's it. Three things to notice:

- **`@agent.tool_plain`** registers the function as a tool. The "plain" suffix means "no `RunContext`" — your function takes only the arguments the model supplies.
- **The docstring matters.** It becomes the tool's description in the JSON Schema sent to the model. Be specific. "Look up the current weather for a city" is fine; "weather thing" is not.
- **Type hints are mandatory.** `city: str` becomes a typed property in the JSON Schema. Without hints, pydantic-ai can't build a schema, and registration fails.

## `@agent.tool` vs `@agent.tool_plain`

Tools come in two flavors, distinguished by whether they receive a `RunContext`:

```python
from pydantic_ai import Agent, RunContext

agent = Agent(model="openai:gpt-4o-mini")

@agent.tool_plain
async def get_weather(city: str) -> dict:
    """City weather. No context needed — public API."""
    return await weather_api.fetch(city)

@agent.tool
async def get_user_notes(ctx: RunContext, query: str) -> list[dict]:
    """Search notes belonging to the current user."""
    user_id = ctx.deps.user_id    # comes from the per-run deps object
    return await notes_repo.search_for_user(user_id, query)
```

`@agent.tool_plain` is for stateless tools — public lookups, pure transformations, anything that doesn't need to know who the user is or what the current request looks like. Use it when you can.

`@agent.tool` is for stateful tools — anything that needs the current user, the current thread, a database connection scoped to the request, or any other context. The `ctx: RunContext` parameter is *not* sent to the model — it's injected by the framework before your function runs. The model still only sees `query: str` in the tool's schema.

The rule of thumb: start with `@agent.tool_plain`. Switch to `@agent.tool` the moment you need anything from the request context. Chapter 4 covers `RunContext` and `deps` in depth.

## Argument validation

The model can produce arbitrary JSON. Your function gets typed, validated arguments. The conversion happens automatically because your type hints become a pydantic model.

```python
from datetime import date
from pydantic import Field

@agent.tool_plain
async def find_meetings(
    on_date: date,
    attendees: list[str],
    duration_minutes: int = Field(ge=15, le=480),
) -> list[dict]:
    """Find meetings on a specific date with given attendees."""
    return await calendar.search(on_date, attendees, duration_minutes)
```

What the model sees: a JSON Schema with `on_date` typed as date (string in ISO format), `attendees` typed as array of strings, `duration_minutes` typed as integer with min/max constraints.

What your function gets: a real `date` object, a real `list[str]`, an `int` between 15 and 480. If the model tries to pass `duration_minutes=600`, pydantic raises a validation error; pydantic-ai catches it and tells the model "your argument is out of range, try again."

This validation is one of the strongest production safeguards you have. It pushes the burden of well-formed arguments onto the model (where retry costs you tokens, not data corruption) and frees your function from defensive parsing.

You can also use pydantic `BaseModel` for complex nested arguments:

```python
from pydantic import BaseModel

class SearchFilters(BaseModel):
    tags: list[str] = []
    project_id: str | None = None
    created_after: date | None = None

@agent.tool_plain
async def search_notes(query: str, filters: SearchFilters) -> list[dict]:
    """Search notes with optional filters."""
    return await notes_repo.search(query, filters)
```

The model sees `filters` as a nested object with typed properties. It builds the nested call structure itself; you receive a real `SearchFilters` instance.

## Returning data

Whatever your tool returns gets serialized and shown to the model. Three patterns work well:

- **Return primitives or dicts** for simple cases: `str`, `int`, `dict`, `list`. Easy to serialize, easy for the model to read.
- **Return a pydantic model** if you want typed structure. The model is `.model_dump()`-ed before going back to the LLM.
- **Return a string for explanations** that you want the model to read literally. Useful for errors that don't deserve to raise an exception (e.g. "no results found" — not an error, just a fact).

What *not* to return: opaque objects (database connections, file handles, datetime-without-tz). They'll either fail to serialize or confuse the model. If you have an entity, return its serializable representation (a dict or a pydantic model), not the entity itself.

```python
@agent.tool
async def get_note(ctx: RunContext, note_id: str) -> dict | str:
    """Fetch a single note by ID."""
    note = await notes_repo.get(note_id)
    if note is None:
        return f"Note {note_id!r} not found."   # string the model can reason about
    return note.model_dump(mode="json")          # serialized typed structure
```

## Marking dangerous tools with `requires_approval=True`

Some tools are irreversible: publishing a post, sending an email, charging a credit card, deleting a record. You don't want the model to call them autonomously. pydantic-ai supports a `requires_approval=True` flag on tool decoration that tells the framework "this tool needs human approval before it runs."

```python
@agent.tool(requires_approval=True)
async def publish_post(ctx: RunContext, title: str, body: str) -> str:
    """Publish a blog post. NEVER runs without human approval."""
    return await blog_client.publish(title=title, body=body)
```

When the model calls this tool, pydantic-ai *doesn't* execute the function. Instead, the agent's run completes with `result.output` set to a `DeferredToolRequests` object. The tool call is *waiting for approval*. Your code is responsible for collecting verdicts and re-running the agent with `deferred_tool_results=...` to either approve (run the tool) or deny (skip it).

In raw pydantic-ai, you'd wire that approval flow yourself: present the pending request to a human, collect the decision, build a `DeferredToolResults` object, call `agent.run(deferred_tool_results=...)` to continue.

In Ballast, you wire it once via `ApprovalCapability`:

```python
from ballast import ApprovalCapability, UICardChannel

approval = ApprovalCapability(tool_card_map={
    "publish_post": (PublishCard, build_publish_card, ui_channel),
})

agent = Agent(model=..., capabilities=[approval])
```

The capability detects `DeferredToolRequests` automatically, opens a UI approval card via the configured channel, waits durably for the human verdict, maps the verdict to `ToolApproved` / `ToolDenied`, and re-runs the agent. The end user sees: "agent wanted to publish this post — approve / edit / reject?" in a UI panel. Chapter 21 covers this in depth.

For now: know that `requires_approval=True` is the right way to mark a dangerous tool. Don't try to enforce approval by leaving the tool unimplemented or wrapping it in custom logic.

## Class-level tool registration on `BallastAgent`

In real apps you typically have multiple agents and want each agent's tools to live with the agent definition rather than at module scope. `BallastAgent` supports class-level tool decorators:

```python
from typing import Any, ClassVar
from ballast import BallastAgent
from pydantic_ai import Agent, RunContext

class NotesAgent(BallastAgent):
    name: ClassVar[str] = "notes"

    def build_agent(self) -> Agent[Any, Any]:
        return Agent(
            model="openai:gpt-4o-mini",
            system_prompt="Help the user manage their notes.",
        )

@NotesAgent.tool
async def search_notes(ctx: RunContext, query: str) -> list[dict]:
    """Search the user's notes."""
    return await notes_repo.search(ctx.deps.user_id, query)

@NotesAgent.tool
async def create_note(ctx: RunContext, title: str, body: str) -> str:
    """Create a new note for the user."""
    return await notes_repo.create(ctx.deps.user_id, title, body)
```

The decorator stores the tool on the class. When `build_agent()` is called and the framework constructs the underlying pydantic-ai `Agent`, all class-registered tools are automatically attached. You don't have to remember to pass `tools=[...]` to the `Agent` constructor — the framework does it for you.

This also walks the MRO, so subclasses inherit their parent's tools. Useful when you have a base agent class with shared tools and specialized subclasses with extra ones.

## Common tool patterns

A few patterns that come up across most apps:

### The "look up something" tool

```python
@agent.tool
async def get_user_settings(ctx: RunContext) -> dict:
    """Look up the current user's settings."""
    return (await user_repo.get(ctx.deps.user_id)).settings.model_dump()
```

Simple, fast, no arguments. Lets the agent ground its responses in the current user's context.

### The "act on something" tool

```python
@agent.tool
async def update_note(ctx: RunContext, note_id: str, new_title: str) -> str:
    """Update a note's title."""
    note = await notes_repo.get(note_id)
    if note is None or note.user_id != ctx.deps.user_id:
        return "Note not found or not yours."
    await notes_repo.update_title(note_id, new_title)
    return f"Updated note {note_id!r} title to {new_title!r}."
```

Three things to notice: (1) returns a string the model can read literally, including the failure case; (2) enforces authorization (`note.user_id != ctx.deps.user_id`) inside the tool, not at the model layer — the model can't bypass it; (3) doesn't raise on "not found" — uses the friendly-error pattern instead.

### The "search-and-return-options" tool

```python
@agent.tool
async def find_candidate_recipients(ctx: RunContext, query: str) -> list[dict]:
    """Find people matching the query. Returns at most 5 candidates."""
    return [
        p.model_dump(mode="json")
        for p in await people_repo.search(ctx.deps.user_id, query, limit=5)
    ]
```

Limit the result set. Models that get 200 results burn tokens trying to process them; models that get 5 pick the right one. Five to ten is the sweet spot for most search tools.

### The "I'm not sure what to do" tool

```python
@agent.tool
async def need_clarification(
    ctx: RunContext,
    question: str,
    options: list[str] = [],
) -> str:
    """Use when you can't decide what to do without user input."""
    # This is just an interaction tool — its real work is signaling to the
    # framework that a clarification panel should open. See chapter 21.
    ...
```

This is a *signal* tool — the agent's only way to say "I'm stuck, I need help." Combined with the HelperAgent pattern (chapter 21), it lets the agent ask the user structured questions.

## Error handling inside tools

Three options when something goes wrong in a tool:

1. **Return a friendly-error string** — recommended for expected failures the model should understand and route around. ("Note not found.", "API quota exceeded for today.", "User does not have permission for this action.")
2. **Raise an exception** — appropriate for programmer errors (bug in tool body, unexpected data shape, DB connection failed). The model sees the exception in the conversation history; pydantic-ai handles the formatting.
3. **Return a typed result with status field** — for tools where success vs. failure matters semantically:

```python
class ToolResult(BaseModel):
    status: Literal["success", "not_found", "denied", "error"]
    message: str
    data: dict | None = None
```

The model can pattern-match on `status` in its reasoning. Use this when the model needs to take different next-step actions depending on outcome.

Don't catch and silently swallow exceptions in tool bodies. The model needs feedback to course-correct; silent failures degrade the agent's behavior over many turns.

## Tools that need to be `Durable.step`-decorated

When your agent is a `DurableAgent` and a tool is non-idempotent (charges money, sends email, modifies state), wrap the tool body in `@Durable.step`:

```python
from ballast import Durable

@MyDurableAgent.tool
async def send_invoice(ctx: RunContext, customer_id: str, amount_usd: int) -> str:
    """Send an invoice to a customer."""
    return await _send_invoice_step(customer_id, amount_usd)

@Durable.step()
async def _send_invoice_step(customer_id: str, amount_usd: int) -> str:
    return await billing_api.send_invoice(customer_id, amount_usd)
```

This makes the tool replay-safe: if the agent's workflow crashes after `send_invoice` succeeded but before the agent's loop wrote the result, DBOS won't re-send the invoice on replay. The step is memoised by (function name, args) — same args → cached result.

For idempotent tools (lookups, reads, anything pure) you don't need `@Durable.step` — re-execution is free. Chapter 24 covers durability in depth.

## What this chapter did NOT cover

- The full `requires_approval` flow with `ApprovalCapability` and UI cards — that's chapter 21.
- `RunContext.deps` setup and the `deps_type` argument — chapter 4.
- Tools that take `Ref[T]` arguments for grounded entity references — chapter 5.
- Returning `Scored[T]` for tools that should advertise confidence in their output — chapter 6.
- Common shared tools the framework provides (HITL, evaluation helpers, etc.) — those are introduced as the relevant subsystems come up in later chapters.

## Where to go next

→ [03-structured-output.md](03-structured-output.md) — making the agent's final reply typed and validated.
