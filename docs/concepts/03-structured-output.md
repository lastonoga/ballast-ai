# 3. Structured output

**Prerequisites:** [01-agents.md](01-agents.md), [02-tools.md](02-tools.md).

**What you'll learn:** how to force the agent's final reply into a typed pydantic model, why this matters for production reliability, and what happens when validation fails.

## Sections

1. Why "ask the LLM for JSON in the prompt" is not enough
2. `Agent(output_type=MyModel)` — pydantic validates every response
3. Provider-side support: OpenAI structured outputs, Anthropic tool_use, etc. (handled by pydantic-ai)
4. What happens on validation failure (and how `Reflection` / `GroundedRetry` can recover)
5. Output unions: `output_type=[Summary, DeferredToolRequests]` for HITL flows
6. The case for keeping output types small and focused
7. Where to go next

## Next

[04-dependencies-and-state.md](04-dependencies-and-state.md) — passing per-run state into tools and capabilities.
