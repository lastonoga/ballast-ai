# 2. Tools

**Prerequisites:** [01-agents.md](01-agents.md).

**What you'll learn:** how to define tools so the agent can take action; how typed arguments turn into JSON Schema for the LLM; the difference between `@agent.tool` and `@agent.tool_plain`; what `requires_approval=True` means (and how Ballast bridges it to UI cards in a later chapter).

## Sections

1. Tools are typed functions the agent can call
2. `@agent.tool` (with `RunContext`) vs `@agent.tool_plain` (without)
3. Argument validation: pydantic infers a JSON Schema from your type hints
4. Returning data the agent can reason about
5. Marking dangerous tools with `requires_approval=True`
6. Error handling inside a tool
7. Registering tools on `BallastAgent` subclasses (the class-level decorator)
8. Where to go next

## Next

[03-structured-output.md](03-structured-output.md) — forcing the agent's reply into a typed shape.
