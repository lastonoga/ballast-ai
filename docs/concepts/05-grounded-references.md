# 5. Grounded references — `Ref[T]`

**Prerequisites:** [03-structured-output.md](03-structured-output.md), [04-dependencies-and-state.md](04-dependencies-and-state.md).

**What you'll learn:** how `Ref[T]` produces typed entity references that prevent the agent from hallucinating UUIDs; how `GroundedAgent` narrows the JSON Schema sent to the LLM into a literal enum of valid IDs; how to hydrate refs into full entities downstream.

## Sections

1. The hallucination problem: agent invents `Note.id = "abc-123"` that doesn't exist
2. `Ref[T]` — a typed wrapper around `id: str` with custom pydantic core schema
3. Using `Ref[Project]` in output models
4. The plain agent doesn't constrain; `GroundedAgent` does
5. `GroundedResolver[T]`: how the framework gets the list of valid IDs
6. Hydration: `result.hydrate(project_repo=...)` returns full models
7. List / optional / nested refs — natural recursion via `scan_output`
8. Cost of large candidate sets (and how to narrow them)
9. Current limitation: tool *input* hydration is not auto (yet)
10. Where to go next

## Next

[06-confidence-and-quality.md](06-confidence-and-quality.md) — adding rationale + confidence signals to outputs.
