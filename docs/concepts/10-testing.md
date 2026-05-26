# 10. Testing

**Prerequisites:** [01-agents.md](01-agents.md), [07-capabilities.md](07-capabilities.md), [08-running-an-app.md](08-running-an-app.md).

**What you'll learn:** how to test agents with pydantic-ai's `TestModel` (no real LLM calls, no network); how to test workflows that use `@Durable.workflow` via the module-scoped DBOS fixture; how to assert on tool-call arguments, capability state, and pattern outcomes.

## Sections

1. The test pyramid for agentic apps (unit / pattern / workflow / integration)
2. `TestModel` for unit tests: scripted outputs, deterministic, fast
3. Asserting on tool-call arguments
4. Testing typed outputs and `Scored[T]`
5. Testing capabilities (counters, drift detection, judge verdicts)
6. The DBOS SQLite fixture for workflow tests (module-scoped)
7. Testing HITL flows with mocked channels
8. Marking integration tests separately (`@pytest.mark.integration`)
9. CI patterns: fast unit suite + nightly integration suite
10. Where to go next

## Next

[11-budget-and-loops.md](11-budget-and-loops.md) — production hardening against runaway behavior.
