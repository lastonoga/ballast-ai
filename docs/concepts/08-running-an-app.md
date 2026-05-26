# 8. Running an app — the `Ballast()` builder

**Prerequisites:** [07-capabilities.md](07-capabilities.md).

**What you'll learn:** how to assemble agents, repositories, and capabilities into a working FastAPI app; the fluent builder API; what the `Engine` actually does; how the streaming router exposes your agents to a frontend over SSE.

## Sections

1. The journey from "I have an Agent" to "I have a running HTTP service"
2. `Ballast()` — the entry point; fluent setters return `self`
3. Registering agents: `.with_agents([NotesAgent(), TodoApprovalAgent()])`
4. Wiring repositories: `.with_thread_repo(...)`, `.with_approval_repo(...)`
5. Adding capabilities at the framework level vs the agent level
6. `Ballast.fastapi_app()` — what you get out of the box (streaming, approvals, A2A, health)
7. The `Engine` — DI container, lifecycle hooks, ServiceProvider protocol
8. Dependency overrides for testing (`app.dependency_overrides`)
9. Configuration via `BallastSettings` + environment variables
10. Where to go next

## Next

[09-persistence.md](09-persistence.md) — making thread/message/approval state survive restarts.
