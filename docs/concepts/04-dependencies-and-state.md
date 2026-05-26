# 4. Dependencies and state

**Prerequisites:** [01-agents.md](01-agents.md), [02-tools.md](02-tools.md).

**What you'll learn:** how to inject per-run state (repositories, user context, request data) into tools and capabilities via `deps_type`; how `RunContext` works; the rule that per-run state must NOT leak across runs.

## Sections

1. The problem: tools need access to repos / user IDs / request context
2. `Agent(deps_type=NoteToolDeps)` + `RunContext[NoteToolDeps]` in tool signatures
3. Where `deps` is constructed (in your route handler, per request)
4. `current_user_id` ContextVar for cross-cutting auth scope
5. The `for_run` capability hook: stateful capabilities clone themselves per run
6. Don't put long-lived state on Agent / Capability instance attrs
7. Where to go next

## Next

[05-grounded-references.md](05-grounded-references.md) — typed entity references that prevent the agent from hallucinating IDs.
