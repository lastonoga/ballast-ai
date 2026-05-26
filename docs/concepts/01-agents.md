# 1. Agents

**Prerequisites:** none. This is the first chapter.

**What you'll learn:** what an `Agent` is in Ballast, what Ballast adds on top of vanilla pydantic-ai, when to use `BallastAgent` vs `DurableAgent`, and the lifecycle of a single agent run.

## Sections

1. The pydantic-ai Agent in 90 seconds
2. What Ballast adds (and what stays the same)
3. `BallastAgent` — the base class with sensible defaults
4. `DurableAgent` — when you need crash-safe replay
5. Constructing an agent: `model`, `system_prompt`, `output_type`, `tools`, `capabilities`, `deps_type`
6. Running an agent: `agent.run`, `agent.iter`, `agent.run_stream`
7. The single-step lifecycle: before_model_request → model call → after_model_request → tool calls → … → after_run
8. Where to go next

## Next

[02-tools.md](02-tools.md) — defining tools the agent can call.
