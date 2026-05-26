# 22. Observability

**Prerequisites:** [07-capabilities.md](07-capabilities.md), [08-running-an-app.md](08-running-an-app.md).

**What you'll learn:** how logfire integration is baked into the framework; how to add custom spans via `@traced`; how to bridge to OTel collectors; how cost extractors attribute token spend per call.

## Sections

1. Why classical monitoring (uptime / latency / error rate) misses agent failures
2. `logfire.configure()` at startup
3. Automatic instrumentation: pydantic-ai, FastAPI, asyncpg, DBOS
4. The `@traced` decorator for app-side spans
5. Span attributes — what to put, what NOT to put (PII)
6. Cost extractors: OpenRouter / OpenAI details / provider-specific
7. Cross-service trace correlation via OTel context headers
8. Sampling for high-volume production
9. Log lines vs spans — when to use each
10. Local development without a logfire account
11. Bridging to OTel collector instead of logfire SaaS
12. Where to go next

## Next

[23-evals.md](23-evals.md) — programmatic quality evaluation.
