# Ballast Documentation

**Ballast is a Python framework for shipping production-grade agentic LLM applications.**
It wires `pydantic-ai` + `DBOS` + `FastAPI` into a coherent stack and ships opinionated patterns, capabilities, and resilience primitives so apps reach production without re-inventing the engineering scaffolding around LLMs.

> **The problem we solve:** agentic systems don't fail because foundation models aren't smart enough. They fail because the *infrastructure* around the LLM — durability, drift detection, HITL gates, circuit breakers, structured I/O, typed handoffs — is hand-rolled per project and inevitably under-built. Ballast IS that infrastructure, opinionated but compositionally open.

---

## Documentation map (Diátaxis)

| Quadrant | Question it answers | Where to go |
|---|---|---|
| **[Tutorial](tutorial/)** | "I'm new. Walk me through it." | Hands-on, end-to-end, builds something real |
| **[How-to](howto/)** | "I have a specific task. How?" | Recipes for common production needs |
| **[Reference](reference/)** | "What does X do exactly?" | API surfaces, types, configuration |
| **[Explanation](explanation/)** | "Why is it built this way?" | Design rationale, article mapping, philosophy |

---

## Start here

- New to Ballast → [tutorial/01-quickstart-notes-app.md](tutorial/01-quickstart-notes-app.md)
- Production-pain mapping → [explanation/article-pain-points.md](explanation/article-pain-points.md)
- "Why this framework?" → [explanation/why-ballast.md](explanation/why-ballast.md)
- "What's the stack?" → [explanation/architecture-overview.md](explanation/architecture-overview.md)
- "Can I customize X?" — yes, almost everywhere → [explanation/customization-everywhere.md](explanation/customization-everywhere.md)

---

## The stack at a glance

| Layer | What it does | Provider |
|---|---|---|
| Model + tool calling | LLM provider abstraction, structured outputs, typed tools, capability hooks | **`pydantic-ai`** |
| Workflow durability | Crash-safe workflows, replay-safe steps, durable signals/HITL waits | **`DBOS`** |
| HTTP surface | SSE streaming, HITL endpoints, A2A protocol, thread CRUD | **`FastAPI`** |
| Data contracts | Models, validation, JSON Schema for LLM-facing types | **`pydantic v2`** |
| Persistence (optional) | Thread/message/approval state | **`SQLModel` + Postgres + Alembic** |
| Observability | Tracing, evals, drift detection | **`logfire`, OTel** |
| Frontend (optional) | Chat UI, HITL panels, streaming | **`assistant-ui`** (notes-app demo) |

**Ballast's job:** the *composition glue*. Patterns (`MapReduce`, `Reflection`, `DivergentConvergent`, `PlanAndExecute`), capabilities (`BudgetGuard`, `GoalDriftDetector`, `SemanticLoopDetector`, `ApprovalCapability`), resilience primitives (`CircuitBreaker`), grounded reference handling (`Ref[T]`, `GroundedAgent`), HITL channels (`UICardChannel`, `ThreadChannel`), and the CoALA cognitive-architecture unit (`CoALAUnit` + 3 adapters).

Each underlying library does its job; Ballast doesn't replace any of them. It *wires* them with a uniform contract surface so apps don't re-invent reliability.

---

## What Ballast is NOT

- ❌ A model provider (use OpenRouter / OpenAI / Anthropic / etc. via pydantic-ai)
- ❌ A vector database (apps wire their own; framework exposes `Embedder` Protocol)
- ❌ A scheduling/queue system (DBOS handles this)
- ❌ A frontend framework (apps use assistant-ui, Next.js, or their own UI)
- ❌ Magic. Every "automatic" thing is implemented as a Protocol you can swap.

---

## Contributing / development

This repository develops the framework + a reference app (`examples/notes-app/`). Skills + specs + plans for each subsystem live under `docs/superpowers/`. See [development docs](development/) for setup.

For framework design history (Why CoALA Unit Architecture? Why no global Episodic Memory facade?) see [explanation/changelog-of-decisions.md](explanation/changelog-of-decisions.md).
