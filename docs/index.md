# Ballast

**A Python framework for shipping production-grade agentic LLM applications.**

Ballast wires [pydantic-ai](https://ai.pydantic.dev) + [DBOS](https://dbos.dev) + [FastAPI](https://fastapi.tiangolo.com) into a coherent stack and ships opinionated patterns, capabilities, and resilience primitives so apps reach production without re-inventing the engineering scaffolding around LLMs.

!!! note "The problem we solve"
    Agentic systems don't fail because foundation models aren't smart enough. They fail because the *infrastructure* around the LLM — durability, drift detection, HITL gates, circuit breakers, structured I/O, typed handoffs — is hand-rolled per project and inevitably under-built. Ballast IS that infrastructure, opinionated but compositionally open.

---

## Where to go

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Tutorial**

    ---

    New to Ballast? Walk through building an end-to-end notes app.

    [:octicons-arrow-right-24: Start the tutorial](tutorial/README.md)

-   :material-book-open-page-variant:{ .lg .middle } **Concepts**

    ---

    25 chapters that take you from agents to production patterns. Read in order.

    [:octicons-arrow-right-24: Start with Agents](concepts/01-agents.md)

-   :material-lightbulb:{ .lg .middle } **Explanation**

    ---

    Why this framework exists. Article pain points, architecture, design choices.

    [:octicons-arrow-right-24: Why Ballast](explanation/why-ballast.md)

-   :material-api:{ .lg .middle } **Reference**

    ---

    API surfaces, types, configuration.

    [:octicons-arrow-right-24: Reference](reference/README.md)

</div>

---

## The stack at a glance

| Layer | What it does | Provider |
|---|---|---|
| Model + tool calling | LLM provider abstraction, structured outputs, typed tools, capability hooks | **pydantic-ai** |
| Workflow durability | Crash-safe workflows, replay-safe steps, durable signals / HITL waits | **DBOS** |
| HTTP surface | SSE streaming, HITL endpoints, A2A protocol, thread CRUD | **FastAPI** |
| Data contracts | Models, validation, JSON Schema for LLM-facing types | **pydantic v2** |
| Persistence (optional) | Thread / message / approval state | **SQLModel + Postgres + Alembic** |
| Observability | Tracing, evals, drift detection | **logfire, OTel** |
| Frontend (optional) | Chat UI, HITL panels, streaming | **assistant-ui** (notes-app demo) |

Ballast's job is the *composition glue*. Patterns (`MapReduce`, `Reflection`, `DivergentConvergent`, `PlanAndExecute`), capabilities (`BudgetGuard`, `GoalDriftDetector`, `SemanticLoopDetector`, `ApprovalCapability`), resilience primitives (`CircuitBreaker`), grounded reference handling (`Ref[T]`, `GroundedAgent`), HITL channels (`UICardChannel`, `ThreadChannel`), and the CoALA cognitive-architecture unit (`CoALAUnit` + 3 adapters).

Each underlying library does its job; Ballast doesn't replace any of them. It *wires* them with a uniform contract surface so apps don't re-invent reliability.

---

## What Ballast is NOT

- A model provider (use OpenRouter / OpenAI / Anthropic / etc. via pydantic-ai)
- A vector database (apps wire their own; framework exposes `Embedder` Protocol)
- A scheduling / queue system (DBOS handles this)
- A frontend framework (apps use assistant-ui, Next.js, or their own UI)
- Magic. Every "automatic" thing is implemented as a Protocol you can swap.

---

## Install

```bash
pip install ballast-ai
```

Requires Python 3.11+. Database optional (in-memory repos for dev).

---

## Quick example

```python
from ballast import Ballast, BallastSettings
from pydantic_ai import Agent

agent = Agent(model="openai:gpt-4o-mini", system_prompt="You take notes.")

app = (
    Ballast(BallastSettings())
    .with_dbos()
    .fastapi(cors="dev")
)
```

That's a runnable FastAPI app with `/threads`, `/approvals`, `/health`, and `/dbos` mounted. Add your own routers for chat streaming on top.

See [chapter 8](concepts/08-running-an-app.md) for the full builder API.

---