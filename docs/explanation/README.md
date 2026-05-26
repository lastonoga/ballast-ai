# Explanation

Understanding-oriented. Background, design rationale, philosophy.

> **You're here because:** you want to know *why* Ballast is built this way before deciding whether to adopt it / contribute to it / fork it.

## Pages

### Mission + motivation
- **[why-ballast.md](why-ballast.md)** — the 5-minute pitch: what production pain we solve, three core beliefs, when not to use this framework
- **[article-pain-points.md](article-pain-points.md)** — a row-by-row mapping from the production-failures article to the specific Ballast primitive that addresses each pain. Code snippets included.

### Architecture
- **[architecture-overview.md](architecture-overview.md)** — stack diagram, layer responsibilities, compositional flow through a single agent step
- **[customization-everywhere.md](customization-everywhere.md)** — Protocol-first design, what's pluggable, what's hardcoded and why

### Subsystem deep-dives (planned)
- `explanation/coala-cognitive-architecture.md` — CoALA paper synthesis, why we chose Unit + 3 adapters over heavy memory facades
- `explanation/compounding-error-economics.md` — quantitative argument for stacking capabilities
- `explanation/hitl-design-philosophy.md` — exception escalation > routine review, why we have multiple channels
- `explanation/durability-without-postgres.md` — DBOS state model, replay semantics, in-memory vs persistent

### Design decision changelog (planned)
- `explanation/changelog-of-decisions.md` — chronological list of significant architectural pivots
  - Why we deleted EpisodicMemory/SemanticMemory facades and switched to CoALAUnit
  - Why `Scored[T]` uses Literal labels instead of int 1-5
  - Why `as_workflow` had to use `DBOSConfiguredInstance` pattern
  - Why we don't auto-hydrate `Ref[T]` tool inputs (yet)

### Comparisons (planned)
- `explanation/vs-langchain-langgraph.md`
- `explanation/vs-crewai.md`
- `explanation/vs-vanilla-pydantic-ai.md`

---

**Start here:** [why-ballast.md](why-ballast.md). Then [article-pain-points.md](article-pain-points.md) for concrete mapping.
