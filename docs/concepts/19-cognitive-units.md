# 19. Cognitive Units — CoALA

**Prerequisites:** [07-capabilities.md](07-capabilities.md), [14-patterns-intro.md](14-patterns-intro.md), [18-plan-and-execute.md](18-plan-and-execute.md).

**What you'll learn:** the CoALA decision procedure (observe / retrieve / act / learn) as a single Protocol; how `CoALABase` provides sensible defaults; the three adapters that deploy the same unit as a tool / workflow / capability; how units compose inside PlanAndExecute via `UnitStep`.

## Sections

1. Why memory-aware computation needs its own abstraction
2. The four CoALA phases: observe, retrieve, act, learn
3. `CoALAUnit` Protocol — what you have to implement
4. `CoALABase` — defaults that let you override only what matters
5. `as_tool(unit)`: deploy as a pydantic-ai Tool
6. `as_workflow(unit)`: deploy as a `@Durable.workflow` with per-phase memoisation
7. `as_capability(unit)`: deploy as a `BallastCapability` (observe+retrieve in before, learn in after)
8. Apps own all storage — framework calls `retrieve()` and `learn()` without prescribing where
9. `UnitStep` inside PlanAndExecute: planner emits CoALA-unit step references
10. The DBOSConfiguredInstance pattern under the hood
11. Where to go next

## Next

[20-composition.md](20-composition.md) — combining patterns + units + capabilities into one pipeline.
