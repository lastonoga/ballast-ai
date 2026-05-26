# ballast-ai

Production-grade orchestration framework for Pydantic AI agents.

Sub-project #1 (Foundation) is currently being implemented:
- L0 GroundedSchema (`Ref[T]`, resolver, hydration)
- `Pattern` Protocol
- `Det` runtime helpers (`uuid_for`, `IdempotencyInput`)

## Install (dev)

```
uv sync --extra dev
```

## Test

```
uv run pytest
```
