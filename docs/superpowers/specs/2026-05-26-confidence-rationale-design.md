# Scored[T] — Confidence + Rationale Typed Return — Design Spec

**Date:** 2026-05-26
**Status:** Approved (proceeding to plan)
**Source motivation:** "Архитектура и надёжность агентных LLM-систем в Production" — Map-phase воркер должен возвращать `(Extracted Information, Rationale, Confidence Score)`. Same article (in the LLM-as-Judge section) warns against abstract numeric scales (1-10) — recommends named categories. We adopt the stronger principle.

## Problem

Agentic systems extracting structured data from documents / API calls / sub-agent results need a uniform way to wrap a payload with:

1. **Rationale** — a short justification (forces LLM through CoT before final answer).
2. **Confidence** — coarse-grained label (so downstream reduce / filter / route by quality).

Today apps must either hand-roll a wrapper per use-case (fragmented) or leave outputs un-scored (lose signal). Framework needs a small composable primitive: `Scored[T]`.

## Goals

- One generic Pydantic BaseModel: `Scored[T, ConfidenceT]` with default `ConfidenceT = Literal["low", "medium", "high"]`.
- Required `rationale: str` field — forces CoT pattern.
- Frozen (immutable after construction).
- Default `Confidence` Literal labels follow article's stronger principle (named categories avoid mean-reversion vs numeric scales like 1-10 / 1-5).
- Composable with existing primitives without modifying them: `MapReduce` / `Reflection` / `PlanAndExecute` / `CoALAUnit` / `CircuitBreaker.is_success` / pydantic-ai `Agent.output_type`.
- Native interop with `scan_output` (no changes needed — walker recurses into `Scored.value` automatically).
- Small set of helpers for the default Confidence shape: `label_to_float`, `aggregate_by_confidence`, `filter_by_min_confidence`, `rank_by_confidence`.
- New top-level subpackage `ballast.quality/` for this and future quality-attribute types (e.g., future `Cited[T]`, `Versioned[T]`).

## Non-goals

- Confidence calibration (does "high" match observed accuracy) — eval concern, not framework primitive.
- Multi-model consensus confidence — separate pattern.
- Token-level perplexity-derived confidence — separate effort.
- Streaming support — pydantic-ai handles structured streaming natively.
- Helpers for custom `ConfidenceT` (apps with non-default scale write their own).
- Mixin / inheritance shape — first cut ships only generic wrapper.

## Architecture

### File structure

```
src/ballast/quality/                       # NEW top-level subpackage
  __init__.py                               # cross-subpackage re-exports
  scored/
    __init__.py                             # subpackage public API
    _model.py                               # Scored[T, ConfidenceT] BaseModel
    _confidence.py                          # Confidence Literal + helpers

tests/quality/
  __init__.py
  scored/
    __init__.py
    test_model.py                           # generic instantiation, validation, frozen
    test_confidence.py                      # label helpers + aggregate/filter/rank
    test_scan_integration.py                # scan_output recurses into Scored.value
    test_integration.py                     # MapReduce / Agent.output_type / CB.is_success
```

Top-level `from ballast import Scored` — yes, consistent with `CircuitBreaker`, `PlanAndExecute`, `GoalDriftDetector`.

### Public API

`from ballast.quality.scored import ...`:
- `Scored[T]` / `Scored[T, ConfidenceT]` — generic BaseModel
- `Confidence` — `Literal["low", "medium", "high"]`
- `label_to_float(label) -> float` — normalize default labels into `[0, 1]`
- `label_rank(label) -> int` — ordering helper (low=0, medium=1, high=2)
- `aggregate_by_confidence(items) -> dict[label, list[T]]`
- `filter_by_min_confidence(items, min_label) -> list[Scored[T]]`
- `rank_by_confidence(items, *, secondary_key=None) -> list[Scored[T]]`

`from ballast.quality import Scored, Confidence` — also exported at subpackage root.

## Components

### `_confidence.py`

```python
"""Default Confidence labels + helpers."""
from __future__ import annotations

from typing import Any, Callable, Literal, TypeVar


Confidence = Literal["low", "medium", "high"]
"""Default 3-bin labeled confidence. Pluggable via Scored[T, ConfidenceT]."""

_LABEL_TO_FLOAT: dict[Confidence, float] = {"low": 0.33, "medium": 0.66, "high": 1.0}
_LABEL_ORDER:    dict[Confidence, int]   = {"low": 0, "medium": 1, "high": 2}


def label_to_float(label: Confidence) -> float:
    """Map a default Confidence label into a [0, 1] float for metrics."""
    return _LABEL_TO_FLOAT[label]


def label_rank(label: Confidence) -> int:
    """Map a default Confidence label into an ordinal rank (low=0, high=2)."""
    return _LABEL_ORDER[label]


T = TypeVar("T")


def aggregate_by_confidence(items: list["Scored[T]"]) -> dict[Confidence, list[T]]:
    """Bucket items by their confidence label. Returns dict with keys
    always present (low / medium / high), empty lists if no entries."""
    out: dict[Confidence, list[T]] = {"low": [], "medium": [], "high": []}
    for it in items:
        out[it.confidence].append(it.value)
    return out


def filter_by_min_confidence(
    items: list["Scored[T]"], min_label: Confidence,
) -> list["Scored[T]"]:
    """Keep only items with confidence rank >= min_label rank."""
    threshold = label_rank(min_label)
    return [it for it in items if label_rank(it.confidence) >= threshold]


def rank_by_confidence(
    items: list["Scored[T]"], *,
    secondary_key: Callable[["Scored[T]"], Any] | None = None,
) -> list["Scored[T]"]:
    """Sort descending by confidence (high → low). Stable sort;
    optional secondary key for tie-breaking."""
    def _key(it: "Scored[T]") -> tuple[int, Any]:
        secondary = secondary_key(it) if secondary_key else 0
        return (-label_rank(it.confidence), secondary)
    return sorted(items, key=_key)


__all__ = [
    "Confidence",
    "aggregate_by_confidence",
    "filter_by_min_confidence",
    "label_rank",
    "label_to_float",
    "rank_by_confidence",
]
```

`from __future__ import annotations` lets the helpers reference `Scored[T]` as a forward string without circular import; actual class lives in `_model.py`.

### `_model.py`

```python
"""``Scored[T, ConfidenceT]`` — generic wrapper carrying value + rationale + confidence."""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from ballast.quality.scored._confidence import Confidence

T = TypeVar("T")
ConfidenceT = TypeVar("ConfidenceT", default=Confidence)  # PEP 696


class Scored(BaseModel, Generic[T, ConfidenceT]):
    """Wraps any value with rationale + confidence label.

    Use as a tool / agent / pattern output type:

        async def search() -> Scored[list[Note]]: ...

        agent = Agent(output_type=Scored[Summary])

        # In a MapReduce map step:
        async def map_fact(item) -> Scored[Fact]: ...

    Default ``ConfidenceT`` is ``Literal["low", "medium", "high"]``
    (named labels — recommended by the article's stronger principle to
    avoid mean-reversion that affects numeric scales like 1-10).

    Apps may override:
        Scored[Fact, int]                              # 1-5 numeric scale
        Scored[Fact, Literal["safe", "uncertain"]]     # binary

    Built-in helpers (``filter_by_min_confidence`` /
    ``rank_by_confidence`` / ``aggregate_by_confidence`` /
    ``label_to_float``) work only with the default ``Confidence`` Literal.
    Apps with custom ``ConfidenceT`` write their own helpers (trivial).

    Composition: ``scan_output`` (from ``ballast.grounded``) recurses
    into ``Scored.value`` automatically — ``Ref[T]`` fields buried inside
    the wrapped value are discovered without any special-case wiring.
    """

    model_config = ConfigDict(frozen=True)

    value: T
    rationale: str
    """One-sentence justification — REQUIRED. Forces CoT-style reasoning
    before the LLM commits to a confidence label."""

    confidence: ConfidenceT


__all__ = ["Scored"]
```

## Data flow (composition examples)

### MapReduce with confidence-aware reduce

```
[caller] await MapReduce(
    map_step=lambda item: extractor.run(item, output_type=Scored[Fact]),
    reduce_step=lambda items: reduce_by_confidence(items),
).run(items)

# reduce_step:
async def reduce_by_confidence(items: list[Scored[Fact]]) -> Summary:
    high_or_med = filter_by_min_confidence(items, "medium")
    ranked = rank_by_confidence(high_or_med)
    return await synthesizer.run(prompt_with(ranked))
```

### Agent native output_type

```
agent = Agent(
    model="openai:gpt-4o",
    system_prompt="Summarize the doc. Provide rationale + confidence.",
    output_type=Scored[Summary],
)
result = await agent.run(doc)
# result.output is a Scored[Summary] instance with .value / .rationale / .confidence
```

### CircuitBreaker treating low confidence as failure

```
cb = CircuitBreaker(
    is_success=lambda r: isinstance(r, Scored) and r.confidence != "low",
)
await cb.call(lambda: extractor.run(item, output_type=Scored[Fact]))
# Repeated low-confidence outputs trip the breaker → fallback policy runs.
```

## Error handling

| Layer | Behaviour |
|---|---|
| LLM returns invalid JSON for `Scored[T]` | pydantic-ai `output_type` validation raises — apps already retry via Reflection or similar |
| LLM returns `confidence: "bogus"` (not in Literal) | Pydantic `ValidationError` — same path as any structured output failure |
| `rationale: ""` empty | Allowed by default — apps wanting non-empty wire a custom Confidence + `Annotated[str, MinLen(1)]` |
| Helper called with non-default `ConfidenceT` value | `KeyError` from `label_rank` — documented as known limitation |
| `Scored[T]` instance mutated | `frozen=True` raises `ValidationError` on assignment |
| `aggregate_by_confidence([])` | Returns `{"low": [], "medium": [], "high": []}` — never raises |
| `filter_by_min_confidence(items, "bogus")` | `KeyError` — caller bug |

## `scan_output` integration

**No code changes required.** `scan_output` recurses into BaseModel-typed fields natively; `Scored.value: T` is just another field. `rationale: str` and `confidence: Literal` contain no `Ref[T]` so the walker ignores them. Verified by `tests/quality/scored/test_scan_integration.py`.

## Testing strategy

```
tests/quality/scored/
  test_model.py             # Scored[T] generic shape:
                            #   - Scored[Foo] instantiation + JSON Schema valid
                            #   - Scored[list[Foo]] supported
                            #   - Scored[Foo, int] custom ConfidenceT works
                            #   - frozen=True: reassignment raises ValidationError
                            #   - rationale required (missing → ValidationError)
                            #   - confidence Literal-validated (bogus → ValidationError)
                            #   - empty rationale="" allowed by default
  test_confidence.py        # Helpers:
                            #   - label_to_float maps low/medium/high → 0.33/0.66/1.0
                            #   - label_rank ordering
                            #   - aggregate_by_confidence: bucketing + empty
                            #   - filter_by_min_confidence: inclusive threshold
                            #   - rank_by_confidence: high → low + stable + secondary_key
                            #   - custom ConfidenceT raises KeyError when fed to helpers
  test_scan_integration.py  # Assert scan_output finds Ref-fields inside Scored.value:
                            #   - Scored[Note] with Ref[Project] in Note.project: Ref found
                            #   - Scored[list[Note]] with multiple Refs: all found
                            #   - rationale/confidence ignored (not Ref-typed)
  test_integration.py       # End-to-end composition:
                            #   - MapReduce: map returns Scored[Fact], reduce uses
                            #     filter_by_min_confidence + rank_by_confidence
                            #   - Agent output_type=Scored[Summary] with TestModel
                            #   - CircuitBreaker.is_success treats low-conf as failure
```

## Composition (no framework changes)

- **`MapReduce`** — map step returns `Scored[T]`, reduce uses filter/rank helpers.
- **`Reflection`** — critic returns `Scored[Critique]`; refiner gates on confidence.
- **`PlanAndExecute`** — step.execute returns `Scored[T]`; dep-consumers access `.value` and `.confidence` via `dep_outputs`.
- **`CoALAUnit`** — `act()` returns `Scored[T]`; learn() receives it as output.
- **`CircuitBreaker`** — `is_success: lambda r: r.confidence != "low"` treats low-conf as failure.
- **`DriftVerdict`** — parallel primitive; no overlap (drift-specific shape with own score).
- **`HelperVerdict[ContextT]`** — apps may compose `Scored[HelperVerdict[X]]` if confidence relevant.
- **pydantic-ai `Agent.output_type=Scored[T]`** — native structured output across all model providers.

## Out of scope

- Confidence calibration metrics + eval harness — apps measure via existing eval framework.
- Multi-model consensus confidence — separate pattern.
- Token-level perplexity-derived confidence — separate effort.
- Streaming-specific support (pydantic-ai handles).
- Mixin / base-class shape — only generic wrapper in first cut.
- Frontend visualisation of confidence buckets — apps build their own UI.
- Auto-recall of low-confidence items into Reflection-loop — separate pattern.

## Open questions for review

None — all design decisions resolved during brainstorm.
