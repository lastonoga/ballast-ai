# Scored[T] Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `Scored[T, ConfidenceT]` generic Pydantic BaseModel + `Confidence` Literal + a small helpers module in a new `ballast.quality.scored/` subpackage. Native composition with existing primitives (no framework changes). 4 test files exercising shape, helpers, scan_output recursion, end-to-end integration.

**Architecture:** `Scored[T, ConfidenceT=Confidence]` (PEP 696 default) — pydantic v2 generic, `frozen=True`. Helpers (`label_to_float`, `aggregate_by_confidence`, `filter_by_min_confidence`, `rank_by_confidence`) live alongside the type in `_confidence.py`. Helpers hardcode default `Confidence` Literal; custom `ConfidenceT` is supported by the model but apps with non-default shape write their own helpers.

**Tech Stack:** Python 3.11+ (typing.TypeVar `default=` requires Python 3.13 OR `typing_extensions.TypeVar` for backports — verify), pydantic v2 (Generic + ConfigDict.frozen), existing `ballast.grounded.scan_output` (no changes), pydantic-ai (native `Agent.output_type` consumption).

**Spec:** `docs/superpowers/specs/2026-05-26-confidence-rationale-design.md`

---

## File Structure (reference for all tasks)

```
src/ballast/quality/                # NEW top-level subpackage
  __init__.py                        # cross-subpackage re-exports (Task 4)
  scored/
    __init__.py                      # public exports (Task 4)
    _confidence.py                   # Confidence + 5 helpers (Task 1)
    _model.py                        # Scored[T, ConfidenceT] (Task 2)

tests/quality/
  __init__.py
  scored/
    __init__.py
    test_confidence.py               # Task 1
    test_model.py                    # Task 2
    test_scan_integration.py         # Task 3
    test_integration.py              # Task 5

src/ballast/__init__.py              # top-level Scored re-export (Task 4)
```

---

## Task 1: `Confidence` Literal + helpers

**Files:**
- Create: `src/ballast/quality/__init__.py` (empty package marker)
- Create: `src/ballast/quality/scored/__init__.py` (empty for now; populated in Task 4)
- Create: `src/ballast/quality/scored/_confidence.py`
- Create: `tests/quality/__init__.py` (empty)
- Create: `tests/quality/scored/__init__.py` (empty)
- Create: `tests/quality/scored/test_confidence.py`

- [ ] **Step 1: Failing test (`tests/quality/scored/test_confidence.py`)**

```python
"""Confidence label + helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ballast.quality.scored._confidence import (
    Confidence,
    aggregate_by_confidence,
    filter_by_min_confidence,
    label_rank,
    label_to_float,
    rank_by_confidence,
)


@dataclass
class _Scored:
    """Minimal stand-in for Scored[T] — helpers duck-type on .confidence/.value."""
    value: str
    confidence: Confidence


def test_label_to_float_mapping() -> None:
    assert label_to_float("low") == pytest.approx(0.33)
    assert label_to_float("medium") == pytest.approx(0.66)
    assert label_to_float("high") == pytest.approx(1.0)


def test_label_rank_ordering() -> None:
    assert label_rank("low") == 0
    assert label_rank("medium") == 1
    assert label_rank("high") == 2


def test_aggregate_buckets_items_by_label() -> None:
    items = [
        _Scored("a", "high"),
        _Scored("b", "low"),
        _Scored("c", "high"),
        _Scored("d", "medium"),
    ]
    out = aggregate_by_confidence(items)
    assert out == {
        "low": ["b"],
        "medium": ["d"],
        "high": ["a", "c"],
    }


def test_aggregate_empty_returns_empty_buckets() -> None:
    out = aggregate_by_confidence([])
    assert out == {"low": [], "medium": [], "high": []}


def test_filter_by_min_confidence_includes_threshold() -> None:
    items = [
        _Scored("a", "low"),
        _Scored("b", "medium"),
        _Scored("c", "high"),
    ]
    assert [it.value for it in filter_by_min_confidence(items, "medium")] == ["b", "c"]
    assert [it.value for it in filter_by_min_confidence(items, "high")] == ["c"]
    assert [it.value for it in filter_by_min_confidence(items, "low")] == ["a", "b", "c"]


def test_rank_by_confidence_descending_with_stable_sort() -> None:
    items = [
        _Scored("a", "medium"),
        _Scored("b", "high"),
        _Scored("c", "medium"),
        _Scored("d", "low"),
        _Scored("e", "high"),
    ]
    ranked = rank_by_confidence(items)
    assert [it.value for it in ranked] == ["b", "e", "a", "c", "d"]


def test_rank_by_confidence_with_secondary_key() -> None:
    items = [
        _Scored("z", "high"),
        _Scored("a", "high"),
        _Scored("m", "low"),
    ]
    ranked = rank_by_confidence(items, secondary_key=lambda it: it.value)
    assert [it.value for it in ranked] == ["a", "z", "m"]


def test_label_rank_raises_on_unknown_label() -> None:
    with pytest.raises(KeyError):
        label_rank("bogus")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run — confirm fail**

Run: `uv run pytest tests/quality/scored/test_confidence.py -v`
Expected: `ModuleNotFoundError: No module named 'ballast.quality'`.

- [ ] **Step 3: Create empty package markers**

- `src/ballast/quality/__init__.py` — empty
- `src/ballast/quality/scored/__init__.py` — empty
- `tests/quality/__init__.py` — empty
- `tests/quality/scored/__init__.py` — empty

- [ ] **Step 4: Implement `src/ballast/quality/scored/_confidence.py`**

```python
"""Default ``Confidence`` Literal + helpers.

Helpers are hardcoded for the default 3-bin Literal labels. Apps using
``Scored[T, CustomConfidenceT]`` write their own helpers — trivial.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from ballast.quality.scored._model import Scored


Confidence = Literal["low", "medium", "high"]
"""Default 3-bin labeled confidence. Pluggable via ``Scored[T, ConfidenceT]``."""

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
    """Bucket items by their confidence label. Returns a dict with all
    three keys always present (empty lists if no entries)."""
    out: dict[Confidence, list[T]] = {"low": [], "medium": [], "high": []}
    for it in items:
        out[it.confidence].append(it.value)
    return out


def filter_by_min_confidence(
    items: list["Scored[T]"], min_label: Confidence,
) -> list["Scored[T]"]:
    """Keep only items with confidence rank >= ``min_label`` rank."""
    threshold = label_rank(min_label)
    return [it for it in items if label_rank(it.confidence) >= threshold]


def rank_by_confidence(
    items: list["Scored[T]"], *,
    secondary_key: Callable[["Scored[T]"], Any] | None = None,
) -> list["Scored[T]"]:
    """Sort descending by confidence (high → low). Stable sort; optional
    secondary key for tie-breaking."""
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

- [ ] **Step 5: Run — confirm pass**

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/ballast/quality tests/quality
git commit -m "feat(scored): Confidence Literal + helpers (label_to_float / aggregate / filter / rank)"
```

---

## Task 2: `Scored[T, ConfidenceT]` generic BaseModel

**Files:**
- Create: `src/ballast/quality/scored/_model.py`
- Create: `tests/quality/scored/test_model.py`

- [ ] **Step 1: Failing test (`tests/quality/scored/test_model.py`)**

```python
"""Scored[T, ConfidenceT] generic BaseModel."""
from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from ballast.quality.scored._model import Scored


class _Note(BaseModel):
    title: str
    body: str


def test_scored_basic_instantiation() -> None:
    note = _Note(title="t", body="b")
    s = Scored[_Note](value=note, rationale="from doc", confidence="high")
    assert s.value is note
    assert s.rationale == "from doc"
    assert s.confidence == "high"


def test_scored_with_list_value() -> None:
    notes = [_Note(title="t1", body="b1"), _Note(title="t2", body="b2")]
    s = Scored[list[_Note]](value=notes, rationale="batched", confidence="medium")
    assert len(s.value) == 2
    assert s.confidence == "medium"


def test_scored_frozen_assignment_raises() -> None:
    s = Scored[str](value="x", rationale="r", confidence="low")
    with pytest.raises(ValidationError):
        s.confidence = "high"  # type: ignore[misc]


def test_scored_rationale_required() -> None:
    with pytest.raises(ValidationError):
        Scored[str](value="x", confidence="high")  # type: ignore[call-arg]


def test_scored_rejects_invalid_confidence_label() -> None:
    with pytest.raises(ValidationError):
        Scored[str](value="x", rationale="r", confidence="bogus")  # type: ignore[arg-type]


def test_scored_empty_rationale_allowed() -> None:
    s = Scored[str](value="x", rationale="", confidence="low")
    assert s.rationale == ""


def test_scored_with_custom_confidence_int() -> None:
    s = Scored[str, int](value="x", rationale="r", confidence=4)
    assert s.confidence == 4


def test_scored_with_custom_confidence_literal() -> None:
    Binary = Literal["safe", "uncertain"]
    s = Scored[str, Binary](value="x", rationale="r", confidence="safe")
    assert s.confidence == "safe"


def test_scored_json_schema_has_required_fields() -> None:
    schema = Scored[_Note].model_json_schema()
    required = set(schema.get("required", []))
    assert "value" in required
    assert "rationale" in required
    assert "confidence" in required


def test_scored_dump_roundtrip() -> None:
    s = Scored[str](value="x", rationale="r", confidence="medium")
    dumped = s.model_dump()
    assert dumped == {"value": "x", "rationale": "r", "confidence": "medium"}
    s2 = Scored[str].model_validate(dumped)
    assert s2 == s
```

- [ ] **Step 2: Run — confirm fail**

Expected: ImportError.

- [ ] **Step 3: Implement `src/ballast/quality/scored/_model.py`**

```python
"""``Scored[T, ConfidenceT]`` — generic wrapper carrying value + rationale + confidence.

Use as a tool / agent / pattern output type:

    async def search() -> Scored[list[Note]]: ...
    agent = Agent(output_type=Scored[Summary])

    async def map_fact(item) -> Scored[Fact]: ...

Default ``ConfidenceT`` is ``Literal["low", "medium", "high"]`` — named
labels recommended by the article's stronger principle to avoid the
mean-reversion that affects numeric scales (1-10).

Apps may override:
    Scored[Fact, int]                              # 1-5 numeric scale
    Scored[Fact, Literal["safe", "uncertain"]]     # binary

Built-in helpers in ``_confidence.py`` (``filter_by_min_confidence`` /
``rank_by_confidence`` / ``aggregate_by_confidence`` / ``label_to_float``)
work only with the default ``Confidence`` Literal. Apps with custom
``ConfidenceT`` write their own helpers.

Composition: ``scan_output`` (from ``ballast.grounded``) recurses into
``Scored.value`` automatically — ``Ref[T]`` fields buried inside the
wrapped value are discovered without any special-case wiring.
"""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

try:
    # PEP 696 — `TypeVar(..., default=...)` lands in CPython 3.13.
    # For 3.11/3.12 we use typing_extensions's backport.
    from typing_extensions import TypeVar as _TypeVar
except ImportError:  # pragma: no cover
    from typing import TypeVar as _TypeVar

from ballast.quality.scored._confidence import Confidence


T = TypeVar("T")
ConfidenceT = _TypeVar("ConfidenceT", default=Confidence)


class Scored(BaseModel, Generic[T, ConfidenceT]):
    """Wraps any value with rationale + confidence label.

    See module docstring for usage examples and composition notes.
    """

    model_config = ConfigDict(frozen=True)

    value: T
    rationale: str
    """One-sentence justification — REQUIRED. Forces CoT-style reasoning
    before the LLM commits to a confidence label."""

    confidence: ConfidenceT


__all__ = ["Scored"]
```

- [ ] **Step 4: Run — confirm pass**

Expected: 10 passed.

NOTE: If `Scored[str, int]` raises at typing-eval time (some pydantic versions don't accept Literal-ConfidenceT replacements via Generic args directly), the test "test_scored_with_custom_confidence_int" or "test_scored_with_custom_confidence_literal" may need adjustment. Run, observe, adjust the test OR the model accordingly. If both tests pass: great. If pydantic complains, mark those two tests as `xfail` with a comment that custom-ConfidenceT support depends on pydantic-version generic handling.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/quality/scored/_model.py tests/quality/scored/test_model.py
git commit -m "feat(scored): Scored[T, ConfidenceT] generic BaseModel (frozen, PEP 696 default)"
```

---

## Task 3: `scan_output` integration verification (no code changes — pure assertion test)

**Files:**
- Create: `tests/quality/scored/test_scan_integration.py`

- [ ] **Step 1: First — locate the grounded scan_output API**

Run:
```
grep -rn "def scan_output\|^scan_output" src/ballast/grounded/ | head -5
```

Note the function signature (e.g., what it returns: a dict, a custom dataclass, etc.). Tests in `tests/grounded/` already exercise it — borrow their patterns.

- [ ] **Step 2: Write the integration test**

`tests/quality/scored/test_scan_integration.py`:
```python
"""scan_output recurses into Scored.value finding Ref[T] fields naturally."""
from __future__ import annotations

from pydantic import BaseModel

from ballast.grounded import Ref, scan_output
from ballast.quality.scored._model import Scored


class _Project(BaseModel):
    id: str
    name: str


class _Note(BaseModel):
    title: str
    project: Ref[_Project]


def test_scan_output_finds_ref_inside_scored_value() -> None:
    note = _Note(
        title="x",
        project=Ref[_Project](id="p-1"),
    )
    scored = Scored[_Note](
        value=note,
        rationale="extracted from doc",
        confidence="high",
    )

    refs = scan_output(scored)
    # Whatever scan_output returns, it MUST contain at least one ref
    # pointing to a _Project entity. Adjust the assertion to match the
    # real scan_output return shape (likely a list[Ref] or dict[type, list[Ref]]).
    assert _has_ref_to(refs, _Project, "p-1"), \
        f"scan_output did not find Ref[_Project] inside Scored.value; got: {refs!r}"


def test_scan_output_finds_refs_inside_scored_list_value() -> None:
    notes = [
        _Note(title="a", project=Ref[_Project](id="p-1")),
        _Note(title="b", project=Ref[_Project](id="p-2")),
    ]
    scored = Scored[list[_Note]](
        value=notes,
        rationale="batch extract",
        confidence="medium",
    )
    refs = scan_output(scored)
    assert _has_ref_to(refs, _Project, "p-1")
    assert _has_ref_to(refs, _Project, "p-2")


def test_scan_output_ignores_rationale_and_confidence_fields() -> None:
    # rationale: str / confidence: Literal contain no Ref-typed values,
    # so scan_output's walker passes over them naturally. This test just
    # asserts that ONLY the Ref inside .value is surfaced — not a string
    # or label scanned as if it were a Ref.
    scored = Scored[_Note](
        value=_Note(title="x", project=Ref[_Project](id="p-1")),
        rationale="the rationale text mentioning p-1 should not produce a Ref",
        confidence="high",
    )
    refs = scan_output(scored)
    # exactly one Ref to _Project (the one in .value.project)
    project_refs = _all_refs_to(refs, _Project)
    assert len(project_refs) == 1


# ---- helpers (adapt to real scan_output return shape) -----------------------

def _has_ref_to(refs, entity_type, id_value) -> bool:
    """Adapt this once you read the real scan_output return shape."""
    for r in _flatten_refs(refs):
        if isinstance(r, Ref) and r.id == id_value and r.entity_type is entity_type:
            return True
    return False


def _all_refs_to(refs, entity_type) -> list:
    return [
        r for r in _flatten_refs(refs)
        if isinstance(r, Ref) and r.entity_type is entity_type
    ]


def _flatten_refs(refs):
    """scan_output may return list[Ref] or dict[type, list[Ref]] — flatten."""
    if isinstance(refs, dict):
        for vs in refs.values():
            yield from vs
        return
    if isinstance(refs, list):
        yield from refs
        return
    # Other shapes — e.g., a dataclass with .by_entity_type
    by_type = getattr(refs, "by_entity_type", None)
    if isinstance(by_type, dict):
        for vs in by_type.values():
            yield from vs
        return
    raise AssertionError(f"unknown scan_output return shape: {refs!r}")
```

- [ ] **Step 3: Run — observe and adjust**

Run: `uv run pytest tests/quality/scored/test_scan_integration.py -v`

Expected: PASS. If the test fails because `_flatten_refs` doesn't match the actual `scan_output` return shape, adjust ONLY the `_has_ref_to` / `_all_refs_to` / `_flatten_refs` helpers to match the real return type. The substantive assertions (Ref found inside `Scored.value`, no Ref from rationale, exactly one Ref) stay the same.

- [ ] **Step 4: If scan_output requires changes to recurse into Scored.value — STOP and report**

Read the actual `scan_output` source in `src/ballast/grounded/_scan.py`. If it iterates `model_fields` and recurses into BaseModel-typed values OR list[BaseModel]-typed values, it will work for `Scored[T]` natively (since `value: T` where `T = SomeBaseModel`-instance recurses). If the walker has a hardcoded blacklist of field names like "rationale" / "confidence" — also fine (it just won't recurse them). If however it has some restriction that would block `Scored[T]` recursion, **STOP and report DONE_WITH_CONCERNS** — we'd need to either special-case Scored OR document the limitation.

- [ ] **Step 5: Commit**

```bash
git add tests/quality/scored/test_scan_integration.py
git commit -m "test(scored): scan_output natively recurses into Scored.value (no code changes)"
```

---

## Task 4: Public API re-exports

**Files:**
- Modify: `src/ballast/quality/scored/__init__.py`
- Modify: `src/ballast/quality/__init__.py`
- Modify: `src/ballast/__init__.py`

- [ ] **Step 1: Subpackage `__init__.py`**

`src/ballast/quality/scored/__init__.py`:
```python
"""``Scored[T, ConfidenceT]`` — typed value + rationale + confidence wrapper."""
from ballast.quality.scored._confidence import (
    Confidence,
    aggregate_by_confidence,
    filter_by_min_confidence,
    label_rank,
    label_to_float,
    rank_by_confidence,
)
from ballast.quality.scored._model import Scored

__all__ = [
    "Confidence",
    "Scored",
    "aggregate_by_confidence",
    "filter_by_min_confidence",
    "label_rank",
    "label_to_float",
    "rank_by_confidence",
]
```

`src/ballast/quality/__init__.py`:
```python
"""Quality-attribute types: Scored (value + rationale + confidence). Future: Cited, Versioned."""
from ballast.quality.scored import Confidence, Scored

__all__ = ["Confidence", "Scored"]
```

- [ ] **Step 2: Edit `src/ballast/__init__.py`**

Read the file. Find existing top-level re-exports block (e.g. near `CircuitBreaker`, `PlanAndExecute`, `GoalDriftDetector`). Add:
```python
from ballast.quality.scored import Scored
```

Add `"Scored"` to `__all__` (alphabetical).

- [ ] **Step 3: Smoke imports**

```
uv run python -c "from ballast import Scored; print('ok')"
```
Expected: `ok`.

```
uv run python -c "
from ballast.quality.scored import (
    Scored, Confidence,
    label_to_float, label_rank,
    aggregate_by_confidence, filter_by_min_confidence, rank_by_confidence,
)
from ballast.quality import Scored as Q_Scored, Confidence as Q_Confidence
print('scored subpackage ok')
"
```
Expected: `scored subpackage ok`.

- [ ] **Step 4: Run full framework suite**

Run: `uv run pytest tests/ -q`
Expected: green. ~20 new tests + existing 638+ still pass.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/__init__.py src/ballast/quality/__init__.py src/ballast/quality/scored/__init__.py
git commit -m "feat(ballast): re-export Scored at top level + quality/ subpackage public API"
```

---

## Task 5: End-to-end integration test + final smoke

**Files:**
- Create: `tests/quality/scored/test_integration.py`

- [ ] **Step 1: Write integration tests exercising composition**

```python
"""End-to-end: Scored[T] with MapReduce / Agent.output_type / CircuitBreaker."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from ballast.quality.scored import (
    Confidence, Scored,
    filter_by_min_confidence, rank_by_confidence,
)
from ballast.resilience.circuit_breaker import (
    Consecutive, CircuitBreaker, BreakerState,
)


# ---- Helpers ---------------------------------------------------------------

class _Clock:
    def __init__(self): self.now = datetime(2026, 1, 1, tzinfo=UTC)
    def __call__(self): return self.now


@pytest.mark.asyncio
async def test_filter_then_rank_pipeline() -> None:
    """Typical reduce-step pattern: low filtered, rest ranked high → low."""
    items: list[Scored[str]] = [
        Scored[str](value="a", rationale="r", confidence="low"),
        Scored[str](value="b", rationale="r", confidence="high"),
        Scored[str](value="c", rationale="r", confidence="medium"),
        Scored[str](value="d", rationale="r", confidence="high"),
    ]
    kept = filter_by_min_confidence(items, "medium")
    ranked = rank_by_confidence(kept, secondary_key=lambda it: it.value)
    assert [it.value for it in ranked] == ["b", "d", "c"]


@pytest.mark.asyncio
async def test_circuit_breaker_treats_low_confidence_as_failure() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        is_success=lambda r: isinstance(r, Scored) and r.confidence != "low",
        clock=_Clock(),
    )

    async def low_conf_extract() -> Scored[str]:
        return Scored[str](value="x", rationale="r", confidence="low")

    # Two low-confidence returns → CB opens
    await cb.call(low_conf_extract)
    await cb.call(low_conf_extract)
    assert cb.stats().state == BreakerState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_high_confidence_keeps_closed() -> None:
    cb = CircuitBreaker(
        threshold_factory=lambda: Consecutive(2),
        is_success=lambda r: isinstance(r, Scored) and r.confidence != "low",
        clock=_Clock(),
    )

    async def high_conf() -> Scored[str]:
        return Scored[str](value="x", rationale="r", confidence="high")

    await cb.call(high_conf)
    await cb.call(high_conf)
    assert cb.stats().state == BreakerState.CLOSED


@pytest.mark.asyncio
async def test_aggregate_then_summarize_pattern() -> None:
    """Apps can bucket items by confidence and feed buckets to LLM separately."""
    from ballast.quality.scored import aggregate_by_confidence

    items: list[Scored[str]] = [
        Scored[str](value="fact-a", rationale="r", confidence="high"),
        Scored[str](value="fact-b", rationale="r", confidence="medium"),
        Scored[str](value="fact-c", rationale="r", confidence="high"),
        Scored[str](value="fact-d", rationale="r", confidence="low"),
    ]
    buckets = aggregate_by_confidence(items)
    assert buckets["high"] == ["fact-a", "fact-c"]
    assert buckets["medium"] == ["fact-b"]
    assert buckets["low"] == ["fact-d"]
```

- [ ] **Step 2: Run — confirm pass**

Run: `uv run pytest tests/quality/scored/test_integration.py -v`
Expected: 4 passed.

- [ ] **Step 3: Run full framework + Scored-specific suite as final smoke**

```
uv run pytest tests/ -q
uv run pytest tests/quality/ -v
```

Expected: full suite green; Scored suite all passing.

- [ ] **Step 4: Smoke import full framework + new type**

```
uv run python -c "
from ballast import (
    Ballast, BallastSettings,
    Scored,
    CircuitBreaker, PlanAndExecute,
    CoALABase, CoALAUnit, as_workflow, as_tool, as_capability,
    GoalDriftDetector, with_drift_monitor,
)
from ballast.quality.scored import (
    Confidence, label_to_float, label_rank,
    aggregate_by_confidence, filter_by_min_confidence, rank_by_confidence,
)
print('all imports ok')
"
```
Expected: `all imports ok`.

- [ ] **Step 5: Commit**

```bash
git add tests/quality/scored/test_integration.py
git commit -m "test(scored): end-to-end composition (filter+rank pipeline, CB is_success bridge)"
```

---

## Self-Review (against the spec)

**Spec coverage:**

| Spec section | Tasks |
|---|---|
| File structure + public API | Tasks 1-4 |
| `Confidence` Literal + helpers (label_to_float / rank / aggregate / filter / rank) | Task 1 |
| `Scored[T, ConfidenceT]` generic BaseModel (frozen, PEP 696 default) | Task 2 |
| `scan_output` integration verification | Task 3 |
| Top-level re-exports + subpackage `__init__` | Task 4 |
| End-to-end integration tests (MapReduce-style pipeline, CB bridge) | Task 5 |

**Placeholder scan:** No TBDs/TODOs/vague-step-without-code. Each step has complete code or exact command + expected output. Task 3 explicitly tolerates `scan_output` return-shape variation via `_flatten_refs` helper — that's intentional defensive design, not a placeholder.

**Type consistency:**
- `Confidence` Literal `["low", "medium", "high"]` consistent across Tasks 1, 2, 4.
- `Scored[T, ConfidenceT]` field names (`value` / `rationale` / `confidence`) consistent across Tasks 1 (helpers duck-type on these), 2 (model definition), 3 (scan-integration test), 5 (integration tests).
- Helper signatures (`aggregate_by_confidence(items) -> dict[Confidence, list[T]]` etc.) consistent in Tasks 1 (impl) and 5 (caller).
- `frozen=True` consistent: Task 2 sets it; Task 2 test asserts it raises on assignment.

**Known plan-vs-spec gap:** Custom `ConfidenceT` (non-default Literal or int) is supported by the model (Task 2's `Scored[str, int]` test) but helpers (Task 1) hardcode default `Confidence` — this is explicitly documented in spec + plan ("apps with non-default scale write their own helpers"). Not a gap, a known scope boundary.
