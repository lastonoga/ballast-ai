# Foundation (Sub-project #1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation layer of `pydantic-ai-stateflow`: L0 GroundedSchema (`Ref[T]`, resolver, hydration, escape hatch), `Pattern` Protocol, and `Det` runtime helpers (including `Det.uuid_for` with `IdempotencyInput`). Project skeleton with uv-based tooling, ruff, mypy, pytest. Standalone library; no DBOS, no state, no agent runtime required.

**Architecture:** Pure Pydantic 2.x for L0 (`Ref[T]` is a Pydantic-native type via `__get_pydantic_core_schema__`). `GroundedResolver` recursively scans output_type → finds `Ref[T]` / `Literal` fields → recursively scans context for entity instances → builds dynamic output model via `pydantic.create_model` with `Literal[*ids]` substitutions. `Det.uuid_for(IdempotencyInput)` produces stable UUID5 from a strictly-typed input (`IdempotencyInput` forbids floats and loose dicts to prevent serialization drift). `Pattern` is a Protocol, not a base class.

**Tech Stack:** Python 3.11+, Pydantic 2.x, `uv` for package management, `ruff` for linting, `mypy` (strict mode) for type checking, `pytest` + `pytest-asyncio` for tests.

**Spec sections covered:** Section 1.4 #17 (Type-driven closed sets), Section 2A (L0 GroundedSchema Implementation), Section 4A.0.4 (Asker Protocol — not in this sub-project), Section 4A.0.5 (UnitOfWork — not in this sub-project), Section 4A.0.11 (Pattern as Protocol), Section 4A — Delta 4 (`Det.uuid_for` + IdempotencyInput).

---

## File Structure

```
philadelphia-v1/                       # repo root
├── pyproject.toml                     # uv project, dependencies, tool config
├── README.md
├── .python-version
├── src/pydantic_ai_stateflow/
│   ├── __init__.py                    # public re-exports
│   ├── _typing.py                     # NewType aliases (ActorId, etc); shared TypeVars
│   ├── grounded/
│   │   ├── __init__.py                # public: Ref, GroundedAgent, GroundedResult, GroundedBuildError
│   │   ├── ref.py                     # Ref[T] class + Pydantic core schema
│   │   ├── _spec.py                   # internal: FieldSpec, FieldRole, OutputSpec
│   │   ├── _scan.py                   # internal: scan_output, scan_context
│   │   ├── _build.py                  # internal: build_dynamic (create_model recursion)
│   │   ├── resolver.py                # GroundedResolver (orchestrator)
│   │   ├── agent.py                   # GroundedAgent + GroundedResult
│   │   ├── hydration.py               # HydrationMap
│   │   └── errors.py                  # GroundedBuildError, GroundedHydrationError
│   ├── runtime/
│   │   ├── __init__.py                # public: Det, IdempotencyInput, IdempotencyValue
│   │   ├── idempotency.py             # IdempotencyInput strict type
│   │   └── det.py                     # Det helpers (uuid_for, uuid4, now, random_choice)
│   └── patterns/
│       ├── __init__.py                # public: Pattern Protocol
│       └── protocol.py                # Pattern Protocol
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── grounded/
│   │   ├── __init__.py
│   │   ├── test_ref.py                # Ref class behavior
│   │   ├── test_ref_pydantic.py       # serialization/deserialization
│   │   ├── test_ref_hydrate.py
│   │   ├── test_resolver_scan_output.py
│   │   ├── test_resolver_scan_context.py
│   │   ├── test_resolver_build_simple_ref.py
│   │   ├── test_resolver_build_collections.py
│   │   ├── test_resolver_build_nested.py
│   │   ├── test_resolver_build_enums.py
│   │   ├── test_resolver_errors.py
│   │   ├── test_resolver_constraints_override.py
│   │   ├── test_grounded_agent.py
│   │   └── test_hydration_map.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── test_idempotency.py
│   │   └── test_det.py
│   ├── patterns/
│   │   ├── __init__.py
│   │   └── test_protocol.py
│   └── integration/
│       ├── __init__.py
│       └── test_smoke_end_to_end.py   # full L0 round-trip with FunctionModel
```

---

## Task 1: Project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `README.md`
- Create: `src/pydantic_ai_stateflow/__init__.py` (empty marker)
- Create: `src/pydantic_ai_stateflow/_typing.py` (empty placeholder for now)
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py` (empty)

- [ ] **Step 1: Create `.python-version`**

```
3.11
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "pydantic-ai-stateflow"
version = "0.1.0"
description = "Production-grade orchestration framework for Pydantic AI agents"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "pydantic-ai>=0.0.13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
    "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pydantic_ai_stateflow"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "PT", "RET", "SIM"]
ignore = ["E501"]   # line length already enforced by formatter

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
disallow_any_explicit = false  # Generic Any needed for Pattern[Any, Any]
```

- [ ] **Step 3: Create `README.md`**

```markdown
# pydantic-ai-stateflow

Production-grade orchestration framework for Pydantic AI agents.

See `docs/superpowers/specs/2026-05-15-pydantic-ai-stateflow-engine-design.md`
for the full architecture spec.

Sub-project #1 (Foundation) is currently being implemented:
- L0 GroundedSchema (`Ref[T]`, resolver, hydration)
- `Pattern` Protocol
- `Det` runtime helpers (`uuid_for`, `IdempotencyInput`)

## Install (dev)

\`\`\`
uv sync --extra dev
\`\`\`

## Test

\`\`\`
uv run pytest
\`\`\`
```

- [ ] **Step 4: Create empty package and test placeholders**

```bash
mkdir -p src/pydantic_ai_stateflow tests
touch src/pydantic_ai_stateflow/__init__.py
touch src/pydantic_ai_stateflow/_typing.py
touch tests/__init__.py
touch tests/conftest.py
```

- [ ] **Step 5: Initialize uv and verify install**

```bash
uv sync --extra dev
uv run pytest --collect-only
```

Expected: pytest collects 0 tests, exits with code 5 (no tests yet) — OK.

- [ ] **Step 6: Verify ruff and mypy work on empty project**

```bash
uv run ruff check
uv run mypy src
```

Expected: both pass with no errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .python-version README.md src tests uv.lock
git commit -m "chore: project skeleton with uv, ruff, mypy, pytest"
```

---

## Task 2: `Ref[T]` minimal class

**Files:**
- Create: `src/pydantic_ai_stateflow/grounded/__init__.py`
- Create: `src/pydantic_ai_stateflow/grounded/ref.py`
- Create: `tests/grounded/__init__.py`
- Create: `tests/grounded/test_ref.py`

- [ ] **Step 1: Write the failing test**

`tests/grounded/test_ref.py`:

```python
from uuid import uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref


class Entity(BaseModel):
    id: str
    name: str


def test_ref_stores_id_and_entity_type():
    ent_id = uuid4()
    ref = Ref[Entity](ent_id)
    assert ref.id == ent_id
    assert ref.entity_type is Entity


def test_ref_class_getitem_creates_subscripted_class():
    subscripted = Ref[Entity]
    # Subscripted form must remember the entity type for resolver later
    assert subscripted.__entity_type__ is Entity


def test_ref_equality_by_id_and_type():
    ent_id = uuid4()
    a = Ref[Entity](ent_id)
    b = Ref[Entity](ent_id)
    assert a == b


def test_ref_inequality_different_types():
    class OtherEntity(BaseModel):
        id: str

    ent_id = uuid4()
    a = Ref[Entity](ent_id)
    b = Ref[OtherEntity](ent_id)
    assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/grounded/test_ref.py -v
```

Expected: ImportError — `pydantic_ai_stateflow.grounded` module not found.

- [ ] **Step 3: Implement `Ref[T]` minimal class**

`src/pydantic_ai_stateflow/grounded/__init__.py`:

```python
from pydantic_ai_stateflow.grounded.ref import Ref

__all__ = ["Ref"]
```

`src/pydantic_ai_stateflow/grounded/ref.py`:

```python
from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel

EntityT = TypeVar("EntityT", bound=BaseModel)


class Ref(Generic[EntityT]):
    """Typed UUID reference to an Entity of type EntityT.

    - JSON / LLM layer: plain UUID string (no wrapper object).
    - Python layer:     typed object with `.id` and `.entity_type`,
                        plus `.hydrate(repo)` for materialization.
    """

    __slots__ = ("id",)
    __entity_type__: ClassVar[type[BaseModel] | None] = None

    def __init__(self, id: UUID) -> None:
        self.id = id

    @property
    def entity_type(self) -> type[BaseModel]:
        if self.__class__.__entity_type__ is None:
            raise TypeError(
                "Ref must be subscripted with an entity type, e.g. Ref[MyEntity](uuid)"
            )
        return self.__class__.__entity_type__

    def __class_getitem__(cls, item: type[BaseModel]) -> type["Ref[Any]"]:
        # Per-class cache (not inherited via MRO). `cls.__dict__` is a
        # `mappingproxy` (immutable view) so we cannot call `.setdefault` on
        # it directly — install the cache via `setattr` and check presence
        # with `in cls.__dict__` (MRO-local).
        if "_subscript_cache" not in cls.__dict__:
            cls._subscript_cache = {}  # type: ignore[attr-defined]
        cache: dict[type, type[Ref[Any]]] = cls._subscript_cache  # type: ignore[attr-defined]

        if item not in cache:
            cls_name = f"Ref[{item.__name__}]"
            cache[item] = type(cls_name, (Ref,), {"__entity_type__": item})
        return cache[item]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ref):
            return NotImplemented
        return self.id == other.id and self.entity_type is other.entity_type

    def __hash__(self) -> int:
        return hash((self.id, self.entity_type))

    def __repr__(self) -> str:
        return f"Ref[{self.entity_type.__name__}](id={self.id!r})"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/grounded/test_ref.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: mypy + ruff pass**

```bash
uv run mypy src
uv run ruff check
```

Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded tests/grounded
git commit -m "feat(grounded): Ref[T] minimal class with subscription cache"
```

---

## Task 3: `Ref[T]` Pydantic core schema (UUID string ↔ Ref)

**Files:**
- Modify: `src/pydantic_ai_stateflow/grounded/ref.py`
- Create: `tests/grounded/test_ref_pydantic.py`

- [ ] **Step 1: Write the failing test**

`tests/grounded/test_ref_pydantic.py`:

```python
import json
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import Ref


class Item(BaseModel):
    id: UUID
    name: str


class Decision(BaseModel):
    chosen: Ref[Item]
    rationale: str


def test_decision_serializes_ref_as_uuid_string():
    item_id = uuid4()
    d = Decision(chosen=Ref[Item](item_id), rationale="best fit")
    dumped = d.model_dump(mode="json")
    assert dumped == {"chosen": str(item_id), "rationale": "best fit"}


def test_decision_deserializes_uuid_string_to_ref():
    item_id = uuid4()
    d = Decision.model_validate({"chosen": str(item_id), "rationale": "best fit"})
    assert isinstance(d.chosen, Ref)
    assert d.chosen.id == item_id
    assert d.chosen.entity_type is Item


def test_decision_roundtrip_via_json():
    item_id = uuid4()
    original = Decision(chosen=Ref[Item](item_id), rationale="r")
    restored = Decision.model_validate_json(original.model_dump_json())
    assert restored.chosen == original.chosen
    assert restored.rationale == original.rationale


def test_decision_rejects_non_uuid_string():
    with pytest.raises(ValidationError):
        Decision.model_validate({"chosen": "not-a-uuid", "rationale": "r"})


def test_decision_accepts_uuid_object_directly():
    item_id = uuid4()
    d = Decision.model_validate({"chosen": item_id, "rationale": "r"})
    assert d.chosen.id == item_id
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/grounded/test_ref_pydantic.py -v
```

Expected: `PydanticSchemaGenerationError` — Ref has no Pydantic schema.

- [ ] **Step 3: Add `__get_pydantic_core_schema__` to `Ref`**

In `src/pydantic_ai_stateflow/grounded/ref.py`, replace the `Ref` class with:

```python
from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

EntityT = TypeVar("EntityT", bound=BaseModel)


class Ref(Generic[EntityT]):
    __slots__ = ("id",)
    __entity_type__: ClassVar[type[BaseModel] | None] = None

    def __init__(self, id: UUID) -> None:
        self.id = id

    @property
    def entity_type(self) -> type[BaseModel]:
        if self.__class__.__entity_type__ is None:
            raise TypeError(
                "Ref must be subscripted with an entity type, e.g. Ref[MyEntity](uuid)"
            )
        return self.__class__.__entity_type__

    def __class_getitem__(cls, item: type[BaseModel]) -> type["Ref[Any]"]:
        if "_subscript_cache" not in cls.__dict__:
            cls._subscript_cache = {}  # type: ignore[attr-defined]
        cache: dict[type, type[Ref[Any]]] = cls._subscript_cache  # type: ignore[attr-defined]
        if item not in cache:
            cls_name = f"Ref[{item.__name__}]"
            cache[item] = type(cls_name, (Ref,), {"__entity_type__": item})
        return cache[item]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # Validate using Pydantic's native UUID schema, then wrap into Ref.
        # `no_info_after_validator_function` gives us proper JSON Schema
        # generation ({"type": "string", "format": "uuid"}) which Tasks
        # 11-13 need for LLM-facing dynamic models.
        def _wrap(value: Any) -> Ref[Any]:
            # Same-type passthrough.
            if isinstance(value, cls):
                return value
            # Different-type Ref → re-wrap to enforce field typing.
            if isinstance(value, Ref):
                return cls(value.id)
            # Bare UUID — wrap.
            return cls(value)

        def _serialize(ref: Ref[Any]) -> str:
            return str(ref.id)

        # Use union to accept either Ref instances (from Python) or UUID strings/objects (from JSON).
        # The uuid_schema validates strings/UUID objects; the is_instance_schema allows
        # already-wrapped Ref instances to pass through.
        return core_schema.no_info_after_validator_function(
            function=_wrap,
            schema=core_schema.union_schema([
                core_schema.is_instance_schema(cls),
                core_schema.is_instance_schema(Ref),
                core_schema.uuid_schema(),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize, return_schema=core_schema.str_schema()
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ref):
            return NotImplemented
        return self.id == other.id and self.entity_type is other.entity_type

    def __hash__(self) -> int:
        return hash((self.id, self.entity_type))

    def __repr__(self) -> str:
        return f"Ref[{self.entity_type.__name__}](id={self.id!r})"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/grounded/test_ref_pydantic.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Re-run full test suite — nothing regressed**

```bash
uv run pytest
```

Expected: all tests pass (Task 2 + Task 3 = 9 tests).

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/ref.py tests/grounded/test_ref_pydantic.py
git commit -m "feat(grounded): Ref[T] Pydantic core schema (UUID string ↔ Ref)"
```

---

## Task 4: `Ref[T].hydrate(repo)` async materialization

**Files:**
- Modify: `src/pydantic_ai_stateflow/grounded/ref.py`
- Create: `tests/grounded/test_ref_hydrate.py`

- [ ] **Step 1: Write the failing test**

`tests/grounded/test_ref_hydrate.py`:

```python
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref


class Order(BaseModel):
    id: UUID
    amount: int


class FakeOrderRepo:
    def __init__(self, orders: dict[UUID, Order]) -> None:
        self._orders = orders

    async def load(self, id: UUID) -> Order:
        if id not in self._orders:
            raise KeyError(id)
        return self._orders[id]


@pytest.mark.asyncio
async def test_hydrate_returns_entity_from_repo():
    oid = uuid4()
    order = Order(id=oid, amount=100)
    repo = FakeOrderRepo({oid: order})

    ref = Ref[Order](oid)
    loaded = await ref.hydrate(repo)
    assert loaded is order


@pytest.mark.asyncio
async def test_hydrate_propagates_repo_errors():
    repo = FakeOrderRepo({})
    ref = Ref[Order](uuid4())
    with pytest.raises(KeyError):
        await ref.hydrate(repo)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/grounded/test_ref_hydrate.py -v
```

Expected: `AttributeError: 'Ref' object has no attribute 'hydrate'`.

- [ ] **Step 3: Implement `hydrate`**

In `src/pydantic_ai_stateflow/grounded/ref.py`, add at the end of the `Ref` class (before `__eq__`):

```python
    async def hydrate(self, repo: "RepositoryLike[EntityT]") -> EntityT:
        """Materialize this reference via a repository.

        `repo` must be any object with `async def load(id: UUID) -> EntityT`.
        We deliberately accept any structurally compatible object, not a
        specific Protocol, to keep this module dependency-free.
        """
        return await repo.load(self.id)
```

Also add at the top of the file (after the imports):

```python
from typing import Protocol


class RepositoryLike(Protocol[EntityT]):
    async def load(self, id: UUID) -> EntityT: ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/grounded/test_ref_hydrate.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Re-run full test suite, mypy, ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/ref.py tests/grounded/test_ref_hydrate.py
git commit -m "feat(grounded): Ref[T].hydrate(repo) async materialization"
```

---

## Task 5: `IdempotencyInput` strict type

**Files:**
- Create: `src/pydantic_ai_stateflow/runtime/__init__.py`
- Create: `src/pydantic_ai_stateflow/runtime/idempotency.py`
- Create: `tests/runtime/__init__.py`
- Create: `tests/runtime/test_idempotency.py`

- [ ] **Step 1: Write the failing test**

`tests/runtime/test_idempotency.py`:

```python
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pydantic_ai_stateflow.runtime import IdempotencyInput


def test_accepts_allowed_primitive_types():
    inp = IdempotencyInput(
        namespace="test",
        parts={
            "str_field": "hello",
            "int_field": 42,
            "uuid_field": uuid4(),
            "dt_field": datetime.now(tz=timezone.utc),
            "dec_field": Decimal("12.34"),
            "bool_field": True,
        },
    )
    assert inp.namespace == "test"


def test_rejects_float_values():
    with pytest.raises(ValidationError, match="float"):
        IdempotencyInput(namespace="test", parts={"bad": 1.5})


def test_rejects_unknown_object():
    class Custom:
        pass

    with pytest.raises(ValidationError):
        IdempotencyInput(namespace="test", parts={"bad": Custom()})


def test_is_frozen():
    inp = IdempotencyInput(namespace="t", parts={"a": 1})
    with pytest.raises(ValidationError):
        inp.namespace = "other"  # type: ignore[misc]


def test_canonical_json_is_stable_across_dict_orderings():
    a = IdempotencyInput(namespace="ns", parts={"x": 1, "y": 2})
    b = IdempotencyInput(namespace="ns", parts={"y": 2, "x": 1})
    assert a.canonical_json() == b.canonical_json()


def test_canonical_json_differs_for_different_inputs():
    a = IdempotencyInput(namespace="ns", parts={"x": 1})
    b = IdempotencyInput(namespace="ns", parts={"x": 2})
    assert a.canonical_json() != b.canonical_json()


def test_canonical_json_distinguishes_namespaces():
    a = IdempotencyInput(namespace="A", parts={"x": 1})
    b = IdempotencyInput(namespace="B", parts={"x": 1})
    assert a.canonical_json() != b.canonical_json()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/runtime/test_idempotency.py -v
```

Expected: ImportError — `pydantic_ai_stateflow.runtime` not found.

- [ ] **Step 3: Implement `IdempotencyInput`**

`src/pydantic_ai_stateflow/runtime/__init__.py`:

```python
from pydantic_ai_stateflow.runtime.idempotency import IdempotencyInput, IdempotencyValue

__all__ = ["IdempotencyInput", "IdempotencyValue"]
```

`src/pydantic_ai_stateflow/runtime/idempotency.py`:

```python
from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

# Allowed value types inside `parts`. Frozen by design — no floats, no
# unbounded objects, only stable primitives. New types must be added
# explicitly here AND in `_strict_encoder` below.
IdempotencyValue: TypeAlias = str | int | UUID | datetime | Decimal | bool


def _strict_encoder(obj: Any) -> str:
    """JSON default-encoder that rejects unknown types instead of falling
    back to str(obj). This catches accidentally-passed objects (e.g. floats
    sneaking in via Decimal arithmetic) at serialization time."""
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        # ISO-8601 with timezone info — stable across versions
        if obj.tzinfo is None:
            raise TypeError(f"datetime in IdempotencyInput must be timezone-aware: {obj!r}")
        return obj.isoformat()
    if isinstance(obj, Decimal):
        # Normalise so 1.0 and 1.00 hash the same
        return format(obj.normalize(), "f")
    raise TypeError(f"IdempotencyInput cannot serialize {type(obj).__name__}")


class IdempotencyInput(BaseModel):
    """Strict input type for `Det.uuid_for`.

    Constraints (enforced):
    - `parts` values are only `IdempotencyValue` types (no floats, no loose
      dicts, no custom objects).
    - Frozen / immutable: no mutation after construction.
    - `canonical_json` produces a stable, sort-ordered JSON string.

    Used as the only acceptable input to `Det.uuid_for`, ensuring that
    deterministic UUID5 derivation is robust across Pydantic / Python
    version drift.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    namespace: str
    parts: dict[str, IdempotencyValue]

    @field_validator("parts", mode="before")
    @classmethod
    def _reject_floats(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        for k, v in value.items():
            if isinstance(v, float):
                raise ValueError(
                    f"IdempotencyInput.parts[{k!r}]: float is not allowed "
                    "(use Decimal as a stable replacement)"
                )
            if not isinstance(v, str | int | UUID | datetime | Decimal | bool):
                raise ValueError(
                    f"IdempotencyInput.parts[{k!r}]: type {type(v).__name__} "
                    "is not an allowed IdempotencyValue"
                )
        return value

    def canonical_json(self) -> str:
        payload = {"ns": self.namespace, "parts": dict(sorted(self.parts.items()))}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_strict_encoder)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/runtime/test_idempotency.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/runtime tests/runtime
git commit -m "feat(runtime): IdempotencyInput strict type rejects floats and unknown objects"
```

---

## Task 6: `Det` helpers (`now`, `uuid4`, `random_choice`) without DBOS

**Files:**
- Create: `src/pydantic_ai_stateflow/runtime/det.py`
- Modify: `src/pydantic_ai_stateflow/runtime/__init__.py`
- Create: `tests/runtime/test_det.py`

> **Note:** In Sub-project #3 (Runtime + DI) these helpers will be wrapped as `@DBOS.step` decorators. For Sub-project #1 they are plain async functions — the DBOS integration is an additive change.

- [ ] **Step 1: Write the failing test**

`tests/runtime/test_det.py`:

```python
from datetime import datetime, timezone
from uuid import UUID

import pytest

from pydantic_ai_stateflow.runtime import Det


@pytest.mark.asyncio
async def test_now_returns_timezone_aware_datetime():
    result = await Det.now()
    assert isinstance(result, datetime)
    assert result.tzinfo is timezone.utc


@pytest.mark.asyncio
async def test_uuid4_returns_unique_uuids():
    a = await Det.uuid4()
    b = await Det.uuid4()
    assert isinstance(a, UUID)
    assert isinstance(b, UUID)
    assert a != b


@pytest.mark.asyncio
async def test_random_choice_returns_one_of_sequence():
    seq = ["a", "b", "c"]
    chosen = await Det.random_choice(seq)
    assert chosen in seq


@pytest.mark.asyncio
async def test_random_choice_empty_raises():
    with pytest.raises(IndexError):
        await Det.random_choice([])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/runtime/test_det.py -v
```

Expected: `ImportError: cannot import name 'Det'`.

- [ ] **Step 3: Implement `Det` minus `uuid_for` (added in Task 7)**

`src/pydantic_ai_stateflow/runtime/det.py`:

```python
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import TypeVar
from uuid import UUID, uuid4 as _uuid4

T = TypeVar("T")


class Det:
    """Deterministic-recorded helpers.

    In Sub-project #3 these methods will be decorated with `@DBOS.step`
    so their results are recorded durably and replayed verbatim. In
    Sub-project #1 they are plain async functions — the decorator is
    an additive non-breaking change.
    """

    @staticmethod
    async def now() -> datetime:
        return datetime.now(tz=timezone.utc)

    @staticmethod
    async def uuid4() -> UUID:
        return _uuid4()

    @staticmethod
    async def random_choice(seq: list[T]) -> T:
        return random.choice(seq)
```

Modify `src/pydantic_ai_stateflow/runtime/__init__.py`:

```python
from pydantic_ai_stateflow.runtime.det import Det
from pydantic_ai_stateflow.runtime.idempotency import IdempotencyInput, IdempotencyValue

__all__ = ["Det", "IdempotencyInput", "IdempotencyValue"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/runtime/test_det.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/runtime/det.py src/pydantic_ai_stateflow/runtime/__init__.py tests/runtime/test_det.py
git commit -m "feat(runtime): Det.now / uuid4 / random_choice (pre-DBOS)"
```

---

## Task 7: `Det.uuid_for(IdempotencyInput)` deterministic UUID5

**Files:**
- Modify: `src/pydantic_ai_stateflow/runtime/det.py`
- Modify: `tests/runtime/test_det.py`

- [ ] **Step 1: Add failing tests for `Det.uuid_for`**

Append to `tests/runtime/test_det.py`:

```python
from uuid import UUID
from pydantic_ai_stateflow.runtime import IdempotencyInput


@pytest.mark.asyncio
async def test_uuid_for_same_input_same_uuid():
    a = await Det.uuid_for(IdempotencyInput(namespace="ns", parts={"x": 1, "y": 2}))
    b = await Det.uuid_for(IdempotencyInput(namespace="ns", parts={"y": 2, "x": 1}))
    assert isinstance(a, UUID)
    assert a == b


@pytest.mark.asyncio
async def test_uuid_for_different_input_different_uuid():
    a = await Det.uuid_for(IdempotencyInput(namespace="ns", parts={"x": 1}))
    b = await Det.uuid_for(IdempotencyInput(namespace="ns", parts={"x": 2}))
    assert a != b


@pytest.mark.asyncio
async def test_uuid_for_different_namespace_different_uuid():
    a = await Det.uuid_for(IdempotencyInput(namespace="A", parts={"x": 1}))
    b = await Det.uuid_for(IdempotencyInput(namespace="B", parts={"x": 1}))
    assert a != b


@pytest.mark.asyncio
async def test_uuid_for_is_uuid5():
    a = await Det.uuid_for(IdempotencyInput(namespace="t", parts={"x": 1}))
    # UUID5 version is 5
    assert a.version == 5
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/runtime/test_det.py -v
```

Expected: 4 new tests fail with `AttributeError: type object 'Det' has no attribute 'uuid_for'`.

- [ ] **Step 3: Implement `Det.uuid_for`**

In `src/pydantic_ai_stateflow/runtime/det.py`, replace the file with:

```python
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import TypeVar
from uuid import UUID, uuid4 as _uuid4, uuid5

from pydantic_ai_stateflow.runtime.idempotency import IdempotencyInput

T = TypeVar("T")

# A fixed namespace UUID for all uuid_for derivations within this framework.
# Generated once via `uuid4()`; hardcoded so derivations are reproducible
# across processes and machines.
_UUID_NAMESPACE = UUID("ad9c8e22-1bc4-4a4f-9c40-d9c4f4ad7e10")


class Det:
    """Deterministic-recorded helpers.

    In Sub-project #3 these methods will be decorated with `@DBOS.step` so
    their results are recorded durably and replayed verbatim across crashes.
    For now they are plain async functions — the decorator is an additive
    non-breaking change.
    """

    @staticmethod
    async def now() -> datetime:
        return datetime.now(tz=timezone.utc)

    @staticmethod
    async def uuid4() -> UUID:
        return _uuid4()

    @staticmethod
    async def random_choice(seq: list[T]) -> T:
        return random.choice(seq)

    @staticmethod
    async def uuid_for(inputs: IdempotencyInput) -> UUID:
        """Deterministic UUID5 from a strict-typed input.

        Why a `@DBOS.step` (in Sub-project #3):
        - The result will be cached by DBOS event log and replayed verbatim.
        - This eliminates ANY risk that serialization variance across
          Pydantic / Python upgrades produces a different UUID on replay.

        Why `IdempotencyInput` (not Any / dict):
        - Type-level guarantee: no floats, no loose dicts, only stable
          primitives. Caller cannot pass `{"amount": 1.0}` accidentally.
        """
        canonical = inputs.canonical_json()
        return uuid5(_UUID_NAMESPACE, canonical)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/runtime/test_det.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/runtime/det.py tests/runtime/test_det.py
git commit -m "feat(runtime): Det.uuid_for deterministic UUID5 from IdempotencyInput"
```

---

## Task 8: `Pattern` Protocol

**Files:**
- Create: `src/pydantic_ai_stateflow/patterns/__init__.py`
- Create: `src/pydantic_ai_stateflow/patterns/protocol.py`
- Create: `tests/patterns/__init__.py`
- Create: `tests/patterns/test_protocol.py`

- [ ] **Step 1: Write the failing test**

`tests/patterns/test_protocol.py`:

```python
from typing import ClassVar
from uuid import UUID, uuid4

import pytest

from pydantic_ai_stateflow.patterns import Pattern


class ConcretePattern:
    """Has all attributes Pattern protocol requires."""

    name: ClassVar[str] = "concrete"

    async def run(self, input: int, *, tenant_id: UUID) -> int:
        return input * 2


class WrongPattern:
    """Missing `run` method."""

    name: ClassVar[str] = "wrong"


def test_concrete_pattern_satisfies_protocol():
    instance: Pattern[int, int] = ConcretePattern()
    assert instance.name == "concrete"


def test_wrong_pattern_does_not_satisfy_protocol_at_runtime_when_checked():
    # Pattern is `@runtime_checkable`; isinstance check enforces structure.
    assert isinstance(ConcretePattern(), Pattern)
    assert not isinstance(WrongPattern(), Pattern)


@pytest.mark.asyncio
async def test_pattern_run_returns_expected():
    p = ConcretePattern()
    result = await p.run(5, tenant_id=uuid4())
    assert result == 10
```

- [ ] **Step 2: Run test — verify it fails**

```bash
uv run pytest tests/patterns/test_protocol.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `Pattern` Protocol**

`src/pydantic_ai_stateflow/patterns/__init__.py`:

```python
from pydantic_ai_stateflow.patterns.protocol import Pattern

__all__ = ["Pattern"]
```

`src/pydantic_ai_stateflow/patterns/protocol.py`:

```python
from __future__ import annotations

from typing import ClassVar, Protocol, TypeVar, runtime_checkable
from uuid import UUID

InT = TypeVar("InT", contravariant=True)
OutT = TypeVar("OutT", covariant=True)


@runtime_checkable
class Pattern(Protocol[InT, OutT]):
    """Structural type — Patterns are plain classes implementing this contract.

    NOT a base class (post code-review). Removes incentive to add hidden
    base behavior. Concrete patterns (`Reflection`, `MapReduce`,
    `MutationPipeline`, etc.) are regular classes that satisfy the protocol.

    Tenant_id is always a kwarg of `run` (canonical carrier per 4A.0.6).
    """

    name: ClassVar[str]

    async def run(self, input: InT, *, tenant_id: UUID) -> OutT: ...
```

- [ ] **Step 4: Run test — verify it passes**

```bash
uv run pytest tests/patterns/test_protocol.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/patterns tests/patterns
git commit -m "feat(patterns): Pattern Protocol (runtime_checkable, structural)"
```

---

## Task 9: `GroundedResolver._scan_output` — detect field roles

**Files:**
- Create: `src/pydantic_ai_stateflow/grounded/_spec.py`
- Create: `src/pydantic_ai_stateflow/grounded/_scan.py`
- Create: `tests/grounded/test_resolver_scan_output.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_resolver_scan_output.py`:

```python
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._scan import scan_output
from pydantic_ai_stateflow.grounded._spec import FieldRole


class Item(BaseModel):
    id: UUID
    name: str


class Status(BaseModel):
    state: Literal["draft", "ready", "sent"]


def test_scan_detects_ref_field():
    class Out(BaseModel):
        chosen: Ref[Item]
        rationale: str

    spec = scan_output(Out)
    assert spec.fields["chosen"].role == FieldRole.REF
    assert spec.fields["chosen"].target_type is Item
    assert spec.fields["rationale"].role == FieldRole.FREE


def test_scan_detects_list_of_refs():
    class Out(BaseModel):
        chosen: list[Ref[Item]]

    spec = scan_output(Out)
    assert spec.fields["chosen"].role == FieldRole.LIST_REF
    assert spec.fields["chosen"].target_type is Item


def test_scan_detects_optional_ref():
    class Out(BaseModel):
        maybe: Optional[Ref[Item]]  # noqa: UP007 — explicit Optional for test

    spec = scan_output(Out)
    assert spec.fields["maybe"].role == FieldRole.OPTIONAL_REF
    assert spec.fields["maybe"].target_type is Item


def test_scan_detects_nested_model():
    class Inner(BaseModel):
        chosen: Ref[Item]

    class Out(BaseModel):
        inner: Inner

    spec = scan_output(Out)
    assert spec.fields["inner"].role == FieldRole.NESTED
    assert spec.fields["inner"].nested_spec is not None
    assert spec.fields["inner"].nested_spec.fields["chosen"].role == FieldRole.REF


def test_scan_detects_list_of_nested_models():
    class Inner(BaseModel):
        chosen: Ref[Item]

    class Out(BaseModel):
        items: list[Inner]

    spec = scan_output(Out)
    assert spec.fields["items"].role == FieldRole.LIST_NESTED
    assert spec.fields["items"].nested_spec is not None


def test_scan_detects_literal_field():
    class Out(BaseModel):
        state: Literal["draft", "ready", "sent"]

    spec = scan_output(Out)
    assert spec.fields["state"].role == FieldRole.LITERAL
    assert spec.fields["state"].literal_values == ("draft", "ready", "sent")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_resolver_scan_output.py -v
```

Expected: ImportError on `_scan` / `_spec`.

- [ ] **Step 3: Implement `_spec.py` and `_scan.py`**

`src/pydantic_ai_stateflow/grounded/_spec.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FieldRole(StrEnum):
    REF = "ref"
    LIST_REF = "list_ref"
    OPTIONAL_REF = "optional_ref"
    NESTED = "nested"
    LIST_NESTED = "list_nested"
    LITERAL = "literal"
    FREE = "free"


@dataclass
class FieldSpec:
    """One field's role within an output template."""

    name: str
    path: str
    role: FieldRole
    target_type: type | None = None
    literal_values: tuple[Any, ...] | None = None
    nested_spec: "OutputSpec | None" = None


@dataclass
class OutputSpec:
    """All fields of a single Pydantic model with their roles."""

    model: type
    fields: dict[str, FieldSpec] = field(default_factory=dict)

    @property
    def referenced_entity_types(self) -> set[type]:
        out: set[type] = set()
        for f in self.fields.values():
            if f.role in (FieldRole.REF, FieldRole.LIST_REF, FieldRole.OPTIONAL_REF):
                if f.target_type is not None:
                    out.add(f.target_type)
            elif f.role in (FieldRole.NESTED, FieldRole.LIST_NESTED) and f.nested_spec:
                out |= f.nested_spec.referenced_entity_types
        return out
```

`src/pydantic_ai_stateflow/grounded/_scan.py`:

```python
from __future__ import annotations

from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded._spec import FieldRole, FieldSpec, OutputSpec
from pydantic_ai_stateflow.grounded.ref import Ref


def scan_output(model: type[BaseModel], path: str = "") -> OutputSpec:
    """Walk Pydantic model fields and classify each by its role.

    Recurses into nested BaseModel and list[BaseModel] fields. Stops at
    primitive / unrecognised fields (FieldRole.FREE).
    """
    spec = OutputSpec(model=model)
    for name, info in model.model_fields.items():
        full_path = f"{path}.{name}" if path else name
        spec.fields[name] = _classify(name, full_path, info.annotation)
    return spec


def _classify(name: str, path: str, annotation: Any) -> FieldSpec:
    # Direct Ref[X]
    if _is_ref_type(annotation):
        return FieldSpec(name=name, path=path, role=FieldRole.REF, target_type=_ref_target(annotation))

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[Ref[X]] / Union[Ref[X], None]
    if origin in (Union, UnionType):
        non_none = [a for a in args if a is not NoneType and a is not type(None)]
        if len(non_none) == 1 and _is_ref_type(non_none[0]):
            return FieldSpec(
                name=name, path=path, role=FieldRole.OPTIONAL_REF, target_type=_ref_target(non_none[0])
            )

    # list[Ref[X]] / list[BaseModel] / list[primitive]
    if origin is list and args:
        inner = args[0]
        if _is_ref_type(inner):
            return FieldSpec(name=name, path=path, role=FieldRole.LIST_REF, target_type=_ref_target(inner))
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return FieldSpec(
                name=name, path=path, role=FieldRole.LIST_NESTED,
                target_type=inner, nested_spec=scan_output(inner, path=f"{path}[*]"),
            )

    # Literal[...]
    if origin is Literal:
        return FieldSpec(name=name, path=path, role=FieldRole.LITERAL, literal_values=args)

    # Nested BaseModel
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return FieldSpec(
            name=name, path=path, role=FieldRole.NESTED,
            target_type=annotation, nested_spec=scan_output(annotation, path=path),
        )

    # Free (primitive / unrecognised — leave as-is)
    return FieldSpec(name=name, path=path, role=FieldRole.FREE)


def _is_ref_type(annotation: Any) -> bool:
    """True iff annotation is `Ref[SomeEntity]` (subscripted form)."""
    return isinstance(annotation, type) and issubclass(annotation, Ref) and annotation is not Ref


def _ref_target(annotation: Any) -> type[BaseModel]:
    target = getattr(annotation, "__entity_type__", None)
    if target is None:
        raise TypeError(f"Subscripted Ref expected, got {annotation!r}")
    return target
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_resolver_scan_output.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/_spec.py src/pydantic_ai_stateflow/grounded/_scan.py tests/grounded/test_resolver_scan_output.py
git commit -m "feat(grounded): scan_output classifies field roles (REF/LIST_REF/OPTIONAL_REF/NESTED/LIST_NESTED/LITERAL/FREE)"
```

---

## Task 10: `GroundedResolver._scan_context` — collect entity instances

**Files:**
- Modify: `src/pydantic_ai_stateflow/grounded/_scan.py`
- Create: `tests/grounded/test_resolver_scan_context.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_resolver_scan_context.py`:

```python
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output


class Item(BaseModel):
    id: UUID
    name: str


class Customer(BaseModel):
    id: UUID
    email: str


class Status(BaseModel):
    code: Literal["a", "b", "c"]


def test_scan_collects_list_of_entities():
    class Ctx(BaseModel):
        items: list[Item]
        notes: str

    class Out(BaseModel):
        chosen_id: Item

    item_ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=item_ids[0], name="a"), Item(id=item_ids[1], name="b")], notes="x")
    out_spec = scan_output(Out)
    sources = scan_context(ctx, out_spec)
    assert sorted(sources.by_entity_type[Item]) == sorted(item_ids)


def test_scan_collects_singleton_entity():
    class Ctx(BaseModel):
        customer: Customer

    class Out(BaseModel):
        ref: Customer

    cust_id = uuid4()
    ctx = Ctx(customer=Customer(id=cust_id, email="x@y.z"))
    out_spec = scan_output(Out)
    sources = scan_context(ctx, out_spec)
    assert sources.by_entity_type[Customer] == [cust_id]


def test_scan_collects_from_nested_pydantic():
    class Holder(BaseModel):
        items: list[Item]

    class Ctx(BaseModel):
        holder: Holder

    class Out(BaseModel):
        ref: Item

    ids = [uuid4(), uuid4(), uuid4()]
    ctx = Ctx(holder=Holder(items=[Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)]))
    sources = scan_context(ctx, scan_output(Out))
    assert sorted(sources.by_entity_type[Item]) == sorted(ids)


def test_scan_returns_empty_for_unreferenced_types():
    class Ctx(BaseModel):
        items: list[Item]

    class Out(BaseModel):
        unrelated: str          # no Ref to Item — no collection requested

    sources = scan_context(Ctx(items=[Item(id=uuid4(), name="a")]), scan_output(Out))
    assert Item not in sources.by_entity_type


def test_scan_unions_multiple_sources_of_same_type():
    class Ctx(BaseModel):
        top: list[Item]
        fallback: list[Item]

    class Out(BaseModel):
        ref: Item

    top_ids = [uuid4()]
    fb_ids = [uuid4(), uuid4()]
    ctx = Ctx(
        top=[Item(id=top_ids[0], name="t")],
        fallback=[Item(id=fb_ids[0], name="f1"), Item(id=fb_ids[1], name="f2")],
    )
    sources = scan_context(ctx, scan_output(Out))
    assert sorted(sources.by_entity_type[Item]) == sorted(top_ids + fb_ids)
```

Note: tests refer to `Out(BaseModel): ref: Item` (not `Ref[Item]`) for scan-context purposes — scan_context only needs to know which entity TYPES the resolver cares about, which we extract from the output spec's `referenced_entity_types`. But scan_output classifies bare `Item` as `NESTED`, not REF. We need to make `referenced_entity_types` include nested-model types too OR adjust the tests to use `Ref[Item]`. Adjusting tests is cleaner — update the `Out` classes to use `Ref[Item]`:

Replace `chosen_id: Item` with `chosen_id: Ref[Item]`, `ref: Customer` with `ref: Ref[Customer]`, `ref: Item` with `ref: Ref[Item]`. Re-paste:

```python
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output


class Item(BaseModel):
    id: UUID
    name: str


class Customer(BaseModel):
    id: UUID
    email: str


def test_scan_collects_list_of_entities():
    class Ctx(BaseModel):
        items: list[Item]
        notes: str

    class Out(BaseModel):
        chosen: Ref[Item]

    item_ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=item_ids[0], name="a"), Item(id=item_ids[1], name="b")], notes="x")
    sources = scan_context(ctx, scan_output(Out))
    assert sorted(sources.by_entity_type[Item]) == sorted(item_ids)


def test_scan_collects_singleton_entity():
    class Ctx(BaseModel):
        customer: Customer

    class Out(BaseModel):
        ref: Ref[Customer]

    cust_id = uuid4()
    sources = scan_context(Ctx(customer=Customer(id=cust_id, email="x@y.z")), scan_output(Out))
    assert sources.by_entity_type[Customer] == [cust_id]


def test_scan_collects_from_nested_pydantic():
    class Holder(BaseModel):
        items: list[Item]

    class Ctx(BaseModel):
        holder: Holder

    class Out(BaseModel):
        ref: Ref[Item]

    ids = [uuid4(), uuid4(), uuid4()]
    ctx = Ctx(holder=Holder(items=[Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)]))
    sources = scan_context(ctx, scan_output(Out))
    assert sorted(sources.by_entity_type[Item]) == sorted(ids)


def test_scan_returns_empty_for_unreferenced_types():
    class Ctx(BaseModel):
        items: list[Item]

    class Out(BaseModel):
        unrelated: str

    sources = scan_context(Ctx(items=[Item(id=uuid4(), name="a")]), scan_output(Out))
    assert Item not in sources.by_entity_type


def test_scan_unions_multiple_sources_of_same_type():
    class Ctx(BaseModel):
        top: list[Item]
        fallback: list[Item]

    class Out(BaseModel):
        ref: Ref[Item]

    top_ids = [uuid4()]
    fb_ids = [uuid4(), uuid4()]
    ctx = Ctx(
        top=[Item(id=top_ids[0], name="t")],
        fallback=[Item(id=fb_ids[0], name="f1"), Item(id=fb_ids[1], name="f2")],
    )
    sources = scan_context(ctx, scan_output(Out))
    assert sorted(sources.by_entity_type[Item]) == sorted(top_ids + fb_ids)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_resolver_scan_context.py -v
```

Expected: `ImportError: cannot import name 'scan_context'`.

- [ ] **Step 3: Implement `scan_context` + `ContextSources`**

In `src/pydantic_ai_stateflow/grounded/_spec.py`, add at the bottom:

```python
@dataclass
class ContextSources:
    by_entity_type: dict[type, list[Any]] = field(default_factory=dict)
    by_enum_type: dict[type, set[Any]] = field(default_factory=dict)
```

In `src/pydantic_ai_stateflow/grounded/_scan.py`, add:

```python
from pydantic_ai_stateflow.grounded._spec import ContextSources


def scan_context(context: BaseModel, output_spec: OutputSpec, *, max_depth: int = 5) -> ContextSources:
    """Walk a Pydantic context, collect instances of types referenced by output_spec.

    Returns a `ContextSources` with `by_entity_type[T]` mapping to all `t.id`
    values for each `T` instance encountered, recursively.
    """
    sources = ContextSources()
    targets = output_spec.referenced_entity_types
    if not targets:
        return sources

    _walk(context, targets, sources, depth=0, max_depth=max_depth)
    return sources


def _walk(obj: Any, targets: set[type], sources: ContextSources, depth: int, max_depth: int) -> None:
    if depth > max_depth:
        return
    if isinstance(obj, BaseModel):
        if type(obj) in targets:
            id_val = getattr(obj, "id", None)
            if id_val is not None:
                sources.by_entity_type.setdefault(type(obj), []).append(id_val)
        for field_name in type(obj).model_fields:
            _walk(getattr(obj, field_name), targets, sources, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            _walk(item, targets, sources, depth + 1, max_depth)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, targets, sources, depth + 1, max_depth)
    # primitives — stop
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_resolver_scan_context.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/_spec.py src/pydantic_ai_stateflow/grounded/_scan.py tests/grounded/test_resolver_scan_context.py
git commit -m "feat(grounded): scan_context collects entity instances recursively"
```

---

## Task 11: `_build_dynamic` for simple `Ref` field

**Files:**
- Create: `src/pydantic_ai_stateflow/grounded/_build.py`
- Create: `src/pydantic_ai_stateflow/grounded/errors.py`
- Create: `tests/grounded/test_resolver_build_simple_ref.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_resolver_build_simple_ref.py`:

```python
from typing import Literal, get_args, get_origin
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


class Item(BaseModel):
    id: UUID
    name: str


def test_build_replaces_ref_with_literal_of_ids():
    class Out(BaseModel):
        chosen: Ref[Item]
        rationale: str

    ids = [uuid4(), uuid4()]
    ctx_items = [Item(id=ids[0], name="a"), Item(id=ids[1], name="b")]

    class Ctx(BaseModel):
        items: list[Item]

    ctx = Ctx(items=ctx_items)
    out_spec = scan_output(Out)
    sources = scan_context(ctx, out_spec)
    Dynamic = build_dynamic(Out, out_spec, sources)

    chosen_field = Dynamic.model_fields["chosen"]
    # The dynamic annotation should be a Literal of UUIDs equal to context.
    assert get_origin(chosen_field.annotation) is Literal
    assert set(get_args(chosen_field.annotation)) == set(ids)


def test_build_validation_passes_for_allowed_value():
    class Out(BaseModel):
        chosen: Ref[Item]

    ids = [uuid4(), uuid4()]
    ctx = type("Ctx", (BaseModel,), {"__annotations__": {"items": list[Item]}})(
        items=[Item(id=ids[0], name="a"), Item(id=ids[1], name="b")]
    )
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))

    obj = Dynamic.model_validate({"chosen": str(ids[0])})
    assert obj.chosen == ids[0]


def test_build_validation_rejects_unknown_value():
    class Out(BaseModel):
        chosen: Ref[Item]

    ids = [uuid4()]
    ctx_data = type("Ctx", (BaseModel,), {"__annotations__": {"items": list[Item]}})(
        items=[Item(id=ids[0], name="a")]
    )
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx_data, scan_output(Out)))

    with pytest.raises(ValidationError):
        Dynamic.model_validate({"chosen": str(uuid4())})


def test_build_raises_when_no_entities_in_context():
    class Out(BaseModel):
        chosen: Ref[Item]

    class Ctx(BaseModel):
        unrelated: str

    ctx = Ctx(unrelated="x")
    with pytest.raises(GroundedBuildError, match="No instances of Item"):
        build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_resolver_build_simple_ref.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `errors.py` + `_build.py`**

`src/pydantic_ai_stateflow/grounded/errors.py`:

```python
class GroundedError(Exception):
    """Base for all grounded-schema errors."""


class GroundedBuildError(GroundedError):
    """Raised at .run() time when the dynamic output model cannot be built
    (e.g., no entity instances in context for a required Ref field)."""


class GroundedHydrationError(GroundedError):
    """Raised when hydration cannot resolve a Ref via the given repos."""
```

`src/pydantic_ai_stateflow/grounded/_build.py`:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, create_model

from pydantic_ai_stateflow.grounded._spec import ContextSources, FieldRole, OutputSpec
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


def build_dynamic(
    model: type[BaseModel],
    spec: OutputSpec,
    sources: ContextSources,
) -> type[BaseModel]:
    """Build a dynamic Pydantic model where Ref/Enum fields become Literals.

    Recursive: nested BaseModel fields are themselves rebuilt with dynamic
    Literals. Existing (non-grounded) fields are passed through unchanged.
    """
    fields: dict[str, Any] = {}
    for name, fspec in spec.fields.items():
        field_info = model.model_fields[name]
        match fspec.role:
            case FieldRole.REF:
                ids = sources.by_entity_type.get(fspec.target_type, [])
                if not ids:
                    raise GroundedBuildError(
                        f"No instances of {fspec.target_type.__name__} in context "
                        f"for {fspec.path}"
                    )
                fields[name] = (Literal[tuple(ids)], field_info)

            case FieldRole.FREE:
                fields[name] = (field_info.annotation, field_info)

            case _:
                # Other roles handled in subsequent tasks; for now passthrough
                fields[name] = (field_info.annotation, field_info)

    return create_model(f"Dynamic{model.__name__}", __base__=BaseModel, **fields)
```

Modify `src/pydantic_ai_stateflow/grounded/__init__.py` to export errors:

```python
from pydantic_ai_stateflow.grounded.errors import (
    GroundedBuildError,
    GroundedError,
    GroundedHydrationError,
)
from pydantic_ai_stateflow.grounded.ref import Ref

__all__ = [
    "Ref",
    "GroundedError",
    "GroundedBuildError",
    "GroundedHydrationError",
]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_resolver_build_simple_ref.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/_build.py src/pydantic_ai_stateflow/grounded/errors.py src/pydantic_ai_stateflow/grounded/__init__.py tests/grounded/test_resolver_build_simple_ref.py
git commit -m "feat(grounded): build_dynamic replaces Ref[T] with Literal[*ids] for simple case"
```

---

## Task 12: `_build_dynamic` for `list[Ref]` and `Optional[Ref]`

**Files:**
- Modify: `src/pydantic_ai_stateflow/grounded/_build.py`
- Create: `tests/grounded/test_resolver_build_collections.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_resolver_build_collections.py`:

```python
from typing import Literal, Optional, get_args, get_origin
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output


class Item(BaseModel):
    id: UUID
    name: str


def _build(out_cls: type[BaseModel], ctx_items: list[Item]) -> type[BaseModel]:
    class Ctx(BaseModel):
        items: list[Item]

    ctx = Ctx(items=ctx_items)
    return build_dynamic(out_cls, scan_output(out_cls), scan_context(ctx, scan_output(out_cls)))


def test_list_ref_becomes_list_of_literal():
    class Out(BaseModel):
        chosen: list[Ref[Item]]

    ids = [uuid4(), uuid4(), uuid4()]
    items = [Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)]
    Dynamic = _build(Out, items)

    ann = Dynamic.model_fields["chosen"].annotation
    assert get_origin(ann) is list
    inner = get_args(ann)[0]
    assert get_origin(inner) is Literal
    assert set(get_args(inner)) == set(ids)


def test_list_ref_validation_passes_for_subset():
    class Out(BaseModel):
        chosen: list[Ref[Item]]

    ids = [uuid4(), uuid4(), uuid4()]
    items = [Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)]
    Dynamic = _build(Out, items)

    obj = Dynamic.model_validate({"chosen": [str(ids[0]), str(ids[2])]})
    assert obj.chosen == [ids[0], ids[2]]


def test_list_ref_validation_rejects_unknown():
    class Out(BaseModel):
        chosen: list[Ref[Item]]

    ids = [uuid4()]
    Dynamic = _build(Out, [Item(id=ids[0], name="a")])

    with pytest.raises(ValidationError):
        Dynamic.model_validate({"chosen": [str(uuid4())]})


def test_optional_ref_becomes_optional_literal():
    class Out(BaseModel):
        maybe: Optional[Ref[Item]] = None  # noqa: UP007 — explicit Optional for test

    ids = [uuid4()]
    Dynamic = _build(Out, [Item(id=ids[0], name="a")])

    # None must validate
    obj_none = Dynamic.model_validate({"maybe": None})
    assert obj_none.maybe is None
    # Valid id must validate
    obj = Dynamic.model_validate({"maybe": str(ids[0])})
    assert obj.maybe == ids[0]
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_resolver_build_collections.py -v
```

Expected: most fail because `_build.py` currently passes through non-REF roles unchanged.

- [ ] **Step 3: Extend `_build.py` to handle `LIST_REF` and `OPTIONAL_REF`**

In `src/pydantic_ai_stateflow/grounded/_build.py`, replace the `match` block in `build_dynamic` with:

```python
        match fspec.role:
            case FieldRole.REF:
                ids = sources.by_entity_type.get(fspec.target_type, [])
                if not ids:
                    raise GroundedBuildError(
                        f"No instances of {fspec.target_type.__name__} in context "
                        f"for {fspec.path}"
                    )
                fields[name] = (Literal[tuple(ids)], field_info)

            case FieldRole.LIST_REF:
                ids = sources.by_entity_type.get(fspec.target_type, [])
                if not ids:
                    raise GroundedBuildError(
                        f"No instances of {fspec.target_type.__name__} in context "
                        f"for {fspec.path} (list[Ref])"
                    )
                fields[name] = (list[Literal[tuple(ids)]], field_info)

            case FieldRole.OPTIONAL_REF:
                ids = sources.by_entity_type.get(fspec.target_type, [])
                if not ids:
                    # Optional with no instances → only None is valid
                    fields[name] = (None | type(None), field_info)
                else:
                    fields[name] = (Literal[tuple(ids)] | None, field_info)

            case FieldRole.FREE:
                fields[name] = (field_info.annotation, field_info)

            case _:
                fields[name] = (field_info.annotation, field_info)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_resolver_build_collections.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/_build.py tests/grounded/test_resolver_build_collections.py
git commit -m "feat(grounded): build_dynamic supports list[Ref] and Optional[Ref]"
```

---

## Task 13: `_build_dynamic` recurses into nested models

**Files:**
- Modify: `src/pydantic_ai_stateflow/grounded/_build.py`
- Create: `tests/grounded/test_resolver_build_nested.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_resolver_build_nested.py`:

```python
from typing import Literal, get_args, get_origin
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output


class Item(BaseModel):
    id: UUID
    name: str


class Customer(BaseModel):
    id: UUID
    email: str


def test_nested_basemodel_with_ref_field_recursively_built():
    class Inner(BaseModel):
        chosen: Ref[Item]
        rationale: str

    class Out(BaseModel):
        inner: Inner

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a"), Item(id=ids[1], name="b")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))

    # Round-trip a valid value
    obj = Dynamic.model_validate({"inner": {"chosen": str(ids[0]), "rationale": "r"}})
    assert obj.inner.chosen == ids[0]
    # Invalid id rejected
    with pytest.raises(ValidationError):
        Dynamic.model_validate({"inner": {"chosen": str(uuid4()), "rationale": "r"}})


def test_list_of_nested_models_recurses_and_broadcasts():
    class Inner(BaseModel):
        chosen: Ref[Item]
        score: int

    class Out(BaseModel):
        items: list[Inner]

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a"), Item(id=ids[1], name="b")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))

    obj = Dynamic.model_validate({"items": [
        {"chosen": str(ids[0]), "score": 1},
        {"chosen": str(ids[1]), "score": 2},
    ]})
    assert obj.items[0].chosen == ids[0]
    assert obj.items[1].chosen == ids[1]


def test_deeply_nested_refs_all_validate():
    class Inner(BaseModel):
        chosen: Ref[Item]

    class Mid(BaseModel):
        ins: list[Inner]

    class Out(BaseModel):
        mid: Mid

    class Ctx(BaseModel):
        items: list[Item]

    ids = [uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))

    obj = Dynamic.model_validate({"mid": {"ins": [{"chosen": str(ids[0])}]}})
    assert obj.mid.ins[0].chosen == ids[0]
    with pytest.raises(ValidationError):
        Dynamic.model_validate({"mid": {"ins": [{"chosen": str(uuid4())}]}})
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_resolver_build_nested.py -v
```

Expected: nested model fields pass through unchanged → inner Ref still rejects valid context IDs.

- [ ] **Step 3: Extend `_build.py` to recurse on NESTED and LIST_NESTED**

In the `match` block of `build_dynamic`, add cases:

```python
            case FieldRole.NESTED:
                inner_dynamic = build_dynamic(fspec.target_type, fspec.nested_spec, sources)
                fields[name] = (inner_dynamic, field_info)

            case FieldRole.LIST_NESTED:
                inner_dynamic = build_dynamic(fspec.target_type, fspec.nested_spec, sources)
                fields[name] = (list[inner_dynamic], field_info)
```

(Place them between `OPTIONAL_REF` and `FREE`.)

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_resolver_build_nested.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/_build.py tests/grounded/test_resolver_build_nested.py
git commit -m "feat(grounded): build_dynamic recurses into nested models and list-of-nested"
```

---

## Task 14: `_build_dynamic` for `Literal` field — intersection with context

**Files:**
- Modify: `src/pydantic_ai_stateflow/grounded/_scan.py`
- Modify: `src/pydantic_ai_stateflow/grounded/_build.py`
- Create: `tests/grounded/test_resolver_build_enums.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_resolver_build_enums.py`:

```python
from typing import Literal, get_args, get_origin
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output


class Order(BaseModel):
    id: UUID
    status: Literal["draft", "ready", "sent", "cancelled"]


def test_literal_in_output_intersects_with_context_values():
    class Out(BaseModel):
        new_status: Literal["draft", "ready", "sent", "cancelled"]

    class Ctx(BaseModel):
        orders: list[Order]

    ctx = Ctx(orders=[
        Order(id=uuid4(), status="ready"),
        Order(id=uuid4(), status="sent"),
    ])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))
    # Only values actually present in context become allowed
    assert set(get_args(Dynamic.model_fields["new_status"].annotation)) == {"ready", "sent"}


def test_literal_without_context_remains_unrestricted():
    class Out(BaseModel):
        new_status: Literal["draft", "ready", "sent", "cancelled"]

    class Ctx(BaseModel):
        unrelated: str

    ctx = Ctx(unrelated="x")
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))
    # No intersection possible — fall back to original Literal
    assert set(get_args(Dynamic.model_fields["new_status"].annotation)) == {
        "draft", "ready", "sent", "cancelled"
    }


def test_literal_validation_rejects_unintersected_value():
    class Out(BaseModel):
        new_status: Literal["draft", "ready", "sent", "cancelled"]

    class Ctx(BaseModel):
        orders: list[Order]

    ctx = Ctx(orders=[Order(id=uuid4(), status="ready")])
    Dynamic = build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))

    with pytest.raises(ValidationError):
        Dynamic.model_validate({"new_status": "draft"})
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_resolver_build_enums.py -v
```

Expected: scan_context doesn't collect Literal values yet, so intersection never happens.

- [ ] **Step 3: Extend `scan_context` to collect Literal values**

In `src/pydantic_ai_stateflow/grounded/_scan.py`, modify `_walk` so that BaseModel walks also collect Literal-typed field values into `sources.by_literal_values`:

First, extend `ContextSources` in `_spec.py`:

```python
@dataclass
class ContextSources:
    by_entity_type: dict[type, list[Any]] = field(default_factory=dict)
    by_literal_values: dict[str, set[Any]] = field(default_factory=dict)
    # Maps Literal-args-tuple-as-string-key -> observed values intersection.

    @staticmethod
    def literal_key(args: tuple[Any, ...]) -> str:
        return "|".join(sorted(str(a) for a in args))
```

In `_scan.py`, modify `_walk` — when traversing a BaseModel, for each field whose annotation is `Literal[...]`, record observed value:

```python
def _walk(obj: Any, targets: set[type], sources: ContextSources, depth: int, max_depth: int) -> None:
    if depth > max_depth:
        return
    if isinstance(obj, BaseModel):
        if type(obj) in targets:
            id_val = getattr(obj, "id", None)
            if id_val is not None:
                sources.by_entity_type.setdefault(type(obj), []).append(id_val)

        for field_name, info in type(obj).model_fields.items():
            value = getattr(obj, field_name)
            # Capture Literal fields for enum intersection
            if get_origin(info.annotation) is Literal:
                key = ContextSources.literal_key(get_args(info.annotation))
                sources.by_literal_values.setdefault(key, set()).add(value)
            _walk(value, targets, sources, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            _walk(item, targets, sources, depth + 1, max_depth)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, targets, sources, depth + 1, max_depth)
```

Also we must call `scan_context` so it walks unconditionally now (not gated on targets being non-empty):

```python
def scan_context(context: BaseModel, output_spec: OutputSpec, *, max_depth: int = 5) -> ContextSources:
    sources = ContextSources()
    targets = output_spec.referenced_entity_types
    _walk(context, targets, sources, depth=0, max_depth=max_depth)
    return sources
```

(Drop the early-return on empty `targets` since Literal scanning is independent.)

- [ ] **Step 4: Implement LITERAL case in `_build.py`**

In `build_dynamic` `match` block, add before FREE:

```python
            case FieldRole.LITERAL:
                allowed = fspec.literal_values or ()
                key = ContextSources.literal_key(allowed)
                observed = sources.by_literal_values.get(key)
                if observed:
                    intersected = tuple(v for v in allowed if v in observed)
                    if not intersected:
                        # Defensive: if intersection is empty, fall back to original.
                        fields[name] = (Literal[allowed], field_info)
                    else:
                        fields[name] = (Literal[intersected], field_info)
                else:
                    fields[name] = (Literal[allowed], field_info)
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_resolver_build_enums.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 7: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/_spec.py src/pydantic_ai_stateflow/grounded/_scan.py src/pydantic_ai_stateflow/grounded/_build.py tests/grounded/test_resolver_build_enums.py
git commit -m "feat(grounded): Literal fields intersect with context-observed values"
```

---

## Task 15: Construction-time errors and warnings

**Files:**
- Modify: `src/pydantic_ai_stateflow/grounded/_build.py`
- Create: `tests/grounded/test_resolver_errors.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_resolver_errors.py`:

```python
import warnings
from typing import Optional
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


class Item(BaseModel):
    id: UUID
    name: str


def test_no_instances_raises_with_helpful_message():
    class Out(BaseModel):
        chosen: Ref[Item]

    class Ctx(BaseModel):
        unrelated: str

    ctx = Ctx(unrelated="x")
    with pytest.raises(GroundedBuildError) as exc_info:
        build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))
    assert "Item" in str(exc_info.value)
    assert "context" in str(exc_info.value).lower()


def test_optional_ref_with_no_instances_only_allows_none():
    class Out(BaseModel):
        maybe: Optional[Ref[Item]] = None  # noqa: UP007

    class Ctx(BaseModel):
        unrelated: str

    Dynamic = build_dynamic(
        Out, scan_output(Out), scan_context(Ctx(unrelated="x"), scan_output(Out))
    )
    # None must validate; any UUID must fail
    obj = Dynamic.model_validate({"maybe": None})
    assert obj.maybe is None


def test_large_allowed_set_emits_warning(recwarn):
    class Out(BaseModel):
        chosen: Ref[Item]

    class Ctx(BaseModel):
        items: list[Item]

    items = [Item(id=uuid4(), name=f"n{i}") for i in range(1500)]
    ctx = Ctx(items=items)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        build_dynamic(Out, scan_output(Out), scan_context(ctx, scan_output(Out)))

    matched = [w for w in recwarn.list if "SemanticRouter" in str(w.message)]
    assert len(matched) >= 1
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_resolver_errors.py -v
```

Expected: large-set warning test fails (no warning emitted yet); other tests may already pass.

- [ ] **Step 3: Add large-set warning**

In `src/pydantic_ai_stateflow/grounded/_build.py`, in the `REF` and `LIST_REF` cases, after `ids = sources.by_entity_type.get(...)`:

```python
                if len(ids) > 1000:
                    import warnings
                    warnings.warn(
                        f"Grounded field {fspec.path} has {len(ids)} allowed IDs; "
                        "consider SemanticRouter pattern for large closed sets.",
                        stacklevel=3,
                    )
```

(Place in both REF and LIST_REF after the empty-check and before assigning `fields[name]`.)

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_resolver_errors.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/_build.py tests/grounded/test_resolver_errors.py
git commit -m "feat(grounded): construction-time errors and large-set warnings"
```

---

## Task 16: Escape-hatch `constraints={...}` in resolver

**Files:**
- Create: `src/pydantic_ai_stateflow/grounded/resolver.py`
- Create: `tests/grounded/test_resolver_constraints_override.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_resolver_constraints_override.py`:

```python
from typing import get_args
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded.resolver import GroundedResolver
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


class Item(BaseModel):
    id: UUID
    name: str


class Ctx(BaseModel):
    items: list[Item]


class Out(BaseModel):
    chosen: Ref[Item]
    rationale: str


def test_constraints_override_restricts_to_subset():
    ids = [uuid4(), uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)])
    resolver = GroundedResolver(Out)

    # Override: only first ID allowed
    Dynamic, _ = resolver.build(ctx, constraints={"chosen": [ids[0]]})

    # Only ids[0] passes validation
    Dynamic.model_validate({"chosen": str(ids[0]), "rationale": "r"})
    with pytest.raises(ValidationError):
        Dynamic.model_validate({"chosen": str(ids[1]), "rationale": "r"})


def test_constraints_override_with_unknown_path_errors():
    ids = [uuid4()]
    ctx = Ctx(items=[Item(id=ids[0], name="a")])
    resolver = GroundedResolver(Out)

    with pytest.raises(GroundedBuildError, match="unknown path"):
        resolver.build(ctx, constraints={"nonexistent_field": [ids[0]]})


def test_constraints_override_singleton_value():
    ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)])
    resolver = GroundedResolver(Out)

    # Constraint is a scalar (single value rather than a list)
    Dynamic, _ = resolver.build(ctx, constraints={"chosen": ids[0]})
    Dynamic.model_validate({"chosen": str(ids[0]), "rationale": "r"})
    with pytest.raises(ValidationError):
        Dynamic.model_validate({"chosen": str(ids[1]), "rationale": "r"})
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_resolver_constraints_override.py -v
```

Expected: ImportError on `resolver.GroundedResolver`.

- [ ] **Step 3: Implement `resolver.py` with `constraints` override**

`src/pydantic_ai_stateflow/grounded/resolver.py`:

```python
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded._build import build_dynamic
from pydantic_ai_stateflow.grounded._scan import scan_context, scan_output
from pydantic_ai_stateflow.grounded._spec import ContextSources, FieldRole, OutputSpec
from pydantic_ai_stateflow.grounded.errors import GroundedBuildError


class GroundedResolver:
    """Per-Pattern scanner + dynamic-model builder.

    Build is per-call (context varies). The output type's spec is cached
    once at construction.
    """

    def __init__(self, output_type: type[BaseModel]) -> None:
        self.output_type = output_type
        self._spec: OutputSpec = scan_output(output_type)

    def build(
        self,
        context: BaseModel,
        constraints: dict[str, Any] | None = None,
    ) -> tuple[type[BaseModel], OutputSpec]:
        """Return (DynamicModel, OutputSpec). OutputSpec used by HydrationMap (Task 18)."""
        sources = scan_context(context, self._spec)
        if constraints:
            sources = self._apply_constraints(sources, constraints)
        dynamic = build_dynamic(self.output_type, self._spec, sources)
        return dynamic, self._spec

    def _apply_constraints(self, sources: ContextSources, constraints: dict[str, Any]) -> ContextSources:
        for path, value in constraints.items():
            fspec = self._find_field_by_path(path)
            if fspec is None:
                raise GroundedBuildError(f"constraints['{path}']: unknown path in output type")
            if fspec.role not in (FieldRole.REF, FieldRole.LIST_REF, FieldRole.OPTIONAL_REF):
                raise GroundedBuildError(
                    f"constraints['{path}']: path role {fspec.role} not overridable in v1"
                )
            values = value if isinstance(value, list) else [value]
            # Coerce strings to UUID for convenience
            coerced = [UUID(v) if isinstance(v, str) else v for v in values]
            sources.by_entity_type[fspec.target_type] = coerced
        return sources

    def _find_field_by_path(self, path: str) -> Any:
        # v1: only top-level paths (no dotted nesting / no [*] glob).
        # Nested-path support deferred to Sub-project #2 if needed.
        if "." in path or "[" in path:
            raise GroundedBuildError(
                f"constraints['{path}']: nested / list paths not supported in v1 "
                "(only top-level field names)"
            )
        return self._spec.fields.get(path)
```

Update `src/pydantic_ai_stateflow/grounded/__init__.py`:

```python
from pydantic_ai_stateflow.grounded.errors import (
    GroundedBuildError,
    GroundedError,
    GroundedHydrationError,
)
from pydantic_ai_stateflow.grounded.ref import Ref
from pydantic_ai_stateflow.grounded.resolver import GroundedResolver

__all__ = [
    "Ref",
    "GroundedResolver",
    "GroundedError",
    "GroundedBuildError",
    "GroundedHydrationError",
]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_resolver_constraints_override.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/resolver.py src/pydantic_ai_stateflow/grounded/__init__.py tests/grounded/test_resolver_constraints_override.py
git commit -m "feat(grounded): GroundedResolver with escape-hatch constraints override"
```

---

## Task 17: `GroundedAgent` wrapper + `GroundedResult`

**Files:**
- Create: `src/pydantic_ai_stateflow/grounded/agent.py`
- Create: `tests/grounded/test_grounded_agent.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_grounded_agent.py`:

```python
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from pydantic_ai_stateflow.grounded import GroundedAgent, Ref


class Item(BaseModel):
    id: UUID
    name: str


class Ctx(BaseModel):
    items: list[Item]


class Decision(BaseModel):
    chosen: Ref[Item]
    rationale: str


def make_function_model_returning_id(item_id: UUID) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Force the agent to return our chosen id via the final_result tool.
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result",
            args={"chosen": str(item_id), "rationale": "always-first"},
        )])
    return FunctionModel(fn)


@pytest.mark.asyncio
async def test_grounded_agent_run_returns_valid_decision():
    item_ids = [uuid4(), uuid4()]
    ctx = Ctx(items=[Item(id=item_ids[0], name="a"), Item(id=item_ids[1], name="b")])
    base_agent: Agent[None, Decision] = Agent(
        model=make_function_model_returning_id(item_ids[0]),
        output_type=Decision,
    )

    grounded = GroundedAgent(base_agent, output_type=Decision)
    result = await grounded.run(ctx, instructions="pick best")

    assert isinstance(result.value.chosen, Ref)
    assert result.value.chosen.id == item_ids[0]
    assert result.value.rationale == "always-first"


@pytest.mark.asyncio
async def test_grounded_agent_blocks_hallucinated_id():
    """If the function-model tries to return an id not in context,
    Pydantic validation must reject it (Literal violation), which
    causes the run to raise."""
    item_ids = [uuid4()]
    ctx = Ctx(items=[Item(id=item_ids[0], name="a")])
    hallucinated = uuid4()
    base_agent: Agent[None, Decision] = Agent(
        model=make_function_model_returning_id(hallucinated),
        output_type=Decision,
    )

    grounded = GroundedAgent(base_agent, output_type=Decision)
    with pytest.raises(Exception):  # pydantic_ai wraps as UnexpectedModelBehavior or similar
        await grounded.run(ctx, instructions="pick best")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_grounded_agent.py -v
```

Expected: ImportError on `GroundedAgent`.

- [ ] **Step 3: Implement `GroundedAgent` + `GroundedResult`**

`src/pydantic_ai_stateflow/grounded/agent.py`:

```python
from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult

from pydantic_ai_stateflow.grounded.resolver import GroundedResolver
from pydantic_ai_stateflow.grounded._spec import OutputSpec

OutT = TypeVar("OutT", bound=BaseModel)


class GroundedResult(BaseModel, Generic[OutT]):
    """Run result, typed as the original OutT for IDE / mypy users.

    Note: the actual model instance is a Dynamic<OutT> with Literal-narrowed
    fields, but every public usage treats it as OutT. Hydration uses the
    underlying OutputSpec to walk Ref fields.
    """
    model_config = {"arbitrary_types_allowed": True}

    value: Any                                       # typed as OutT externally; runtime is DynamicOutT
    raw: Any                                         # AgentRunResult
    _spec: OutputSpec                                # internal; used by HydrationMap (Task 18)


class GroundedAgent(Generic[OutT]):
    """Wrapper that builds a per-call dynamic output type and delegates to agent.run."""

    def __init__(self, agent: Agent[Any, OutT], *, output_type: type[OutT]) -> None:
        self.agent = agent
        self.output_type = output_type
        self._resolver = GroundedResolver(output_type)

    async def run(
        self,
        context: BaseModel,
        *,
        instructions: str | None = None,
        constraints: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> GroundedResult[OutT]:
        dynamic_output, spec = self._resolver.build(context, constraints=constraints)
        # Use Agent.run with the dynamic output type via .override
        with self.agent.override(output_type=dynamic_output):
            user_prompt = instructions or "Produce output matching the schema."
            run_result: AgentRunResult[Any] = await self.agent.run(user_prompt, **agent_kwargs)
        return GroundedResult(value=run_result.output, raw=run_result, _spec=spec)
```

> **Note on `agent.override(output_type=...)`:** pydantic-ai's `agent.override` context manager supports overriding the output type per-call. If a version doesn't support this exact kwarg, instead construct a fresh `Agent(model=self.agent.model, output_type=dynamic_output)` and call its `.run`. The first test will guide if a fallback is needed.

Update `src/pydantic_ai_stateflow/grounded/__init__.py`:

```python
from pydantic_ai_stateflow.grounded.agent import GroundedAgent, GroundedResult
from pydantic_ai_stateflow.grounded.errors import (
    GroundedBuildError,
    GroundedError,
    GroundedHydrationError,
)
from pydantic_ai_stateflow.grounded.ref import Ref
from pydantic_ai_stateflow.grounded.resolver import GroundedResolver

__all__ = [
    "Ref",
    "GroundedAgent",
    "GroundedResult",
    "GroundedResolver",
    "GroundedError",
    "GroundedBuildError",
    "GroundedHydrationError",
]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_grounded_agent.py -v
```

Expected: 2 tests pass. If `agent.override(output_type=...)` is not supported, fix to the fresh-Agent fallback noted above and re-run.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/agent.py src/pydantic_ai_stateflow/grounded/__init__.py tests/grounded/test_grounded_agent.py
git commit -m "feat(grounded): GroundedAgent wrapper + GroundedResult"
```

---

## Task 18: `HydrationMap` + `GroundedResult.hydrate(**repos)`

**Files:**
- Create: `src/pydantic_ai_stateflow/grounded/hydration.py`
- Modify: `src/pydantic_ai_stateflow/grounded/agent.py`
- Create: `tests/grounded/test_hydration_map.py`

- [ ] **Step 1: Write failing tests**

`tests/grounded/test_hydration_map.py`:

```python
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from pydantic_ai_stateflow.grounded import Ref
from pydantic_ai_stateflow.grounded._scan import scan_output
from pydantic_ai_stateflow.grounded.hydration import HydrationMap


class Item(BaseModel):
    id: UUID
    name: str


class Decision(BaseModel):
    chosen: Ref[Item]
    rationale: str


class FakeRepo:
    def __init__(self, items: dict[UUID, Item]) -> None:
        self._items = items

    async def load(self, id: UUID) -> Item:
        return self._items[id]


@pytest.mark.asyncio
async def test_hydrate_replaces_single_ref_with_entity():
    item_id = uuid4()
    item = Item(id=item_id, name="hydrated")
    repo = FakeRepo({item_id: item})

    decision = Decision(chosen=Ref[Item](item_id), rationale="r")
    hmap = HydrationMap(scan_output(Decision))
    hydrated = await hmap.hydrate(decision, repos={Item: repo})

    # hydrated.chosen is now an Item, not a Ref
    assert isinstance(hydrated["chosen"], Item)
    assert hydrated["chosen"].name == "hydrated"
    assert hydrated["rationale"] == "r"


@pytest.mark.asyncio
async def test_hydrate_works_on_list_of_refs():
    class Out(BaseModel):
        items: list[Ref[Item]]

    ids = [uuid4(), uuid4()]
    items = {i: Item(id=i, name=f"n{idx}") for idx, i in enumerate(ids)}
    repo = FakeRepo(items)

    obj = Out(items=[Ref[Item](ids[0]), Ref[Item](ids[1])])
    hmap = HydrationMap(scan_output(Out))
    hydrated = await hmap.hydrate(obj, repos={Item: repo})

    assert len(hydrated["items"]) == 2
    assert all(isinstance(it, Item) for it in hydrated["items"])


@pytest.mark.asyncio
async def test_hydrate_missing_repo_raises():
    item_id = uuid4()
    decision = Decision(chosen=Ref[Item](item_id), rationale="r")
    hmap = HydrationMap(scan_output(Decision))
    with pytest.raises(KeyError, match="Item"):
        await hmap.hydrate(decision, repos={})
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/grounded/test_hydration_map.py -v
```

Expected: ImportError on `hydration`.

- [ ] **Step 3: Implement `HydrationMap`**

`src/pydantic_ai_stateflow/grounded/hydration.py`:

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from pydantic_ai_stateflow.grounded._spec import FieldRole, OutputSpec
from pydantic_ai_stateflow.grounded.errors import GroundedHydrationError
from pydantic_ai_stateflow.grounded.ref import Ref


class HydrationMap:
    """Walks an output value and replaces Ref instances with entities."""

    def __init__(self, spec: OutputSpec) -> None:
        self._spec = spec

    async def hydrate(self, value: BaseModel, *, repos: dict[type, Any]) -> dict[str, Any]:
        """Return a dict-shaped hydrated view of `value`.

        We deliberately return a dict (not a typed BaseModel) so that the
        consumer is not forced to maintain a separate hydrated-output type
        per pattern. Repos must be indexed by entity TYPE (per 4A.0.4).
        """
        return await _hydrate_model(value, self._spec, repos)


async def _hydrate_model(
    obj: BaseModel, spec: OutputSpec, repos: dict[type, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fspec in spec.fields.items():
        value = getattr(obj, name)
        match fspec.role:
            case FieldRole.REF:
                if not isinstance(value, Ref):
                    out[name] = value
                else:
                    repo = repos.get(fspec.target_type)
                    if repo is None:
                        raise KeyError(
                            f"hydrate: missing repo for {fspec.target_type.__name__}"
                        )
                    out[name] = await value.hydrate(repo)

            case FieldRole.LIST_REF:
                repo = repos.get(fspec.target_type)
                if repo is None:
                    raise KeyError(
                        f"hydrate: missing repo for {fspec.target_type.__name__}"
                    )
                out[name] = [await r.hydrate(repo) if isinstance(r, Ref) else r for r in value]

            case FieldRole.OPTIONAL_REF:
                if value is None:
                    out[name] = None
                elif isinstance(value, Ref):
                    repo = repos.get(fspec.target_type)
                    if repo is None:
                        raise KeyError(
                            f"hydrate: missing repo for {fspec.target_type.__name__}"
                        )
                    out[name] = await value.hydrate(repo)
                else:
                    out[name] = value

            case FieldRole.NESTED:
                out[name] = await _hydrate_model(value, fspec.nested_spec, repos)

            case FieldRole.LIST_NESTED:
                out[name] = [await _hydrate_model(v, fspec.nested_spec, repos) for v in value]

            case _:
                out[name] = value
    return out
```

Add `hydrate` method on `GroundedResult` — modify `src/pydantic_ai_stateflow/grounded/agent.py`:

```python
from pydantic_ai_stateflow.grounded.hydration import HydrationMap


class GroundedResult(BaseModel, Generic[OutT]):
    model_config = {"arbitrary_types_allowed": True}

    value: Any
    raw: Any
    _spec: OutputSpec

    async def hydrate(self, **repos: Any) -> dict[str, Any]:
        """Replace Ref instances in `value` with entities loaded via repos.

        `repos` is type-keyed: pass `Item=item_repo, Customer=customer_repo`,
        where the key name MUST match the entity-type's class __name__.

        Example:
            hydrated = await result.hydrate(Item=item_repo, Customer=cust_repo)
        """
        repos_by_type: dict[type, Any] = {}
        for type_name, repo in repos.items():
            # Find the type in spec's referenced_entity_types by class name
            for t in self._spec.referenced_entity_types:
                if t.__name__ == type_name:
                    repos_by_type[t] = repo
                    break
            else:
                # Unused repo is OK — caller might pass extras
                pass
        hmap = HydrationMap(self._spec)
        return await hmap.hydrate(self.value, repos=repos_by_type)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/grounded/test_hydration_map.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/pydantic_ai_stateflow/grounded/hydration.py src/pydantic_ai_stateflow/grounded/agent.py tests/grounded/test_hydration_map.py
git commit -m "feat(grounded): HydrationMap + GroundedResult.hydrate(**repos) by type-name"
```

---

## Task 19: End-to-end smoke test and public API

**Files:**
- Modify: `src/pydantic_ai_stateflow/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_smoke_end_to_end.py`

- [ ] **Step 1: Wire top-level public API**

`src/pydantic_ai_stateflow/__init__.py`:

```python
"""pydantic-ai-stateflow — Sub-project #1 (Foundation) public API.

Layer 0 (GroundedSchema):
    Ref, GroundedAgent, GroundedResult, GroundedResolver
    GroundedError, GroundedBuildError, GroundedHydrationError

Runtime helpers:
    Det, IdempotencyInput, IdempotencyValue

Patterns:
    Pattern (Protocol)
"""

from pydantic_ai_stateflow.grounded import (
    GroundedAgent,
    GroundedBuildError,
    GroundedError,
    GroundedHydrationError,
    GroundedResolver,
    GroundedResult,
    Ref,
)
from pydantic_ai_stateflow.patterns import Pattern
from pydantic_ai_stateflow.runtime import Det, IdempotencyInput, IdempotencyValue

__all__ = [
    "Det",
    "GroundedAgent",
    "GroundedBuildError",
    "GroundedError",
    "GroundedHydrationError",
    "GroundedResolver",
    "GroundedResult",
    "IdempotencyInput",
    "IdempotencyValue",
    "Pattern",
    "Ref",
]
```

- [ ] **Step 2: Write end-to-end smoke test**

`tests/integration/__init__.py`: (empty)

`tests/integration/test_smoke_end_to_end.py`:

```python
"""End-to-end smoke test exercising every Sub-project #1 component."""

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_stateflow import (
    Det,
    GroundedAgent,
    IdempotencyInput,
    Ref,
)


class Candidate(BaseModel):
    id: UUID
    label: str
    score: float


class Customer(BaseModel):
    id: UUID
    email: str


class Context(BaseModel):
    customer: Customer
    candidates: list[Candidate]


class Decision(BaseModel):
    chosen_customer: Ref[Customer]
    chosen_candidate: Ref[Candidate]
    rationale: str


class FakeRepo:
    def __init__(self, mapping: dict[UUID, object]) -> None:
        self._mapping = mapping

    async def load(self, id: UUID):
        return self._mapping[id]


@pytest.mark.asyncio
async def test_full_grounded_flow_with_hydration_and_idempotency():
    customer_id = uuid4()
    candidate_ids = [uuid4(), uuid4(), uuid4()]
    customer = Customer(id=customer_id, email="who@where.com")
    candidates = [Candidate(id=i, label=f"c{idx}", score=idx * 0.1)
                  for idx, i in enumerate(candidate_ids)]

    ctx = Context(customer=customer, candidates=candidates)

    # FunctionModel that returns the *second* candidate
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(
            tool_name="final_result",
            args={
                "chosen_customer": str(customer_id),
                "chosen_candidate": str(candidate_ids[1]),
                "rationale": "highest available score",
            },
        )])

    base_agent: Agent[None, Decision] = Agent(
        model=FunctionModel(fn), output_type=Decision,
    )
    grounded = GroundedAgent(base_agent, output_type=Decision)
    result = await grounded.run(ctx, instructions="pick best candidate")

    assert result.value.chosen_customer.id == customer_id
    assert result.value.chosen_candidate.id == candidate_ids[1]

    # Hydrate
    cust_repo = FakeRepo({customer_id: customer})
    cand_repo = FakeRepo({c.id: c for c in candidates})
    hydrated = await result.hydrate(Customer=cust_repo, Candidate=cand_repo)

    assert isinstance(hydrated["chosen_customer"], Customer)
    assert isinstance(hydrated["chosen_candidate"], Candidate)
    assert hydrated["chosen_candidate"].label == "c1"

    # Det.uuid_for produces a stable idempotency key for this run
    key = await Det.uuid_for(IdempotencyInput(
        namespace="smoke",
        parts={
            "customer_id": customer_id,
            "chosen_candidate_id": result.value.chosen_candidate.id,
        },
    ))
    assert isinstance(key, UUID)
    assert key.version == 5
```

- [ ] **Step 3: Run test — verify it passes**

```bash
uv run pytest tests/integration/test_smoke_end_to_end.py -v
```

Expected: 1 test passes.

- [ ] **Step 4: Run full suite — everything green**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

Expected: all tests pass, no type or lint errors.

- [ ] **Step 5: Commit**

```bash
git add src/pydantic_ai_stateflow/__init__.py tests/integration
git commit -m "feat: public API + end-to-end smoke test (Sub-project #1 complete)"
```

---

## Sub-project #1 acceptance criteria

After all 19 tasks:

- ✅ `from pydantic_ai_stateflow import Ref, GroundedAgent, Det, IdempotencyInput, Pattern` works
- ✅ A `Ref[Entity]` field serializes to / deserializes from a plain UUID string in Pydantic models
- ✅ `GroundedResolver` builds a dynamic Pydantic class from output_type + context where every Ref becomes `Literal[*context_ids]`
- ✅ Hallucinated UUIDs are rejected by Pydantic validation at parse time (no possible silent acceptance)
- ✅ `result.hydrate(**repos)` returns a dict-shaped view with Refs replaced by loaded entities
- ✅ `Det.uuid_for(IdempotencyInput(...))` produces stable UUID5 keys; floats and unknown types rejected at construction
- ✅ `Pattern` Protocol is structurally checkable at runtime
- ✅ Full suite passes; mypy strict + ruff clean
- ✅ One end-to-end test exercises Ref → Resolver → GroundedAgent → hydrate → Det.uuid_for in one go
