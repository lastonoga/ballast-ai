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
