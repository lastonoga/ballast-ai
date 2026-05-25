"""``UICardChannel`` — out-of-thread approval card delivered via a
side-panel SSE stream.

This module ships:
  - ``CardVerdict[OutT]`` — the standard verdict shape for card-style
    approvals (`decision` + optional `modified` payload).
  - ``card_kind_registry`` — `__hitl_kind__` → payload BaseModel
    lookup; the REST decision endpoint uses this to validate the
    incoming ``modified`` payload against the right type.
  - ``approval_card_requested`` / ``approval_card_decided`` signals —
    SSE multiplexer subscribes to both.

The actual ``UICardChannel`` class lands in the next task.
"""
from __future__ import annotations

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

from ballast.events.signals import Signal

OutT = TypeVar("OutT", bound=BaseModel)


class CardVerdict(BaseModel, Generic[OutT]):
    """Standard verdict for card-style approvals.

    Custom channels are free to ship their own verdict shapes; this
    one covers the common UI card case (approve/reject with optional
    edits coming back).
    """

    decision: Literal["approve", "reject"]
    modified: OutT | None = None
    feedback: str | None = None
    answered_at: datetime


# ── kind registry ───────────────────────────────────────────────────

card_kind_registry: dict[str, type[BaseModel]] = {}


def register_card_kind(model: type[BaseModel]) -> type[BaseModel]:
    """Register a payload model under its ``__hitl_kind__``.

    The REST decision endpoint reads this to know how to validate the
    incoming ``modified`` body against the right type. Idempotent:
    re-registering the same class is a no-op; re-registering a
    different class under the same kind raises.
    """
    kind = getattr(model, "__hitl_kind__", None)
    if not kind:
        raise AttributeError(
            f"{model.__name__} must declare __hitl_kind__ to register",
        )
    existing = card_kind_registry.get(kind)
    if existing is not None and existing is not model:
        raise ValueError(
            f"__hitl_kind__={kind!r} already registered to "
            f"{existing.__name__}; cannot reassign to {model.__name__}",
        )
    card_kind_registry[kind] = model
    return model


# ── signals ─────────────────────────────────────────────────────────

approval_card_requested: Signal = Signal("approval-card-requested")
approval_card_decided:   Signal = Signal("approval-card-decided")


__all__ = [
    "CardVerdict",
    "approval_card_decided",
    "approval_card_requested",
    "card_kind_registry",
    "register_card_kind",
]
