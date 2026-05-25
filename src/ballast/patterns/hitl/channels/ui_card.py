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

from datetime import datetime, timezone
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, TypeAdapter

from ballast.events.signals import Signal
from ballast.patterns.hitl.channels._base import DBOSHITLChannel
from ballast.patterns.hitl.channels._protocol import InT

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


class UICardChannel(DBOSHITLChannel[InT, "CardVerdict[InT]"]):
    """Persists an ApprovalCard row + fires the request signal so the
    UI panel SSE picks it up. Verdict comes back via
    ``POST /approvals/{id}/decision`` → ``Durable.send_async`` → the
    suspended ``recv_async`` inside ``DBOSHITLChannel.request``.

    Take ``payload_type`` in the constructor so ``decode_verdict`` can
    type-validate the inbound dict — Python's runtime erases generic
    parameters from the class so we can't reach for them via
    ``__orig_class__`` on every instance.
    """

    def __init__(self, payload_type: type[InT]) -> None:
        super().__init__()
        self._payload_type = payload_type

    async def deliver(
        self, *,
        request_id: str, workflow_id: str, respond_topic: str,
        payload: InT,
    ) -> None:
        from ballast.auth.context import current_user_id              # noqa: PLC0415
        from ballast.events.context import current_parent_thread_id   # noqa: PLC0415
        from ballast.persistence.approval_card import (                # noqa: PLC0415
            ApprovalCard, approval_card_repo,
        )

        card = ApprovalCard(
            id=request_id, workflow_id=workflow_id,
            respond_topic=respond_topic,
            kind=type(payload).__hitl_kind__,
            payload=payload.model_dump(mode="json"),
            parent_thread_id=current_parent_thread_id(),
            user_id=current_user_id(),
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        await approval_card_repo.add(card)
        await approval_card_requested.send(self, card=card)

    async def decode_verdict(self, raw: Any) -> "CardVerdict[InT]":
        return TypeAdapter(CardVerdict[self._payload_type]).validate_python(raw)


__all__ = [
    "CardVerdict",
    "UICardChannel",
    "approval_card_decided",
    "approval_card_requested",
    "card_kind_registry",
    "register_card_kind",
]
