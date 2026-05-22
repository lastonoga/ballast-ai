from __future__ import annotations

from uuid import UUID


def _hitl_topic(request_id: UUID) -> str:
    """DBOS topic for HITL replies, scoped per-request.

    Format ``hitl:{request_id}``. UUIDs are globally unique so a
    request-id-only topic is collision-safe.
    """
    return f"hitl:{request_id}"
