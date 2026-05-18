from __future__ import annotations

from uuid import UUID


def _hitl_topic(tenant_id: UUID, request_id: UUID) -> str:
    """Tenant-scoped DBOS topic for HITL replies.

    Format `hitl:{tenant_id}:{request_id}` per spec 2C.4. The tenant
    prefix prevents cross-tenant collisions if a request_id is ever
    reused across tenants (defensive — UUIDs should be globally
    unique, but topic isolation is still required).
    """
    return f"hitl:{tenant_id}:{request_id}"
