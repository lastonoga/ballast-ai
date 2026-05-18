from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@dataclass(frozen=True)
class AccessDecision:
    """Result of Policy.can. `votes` is a per-voter audit trail (DRY across denials)."""

    is_grant: bool
    votes: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return ("GRANT" if self.is_grant else "DENY") + f" votes={self.votes}"


@runtime_checkable
class Voter(Protocol):
    name: str

    def supports(self, action: str, resource: Any) -> bool: ...
    async def vote(
        self, *, actor: Any, action: str, resource: Any, tenant_id: UUID,
    ) -> str: ...  # returns "grant" | "deny" | "abstain"


@runtime_checkable
class Policy(Protocol):
    """Yes/no decision on (actor, action, resource) — the DRY authz port.

    Used in three places (spec 2E.3):
    1. MutationPipeline policy stages
    2. HITLGate.run() — who can respond
    3. FastAPI endpoints — who can hit the route
    """

    async def can(
        self, *, actor: Any, action: str, resource: Any, tenant_id: UUID,
    ) -> AccessDecision: ...


class AllowAll:
    """Reference impl: grants any (actor, action, resource). Useful in tests / dev."""

    async def can(
        self, *, actor: Any, action: str, resource: Any, tenant_id: UUID,
    ) -> AccessDecision:
        return AccessDecision(is_grant=True, votes={"allow_all": "grant"})


class DenyAll:
    """Reference impl: denies anything. Exists to force test/coverage of denial paths."""

    async def can(
        self, *, actor: Any, action: str, resource: Any, tenant_id: UUID,
    ) -> AccessDecision:
        return AccessDecision(is_grant=False, votes={"deny_all": "deny"})
