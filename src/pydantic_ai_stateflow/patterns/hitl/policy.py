from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AccessDecision:
    """Result of Policy.can. ``votes`` is a per-voter audit trail."""

    is_grant: bool
    votes: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        return ("GRANT" if self.is_grant else "DENY") + f" votes={self.votes}"


@runtime_checkable
class Voter(Protocol):
    name: str

    def supports(self, action: str, resource: Any) -> bool: ...
    async def vote(
        self, *, actor: Any, action: str, resource: Any,
    ) -> str: ...  # returns "grant" | "deny" | "abstain"


@runtime_checkable
class Policy(Protocol):
    """Yes/no decision on (actor, action, resource) — the DRY authz port.

    Apps that need tenant / workspace scoping carry it inside the
    ``resource`` argument (or in their custom ``actor`` type); the
    framework's Policy contract is identity-agnostic.
    """

    async def can(
        self, *, actor: Any, action: str, resource: Any,
    ) -> AccessDecision: ...


class AllowAll:
    """Reference impl: grants any (actor, action, resource)."""

    async def can(
        self, *, actor: Any, action: str, resource: Any,
    ) -> AccessDecision:
        return AccessDecision(is_grant=True, votes={"allow_all": "grant"})


class DenyAll:
    """Reference impl: denies anything."""

    async def can(
        self, *, actor: Any, action: str, resource: Any,
    ) -> AccessDecision:
        return AccessDecision(is_grant=False, votes={"deny_all": "deny"})
