"""``DomainSemanticSource`` — convenience base for repo-wrapping sources."""
from __future__ import annotations

from abc import ABC
from typing import ClassVar

from ballast.memory.semantic._protocol import SemanticSource


class DomainSemanticSource(SemanticSource, ABC):
    """Base class for semantic sources that wrap domain repositories.

    Convention: subclass, set ``name``, add ``@memory_tool`` methods
    that delegate to repo singletons. Scope (user_id, tenant_id) is
    enforced by the underlying repo via ``current_user_id()``
    ContextVar (installed by Phase 1 ``ballast.auth.context``) — no
    scope parameter on the source methods.

    Pure convenience — DOES NOT enforce any structural shape beyond
    ``name``. Subclasses use ``@memory_tool`` freely on as few or as
    many methods as they want.
    """

    name: ClassVar[str]


__all__ = ["DomainSemanticSource"]
