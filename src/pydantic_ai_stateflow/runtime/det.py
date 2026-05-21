from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TypeVar
from uuid import UUID, uuid5
from uuid import uuid4 as _uuid4

from dbos import DBOS

from pydantic_ai_stateflow.runtime.idempotency import IdempotencyInput

T = TypeVar("T")

# A fixed namespace UUID for all uuid_for derivations within this framework.
# Generated once via `uuid4()`; hardcoded so derivations are reproducible
# across processes and machines.
_UUID_NAMESPACE = UUID("ad9c8e22-1bc4-4a4f-9c40-d9c4f4ad7e10")


class Det:
    """Deterministic-recorded helpers wrapped as DBOS steps.

    Each method's result is recorded by DBOS on first invocation and
    replayed verbatim on recovery. This makes deterministic UUIDs and
    timestamps robust across crashes — replay produces identical values
    without re-running the function.

    Side-effects outside `@DBOS.step` boundaries are forbidden by lint
    rule STATEFLOW001-007 (Sub-project #3 Task 10).
    """

    @staticmethod
    @DBOS.step()
    async def now() -> datetime:
        return datetime.now(tz=UTC)

    @staticmethod
    @DBOS.step()
    async def uuid4() -> UUID:
        return _uuid4()

    @staticmethod
    @DBOS.step()
    async def random_choice(seq: Sequence[T]) -> T:
        # Accepts any Sequence (list, tuple, etc.) — broader than list-only.
        # CPython random.choice is thread-safe (module-level Random instance
        # is GIL-protected); do not replace with a non-thread-safe variant.
        return random.choice(seq)

    @staticmethod
    @DBOS.step()
    async def uuid_for(inputs: IdempotencyInput) -> UUID:
        """Deterministic UUID5 from a strict-typed input.

        Wrapped as @DBOS.step (Critical Fix #1, code-review pass): the
        result is durably recorded so replay returns the same UUID
        regardless of serialization-version drift across Pydantic /
        Python upgrades.
        """
        canonical = inputs.canonical_json()
        return uuid5(_UUID_NAMESPACE, canonical)
