from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import TypeVar
from uuid import UUID, uuid5
from uuid import uuid4 as _uuid4

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
        return datetime.now(tz=UTC)

    @staticmethod
    async def uuid4() -> UUID:
        return _uuid4()

    @staticmethod
    async def random_choice(seq: list[T]) -> T:
        return random.choice(seq)

    @staticmethod
    async def uuid_for(inputs: IdempotencyInput) -> UUID:
        """Deterministic UUID5 from a strict-typed input.

        In Sub-project #3 this will be wrapped as `@DBOS.step` so the
        result is durably recorded and replayed verbatim. This eliminates
        ANY risk that serialization variance across Pydantic / Python
        upgrades produces a different UUID on replay.

        `IdempotencyInput` enforces input contract at type level: no floats,
        no loose dicts, only stable primitives. Caller cannot pass
        `{"amount": 1.0}` accidentally.
        """
        canonical = inputs.canonical_json()
        return uuid5(_UUID_NAMESPACE, canonical)
