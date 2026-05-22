from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

ContextT = TypeVar("ContextT")


class HelperVerdict(BaseModel, Generic[ContextT]):
    """Structured verdict from a `HelperAgent`. Domain-agnostic base (spec 3J.2).

    Apps embed domain context via `context: ContextT | None`; concrete usages:

    - `HelperVerdict[None]` — no domain extension (simple approve/reject).
    - `HelperVerdict[StrategyReviewContext]` — strategy-review specific.

    NOTE (project quirk): parameterized aliases of `HelperVerdict[Foo]` that
    cross `@DBOS.workflow()` boundaries MUST be defined as module-level
    constants (e.g. `_StrategyVerdict = HelperVerdict[StrategyReviewContext]`)
    so DBOS's pickler can resolve the type on replay.
    """

    model_config = ConfigDict(frozen=True)

    rationale: str
    confidence: float
    conversation_turn_count: int
    tools_invoked: list[str]
    autopilot_eligible: bool = False
    autopilot_confidence: float | None = None
    context: ContextT | None = None
