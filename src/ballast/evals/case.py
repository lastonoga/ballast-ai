from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    """A single eval input + expected output (when known)."""

    model_config = ConfigDict(frozen=True)
    name: str
    inputs: Any
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRunOutput(BaseModel):
    """What the runner produced for a case + any framework signals.

    `retries` is the BaseModel-level retry count from pydantic-ai
    (`run_result.retries`). 0 means structured output was valid first try.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    output: Any = None
    retries: int = 0
    error: str | None = None
