from pydantic import BaseModel, ValidationError

from ballast.capabilities.grounded_retry import (
    GroundedRetry,
    _build_feedback,
)


class _Out(BaseModel):
    choice: str
    score: int


def test_build_feedback_for_missing_field():
    try:
        _Out.model_validate({"choice": "a"})
    except ValidationError as err:
        feedback = _build_feedback(err, raw_output={"choice": "a"})
    assert "score" in feedback
    assert "missing" in feedback.lower()


def test_build_feedback_for_literal_violation():
    """Literal-type errors should mention the allowed values."""
    from typing import Literal

    class L(BaseModel):
        status: Literal["a", "b", "c"]

    try:
        L.model_validate({"status": "z"})
    except ValidationError as err:
        feedback = _build_feedback(err, raw_output={"status": "z"})
    assert "status" in feedback
    assert "z" in feedback or "'z'" in feedback


def test_grounded_retry_has_max_retries_default():
    cap = GroundedRetry()
    assert cap.max_retries == 3


def test_grounded_retry_accepts_custom_max_retries():
    cap = GroundedRetry(max_retries=5)
    assert cap.max_retries == 5
