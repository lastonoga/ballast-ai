from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pydantic_ai_stateflow.patterns.hitl import HITLPrompt


def test_prompt_requires_tenant_id() -> None:
    """4A.0.2: tenant_id is required on the prompt (NOT a separate kwarg)."""
    with pytest.raises(ValidationError):
        HITLPrompt(title="x", context="y", decision_kinds={"approved"})  # type: ignore[call-arg]


def test_prompt_constructed_with_minimum_fields() -> None:
    p = HITLPrompt(
        tenant_id=uuid4(),
        title="Approve refund",
        context="$5000 over policy",
        decision_kinds={"approved", "rejected"},
    )
    assert p.title == "Approve refund"
    assert p.timeout is None


def test_prompt_supports_timeout() -> None:
    p = HITLPrompt(
        tenant_id=uuid4(), title="x", context="y",
        decision_kinds={"approved"}, timeout=timedelta(seconds=30),
    )
    assert p.timeout == timedelta(seconds=30)
