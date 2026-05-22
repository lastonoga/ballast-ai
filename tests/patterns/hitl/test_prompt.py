from __future__ import annotations

from datetime import timedelta

from ballast.patterns.hitl import HITLPrompt


def test_prompt_constructed_with_minimum_fields() -> None:
    p = HITLPrompt(
        title="Approve refund",
        context="$5000 over policy",
        decision_kinds={"approved", "rejected"},
    )
    assert p.title == "Approve refund"
    assert p.timeout is None


def test_prompt_supports_timeout() -> None:
    p = HITLPrompt(
        title="x", context="y",
        decision_kinds={"approved"}, timeout=timedelta(seconds=30),
    )
    assert p.timeout == timedelta(seconds=30)
