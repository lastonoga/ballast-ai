"""Smoke check that the live pattern surface stays importable from
the top-level ``ballast`` package."""
from __future__ import annotations

from ballast import (
    ApprovedResponse,
    DivergentAgent,
    DivergentBranch,
    DivergentConvergent,
    DurableHITLWorkflow,
    HITLDenied,
    HITLResponse,
    HITLTimedOut,
    ModifiedResponse,
    Pattern,
    PatternError,
    RejectedResponse,
    Synthesizer,
    TimeoutResponse,
    Verifier,
    ask_human,
)


def test_live_pattern_surface_is_exported() -> None:
    """Every name the framework currently ships from ``ballast.patterns``
    (plus the HITL response types and entry points) must resolve via
    the top-level re-export."""
    for symbol in (
        ApprovedResponse,
        DivergentAgent,
        DivergentBranch,
        DivergentConvergent,
        DurableHITLWorkflow,
        HITLDenied,
        HITLResponse,
        HITLTimedOut,
        ModifiedResponse,
        Pattern,
        PatternError,
        RejectedResponse,
        Synthesizer,
        TimeoutResponse,
        Verifier,
        ask_human,
    ):
        assert symbol is not None
