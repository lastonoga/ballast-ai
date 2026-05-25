"""Smoke check that the live pattern surface stays importable from
the top-level ``ballast`` package."""
from __future__ import annotations

from ballast import (
    CardVerdict,
    DBOSHITLChannel,
    DivergentAgent,
    DivergentBranch,
    DivergentConvergent,
    HITLChannel,
    HITLDenied,
    HITLTimedOut,
    Pattern,
    PatternError,
    Synthesizer,
    ThreadChannel,
    UICardChannel,
    Verifier,
    register_card_kind,
)


def test_live_pattern_surface_is_exported() -> None:
    """Every name the framework currently ships from ``ballast.patterns``
    (plus the HITL channel primitives and entry points) must resolve via
    the top-level re-export."""
    for symbol in (
        CardVerdict,
        DBOSHITLChannel,
        DivergentAgent,
        DivergentBranch,
        DivergentConvergent,
        HITLChannel,
        HITLDenied,
        HITLTimedOut,
        Pattern,
        PatternError,
        Synthesizer,
        ThreadChannel,
        UICardChannel,
        Verifier,
        register_card_kind,
    ):
        assert symbol is not None


def test_legacy_hitl_symbols_are_gone() -> None:
    """Legacy ask_human / DurableHITLWorkflow / HITLResponse union must no
    longer be importable from the top-level ``ballast`` package."""
    import ballast

    for name in (
        "ask_human",
        "DurableHITLWorkflow",
        "HITLResponse",
        "ApprovedResponse",
        "RejectedResponse",
        "ModifiedResponse",
        "TimeoutResponse",
    ):
        assert not hasattr(ballast, name), f"ballast.{name} should have been removed"
