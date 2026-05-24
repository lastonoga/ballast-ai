from ballast.errors import BallastError
from ballast.patterns import (
    HITLDenied,
    HITLTimedOut,
    InsufficientDivergence,
    PatternError,
)


def test_pattern_error_is_exception_root():
    """All pattern-specific errors share a single PatternError root."""
    assert issubclass(HITLTimedOut, PatternError)
    assert issubclass(HITLDenied, PatternError)
    assert issubclass(InsufficientDivergence, PatternError)


def test_hitl_timed_out_carries_request_id():
    from uuid import uuid4
    rid = uuid4()
    err = HITLTimedOut(request_id=rid)
    assert err.request_id == rid


def test_hitl_denied_carries_actor_and_votes():
    err = HITLDenied(actor_id="alice", votes={"admin_voter": "deny"})
    assert err.actor_id == "alice"
    assert err.votes == {"admin_voter": "deny"}
    assert "alice" in str(err)


def test_pattern_errors_inherit_ballast_error():
    """Every PatternError subclass is a BallastError."""
    assert issubclass(PatternError, BallastError)
    assert issubclass(HITLTimedOut, BallastError)
    assert issubclass(HITLDenied, BallastError)
    assert issubclass(InsufficientDivergence, BallastError)


def test_pattern_errors_have_codes_and_status():
    assert HITLTimedOut.code == "BALLAST_PATTERN_HITL_TIMED_OUT"
    assert HITLTimedOut.status_code == 504
    assert HITLDenied.code == "BALLAST_PATTERN_HITL_DENIED"
    assert HITLDenied.status_code == 403
    assert InsufficientDivergence.code == "BALLAST_PATTERN_INSUFFICIENT_DIVERGENCE"
    assert InsufficientDivergence.status_code == 500


def test_insufficient_divergence_backcompat_attrs():
    err = InsufficientDivergence(
        produced=1,
        required=2,
        branch_outcomes={"practical": "ok", "creative": "failed"},
    )
    assert err.produced == 1
    assert err.required == 2
    assert err.branch_outcomes == {"practical": "ok", "creative": "failed"}
    d = err.to_dict()
    assert d["code"] == "BALLAST_PATTERN_INSUFFICIENT_DIVERGENCE"
    assert d["context"]["produced"] == 1
    assert d["context"]["required"] == 2
    assert d["hint"] is not None
