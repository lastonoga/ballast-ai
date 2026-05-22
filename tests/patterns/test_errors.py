from ballast.errors import BallastError
from ballast.patterns import (
    HITLDenied,
    HITLTimedOut,
    InsufficientDivergence,
    MutationRejected,
    PatternError,
    ReflectionExhausted,
)


def test_pattern_error_is_exception_root():
    """All pattern-specific errors share a single PatternError root for `except`."""
    assert issubclass(ReflectionExhausted, PatternError)
    assert issubclass(MutationRejected, PatternError)
    assert issubclass(HITLTimedOut, PatternError)
    assert issubclass(HITLDenied, PatternError)


def test_reflection_exhausted_carries_iterations_and_feedback():
    err = ReflectionExhausted(iterations=5, last_feedback=[{"issue": "bad"}])
    assert err.iterations == 5
    assert err.last_feedback == [{"issue": "bad"}]
    assert "5" in str(err)


def test_mutation_rejected_carries_stage_and_reason():
    err = MutationRejected(stage="validation", reason="schema invalid")
    assert err.stage == "validation"
    assert err.reason == "schema invalid"
    assert "validation" in str(err)
    assert "schema invalid" in str(err)


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


def test_pattern_errors_inherit_stateflow_error():
    """Migration: every PatternError subclass is now a BallastError."""
    assert issubclass(PatternError, BallastError)
    assert issubclass(ReflectionExhausted, BallastError)
    assert issubclass(MutationRejected, BallastError)
    assert issubclass(HITLTimedOut, BallastError)
    assert issubclass(HITLDenied, BallastError)
    assert issubclass(InsufficientDivergence, BallastError)


def test_pattern_errors_have_codes_and_status():
    """Stable codes + status_codes per spec §E."""
    assert ReflectionExhausted.code == "BALLAST_PATTERN_REFLECTION_EXHAUSTED"
    assert ReflectionExhausted.status_code == 500
    assert MutationRejected.code == "BALLAST_PATTERN_MUTATION_REJECTED"
    assert MutationRejected.status_code == 500
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
    # Back-compat instance attributes still readable.
    assert err.produced == 1
    assert err.required == 2
    assert err.branch_outcomes == {"practical": "ok", "creative": "failed"}
    # New BallastError contract.
    d = err.to_dict()
    assert d["code"] == "BALLAST_PATTERN_INSUFFICIENT_DIVERGENCE"
    assert d["context"]["produced"] == 1
    assert d["context"]["required"] == 2
    assert d["hint"] is not None
