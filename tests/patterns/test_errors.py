from pydantic_ai_stateflow.patterns import (
    HITLDenied,
    HITLTimedOut,
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
