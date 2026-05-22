"""Verify Det.* methods are registered as @DBOS.step.

We can't easily run them through a real DBOS workflow without a PG +
DBOS launch (covered by Task 11 smoke test). Instead this test inspects
the DBOS step registry to confirm registration.
"""

from ballast.runtime import Det


def _is_dbos_step(fn) -> bool:
    return (
        hasattr(fn, "__dbos_step__")
        or hasattr(fn, "__wrapped__")
        or getattr(fn, "_is_dbos_step", False)
    )


def test_det_now_is_dbos_step():
    fn = Det.now
    assert _is_dbos_step(fn), f"Det.now is not marked as a DBOS step: {fn!r}"


def test_det_uuid4_is_dbos_step():
    fn = Det.uuid4
    assert _is_dbos_step(fn), f"Det.uuid4 is not marked as a DBOS step: {fn!r}"


def test_det_uuid_for_is_dbos_step():
    fn = Det.uuid_for
    assert _is_dbos_step(fn), f"Det.uuid_for is not marked as a DBOS step: {fn!r}"


def test_det_random_choice_is_dbos_step():
    fn = Det.random_choice
    assert _is_dbos_step(fn), f"Det.random_choice is not marked as a DBOS step: {fn!r}"
