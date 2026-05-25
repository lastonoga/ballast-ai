"""``current_user_id`` ContextVar — set/get/reset via ``acting_as`` cm."""
from __future__ import annotations

from ballast.auth.context import acting_as, current_user_id


def test_default_is_none() -> None:
    assert current_user_id() is None


def test_acting_as_sets_and_resets() -> None:
    assert current_user_id() is None
    with acting_as("user-1"):
        assert current_user_id() == "user-1"
    assert current_user_id() is None


def test_nested_acting_as_restores_outer() -> None:
    with acting_as("outer"):
        assert current_user_id() == "outer"
        with acting_as("inner"):
            assert current_user_id() == "inner"
        assert current_user_id() == "outer"
    assert current_user_id() is None
