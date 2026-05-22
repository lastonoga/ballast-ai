"""Tests for ``ballast.errors`` — base hierarchy + formatters."""
from __future__ import annotations

import re
from uuid import uuid4

import pytest

from ballast.errors import (
    AuthError,
    AuthorizationDenied,
    ConfigurationError,
    ConfigurationInvariantViolation,
    MissingDependencyError,
    PersistenceError,
    SettingsValidationError,
    BallastError,
    ThreadMetadataInvalid,
    ThreadNotFound,
    format_error,
)


# ---------- Base contract ----------


def test_to_dict_shape() -> None:
    err = BallastError(
        "something blew up",
        hint="restart it",
        context={"k": "v"},
    )
    assert err.to_dict() == {
        "code": "BALLAST_UNKNOWN",
        "detail": "something blew up",
        "hint": "restart it",
        "context": {"k": "v"},
    }


def test_default_status_code() -> None:
    assert BallastError.status_code == 500
    err = BallastError("x")
    assert err.status_code == 500


def test_repr_mentions_code() -> None:
    err = BallastError("boom")
    r = repr(err)
    assert "BALLAST_UNKNOWN" in r
    assert "boom" in r


def test_context_defaults_to_empty_dict() -> None:
    err = BallastError("x")
    assert err.context == {}
    assert err.hint is None


# ---------- Subclass behaviour ----------


def test_custom_subclass_code_and_status() -> None:
    class MyErr(BallastError):
        code = "BALLAST_TEST_THING"
        status_code = 418

    err = MyErr("teapot")
    assert err.code == "BALLAST_TEST_THING"
    assert err.status_code == 418
    assert err.to_dict()["code"] == "BALLAST_TEST_THING"


def test_hint_and_context_propagate_through_subclass() -> None:
    err = ConfigurationError("bad", hint="fix it", context={"field": "x"})
    d = err.to_dict()
    assert d["hint"] == "fix it"
    assert d["context"] == {"field": "x"}


def test_hierarchy_inheritance() -> None:
    assert issubclass(ConfigurationError, BallastError)
    assert issubclass(SettingsValidationError, ConfigurationError)
    assert issubclass(MissingDependencyError, ConfigurationError)
    assert issubclass(ConfigurationInvariantViolation, ConfigurationError)
    assert issubclass(ThreadNotFound, PersistenceError)
    assert issubclass(ThreadMetadataInvalid, PersistenceError)
    assert issubclass(AuthorizationDenied, AuthError)


# ---------- Specific subclasses ----------


def test_thread_not_found_with_thread_id() -> None:
    tid = str(uuid4())
    err = ThreadNotFound(thread_id=tid)
    assert err.status_code == 404
    assert err.code == "BALLAST_PERSISTENCE_THREAD_NOT_FOUND"
    assert tid in err.detail
    assert err.context["thread_id"] == tid
    assert err.hint is not None


def test_thread_not_found_with_explicit_detail() -> None:
    err = ThreadNotFound("custom message", thread_id="abc")
    assert err.detail == "custom message"
    assert err.context["thread_id"] == "abc"


def test_thread_not_found_minimal() -> None:
    err = ThreadNotFound()
    assert err.detail == "thread not found"
    assert err.status_code == 404


def test_thread_metadata_invalid_status() -> None:
    err = ThreadMetadataInvalid("bad shape")
    assert err.status_code == 422


def test_auth_error_status() -> None:
    assert AuthError.status_code == 401
    assert AuthorizationDenied.status_code == 403


def test_configuration_invariant_violation_code() -> None:
    err = ConfigurationInvariantViolation("startup invariant broken")
    assert err.code == "BALLAST_CONFIG_INVARIANT"
    assert isinstance(err, ConfigurationError)


# ---------- format_error ----------


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def test_format_error_plain_no_ansi() -> None:
    err = ConfigurationError(
        "missing API key",
        hint="set OPENROUTER_API_KEY",
        context={"provider": "openrouter"},
    )
    out = format_error(err, color=False)
    assert "BALLAST_CONFIG" in out
    assert "missing API key" in out
    assert "hint" in out
    assert "set OPENROUTER_API_KEY" in out
    assert "provider" in out
    assert _ANSI_RE.search(out) is None


def test_format_error_plain_omits_hint_when_none() -> None:
    err = BallastError("simple")
    out = format_error(err, color=False)
    assert "hint" not in out
    assert "context" not in out
    assert "BALLAST_UNKNOWN" in out


def test_format_error_color_true_emits_ansi() -> None:
    pytest.importorskip("rich")
    err = BallastError("colorful", hint="fix")
    out = format_error(err, color=True)
    assert _ANSI_RE.search(out) is not None
    assert "BALLAST_UNKNOWN" in out
