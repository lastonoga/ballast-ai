"""Tests for ``pydantic_ai_stateflow.errors`` — base hierarchy + formatters."""
from __future__ import annotations

import re
from uuid import uuid4

import pytest

from pydantic_ai_stateflow.errors import (
    AuthError,
    AuthorizationDenied,
    ConfigurationError,
    ConfigurationInvariantViolation,
    MissingDependencyError,
    PersistenceError,
    SettingsValidationError,
    StateflowError,
    ThreadMetadataInvalid,
    ThreadNotFound,
    format_error,
)


# ---------- Base contract ----------


def test_to_dict_shape() -> None:
    err = StateflowError(
        "something blew up",
        hint="restart it",
        context={"k": "v"},
    )
    assert err.to_dict() == {
        "code": "STATEFLOW_UNKNOWN",
        "detail": "something blew up",
        "hint": "restart it",
        "context": {"k": "v"},
    }


def test_default_status_code() -> None:
    assert StateflowError.status_code == 500
    err = StateflowError("x")
    assert err.status_code == 500


def test_repr_mentions_code() -> None:
    err = StateflowError("boom")
    r = repr(err)
    assert "STATEFLOW_UNKNOWN" in r
    assert "boom" in r


def test_context_defaults_to_empty_dict() -> None:
    err = StateflowError("x")
    assert err.context == {}
    assert err.hint is None


# ---------- Subclass behaviour ----------


def test_custom_subclass_code_and_status() -> None:
    class MyErr(StateflowError):
        code = "STATEFLOW_TEST_THING"
        status_code = 418

    err = MyErr("teapot")
    assert err.code == "STATEFLOW_TEST_THING"
    assert err.status_code == 418
    assert err.to_dict()["code"] == "STATEFLOW_TEST_THING"


def test_hint_and_context_propagate_through_subclass() -> None:
    err = ConfigurationError("bad", hint="fix it", context={"field": "x"})
    d = err.to_dict()
    assert d["hint"] == "fix it"
    assert d["context"] == {"field": "x"}


def test_hierarchy_inheritance() -> None:
    assert issubclass(ConfigurationError, StateflowError)
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
    assert err.code == "STATEFLOW_PERSISTENCE_THREAD_NOT_FOUND"
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
    assert err.code == "STATEFLOW_CONFIG_INVARIANT"
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
    assert "STATEFLOW_CONFIG" in out
    assert "missing API key" in out
    assert "hint" in out
    assert "set OPENROUTER_API_KEY" in out
    assert "provider" in out
    assert _ANSI_RE.search(out) is None


def test_format_error_plain_omits_hint_when_none() -> None:
    err = StateflowError("simple")
    out = format_error(err, color=False)
    assert "hint" not in out
    assert "context" not in out
    assert "STATEFLOW_UNKNOWN" in out


def test_format_error_color_true_emits_ansi() -> None:
    pytest.importorskip("rich")
    err = StateflowError("colorful", hint="fix")
    out = format_error(err, color=True)
    assert _ANSI_RE.search(out) is not None
    assert "STATEFLOW_UNKNOWN" in out
