"""``Scope`` — app-subclassable scope BaseModel with extra=allow."""
from __future__ import annotations

import pytest

from ballast.memory import Scope


def test_default_scope_has_optional_user_tenant_thread() -> None:
    s = Scope()
    assert s.user_id is None
    assert s.tenant_id is None
    assert s.thread_id is None


def test_explicit_scope_fields() -> None:
    s = Scope(user_id="u-1", tenant_id="t-1", thread_id="th-1")
    assert (s.user_id, s.tenant_id, s.thread_id) == ("u-1", "t-1", "th-1")


def test_extra_fields_allowed_for_app_dimensions() -> None:
    s = Scope(user_id="u-1", project_id="p-99")  # type: ignore[call-arg]
    assert getattr(s, "project_id", None) == "p-99"


def test_subclass_adds_typed_dimensions() -> None:
    class ProjectScope(Scope):
        project_id: str | None = None

    s = ProjectScope(user_id="u-1", project_id="p-9")
    assert s.project_id == "p-9"
    assert isinstance(s, Scope)
