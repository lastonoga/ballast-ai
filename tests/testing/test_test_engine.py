"""Unit tests for ``TestEngine``."""
from __future__ import annotations

import pytest

from pydantic_ai_stateflow.observability.config import (
    _reset_observability_for_tests,
)
from pydantic_ai_stateflow.persistence import InMemoryThreadRepository
from pydantic_ai_stateflow.persistence.thread.repository import ThreadRepository
from pydantic_ai_stateflow.testing import TestEngine


@pytest.fixture(autouse=True)
def _reset_obs():
    _reset_observability_for_tests()
    yield
    _reset_observability_for_tests()


def test_default_engine_boots(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    with TestEngine.default().test_client() as client:
        r = client.get("/threads")
        assert r.status_code == 200, r.text


def test_override_thread_repo(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    engine = TestEngine.default()
    custom = InMemoryThreadRepository()
    engine.override(ThreadRepository, custom)
    with engine.test_client() as client:
        r = client.get("/threads")
        assert r.status_code == 200, r.text


def test_engine_cleans_up_on_exit(fresh_dbos_executor: None) -> None:
    """No DBOS state leaks across TestEngine context boundaries in same process."""
    del fresh_dbos_executor
    e1 = TestEngine.default()
    with e1.test_client() as c1:
        assert c1.get("/threads").status_code == 200
    e2 = TestEngine.default()
    with e2.test_client() as c2:
        assert c2.get("/threads").status_code == 200


def test_override_chains(fresh_dbos_executor: None) -> None:
    del fresh_dbos_executor
    engine = TestEngine.default()
    repo = InMemoryThreadRepository()
    result = engine.override(ThreadRepository, repo)
    assert result is engine


def test_override_unsupported_target_raises(fresh_dbos_executor: None) -> None:
    """Workflow / agent overrides aren't supported anymore — the registry
    is gone."""
    del fresh_dbos_executor
    engine = TestEngine.default()

    class _NotARepo:
        pass

    with pytest.raises(TypeError, match="only ThreadRepository"):
        engine.override(_NotARepo, _NotARepo())  # type: ignore[arg-type]
