"""Unit tests for the ``Engine`` config-holder + process-wide singleton."""
from __future__ import annotations

import pytest

from ballast.errors import ConfigurationError
from ballast.persistence import (
    InMemoryEventLogRepository,
    InMemoryThreadRepository,
)
from ballast.runtime.engine import (
    Engine,
    _reset_engine_for_tests,
    _set_engine,
    get_engine,
)
from ballast.runtime.event_stream import InProcessEventStream
from ballast.runtime.thread_events import ThreadEventBroadcaster


def _build_engine() -> Engine:
    return Engine(
        thread_repo=InMemoryThreadRepository(),
        event_log=InMemoryEventLogRepository(),
        event_stream=InProcessEventStream(),
    )


def test_engine_broadcaster_cached() -> None:
    engine = _build_engine()
    assert engine.broadcaster is engine.broadcaster
    assert isinstance(engine.broadcaster, ThreadEventBroadcaster)


def test_engine_is_frozen() -> None:
    engine = _build_engine()
    with pytest.raises(Exception):
        engine.thread_repo = InMemoryThreadRepository()  # type: ignore[misc]


def test_get_engine_raises_when_uninitialized() -> None:
    _reset_engine_for_tests()
    with pytest.raises(ConfigurationError):
        get_engine()


def test_set_engine_idempotent_same_instance() -> None:
    _reset_engine_for_tests()
    engine = _build_engine()
    _set_engine(engine)
    _set_engine(engine)  # idempotent
    assert get_engine() is engine
    _reset_engine_for_tests()


def test_set_engine_raises_on_different_instance() -> None:
    _reset_engine_for_tests()
    engine1 = _build_engine()
    engine2 = _build_engine()
    _set_engine(engine1)
    with pytest.raises(ConfigurationError):
        _set_engine(engine2)
    _reset_engine_for_tests()


def test_reset_clears_singleton() -> None:
    _reset_engine_for_tests()
    engine = _build_engine()
    _set_engine(engine)
    assert get_engine() is engine
    _reset_engine_for_tests()
    with pytest.raises(ConfigurationError):
        get_engine()
