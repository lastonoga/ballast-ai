"""Pytest plugin: ``engine`` + ``client`` fixtures for stateflow apps.

Apps opt in via ``pytest_plugins = ["ballast.testing.pytest_plugin"]``
in their ``conftest.py``.

The ``engine`` fixture yields a ``TestEngine.default()`` (in-memory
repos, no workflows/agents). Tests that need additional registrations
or overrides do so on the fixture before entering ``client``.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ballast.testing.engine import TestEngine


@pytest.fixture
def engine() -> Iterator[TestEngine]:
    e = TestEngine.default()
    yield e


@pytest.fixture
def client(engine: TestEngine) -> Iterator[Any]:
    with engine.test_client() as c:
        yield c
