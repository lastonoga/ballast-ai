"""Suite-wide pytest fixtures for the framework tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from ballast.events import helper_thread_created, message_added
from ballast.patterns.divergent_convergent.events import (
    divergent_convergent_progress,
)
from ballast.patterns.reflection import reflection_progress


@pytest.fixture(autouse=True)
def _isolate_signals() -> Iterator[None]:
    """Snapshot + restore signal receivers across tests.

    Framework signals are module-level singletons. Tests that install
    fresh defaults — e.g. via ``Ballast.with_events`` — would
    otherwise leak their receivers into every following test, causing
    duplicate fires or stale closures pointing at torn-down engines.
    Snapshotting on enter + restoring on exit keeps the receiver
    lists stable across the suite.
    """
    snapshots = {
        s: list(s._receivers)
        for s in (
            message_added,
            helper_thread_created,
            divergent_convergent_progress,
            reflection_progress,
        )
    }
    try:
        yield
    finally:
        for s, rec in snapshots.items():
            s._receivers = list(rec)
