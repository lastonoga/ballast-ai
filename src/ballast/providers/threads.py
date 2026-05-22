"""ThreadsProvider — wires the app's :class:`ThreadRepository` onto Ballast."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ballast.app import Ballast
    from ballast.persistence.thread.repository import ThreadRepository


class ThreadsProvider:
    """Set the :class:`ThreadRepository` used by the :class:`Engine`."""

    def __init__(self, thread_repo: "ThreadRepository") -> None:
        self._thread_repo = thread_repo

    def register(self, ballast: "Ballast") -> None:
        ballast._set_thread_repo(self._thread_repo)


__all__ = ["ThreadsProvider"]
