"""Django-style :class:`Signal` primitive used by Ballast for in-process
fan-out of framework events.

Why this exists
---------------

Without signals, every framework-level write that "needs" downstream
processing has to be open-coded by each caller — typically a three-call
dance::

    msg = await thread_repo.add_message(...)
    ev  = await event_log.append(thread_id=tid, kind="message-added", ...)
    await event_stream.publish(thread_channel(tid), EventNotification(...))

That couples the call site to the event log + event stream wiring AND
forces every flow that appends a message to re-implement the same
boilerplate. With signals the repo emits ``message_added`` and the
framework's default handler (registered by :class:`EventsProvider`) does
the log + publish — callers only call ``add_message`` and the rest
follows.

Receivers can be sync **or** async. Order of invocation is registration
order. :meth:`Signal.send` aborts on the first exception (fail-loud for
framework defaults — a broken default should not be silently dropped);
:meth:`Signal.send_robust` collects exceptions and never raises (suits
audit-style observers that should not break the producer).

A ``sender=`` filter at connect time scopes the receiver to a specific
sender class / instance. Most receivers care about the signal itself, not
who emitted it, so leave ``sender=None``.

``connect`` is **idempotent**: calling it twice with the same
``(receiver, sender)`` pair is a no-op. This matters because providers
that register defaults (e.g. :class:`EventsProvider`) may run more than
once across test fixtures, and we don't want the default to fire twice
per emit.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any


class Signal:
    """In-process pub/sub channel for a single named event.

    Receivers are stored in registration order and invoked sequentially.
    Async receivers are awaited; sync receivers are called directly.

    The receivers list is exposed as ``_receivers`` for test-fixture
    snapshot/restore (see ``tests/conftest.py``); production code MUST
    NOT mutate it directly — use :meth:`connect` / :meth:`disconnect`.
    """

    __slots__ = ("name", "_receivers")

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("Signal name must be non-empty")
        self.name = name
        # Each entry is ``(receiver, sender_filter)``. ``sender_filter``
        # ``None`` means "match every sender".
        self._receivers: list[tuple[Callable[..., Any], Any]] = []

    def connect(
        self,
        receiver: Callable[..., Any],
        *,
        sender: Any = None,
    ) -> None:
        """Register ``receiver`` for this signal.

        Idempotent: a repeated ``connect`` with the same
        ``(receiver, sender)`` pair is a no-op. Required so providers
        that register the framework's default handlers may run more
        than once (test fixtures, re-build cycles) without producing
        duplicate fires.
        """
        for existing_receiver, existing_sender in self._receivers:
            if existing_receiver is receiver and existing_sender is sender:
                return
        self._receivers.append((receiver, sender))

    def disconnect(
        self,
        receiver: Callable[..., Any],
        *,
        sender: Any = None,
    ) -> None:
        """Unregister ``receiver``. No-op if it isn't connected."""
        self._receivers = [
            (existing_receiver, existing_sender)
            for existing_receiver, existing_sender in self._receivers
            if not (
                existing_receiver is receiver and existing_sender is sender
            )
        ]

    async def send(self, sender: Any, **kwargs: Any) -> None:
        """Invoke every matching receiver in registration order.

        The first exception aborts the loop and propagates to the
        caller — appropriate for framework defaults whose failure
        should be loud. Use :meth:`send_robust` from observers that
        must not break the producer.
        """
        for receiver, sender_filter in self._receivers:
            if not _sender_matches(sender_filter, sender):
                continue
            result = receiver(sender, **kwargs)
            if inspect.isawaitable(result):
                await result

    async def send_robust(
        self,
        sender: Any,
        **kwargs: Any,
    ) -> list[tuple[Callable[..., Any], BaseException | None]]:
        """Run every matching receiver, collecting exceptions.

        Returns one ``(receiver, error)`` tuple per receiver matched,
        in registration order. ``error`` is ``None`` on success.
        Never raises — suits audit-style observers (logfire spans,
        metric counters) that should never break the producer.
        """
        results: list[tuple[Callable[..., Any], BaseException | None]] = []
        for receiver, sender_filter in self._receivers:
            if not _sender_matches(sender_filter, sender):
                continue
            try:
                outcome = receiver(sender, **kwargs)
                if inspect.isawaitable(outcome):
                    await outcome
            except (Exception, asyncio.CancelledError) as exc:
                results.append((receiver, exc))
            else:
                results.append((receiver, None))
        return results


def _sender_matches(sender_filter: Any, sender: Any) -> bool:
    """Match policy for the ``sender=`` connect-time filter.

    - ``None`` — wildcard, always matches.
    - a class — matches when ``isinstance(sender, sender_filter)``.
    - any other value — matches by identity (``sender is sender_filter``).
    """
    if sender_filter is None:
        return True
    if isinstance(sender_filter, type):
        return isinstance(sender, sender_filter)
    return sender is sender_filter


def receiver(
    signal: Signal,
    *,
    sender: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that connects ``fn`` to ``signal`` at module-load.

    Usage::

        from ballast import message_added, receiver

        @receiver(message_added)
        async def audit_message(sender, *, thread_id, message, **_):
            ...

    Returns the function unchanged so the import side-effect is purely
    the ``signal.connect`` call.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        signal.connect(fn, sender=sender)
        return fn

    return decorator


__all__ = ["Signal", "receiver"]
