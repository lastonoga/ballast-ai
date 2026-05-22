"""Unit tests for :class:`ballast.events.Signal`."""

from __future__ import annotations

import pytest

from ballast.events import Signal, receiver


@pytest.mark.asyncio
async def test_connect_then_send_fires_receiver() -> None:
    sig = Signal("t")
    seen: list[object] = []

    def fn(sender: object, **kwargs: object) -> None:
        seen.append((sender, kwargs))

    sig.connect(fn)
    await sig.send("S", a=1)
    assert seen == [("S", {"a": 1})]


@pytest.mark.asyncio
async def test_sync_and_async_receivers_both_fire() -> None:
    sig = Signal("t")
    order: list[str] = []

    def sync_fn(_sender: object, **_: object) -> None:
        order.append("sync")

    async def async_fn(_sender: object, **_: object) -> None:
        order.append("async")

    sig.connect(sync_fn)
    sig.connect(async_fn)
    await sig.send(None)
    assert order == ["sync", "async"]


@pytest.mark.asyncio
async def test_send_reraises_first_exception() -> None:
    sig = Signal("t")

    def boom(_sender: object, **_: object) -> None:
        raise RuntimeError("nope")

    sig.connect(boom)
    with pytest.raises(RuntimeError, match="nope"):
        await sig.send(None)


@pytest.mark.asyncio
async def test_send_robust_collects_without_raising() -> None:
    sig = Signal("t")
    ran: list[str] = []

    def good(_sender: object, **_: object) -> None:
        ran.append("good")

    def bad(_sender: object, **_: object) -> None:
        raise ValueError("oops")

    async def good_async(_sender: object, **_: object) -> None:
        ran.append("good_async")

    sig.connect(good)
    sig.connect(bad)
    sig.connect(good_async)
    results = await sig.send_robust(None)
    assert ran == ["good", "good_async"]
    assert len(results) == 3
    assert results[0][1] is None
    assert isinstance(results[1][1], ValueError)
    assert results[2][1] is None


@pytest.mark.asyncio
async def test_sender_filter_class_match() -> None:
    sig = Signal("t")

    class Sender:
        pass

    class Other:
        pass

    fired: list[str] = []

    def only_sender(_s: object, **_: object) -> None:
        fired.append("sender")

    sig.connect(only_sender, sender=Sender)
    await sig.send(Other())
    assert fired == []
    await sig.send(Sender())
    assert fired == ["sender"]


@pytest.mark.asyncio
async def test_sender_filter_identity_match() -> None:
    sig = Signal("t")
    target = object()
    other = object()

    fired: list[str] = []

    def watch(_s: object, **_: object) -> None:
        fired.append("hit")

    sig.connect(watch, sender=target)
    await sig.send(other)
    assert fired == []
    await sig.send(target)
    assert fired == ["hit"]


@pytest.mark.asyncio
async def test_disconnect_removes_receiver() -> None:
    sig = Signal("t")
    fired: list[int] = []

    def fn(_s: object, **_: object) -> None:
        fired.append(1)

    sig.connect(fn)
    sig.disconnect(fn)
    await sig.send(None)
    assert fired == []


def test_disconnect_unknown_is_noop() -> None:
    sig = Signal("t")

    def fn(_s: object, **_: object) -> None:
        pass

    sig.disconnect(fn)  # never connected — must not raise


def test_connect_is_idempotent() -> None:
    sig = Signal("t")

    def fn(_s: object, **_: object) -> None:
        pass

    sig.connect(fn)
    sig.connect(fn)
    sig.connect(fn)
    assert len(sig._receivers) == 1


def test_connect_distinct_senders_not_deduped() -> None:
    sig = Signal("t")

    class A:
        pass

    class B:
        pass

    def fn(_s: object, **_: object) -> None:
        pass

    sig.connect(fn, sender=A)
    sig.connect(fn, sender=B)
    assert len(sig._receivers) == 2


@pytest.mark.asyncio
async def test_receiver_decorator_form() -> None:
    sig = Signal("t")
    fired: list[str] = []

    @receiver(sig)
    async def handler(_sender: object, **_: object) -> None:
        fired.append("ok")

    await sig.send(None)
    assert fired == ["ok"]


def test_signal_requires_non_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Signal("")
