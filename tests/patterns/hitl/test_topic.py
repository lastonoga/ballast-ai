from __future__ import annotations

from uuid import UUID

from ballast.patterns.hitl.topic import _hitl_topic


def test_topic_format() -> None:
    rid = UUID("22222222-2222-2222-2222-222222222222")
    assert _hitl_topic(rid) == f"hitl:{rid}"


def test_topic_is_string() -> None:
    rid = UUID("22222222-2222-2222-2222-222222222222")
    assert isinstance(_hitl_topic(rid), str)


def test_topic_distinct_per_request() -> None:
    a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    assert _hitl_topic(a) != _hitl_topic(b)
