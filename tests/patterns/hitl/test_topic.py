from __future__ import annotations

from uuid import UUID

from pydantic_ai_stateflow.patterns.hitl.topic import _hitl_topic


def test_topic_format_is_tenant_then_request() -> None:
    tid = UUID("11111111-1111-1111-1111-111111111111")
    rid = UUID("22222222-2222-2222-2222-222222222222")
    assert _hitl_topic(tid, rid) == f"hitl:{tid}:{rid}"


def test_topic_is_string() -> None:
    tid = UUID("11111111-1111-1111-1111-111111111111")
    rid = UUID("22222222-2222-2222-2222-222222222222")
    assert isinstance(_hitl_topic(tid, rid), str)


def test_topic_distinct_per_request() -> None:
    tid = UUID("11111111-1111-1111-1111-111111111111")
    a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    assert _hitl_topic(tid, a) != _hitl_topic(tid, b)


def test_topic_distinct_per_tenant() -> None:
    rid = UUID("22222222-2222-2222-2222-222222222222")
    t1 = UUID("11111111-1111-1111-1111-111111111111")
    t2 = UUID("99999999-9999-9999-9999-999999999999")
    assert _hitl_topic(t1, rid) != _hitl_topic(t2, rid)
