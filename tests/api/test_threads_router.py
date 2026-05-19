from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pydantic_ai_stateflow.api.threads import build_threads_router
from pydantic_ai_stateflow.persistence.thread.repository import (
    InMemoryThreadRepository,
)


def _app(repo: InMemoryThreadRepository) -> FastAPI:
    app = FastAPI()
    app.include_router(build_threads_router(thread_repo=repo))
    return app


@pytest.mark.asyncio
async def test_create_thread_201_returns_id():
    repo = InMemoryThreadRepository()
    app = _app(repo)
    tid = uuid4()
    body = {"purpose": "conversation", "purpose_metadata": {}, "actor_id": "alice"}
    with TestClient(app) as c:
        r = c.post("/threads", json=body, headers={"X-Tenant-Id": str(tid)})
    assert r.status_code == 201
    payload = r.json()
    assert "id" in payload
    assert payload["actor_id"] == "alice"
    assert payload["tenant_id"] == str(tid)


@pytest.mark.asyncio
async def test_get_thread_404_when_unknown():
    repo = InMemoryThreadRepository()
    app = _app(repo)
    with TestClient(app) as c:
        r = c.get(f"/threads/{uuid4()}", headers={"X-Tenant-Id": str(uuid4())})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_thread_200_when_owned():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        r = c.get(f"/threads/{th.id}", headers={"X-Tenant-Id": str(tid)})
    assert r.status_code == 200
    assert r.json()["id"] == str(th.id)


@pytest.mark.asyncio
async def test_get_thread_404_cross_tenant():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    other = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        r = c.get(f"/threads/{th.id}", headers={"X-Tenant-Id": str(other)})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_history_returns_messages():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    await repo.add_message(
        th.id, role="user", parts=[{"kind": "text", "text": "hi"}], tenant_id=tid,
    )
    await repo.add_message(
        th.id, role="assistant", parts=[{"kind": "text", "text": "hello"}],
        tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        r = c.get(
            f"/threads/{th.id}/messages", headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 200
    msgs = r.json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_router_respects_prefix():
    repo = InMemoryThreadRepository()
    app = FastAPI()
    app.include_router(build_threads_router(thread_repo=repo, prefix="/api"))
    body = {"purpose": "conversation", "purpose_metadata": {}, "actor_id": "x"}
    with TestClient(app) as c:
        r = c.post("/api/threads", json=body, headers={"X-Tenant-Id": str(uuid4())})
    assert r.status_code == 201


# ── F6: list / rename / archive / unarchive / delete ─────────────────────────


@pytest.mark.asyncio
async def test_list_endpoint_200_respects_include_archived():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    t1 = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    t2 = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    await repo.archive(t1.id, tenant_id=tid)
    app = _app(repo)
    with TestClient(app) as c:
        r = c.get("/threads", headers={"X-Tenant-Id": str(tid)})
        assert r.status_code == 200
        ids = {row["id"] for row in r.json()}
        assert str(t2.id) in ids
        assert str(t1.id) not in ids

        r2 = c.get(
            "/threads?include_archived=true",
            headers={"X-Tenant-Id": str(tid)},
        )
        ids2 = {row["id"] for row in r2.json()}
        assert {str(t1.id), str(t2.id)} <= ids2


@pytest.mark.asyncio
async def test_list_endpoint_tenant_scoped():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    other = uuid4()
    await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        r = c.get("/threads", headers={"X-Tenant-Id": str(other)})
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_patch_thread_sets_title_200():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        r = c.patch(
            f"/threads/{th.id}",
            json={"title": "My session"},
            headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 200
    assert r.json()["title"] == "My session"


@pytest.mark.asyncio
async def test_patch_thread_404_when_unknown():
    repo = InMemoryThreadRepository()
    app = _app(repo)
    with TestClient(app) as c:
        r = c.patch(
            f"/threads/{uuid4()}",
            json={"title": "x"},
            headers={"X-Tenant-Id": str(uuid4())},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_patch_thread_404_cross_tenant():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    other = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        r = c.patch(
            f"/threads/{th.id}",
            json={"title": "x"},
            headers={"X-Tenant-Id": str(other)},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_archive_endpoint_sets_status():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/archive", headers={"X-Tenant-Id": str(tid)},
        )
        assert r.status_code == 200
        assert r.json()["status"] == "archived"
        r2 = c.post(
            f"/threads/{th.id}/unarchive", headers={"X-Tenant-Id": str(tid)},
        )
        assert r2.status_code == 200
        assert r2.json()["status"] == "open"


@pytest.mark.asyncio
async def test_archive_endpoint_404_cross_tenant():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    other = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/archive", headers={"X-Tenant-Id": str(other)},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_endpoint_204_and_idempotent():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        r = c.delete(
            f"/threads/{th.id}", headers={"X-Tenant-Id": str(tid)},
        )
        assert r.status_code == 204
        # idempotent: second delete still 204
        r2 = c.delete(
            f"/threads/{th.id}", headers={"X-Tenant-Id": str(tid)},
        )
        assert r2.status_code == 204
        # thread is gone
        r3 = c.get(
            f"/threads/{th.id}", headers={"X-Tenant-Id": str(tid)},
        )
        assert r3.status_code == 404


@pytest.mark.asyncio
async def test_delete_endpoint_cross_tenant_does_not_remove():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    other = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )
    app = _app(repo)
    with TestClient(app) as c:
        # cross-tenant delete returns 204 (idempotent) but does not remove
        r = c.delete(
            f"/threads/{th.id}", headers={"X-Tenant-Id": str(other)},
        )
        assert r.status_code == 204
        # thread still accessible under its actual tenant
        r2 = c.get(
            f"/threads/{th.id}", headers={"X-Tenant-Id": str(tid)},
        )
        assert r2.status_code == 200
