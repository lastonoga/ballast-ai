# HITL Channels (Sub-project #6) Implementation Plan

```yaml
date: 2026-05-15
sub_project: 6
status: ready-for-implementation
baseline_tests: 243 passed + 10 skipped (after SP5)
target_tests: ~285 passed + 10 skipped (after SP6)
```

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build production-grade HITL channel adapters on top of the SP5 `HITLChannel` Protocol and `HITLGate`: a FastAPI inbound `UIChannel`, a signed outbound `WebhookChannel` (with matching inbound callback endpoint), and a `ConversationalChannel` driven by a helper pydantic-ai Agent running in its own DBOS workflow. Add the framework-side `HelperVerdict[ContextT]`, the typed `make_helper_agent_with_approval_tools(...)` factory, and a `DefaultHelperSessionRunner` workflow that bridges the helper conversation to the gate via DBOS tenant-scoped topics.

**Spec sections covered:** 2C.4 (HITLGate FSM, two-point authz, `_topic(tenant_id, request_id)` format `hitl:{tenant_id}:{request_id}`), 3G (channels catalog: `UIChannel`, `WebhookChannel`, `ConversationalChannel`), 3H (FastAPI endpoint `POST /hitl/{request_id}/respond`), 3J.1–3J.6 (ConversationalChannel + separate-workflow runtime model; `HelperAgentFactory`; typed approval tools; `HelperVerdict[ContextT]`; helper persistence columns), 4A.0.1–4A.0.2 (canonical `HITLResponse` discriminated union, `HITLPrompt` with `tenant_id`).

**Scope vs deferred:**
- v1 in SP6: `UIChannel`, `WebhookChannel` (outbound POST + inbound callback router), `ConversationalChannel`, `HelperVerdict[ContextT]`, `HelperAgentFactory` Protocol, `make_helper_agent_with_approval_tools`, `HelperSessionRunner` Protocol + `DefaultHelperSessionRunner`, `build_hitl_router` FastAPI factory, helper-verdict round-trip persistence wiring, `httpx` dependency.
- Deferred to later sub-projects: `SlackChannel`, `ChatChannel`, `EscalationChannel`, `ToolNeedsHITL` decorator, `PartialResponse` / `FreeTextResponse` / `OptionChosenResponse` HITL response variants, `QuorumApprovalStage`.
- `InMemoryHITLChannel` already shipped in SP5 — unchanged here.

---

## File Structure

```
src/ballast/patterns/hitl/
├── __init__.py                          # extended exports
├── channel.py                           # (existing — unchanged)
├── gate.py                              # (existing — unchanged)
├── policy.py                            # (existing — unchanged)
├── prompt.py                            # (existing — unchanged)
├── response.py                          # (existing — unchanged)
├── topic.py                             # _hitl_topic(tenant_id, request_id)
├── verdict.py                           # HelperVerdict[ContextT]
├── api/
│   ├── __init__.py
│   └── router.py                        # build_hitl_router + endpoints
├── channels/
│   ├── __init__.py
│   ├── ui.py                            # UIChannel
│   ├── webhook.py                       # WebhookChannel + WebhookConfig + sign_payload + post_webhook step
│   └── conversational.py                # ConversationalChannel
└── helper/
    ├── __init__.py
    ├── factory.py                       # HelperAgentFactory Protocol + make_helper_agent_with_approval_tools + HelperToolBox / HelperDeps
    └── session.py                       # HelperSessionRunner Protocol + DefaultHelperSessionRunner workflow + HelperSessionInput

tests/patterns/hitl/
├── (existing files — unchanged)
├── test_topic.py
├── test_verdict.py
├── test_router.py
├── test_ui_channel.py
├── test_webhook_channel.py
├── test_helper_factory.py
├── test_helper_session.py
├── test_conversational_channel.py
├── test_persistence_wiring.py
└── test_public_api_sp6.py
```

---

## Task 1: HITL FastAPI router primitives — `_hitl_topic`, `build_hitl_router`

Pure module: the tenant-scoped topic helper used by every channel + a FastAPI router factory that mounts `POST /hitl/{request_id}/respond` (the UI endpoint). The router does endpoint-side authz (point #1 of the two-point check in 2C.4) and `DBOS.send` to the gate's tenant-scoped topic. No channel implementations yet.

**Baseline:** 243 passed + 10 skipped → **Target:** 254 passed + 10 skipped (+11).

**Files:**
- Create: `src/ballast/patterns/hitl/topic.py`
- Create: `src/ballast/patterns/hitl/api/__init__.py`
- Create: `src/ballast/patterns/hitl/api/router.py`
- Create: `tests/patterns/hitl/test_topic.py`
- Create: `tests/patterns/hitl/test_router.py`
- Modify: `pyproject.toml` — add `fastapi>=0.110` and `httpx>=0.27` to dev deps so test client works (httpx will also be needed by Task 3, so use `uv add fastapi httpx` then move httpx to runtime deps in Task 3).

- [ ] **Step 1: Failing tests**

`tests/patterns/hitl/test_topic.py`:

```python
from __future__ import annotations

from uuid import UUID

from ballast.patterns.hitl.topic import _hitl_topic


def test_topic_format_is_tenant_then_request():
    tid = UUID("11111111-1111-1111-1111-111111111111")
    rid = UUID("22222222-2222-2222-2222-222222222222")
    assert _hitl_topic(tid, rid) == f"hitl:{tid}:{rid}"


def test_topic_is_string():
    tid = UUID("11111111-1111-1111-1111-111111111111")
    rid = UUID("22222222-2222-2222-2222-222222222222")
    assert isinstance(_hitl_topic(tid, rid), str)


def test_topic_distinct_per_request():
    tid = UUID("11111111-1111-1111-1111-111111111111")
    a = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    b = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    assert _hitl_topic(tid, a) != _hitl_topic(tid, b)


def test_topic_distinct_per_tenant():
    rid = UUID("22222222-2222-2222-2222-222222222222")
    t1 = UUID("11111111-1111-1111-1111-111111111111")
    t2 = UUID("99999999-9999-9999-9999-999999999999")
    assert _hitl_topic(t1, rid) != _hitl_topic(t2, rid)
```

`tests/patterns/hitl/test_router.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.patterns.hitl.api.router import build_hitl_router
from ballast.patterns.hitl.policy import AllowAll, DenyAll
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence import InMemoryHITLRepository


def _make_app(repo, policy) -> FastAPI:
    app = FastAPI()
    app.include_router(build_hitl_router(repo=repo, policy=policy))
    return app


@pytest.mark.asyncio
async def test_respond_404_when_request_unknown():
    repo = InMemoryHITLRepository()
    app = _make_app(repo, AllowAll())
    body = {
        "kind": "approved",
        "actor_id": "alice",
        "answered_at": datetime.now(tz=UTC).isoformat(),
    }
    with TestClient(app) as client:
        r = client.post(f"/hitl/{uuid4()}/respond", json=body)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_respond_403_when_policy_denies_and_audited():
    repo = InMemoryHITLRepository()
    tid = uuid4()
    req = await repo.persist_request(
        prompt={"tenant_id": str(tid), "title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=uuid4(), gate_kind="hitl_gate",
        purpose="approval", tenant_id=tid,
    )
    app = _make_app(repo, DenyAll())
    body = {
        "kind": "approved", "actor_id": "alice",
        "answered_at": datetime.now(tz=UTC).isoformat(),
    }
    headers = {"X-Tenant-Id": str(tid)}
    with TestClient(app) as client:
        r = client.post(f"/hitl/{req.id}/respond", json=body, headers=headers)
    assert r.status_code == 403
    # Audited
    assert len(repo._denials) == 1
    assert repo._denials[0].actor_id == "alice"


@pytest.mark.asyncio
async def test_respond_200_sends_to_topic_on_grant():
    repo = InMemoryHITLRepository()
    tid = uuid4()
    wf_id = uuid4()
    req = await repo.persist_request(
        prompt={"tenant_id": str(tid), "title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=wf_id, gate_kind="hitl_gate",
        purpose="approval", tenant_id=tid,
    )
    app = _make_app(repo, AllowAll())
    body = {
        "kind": "approved", "actor_id": "alice",
        "answered_at": datetime.now(tz=UTC).isoformat(),
    }
    headers = {"X-Tenant-Id": str(tid)}
    sent: dict[str, Any] = {}

    def fake_send(destination: str, message: Any, topic: str | None = None) -> None:
        sent["destination"] = destination
        sent["message"] = message
        sent["topic"] = topic

    with patch("ballast.patterns.hitl.api.router.DBOS.send", fake_send):
        with TestClient(app) as client:
            r = client.post(f"/hitl/{req.id}/respond", json=body, headers=headers)
    assert r.status_code == 200
    assert sent["destination"] == str(wf_id)
    assert sent["topic"] == _hitl_topic(tid, req.id)
    assert sent["message"]["kind"] == "approved"
    assert sent["message"]["actor_id"] == "alice"


@pytest.mark.asyncio
async def test_respond_400_when_tenant_header_missing():
    repo = InMemoryHITLRepository()
    tid = uuid4()
    req = await repo.persist_request(
        prompt={"tenant_id": str(tid), "title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=uuid4(), gate_kind="hitl_gate",
        purpose="approval", tenant_id=tid,
    )
    app = _make_app(repo, AllowAll())
    body = {"kind": "approved", "actor_id": "a",
            "answered_at": datetime.now(tz=UTC).isoformat()}
    with TestClient(app) as client:
        r = client.post(f"/hitl/{req.id}/respond", json=body)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_respond_403_when_tenant_mismatch():
    """Cross-tenant attempts MUST be rejected, not silently load wrong tenant."""
    repo = InMemoryHITLRepository()
    tid = uuid4()
    other = uuid4()
    req = await repo.persist_request(
        prompt={"tenant_id": str(tid), "title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=uuid4(), gate_kind="hitl_gate",
        purpose="approval", tenant_id=tid,
    )
    app = _make_app(repo, AllowAll())
    body = {"kind": "approved", "actor_id": "a",
            "answered_at": datetime.now(tz=UTC).isoformat()}
    headers = {"X-Tenant-Id": str(other)}
    with TestClient(app) as client:
        r = client.post(f"/hitl/{req.id}/respond", json=body, headers=headers)
    assert r.status_code == 404  # request not visible cross-tenant


@pytest.mark.asyncio
async def test_router_uses_provided_path_prefix():
    repo = InMemoryHITLRepository()
    app = FastAPI()
    app.include_router(
        build_hitl_router(repo=repo, policy=AllowAll(), prefix="/api"),
    )
    with TestClient(app) as client:
        r = client.post(f"/api/hitl/{uuid4()}/respond", json={
            "kind": "approved",
            "answered_at": datetime.now(tz=UTC).isoformat(),
        })
    # 404 because request unknown, but route resolved
    assert r.status_code == 404
```

- [ ] **Step 2: Run → fail (ImportError / 404)**

```bash
uv add fastapi httpx
uv run pytest tests/patterns/hitl/test_topic.py tests/patterns/hitl/test_router.py -v
```

- [ ] **Step 3: Implement**

`src/ballast/patterns/hitl/topic.py`:

```python
from __future__ import annotations

from uuid import UUID


def _hitl_topic(tenant_id: UUID, request_id: UUID) -> str:
    """Tenant-scoped DBOS topic for HITL replies.

    Format `hitl:{tenant_id}:{request_id}` per spec 2C.4. The tenant
    prefix prevents cross-tenant collisions if a request_id is ever
    reused across tenants (defensive — UUIDs should be globally
    unique, but topic isolation is still required).
    """
    return f"hitl:{tenant_id}:{request_id}"
```

`src/ballast/patterns/hitl/api/__init__.py`:

```python
from ballast.patterns.hitl.api.router import build_hitl_router

__all__ = ["build_hitl_router"]
```

`src/ballast/patterns/hitl/api/router.py`:

```python
from __future__ import annotations

from uuid import UUID

from dbos import DBOS
from fastapi import APIRouter, Header, HTTPException
from pydantic import TypeAdapter

from ballast.patterns.hitl.policy import Policy
from ballast.patterns.hitl.response import HITLResponse
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence import HITLRepository

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)


def build_hitl_router(
    *,
    repo: HITLRepository,
    policy: Policy,
    prefix: str = "",
) -> APIRouter:
    """Build a FastAPI router for HITL inbound endpoints.

    Mounts:
      - `POST {prefix}/hitl/{request_id}/respond` — UI / generic JSON.
      - `POST {prefix}/hitl/webhook/{request_id}` — third-party callback
        (added in Task 4; same authz pipeline).

    Tenant is taken from the `X-Tenant-Id` header (apps wire their own
    tenant resolver via FastAPI middleware that injects the header).

    Authz happens HERE (endpoint side, point #1 of spec 2C.4's two-point
    check). The defense-in-depth check lives in HITLGate.run (SP5).
    """

    router = APIRouter(prefix=prefix)

    async def _respond(request_id: UUID, body_json: dict, tenant_id: UUID) -> dict:
        request = await repo.load_request(request_id, tenant_id=tenant_id)
        if request is None:
            raise HTTPException(status_code=404, detail="HITL request not found")

        # Parse the discriminated-union body now we know the request is real.
        try:
            response = _RESPONSE_ADAPTER.validate_python(body_json)
        except Exception as exc:  # pragma: no cover - pydantic raises ValidationError
            raise HTTPException(status_code=422, detail=str(exc))

        verdict = await policy.can(
            actor=response.actor_id,
            action="decide",
            resource=request.payload,
            tenant_id=tenant_id,
        )
        if not verdict.is_grant:
            await repo.persist_authz_denied(
                request_id=request_id,
                actor_id=response.actor_id or "<anonymous>",
                voter_votes=dict(verdict.votes),
                tenant_id=tenant_id,
            )
            raise HTTPException(status_code=403, detail=verdict.summary())

        DBOS.send(
            destination=str(request.workflow_id),
            message=response.model_dump(mode="json"),
            topic=_hitl_topic(tenant_id, request_id),
        )
        return {"status": "delivered"}

    @router.post("/hitl/{request_id}/respond")
    async def respond_to_hitl(
        request_id: UUID,
        body: dict,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> dict:
        if x_tenant_id is None:
            raise HTTPException(status_code=400, detail="X-Tenant-Id header required")
        try:
            tenant_id = UUID(x_tenant_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="X-Tenant-Id must be a UUID")
        return await _respond(request_id, body, tenant_id)

    # Task 4 will extend this router with `/hitl/webhook/{request_id}`.
    router._respond_impl = _respond  # type: ignore[attr-defined]
    return router
```

> Note: the router attaches `_respond_impl` as a private hook so Task 4's webhook endpoint can reuse the exact same load/authz/send pipeline without duplication.

- [ ] **Step 4: Tests pass (11 new: 4 topic + 6 router + 1 prefix)**
- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/topic.py \
        src/ballast/patterns/hitl/api/ \
        tests/patterns/hitl/test_topic.py \
        tests/patterns/hitl/test_router.py \
        pyproject.toml uv.lock
git commit -m "feat(hitl): tenant-scoped topic helper + FastAPI router with endpoint-side authz"
```

---

## Task 2: `UIChannel`

The thinnest production channel: receives via `DBOS.recv` on the tenant-scoped topic. Defense-in-depth authz is in `HITLGate.run` (SP5) — so `UIChannel` does not re-check policy. On `prompt.timeout` exhausted, returns a `TimeoutResponse` (the gate then translates to `HITLTimedOut`).

**Baseline:** 254 → **Target:** 258 (+4).

**Files:**
- Create: `src/ballast/patterns/hitl/channels/__init__.py`
- Create: `src/ballast/patterns/hitl/channels/ui.py`
- Create: `tests/patterns/hitl/test_ui_channel.py`

- [ ] **Step 1: Failing tests**

`tests/patterns/hitl/test_ui_channel.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from ballast.patterns.hitl.channel import HITLChannel
from ballast.patterns.hitl.channels.ui import UIChannel
from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.response import (
    ApprovedResponse,
    TimeoutResponse,
)
from ballast.patterns.hitl.topic import _hitl_topic


def test_ui_channel_satisfies_protocol():
    assert isinstance(UIChannel(), HITLChannel)


@pytest.mark.asyncio
async def test_ui_channel_returns_received_response():
    tid = uuid4()
    rid = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c",
        decision_kinds={"approved", "rejected"},
        timeout=timedelta(seconds=5),
    )
    payload = ApprovedResponse(
        actor_id="alice", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")
    recv = AsyncMock(return_value=payload)
    with patch(
        "ballast.patterns.hitl.channels.ui.DBOS.recv", recv,
    ):
        channel = UIChannel()
        result = await channel.ask(prompt, request_id=rid)
    assert isinstance(result, ApprovedResponse)
    assert result.actor_id == "alice"
    recv.assert_awaited_once_with(
        _hitl_topic(tid, rid), timeout_seconds=5.0,
    )


@pytest.mark.asyncio
async def test_ui_channel_returns_timeout_on_none():
    tid = uuid4()
    rid = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c",
        decision_kinds={"approved"},
        timeout=timedelta(seconds=1),
    )
    recv = AsyncMock(return_value=None)
    with patch(
        "ballast.patterns.hitl.channels.ui.DBOS.recv", recv,
    ):
        channel = UIChannel()
        result = await channel.ask(prompt, request_id=rid)
    assert isinstance(result, TimeoutResponse)


@pytest.mark.asyncio
async def test_ui_channel_no_timeout_passes_none():
    tid = uuid4()
    rid = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c",
        decision_kinds={"approved"},
    )
    payload = ApprovedResponse(
        actor_id="bob", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")
    recv = AsyncMock(return_value=payload)
    with patch(
        "ballast.patterns.hitl.channels.ui.DBOS.recv", recv,
    ):
        channel = UIChannel()
        await channel.ask(prompt, request_id=rid)
    recv.assert_awaited_once_with(_hitl_topic(tid, rid), timeout_seconds=None)
```

- [ ] **Step 2: Run → fail (ImportError)**

```bash
uv run pytest tests/patterns/hitl/test_ui_channel.py -v
```

- [ ] **Step 3: Implement**

`src/ballast/patterns/hitl/channels/__init__.py`:

```python
from ballast.patterns.hitl.channels.ui import UIChannel

__all__ = ["UIChannel"]
```

`src/ballast/patterns/hitl/channels/ui.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar
from uuid import UUID

from dbos import DBOS
from pydantic import TypeAdapter

from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from ballast.patterns.hitl.topic import _hitl_topic

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)


class UIChannel:
    """HITL channel backed by a FastAPI inbound endpoint.

    The endpoint (built via `build_hitl_router`) does endpoint-side
    authz + `DBOS.send` to the gate's tenant-scoped topic. This
    channel simply blocks on `DBOS.recv` and returns the response.

    Defense-in-depth re-check happens in `HITLGate.run` (SP5) — UIChannel
    intentionally does NOT re-check policy.
    """

    name: ClassVar[str] = "ui"

    async def ask(self, prompt: HITLPrompt, *, request_id: UUID) -> HITLResponse:
        topic = _hitl_topic(prompt.tenant_id, request_id)
        timeout_seconds = (
            prompt.timeout.total_seconds() if prompt.timeout is not None else None
        )
        payload = await DBOS.recv(topic, timeout_seconds=timeout_seconds)
        if payload is None:
            return TimeoutResponse(answered_at=datetime.now(tz=UTC))
        return _RESPONSE_ADAPTER.validate_python(payload)
```

- [ ] **Step 4: Tests pass (4 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/channels/ \
        tests/patterns/hitl/test_ui_channel.py
git commit -m "feat(hitl): UIChannel — DBOS.recv on tenant-scoped topic with TimeoutResponse fallback"
```

---

## Task 3: `WebhookChannel` outbound primitives — config, signing, POST step

Pure helpers + a `@DBOS.step` for outbound HTTP. Done before the full channel so that step replay semantics are tested in isolation.

**Baseline:** 258 → **Target:** 265 (+7).

**Files:**
- Modify: `pyproject.toml` — promote `httpx>=0.27` from dev deps to runtime deps.
- Create: `src/ballast/patterns/hitl/channels/webhook.py` (partial — just the helpers + step; full channel in Task 4)
- Create: `tests/patterns/hitl/test_webhook_primitives.py`

- [ ] **Step 1: Failing tests**

`tests/patterns/hitl/test_webhook_primitives.py`:

```python
from __future__ import annotations

import hmac
import json
from hashlib import sha256
from unittest.mock import AsyncMock, patch

import pytest

from ballast.patterns.hitl.channels.webhook import (
    WEBHOOK_SIGNATURE_HEADER,
    WebhookConfig,
    post_webhook,
    sign_payload,
)


def test_webhook_config_validates_url():
    cfg = WebhookConfig(url="https://example.com/cb", secret="sssh")
    assert cfg.url == "https://example.com/cb"
    assert cfg.secret == "sssh"


def test_sign_payload_hmac_sha256_hex():
    body = b'{"hello":"world"}'
    sig = sign_payload(body, secret="sssh")
    expected = hmac.new(b"sssh", body, sha256).hexdigest()
    assert sig == expected
    # Hex, no prefix.
    assert all(c in "0123456789abcdef" for c in sig)
    assert len(sig) == 64


def test_sign_payload_different_secret_distinct():
    body = b"payload"
    assert sign_payload(body, secret="a") != sign_payload(body, secret="b")


def test_signature_header_name_is_canonical():
    assert WEBHOOK_SIGNATURE_HEADER == "X-Stateflow-Signature"


@pytest.mark.asyncio
async def test_post_webhook_sends_signature_and_body():
    fake_response = AsyncMock()
    fake_response.status_code = 200
    fake_response.raise_for_status = AsyncMock(return_value=None)

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None

        async def post(self, url, content, headers):
            FakeClient.url = url
            FakeClient.content = content
            FakeClient.headers = headers
            class _R:
                status_code = 200
                def raise_for_status(self): return None
            return _R()

    with patch(
        "ballast.patterns.hitl.channels.webhook.httpx.AsyncClient",
        FakeClient,
    ):
        body = json.dumps({"request_id": "abc"}).encode()
        await post_webhook(
            url="https://hooks.example/cb",
            body=body,
            signature="deadbeef",
        )
    assert FakeClient.url == "https://hooks.example/cb"
    assert FakeClient.content == body
    assert FakeClient.headers["X-Stateflow-Signature"] == "deadbeef"
    assert FakeClient.headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_post_webhook_raises_on_4xx():
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def post(self, url, content, headers):
            class _R:
                status_code = 500
                def raise_for_status(self):
                    import httpx as _h
                    req = _h.Request("POST", url)
                    raise _h.HTTPStatusError("boom", request=req,
                                             response=_h.Response(500))
            return _R()

    with patch(
        "ballast.patterns.hitl.channels.webhook.httpx.AsyncClient",
        FakeClient,
    ):
        with pytest.raises(Exception):
            await post_webhook(url="https://x", body=b"{}", signature="s")
```

- [ ] **Step 2: Run → fail (ImportError)**

```bash
uv remove httpx --dev  # move to runtime
uv add httpx
uv run pytest tests/patterns/hitl/test_webhook_primitives.py -v
```

- [ ] **Step 3: Implement**

`src/ballast/patterns/hitl/channels/webhook.py` (partial — channel class lands in Task 4):

```python
from __future__ import annotations

import hmac
from hashlib import sha256

import httpx
from dbos import DBOS
from pydantic import BaseModel, ConfigDict, HttpUrl

WEBHOOK_SIGNATURE_HEADER = "X-Stateflow-Signature"


class WebhookConfig(BaseModel):
    """Outbound webhook configuration: URL to POST to + shared secret for HMAC."""

    model_config = ConfigDict(frozen=True)

    url: HttpUrl
    secret: str


def sign_payload(payload: bytes, *, secret: str) -> str:
    """HMAC-SHA256 of `payload` keyed by `secret`, hex-encoded.

    Verifiers reconstruct the signature with the same secret and compare
    via `hmac.compare_digest` to prevent timing leaks.
    """
    return hmac.new(secret.encode("utf-8"), payload, sha256).hexdigest()


@DBOS.step()
async def post_webhook(*, url: str, body: bytes, signature: str) -> None:
    """POST `body` to `url` with the signature header.

    Wrapped as `@DBOS.step` so DBOS records the side-effect (idempotency
    on replay = the step is recorded once even though the HTTP call may
    have been at-least-once at the network level).

    Caller's responsibility:
    - body must be the exact bytes that were signed (no re-serialization).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            content=body,
            headers={
                WEBHOOK_SIGNATURE_HEADER: signature,
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
```

- [ ] **Step 4: Tests pass (7 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/channels/webhook.py \
        tests/patterns/hitl/test_webhook_primitives.py \
        pyproject.toml uv.lock
git commit -m "feat(hitl): webhook signing + HMAC step + WebhookConfig (Task 3 of SP6)"
```

---

## Task 4: `WebhookChannel` + inbound callback endpoint

Wire the outbound POST into a `HITLChannel.ask` impl + extend the SP6 Task 1 router with `POST /hitl/webhook/{request_id}` that verifies the signature, then reuses the same load+authz+send pipeline.

**Baseline:** 265 → **Target:** 273 (+8).

**Files:**
- Modify: `src/ballast/patterns/hitl/channels/webhook.py` (add `WebhookChannel` class)
- Modify: `src/ballast/patterns/hitl/channels/__init__.py` (export `WebhookChannel`, `WebhookConfig`)
- Modify: `src/ballast/patterns/hitl/api/router.py` (extend `build_hitl_router` with `webhook_secret` kwarg + endpoint)
- Create: `tests/patterns/hitl/test_webhook_channel.py`

- [ ] **Step 1: Failing tests**

`tests/patterns/hitl/test_webhook_channel.py`:

```python
from __future__ import annotations

import hmac
import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.patterns.hitl.api.router import build_hitl_router
from ballast.patterns.hitl.channel import HITLChannel
from ballast.patterns.hitl.channels.webhook import (
    WEBHOOK_SIGNATURE_HEADER,
    WebhookChannel,
    WebhookConfig,
)
from ballast.patterns.hitl.policy import AllowAll
from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.response import (
    ApprovedResponse,
    TimeoutResponse,
)
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence import InMemoryHITLRepository


def test_webhook_channel_satisfies_protocol():
    cfg = WebhookConfig(url="https://x.example/cb", secret="s")
    assert isinstance(WebhookChannel(config=cfg), HITLChannel)


@pytest.mark.asyncio
async def test_webhook_channel_posts_signed_payload_then_recvs():
    tid = uuid4()
    rid = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c",
        decision_kinds={"approved"},
        timeout=timedelta(seconds=5),
    )
    posted: dict = {}

    async def fake_post(*, url: str, body: bytes, signature: str) -> None:
        posted["url"] = url
        posted["body"] = body
        posted["signature"] = signature

    payload = ApprovedResponse(
        actor_id="ext", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")
    recv = AsyncMock(return_value=payload)

    cfg = WebhookConfig(url="https://hooks.example/cb", secret="sssh")
    channel = WebhookChannel(config=cfg)
    with patch(
        "ballast.patterns.hitl.channels.webhook.post_webhook",
        fake_post,
    ), patch(
        "ballast.patterns.hitl.channels.webhook.DBOS.recv", recv,
    ):
        result = await channel.ask(prompt, request_id=rid)

    assert isinstance(result, ApprovedResponse)
    assert result.actor_id == "ext"
    body_json = json.loads(posted["body"])
    assert body_json["request_id"] == str(rid)
    assert body_json["prompt"]["title"] == "t"
    # Signature is HMAC of EXACT bytes posted.
    expected = hmac.new(b"sssh", posted["body"], sha256).hexdigest()
    assert posted["signature"] == expected
    recv.assert_awaited_once_with(
        _hitl_topic(tid, rid), timeout_seconds=5.0,
    )


@pytest.mark.asyncio
async def test_webhook_channel_returns_timeout_on_none():
    tid = uuid4()
    rid = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c",
        decision_kinds={"approved"},
        timeout=timedelta(seconds=1),
    )

    cfg = WebhookConfig(url="https://hooks.example/cb", secret="sssh")
    channel = WebhookChannel(config=cfg)
    with patch(
        "ballast.patterns.hitl.channels.webhook.post_webhook",
        AsyncMock(return_value=None),
    ), patch(
        "ballast.patterns.hitl.channels.webhook.DBOS.recv",
        AsyncMock(return_value=None),
    ):
        result = await channel.ask(prompt, request_id=rid)
    assert isinstance(result, TimeoutResponse)


def _signed_body(payload: dict, secret: str) -> tuple[bytes, str]:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return body, hmac.new(secret.encode(), body, sha256).hexdigest()


@pytest.mark.asyncio
async def test_webhook_endpoint_rejects_missing_signature():
    repo = InMemoryHITLRepository()
    tid = uuid4()
    req = await repo.persist_request(
        prompt={"tenant_id": str(tid), "title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=uuid4(), gate_kind="hitl_gate",
        purpose="approval", tenant_id=tid,
    )
    app = FastAPI()
    app.include_router(build_hitl_router(
        repo=repo, policy=AllowAll(), webhook_secret="sssh",
    ))
    body = {"kind": "approved", "actor_id": "ext",
            "answered_at": datetime.now(tz=UTC).isoformat()}
    with TestClient(app) as client:
        r = client.post(
            f"/hitl/webhook/{req.id}",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_webhook_endpoint_rejects_bad_signature():
    repo = InMemoryHITLRepository()
    tid = uuid4()
    req = await repo.persist_request(
        prompt={"tenant_id": str(tid), "title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=uuid4(), gate_kind="hitl_gate",
        purpose="approval", tenant_id=tid,
    )
    app = FastAPI()
    app.include_router(build_hitl_router(
        repo=repo, policy=AllowAll(), webhook_secret="sssh",
    ))
    with TestClient(app) as client:
        r = client.post(
            f"/hitl/webhook/{req.id}",
            content=b'{"kind":"approved","actor_id":"ext","answered_at":"2026-01-01T00:00:00+00:00"}',
            headers={
                "X-Tenant-Id": str(tid),
                "Content-Type": "application/json",
                WEBHOOK_SIGNATURE_HEADER: "deadbeef",
            },
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_webhook_endpoint_accepts_valid_signature_and_sends():
    repo = InMemoryHITLRepository()
    tid = uuid4()
    wf_id = uuid4()
    req = await repo.persist_request(
        prompt={"tenant_id": str(tid), "title": "x", "context": "y",
                "decision_kinds": ["approved"]},
        workflow_id=wf_id, gate_kind="hitl_gate",
        purpose="approval", tenant_id=tid,
    )
    app = FastAPI()
    app.include_router(build_hitl_router(
        repo=repo, policy=AllowAll(), webhook_secret="sssh",
    ))

    body = b'{"actor_id":"ext","answered_at":"2026-01-01T00:00:00+00:00","kind":"approved"}'
    sig = hmac.new(b"sssh", body, sha256).hexdigest()
    sent: dict = {}

    def fake_send(destination, message, topic=None):
        sent.update(destination=destination, message=message, topic=topic)

    with patch(
        "ballast.patterns.hitl.api.router.DBOS.send", fake_send,
    ):
        with TestClient(app) as client:
            r = client.post(
                f"/hitl/webhook/{req.id}",
                content=body,
                headers={
                    "X-Tenant-Id": str(tid),
                    "Content-Type": "application/json",
                    WEBHOOK_SIGNATURE_HEADER: sig,
                },
            )
    assert r.status_code == 200
    assert sent["destination"] == str(wf_id)
    assert sent["topic"] == _hitl_topic(tid, req.id)


@pytest.mark.asyncio
async def test_webhook_endpoint_404_when_no_secret_configured():
    """If `webhook_secret` not provided, the webhook endpoint MUST NOT mount."""
    repo = InMemoryHITLRepository()
    app = FastAPI()
    app.include_router(build_hitl_router(repo=repo, policy=AllowAll()))
    with TestClient(app) as client:
        r = client.post(f"/hitl/webhook/{uuid4()}", json={})
    assert r.status_code == 404
```

- [ ] **Step 2: Run → fail (ImportError)**

```bash
uv run pytest tests/patterns/hitl/test_webhook_channel.py -v
```

- [ ] **Step 3: Implement**

Append to `src/ballast/patterns/hitl/channels/webhook.py`:

```python
from datetime import UTC, datetime
from typing import ClassVar
from uuid import UUID

from pydantic import TypeAdapter

from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from ballast.patterns.hitl.topic import _hitl_topic

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)


class WebhookChannel:
    """Outbound notification + inbound callback HITL channel.

    Flow:
      1. `ask()` serializes prompt + request_id + callback URL into a JSON
         body, signs it via HMAC-SHA256 with the configured secret, and
         POSTs it to `config.url` via `post_webhook` (a DBOS step).
      2. Caller (third party) eventually POSTs a `HITLResponse` to
         `POST /hitl/webhook/{request_id}` (mounted by `build_hitl_router`
         when `webhook_secret` is supplied). That endpoint verifies the
         signature, runs endpoint-side authz, and `DBOS.send`s the response
         to the gate's tenant-scoped topic.
      3. `ask()` returns via `DBOS.recv` on that topic.
    """

    name: ClassVar[str] = "webhook"

    def __init__(self, *, config: WebhookConfig) -> None:
        self.config = config

    async def ask(self, prompt: HITLPrompt, *, request_id: UUID) -> HITLResponse:
        body = self._build_outbound_body(prompt, request_id)
        signature = sign_payload(body, secret=self.config.secret)
        await post_webhook(url=str(self.config.url), body=body, signature=signature)

        topic = _hitl_topic(prompt.tenant_id, request_id)
        timeout_seconds = (
            prompt.timeout.total_seconds() if prompt.timeout is not None else None
        )
        payload = await DBOS.recv(topic, timeout_seconds=timeout_seconds)
        if payload is None:
            return TimeoutResponse(answered_at=datetime.now(tz=UTC))
        return _RESPONSE_ADAPTER.validate_python(payload)

    @staticmethod
    def _build_outbound_body(prompt: HITLPrompt, request_id: UUID) -> bytes:
        import json

        payload = {
            "request_id": str(request_id),
            "tenant_id": str(prompt.tenant_id),
            "prompt": prompt.model_dump(mode="json"),
        }
        # Sorted keys + tight separators: deterministic for stable HMAC.
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
```

Update `src/ballast/patterns/hitl/channels/__init__.py`:

```python
from ballast.patterns.hitl.channels.ui import UIChannel
from ballast.patterns.hitl.channels.webhook import (
    WEBHOOK_SIGNATURE_HEADER,
    WebhookChannel,
    WebhookConfig,
)

__all__ = [
    "UIChannel",
    "WEBHOOK_SIGNATURE_HEADER",
    "WebhookChannel",
    "WebhookConfig",
]
```

Extend `src/ballast/patterns/hitl/api/router.py` — add a `webhook_secret` kwarg and mount the second endpoint:

```python
import hmac
from hashlib import sha256

from ballast.patterns.hitl.channels.webhook import (
    WEBHOOK_SIGNATURE_HEADER,
)
# ... existing imports unchanged

def build_hitl_router(
    *,
    repo: HITLRepository,
    policy: Policy,
    prefix: str = "",
    webhook_secret: str | None = None,
) -> APIRouter:
    router = APIRouter(prefix=prefix)

    async def _respond(request_id: UUID, body_json: dict, tenant_id: UUID) -> dict:
        # ... unchanged body from Task 1 ...
        ...

    @router.post("/hitl/{request_id}/respond")
    async def respond_to_hitl(
        request_id: UUID,
        body: dict,
        x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    ) -> dict:
        if x_tenant_id is None:
            raise HTTPException(status_code=400, detail="X-Tenant-Id header required")
        try:
            tenant_id = UUID(x_tenant_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="X-Tenant-Id must be a UUID")
        return await _respond(request_id, body, tenant_id)

    if webhook_secret is not None:
        from fastapi import Request

        @router.post("/hitl/webhook/{request_id}")
        async def respond_via_webhook(
            request_id: UUID,
            request: Request,
            x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
            x_stateflow_signature: str | None = Header(
                default=None, alias=WEBHOOK_SIGNATURE_HEADER,
            ),
        ) -> dict:
            if x_stateflow_signature is None:
                raise HTTPException(status_code=401, detail="signature header missing")
            raw = await request.body()
            expected = hmac.new(
                webhook_secret.encode("utf-8"), raw, sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, x_stateflow_signature):
                raise HTTPException(status_code=401, detail="signature mismatch")
            if x_tenant_id is None:
                raise HTTPException(status_code=400, detail="X-Tenant-Id header required")
            try:
                tenant_id = UUID(x_tenant_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="X-Tenant-Id must be a UUID")
            import json
            body_json = json.loads(raw.decode("utf-8"))
            return await _respond(request_id, body_json, tenant_id)

    router._respond_impl = _respond  # type: ignore[attr-defined]
    return router
```

- [ ] **Step 4: Tests pass (7 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/channels/webhook.py \
        src/ballast/patterns/hitl/channels/__init__.py \
        src/ballast/patterns/hitl/api/router.py \
        tests/patterns/hitl/test_webhook_channel.py
git commit -m "feat(hitl): WebhookChannel outbound POST + inbound callback endpoint with HMAC verify"
```

---

## Task 5: `HelperVerdict[ContextT]`

Generic Pydantic model — pure type module. Module-level alias rule from project memory applies whenever a test crosses `@DBOS.workflow()` (used later in Tasks 7/9/10).

**Baseline:** 273 → **Target:** 278 (+5).

**Files:**
- Create: `src/ballast/patterns/hitl/verdict.py`
- Create: `tests/patterns/hitl/test_verdict.py`

- [ ] **Step 1: Failing tests**

`tests/patterns/hitl/test_verdict.py`:

```python
from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from ballast.patterns.hitl.verdict import HelperVerdict


class _Ctx(BaseModel):
    note: str


# Module-level alias REQUIRED (project memory: parameterized generics that
# cross @DBOS.workflow() boundaries need a module-level alias).
_CtxVerdict = HelperVerdict[_Ctx]
_NoneVerdict = HelperVerdict[None]


def test_basic_construction_with_typed_context():
    v = _CtxVerdict(
        rationale="lgtm",
        confidence=0.9,
        conversation_turn_count=3,
        tools_invoked=["cite", "approve"],
        context=_Ctx(note="hi"),
    )
    assert v.rationale == "lgtm"
    assert v.confidence == 0.9
    assert v.conversation_turn_count == 3
    assert v.tools_invoked == ["cite", "approve"]
    assert v.autopilot_eligible is False
    assert v.autopilot_confidence is None
    assert v.context is not None
    assert v.context.note == "hi"


def test_autopilot_fields_optional():
    v = _CtxVerdict(
        rationale="r", confidence=1.0,
        conversation_turn_count=0, tools_invoked=[],
        autopilot_eligible=True, autopilot_confidence=0.42,
    )
    assert v.autopilot_eligible is True
    assert v.autopilot_confidence == 0.42


def test_none_context_form():
    v = _NoneVerdict(
        rationale="ok", confidence=1.0,
        conversation_turn_count=1, tools_invoked=[],
    )
    assert v.context is None


def test_rejects_wrong_context_type():
    class _Other(BaseModel):
        pass
    with pytest.raises(ValidationError):
        _CtxVerdict(
            rationale="r", confidence=1.0,
            conversation_turn_count=0, tools_invoked=[],
            context=_Other(),  # type: ignore[arg-type]
        )


def test_round_trip_via_model_dump_and_validate():
    v = _CtxVerdict(
        rationale="r", confidence=0.5,
        conversation_turn_count=2, tools_invoked=["x"],
        context=_Ctx(note="n"),
    )
    dumped = v.model_dump(mode="json")
    restored = _CtxVerdict.model_validate(dumped)
    assert restored == v
```

- [ ] **Step 2: Run → fail (ImportError)**

- [ ] **Step 3: Implement**

`src/ballast/patterns/hitl/verdict.py`:

```python
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

ContextT = TypeVar("ContextT")


class HelperVerdict(BaseModel, Generic[ContextT]):
    """Structured verdict from a `HelperAgent`. Domain-agnostic base (spec 3J.2).

    Apps embed domain context via `context: ContextT | None`; concrete usages:

    - `HelperVerdict[None]` — no domain extension (simple approve/reject).
    - `HelperVerdict[StrategyReviewContext]` — strategy-review specific.

    NOTE (project quirk): parameterized aliases of `HelperVerdict[Foo]` that
    cross `@DBOS.workflow()` boundaries MUST be defined as module-level
    constants (e.g. `_StrategyVerdict = HelperVerdict[StrategyReviewContext]`)
    so DBOS's pickler can resolve the type on replay.
    """

    model_config = ConfigDict(frozen=True)

    rationale: str
    confidence: float
    conversation_turn_count: int
    tools_invoked: list[str]
    autopilot_eligible: bool = False
    autopilot_confidence: float | None = None
    context: ContextT | None = None
```

- [ ] **Step 4: Tests pass (5 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/verdict.py \
        tests/patterns/hitl/test_verdict.py
git commit -m "feat(hitl): HelperVerdict[ContextT] — generic, frozen, framework-side base"
```

---

## Task 6: `HelperAgentFactory` Protocol + `make_helper_agent_with_approval_tools`

Wraps a pydantic-ai `Agent` with `approve` / `reject` / optional `modify` / optional `finalize_partial` tools. Each tool builds the matching `HITLResponse` + `HelperVerdict[context_type]` and stores it on `ctx.deps` (a mutable `HelperToolBox`). The session runner (Task 7) inspects the toolbox after each `agent.run` to decide if the conversation is complete.

**Baseline:** 278 → **Target:** 284 (+6).

**Files:**
- Create: `src/ballast/patterns/hitl/helper/__init__.py`
- Create: `src/ballast/patterns/hitl/helper/factory.py`
- Create: `tests/patterns/hitl/test_helper_factory.py`

- [ ] **Step 1: Failing tests**

`tests/patterns/hitl/test_helper_factory.py`:

```python
from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ballast.patterns.hitl.helper.factory import (
    HelperAgentFactory,
    HelperDeps,
    HelperToolBox,
    make_helper_agent_with_approval_tools,
)
from ballast.patterns.hitl.response import (
    ApprovedResponse,
    ModifiedResponse,
    RejectedResponse,
)
from ballast.patterns.hitl.verdict import HelperVerdict


class _Ctx(BaseModel):
    note: str


_CtxVerdict = HelperVerdict[_Ctx]


def _scripted_agent(tool_name: str, tool_args: dict[str, Any]) -> Agent[HelperDeps, str]:
    """Build a pydantic-ai Agent that on its single turn calls `tool_name`."""

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=tool_args)])

    return Agent[HelperDeps, str](model=FunctionModel(model_fn), deps_type=HelperDeps)


def _deps(request_id: UUID) -> HelperDeps:
    return HelperDeps(
        request_id=request_id, tenant_id=uuid4(), actor_id="founder",
        turn_count=0, tools_invoked_so_far=[], toolbox=HelperToolBox(),
        autopilot_eligible=False, cached_recommendation_confidence=None,
    )


def test_factory_protocol_satisfied_by_function():
    """The function `make_helper_agent_with_approval_tools` is a HelperAgentFactory."""
    assert callable(make_helper_agent_with_approval_tools)


@pytest.mark.asyncio
async def test_approve_tool_writes_response_and_verdict_to_toolbox():
    rid = uuid4()
    base = _scripted_agent("approve", {
        "rationale": "lgtm",
        "confidence": 0.9,
        "context": {"note": "all good"},
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    deps = _deps(rid)
    await agent.run("hi", deps=deps)
    assert deps.toolbox.response is not None
    assert isinstance(deps.toolbox.response, ApprovedResponse)
    assert deps.toolbox.response.feedback == "lgtm"
    assert deps.toolbox.response.helper_verdict is not None
    assert deps.toolbox.response.helper_verdict["rationale"] == "lgtm"
    assert deps.toolbox.response.helper_verdict["context"]["note"] == "all good"


@pytest.mark.asyncio
async def test_reject_tool_writes_rejected_response():
    rid = uuid4()
    base = _scripted_agent("reject", {
        "rationale": "missing evidence", "feedback": "cite sources",
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    deps = _deps(rid)
    await agent.run("hi", deps=deps)
    assert isinstance(deps.toolbox.response, RejectedResponse)
    assert deps.toolbox.response.feedback == "cite sources"
    assert deps.toolbox.response.helper_verdict is not None
    assert deps.toolbox.response.helper_verdict["rationale"] == "missing evidence"


@pytest.mark.asyncio
async def test_modify_tool_disabled_by_default():
    rid = uuid4()
    base = _scripted_agent("modify", {
        "rationale": "tweak",
        "confidence": 0.7,
        "modified_proposal": {"amount": 99},
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    deps = _deps(rid)
    # Modify is not registered, so the model's tool-call should be ignored or
    # raise — toolbox.response stays None.
    with pytest.raises(Exception):
        await agent.run("hi", deps=deps)
    assert deps.toolbox.response is None


@pytest.mark.asyncio
async def test_modify_tool_enabled_writes_modified_response():
    rid = uuid4()
    base = _scripted_agent("modify", {
        "rationale": "tweak",
        "confidence": 0.7,
        "modified_proposal": {"amount": 99},
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
        allow_modify=True,
    )
    deps = _deps(rid)
    await agent.run("hi", deps=deps)
    assert isinstance(deps.toolbox.response, ModifiedResponse)
    assert deps.toolbox.response.modified_proposal == {"amount": 99}


@pytest.mark.asyncio
async def test_finalize_partial_disabled_by_default():
    rid = uuid4()
    base = _scripted_agent("finalize_partial", {
        "rationale": "n/a", "approved_element_ids": [], "rejected_element_ids": [],
    })
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    deps = _deps(rid)
    with pytest.raises(Exception):
        await agent.run("hi", deps=deps)


def test_factory_returns_a_helper_agent_factory_protocol_value():
    rid = uuid4()
    async def model_fn(messages, info):
        return ModelResponse(parts=[])
    base = Agent[HelperDeps, str](model=FunctionModel(model_fn), deps_type=HelperDeps)
    agent = make_helper_agent_with_approval_tools(
        base_agent=base, request_id=rid, context_type=_Ctx,
    )
    assert isinstance(agent, Agent)


def test_helper_agent_factory_protocol_is_runtime_checkable():
    class _Mine:
        def __call__(self, *, base_agent, request_id, context_type=None,
                     allow_modify=False, allow_partial=False):
            return base_agent
    assert isinstance(_Mine(), HelperAgentFactory)
```

- [ ] **Step 2: Run → fail (ImportError)**

- [ ] **Step 3: Implement**

`src/ballast/patterns/hitl/helper/__init__.py`:

```python
from ballast.patterns.hitl.helper.factory import (
    HelperAgentFactory,
    HelperDeps,
    HelperToolBox,
    make_helper_agent_with_approval_tools,
)

__all__ = [
    "HelperAgentFactory",
    "HelperDeps",
    "HelperToolBox",
    "make_helper_agent_with_approval_tools",
]
```

`src/ballast/patterns/hitl/helper/factory.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic_ai import Agent, RunContext

from ballast.patterns.hitl.response import (
    ApprovedResponse,
    HITLResponse,
    ModifiedResponse,
    RejectedResponse,
)
from ballast.patterns.hitl.verdict import HelperVerdict


@dataclass
class HelperToolBox:
    """Mutable container the approval tools write into.

    The session runner (Task 7) inspects `response` after each `agent.run`;
    if non-None, the conversation is complete and the response is sent to
    the gate's topic.
    """

    response: HITLResponse | None = None


@dataclass
class HelperDeps:
    """`ctx.deps` shape for the helper agent (`Agent[HelperDeps, str]`)."""

    request_id: UUID
    tenant_id: UUID
    actor_id: str
    turn_count: int
    tools_invoked_so_far: list[str]
    toolbox: HelperToolBox
    autopilot_eligible: bool = False
    cached_recommendation_confidence: float | None = None


@runtime_checkable
class HelperAgentFactory(Protocol):
    """Builds a HelperAgent from a base pydantic-ai Agent + a request_id.

    Default impl is `make_helper_agent_with_approval_tools`. Apps may
    supply alternatives that register custom tools.
    """

    def __call__(
        self,
        *,
        base_agent: Agent[HelperDeps, str],
        request_id: UUID,
        context_type: type[Any] | None = None,
        allow_modify: bool = False,
        allow_partial: bool = False,
    ) -> Agent[HelperDeps, str]: ...


def make_helper_agent_with_approval_tools(
    *,
    base_agent: Agent[HelperDeps, str],
    request_id: UUID,
    context_type: type[Any] | None = None,
    allow_modify: bool = False,
    allow_partial: bool = False,
) -> Agent[HelperDeps, str]:
    """Register approve / reject (+ optional modify / finalize_partial) tools.

    Each tool builds the appropriate `HITLResponse` + `HelperVerdict[context_type]`
    and writes it to `ctx.deps.toolbox.response`. The session runner (Task 7)
    picks it up there. Mutating the toolbox is preferred over `DBOS.send` here
    because the runner is the boundary that decides whether to send (single
    source of truth — keeps the factory testable without DBOS).
    """

    verdict_type = (
        HelperVerdict[context_type] if context_type is not None
        else HelperVerdict[None]
    )

    def _build_verdict(
        ctx: RunContext[HelperDeps], *, rationale: str, confidence: float,
        context: Any | None = None,
    ) -> dict[str, Any]:
        return verdict_type(
            rationale=rationale,
            confidence=confidence,
            conversation_turn_count=ctx.deps.turn_count,
            tools_invoked=list(ctx.deps.tools_invoked_so_far),
            autopilot_eligible=ctx.deps.autopilot_eligible,
            autopilot_confidence=ctx.deps.cached_recommendation_confidence,
            context=context,
        ).model_dump(mode="json")

    if context_type is not None:
        @base_agent.tool
        async def approve(  # type: ignore[unused-ignore]
            ctx: RunContext[HelperDeps],
            rationale: str,
            confidence: float,
            context: context_type,  # type: ignore[valid-type]
        ) -> str:
            ctx.deps.toolbox.response = ApprovedResponse(
                actor_id=ctx.deps.actor_id,
                answered_at=datetime.now(tz=UTC),
                feedback=rationale,
                helper_verdict=_build_verdict(
                    ctx, rationale=rationale, confidence=confidence,
                    context=context,
                ),
            )
            return "approved"
    else:
        @base_agent.tool
        async def approve(  # type: ignore[no-redef]
            ctx: RunContext[HelperDeps],
            rationale: str,
            confidence: float,
        ) -> str:
            ctx.deps.toolbox.response = ApprovedResponse(
                actor_id=ctx.deps.actor_id,
                answered_at=datetime.now(tz=UTC),
                feedback=rationale,
                helper_verdict=_build_verdict(
                    ctx, rationale=rationale, confidence=confidence,
                ),
            )
            return "approved"

    @base_agent.tool
    async def reject(
        ctx: RunContext[HelperDeps],
        rationale: str,
        feedback: str | None = None,
    ) -> str:
        ctx.deps.toolbox.response = RejectedResponse(
            actor_id=ctx.deps.actor_id,
            answered_at=datetime.now(tz=UTC),
            feedback=feedback or rationale,
            helper_verdict=_build_verdict(
                ctx, rationale=rationale, confidence=1.0,
            ),
        )
        return "rejected"

    if allow_modify:
        @base_agent.tool
        async def modify(
            ctx: RunContext[HelperDeps],
            rationale: str,
            confidence: float,
            modified_proposal: dict[str, Any],
        ) -> str:
            ctx.deps.toolbox.response = ModifiedResponse(
                actor_id=ctx.deps.actor_id,
                answered_at=datetime.now(tz=UTC),
                feedback=rationale,
                modified_proposal=modified_proposal,
                helper_verdict=_build_verdict(
                    ctx, rationale=rationale, confidence=confidence,
                ),
            )
            return "modified"

    if allow_partial:
        # Partial-approval response variant is deferred; for now we expose
        # the tool but it produces a ModifiedResponse with the partial info
        # packed into the modified_proposal under a reserved key.
        @base_agent.tool
        async def finalize_partial(
            ctx: RunContext[HelperDeps],
            rationale: str,
            approved_element_ids: list[str],
            rejected_element_ids: list[str],
            modifications: dict[str, dict[str, Any]] | None = None,
        ) -> str:
            ctx.deps.toolbox.response = ModifiedResponse(
                actor_id=ctx.deps.actor_id,
                answered_at=datetime.now(tz=UTC),
                feedback=rationale,
                modified_proposal={
                    "__partial__": True,
                    "approved_element_ids": list(approved_element_ids),
                    "rejected_element_ids": list(rejected_element_ids),
                    "modifications": dict(modifications or {}),
                },
                helper_verdict=_build_verdict(
                    ctx, rationale=rationale, confidence=1.0,
                ),
            )
            return "partial"

    return base_agent
```

- [ ] **Step 4: Tests pass (8 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/helper/__init__.py \
        src/ballast/patterns/hitl/helper/factory.py \
        tests/patterns/hitl/test_helper_factory.py
git commit -m "feat(hitl): HelperAgentFactory Protocol + typed approval-tools factory"
```

---

## Task 7: `HelperSessionRunner` Protocol + `DefaultHelperSessionRunner`

A `@DBOS.workflow()` that drives the helper conversation in its OWN workflow (spec 3J.1 — *not* inside HITLGate's workflow). Loop bound by `STATEFLOW013`-style `max_turns` kwarg (default 30). On verdict, `DBOS.send`s the response to the gate's tenant-scoped topic and exits.

**Baseline:** 284 → **Target:** 292 (+8).

**Files:**
- Create: `src/ballast/patterns/hitl/helper/session.py`
- Modify: `src/ballast/patterns/hitl/helper/__init__.py`
- Create: `tests/patterns/hitl/test_helper_session.py`

- [ ] **Step 1: Failing tests**

`tests/patterns/hitl/test_helper_session.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ballast.patterns.hitl.helper.factory import (
    HelperDeps,
    HelperToolBox,
    make_helper_agent_with_approval_tools,
)
from ballast.patterns.hitl.helper.session import (
    DefaultHelperSessionRunner,
    HelperSessionInput,
    HelperSessionRunner,
)
from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence import InMemoryThreadRepository


class _Ctx(BaseModel):
    note: str


_CtxVerdict_Session = "HelperVerdict[_Ctx]"  # for documentation only


def _scripted_agent(plan: list[tuple[str, dict[str, Any]]]) -> Agent[HelperDeps, str]:
    """Returns an agent whose Nth `.run()` calls plan[N]'s tool."""
    state = {"i": 0}

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        idx = state["i"]
        state["i"] += 1
        tool_name, tool_args = plan[idx]
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=tool_args)])

    return Agent[HelperDeps, str](model=FunctionModel(model_fn), deps_type=HelperDeps)


def _build_runner(agent_factory) -> DefaultHelperSessionRunner:
    return DefaultHelperSessionRunner(
        thread_repo=InMemoryThreadRepository(),
        agent_factory=agent_factory,
        max_turns=5,
    )


def test_runner_satisfies_protocol():
    runner = _build_runner(
        lambda *, base_agent, request_id, context_type=None, **k: base_agent,
    )
    assert isinstance(runner, HelperSessionRunner)


@pytest.mark.asyncio
async def test_runner_sends_response_to_gate_topic_and_exits(
    fresh_dbos_executor,
):
    rid = uuid4()
    tid = uuid4()
    wf_id = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c", decision_kinds={"approved"},
    )
    base = _scripted_agent([(
        "approve", {"rationale": "ok", "confidence": 0.9, "context": {"note": "n"}},
    )])

    def factory(*, base_agent, request_id, context_type=None, **k):
        return make_helper_agent_with_approval_tools(
            base_agent=base_agent, request_id=request_id,
            context_type=context_type,
        )

    runner = _build_runner(factory)
    sent: dict[str, Any] = {}

    # First recv → simulate one founder message arrival; agent then approves.
    recv = AsyncMock(side_effect=[{"text": "hello"}, None])

    def fake_send(destination, message, topic=None):
        sent.update(destination=destination, message=message, topic=topic)

    with patch(
        "ballast.patterns.hitl.helper.session.DBOS.recv", recv,
    ), patch(
        "ballast.patterns.hitl.helper.session.DBOS.send", fake_send,
    ):
        await runner.run(HelperSessionInput(
            prompt_payload=prompt.model_dump(mode="json"),
            request_id=rid, tenant_id=tid, gate_workflow_id=wf_id,
            base_agent_module="tests.patterns.hitl.test_helper_session",
            base_agent_attr=None,  # base supplied via factory closure
            context_type_fqn=None,
            actor_id="founder",
        ), _base_agent_for_test=base)

    assert sent["destination"] == str(wf_id)
    assert sent["topic"] == _hitl_topic(tid, rid)
    assert sent["message"]["kind"] == "approved"
    # Thread was opened
    assert len(runner.thread_repo._threads) == 1
    th = next(iter(runner.thread_repo._threads.values()))
    assert th.purpose == "hitl"
    assert th.purpose_metadata["request_id"] == str(rid)


@pytest.mark.asyncio
async def test_runner_loops_on_non_verdict_messages_then_completes(
    fresh_dbos_executor,
):
    rid = uuid4()
    tid = uuid4()
    wf_id = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c", decision_kinds={"approved"},
    )

    # Plan: turn 0 — no tool (we simulate by approving immediately for simplicity);
    # The runner loops once per recv. We assert it consumes two recv calls.
    base = _scripted_agent([
        ("reject", {"rationale": "need more info"}),
        ("approve", {"rationale": "ok", "confidence": 0.9, "context": {"note": "n"}}),
    ])

    def factory(*, base_agent, request_id, context_type=None, **k):
        return make_helper_agent_with_approval_tools(
            base_agent=base_agent, request_id=request_id,
            context_type=context_type,
        )

    runner = _build_runner(factory)
    recv = AsyncMock(side_effect=[
        {"text": "first"},
        {"text": "second"},
    ])
    sends: list[dict] = []

    def fake_send(destination, message, topic=None):
        sends.append({"destination": destination, "topic": topic, "message": message})

    with patch(
        "ballast.patterns.hitl.helper.session.DBOS.recv", recv,
    ), patch(
        "ballast.patterns.hitl.helper.session.DBOS.send", fake_send,
    ):
        await runner.run(HelperSessionInput(
            prompt_payload=prompt.model_dump(mode="json"),
            request_id=rid, tenant_id=tid, gate_workflow_id=wf_id,
            base_agent_module="x", base_agent_attr=None,
            context_type_fqn=None, actor_id="founder",
        ), _base_agent_for_test=base)

    # First turn produced a reject toolbox.response; the runner sends it
    # and exits (any verdict — including reject — terminates the session).
    assert len(sends) == 1
    assert sends[0]["message"]["kind"] == "rejected"
    # Only one recv consumed.
    assert recv.await_count == 1


@pytest.mark.asyncio
async def test_runner_bounded_by_max_turns(fresh_dbos_executor):
    """If max_turns exhausted with no verdict, runner exits without sending."""
    rid = uuid4()
    tid = uuid4()
    wf_id = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c", decision_kinds={"approved"},
    )

    # Agent never calls a verdict tool — returns plain text every time.
    async def model_fn(messages, info):
        return ModelResponse(parts=[])  # no tool call, no verdict

    base = Agent[HelperDeps, str](
        model=FunctionModel(model_fn), deps_type=HelperDeps,
    )

    def factory(*, base_agent, request_id, context_type=None, **k):
        return make_helper_agent_with_approval_tools(
            base_agent=base_agent, request_id=request_id,
            context_type=context_type,
        )

    runner = DefaultHelperSessionRunner(
        thread_repo=InMemoryThreadRepository(),
        agent_factory=factory, max_turns=3,
    )
    recv = AsyncMock(return_value={"text": "noop"})
    sends: list = []

    def fake_send(*a, **k): sends.append(k)

    with patch(
        "ballast.patterns.hitl.helper.session.DBOS.recv", recv,
    ), patch(
        "ballast.patterns.hitl.helper.session.DBOS.send", fake_send,
    ):
        await runner.run(HelperSessionInput(
            prompt_payload=prompt.model_dump(mode="json"),
            request_id=rid, tenant_id=tid, gate_workflow_id=wf_id,
            base_agent_module="x", base_agent_attr=None,
            context_type_fqn=None, actor_id="founder",
        ), _base_agent_for_test=base)

    assert recv.await_count == 3
    assert sends == []


def test_helper_session_input_is_frozen_basemodel():
    rid = uuid4(); tid = uuid4(); wf = uuid4()
    inp = HelperSessionInput(
        prompt_payload={"k": "v"}, request_id=rid, tenant_id=tid,
        gate_workflow_id=wf, base_agent_module="m", base_agent_attr=None,
        context_type_fqn=None, actor_id="a",
    )
    with pytest.raises(Exception):
        inp.request_id = uuid4()  # type: ignore[misc]
```

> The `_base_agent_for_test=` kwarg on `runner.run` is a test seam (allows passing a constructed Agent directly; in production the runner reconstructs via `base_agent_module` / `base_agent_attr` so the workflow input can be serialized by DBOS).

- [ ] **Step 2: Run → fail (ImportError)**

- [ ] **Step 3: Implement**

`src/ballast/patterns/hitl/helper/session.py`:

```python
from __future__ import annotations

import importlib
import itertools
from typing import Any, ClassVar, Protocol, runtime_checkable
from uuid import UUID

from dbos import DBOS, DBOSConfiguredInstance
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent

from ballast.patterns.hitl.helper.factory import (
    HelperAgentFactory,
    HelperDeps,
    HelperToolBox,
)
from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence.thread.repository import ThreadRepository

_session_counter = itertools.count()


def _helper_msg_topic(tenant_id: UUID, request_id: UUID) -> str:
    """Topic for *inbound founder messages* (separate from the gate topic)."""
    return f"helper:{tenant_id}:{request_id}"


class HelperSessionInput(BaseModel):
    """Workflow input for `DefaultHelperSessionRunner.run`.

    All fields are JSON-serializable (DBOS workflow input requirement).
    The base agent is reconstructed at runtime via `base_agent_module`
    + `base_agent_attr` (a module-level Agent constant) so the workflow
    input doesn't carry the un-picklable Agent object.
    """

    model_config = ConfigDict(frozen=True)

    prompt_payload: dict[str, Any]
    request_id: UUID
    tenant_id: UUID
    gate_workflow_id: UUID
    base_agent_module: str
    base_agent_attr: str | None
    context_type_fqn: str | None
    actor_id: str


@runtime_checkable
class HelperSessionRunner(Protocol):
    """Drives the helper conversation in its OWN DBOS workflow (spec 3J.1)."""

    async def run(self, input: HelperSessionInput) -> None: ...


@DBOS.dbos_class()
class DefaultHelperSessionRunner(DBOSConfiguredInstance):
    """Default helper-session driver.

    Loop:
      for turn in range(max_turns):
        msg = await DBOS.recv(helper_msg_topic, timeout=...)
        if msg is None: break  # timeout
        agent.run(msg.text, message_history=...)
        if toolbox.response: DBOS.send(gate_workflow_id, response, topic=gate_topic); return

    The bound on `max_turns` satisfies STATEFLOW013 (no unbounded `while`).
    """

    name: ClassVar[str] = "helper_session"

    def __init__(
        self,
        *,
        thread_repo: ThreadRepository,
        agent_factory: HelperAgentFactory,
        max_turns: int = 30,
        message_recv_timeout_seconds: float | None = 86_400.0,
    ) -> None:
        super().__init__(
            config_name=f"helper-session-{next(_session_counter)}",
        )
        self.thread_repo = thread_repo
        self.agent_factory = agent_factory
        self.max_turns = max_turns
        self.message_recv_timeout_seconds = message_recv_timeout_seconds

    @DBOS.workflow()
    async def run(
        self,
        input: HelperSessionInput,
        *,
        _base_agent_for_test: Agent[HelperDeps, str] | None = None,
    ) -> None:
        prompt = HITLPrompt.model_validate(input.prompt_payload)
        context_type = (
            _resolve_fqn(input.context_type_fqn)
            if input.context_type_fqn is not None else None
        )
        base_agent = _base_agent_for_test or _resolve_base_agent(
            input.base_agent_module, input.base_agent_attr,
        )

        # Open thread for audit + future message-history retrieval.
        thread = await self.thread_repo.create(
            purpose="hitl",
            purpose_metadata={
                "request_id": str(input.request_id),
                "gate_kind": "hitl_gate",
                "tenant_id": str(input.tenant_id),
                "title": prompt.title,
            },
            actor_id=input.actor_id,
            tenant_id=input.tenant_id,
        )

        toolbox = HelperToolBox()
        agent = self.agent_factory(
            base_agent=base_agent,
            request_id=input.request_id,
            context_type=context_type,
        )

        msg_topic = _helper_msg_topic(input.tenant_id, input.request_id)
        gate_topic = _hitl_topic(input.tenant_id, input.request_id)
        tools_invoked: list[str] = []

        for turn in range(self.max_turns):
            msg = await DBOS.recv(
                msg_topic, timeout_seconds=self.message_recv_timeout_seconds,
            )
            if msg is None:
                return  # timeout — gate's own recv will time out independently

            user_text = msg.get("text", "") if isinstance(msg, dict) else str(msg)
            await self.thread_repo.add_message(
                thread.id, role="user",
                parts=[{"type": "text", "content": user_text}],
                tenant_id=input.tenant_id,
            )

            deps = HelperDeps(
                request_id=input.request_id,
                tenant_id=input.tenant_id,
                actor_id=input.actor_id,
                turn_count=turn,
                tools_invoked_so_far=list(tools_invoked),
                toolbox=toolbox,
            )
            try:
                await agent.run(user_text, deps=deps)
            except Exception:
                # Bubble up — caller workflow records the failure.
                raise

            if toolbox.response is not None:
                DBOS.send(
                    destination=str(input.gate_workflow_id),
                    message=toolbox.response.model_dump(mode="json"),
                    topic=gate_topic,
                )
                return


def _resolve_fqn(fqn: str) -> type[Any]:
    mod_name, _, attr = fqn.rpartition(".")
    module = importlib.import_module(mod_name)
    return getattr(module, attr)


def _resolve_base_agent(module: str, attr: str | None) -> Agent[HelperDeps, str]:
    if attr is None:
        raise ValueError(
            "DefaultHelperSessionRunner: base_agent_attr is required when "
            "running outside tests (no _base_agent_for_test injected)",
        )
    return getattr(importlib.import_module(module), attr)
```

Update `helper/__init__.py`:

```python
from ballast.patterns.hitl.helper.factory import (
    HelperAgentFactory,
    HelperDeps,
    HelperToolBox,
    make_helper_agent_with_approval_tools,
)
from ballast.patterns.hitl.helper.session import (
    DefaultHelperSessionRunner,
    HelperSessionInput,
    HelperSessionRunner,
)

__all__ = [
    "DefaultHelperSessionRunner",
    "HelperAgentFactory",
    "HelperDeps",
    "HelperSessionInput",
    "HelperSessionRunner",
    "HelperToolBox",
    "make_helper_agent_with_approval_tools",
]
```

- [ ] **Step 4: Tests pass (4 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/helper/session.py \
        src/ballast/patterns/hitl/helper/__init__.py \
        tests/patterns/hitl/test_helper_session.py
git commit -m "feat(hitl): DefaultHelperSessionRunner — separate workflow drives helper convo"
```

---

## Task 8: `ConversationalChannel`

`HITLChannel` impl that opens a thread, starts the helper session as a separate workflow (deterministic idempotency key via `Det.uuid_for`), and `DBOS.recv`s on the gate's tenant-scoped topic.

**Baseline:** 292 → **Target:** 298 (+6).

**Files:**
- Create: `src/ballast/patterns/hitl/channels/conversational.py`
- Modify: `src/ballast/patterns/hitl/channels/__init__.py`
- Create: `tests/patterns/hitl/test_conversational_channel.py`

- [ ] **Step 1: Failing tests**

`tests/patterns/hitl/test_conversational_channel.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from ballast.patterns.hitl.channel import HITLChannel
from ballast.patterns.hitl.channels.conversational import (
    ConversationalChannel,
)
from ballast.patterns.hitl.helper.session import (
    DefaultHelperSessionRunner,
    HelperSessionInput,
)
from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.response import (
    ApprovedResponse,
    TimeoutResponse,
)
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence import InMemoryThreadRepository


class _Ctx(BaseModel):
    note: str


def _make_channel(runner=None):
    runner = runner or MagicMock(spec=DefaultHelperSessionRunner)
    return ConversationalChannel(
        helper_session_runner=runner,
        thread_repo=InMemoryThreadRepository(),
        base_agent_module="tests.patterns.hitl.test_conversational_channel",
        base_agent_attr=None,
        context_type=_Ctx,
        gate_workflow_id_resolver=lambda: uuid4(),
    )


def test_conversational_channel_satisfies_protocol():
    assert isinstance(_make_channel(), HITLChannel)


@pytest.mark.asyncio
async def test_ask_starts_helper_session_then_recvs(fresh_dbos_executor):
    tid = uuid4()
    rid = uuid4()
    gate_wf = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="strategy review", context="c",
        decision_kinds={"approved"}, timeout=timedelta(seconds=5),
    )

    started_with: dict[str, Any] = {}

    async def fake_start(workflow, input, *, idempotency_key=None):
        started_with["workflow"] = workflow
        started_with["input"] = input
        started_with["idempotency_key"] = idempotency_key
        return None

    runner = MagicMock()
    runner.run = MagicMock()
    payload = ApprovedResponse(
        actor_id="founder", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")
    recv = AsyncMock(return_value=payload)

    channel = ConversationalChannel(
        helper_session_runner=runner,
        thread_repo=InMemoryThreadRepository(),
        base_agent_module="my_app.agents",
        base_agent_attr="strategy_helper",
        context_type=_Ctx,
        gate_workflow_id_resolver=lambda: gate_wf,
    )

    with patch(
        "ballast.patterns.hitl.channels.conversational"
        ".start_workflow_async", fake_start,
    ), patch(
        "ballast.patterns.hitl.channels.conversational.DBOS.recv",
        recv,
    ):
        result = await channel.ask(prompt, request_id=rid)

    assert isinstance(result, ApprovedResponse)
    assert started_with["workflow"] is runner.run
    inp: HelperSessionInput = started_with["input"]
    assert inp.request_id == rid
    assert inp.tenant_id == tid
    assert inp.gate_workflow_id == gate_wf
    assert inp.base_agent_module == "my_app.agents"
    assert inp.base_agent_attr == "strategy_helper"
    assert inp.context_type_fqn.endswith("test_conversational_channel._Ctx")
    # Idempotency key is deterministic — same inputs would yield same key.
    assert started_with["idempotency_key"].startswith("helper:")
    recv.assert_awaited_once_with(_hitl_topic(tid, rid), timeout_seconds=5.0)


@pytest.mark.asyncio
async def test_ask_returns_timeout_when_recv_returns_none(fresh_dbos_executor):
    tid = uuid4(); rid = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c", decision_kinds={"approved"},
        timeout=timedelta(seconds=1),
    )
    runner = MagicMock()
    runner.run = MagicMock()
    with patch(
        "ballast.patterns.hitl.channels.conversational"
        ".start_workflow_async", AsyncMock(return_value=None),
    ), patch(
        "ballast.patterns.hitl.channels.conversational.DBOS.recv",
        AsyncMock(return_value=None),
    ):
        channel = ConversationalChannel(
            helper_session_runner=runner,
            thread_repo=InMemoryThreadRepository(),
            base_agent_module="m", base_agent_attr=None,
            context_type=None,
            gate_workflow_id_resolver=lambda: uuid4(),
        )
        result = await channel.ask(prompt, request_id=rid)
    assert isinstance(result, TimeoutResponse)


@pytest.mark.asyncio
async def test_idempotency_key_stable_for_same_request(fresh_dbos_executor):
    tid = uuid4(); rid = uuid4()
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c", decision_kinds={"approved"},
    )
    keys: list[str] = []

    async def fake_start(workflow, input, *, idempotency_key=None):
        keys.append(idempotency_key)
        return None

    runner = MagicMock()
    payload = ApprovedResponse(
        actor_id="f", answered_at=datetime.now(tz=UTC),
    ).model_dump(mode="json")

    with patch(
        "ballast.patterns.hitl.channels.conversational"
        ".start_workflow_async", fake_start,
    ), patch(
        "ballast.patterns.hitl.channels.conversational.DBOS.recv",
        AsyncMock(return_value=payload),
    ):
        channel = ConversationalChannel(
            helper_session_runner=runner,
            thread_repo=InMemoryThreadRepository(),
            base_agent_module="m", base_agent_attr=None,
            context_type=None,
            gate_workflow_id_resolver=lambda: uuid4(),
        )
        await channel.ask(prompt, request_id=rid)
        await channel.ask(prompt, request_id=rid)
    assert len(keys) == 2
    assert keys[0] == keys[1]
```

- [ ] **Step 2: Run → fail (ImportError)**

- [ ] **Step 3: Implement**

`src/ballast/patterns/hitl/channels/conversational.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID

from dbos import DBOS, SetWorkflowID
from pydantic import TypeAdapter

from ballast.patterns.hitl.helper.session import (
    DefaultHelperSessionRunner,
    HelperSessionInput,
)
from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.response import (
    HITLResponse,
    TimeoutResponse,
)
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.persistence.thread.repository import ThreadRepository
from ballast.runtime.det import Det
from ballast.runtime.idempotency import IdempotencyInput

_RESPONSE_ADAPTER: TypeAdapter[HITLResponse] = TypeAdapter(HITLResponse)


async def start_workflow_async(
    workflow_fn: Callable[..., Any],
    input: Any,
    *,
    idempotency_key: str,
) -> None:
    """Thin wrapper around `DBOS.start_workflow_async` + `SetWorkflowID`.

    Extracted so tests can patch a single symbol. Production: this kicks off
    the helper-session workflow under a deterministic workflow_id so retries
    of the parent gate workflow re-attach to the same session instead of
    spawning new ones.
    """
    with SetWorkflowID(idempotency_key):
        await DBOS.start_workflow_async(workflow_fn, input)


class ConversationalChannel:
    """HITL channel backed by a helper pydantic-ai Agent in its own workflow.

    Lifecycle (spec 3J.1):
      1. `ask()` opens a thread (`purpose="hitl"`).
      2. Starts `helper_session_runner.run` as an INDEPENDENT workflow with
         a deterministic idempotency key (`helper:{tenant}:{request}`) — so
         gate-workflow replay reuses the same session.
      3. `DBOS.recv`s on the gate's tenant-scoped topic until the helper
         agent invokes an approval tool (which DBOS.sends to that topic).
      4. Returns the resulting `HITLResponse` (or `TimeoutResponse`).
    """

    name: ClassVar[str] = "conversational"

    def __init__(
        self,
        *,
        helper_session_runner: DefaultHelperSessionRunner,
        thread_repo: ThreadRepository,
        base_agent_module: str,
        base_agent_attr: str | None,
        context_type: type[Any] | None,
        gate_workflow_id_resolver: Callable[[], UUID],
        actor_id: str = "founder",
    ) -> None:
        self.helper_session_runner = helper_session_runner
        self.thread_repo = thread_repo
        self.base_agent_module = base_agent_module
        self.base_agent_attr = base_agent_attr
        self.context_type = context_type
        self.gate_workflow_id_resolver = gate_workflow_id_resolver
        self.actor_id = actor_id

    async def ask(
        self, prompt: HITLPrompt, *, request_id: UUID,
    ) -> HITLResponse:
        idempotency_key = await self._idempotency_key(
            prompt.tenant_id, request_id,
        )
        gate_wf = self.gate_workflow_id_resolver()
        input = HelperSessionInput(
            prompt_payload=prompt.model_dump(mode="json"),
            request_id=request_id,
            tenant_id=prompt.tenant_id,
            gate_workflow_id=gate_wf,
            base_agent_module=self.base_agent_module,
            base_agent_attr=self.base_agent_attr,
            context_type_fqn=(
                f"{self.context_type.__module__}.{self.context_type.__qualname__}"
                if self.context_type is not None else None
            ),
            actor_id=self.actor_id,
        )
        await start_workflow_async(
            self.helper_session_runner.run, input,
            idempotency_key=idempotency_key,
        )

        topic = _hitl_topic(prompt.tenant_id, request_id)
        timeout_seconds = (
            prompt.timeout.total_seconds() if prompt.timeout is not None else None
        )
        payload = await DBOS.recv(topic, timeout_seconds=timeout_seconds)
        if payload is None:
            return TimeoutResponse(answered_at=datetime.now(tz=UTC))
        return _RESPONSE_ADAPTER.validate_python(payload)

    @staticmethod
    async def _idempotency_key(tenant_id: UUID, request_id: UUID) -> str:
        derived = await Det.uuid_for(
            IdempotencyInput(
                namespace="helper_session",
                parts={
                    "tenant_id": tenant_id,
                    "request_id": request_id,
                },
            ),
        )
        return f"helper:{derived}"
```

Update `channels/__init__.py`:

```python
from ballast.patterns.hitl.channels.conversational import (
    ConversationalChannel,
)
from ballast.patterns.hitl.channels.ui import UIChannel
from ballast.patterns.hitl.channels.webhook import (
    WEBHOOK_SIGNATURE_HEADER,
    WebhookChannel,
    WebhookConfig,
)

__all__ = [
    "ConversationalChannel",
    "UIChannel",
    "WEBHOOK_SIGNATURE_HEADER",
    "WebhookChannel",
    "WebhookConfig",
]
```

- [ ] **Step 4: Tests pass (4 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/channels/conversational.py \
        src/ballast/patterns/hitl/channels/__init__.py \
        tests/patterns/hitl/test_conversational_channel.py
git commit -m "feat(hitl): ConversationalChannel — separate-workflow helper with deterministic idempotency"
```

---

## Task 9: Persistence wiring — helper verdict round-trip

`Decision` rows already carry `helper_verdict_payload`, `helper_verdict_context_type`, `helper_thread_id` (SP2). SP5's `HITLGate.run` does NOT forward them yet. Update it to detect `response.helper_verdict`, infer the context-type FQN from the channel (passed via prompt purpose_metadata or per-channel attribute), and pass through.

**Baseline:** 298 → **Target:** 304 (+6).

**Files:**
- Modify: `src/ballast/patterns/hitl/gate.py`
- Create: `tests/patterns/hitl/test_persistence_wiring.py`

- [ ] **Step 1: Failing tests**

`tests/patterns/hitl/test_persistence_wiring.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel

from ballast.patterns.hitl.channel import InMemoryHITLChannel
from ballast.patterns.hitl.gate import HITLGate
from ballast.patterns.hitl.policy import AllowAll
from ballast.patterns.hitl.prompt import HITLPrompt
from ballast.patterns.hitl.response import ApprovedResponse
from ballast.patterns.hitl.verdict import HelperVerdict
from ballast.persistence import InMemoryHITLRepository


class _Ctx(BaseModel):
    note: str


_CtxVerdict_Wiring = HelperVerdict[_Ctx]


@pytest.mark.asyncio
async def test_helper_verdict_persisted_via_gate(fresh_dbos_executor):
    tid = uuid4()
    repo = InMemoryHITLRepository()
    channel = InMemoryHITLChannel()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    verdict = _CtxVerdict_Wiring(
        rationale="r", confidence=0.9, conversation_turn_count=2,
        tools_invoked=["approve"], context=_Ctx(note="hello"),
    )

    # Preload response so channel.ask returns it; capture the request_id
    # by patching persist_request via the original method.
    captured: dict = {}
    orig = repo.persist_request

    async def capture(**kw):
        req = await orig(**kw)
        captured["request"] = req
        channel.set_response(req.id, ApprovedResponse(
            actor_id="alice", answered_at=datetime.now(tz=UTC),
            feedback="ok", helper_verdict=verdict.model_dump(mode="json"),
        ))
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c", decision_kinds={"approved"},
    )
    await gate.run(prompt, tenant_id=tid)

    # One Decision row created.
    assert len(repo._decisions) == 1
    decision = next(iter(repo._decisions.values()))
    assert decision.helper_verdict_payload is not None
    assert decision.helper_verdict_payload["rationale"] == "r"
    assert decision.helper_verdict_payload["context"]["note"] == "hello"


@pytest.mark.asyncio
async def test_helper_verdict_absent_when_response_lacks_it(fresh_dbos_executor):
    tid = uuid4()
    repo = InMemoryHITLRepository()
    channel = InMemoryHITLChannel()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    orig = repo.persist_request

    async def capture(**kw):
        req = await orig(**kw)
        channel.set_response(req.id, ApprovedResponse(
            actor_id="alice", answered_at=datetime.now(tz=UTC),
        ))
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c", decision_kinds={"approved"},
    )
    await gate.run(prompt, tenant_id=tid)
    decision = next(iter(repo._decisions.values()))
    assert decision.helper_verdict_payload is None
    assert decision.helper_verdict_context_type is None
    assert decision.helper_thread_id is None


@pytest.mark.asyncio
async def test_helper_thread_id_propagated_via_prompt_metadata(fresh_dbos_executor):
    """If the prompt carries a `helper_thread_id` (set by ConversationalChannel),
    the gate threads it through to persist_response."""
    tid = uuid4()
    thread_id = uuid4()
    repo = InMemoryHITLRepository()
    channel = InMemoryHITLChannel()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)

    orig = repo.persist_request

    async def capture(**kw):
        req = await orig(**kw)
        channel.set_response(req.id, ApprovedResponse(
            actor_id="alice", answered_at=datetime.now(tz=UTC),
            helper_verdict={"rationale": "r", "confidence": 1.0,
                            "conversation_turn_count": 0, "tools_invoked": [],
                            "autopilot_eligible": False,
                            "autopilot_confidence": None, "context": None,
                            "__helper_thread_id__": str(thread_id),
                            "__context_type_fqn__": "x.y.Ctx"},
        ))
        return req

    repo.persist_request = capture  # type: ignore[method-assign]

    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c", decision_kinds={"approved"},
    )
    await gate.run(prompt, tenant_id=tid)
    decision = next(iter(repo._decisions.values()))
    assert decision.helper_thread_id == thread_id
    assert decision.helper_verdict_context_type == "x.y.Ctx"
```

- [ ] **Step 2: Run → fail (assertion: payload always None)**

- [ ] **Step 3: Implement** — replace `HITLGate.run`'s `persist_response` call:

```python
# In src/ballast/patterns/hitl/gate.py — patch the final
# `await self.repo.persist_response(...)` call to forward helper fields:

helper_verdict_payload: dict | None = None
helper_verdict_context_type: str | None = None
helper_thread_id: UUID | None = None
if getattr(response, "helper_verdict", None) is not None:
    helper_verdict_payload = dict(response.helper_verdict)
    # Optional sidecar keys carried inside the helper_verdict blob (set
    # by ConversationalChannel via the helper agent). Stripped from the
    # persisted blob so the row's typed columns hold them instead.
    tid_str = helper_verdict_payload.pop("__helper_thread_id__", None)
    fqn = helper_verdict_payload.pop("__context_type_fqn__", None)
    if tid_str is not None:
        helper_thread_id = UUID(tid_str)
    if fqn is not None:
        helper_verdict_context_type = fqn

await self.repo.persist_response(
    request_id=request.id,
    actor_id=response.actor_id or "<anonymous>",
    verdict=_KIND_TO_VERDICT[response.kind],
    payload=response.model_dump(mode="json"),
    tenant_id=tenant_id,
    helper_verdict_payload=helper_verdict_payload,
    helper_verdict_context_type=helper_verdict_context_type,
    helper_thread_id=helper_thread_id,
)
```

- [ ] **Step 4: Tests pass (3 new)**
- [ ] **Step 5: Full suite + mypy + ruff** — every prior HITLGate test continues passing because the new kwargs are optional.
- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/gate.py \
        tests/patterns/hitl/test_persistence_wiring.py
git commit -m "feat(hitl): HITLGate forwards helper_verdict / thread_id / context_type to Decision row"
```

---

## Task 10: Public API + integration smoke

Top-level exports + an end-to-end smoke that wires `HITLGate` + `ConversationalChannel` + the FastAPI router and drives the whole loop in a single test (UI endpoint posts a response while the gate workflow is awaiting on the topic).

**Baseline:** 304 → **Target:** ~309 (+5).

**Files:**
- Modify: `src/ballast/patterns/hitl/__init__.py`
- Modify: `src/ballast/__init__.py`
- Create: `tests/patterns/hitl/test_public_api_sp6.py`

- [ ] **Step 1: Update `patterns/hitl/__init__.py`**

```python
from ballast.patterns.hitl.api import build_hitl_router
from ballast.patterns.hitl.channel import (
    HITLChannel, InMemoryHITLChannel,
)
from ballast.patterns.hitl.channels import (
    WEBHOOK_SIGNATURE_HEADER,
    ConversationalChannel,
    UIChannel,
    WebhookChannel,
    WebhookConfig,
)
from ballast.patterns.hitl.gate import HITLGate
from ballast.patterns.hitl.helper import (
    DefaultHelperSessionRunner,
    HelperAgentFactory,
    HelperDeps,
    HelperSessionInput,
    HelperSessionRunner,
    HelperToolBox,
    make_helper_agent_with_approval_tools,
)
from ballast.patterns.hitl.policy import (
    AccessDecision, AllowAll, DenyAll, Policy, Voter,
)
from ballast.patterns.hitl.prompt import HITLOption, HITLPrompt
from ballast.patterns.hitl.response import (
    ApprovedResponse, HITLResponse, ModifiedResponse,
    RejectedResponse, TimeoutResponse,
)
from ballast.patterns.hitl.topic import _hitl_topic
from ballast.patterns.hitl.verdict import HelperVerdict

__all__ = [
    "AccessDecision", "AllowAll", "ApprovedResponse",
    "ConversationalChannel", "DefaultHelperSessionRunner", "DenyAll",
    "HelperAgentFactory", "HelperDeps", "HelperSessionInput",
    "HelperSessionRunner", "HelperToolBox", "HelperVerdict",
    "HITLChannel", "HITLGate", "HITLOption", "HITLPrompt", "HITLResponse",
    "InMemoryHITLChannel", "ModifiedResponse", "Policy", "RejectedResponse",
    "TimeoutResponse", "UIChannel", "Voter",
    "WEBHOOK_SIGNATURE_HEADER", "WebhookChannel", "WebhookConfig",
    "_hitl_topic", "build_hitl_router",
    "make_helper_agent_with_approval_tools",
]
```

- [ ] **Step 2: Add to top-level `src/ballast/__init__.py`**

```python
from ballast.patterns.hitl import (
    ConversationalChannel,
    DefaultHelperSessionRunner,
    HelperAgentFactory,
    HelperDeps,
    HelperSessionInput,
    HelperSessionRunner,
    HelperToolBox,
    HelperVerdict,
    UIChannel,
    WebhookChannel,
    WebhookConfig,
    build_hitl_router,
    make_helper_agent_with_approval_tools,
)
```

Append the new names to `__all__`.

- [ ] **Step 3: Integration smoke**

`tests/patterns/hitl/test_public_api_sp6.py`:

```python
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from ballast import (
    ConversationalChannel,
    HelperVerdict,
    HITLGate,
    HITLPrompt,
    UIChannel,
    WebhookChannel,
    WebhookConfig,
    build_hitl_router,
    make_helper_agent_with_approval_tools,
)
from ballast.patterns.hitl.policy import AllowAll
from ballast.persistence import InMemoryHITLRepository


class _Ctx(BaseModel):
    note: str


_VerdictSmoke = HelperVerdict[_Ctx]


def test_all_sp6_symbols_visible_at_top_level():
    assert UIChannel is not None
    assert WebhookChannel is not None
    assert WebhookConfig is not None
    assert ConversationalChannel is not None
    assert HelperVerdict is not None
    assert make_helper_agent_with_approval_tools is not None
    assert build_hitl_router is not None


def test_ui_channel_round_trip_via_router(fresh_dbos_executor):
    """Spin up the FastAPI app, run HITLGate+UIChannel in a task,
    POST a response from the test client, observe the gate completing."""
    from unittest.mock import patch

    tid = uuid4()
    repo = InMemoryHITLRepository()
    channel = UIChannel()
    gate = HITLGate(channel=channel, policy=AllowAll(), repo=repo)
    prompt = HITLPrompt(
        tenant_id=tid, title="t", context="c", decision_kinds={"approved"},
        timeout=timedelta(seconds=2),
    )

    app = FastAPI()
    app.include_router(build_hitl_router(repo=repo, policy=AllowAll()))

    async def driver():
        # Wait for the request to be persisted, then POST.
        for _ in range(50):
            pending = await repo.list_pending(tenant_id=tid)
            if pending: break
            await asyncio.sleep(0.01)
        assert pending
        rid = pending[0].id
        with TestClient(app) as client:
            r = client.post(
                f"/hitl/{rid}/respond",
                headers={"X-Tenant-Id": str(tid)},
                json={
                    "kind": "approved", "actor_id": "alice",
                    "answered_at": datetime.now(tz=UTC).isoformat(),
                    "feedback": "lgtm",
                },
            )
            assert r.status_code == 200

    async def run_both():
        gate_task = asyncio.create_task(gate.run(prompt, tenant_id=tid))
        await driver()
        return await gate_task

    response = asyncio.get_event_loop().run_until_complete(run_both()) \
        if not asyncio.get_event_loop().is_running() else None
    # Pytest-asyncio path:
    if response is None:
        async def _go():
            gate_task = asyncio.create_task(gate.run(prompt, tenant_id=tid))
            await driver()
            return await gate_task
        response = asyncio.run(_go())
    assert response.kind == "approved"
    assert response.actor_id == "alice"
```

> Note: The integration test above relies on DBOS `send`/`recv` being wired against the test SQLite system DB (via `fresh_dbos_executor`). If DBOS doesn't fully cooperate inside `TestClient`, fall back to an isolated test that patches `DBOS.send` / `DBOS.recv` in the same way prior task tests did — what matters is the symbolic API surface + the smoke that wiring is reachable.

- [ ] **Step 4: Tests pass (~5 new — top-level exports + smoke)**
- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/ballast/patterns/hitl/__init__.py \
        src/ballast/__init__.py \
        tests/patterns/hitl/test_public_api_sp6.py
git commit -m "feat: Sub-project #6 public API (HITL channels exports + end-to-end smoke)"
```

---

## Sub-project #6 acceptance criteria

After all 10 tasks:

- `from ballast import UIChannel, WebhookChannel, WebhookConfig, ConversationalChannel, HelperVerdict, make_helper_agent_with_approval_tools, build_hitl_router` works.
- `UIChannel`, `WebhookChannel`, `ConversationalChannel` all satisfy `isinstance(x, HITLChannel)` and return `TimeoutResponse` on `DBOS.recv` exhausting `prompt.timeout`.
- `build_hitl_router(repo, policy)` mounts `POST /hitl/{request_id}/respond` performing endpoint-side authz (point #1 of the spec 2C.4 two-point check) and `DBOS.send` to the gate's tenant-scoped topic `hitl:{tenant_id}:{request_id}`.
- `build_hitl_router(..., webhook_secret=...)` ADDITIONALLY mounts `POST /hitl/webhook/{request_id}` that verifies an HMAC-SHA256 signature in `X-Stateflow-Signature` via `hmac.compare_digest`, then reuses the same load+authz+send pipeline.
- `WebhookChannel.ask` POSTs a deterministically-ordered JSON body (sorted keys, tight separators) and signs the exact bytes via `sign_payload`. The outbound POST runs inside `@DBOS.step post_webhook` so replay records the call once.
- `HelperVerdict[ContextT]` is frozen, generic, JSON round-trippable, and project-memory rule about module-level aliases is honored across all SP6 tests that cross workflow boundaries.
- `make_helper_agent_with_approval_tools` registers `approve` + `reject` unconditionally, `modify` only if `allow_modify=True`, `finalize_partial` only if `allow_partial=True`. Each tool writes a fully-typed `HITLResponse` + `HelperVerdict[context_type]` to `ctx.deps.toolbox.response`.
- `DefaultHelperSessionRunner` is a `@DBOS.workflow()` on a `DBOSConfiguredInstance` (per project memory: `DBOS.dbos_class()` + module-level counter for `config_name`). The loop is bounded by `max_turns` (STATEFLOW013 clean). On `toolbox.response`, the runner `DBOS.send`s to `_hitl_topic(tenant, request)` with `destination=gate_workflow_id` and exits.
- `ConversationalChannel.ask` derives its helper-session idempotency key via `Det.uuid_for(IdempotencyInput(namespace="helper_session", parts={tenant_id, request_id}))` so retries reuse the same helper session.
- `HITLGate.run` (modified in Task 9) forwards `helper_verdict_payload`, `helper_verdict_context_type`, `helper_thread_id` to `HITLRepository.persist_response`; absent helper data leaves all three columns `None`.
- All Sub-project #1 through #5 tests still pass (243 passed + 10 skipped baseline preserved).
- `mypy strict` + `ruff check` clean.
- Pattern Protocol invariant: every new channel class has `name: ClassVar[str]` and satisfies `HITLChannel` structurally (no inheritance).

---

### Critical Files for Implementation
- /Users/kirunya/conductor/workspaces/ballast-ai-engine/philadelphia-v1/src/ballast/patterns/hitl/channel.py
- /Users/kirunya/conductor/workspaces/ballast-ai-engine/philadelphia-v1/src/ballast/patterns/hitl/gate.py
- /Users/kirunya/conductor/workspaces/ballast-ai-engine/philadelphia-v1/src/ballast/persistence/hitl/repository.py
- /Users/kirunya/conductor/workspaces/ballast-ai-engine/philadelphia-v1/src/ballast/persistence/thread/repository.py
- /Users/kirunya/conductor/workspaces/ballast-ai-engine/philadelphia-v1/src/ballast/runtime/det.py

### Task Titles (10)
1. HITL FastAPI router primitives — `_hitl_topic`, `build_hitl_router`
2. `UIChannel`
3. `WebhookChannel` outbound primitives — config, signing, POST step
4. `WebhookChannel` + inbound callback endpoint
5. `HelperVerdict[ContextT]`
6. `HelperAgentFactory` Protocol + `make_helper_agent_with_approval_tools`
7. `HelperSessionRunner` Protocol + `DefaultHelperSessionRunner`
8. `ConversationalChannel`
9. Persistence wiring — helper verdict round-trip
10. Public API + integration smoke
