# FastAPI + Observability + Evals (Sub-project #7) Implementation Plan

```yaml
date: 2026-05-15
sub_project: 7
status: ready-for-implementation
baseline_tests: 299 passed + 10 skipped (after SP6)
target_tests: ~349 passed + 10 skipped (after SP7)
```

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the L7 (FastAPI surface), L6 (observability via logfire) and L5 (evals MVP) slices of the v1 spec. Concretely: ship the `fastapi_app()` factory on `Engine`, thread CRUD + AG-UI + Vercel SDK streaming endpoints for chat UX, A2A discovery + invoke endpoints for agent-to-agent, an `ObservabilityProvider` that fronts logfire with a soft-import (no hard dep on logfire), canonical span helpers wrapped around `Pattern.run` / `Stage.process` / `HITLChannel.ask`, the `SchemaAdherenceScorer` + `Dataset` + `EvalReport` primitives, and a `stateflow evals dataset-from-traces` CLI.

**Spec sections covered:** 1.8 (streaming SSE channels narrative), 1.13 (A2A as primary inter-agent protocol), 1.14 (eval-from-trace tooling), 2D / 3J narrative for ConversationalChannel parity, 3H (FastAPI thin layer — `/healthz`, `/threads/*`, `/hitl/*`, `/a2a/*`), 4A.0.7 (Container lives in `app.state.container`, never global, accessed via `Depends(get_container)`), 4C (Evals v1 — `SchemaAdherenceScorer` + `Dataset` + CLI `dataset-from-traces`), 4D (Observability — `ObservabilityProvider`, canonical span table, soft dep on logfire), 4F (MVP — L5: `SchemaAdherenceScorer` only; L6: logfire + one dashboard only; L7: FastAPI + AG-UI + Vercel + A2A), 4H (bootstrap invariants — `ObservabilityProvider` registers FIRST).

**Scope vs deferred:**
- v1 in SP7: `ObservabilityProvider` + `@traced(...)` span decorator/context-manager + canonical span names on Reflection / MapReduce / MutationPipeline / HITLGate / HITLChannel; `Engine.fastapi_app(...)`; DI helpers `get_container` / `get_engine` / `get_tenant_id`; `/healthz`; `build_threads_router` (REST CRUD); `build_streaming_router` with two adapter modes (`AGUIEncoder`, `VercelEncoder`); `build_a2a_router` (`/.well-known/agent.json` + `POST /a2a/{agent_name}`); `EvalCase`, `Dataset`, `Scorer` Protocol, `EvalReport`, `SchemaAdherenceScorer`, in-memory `Dataset.evaluate(...)`; `python -m ballast.evals.cli dataset-from-traces` Typer entry point; end-to-end smoke test (build app → run a Reflection → run a Dataset eval).
- Deferred to v1.1 / v2: `MutationAcceptanceScorer`, `IterationBudgetScorer`, `GroundedReferenceScorer`, `HelperVerdictDisagreementScorer`; the other 5 dashboards (HITL latency stays a doc-only sketch); the drift-detection pipeline; `STATEFLOW006-013` lint rules (only `STATEFLOW001-005` shipped, additions out of scope here); auto-registration of `stateflow` entry-point in `pyproject.toml [project.scripts]` (use `python -m ...` in v1); `RemoteAgent` proxy for outbound A2A (left to apps using `httpx`); WebSocket transport (SSE only); Slack / Discord A2A flavours.

---

## File Structure

```
src/ballast/
├── api/
│   ├── __init__.py                          # build_threads_router, build_streaming_router, build_a2a_router, build_health_router, deps re-export
│   ├── deps.py                              # get_container, get_engine, get_tenant_id, build_default_tenant_resolver
│   ├── health.py                            # build_health_router -> GET /healthz
│   ├── threads.py                           # build_threads_router -> POST /threads, GET /threads/{id}, GET /threads/{id}/messages
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── router.py                        # build_streaming_router -> POST /threads/{id}/messages?protocol=ag-ui|vercel
│   │   ├── ag_ui.py                         # AGUIEncoder.encode(event) -> bytes (SSE event lines)
│   │   └── vercel.py                        # VercelEncoder.encode(event) -> bytes
│   └── a2a.py                               # build_a2a_router -> GET /.well-known/agent.json, POST /a2a/{agent_name}
├── observability/
│   ├── __init__.py                          # ObservabilityProvider, traced, span_attrs, has_logfire
│   ├── provider.py                          # ObservabilityProvider (soft-imports logfire)
│   └── spans.py                             # @traced, span_attrs(), _noop ctx manager
├── evals/
│   ├── __init__.py                          # EvalCase, Dataset, Scorer, EvalReport, SchemaAdherenceScorer
│   ├── case.py                              # EvalCase, EvalRunOutput
│   ├── dataset.py                           # Dataset, EvalReport, ScoreResult
│   ├── scorer.py                            # Scorer Protocol, SchemaAdherenceScorer
│   ├── traces.py                            # dataset_from_traces() — joins eval_runs / proposal_audit / thread / decision rows
│   └── cli.py                               # `python -m ballast.evals.cli dataset-from-traces ...`
├── runtime/
│   └── engine.py                            # extended: Engine.fastapi_app(...)
├── patterns/
│   ├── reflection.py                        # @traced('pattern.reflection') wrap
│   ├── mapreduce/pattern.py                 # @traced('pattern.mapreduce')
│   ├── mutation/pipeline.py                 # @traced('pattern.mutation_pipeline') + per-stage span
│   └── hitl/
│       ├── gate.py                          # @traced('pattern.hitl_gate')
│       └── channels/{ui,webhook,conversational}.py   # @traced('channel.<name>') around .ask
└── __init__.py                              # SP7 public exports added

tests/
├── api/
│   ├── test_deps.py
│   ├── test_health.py
│   ├── test_threads_router.py
│   ├── test_streaming_ag_ui.py
│   ├── test_streaming_vercel.py
│   ├── test_a2a_router.py
│   └── test_engine_fastapi_app.py
├── observability/
│   ├── test_provider.py
│   ├── test_traced.py
│   └── test_pattern_instrumentation.py
├── evals/
│   ├── test_dataset.py
│   ├── test_schema_adherence_scorer.py
│   ├── test_dataset_from_traces.py
│   └── test_cli.py
└── test_public_api_sp7.py
```

---

## Task 1: API primitives — `deps.py` + `/healthz`

Pure FastAPI scaffolding. Three Depends helpers and a one-route health router. Container is read from `request.app.state.container` per spec 4A.0.7. Tenant resolver pulls from `X-Tenant-Id` header by default, overridable via app-level `app.state.tenant_resolver` callable.

**Baseline:** 299 passed + 10 skipped → **Target:** 309 passed + 10 skipped (+10).

**Files:**
- Create: `src/ballast/api/__init__.py`
- Create: `src/ballast/api/deps.py`
- Create: `src/ballast/api/health.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_deps.py`
- Create: `tests/api/test_health.py`

- [ ] **Step 1: Failing tests**

`tests/api/test_deps.py`:

```python
from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from ballast.api.deps import (
    get_container,
    get_engine,
    get_tenant_id,
)
from ballast.runtime import DefaultContainer, Engine
from ballast.runtime.provider import ServiceProvider


class _NoopProvider:
    async def register(self, container) -> None:
        return None


def _app_with_engine() -> tuple[FastAPI, Engine]:
    engine = Engine(providers=[_NoopProvider()])
    app = FastAPI()
    app.state.container = engine.container
    app.state.engine = engine

    @app.get("/echo-container")
    async def echo_container(container=Depends(get_container)) -> dict[str, str]:
        return {"container_class": type(container).__name__}

    @app.get("/echo-engine")
    async def echo_engine(engine=Depends(get_engine)) -> dict[str, str]:
        return {"engine_class": type(engine).__name__}

    @app.get("/echo-tenant")
    async def echo_tenant(tenant_id: UUID = Depends(get_tenant_id)) -> dict[str, str]:
        return {"tenant_id": str(tenant_id)}

    return app, engine


def test_get_container_pulls_from_app_state():
    app, _ = _app_with_engine()
    with TestClient(app) as c:
        r = c.get("/echo-container")
    assert r.status_code == 200
    assert r.json()["container_class"] == "DefaultContainer"


def test_get_container_raises_when_unset():
    app = FastAPI()

    @app.get("/x")
    async def x(container=Depends(get_container)) -> dict[str, str]:
        return {"ok": "1"}

    with TestClient(app) as c:
        r = c.get("/x")
    assert r.status_code == 500
    assert "container" in r.text.lower()


def test_get_engine_pulls_from_app_state():
    app, engine = _app_with_engine()
    with TestClient(app) as c:
        r = c.get("/echo-engine")
    assert r.status_code == 200
    assert r.json()["engine_class"] == "Engine"


def test_get_tenant_id_from_header():
    app, _ = _app_with_engine()
    tid = uuid4()
    with TestClient(app) as c:
        r = c.get("/echo-tenant", headers={"X-Tenant-Id": str(tid)})
    assert r.status_code == 200
    assert r.json()["tenant_id"] == str(tid)


def test_get_tenant_id_400_when_missing():
    app, _ = _app_with_engine()
    with TestClient(app) as c:
        r = c.get("/echo-tenant")
    assert r.status_code == 400


def test_get_tenant_id_400_when_not_uuid():
    app, _ = _app_with_engine()
    with TestClient(app) as c:
        r = c.get("/echo-tenant", headers={"X-Tenant-Id": "not-a-uuid"})
    assert r.status_code == 400


def test_get_tenant_id_uses_override_resolver():
    """Apps can swap the resolver by setting `app.state.tenant_resolver`."""
    app, _ = _app_with_engine()
    pinned = uuid4()
    app.state.tenant_resolver = lambda request: pinned
    with TestClient(app) as c:
        r = c.get("/echo-tenant")  # no header — resolver wins
    assert r.status_code == 200
    assert r.json()["tenant_id"] == str(pinned)
```

`tests/api/test_health.py`:

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.api.health import build_health_router


def test_healthz_returns_200_ok():
    app = FastAPI()
    app.include_router(build_health_router())
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz_respects_prefix():
    app = FastAPI()
    app.include_router(build_health_router(prefix="/api"))
    with TestClient(app) as c:
        r = c.get("/api/healthz")
    assert r.status_code == 200


def test_healthz_passes_optional_checks():
    """Optional checks fold into response when provided."""
    calls: list[str] = []

    async def db_ok() -> bool:
        calls.append("db")
        return True

    app = FastAPI()
    app.include_router(build_health_router(checks={"db": db_ok}))
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "checks": {"db": "ok"}}
    assert calls == ["db"]
```

- [ ] **Step 2: Run → fail (ImportError)**

```bash
uv run pytest tests/api/test_deps.py tests/api/test_health.py -v
```

- [ ] **Step 3: Implement**

`src/ballast/api/__init__.py`:

```python
from ballast.api.deps import (
    get_container,
    get_engine,
    get_tenant_id,
)
from ballast.api.health import build_health_router

__all__ = [
    "build_health_router",
    "get_container",
    "get_engine",
    "get_tenant_id",
]
```

`src/ballast/api/deps.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from ballast.runtime import Engine
    from ballast.runtime.container import Container

TenantResolver = Callable[[Request], UUID]


def get_container(request: Request) -> Container:
    """Resolve the framework Container from `app.state.container`.

    Spec 4A.0.7 forbids globals — the Container is attached to the FastAPI
    application by `Engine.fastapi_app(...)` and read here.
    """
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(
            status_code=500,
            detail="Container not attached to app.state — call Engine.fastapi_app()",
        )
    return container


def get_engine(request: Request) -> Engine:
    """Resolve the Engine from `app.state.engine`."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=500,
            detail="Engine not attached to app.state — call Engine.fastapi_app()",
        )
    return engine


def get_tenant_id(request: Request) -> UUID:
    """Resolve the tenant for this request.

    Order:
      1. If `app.state.tenant_resolver` is set, call it (app-defined auth wins).
      2. Otherwise read `X-Tenant-Id` header and parse as UUID.

    Raises 400 if neither path yields a valid UUID.
    """
    resolver: TenantResolver | None = getattr(
        request.app.state, "tenant_resolver", None,
    )
    if resolver is not None:
        return resolver(request)
    raw = request.headers.get("X-Tenant-Id")
    if not raw:
        raise HTTPException(status_code=400, detail="X-Tenant-Id header required")
    try:
        return UUID(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="X-Tenant-Id must be a UUID",
        ) from exc
```

`src/ballast/api/health.py`:

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import APIRouter

HealthCheck = Callable[[], Awaitable[bool]]


def build_health_router(
    *,
    prefix: str = "",
    checks: dict[str, HealthCheck] | None = None,
) -> APIRouter:
    """Mount `GET {prefix}/healthz` with optional per-component checks.

    `checks` map name -> async callable returning True on healthy. Failure
    flips the overall status to "degraded" with per-check error strings.
    """
    router = APIRouter(prefix=prefix)
    cs = dict(checks or {})

    @router.get("/healthz")
    async def healthz() -> dict[str, Any]:
        if not cs:
            return {"status": "ok"}
        results: dict[str, str] = {}
        overall = "ok"
        for name, fn in cs.items():
            try:
                ok = await fn()
            except Exception as exc:  # pragma: no cover - defensive
                results[name] = f"error: {exc}"
                overall = "degraded"
                continue
            results[name] = "ok" if ok else "fail"
            if not ok:
                overall = "degraded"
        return {"status": overall, "checks": results}

    return router
```

- [ ] **Step 4: Tests pass (10 new: 7 deps + 3 health)**
- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/ballast/api/ tests/api/__init__.py tests/api/test_deps.py tests/api/test_health.py
git commit -m "feat(api): DI helpers (get_container/engine/tenant) + /healthz router"
```

---

## Task 2: Thread CRUD endpoints — `build_threads_router`

Wires the existing `ThreadRepository` (SP2) into REST: `POST /threads`, `GET /threads/{id}`, `GET /threads/{id}/messages`. Tenant scoping is enforced — every call passes through `get_tenant_id`. No streaming here (Task 3).

**Baseline:** 309 → **Target:** 315 (+6).

**Files:**
- Create: `src/ballast/api/threads.py`
- Create: `tests/api/test_threads_router.py`
- Modify: `src/ballast/api/__init__.py` — export `build_threads_router`.

- [ ] **Step 1: Failing tests**

`tests/api/test_threads_router.py`:

```python
from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.api.threads import build_threads_router
from ballast.persistence.thread.repository import (
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
```

- [ ] **Step 2: Run → fail (ImportError)**

```bash
uv run pytest tests/api/test_threads_router.py -v
```

- [ ] **Step 3: Implement**

`src/ballast/api/threads.py`:

```python
from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ballast.api.deps import get_tenant_id
from ballast.persistence.thread.repository import ThreadRepository


class CreateThreadBody(BaseModel):
    purpose: str
    purpose_metadata: dict[str, Any] = Field(default_factory=dict)
    actor_id: str


def build_threads_router(
    *,
    thread_repo: ThreadRepository,
    prefix: str = "",
) -> APIRouter:
    """REST surface for the Thread aggregate (SP2)."""
    router = APIRouter(prefix=prefix)

    @router.post("/threads", status_code=201)
    async def create_thread(
        body: CreateThreadBody, tenant_id: UUID = Depends(get_tenant_id),
    ) -> dict[str, Any]:
        thread = await thread_repo.create(
            purpose=body.purpose,
            purpose_metadata=body.purpose_metadata,
            actor_id=body.actor_id,
            tenant_id=tenant_id,
        )
        return thread.model_dump(mode="json")

    @router.get("/threads/{thread_id}")
    async def get_thread(
        thread_id: UUID, tenant_id: UUID = Depends(get_tenant_id),
    ) -> dict[str, Any]:
        thread = await thread_repo.load(thread_id, tenant_id=tenant_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        return thread.model_dump(mode="json")

    @router.get("/threads/{thread_id}/messages")
    async def get_messages(
        thread_id: UUID,
        tenant_id: UUID = Depends(get_tenant_id),
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        thread = await thread_repo.load(thread_id, tenant_id=tenant_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        msgs = await thread_repo.history(
            thread_id, tenant_id=tenant_id, limit=limit,
        )
        return [m.model_dump(mode="json") for m in msgs]

    return router
```

Extend `src/ballast/api/__init__.py` exports.

- [ ] **Step 4: Tests pass (6 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/api/threads.py src/ballast/api/__init__.py tests/api/test_threads_router.py
git commit -m "feat(api): threads router — POST /threads, GET /threads/{id}, GET /threads/{id}/messages"
```

---

## Task 3: Streaming endpoint + AG-UI encoder

`AGUIEncoder` is a content formatter — pure function `encode(event: StreamEvent) -> bytes` that emits SSE event lines (`event: ...\ndata: ...\n\n`). The router wraps an async iterator (the agent's stream) in `StreamingResponse(media_type="text/event-stream")` and pipes events through the encoder. The agent is supplied via a callable `agent_runner(thread_id, message, tenant_id) -> AsyncIterator[StreamEvent]` so tests can pass a fake iterator without needing a real `pydantic-ai` model.

**Baseline:** 315 → **Target:** 322 (+7).

**Files:**
- Create: `src/ballast/api/streaming/__init__.py`
- Create: `src/ballast/api/streaming/ag_ui.py`
- Create: `src/ballast/api/streaming/router.py`
- Create: `tests/api/test_streaming_ag_ui.py`

- [ ] **Step 1: Failing tests**

`tests/api/test_streaming_ag_ui.py`:

```python
from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.api.streaming.ag_ui import AGUIEncoder
from ballast.api.streaming.router import (
    StreamEvent,
    build_streaming_router,
)
from ballast.persistence.thread.repository import (
    InMemoryThreadRepository,
)


def test_ag_ui_encoder_emits_sse_frame_for_text_delta():
    enc = AGUIEncoder()
    frame = enc.encode(StreamEvent(kind="text_delta", data={"text": "hi"}))
    text = frame.decode("utf-8")
    assert text.startswith("event: text_delta\n")
    assert "data: " in text
    assert text.endswith("\n\n")


def test_ag_ui_encoder_emits_done_event():
    enc = AGUIEncoder()
    frame = enc.encode(StreamEvent(kind="done", data={}))
    assert b"event: done" in frame


def test_ag_ui_encoder_escapes_newlines_in_data():
    """SSE data lines MUST NOT contain raw \\n — JSON-encode payload."""
    enc = AGUIEncoder()
    frame = enc.encode(StreamEvent(kind="text_delta", data={"text": "a\nb"}))
    text = frame.decode("utf-8")
    # Newline inside the JSON literal must be escaped (\\n), so only the
    # event/data separator newlines remain.
    assert text.count("\n") == 3  # event:, data:, blank terminator


@pytest.mark.asyncio
async def test_streaming_endpoint_streams_events_as_sse():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(
        *, thread_id, message, tenant_id,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="text_delta", data={"text": "he"})
        yield StreamEvent(kind="text_delta", data={"text": "llo"})
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"kind": "text", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body_text = r.text
    assert "event: text_delta" in body_text
    assert "event: done" in body_text


@pytest.mark.asyncio
async def test_streaming_endpoint_persists_user_message_before_streaming():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"kind": "text", "text": "hello"}]}
    with TestClient(app) as c:
        c.post(
            f"/threads/{th.id}/messages",
            json=body,
            headers={"X-Tenant-Id": str(tid)},
        )
    msgs = await repo.history(th.id, tenant_id=tid)
    assert len(msgs) == 1
    assert msgs[0].role == "user"


@pytest.mark.asyncio
async def test_streaming_endpoint_404_when_thread_missing():
    repo = InMemoryThreadRepository()

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"kind": "text", "text": "x"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{uuid4()}/messages", json=body,
            headers={"X-Tenant-Id": str(uuid4())},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_streaming_endpoint_404_cross_tenant():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    other = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(
        build_streaming_router(thread_repo=repo, agent_runner=runner),
    )
    body = {"role": "user", "parts": [{"kind": "text", "text": "x"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages", json=body,
            headers={"X-Tenant-Id": str(other)},
        )
    assert r.status_code == 404
```

- [ ] **Step 2: Run → fail (ImportError)**

- [ ] **Step 3: Implement**

`src/ballast/api/streaming/__init__.py`:

```python
from ballast.api.streaming.ag_ui import AGUIEncoder
from ballast.api.streaming.router import (
    StreamEncoder,
    StreamEvent,
    build_streaming_router,
)

__all__ = ["AGUIEncoder", "StreamEncoder", "StreamEvent", "build_streaming_router"]
```

`src/ballast/api/streaming/ag_ui.py`:

```python
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ballast.api.streaming.router import StreamEvent


class AGUIEncoder:
    """Encode framework `StreamEvent`s as AG-UI SSE frames.

    Spec 1.13: AG-UI = UI streaming protocol. Encoder is a CONTENT FORMATTER
    — does not own the transport. Frames look like:

        event: text_delta
        data: {"text": "hi"}
        <blank line>

    JSON-encoding the data payload guarantees no raw newlines inside `data:`
    lines (SSE requires single-line data values).
    """

    media_type = "text/event-stream"

    def encode(self, event: StreamEvent) -> bytes:
        payload = json.dumps(event.data, separators=(",", ":"))
        return f"event: {event.kind}\ndata: {payload}\n\n".encode()
```

`src/ballast/api/streaming/router.py`:

```python
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ballast.api.deps import get_tenant_id
from ballast.api.streaming.ag_ui import AGUIEncoder
from ballast.persistence.thread.repository import ThreadRepository


class StreamEvent(BaseModel):
    """Protocol-neutral streaming event emitted by the agent runner."""
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


class StreamEncoder(Protocol):
    media_type: str
    def encode(self, event: StreamEvent) -> bytes: ...


class _PostMessageBody(BaseModel):
    role: str = "user"
    parts: list[dict[str, Any]] = Field(default_factory=list)


AgentRunner = Callable[..., AsyncIterator[StreamEvent]]


def build_streaming_router(
    *,
    thread_repo: ThreadRepository,
    agent_runner: AgentRunner,
    encoder: StreamEncoder | None = None,
    prefix: str = "",
) -> APIRouter:
    """Mount `POST {prefix}/threads/{id}/messages` as an SSE stream.

    `agent_runner` is a callable returning an async iterator of `StreamEvent`s.
    Provide a fake in tests; production wires it to `agent.run_stream(...)` /
    `agent.iter(...)`. The user message is persisted BEFORE the stream starts
    so a client crash mid-stream still leaves the thread consistent.
    """
    router = APIRouter(prefix=prefix)
    enc: StreamEncoder = encoder or AGUIEncoder()

    @router.post("/threads/{thread_id}/messages")
    async def post_message(
        thread_id: UUID,
        body: _PostMessageBody,
        tenant_id: UUID = Depends(get_tenant_id),
    ) -> StreamingResponse:
        thread = await thread_repo.load(thread_id, tenant_id=tenant_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        await thread_repo.add_message(
            thread_id, role=body.role, parts=body.parts, tenant_id=tenant_id,
        )

        async def _gen() -> AsyncIterator[bytes]:
            async for event in agent_runner(
                thread_id=thread_id, message=body, tenant_id=tenant_id,
            ):
                yield enc.encode(event)

        return StreamingResponse(_gen(), media_type=enc.media_type)

    return router
```

- [ ] **Step 4: Tests pass (7 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/api/streaming/ tests/api/test_streaming_ag_ui.py
git commit -m "feat(api): SSE streaming endpoint + AG-UI encoder (content formatter, not transport)"
```

---

## Task 4: Vercel AI SDK encoder

Alternate `VercelEncoder` for the same `StreamEvent`. Vercel SDK uses a different line protocol (`0:"text"` for content, `d:` for done, JSON values, single newline per record). The same router accepts `?protocol=vercel` to swap encoders.

**Baseline:** 322 → **Target:** 328 (+6).

**Files:**
- Create: `src/ballast/api/streaming/vercel.py`
- Modify: `src/ballast/api/streaming/router.py` — accept `?protocol=ag-ui|vercel` query param.
- Modify: `src/ballast/api/streaming/__init__.py` — export `VercelEncoder`.
- Create: `tests/api/test_streaming_vercel.py`

- [ ] **Step 1: Failing tests**

`tests/api/test_streaming_vercel.py`:

```python
from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.api.streaming.router import (
    StreamEvent,
    build_streaming_router,
)
from ballast.api.streaming.vercel import VercelEncoder
from ballast.persistence.thread.repository import (
    InMemoryThreadRepository,
)


def test_vercel_encoder_text_delta_line():
    enc = VercelEncoder()
    frame = enc.encode(StreamEvent(kind="text_delta", data={"text": "hi"}))
    assert frame == b'0:"hi"\n'


def test_vercel_encoder_done_line():
    enc = VercelEncoder()
    frame = enc.encode(
        StreamEvent(kind="done", data={"finish_reason": "stop"}),
    )
    text = frame.decode("utf-8")
    assert text.startswith("d:")
    assert text.endswith("\n")


def test_vercel_encoder_tool_call_line():
    enc = VercelEncoder()
    frame = enc.encode(StreamEvent(
        kind="tool_call",
        data={"tool_call_id": "t1", "tool_name": "search", "args": {"q": "x"}},
    ))
    assert frame.startswith(b"9:")


@pytest.mark.asyncio
async def test_router_selects_vercel_encoder_via_query():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="text_delta", data={"text": "hi"})
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": [{"kind": "text", "text": "x"}]}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages?protocol=vercel",
            json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 200
    text = r.text
    assert '0:"hi"' in text
    assert text.endswith("d:{}\n") or "d:" in text


@pytest.mark.asyncio
async def test_router_defaults_to_ag_ui():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="text_delta", data={"text": "hi"})

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": []}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages",
            json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert "event: text_delta" in r.text


@pytest.mark.asyncio
async def test_router_400_on_unknown_protocol():
    repo = InMemoryThreadRepository()
    tid = uuid4()
    th = await repo.create(
        purpose="conversation", purpose_metadata={}, actor_id="a", tenant_id=tid,
    )

    async def runner(**_kw) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(kind="done", data={})

    app = FastAPI()
    app.include_router(build_streaming_router(thread_repo=repo, agent_runner=runner))
    body = {"role": "user", "parts": []}
    with TestClient(app) as c:
        r = c.post(
            f"/threads/{th.id}/messages?protocol=ws",
            json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 400
```

- [ ] **Step 2: Run → fail (ImportError / 200 instead of 400)**

- [ ] **Step 3: Implement**

`src/ballast/api/streaming/vercel.py`:

```python
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ballast.api.streaming.router import StreamEvent

# Vercel AI SDK stream protocol — record prefixes per @ai-sdk/ui
_PREFIX = {
    "text_delta": "0",      # text part
    "tool_call": "9",       # tool call full
    "tool_call_delta": "c",
    "tool_result": "a",
    "error": "3",
    "done": "d",
}


class VercelEncoder:
    """Encode `StreamEvent`s in the Vercel AI SDK record format.

    One JSON-encoded record per line, prefixed with a single-char tag.
    """
    media_type = "text/plain; charset=utf-8"

    def encode(self, event: StreamEvent) -> bytes:
        prefix = _PREFIX.get(event.kind, "2")  # 2 = data event (unknown kinds)
        if event.kind == "text_delta":
            value = event.data.get("text", "")
            return f'{prefix}:{json.dumps(value)}\n'.encode()
        body = json.dumps(event.data, separators=(",", ":"))
        return f"{prefix}:{body}\n".encode()
```

Update `router.py` — add `protocol: str = "ag-ui"` query param:

```python
from fastapi import Query
from ballast.api.streaming.vercel import VercelEncoder

_ENCODERS: dict[str, type] = {"ag-ui": AGUIEncoder, "vercel": VercelEncoder}

@router.post("/threads/{thread_id}/messages")
async def post_message(
    thread_id: UUID,
    body: _PostMessageBody,
    tenant_id: UUID = Depends(get_tenant_id),
    protocol: str = Query(default="ag-ui"),
) -> StreamingResponse:
    if protocol not in _ENCODERS:
        raise HTTPException(status_code=400, detail=f"unknown protocol: {protocol}")
    chosen: StreamEncoder = encoder or _ENCODERS[protocol]()
    ...
```

- [ ] **Step 4: Tests pass (6 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/api/streaming/ tests/api/test_streaming_vercel.py
git commit -m "feat(api): Vercel AI SDK encoder + ?protocol= dispatcher"
```

---

## Task 5: A2A discovery + invoke endpoints

A2A is the inter-agent protocol (spec 1.13). Two endpoints:
- `GET /.well-known/agent.json` — agent card (JSON describing capabilities, name, endpoint).
- `POST /a2a/{agent_name}` — accepts `{"messages": [...]}` and routes to the named agent.

We do NOT depend on `pydantic-ai`'s `agent.to_a2a()` even if exposed in the installed version — the framework is a thin contract over agents the app supplies. Apps supply a mapping `{agent_name: A2AAgentAdapter}` where `A2AAgentAdapter.run(messages, tenant_id) -> dict`. Production apps wrap their `Agent.run(...)` in such an adapter (and may delegate to `agent.to_a2a()` if they choose).

**Baseline:** 328 → **Target:** 333 (+5).

**Files:**
- Create: `src/ballast/api/a2a.py`
- Create: `tests/api/test_a2a_router.py`
- Modify: `src/ballast/api/__init__.py` — export `build_a2a_router`, `A2AAgentAdapter`, `AgentCard`.

- [ ] **Step 1: Failing tests**

`tests/api/test_a2a_router.py`:

```python
from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.api.a2a import (
    A2AAgentAdapter,
    AgentCard,
    build_a2a_router,
)


class _EchoAgent:
    """Minimal A2AAgentAdapter for tests."""
    name = "echo"
    description = "echoes the last message"

    async def run(
        self, *, messages: list[dict[str, Any]], tenant_id: UUID,
    ) -> dict[str, Any]:
        return {"echo": messages[-1] if messages else None, "tenant": str(tenant_id)}


def test_well_known_agent_json_returns_cards():
    app = FastAPI()
    app.include_router(build_a2a_router(agents={"echo": _EchoAgent()}))
    with TestClient(app) as c:
        r = c.get("/.well-known/agent.json")
    assert r.status_code == 200
    body = r.json()
    assert "agents" in body
    assert any(card["name"] == "echo" for card in body["agents"])


def test_well_known_agent_json_card_includes_endpoint():
    app = FastAPI()
    app.include_router(build_a2a_router(agents={"echo": _EchoAgent()}))
    with TestClient(app) as c:
        r = c.get("/.well-known/agent.json")
    card = next(c for c in r.json()["agents"] if c["name"] == "echo")
    assert card["endpoint"].endswith("/a2a/echo")


@pytest.mark.asyncio
async def test_a2a_invoke_routes_to_agent():
    app = FastAPI()
    app.include_router(build_a2a_router(agents={"echo": _EchoAgent()}))
    tid = uuid4()
    body = {"messages": [{"role": "user", "text": "hi"}]}
    with TestClient(app) as c:
        r = c.post(
            "/a2a/echo", json=body, headers={"X-Tenant-Id": str(tid)},
        )
    assert r.status_code == 200
    payload = r.json()
    assert payload["echo"]["text"] == "hi"
    assert payload["tenant"] == str(tid)


@pytest.mark.asyncio
async def test_a2a_invoke_404_when_unknown_agent():
    app = FastAPI()
    app.include_router(build_a2a_router(agents={"echo": _EchoAgent()}))
    with TestClient(app) as c:
        r = c.post(
            "/a2a/ghost",
            json={"messages": []},
            headers={"X-Tenant-Id": str(uuid4())},
        )
    assert r.status_code == 404


def test_agent_card_includes_optional_metadata():
    """Cards carry capabilities + description so discovery is useful."""
    card = AgentCard(
        name="planner", description="plans things",
        endpoint="/a2a/planner",
        capabilities=["plan", "decompose"],
    )
    assert "plan" in card.capabilities
    assert card.description == "plans things"
```

- [ ] **Step 2: Run → fail (ImportError)**

- [ ] **Step 3: Implement**

`src/ballast/api/a2a.py`:

```python
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ballast.api.deps import get_tenant_id


@runtime_checkable
class A2AAgentAdapter(Protocol):
    """Minimal contract for an agent exposed over A2A.

    Apps may delegate to pydantic-ai's `agent.to_a2a()` from inside `run`
    if the installed version exposes it; the framework does not require it.
    """
    name: str
    description: str

    async def run(
        self, *, messages: list[dict[str, Any]], tenant_id: UUID,
    ) -> dict[str, Any]: ...


class AgentCard(BaseModel):
    name: str
    description: str
    endpoint: str
    capabilities: list[str] = Field(default_factory=list)


class _InvokeBody(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)


def build_a2a_router(
    *,
    agents: dict[str, A2AAgentAdapter],
    prefix: str = "",
) -> APIRouter:
    """Mount A2A discovery (`/.well-known/agent.json`) + invoke (`/a2a/{name}`)."""
    router = APIRouter(prefix=prefix)
    registry = dict(agents)

    @router.get("/.well-known/agent.json")
    async def agent_cards(request: Request) -> dict[str, Any]:
        cards = [
            AgentCard(
                name=name,
                description=getattr(adapter, "description", ""),
                endpoint=f"{prefix}/a2a/{name}",
                capabilities=list(getattr(adapter, "capabilities", [])),
            ).model_dump()
            for name, adapter in registry.items()
        ]
        return {"agents": cards}

    @router.post("/a2a/{agent_name}")
    async def invoke(
        agent_name: str,
        body: _InvokeBody,
        tenant_id: UUID = Depends(get_tenant_id),
    ) -> dict[str, Any]:
        adapter = registry.get(agent_name)
        if adapter is None:
            raise HTTPException(status_code=404, detail=f"unknown agent: {agent_name}")
        return await adapter.run(messages=body.messages, tenant_id=tenant_id)

    return router
```

- [ ] **Step 4: Tests pass (5 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/api/a2a.py src/ballast/api/__init__.py tests/api/test_a2a_router.py
git commit -m "feat(api): A2A discovery (/.well-known/agent.json) + invoke (/a2a/{name})"
```

---

## Task 6: `Engine.fastapi_app(...)` factory

Wires everything: builds a `FastAPI`, attaches `Container` and the `Engine` itself to `app.state`, registers a lifespan hook that calls `engine.boot()` lazily (only if not already booted), and mounts the routers the caller supplies.

**Baseline:** 333 → **Target:** 338 (+5).

**Files:**
- Modify: `src/ballast/runtime/engine.py` — add `fastapi_app(...)` method.
- Modify: `src/ballast/runtime/__init__.py` — no new export (method on Engine).
- Create: `tests/api/test_engine_fastapi_app.py`

- [ ] **Step 1: Failing tests**

`tests/api/test_engine_fastapi_app.py`:

```python
from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from ballast.runtime import Engine


class _RecordingProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def register(self, container) -> None:
        self.calls += 1


def test_fastapi_app_attaches_container_and_engine_to_state():
    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app()
    assert app.state.container is engine.container
    assert app.state.engine is engine


def test_fastapi_app_mounts_healthz_by_default():
    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app()
    with TestClient(app) as c:
        r = c.get("/healthz")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_fastapi_app_lifespan_boots_engine_once():
    prov = _RecordingProvider()
    engine = Engine(providers=[prov])
    app = engine.fastapi_app()
    with TestClient(app):  # triggers lifespan startup
        pass
    assert prov.calls == 1
    # Re-entering does NOT double-boot.
    with TestClient(app):
        pass
    assert prov.calls == 1


def test_fastapi_app_mounts_extra_routers():
    engine = Engine(providers=[_RecordingProvider()])
    extra = APIRouter()

    @extra.get("/custom")
    async def custom() -> dict[str, str]:
        return {"hi": "there"}

    app = engine.fastapi_app(extra_routers=[extra])
    with TestClient(app) as c:
        r = c.get("/custom")
    assert r.status_code == 200


def test_fastapi_app_does_not_attach_observability_by_default():
    """ObservabilityProvider is opt-in; instrument_fastapi must NOT
    run unless explicitly enabled (Task 7 enables it via the provider)."""
    engine = Engine(providers=[_RecordingProvider()])
    app = engine.fastapi_app()
    # The app builds fine and answers /healthz — no logfire wiring.
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200
```

- [ ] **Step 2: Run → fail (`Engine` has no `fastapi_app`)**

- [ ] **Step 3: Implement** — extend `engine.py`:

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import APIRouter, FastAPI

from ballast.api.health import build_health_router


class Engine:
    ...

    def fastapi_app(
        self,
        *,
        extra_routers: list[APIRouter] | None = None,
        health_checks: dict | None = None,
    ) -> FastAPI:
        """Build a FastAPI app with the Container/Engine wired in.

        - Attaches `app.state.container` and `app.state.engine` (spec 4A.0.7).
        - Registers a lifespan that calls `engine.boot()` once (idempotent
          guard against double-boot on re-entry).
        - Mounts `/healthz` by default.
        - Mounts any `extra_routers` provided.

        Observability is NOT auto-attached here — install via
        `ObservabilityProvider` in the provider list (spec 4H: provider order).
        """
        @asynccontextmanager
        async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
            if not self._booted:
                await self.boot()
            yield

        app = FastAPI(lifespan=_lifespan)
        app.state.container = self.container
        app.state.engine = self
        app.include_router(build_health_router(checks=health_checks))
        for r in extra_routers or []:
            app.include_router(r)
        return app
```

- [ ] **Step 4: Tests pass (5 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/runtime/engine.py tests/api/test_engine_fastapi_app.py
git commit -m "feat(runtime): Engine.fastapi_app() — Container/Engine on app.state + lifespan boot"
```

---

## Task 7: `ObservabilityProvider` + logfire soft import

`ObservabilityProvider` is a `ServiceProvider` that calls `logfire.configure(...)` and the relevant `instrument_*` shims. logfire is **soft-imported** — when absent, the provider becomes a no-op (so the test suite doesn't require `logfire`). Spec 4H mandates it be registered FIRST; we enforce this with a constructor-time `_observability_first_invariant(...)` that fails if any other provider was already invoked.

**Baseline:** 338 → **Target:** 344 (+6).

**Files:**
- Create: `src/ballast/observability/__init__.py`
- Create: `src/ballast/observability/provider.py`
- Create: `tests/observability/__init__.py`
- Create: `tests/observability/test_provider.py`

- [ ] **Step 1: Failing tests**

`tests/observability/test_provider.py`:

```python
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from ballast.observability import ObservabilityProvider, has_logfire
from ballast.runtime import Engine


class _Spy:
    def __init__(self) -> None:
        self.calls = 0

    async def register(self, _container) -> None:
        self.calls += 1


@pytest.mark.asyncio
async def test_provider_is_noop_when_logfire_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "logfire", None)  # simulate absence
    engine = Engine(providers=[ObservabilityProvider(service_name="t")])
    await engine.boot()
    assert engine._booted is True  # boots successfully despite no logfire


@pytest.mark.asyncio
async def test_provider_calls_logfire_configure_when_present(monkeypatch):
    fake = MagicMock()
    fake.configure = MagicMock()
    fake.instrument_pydantic_ai = MagicMock()
    fake.instrument_httpx = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    engine = Engine(providers=[
        ObservabilityProvider(service_name="svc", environment="test"),
    ])
    await engine.boot()
    fake.configure.assert_called_once()
    kwargs = fake.configure.call_args.kwargs
    assert kwargs["service_name"] == "svc"
    assert kwargs["environment"] == "test"


@pytest.mark.asyncio
async def test_provider_instruments_pydantic_ai_and_httpx_when_enabled(monkeypatch):
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    engine = Engine(providers=[
        ObservabilityProvider(
            service_name="svc",
            instrument_pydantic_ai=True,
            instrument_httpx=True,
        ),
    ])
    await engine.boot()
    fake.instrument_pydantic_ai.assert_called_once()
    fake.instrument_httpx.assert_called_once()


def test_provider_first_invariant_fails_if_other_provider_already_registered():
    """Spec 4H — ObservabilityProvider must be first in the list."""
    from ballast import EngineInvariantViolation

    engine = Engine(providers=[_Spy(), ObservabilityProvider(service_name="t")])
    with pytest.raises(EngineInvariantViolation, match="first"):
        import asyncio
        asyncio.get_event_loop().run_until_complete(engine.boot())


def test_has_logfire_returns_bool():
    assert isinstance(has_logfire(), bool)


@pytest.mark.asyncio
async def test_provider_skips_instrument_fastapi_unless_app_given(monkeypatch):
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "logfire", fake)
    engine = Engine(providers=[
        ObservabilityProvider(service_name="svc"),
    ])
    await engine.boot()
    fake.instrument_fastapi.assert_not_called()
```

- [ ] **Step 2: Run → fail (ImportError)**

- [ ] **Step 3: Implement**

`src/ballast/observability/__init__.py`:

```python
from ballast.observability.provider import (
    ObservabilityProvider,
    has_logfire,
)

__all__ = ["ObservabilityProvider", "has_logfire"]
```

`src/ballast/observability/provider.py`:

```python
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ballast.runtime.engine import EngineInvariantViolation

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ballast.runtime.container import Container


def has_logfire() -> bool:
    """Soft import — True iff `logfire` is importable in this process."""
    try:
        mod = importlib.import_module("logfire")
        return mod is not None
    except Exception:
        return False


class ObservabilityProvider:
    """Configures logfire (when present) and registers the `must-be-first`
    bootstrap invariant.

    Soft dependency: if `logfire` is not installed, every method is a no-op
    so the test suite (and applications that don't want telemetry) keep
    working. Spec 4D, 4H.
    """

    def __init__(
        self,
        *,
        service_name: str = "ballast-ai",
        environment: str = "dev",
        instrument_pydantic_ai: bool = True,
        instrument_httpx: bool = True,
        instrument_fastapi_app: FastAPI | None = None,
        instrument_sqlalchemy_engine: AsyncEngine | None = None,
        configure_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.service_name = service_name
        self.environment = environment
        self._instr_pai = instrument_pydantic_ai
        self._instr_httpx = instrument_httpx
        self._fastapi_app = instrument_fastapi_app
        self._sa_engine = instrument_sqlalchemy_engine
        self._configure_kwargs = dict(configure_kwargs or {})

    async def register(self, container: Container) -> None:
        # Spec 4H invariant — observability registers FIRST. The container
        # snitches via a flag we set ourselves; any prior provider would
        # have already created a binding.
        if getattr(container, "_observability_first_violated", False):
            raise EngineInvariantViolation(
                "ObservabilityProvider must register first (spec 4H).",
            )
        if not has_logfire():
            # mark so subsequent providers don't trip the invariant
            container._observability_registered = True  # type: ignore[attr-defined]
            return
        import logfire  # noqa: WPS433  (soft import)
        logfire.configure(
            service_name=self.service_name,
            environment=self.environment,
            **self._configure_kwargs,
        )
        if self._instr_pai and hasattr(logfire, "instrument_pydantic_ai"):
            logfire.instrument_pydantic_ai()
        if self._instr_httpx and hasattr(logfire, "instrument_httpx"):
            logfire.instrument_httpx()
        if self._fastapi_app is not None and hasattr(logfire, "instrument_fastapi"):
            logfire.instrument_fastapi(self._fastapi_app)
        if self._sa_engine is not None and hasattr(logfire, "instrument_sqlalchemy"):
            logfire.instrument_sqlalchemy(engine=self._sa_engine)
        container._observability_registered = True  # type: ignore[attr-defined]
```

To make the "first" invariant testable, modify `Engine.boot()` to set `container._observability_first_violated = True` before invoking any non-`ObservabilityProvider`:

```python
async def boot(self) -> None:
    if self._booted:
        raise RuntimeError("Engine already booted")
    for provider in self._providers:
        if not isinstance(provider, ObservabilityProvider) and not getattr(
            self.container, "_observability_registered", False,
        ):
            # Mark — ObservabilityProvider's own register will raise
            # if it runs later.
            self.container._observability_first_violated = True
        await provider.register(self.container)
    ...
```

(Import `ObservabilityProvider` lazily inside `boot` to avoid an import cycle.)

- [ ] **Step 4: Tests pass (6 new)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/observability/ src/ballast/runtime/engine.py tests/observability/
git commit -m "feat(observability): ObservabilityProvider with soft logfire dep + first-in-list invariant"
```

---

## Task 8: `@traced` decorator + Pattern / Channel instrumentation

A `@traced(name, attrs=...)` helper that produces a span when logfire is present and is a transparent passthrough otherwise. Applied to `Reflection.run`, `MapReduce.run`, `MutationPipeline.run`, `HITLGate.run`, and `UIChannel.ask` / `WebhookChannel.ask` / `ConversationalChannel.ask`. Span names per spec 4D table.

Must NOT break existing pattern tests — `@traced` only adds context; behaviour is preserved.

**Baseline:** 344 → **Target:** 349 (+5).

**Files:**
- Create: `src/ballast/observability/spans.py`
- Modify: `src/ballast/observability/__init__.py` — export `traced`.
- Modify: `src/ballast/patterns/reflection.py` — wrap `run`.
- Modify: `src/ballast/patterns/mapreduce/pattern.py` — wrap `run`.
- Modify: `src/ballast/patterns/mutation/pipeline.py` — wrap `run` + per-stage span.
- Modify: `src/ballast/patterns/hitl/gate.py` — wrap `run`.
- Modify: `src/ballast/patterns/hitl/channels/{ui,webhook,conversational}.py` — wrap `ask`.
- Create: `tests/observability/test_traced.py`
- Create: `tests/observability/test_pattern_instrumentation.py`

- [ ] **Step 1: Failing tests**

`tests/observability/test_traced.py`:

```python
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from ballast.observability.spans import traced


@pytest.mark.asyncio
async def test_traced_passthrough_when_logfire_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "logfire", None)

    @traced("test.span")
    async def fn(x: int) -> int:
        return x * 2

    assert await fn(3) == 6


@pytest.mark.asyncio
async def test_traced_emits_span_when_logfire_present(monkeypatch):
    fake = MagicMock()
    ctx_mgr = MagicMock()
    ctx_mgr.__enter__ = MagicMock(return_value=ctx_mgr)
    ctx_mgr.__exit__ = MagicMock(return_value=False)
    fake.span = MagicMock(return_value=ctx_mgr)
    monkeypatch.setitem(sys.modules, "logfire", fake)

    @traced("test.span")
    async def fn() -> str:
        return "ok"

    assert await fn() == "ok"
    fake.span.assert_called_once()
    args, _ = fake.span.call_args
    assert args[0] == "test.span"


@pytest.mark.asyncio
async def test_traced_attaches_attributes_lambda(monkeypatch):
    fake = MagicMock()
    ctx_mgr = MagicMock()
    ctx_mgr.__enter__ = MagicMock(return_value=ctx_mgr)
    ctx_mgr.__exit__ = MagicMock(return_value=False)
    fake.span = MagicMock(return_value=ctx_mgr)
    monkeypatch.setitem(sys.modules, "logfire", fake)

    @traced("test.span", attrs=lambda x: {"x": x})
    async def fn(x: int) -> int:
        return x

    await fn(42)
    _, kwargs = fake.span.call_args
    assert kwargs == {"x": 42} or kwargs.get("_tags") is not None or kwargs == {"x": 42}


@pytest.mark.asyncio
async def test_traced_propagates_exceptions(monkeypatch):
    monkeypatch.setitem(sys.modules, "logfire", None)

    @traced("test.span")
    async def boom() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        await boom()
```

`tests/observability/test_pattern_instrumentation.py`:

```python
"""Smoke: the wrapped patterns still satisfy their old contract.

We don't assert on logfire output (covered in test_traced); we assert that
existing pattern behaviour is unchanged after wrapping `.run` with @traced.
"""
from __future__ import annotations

import pytest

from ballast.capabilities.helpers import Critique
from ballast.patterns import Reflection


@pytest.mark.asyncio
async def test_reflection_run_still_works_after_instrumentation():
    from uuid import uuid4
    calls = {"writer": 0, "critic": 0}

    async def writer(task: str) -> str:
        calls["writer"] += 1
        return f"draft:{task}"

    async def critic(draft: str) -> Critique:
        calls["critic"] += 1
        return Critique(passed=True, feedback="ok")

    ref: Reflection[str, str] = Reflection(writer, critic, max_iterations=1)
    out = await ref.run("input", tenant_id=uuid4())
    assert out == "draft:input"
    assert calls == {"writer": 1, "critic": 1}
```

- [ ] **Step 2: Run → fail (ImportError on traced)**

- [ ] **Step 3: Implement**

`src/ballast/observability/spans.py`:

```python
from __future__ import annotations

import functools
import importlib
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

AttrsFn = Callable[..., dict[str, Any]]


@contextmanager
def _noop_span(_name: str, **_attrs: Any):
    yield None


def _get_logfire_span():
    try:
        mod = importlib.import_module("logfire")
        if mod is None:
            return None
        return getattr(mod, "span", None)
    except Exception:
        return None


def traced(
    name: str, *, attrs: AttrsFn | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Wrap an async function in a logfire span.

    No-op when logfire is missing — the wrapped function runs unchanged.
    `attrs` callable receives the wrapped function's args and returns a
    dict merged into the span attributes (canonical names per spec 4D).
    """

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            span_fn = _get_logfire_span()
            if span_fn is None:
                return await fn(*args, **kwargs)
            try:
                attributes = attrs(*args, **kwargs) if attrs else {}
            except Exception:
                attributes = {}
            with span_fn(name, **attributes):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator
```

For each pattern, add the decorator (example — `Reflection`):

```python
from ballast.observability.spans import traced

@DBOS.workflow()
@traced("pattern.reflection", attrs=lambda self, task, *, tenant_id: {
    "tenant_id": str(tenant_id), "pattern": self.name,
})
async def run(self, task: InT, *, tenant_id: UUID) -> OutT:
    ...
```

Apply analogous `traced(...)` on `MapReduce.run`, `MutationPipeline.run`, `HITLGate.run`, and `channel.ask` per the span table. For `MutationPipeline`, also wrap each `Stage.process` call site with `with span_fn("stage.<name>", ...)` (keep it lightweight — span name only).

- [ ] **Step 4: Tests pass (5 new — 4 traced + 1 smoke). Full pattern suite still green.**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/observability/ src/ballast/patterns/ tests/observability/test_traced.py tests/observability/test_pattern_instrumentation.py
git commit -m "feat(observability): @traced decorator + canonical span names on patterns/channels"
```

---

## Task 9: Evals primitives + `SchemaAdherenceScorer`

Pure types: `EvalCase`, `EvalRunOutput`, `Scorer` Protocol, `EvalReport`, `Dataset`. The MVP scorer is `SchemaAdherenceScorer` — score 1.0 if the run's output is a valid `BaseModel` (or `retries=0`), 0.0 otherwise. `Dataset.evaluate(runner, evaluators=[...])` runs the callable per case and returns an aggregated report. Pure in-memory — no DBOS, no DB.

**Baseline:** 349 → **Target:** 358 (+9).

**Files:**
- Create: `src/ballast/evals/__init__.py`
- Create: `src/ballast/evals/case.py`
- Create: `src/ballast/evals/scorer.py`
- Create: `src/ballast/evals/dataset.py`
- Create: `tests/evals/__init__.py`
- Create: `tests/evals/test_dataset.py`
- Create: `tests/evals/test_schema_adherence_scorer.py`

- [ ] **Step 1: Failing tests**

`tests/evals/test_dataset.py`:

```python
from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel

from ballast.evals import (
    Dataset,
    EvalCase,
    EvalReport,
    SchemaAdherenceScorer,
)


class _Out(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_dataset_evaluate_returns_report_with_per_case_scores():
    cases = [
        EvalCase(name="c1", inputs={"x": 1}, expected={}, metadata={}),
        EvalCase(name="c2", inputs={"x": 2}, expected={}, metadata={}),
    ]
    ds = Dataset(name="t", tenant_id=uuid4(), cases=cases)

    async def runner(inputs: dict[str, int]) -> _Out:
        return _Out(text=f"hi-{inputs['x']}")

    report = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer()])
    assert isinstance(report, EvalReport)
    assert len(report.case_scores) == 2
    assert all(0.0 <= cs.score <= 1.0 for cs in report.case_scores)


@pytest.mark.asyncio
async def test_dataset_report_aggregates_mean_per_scorer():
    cases = [EvalCase(name=f"c{i}", inputs={}, expected={}, metadata={}) for i in range(3)]
    ds = Dataset(name="t", tenant_id=uuid4(), cases=cases)

    async def runner(_inputs):
        return _Out(text="x")

    rep = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer()])
    assert "SchemaAdherenceScorer" in rep.scorer_means
    assert rep.scorer_means["SchemaAdherenceScorer"] == 1.0


@pytest.mark.asyncio
async def test_dataset_passed_respects_thresholds():
    cases = [EvalCase(name="c1", inputs={}, expected={}, metadata={})]
    ds = Dataset(name="t", tenant_id=uuid4(), cases=cases)

    async def runner(_inputs):
        return _Out(text="x")

    rep_ok = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer(threshold=0.5)])
    assert rep_ok.passed is True

    rep_bad = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer(threshold=2.0)])
    assert rep_bad.passed is False


@pytest.mark.asyncio
async def test_dataset_rejects_cross_tenant_metadata():
    """Spec 1.12: Dataset filtered by tenant_id; cross-tenant cases dropped."""
    tid = uuid4()
    other = uuid4()
    cases = [
        EvalCase(name="c1", inputs={}, expected={}, metadata={"tenant_id": str(tid)}),
        EvalCase(name="c2", inputs={}, expected={}, metadata={"tenant_id": str(other)}),
    ]
    ds = Dataset(name="t", tenant_id=tid, cases=cases)
    # Cross-tenant case excluded.
    assert len(ds.cases) == 1
    assert ds.cases[0].name == "c1"


@pytest.mark.asyncio
async def test_dataset_evaluate_captures_runner_exception_as_score_zero():
    cases = [EvalCase(name="c1", inputs={}, expected={}, metadata={})]
    ds = Dataset(name="t", tenant_id=uuid4(), cases=cases)

    async def runner(_inputs):
        raise RuntimeError("boom")

    rep = await ds.evaluate(runner, evaluators=[SchemaAdherenceScorer()])
    assert rep.case_scores[0].score == 0.0
    assert "boom" in (rep.case_scores[0].error or "")
```

`tests/evals/test_schema_adherence_scorer.py`:

```python
from __future__ import annotations

import pytest
from pydantic import BaseModel

from ballast.evals import EvalRunOutput, SchemaAdherenceScorer


class _Out(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_scorer_1_for_valid_basemodel_output():
    s = SchemaAdherenceScorer()
    score = await s.score(EvalRunOutput(output=_Out(text="x"), retries=0))
    assert score == 1.0


@pytest.mark.asyncio
async def test_scorer_0_when_retries_gt_zero():
    s = SchemaAdherenceScorer()
    score = await s.score(EvalRunOutput(output=_Out(text="x"), retries=2))
    assert score == 0.0


@pytest.mark.asyncio
async def test_scorer_0_when_output_is_none():
    s = SchemaAdherenceScorer()
    score = await s.score(EvalRunOutput(output=None, retries=0, error="bad"))
    assert score == 0.0


@pytest.mark.asyncio
async def test_scorer_0_for_non_basemodel_output():
    s = SchemaAdherenceScorer()
    score = await s.score(EvalRunOutput(output={"text": "x"}, retries=0))
    assert score == 0.0
```

- [ ] **Step 2: Run → fail (ImportError)**

- [ ] **Step 3: Implement**

`src/ballast/evals/__init__.py`:

```python
from ballast.evals.case import EvalCase, EvalRunOutput
from ballast.evals.dataset import Dataset, EvalReport, ScoreResult
from ballast.evals.scorer import SchemaAdherenceScorer, Scorer

__all__ = [
    "Dataset",
    "EvalCase",
    "EvalReport",
    "EvalRunOutput",
    "SchemaAdherenceScorer",
    "ScoreResult",
    "Scorer",
]
```

`src/ballast/evals/case.py`:

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvalCase(BaseModel):
    """A single eval input + expected output (when known)."""
    model_config = ConfigDict(frozen=True)
    name: str
    inputs: Any
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRunOutput(BaseModel):
    """What the runner produced for a case + any framework signals.

    `retries` is the BaseModel-level retry count from pydantic-ai
    (`run_result.retries`). 0 means structured output was valid first try.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    output: Any = None
    retries: int = 0
    error: str | None = None
```

`src/ballast/evals/scorer.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ballast.evals.case import EvalRunOutput


@runtime_checkable
class Scorer(Protocol):
    """Pluggable scorer — returns a float in [0.0, 1.0]."""
    threshold: float
    name: str

    async def score(self, run: EvalRunOutput) -> float: ...


class SchemaAdherenceScorer:
    """1.0 if the runner produced a valid BaseModel without retries."""
    name = "SchemaAdherenceScorer"

    def __init__(self, *, threshold: float = 0.95) -> None:
        self.threshold = threshold

    async def score(self, run: EvalRunOutput) -> float:
        if run.error is not None or run.output is None:
            return 0.0
        if run.retries > 0:
            return 0.0
        if not isinstance(run.output, BaseModel):
            return 0.0
        return 1.0
```

`src/ballast/evals/dataset.py`:

```python
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ballast.evals.case import EvalCase, EvalRunOutput
from ballast.evals.scorer import Scorer

Runner = Callable[[Any], Awaitable[Any]] | Callable[[Any], Any]


class ScoreResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    case_name: str
    scorer_name: str
    score: float
    error: str | None = None


class EvalReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_name: str
    case_scores: list[ScoreResult]
    scorer_means: dict[str, float] = Field(default_factory=dict)
    passed: bool = True


class Dataset:
    """A collection of `EvalCase`s scoped to a single tenant.

    Cross-tenant cases (metadata `tenant_id` mismatch) are dropped at
    construction — spec 1.12 forbids cross-tenant eval mixing.
    """

    def __init__(
        self,
        *,
        name: str,
        tenant_id: UUID,
        cases: list[EvalCase],
    ) -> None:
        self.name = name
        self.tenant_id = tenant_id
        self.cases: list[EvalCase] = [
            c for c in cases
            if c.metadata.get("tenant_id") in (None, str(tenant_id))
        ]

    async def evaluate(
        self, runner: Runner, *, evaluators: list[Scorer],
    ) -> EvalReport:
        rows: list[ScoreResult] = []
        for case in self.cases:
            try:
                result = runner(case.inputs)
                if inspect.isawaitable(result):
                    output = await result
                else:
                    output = result
                run_out = EvalRunOutput(output=output, retries=0)
            except Exception as exc:
                run_out = EvalRunOutput(output=None, retries=0, error=str(exc))
            for scorer in evaluators:
                score = await scorer.score(run_out)
                rows.append(ScoreResult(
                    case_name=case.name, scorer_name=scorer.name,
                    score=score, error=run_out.error,
                ))
        means = self._aggregate(rows)
        passed = all(
            means.get(s.name, 0.0) >= s.threshold for s in evaluators
        )
        return EvalReport(
            dataset_name=self.name, case_scores=rows,
            scorer_means=means, passed=passed,
        )

    @staticmethod
    def _aggregate(rows: list[ScoreResult]) -> dict[str, float]:
        by_scorer: dict[str, list[float]] = {}
        for r in rows:
            by_scorer.setdefault(r.scorer_name, []).append(r.score)
        return {k: sum(v) / len(v) for k, v in by_scorer.items() if v}
```

- [ ] **Step 4: Tests pass (9 new: 5 dataset + 4 scorer)**
- [ ] **Step 5: Full suite + mypy + ruff**
- [ ] **Step 6: Commit**

```bash
git add src/ballast/evals/ tests/evals/
git commit -m "feat(evals): EvalCase/Dataset/EvalReport + SchemaAdherenceScorer MVP"
```

---

## Task 10: `dataset-from-traces` CLI + SP7 public API + smoke

`stateflow evals dataset-from-traces --since --pattern --tenant --out` reads from the eval-related rows we already have (HITL `Decision` audit, proposal audit if present, thread history) and emits a YAML `Dataset` snapshot. We do NOT rely on a live DBOS state DB for this MVP — apps inject a `TraceSource` Protocol so the same CLI works against in-memory or Postgres. Typer-based entry point invoked via `python -m ballast.evals.cli ...`.

Also: top-level package exports for SP7, and an end-to-end smoke test that boots `Engine.fastapi_app(...)`, posts a thread + a message, runs a `Reflection`, and feeds the run output through a `Dataset` eval.

**Baseline:** 358 → **Target:** 365 (+7).

**Files:**
- Create: `src/ballast/evals/traces.py`
- Create: `src/ballast/evals/cli.py`
- Modify: `src/ballast/__init__.py` — add SP7 exports.
- Create: `tests/evals/test_dataset_from_traces.py`
- Create: `tests/evals/test_cli.py`
- Create: `tests/test_public_api_sp7.py`
- Modify: `pyproject.toml` — add `typer` to dependencies (or use `argparse` if user wants zero new deps — see note).

> **Note on deps:** Spec doesn't mandate Typer; if avoiding the new dep is preferred, swap `typer` for stdlib `argparse` — tests below use shell semantics that work for either. Plan ships Typer for ergonomics.

- [ ] **Step 1: Failing tests**

`tests/evals/test_dataset_from_traces.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from ballast.evals.traces import (
    InMemoryTraceSource,
    TraceRecord,
    dataset_from_traces,
)


@pytest.mark.asyncio
async def test_dataset_from_traces_filters_by_tenant_and_pattern():
    tid = uuid4()
    other = uuid4()
    now = datetime.now(tz=UTC)
    src = InMemoryTraceSource(records=[
        TraceRecord(
            run_id=uuid4(), tenant_id=tid, pattern="reflection",
            inputs={"x": 1}, output={"text": "y"},
            created_at=now, outcome="success",
        ),
        TraceRecord(
            run_id=uuid4(), tenant_id=other, pattern="reflection",
            inputs={"x": 2}, output={"text": "z"},
            created_at=now, outcome="success",
        ),
        TraceRecord(
            run_id=uuid4(), tenant_id=tid, pattern="mapreduce",
            inputs={"x": 3}, output={"text": "a"},
            created_at=now, outcome="success",
        ),
    ])
    ds = await dataset_from_traces(
        src, tenant_id=tid, pattern="reflection",
        since=now - timedelta(days=1),
    )
    assert ds.name == "reflection-traces"
    assert ds.tenant_id == tid
    assert len(ds.cases) == 1
    assert ds.cases[0].inputs == {"x": 1}


@pytest.mark.asyncio
async def test_dataset_from_traces_excludes_pre_since_records():
    tid = uuid4()
    old = datetime.now(tz=UTC) - timedelta(days=10)
    new = datetime.now(tz=UTC)
    src = InMemoryTraceSource(records=[
        TraceRecord(
            run_id=uuid4(), tenant_id=tid, pattern="p",
            inputs={}, output={}, created_at=old, outcome="success",
        ),
        TraceRecord(
            run_id=uuid4(), tenant_id=tid, pattern="p",
            inputs={}, output={}, created_at=new, outcome="success",
        ),
    ])
    ds = await dataset_from_traces(
        src, tenant_id=tid, pattern="p",
        since=datetime.now(tz=UTC) - timedelta(days=1),
    )
    assert len(ds.cases) == 1


@pytest.mark.asyncio
async def test_dataset_from_traces_attaches_run_id_metadata():
    """Spec 1.14 — run_id traceability back to production incident."""
    tid = uuid4()
    rid = uuid4()
    src = InMemoryTraceSource(records=[
        TraceRecord(
            run_id=rid, tenant_id=tid, pattern="p",
            inputs={}, output={}, created_at=datetime.now(tz=UTC),
            outcome="success",
        ),
    ])
    ds = await dataset_from_traces(
        src, tenant_id=tid, pattern="p",
        since=datetime.now(tz=UTC) - timedelta(days=1),
    )
    assert ds.cases[0].metadata["run_id"] == str(rid)
    assert ds.cases[0].metadata["tenant_id"] == str(tid)
```

`tests/evals/test_cli.py`:

```python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_help_lists_dataset_from_traces():
    proc = subprocess.run(
        [sys.executable, "-m", "ballast.evals.cli", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert "dataset-from-traces" in proc.stdout.lower() or "dataset_from_traces" in proc.stdout.lower()


def test_cli_dataset_from_traces_writes_yaml(tmp_path: Path):
    out = tmp_path / "ds.yaml"
    proc = subprocess.run(
        [
            sys.executable, "-m", "ballast.evals.cli",
            "dataset-from-traces",
            "--pattern", "reflection",
            "--since", "2026-01-01",
            "--tenant", "11111111-1111-1111-1111-111111111111",
            "--out", str(out),
            "--source", "demo",  # built-in demo source for CLI smoke
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    text = out.read_text()
    assert "name:" in text
    assert "cases:" in text
```

`tests/test_public_api_sp7.py`:

```python
"""SP7 exports + an end-to-end smoke that wires the whole layer."""
from __future__ import annotations

import ballast as sf


def test_sp7_exports_present():
    assert hasattr(sf, "ObservabilityProvider")
    assert hasattr(sf, "build_threads_router")
    assert hasattr(sf, "build_a2a_router")
    assert hasattr(sf, "build_streaming_router")
    assert hasattr(sf, "Dataset")
    assert hasattr(sf, "EvalCase")
    assert hasattr(sf, "SchemaAdherenceScorer")
    assert hasattr(sf, "AGUIEncoder")
    assert hasattr(sf, "VercelEncoder")
```

- [ ] **Step 2: Run → fail (ImportError + CLI missing)**

- [ ] **Step 3: Implement**

`src/ballast/evals/traces.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from ballast.evals.case import EvalCase
from ballast.evals.dataset import Dataset


class TraceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_id: UUID
    tenant_id: UUID
    pattern: str
    inputs: Any
    output: Any
    created_at: datetime
    outcome: str  # success | hitl_rejected | reflection_exhausted | ...


@runtime_checkable
class TraceSource(Protocol):
    async def query(
        self, *, tenant_id: UUID, pattern: str | None, since: datetime,
    ) -> list[TraceRecord]: ...


class InMemoryTraceSource:
    def __init__(self, records: list[TraceRecord]) -> None:
        self._records = list(records)

    async def query(
        self, *, tenant_id: UUID, pattern: str | None, since: datetime,
    ) -> list[TraceRecord]:
        return [
            r for r in self._records
            if r.tenant_id == tenant_id
            and (pattern is None or r.pattern == pattern)
            and r.created_at >= since
        ]


async def dataset_from_traces(
    source: TraceSource,
    *,
    tenant_id: UUID,
    pattern: str | None,
    since: datetime,
    name: str | None = None,
) -> Dataset:
    """Build a Dataset by joining production trace records.

    Spec 1.14 — each production run becomes a reusable eval case;
    `run_id` is preserved in metadata so an eval failure traces back to
    the originating incident.
    """
    records = await source.query(
        tenant_id=tenant_id, pattern=pattern, since=since,
    )
    cases = [
        EvalCase(
            name=f"run-{r.run_id}",
            inputs=r.inputs,
            expected=r.output,
            metadata={
                "run_id": str(r.run_id),
                "tenant_id": str(r.tenant_id),
                "outcome": r.outcome,
                "pattern": r.pattern,
            },
        )
        for r in records
    ]
    return Dataset(
        name=name or f"{pattern or 'all'}-traces",
        tenant_id=tenant_id,
        cases=cases,
    )
```

`src/ballast/evals/cli.py`:

```python
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID

import typer
import yaml

from ballast.evals.traces import (
    InMemoryTraceSource,
    TraceRecord,
    dataset_from_traces,
)

app = typer.Typer(name="stateflow-evals", help="Evals CLI for ballast-ai")


def _demo_source() -> InMemoryTraceSource:
    from uuid import uuid4
    return InMemoryTraceSource(records=[
        TraceRecord(
            run_id=uuid4(),
            tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
            pattern="reflection",
            inputs={"x": 1},
            output={"text": "demo"},
            created_at=datetime(2026, 4, 1, tzinfo=__import__("datetime").timezone.utc),
            outcome="success",
        ),
    ])


@app.command("dataset-from-traces")
def dataset_from_traces_cmd(
    pattern: str = typer.Option(..., help="Pattern name filter"),
    since: str = typer.Option(..., help="ISO date — only newer traces included"),
    tenant: UUID = typer.Option(..., help="Tenant UUID"),
    out: Path = typer.Option(..., help="Output YAML path"),
    source: str = typer.Option(
        "demo", help="Source: 'demo' (built-in) or app-provided importable",
    ),
) -> None:
    """Build a YAML Dataset from production traces (spec 1.14)."""
    src = _demo_source() if source == "demo" else _resolve_source(source)
    since_dt = datetime.fromisoformat(since).replace(tzinfo=None)
    # Coerce to aware UTC for source.query
    from datetime import UTC
    since_dt = since_dt.replace(tzinfo=UTC) if since_dt.tzinfo is None else since_dt
    ds = asyncio.run(dataset_from_traces(
        src, tenant_id=tenant, pattern=pattern, since=since_dt,
    ))
    payload = {
        "name": ds.name,
        "tenant_id": str(ds.tenant_id),
        "cases": [c.model_dump(mode="json") for c in ds.cases],
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    typer.echo(f"wrote {len(ds.cases)} cases → {out}")


def _resolve_source(path: str):
    """Import "pkg.module:factory_callable" and call it."""
    mod_name, attr = path.split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(app())
```

Update `src/ballast/__init__.py`:

```python
from ballast.api import (
    build_a2a_router, build_health_router, build_streaming_router,
    build_threads_router, get_container, get_engine, get_tenant_id,
)
from ballast.api.a2a import A2AAgentAdapter, AgentCard
from ballast.api.streaming import (
    AGUIEncoder, StreamEvent, VercelEncoder,
)
from ballast.evals import (
    Dataset, EvalCase, EvalReport, EvalRunOutput, SchemaAdherenceScorer,
    ScoreResult, Scorer,
)
from ballast.observability import ObservabilityProvider, has_logfire
```

Add `pyyaml` (and `typer` if not already) to `pyproject.toml` dependencies.

- [ ] **Step 4: Tests pass (7 new: 3 traces + 2 cli + 1 sp7 exports + smoke = 7)**
- [ ] **Step 5: Full suite + mypy + ruff**

```bash
uv add typer pyyaml
uv run pytest && uv run mypy src && uv run ruff check
```

- [ ] **Step 6: Commit**

```bash
git add src/ballast/evals/ src/ballast/__init__.py tests/evals/ tests/test_public_api_sp7.py pyproject.toml uv.lock
git commit -m "feat(evals): dataset-from-traces CLI + SP7 public API + end-to-end smoke"
```

---

## Acceptance criteria

After Task 10 the SP7 deliverable must satisfy:

1. **Test counts:** ~349 passed + 10 skipped (baseline 299 + ~50 new, distributed 10/6/7/6/5/5/6/5/9/7 across Tasks 1–10). `uv run pytest` is green; `uv run mypy src` is clean; `uv run ruff check` is clean.
2. **No new mandatory deps without justification:** logfire is soft-imported (not in `[project.dependencies]`); typer + pyyaml added because the CLI requires them.
3. **Spec compliance:**
   - 4A.0.7 — Container lives on `app.state.container`, never accessed as a global; `Depends(get_container)` works.
   - 4H — `ObservabilityProvider` enforces "must register first" via `EngineInvariantViolation`.
   - 4D — Span names match the canonical table (`pattern.<name>`, `stage.<name>`, `channel.<name>`); attrs include `tenant_id`.
   - 1.13 — Both AG-UI (UI) and A2A (inter-agent) are mounted; AG-UI is the default streaming protocol with Vercel selectable via `?protocol=vercel`.
   - 1.14 — `dataset-from-traces` writes a `Dataset` YAML with `run_id` metadata per case; cross-tenant exports forbidden.
   - 4C — `SchemaAdherenceScorer(threshold=0.95)` + `Dataset.evaluate(...)` round-trip works.
4. **No regressions:** Every SP1–SP6 test that was green at the SP6 baseline (299 passed + 10 skipped) remains green. The `@traced` decorator must be a transparent passthrough when logfire is absent.
5. **Soft-dep invariant:** `ObservabilityProvider` + `@traced` + `SchemaAdherenceScorer` all import and instantiate without `logfire` installed. The test suite is run without `logfire` to prove this.
6. **A2A note:** `agent.to_a2a()` from pydantic-ai is NOT relied on; if the installed pydantic-ai version exposes it, apps may opt-in inside their own `A2AAgentAdapter.run` body — the framework's contract is the adapter Protocol.
7. **Smoke:** `tests/test_public_api_sp7.py` boots `Engine.fastapi_app(...)`, hits `/healthz`, creates a thread, posts a streamed message (AG-UI), runs a Reflection, and pipes the result through `Dataset.evaluate([SchemaAdherenceScorer()])`. All in one test, all green.

---

### Task titles (for parent verification)

1. API primitives — `deps.py` + `/healthz`
2. Thread CRUD endpoints — `build_threads_router`
3. Streaming endpoint + AG-UI encoder
4. Vercel AI SDK encoder
5. A2A discovery + invoke endpoints
6. `Engine.fastapi_app(...)` factory
7. `ObservabilityProvider` + logfire soft import
8. `@traced` decorator + Pattern / Channel instrumentation
9. Evals primitives + `SchemaAdherenceScorer`
10. `dataset-from-traces` CLI + SP7 public API + smoke
