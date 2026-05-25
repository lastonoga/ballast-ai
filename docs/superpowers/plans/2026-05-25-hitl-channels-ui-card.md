# HITL Channels + UI Card Approval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an out-of-thread approval gate for `create_note` — the tool suspends on a `UICardChannel.request(payload)`, the user sees the card in a side-panel "Approvals" drawer (separate from chat), clicks approve/reject, the workflow resumes with the typed verdict.

**Architecture:** A single `HITLChannel` Protocol owns the whole "ask a human" lifecycle (deliver + suspend + decode); `DBOSHITLChannel` ABC implements suspend via `Durable.recv_async` so concrete channels only fill in `deliver` + `decode_verdict`. `UICardChannel` persists an `ApprovalCard` row + fires a signal; a `/approvals` REST router serves the panel and pumps verdicts back through `Durable.send_async`. `create_note` extracts its body into a child `@Durable.workflow` so the human-pause unit is explicit in DBOS topology. Spike at `tests/spikes/test_hitl_channel_wire.py` validates the wire.

**Tech Stack:** Python 3.11+, pydantic v2, DBOS (`Durable.workflow` / `Durable.step` / `recv_async` / `send_async`), FastAPI, blinker signals, React + assistant-ui (Vercel AI SDK).

**Out of scope** (follow-up plan): `ThreadChannel` extraction from `DurableHITLWorkflow`; `propose_todo` migration; deletion of legacy `ask_human` / `DurableHITLWorkflow` / `HITLResponse` union / `HITLContext`; Postgres repo + Alembic migration; multi-user RBAC; card-edit form; Slack channels.

---

## File Map

**Framework — new:**
- `src/ballast/patterns/hitl/channels/__init__.py` — re-exports `HITLChannel`, `DBOSHITLChannel`, `UICardChannel`, `CardVerdict`, `register_card_kind`.
- `src/ballast/patterns/hitl/channels/_protocol.py` — `HITLChannel` Protocol.
- `src/ballast/patterns/hitl/channels/_base.py` — `DBOSHITLChannel` ABC (suspend boilerplate).
- `src/ballast/patterns/hitl/channels/ui_card.py` — `UICardChannel`, `CardVerdict`, `approval_card_requested` / `approval_card_decided` signals, `__hitl_kind__` registry.
- `src/ballast/auth/__init__.py` — re-export of context helpers.
- `src/ballast/auth/context.py` — `current_user_id()` ContextVar + `acting_as()` cm.
- `src/ballast/persistence/approval_card/__init__.py` — re-exports model, Protocol, module singleton.
- `src/ballast/persistence/approval_card/_models.py` — `ApprovalCard` model.
- `src/ballast/persistence/approval_card/_repo.py` — `ApprovalCardRepository` Protocol.
- `src/ballast/persistence/approval_card/_memory.py` — `InMemoryApprovalCardRepository`.
- `src/ballast/api/approvals/__init__.py` — re-export of router.
- `src/ballast/api/approvals/router.py` — REST endpoints + SSE.

**Framework — modify:**
- `src/ballast/app.py` — `Ballast.with_approval_repo(...)` setter + wire `/approvals` router in `fastapi_app()`.

**Notes-app — new:**
- `examples/notes-app/backend/src/notes_app/workflows/create_note.py` — `create_note_flow` `@Durable.workflow`.
- `examples/notes-app/frontend/src/components/approvals/approvals-panel.tsx` — side drawer + list + decide buttons.
- `examples/notes-app/frontend/src/components/approvals/use-approvals.ts` — hook owning fetch + SSE + decide POST.
- `examples/notes-app/frontend/src/components/approvals/card-renderers/note-create.tsx` — per-kind renderer for `"note.create"`.
- `examples/notes-app/frontend/src/components/data-parts/data-approval-pending.tsx` — chat marker (pending).
- `examples/notes-app/frontend/src/components/data-parts/data-approval-resolved.tsx` — chat marker (resolved).

**Notes-app — modify:**
- `examples/notes-app/backend/src/notes_app/agents/notes.py` — `create_note` body → child-flow call.
- `examples/notes-app/backend/src/notes_app/main.py` — `.with_approval_repo(InMemoryApprovalCardRepository())` in builder chain.
- `examples/notes-app/frontend/src/app/page.tsx` (or app shell) — drawer toggle button + badge + render `<ApprovalsPanel/>`.

**Tests:**
- `tests/auth/test_context.py` — current_user_id ContextVar.
- `tests/persistence/test_approval_card_memory.py` — in-memory repo behavior.
- `tests/patterns/hitl/test_channels_protocol.py` — DBOSHITLChannel base contract.
- `tests/patterns/hitl/test_ui_card_channel.py` — UICardChannel deliver + decode.
- `tests/api/test_approvals_router.py` — REST + SSE.
- `examples/notes-app/backend/tests/test_create_note_approval.py` — end-to-end with stub channel.

---

## Task 1: `current_user_id` ContextVar

**Files:**
- Create: `src/ballast/auth/__init__.py`
- Create: `src/ballast/auth/context.py`
- Create: `tests/auth/__init__.py`
- Create: `tests/auth/test_context.py`

- [ ] **Step 1: Write the failing test**

Create `tests/auth/__init__.py` (empty), then `tests/auth/test_context.py`:

```python
"""``current_user_id`` ContextVar — set/get/reset via ``acting_as`` cm."""
from __future__ import annotations

from ballast.auth.context import acting_as, current_user_id


def test_default_is_none() -> None:
    assert current_user_id() is None


def test_acting_as_sets_and_resets() -> None:
    assert current_user_id() is None
    with acting_as("user-1"):
        assert current_user_id() == "user-1"
    assert current_user_id() is None


def test_nested_acting_as_restores_outer() -> None:
    with acting_as("outer"):
        assert current_user_id() == "outer"
        with acting_as("inner"):
            assert current_user_id() == "inner"
        assert current_user_id() == "outer"
    assert current_user_id() is None
```

- [ ] **Step 2: Run test — confirm import failure**

```
uv run pytest tests/auth/test_context.py -v
```

Expected: `ModuleNotFoundError: No module named 'ballast.auth'`.

- [ ] **Step 3: Implement minimal module**

Create `src/ballast/auth/__init__.py`:

```python
"""Authentication context primitives.

Currently only the ``current_user_id`` ContextVar — read by repos and
channels to scope visibility / persist `user_id` stamps. Filled by API
middleware in production; tests use ``acting_as`` directly.
"""
from ballast.auth.context import acting_as, current_user_id

__all__ = ["acting_as", "current_user_id"]
```

Create `src/ballast/auth/context.py`:

```python
"""Ambient ``current_user_id`` ContextVar."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_user_id: ContextVar[str | None] = ContextVar(
    "current_user_id", default=None,
)


def current_user_id() -> str | None:
    """Return the user id bound to the current context, if any."""
    return _current_user_id.get()


@contextmanager
def acting_as(user_id: str) -> Iterator[None]:
    """Bind ``user_id`` to the current context for the duration of the
    block. API middleware wraps each request handler with this so
    downstream code (repos, channels) reads the caller's identity from
    the ambient context instead of plumbing it through every signature.
    """
    tok = _current_user_id.set(user_id)
    try:
        yield
    finally:
        _current_user_id.reset(tok)


__all__ = ["acting_as", "current_user_id"]
```

- [ ] **Step 4: Run tests — confirm pass**

```
uv run pytest tests/auth/test_context.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/auth tests/auth
git commit -m "feat(auth): current_user_id ContextVar + acting_as cm"
```

---

## Task 2: `ApprovalCard` model

**Files:**
- Create: `src/ballast/persistence/approval_card/__init__.py`
- Create: `src/ballast/persistence/approval_card/_models.py`
- Create: `tests/persistence/test_approval_card_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_approval_card_model.py`:

```python
"""``ApprovalCard`` Pydantic model — shape + status transitions."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.persistence.approval_card import ApprovalCard


def _now() -> datetime:
    return datetime(2026, 5, 25, tzinfo=UTC)


def test_pending_card_has_no_resolution() -> None:
    card = ApprovalCard(
        id="req-1", workflow_id="wf-1",
        respond_topic="hitl:req-1", kind="note.create",
        payload={"title": "x", "body": "y"},
        parent_thread_id=None, user_id=None,
        status="pending", created_at=_now(),
    )
    assert card.status == "pending"
    assert card.resolution is None
    assert card.resolved_at is None


def test_status_validates() -> None:
    with pytest.raises(ValueError):
        ApprovalCard(
            id="req-1", workflow_id="wf-1",
            respond_topic="hitl:req-1", kind="note.create",
            payload={}, parent_thread_id=None, user_id=None,
            status="bogus",  # type: ignore[arg-type]
            created_at=_now(),
        )


def test_json_round_trip_preserves_fields() -> None:
    card = ApprovalCard(
        id="req-1", workflow_id="wf-1",
        respond_topic="hitl:req-1", kind="note.create",
        payload={"title": "x", "body": "y"},
        parent_thread_id="t-1", user_id="user-1",
        status="approved", resolution={"decision": "approve"},
        created_at=_now(), resolved_at=_now(),
    )
    dump = card.model_dump_json()
    again = ApprovalCard.model_validate_json(dump)
    assert again == card
```

- [ ] **Step 2: Run test — confirm import failure**

```
uv run pytest tests/persistence/test_approval_card_model.py -v
```

Expected: `ModuleNotFoundError: No module named 'ballast.persistence.approval_card'`.

- [ ] **Step 3: Implement model**

Create `src/ballast/persistence/approval_card/__init__.py`:

```python
"""Approval card persistence — model + Protocol + module singleton."""
from ballast.persistence.approval_card._models import ApprovalCard

__all__ = ["ApprovalCard"]
```

Create `src/ballast/persistence/approval_card/_models.py`:

```python
"""``ApprovalCard`` — one human approval request awaiting a decision."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

CardStatus = Literal["pending", "approved", "rejected", "timeout"]


class ApprovalCard(BaseModel):
    """One pending / resolved approval request displayed in the inbox.

    ``id`` doubles as the HITL ``request_id`` so the wire topic
    (`f"hitl:{id}"`) is stable across the channel ↔ workflow ↔ router
    hops. ``payload`` is the channel's input model as JSON; ``resolution``
    is the verdict dump once decided.
    """

    id: str
    workflow_id: str
    respond_topic: str
    kind: str
    payload: dict[str, Any]
    parent_thread_id: str | None
    user_id: str | None
    status: CardStatus
    resolution: dict[str, Any] | None = None
    created_at: datetime
    resolved_at: datetime | None = None


__all__ = ["ApprovalCard", "CardStatus"]
```

- [ ] **Step 4: Run tests — confirm pass**

```
uv run pytest tests/persistence/test_approval_card_model.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/persistence/approval_card tests/persistence/test_approval_card_model.py
git commit -m "feat(persistence): ApprovalCard model"
```

---

## Task 3: `ApprovalCardRepository` Protocol + in-memory impl

**Files:**
- Create: `src/ballast/persistence/approval_card/_repo.py`
- Create: `src/ballast/persistence/approval_card/_memory.py`
- Modify: `src/ballast/persistence/approval_card/__init__.py`
- Create: `tests/persistence/test_approval_card_memory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_approval_card_memory.py`:

```python
"""``InMemoryApprovalCardRepository`` — add / get / list_pending / resolve.

Per-user visibility is enforced at the repo edge by reading
``current_user_id()``. Tests exercise both the unscoped (None) and
scoped behaviors.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ballast.auth.context import acting_as
from ballast.persistence.approval_card import (
    ApprovalCard,
    InMemoryApprovalCardRepository,
)


def _card(id_: str, *, user_id: str | None, status: str = "pending") -> ApprovalCard:
    return ApprovalCard(
        id=id_, workflow_id=f"wf-{id_}",
        respond_topic=f"hitl:{id_}", kind="note.create",
        payload={}, parent_thread_id=None, user_id=user_id,
        status=status,  # type: ignore[arg-type]
        created_at=datetime(2026, 5, 25, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_add_then_get() -> None:
    repo = InMemoryApprovalCardRepository()
    await repo.add(_card("a", user_id="u-1"))
    got = await repo.get("a")
    assert got is not None and got.id == "a"


@pytest.mark.asyncio
async def test_get_returns_none_for_unknown() -> None:
    repo = InMemoryApprovalCardRepository()
    assert await repo.get("nope") is None


@pytest.mark.asyncio
async def test_list_pending_filters_by_current_user_id() -> None:
    repo = InMemoryApprovalCardRepository()
    await repo.add(_card("a", user_id="u-1"))
    await repo.add(_card("b", user_id="u-2"))
    await repo.add(_card("c", user_id="u-1", status="approved"))

    with acting_as("u-1"):
        listed = await repo.list_pending()
    ids = [c.id for c in listed]
    assert ids == ["a"]  # only pending + matches u-1


@pytest.mark.asyncio
async def test_list_pending_unscoped_returns_all_pending() -> None:
    """No acting_as scope → no user filter (admin / single-user use)."""
    repo = InMemoryApprovalCardRepository()
    await repo.add(_card("a", user_id="u-1"))
    await repo.add(_card("b", user_id="u-2"))
    await repo.add(_card("c", user_id=None, status="approved"))

    listed = await repo.list_pending()
    assert {c.id for c in listed} == {"a", "b"}


@pytest.mark.asyncio
async def test_resolve_flips_status_and_stamps_resolution() -> None:
    repo = InMemoryApprovalCardRepository()
    await repo.add(_card("a", user_id="u-1"))

    from pydantic import BaseModel
    class _V(BaseModel): decision: str

    resolved = await repo.resolve("a", verdict=_V(decision="approve"))
    assert resolved.status == "approved"
    assert resolved.resolution == {"decision": "approve"}
    assert resolved.resolved_at is not None


@pytest.mark.asyncio
async def test_resolve_unknown_raises() -> None:
    repo = InMemoryApprovalCardRepository()
    from pydantic import BaseModel
    class _V(BaseModel): decision: str

    with pytest.raises(KeyError):
        await repo.resolve("nope", verdict=_V(decision="approve"))
```

- [ ] **Step 2: Run test — confirm failure**

```
uv run pytest tests/persistence/test_approval_card_memory.py -v
```

Expected: ImportError for `InMemoryApprovalCardRepository`.

- [ ] **Step 3: Implement Protocol**

Create `src/ballast/persistence/approval_card/_repo.py`:

```python
"""``ApprovalCardRepository`` Protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from ballast.persistence.approval_card._models import ApprovalCard


@runtime_checkable
class ApprovalCardRepository(Protocol):
    """Read/write of approval card rows.

    Visibility (``list_pending`` / ``get``) is the implementation's
    concern: the in-memory and SQL repos filter by ``current_user_id()``
    when set, returning all rows when unscoped.
    """

    async def add(self, card: ApprovalCard) -> None: ...

    async def get(self, card_id: str) -> ApprovalCard | None: ...

    async def list_pending(
        self, *, limit: int = 50,
    ) -> list[ApprovalCard]: ...

    async def resolve(
        self, card_id: str, *, verdict: BaseModel,
    ) -> ApprovalCard: ...


__all__ = ["ApprovalCardRepository"]
```

- [ ] **Step 4: Implement in-memory repo**

Create `src/ballast/persistence/approval_card/_memory.py`:

```python
"""In-memory ``ApprovalCardRepository`` — tests + local dev."""
from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from ballast.auth.context import current_user_id
from ballast.persistence.approval_card._models import ApprovalCard
from ballast.persistence.approval_card._repo import ApprovalCardRepository


class InMemoryApprovalCardRepository(ApprovalCardRepository):
    """Process-local dict-backed repo. Filters by ``current_user_id``
    when set; returns all rows otherwise (admin / single-user mode)."""

    def __init__(self) -> None:
        self._rows: dict[str, ApprovalCard] = {}

    async def add(self, card: ApprovalCard) -> None:
        self._rows[card.id] = card

    async def get(self, card_id: str) -> ApprovalCard | None:
        card = self._rows.get(card_id)
        if card is None:
            return None
        scope = current_user_id()
        if scope is not None and card.user_id != scope:
            return None
        return card

    async def list_pending(
        self, *, limit: int = 50,
    ) -> list[ApprovalCard]:
        scope = current_user_id()
        out = [
            c for c in self._rows.values()
            if c.status == "pending"
            and (scope is None or c.user_id == scope)
        ]
        out.sort(key=lambda c: c.created_at)
        return out[:limit]

    async def resolve(
        self, card_id: str, *, verdict: BaseModel,
    ) -> ApprovalCard:
        card = self._rows.get(card_id)
        if card is None:
            raise KeyError(card_id)
        decision = getattr(verdict, "decision", None)
        match decision:
            case "approve": new_status = "approved"
            case "reject":  new_status = "rejected"
            case _:         new_status = "timeout"
        updated = card.model_copy(update={
            "status": new_status,
            "resolution": verdict.model_dump(mode="json"),
            "resolved_at": datetime.now(UTC),
        })
        self._rows[card_id] = updated
        return updated


__all__ = ["InMemoryApprovalCardRepository"]
```

- [ ] **Step 5: Update package exports**

Replace `src/ballast/persistence/approval_card/__init__.py`:

```python
"""Approval card persistence — model + Protocol + in-memory impl."""
from ballast.persistence.approval_card._memory import (
    InMemoryApprovalCardRepository,
)
from ballast.persistence.approval_card._models import (
    ApprovalCard,
    CardStatus,
)
from ballast.persistence.approval_card._repo import ApprovalCardRepository

__all__ = [
    "ApprovalCard",
    "ApprovalCardRepository",
    "CardStatus",
    "InMemoryApprovalCardRepository",
]
```

- [ ] **Step 6: Run tests — confirm pass**

```
uv run pytest tests/persistence/test_approval_card_memory.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add src/ballast/persistence/approval_card tests/persistence/test_approval_card_memory.py
git commit -m "feat(persistence): InMemoryApprovalCardRepository + Protocol"
```

---

## Task 4: Module-level `approval_card_repo` singleton

**Files:**
- Modify: `src/ballast/persistence/approval_card/__init__.py`
- Create: `tests/persistence/test_approval_card_singleton.py`

- [ ] **Step 1: Write the failing test**

Create `tests/persistence/test_approval_card_singleton.py`:

```python
"""The module exposes a swappable ``approval_card_repo`` singleton —
production reassigns via ``Ballast.with_approval_repo(...)``; tests
monkeypatch the same attribute.
"""
from __future__ import annotations

import pytest

from ballast.persistence import approval_card as mod


def test_default_singleton_is_inmemory() -> None:
    from ballast.persistence.approval_card import InMemoryApprovalCardRepository
    assert isinstance(mod.approval_card_repo, InMemoryApprovalCardRepository)


@pytest.mark.asyncio
async def test_singleton_is_monkeypatchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests swap the singleton via monkeypatch — same convention as
    notes_repo."""
    from ballast.persistence.approval_card import InMemoryApprovalCardRepository
    fresh = InMemoryApprovalCardRepository()
    monkeypatch.setattr(
        "ballast.persistence.approval_card.approval_card_repo",
        fresh,
    )
    assert mod.approval_card_repo is fresh
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/persistence/test_approval_card_singleton.py -v
```

Expected: `AttributeError: module ... has no attribute 'approval_card_repo'`.

- [ ] **Step 3: Add singleton to package init**

Edit `src/ballast/persistence/approval_card/__init__.py` — append:

```python
# Module-level singleton, reassigned at app-build time when the user
# configures a custom repo (see ``Ballast.with_approval_repo``). Tests
# monkeypatch this attribute directly — same pattern as ``notes_repo``.
approval_card_repo: ApprovalCardRepository = InMemoryApprovalCardRepository()
```

And add to `__all__`:

```python
__all__ = [
    "ApprovalCard",
    "ApprovalCardRepository",
    "CardStatus",
    "InMemoryApprovalCardRepository",
    "approval_card_repo",
]
```

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/persistence/test_approval_card_singleton.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/persistence/approval_card/__init__.py tests/persistence/test_approval_card_singleton.py
git commit -m "feat(persistence): module-level approval_card_repo singleton"
```

---

## Task 5: `HITLChannel` Protocol + `DBOSHITLChannel` base

**Files:**
- Create: `src/ballast/patterns/hitl/channels/__init__.py`
- Create: `src/ballast/patterns/hitl/channels/_protocol.py`
- Create: `src/ballast/patterns/hitl/channels/_base.py`
- Create: `tests/patterns/hitl/test_dbos_hitl_channel.py`

- [ ] **Step 1: Write the failing test**

The spike already proved the wire works (see `tests/spikes/test_hitl_channel_wire.py`). This test exercises the public `DBOSHITLChannel` ABC contract: subclassing requires `deliver` + `decode_verdict`; `request()` orchestrates them with `Durable.recv_async`.

Create `tests/patterns/hitl/__init__.py` if missing (empty). Then create `tests/patterns/hitl/test_dbos_hitl_channel.py`:

```python
"""``DBOSHITLChannel`` ABC — request() = deliver + recv + decode_verdict."""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

from ballast.durable import Durable
from ballast.patterns.hitl.channels import DBOSHITLChannel


class _Payload(BaseModel):
    title: str


class _Verdict(BaseModel):
    decision: str


# Spy slot — child writes (workflow_id, topic) so the test body can
# dial send_async at the right destination.
_DELIVERIES: dict[str, tuple[str, str]] = {}


class _SpyChannel(DBOSHITLChannel[_Payload, _Verdict]):
    async def deliver(self, *, request_id, workflow_id,
                      respond_topic, payload) -> None:
        _DELIVERIES[request_id] = (workflow_id, respond_topic)

    async def decode_verdict(self, raw: Any) -> _Verdict:
        return _Verdict.model_validate(raw)


@Durable.workflow()
async def _flow(payload: _Payload, request_id: str) -> _Verdict:
    chan = _SpyChannel()
    # request_id is overridden internally by the base — to keep the
    # spike-style test simple we'll just await the channel and let the
    # outer code dial send_async at whatever id the spy captured.
    return await chan.request(payload, timeout=10.0)


async def _wait(rid_seen_in: dict, want: int, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if len(rid_seen_in) >= want:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError


@pytest.mark.asyncio
async def test_request_delivers_then_decodes_recv(
    fresh_dbos_executor: None,
) -> None:
    _DELIVERIES.clear()
    handle = await Durable.start_workflow(
        _flow, _Payload(title="x"), str(uuid4()),
    )
    await _wait(_DELIVERIES, want=1)
    (wfid, topic), = _DELIVERIES.values()

    await Durable.send_async(
        destination_id=wfid, topic=topic,
        message={"decision": "approve"},
    )
    result = await handle.get_result()
    assert isinstance(result, _Verdict)
    assert result.decision == "approve"


@pytest.mark.asyncio
async def test_abstract_methods_must_be_overridden() -> None:
    with pytest.raises(TypeError, match="abstract"):
        DBOSHITLChannel()  # type: ignore[abstract]
```

Add the `fresh_dbos_executor` + `dbos_runtime` fixtures to `tests/patterns/hitl/conftest.py` (mirror `tests/patterns/conftest.py`):

```python
"""DBOS bootstrap for HITL channel tests — mirror of tests/patterns/conftest.py."""
from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pytest_asyncio
from dbos import DBOS, DBOSConfig


@pytest.fixture(scope="module")
def dbos_runtime() -> Iterator[type[DBOS]]:
    tmpdir = tempfile.mkdtemp(prefix="dbos-hitl-")
    db_path = Path(tmpdir) / "dbos.sqlite"
    DBOS(config=DBOSConfig(
        name="stateflow-hitl-test",
        system_database_url=f"sqlite:///{db_path}",
    ))
    DBOS.launch()
    try:
        yield DBOS
    finally:
        DBOS.destroy(destroy_registry=False)


@pytest_asyncio.fixture
async def fresh_dbos_executor(dbos_runtime: type[DBOS]) -> AsyncIterator[None]:
    from dbos._dbos import _get_dbos_instance
    dbos = _get_dbos_instance()
    fresh = ThreadPoolExecutor(max_workers=8, thread_name_prefix="dbos-test-")
    dbos._executor_field = fresh
    yield
```

- [ ] **Step 2: Run — confirm import failure**

```
uv run pytest tests/patterns/hitl/test_dbos_hitl_channel.py -v
```

Expected: `ModuleNotFoundError: No module named 'ballast.patterns.hitl.channels'`.

- [ ] **Step 3: Implement Protocol**

Create `src/ballast/patterns/hitl/channels/_protocol.py`:

```python
"""``HITLChannel`` Protocol — one method, full lifecycle."""
from __future__ import annotations

from datetime import timedelta
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

InT      = TypeVar("InT",      bound=BaseModel)
VerdictT = TypeVar("VerdictT", bound=BaseModel)


class HITLChannel(Protocol, Generic[InT, VerdictT]):
    """Owns the full request lifecycle for one human decision.

    A channel knows what payload type it accepts, how to surface the
    request (UI card, chat thread, Slack, …), how to wait for the
    verdict, and how to decode the response into a typed model. The
    framework knows nothing about the medium.
    """

    async def request(
        self,
        payload: InT,
        *,
        timeout: timedelta | None = None,
    ) -> VerdictT: ...


__all__ = ["HITLChannel", "InT", "VerdictT"]
```

- [ ] **Step 4: Implement DBOS base**

Create `src/ballast/patterns/hitl/channels/_base.py`:

```python
"""``DBOSHITLChannel`` ABC — shared suspend boilerplate for any channel
that delivers verdicts via DBOS topics (the common case)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Generic
from uuid import uuid4

from ballast.durable import Durable
from ballast.patterns.hitl.channels._protocol import InT, VerdictT


class DBOSHITLChannel(Generic[InT, VerdictT], ABC):
    """Channels that use DBOS topics for verdict delivery.

    Subclasses fill in ``deliver`` (surface the request to the human)
    and ``decode_verdict`` (re-hydrate the dict that arrived on the
    DBOS topic into a typed VerdictT). ``request`` orchestrates them
    with ``Durable.recv_async`` so the calling workflow is recoverable
    across crashes.
    """

    async def request(
        self,
        payload: InT,
        *,
        timeout: timedelta | None = None,
    ) -> VerdictT:
        request_id  = str(uuid4())
        workflow_id = Durable.current_workflow_id()
        topic       = f"hitl:{request_id}"
        await self.deliver(
            request_id=request_id, workflow_id=workflow_id,
            respond_topic=topic, payload=payload,
        )
        timeout_seconds = (
            timeout.total_seconds() if timeout is not None else None
        )
        raw = await Durable.recv_async(
            topic=topic, timeout_seconds=timeout_seconds,
        )
        return await self.decode_verdict(raw)

    @abstractmethod
    async def deliver(
        self, *,
        request_id: str, workflow_id: str, respond_topic: str,
        payload: InT,
    ) -> None: ...

    @abstractmethod
    async def decode_verdict(self, raw: Any) -> VerdictT: ...


__all__ = ["DBOSHITLChannel"]
```

- [ ] **Step 5: Implement package __init__**

Create `src/ballast/patterns/hitl/channels/__init__.py`:

```python
"""HITL channels — request delivery surfaces.

A channel is the unit of "how a human is asked" — UI card, helper
thread, Slack DM, custom user-written. ``HITLChannel`` is the Protocol;
``DBOSHITLChannel`` is the convenient base for the common case where
verdicts arrive on a DBOS topic.
"""
from ballast.patterns.hitl.channels._base import DBOSHITLChannel
from ballast.patterns.hitl.channels._protocol import (
    HITLChannel,
    InT,
    VerdictT,
)

__all__ = ["DBOSHITLChannel", "HITLChannel", "InT", "VerdictT"]
```

- [ ] **Step 6: Run tests — confirm pass**

```
uv run pytest tests/patterns/hitl/test_dbos_hitl_channel.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/ballast/patterns/hitl/channels tests/patterns/hitl/__init__.py tests/patterns/hitl/conftest.py tests/patterns/hitl/test_dbos_hitl_channel.py
git commit -m "feat(hitl): HITLChannel Protocol + DBOSHITLChannel base"
```

---

## Task 6: `CardVerdict` + `__hitl_kind__` registry + signals

**Files:**
- Create: `src/ballast/patterns/hitl/channels/ui_card.py` (partial — only verdict + signals + registry; channel class lands in Task 7)
- Create: `tests/patterns/hitl/test_card_verdict.py`

- [ ] **Step 1: Write the failing test**

Create `tests/patterns/hitl/test_card_verdict.py`:

```python
"""``CardVerdict[OutT]`` + ``__hitl_kind__`` registry + signals."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict,
    card_kind_registry,
    register_card_kind,
)


class _Note(BaseModel):
    __hitl_kind__ = "note.create"
    title: str
    body: str


def test_verdict_typed_modified_field() -> None:
    v = CardVerdict[_Note](
        decision="approve",
        modified=_Note(title="t", body="b"),
        answered_at=datetime(2026, 5, 25, tzinfo=UTC),
    )
    assert v.modified is not None and v.modified.title == "t"


def test_verdict_reject_no_modified() -> None:
    v = CardVerdict[_Note](
        decision="reject",
        answered_at=datetime(2026, 5, 25, tzinfo=UTC),
    )
    assert v.modified is None
    assert v.decision == "reject"


def test_register_card_kind_indexes_by_hitl_kind_attr() -> None:
    register_card_kind(_Note)
    assert card_kind_registry["note.create"] is _Note


def test_register_card_kind_requires_attr() -> None:
    class NoKind(BaseModel):
        x: str
    with pytest.raises(AttributeError, match="__hitl_kind__"):
        register_card_kind(NoKind)
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/patterns/hitl/test_card_verdict.py -v
```

Expected: ImportError for `CardVerdict`, `register_card_kind`, `card_kind_registry`.

- [ ] **Step 3: Implement (partial ui_card.py)**

Create `src/ballast/patterns/hitl/channels/ui_card.py`:

```python
"""``UICardChannel`` — out-of-thread approval card delivered via a
side-panel SSE stream.

This module ships:
  - ``CardVerdict[OutT]`` — the standard verdict shape for card-style
    approvals (`decision` + optional `modified` payload).
  - ``card_kind_registry`` — `__hitl_kind__` → payload BaseModel
    lookup; the REST decision endpoint uses this to validate the
    incoming ``modified`` payload against the right type.
  - ``approval_card_requested`` / ``approval_card_decided`` signals —
    SSE multiplexer subscribes to both.

The actual ``UICardChannel`` class lands in the next task.
"""
from __future__ import annotations

from datetime import datetime
from typing import Generic, Literal, TypeVar

from blinker import Signal
from pydantic import BaseModel

OutT = TypeVar("OutT", bound=BaseModel)


class CardVerdict(BaseModel, Generic[OutT]):
    """Standard verdict for card-style approvals.

    Custom channels are free to ship their own verdict shapes; this
    one covers the common UI card case (approve/reject with optional
    edits coming back).
    """

    decision: Literal["approve", "reject"]
    modified: OutT | None = None
    feedback: str | None = None
    answered_at: datetime


# ── kind registry ───────────────────────────────────────────────────

card_kind_registry: dict[str, type[BaseModel]] = {}


def register_card_kind(model: type[BaseModel]) -> type[BaseModel]:
    """Register a payload model under its ``__hitl_kind__``.

    The REST decision endpoint reads this to know how to validate the
    incoming ``modified`` body against the right type. Idempotent:
    re-registering the same class is a no-op; re-registering a
    different class under the same kind raises.
    """
    kind = getattr(model, "__hitl_kind__", None)
    if not kind:
        raise AttributeError(
            f"{model.__name__} must declare __hitl_kind__ to register",
        )
    existing = card_kind_registry.get(kind)
    if existing is not None and existing is not model:
        raise ValueError(
            f"__hitl_kind__={kind!r} already registered to "
            f"{existing.__name__}; cannot reassign to {model.__name__}",
        )
    card_kind_registry[kind] = model
    return model


# ── signals ─────────────────────────────────────────────────────────

approval_card_requested: Signal = Signal("approval-card-requested")
approval_card_decided:   Signal = Signal("approval-card-decided")


__all__ = [
    "CardVerdict",
    "approval_card_decided",
    "approval_card_requested",
    "card_kind_registry",
    "register_card_kind",
]
```

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/patterns/hitl/test_card_verdict.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/hitl/channels/ui_card.py tests/patterns/hitl/test_card_verdict.py
git commit -m "feat(hitl): CardVerdict + __hitl_kind__ registry + signals"
```

---

## Task 7: `UICardChannel` class

**Files:**
- Modify: `src/ballast/patterns/hitl/channels/ui_card.py` — append `UICardChannel` + module singleton.
- Modify: `src/ballast/patterns/hitl/channels/__init__.py` — re-export.
- Create: `tests/patterns/hitl/test_ui_card_channel.py`

- [ ] **Step 1: Write the failing test**

Create `tests/patterns/hitl/test_ui_card_channel.py`:

```python
"""``UICardChannel`` — deliver persists a card row + fires the signal;
decode_verdict re-validates dict → CardVerdict[InT]."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from ballast.auth.context import acting_as
from ballast.events.context import progress_to_thread
from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict,
    UICardChannel,
    approval_card_requested,
    register_card_kind,
)
from ballast.persistence.approval_card import (
    InMemoryApprovalCardRepository,
    approval_card_repo,
)


class _Note(BaseModel):
    __hitl_kind__ = "note.create"
    title: str
    body: str


register_card_kind(_Note)


@pytest.fixture
def fresh_repo(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryApprovalCardRepository]:
    fresh = InMemoryApprovalCardRepository()
    monkeypatch.setattr(
        "ballast.persistence.approval_card.approval_card_repo",
        fresh,
    )
    yield fresh


@pytest.mark.asyncio
async def test_deliver_persists_card_and_fires_signal(
    fresh_repo: InMemoryApprovalCardRepository,
) -> None:
    seen: list[Any] = []
    approval_card_requested.connect(
        lambda sender, *, card, **_: seen.append(card),
        weak=False,
    )

    chan: UICardChannel[_Note] = UICardChannel(payload_type=_Note)
    with acting_as("user-1"), progress_to_thread("thread-1"):
        await chan.deliver(
            request_id="req-1", workflow_id="wf-1",
            respond_topic="hitl:req-1",
            payload=_Note(title="t", body="b"),
        )

    stored = await fresh_repo.get("req-1")
    assert stored is not None
    assert stored.kind == "note.create"
    assert stored.payload == {"title": "t", "body": "b"}
    assert stored.user_id == "user-1"
    assert stored.parent_thread_id == "thread-1"

    assert len(seen) == 1 and seen[0].id == "req-1"


@pytest.mark.asyncio
async def test_decode_verdict_typed() -> None:
    chan: UICardChannel[_Note] = UICardChannel(payload_type=_Note)
    verdict = await chan.decode_verdict({
        "decision": "approve",
        "modified": {"title": "x", "body": "y"},
        "answered_at": datetime(2026, 5, 25, tzinfo=UTC).isoformat(),
    })
    assert isinstance(verdict, CardVerdict)
    assert verdict.decision == "approve"
    assert verdict.modified is not None
    assert verdict.modified.title == "x"
```

Note: `progress_to_thread` already exists in the framework (used by Reflection events). Confirm import path with `grep -r 'def progress_to_thread' src/ballast/` before running — adjust if it lives elsewhere.

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/patterns/hitl/test_ui_card_channel.py -v
```

Expected: ImportError for `UICardChannel`.

- [ ] **Step 3: Implement `UICardChannel`**

Append to `src/ballast/patterns/hitl/channels/ui_card.py`:

```python
from datetime import datetime, timezone
from typing import Any

from ballast.patterns.hitl.channels._base import DBOSHITLChannel
from ballast.patterns.hitl.channels._protocol import InT
from pydantic import TypeAdapter


class UICardChannel(DBOSHITLChannel[InT, "CardVerdict[InT]"]):
    """Persists an ApprovalCard row + fires the request signal so the
    UI panel SSE picks it up. Verdict comes back via
    ``POST /approvals/{id}/decision`` → ``Durable.send_async`` → the
    suspended ``recv_async`` inside ``DBOSHITLChannel.request``.

    Take ``payload_type`` in the constructor so ``decode_verdict`` can
    type-validate the inbound dict — Python's runtime erases generic
    parameters from the class so we can't reach for them via
    ``__orig_class__`` on every instance.
    """

    def __init__(self, payload_type: type[InT]) -> None:
        super().__init__()
        self._payload_type = payload_type

    async def deliver(
        self, *,
        request_id: str, workflow_id: str, respond_topic: str,
        payload: InT,
    ) -> None:
        # Lazy imports keep this module importable in environments
        # without the auth / events ContextVars wired (tests, etc.).
        from ballast.auth.context import current_user_id              # noqa: PLC0415
        from ballast.events.context import current_parent_thread_id   # noqa: PLC0415
        from ballast.persistence.approval_card import (                # noqa: PLC0415
            ApprovalCard, approval_card_repo,
        )

        card = ApprovalCard(
            id=request_id, workflow_id=workflow_id,
            respond_topic=respond_topic,
            kind=type(payload).__hitl_kind__,
            payload=payload.model_dump(mode="json"),
            parent_thread_id=current_parent_thread_id(),
            user_id=current_user_id(),
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        await approval_card_repo.add(card)
        await approval_card_requested.send_async(self, card=card)

    async def decode_verdict(self, raw: Any) -> "CardVerdict[InT]":
        return TypeAdapter(CardVerdict[self._payload_type]).validate_python(raw)
```

Append `UICardChannel` to `__all__`. **No module singleton** — channels
are constructed per-payload-type at the call site (e.g.
`UICardChannel(payload_type=ProposedNote)`) because `decode_verdict`
needs the InT to type-validate the inbound dict.

- [ ] **Step 4: Re-export from channels package**

Edit `src/ballast/patterns/hitl/channels/__init__.py`:

```python
"""HITL channels — request delivery surfaces."""
from ballast.patterns.hitl.channels._base import DBOSHITLChannel
from ballast.patterns.hitl.channels._protocol import (
    HITLChannel,
    InT,
    VerdictT,
)
from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict,
    UICardChannel,
    approval_card_decided,
    approval_card_requested,
    card_kind_registry,
    register_card_kind,
)

__all__ = [
    "CardVerdict",
    "DBOSHITLChannel",
    "HITLChannel",
    "InT",
    "UICardChannel",
    "VerdictT",
    "approval_card_decided",
    "approval_card_requested",
    "card_kind_registry",
    "register_card_kind",
]
```

- [ ] **Step 5: Verify `current_parent_thread_id` exists in the framework**

Run:

```
uv run python -c "from ballast.events.context import current_parent_thread_id; print(current_parent_thread_id)"
```

If `ImportError`, search for the canonical reader:

```
grep -rn "progress_to_thread" src/ballast/events/
```

Use whatever the codebase already calls the getter. If only the cm exists but no getter, add a getter in `src/ballast/events/context.py` that returns the bound thread id or `None`. Adjust the `UICardChannel.deliver` import accordingly.

- [ ] **Step 6: Run — confirm pass**

```
uv run pytest tests/patterns/hitl/test_ui_card_channel.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/ballast/patterns/hitl/channels tests/patterns/hitl/test_ui_card_channel.py
# include any events/context.py addition if needed
git add src/ballast/events/context.py 2>/dev/null || true
git commit -m "feat(hitl): UICardChannel — persist + signal + decode"
```

---

## Task 8: `Ballast.with_approval_repo(...)` builder

**Files:**
- Modify: `src/ballast/app.py` — add fluent setter that reassigns the module singleton.
- Create: `tests/app/test_with_approval_repo.py`

- [ ] **Step 1: Find the existing builder method shape**

```
grep -n "with_judge_defaults\|with_persistence\|with_events" src/ballast/app.py | head
```

Mirror the closest sibling (likely `with_judge_defaults` which we built earlier). The shape is fluent: returns `self`, mutates the in-progress Ballast spec, takes effect at build time.

- [ ] **Step 2: Write the failing test**

Create `tests/app/test_with_approval_repo.py`:

```python
"""``Ballast.with_approval_repo`` swaps the module singleton at build."""
from __future__ import annotations

from ballast.app import Ballast
from ballast.persistence.approval_card import (
    InMemoryApprovalCardRepository,
)


def test_with_approval_repo_installs_singleton() -> None:
    fresh = InMemoryApprovalCardRepository()
    app = Ballast().with_approval_repo(fresh).build()

    from ballast.persistence import approval_card as mod
    assert mod.approval_card_repo is fresh

    # Cleanup: rebuild a default Ballast so other tests aren't poisoned.
    app  # noqa: B018 — touch the var
```

- [ ] **Step 3: Implement the setter**

In `src/ballast/app.py`, follow the same pattern as `with_judge_defaults`. Add:

```python
def with_approval_repo(
    self, repo: "ApprovalCardRepository",
) -> "Ballast":
    """Configure the approval-card repository (defaults to in-memory).

    Reassigns the module-level ``approval_card_repo`` singleton at
    build time so tools / channels that do
    ``from ballast.persistence.approval_card import approval_card_repo``
    pick up the configured instance without explicit DI.
    """
    self._approval_repo = repo
    return self
```

And in `build()` (or whatever finalisation method exists — match the existing pattern):

```python
if self._approval_repo is not None:
    import ballast.persistence.approval_card as _ac
    _ac.approval_card_repo = self._approval_repo
```

Plus initialise `self._approval_repo = None` in `__init__`.

Import `ApprovalCardRepository` at the top in a `TYPE_CHECKING` block:

```python
if TYPE_CHECKING:
    from ballast.persistence.approval_card import ApprovalCardRepository
```

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/app/test_with_approval_repo.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/app.py tests/app/test_with_approval_repo.py
git commit -m "feat(app): Ballast.with_approval_repo() fluent setter"
```

---

## Task 9: REST router — list + get + decide

**Files:**
- Create: `src/ballast/api/approvals/__init__.py`
- Create: `src/ballast/api/approvals/router.py`
- Create: `tests/api/test_approvals_router.py`

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_approvals_router.py`:

```python
"""``/approvals`` REST endpoints — list / get / decide."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ballast.api.approvals import approvals_router
from ballast.auth.context import acting_as
from ballast.persistence.approval_card import (
    ApprovalCard, InMemoryApprovalCardRepository,
)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Iterator[FastAPI]:
    """Throwaway FastAPI app with only the approvals router mounted +
    a per-test approval repo singleton."""
    repo = InMemoryApprovalCardRepository()
    monkeypatch.setattr(
        "ballast.persistence.approval_card.approval_card_repo", repo,
    )
    f = FastAPI()
    f.include_router(approvals_router)
    yield f


def _seed(*, repo_id: str, user_id: str, status: str = "pending") -> ApprovalCard:
    return ApprovalCard(
        id=repo_id, workflow_id=f"wf-{repo_id}",
        respond_topic=f"hitl:{repo_id}", kind="note.create",
        payload={"title": "t", "body": "b"},
        parent_thread_id=None, user_id=user_id,
        status=status,  # type: ignore[arg-type]
        created_at=datetime(2026, 5, 25, tzinfo=UTC),
    )


def test_list_filters_by_acting_user(app: FastAPI) -> None:
    import asyncio
    from ballast.persistence import approval_card as mod
    asyncio.run(mod.approval_card_repo.add(_seed(repo_id="a", user_id="u-1")))
    asyncio.run(mod.approval_card_repo.add(_seed(repo_id="b", user_id="u-2")))

    with TestClient(app) as client, acting_as("u-1"):
        r = client.get("/approvals?status=pending")
    assert r.status_code == 200
    ids = [c["id"] for c in r.json()]
    assert ids == ["a"]


def test_get_403_when_not_owner(app: FastAPI) -> None:
    import asyncio
    from ballast.persistence import approval_card as mod
    asyncio.run(mod.approval_card_repo.add(_seed(repo_id="a", user_id="u-1")))

    with TestClient(app) as client, acting_as("u-2"):
        r = client.get("/approvals/a")
    assert r.status_code in (403, 404)


def test_decide_resolves_card_and_returns(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from ballast.persistence import approval_card as mod
    asyncio.run(mod.approval_card_repo.add(_seed(repo_id="a", user_id="u-1")))

    # Stub send_async — the wire is exercised by the spike / integration
    # test; here we only check the router builds the verdict correctly.
    sent: list[tuple[str, dict, str]] = []

    async def _fake_send_async(destination_id, message, topic=None):
        sent.append((destination_id, message, topic))

    monkeypatch.setattr(
        "ballast.api.approvals.router.Durable.send_async",
        _fake_send_async,
    )

    # Register the kind so the router can validate ``modified``.
    from pydantic import BaseModel
    from ballast.patterns.hitl.channels.ui_card import register_card_kind

    class _Note(BaseModel):
        __hitl_kind__ = "note.create"
        title: str
        body: str

    register_card_kind(_Note)

    with TestClient(app) as client, acting_as("u-1"):
        r = client.post(
            "/approvals/a/decision",
            json={"decision": "approve", "modified": None, "feedback": None},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "approved"
    assert body["resolution"]["decision"] == "approve"

    assert len(sent) == 1
    dest, msg, topic = sent[0]
    assert dest == "wf-a" and topic == "hitl:a"
    assert msg["decision"] == "approve"


def test_decide_409_when_already_resolved(app: FastAPI) -> None:
    import asyncio
    from ballast.persistence import approval_card as mod
    asyncio.run(mod.approval_card_repo.add(
        _seed(repo_id="a", user_id="u-1", status="approved"),
    ))

    with TestClient(app) as client, acting_as("u-1"):
        r = client.post(
            "/approvals/a/decision",
            json={"decision": "reject", "modified": None, "feedback": None},
        )
    assert r.status_code == 409
```

- [ ] **Step 2: Run — confirm fail**

```
uv run pytest tests/api/test_approvals_router.py -v
```

Expected: ImportError for `approvals_router`.

- [ ] **Step 3: Implement router**

Create `src/ballast/api/approvals/__init__.py`:

```python
"""``/approvals`` REST + SSE router."""
from ballast.api.approvals.router import approvals_router

__all__ = ["approvals_router"]
```

Create `src/ballast/api/approvals/router.py`:

```python
"""Approval card REST + SSE endpoints.

  GET    /approvals                          → list pending (filtered by current_user_id)
  GET    /approvals/{card_id}                → single card (403 if not yours)
  POST   /approvals/{card_id}/decision       → verdict → Durable.send_async to the suspended workflow
  GET    /approvals/stream                   → SSE multiplexer (Task 10)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, TypeAdapter

from ballast.auth.context import current_user_id
from ballast.durable import Durable
from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict,
    approval_card_decided,
    card_kind_registry,
)
from ballast.persistence.approval_card import ApprovalCard

approvals_router = APIRouter(prefix="/approvals", tags=["approvals"])


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    modified: dict[str, Any] | None = None
    feedback:  str | None = None


@approvals_router.get("", response_model=list[ApprovalCard])
async def list_approvals(
    status: Literal["pending"] = Query("pending"),
    limit:  int = Query(50, ge=1, le=200),
) -> list[ApprovalCard]:
    """Pending approvals visible to the caller (filtered by user_id
    when ``current_user_id()`` is set, unscoped otherwise)."""
    from ballast.persistence.approval_card import approval_card_repo  # noqa: PLC0415
    return await approval_card_repo.list_pending(limit=limit)


@approvals_router.get("/{card_id}", response_model=ApprovalCard)
async def get_approval(card_id: str) -> ApprovalCard:
    from ballast.persistence.approval_card import approval_card_repo  # noqa: PLC0415
    card = await approval_card_repo.get(card_id)
    if card is None:
        raise HTTPException(404, "Approval not found")
    return card


@approvals_router.post("/{card_id}/decision", response_model=ApprovalCard)
async def decide_approval(
    card_id: str, body: DecisionRequest,
) -> ApprovalCard:
    from ballast.persistence.approval_card import approval_card_repo  # noqa: PLC0415

    card = await approval_card_repo.get(card_id)
    if card is None:
        raise HTTPException(404, "Approval not found")
    if current_user_id() is not None and card.user_id != current_user_id():
        raise HTTPException(403, "Not your approval")
    if card.status != "pending":
        raise HTTPException(
            409, f"Card already {card.status}",
        )

    # Validate ``modified`` (if present) against the kind's registered model.
    payload_model = card_kind_registry.get(card.kind)
    if body.modified is not None and payload_model is not None:
        modified_typed = TypeAdapter(payload_model).validate_python(body.modified)
    else:
        modified_typed = body.modified

    verdict = CardVerdict[payload_model or dict](
        decision=body.decision,
        modified=modified_typed,
        feedback=body.feedback,
        answered_at=datetime.now(timezone.utc),
    )

    await Durable.send_async(
        destination_id=card.workflow_id,
        message=verdict.model_dump(mode="json"),
        topic=card.respond_topic,
    )
    resolved = await approval_card_repo.resolve(card.id, verdict=verdict)
    await approval_card_decided.send_async(None, card=resolved)
    return resolved
```

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/api/test_approvals_router.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/api/approvals tests/api/test_approvals_router.py
git commit -m "feat(api): /approvals REST router — list/get/decide"
```

---

## Task 10: `/approvals/stream` SSE multiplexer

**Files:**
- Modify: `src/ballast/api/approvals/router.py` — add stream endpoint.
- Modify: `tests/api/test_approvals_router.py` — add SSE smoke test.

- [ ] **Step 1: Find the existing SSE pattern**

```
grep -rn "sse_starlette\|StreamingResponse\|EventSource" src/ballast/api/ | head
```

Use the same SSE helper / encoder the existing streaming router uses. The most likely existing tool is `sse-starlette`'s `EventSourceResponse` or a project helper.

- [ ] **Step 2: Add the SSE test**

Append to `tests/api/test_approvals_router.py`:

```python
@pytest.mark.asyncio
async def test_stream_emits_card_events(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A card-requested signal fires an SSE 'card-requested' event."""
    import asyncio
    import json

    from ballast.persistence import approval_card as mod
    card = _seed(repo_id="a", user_id="u-1")

    with TestClient(app) as client:
        # Open stream then fire signal in another task.
        async def _emit() -> None:
            await asyncio.sleep(0.1)
            from ballast.patterns.hitl.channels.ui_card import approval_card_requested
            await mod.approval_card_repo.add(card)
            await approval_card_requested.send_async(None, card=card)

        # TestClient.stream is sync; run signal emission in background thread.
        import threading
        threading.Thread(
            target=lambda: asyncio.run(_emit()), daemon=True,
        ).start()

        events: list[str] = []
        with client.stream("GET", "/approvals/stream") as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("event:") or line.startswith("data:"):
                    events.append(line)
                if len(events) >= 2:
                    break

    assert any("card-requested" in e for e in events)
```

- [ ] **Step 3: Implement SSE endpoint**

Append to `src/ballast/api/approvals/router.py`:

```python
from collections.abc import AsyncIterator
from sse_starlette.sse import EventSourceResponse


@approvals_router.get("/stream")
async def stream_approvals(request: Request) -> EventSourceResponse:
    """Multiplex ``approval_card_requested`` + ``approval_card_decided``
    signals as SSE events. Disconnect-aware via ``request.is_disconnected``.
    """
    from ballast.patterns.hitl.channels.ui_card import (             # noqa: PLC0415
        approval_card_decided, approval_card_requested,
    )

    queue: asyncio.Queue[tuple[str, ApprovalCard]] = asyncio.Queue()

    async def _on_request(sender: Any, *, card: ApprovalCard, **_: Any) -> None:
        await queue.put(("card-requested", card))

    async def _on_decided(sender: Any, *, card: ApprovalCard, **_: Any) -> None:
        await queue.put(("card-decided", card))

    approval_card_requested.connect(_on_request, weak=False)
    approval_card_decided.connect(_on_decided, weak=False)

    async def _gen() -> AsyncIterator[dict[str, str]]:
        try:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event_name, card = await asyncio.wait_for(
                        queue.get(), timeout=15.0,
                    )
                    yield {
                        "event": event_name,
                        "data":  card.model_dump_json(),
                    }
                except asyncio.TimeoutError:
                    yield {"event": "heartbeat", "data": ""}
        finally:
            approval_card_requested.disconnect(_on_request)
            approval_card_decided.disconnect(_on_decided)

    return EventSourceResponse(_gen())
```

Add `import asyncio` near the top if missing.

If `sse_starlette` isn't a dep, use whichever SSE helper the streaming router uses (probably exposed via `ballast.api.streaming`). Mirror that.

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/api/test_approvals_router.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/api/approvals/router.py tests/api/test_approvals_router.py
git commit -m "feat(api): /approvals/stream SSE multiplexer for card signals"
```

---

## Task 11: Wire `/approvals` router into `Ballast.fastapi_app`

**Files:**
- Modify: `src/ballast/app.py` — include the router in the assembled FastAPI app.
- Modify: existing `tests/api/...` smoke test (or add one) — verify the route is mounted on the engine-assembled app.

- [ ] **Step 1: Find where existing routers are mounted**

```
grep -n "include_router\|fastapi_app" src/ballast/app.py | head
```

- [ ] **Step 2: Add include_router call alongside the existing ones**

In `src/ballast/app.py`, wherever the engine builds the FastAPI instance:

```python
from ballast.api.approvals import approvals_router
...
app.include_router(approvals_router)
```

Match the existing pattern's import-location (lazy import inside the factory if that's the convention).

- [ ] **Step 3: Smoke test the route is registered**

Add to whichever app-level test file already exists (e.g. `tests/app/test_fastapi_app.py`):

```python
def test_approvals_router_mounted() -> None:
    from ballast.app import Ballast
    app = Ballast().build().fastapi_app()
    routes = {r.path for r in app.routes}
    assert "/approvals" in routes
    assert "/approvals/{card_id}/decision" in routes
    assert "/approvals/stream" in routes
```

If `tests/app/test_fastapi_app.py` doesn't exist, create it with just this test.

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/app/ -v
```

Expected: app tests pass including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/app.py tests/app/test_fastapi_app.py
git commit -m "feat(app): mount /approvals router on Ballast.fastapi_app"
```

---

## Task 12: Chat-marker thread events

**Files:**
- Modify: `src/ballast/patterns/hitl/channels/ui_card.py` — emit chat marker from `deliver`.
- Modify: `tests/patterns/hitl/test_ui_card_channel.py` — assert the marker event is fired.

- [ ] **Step 1: Find the persistent thread-event emitter API**

The framework already has a primitive for emitting persistent `data-*` thread events (used by Reflection / DivergentConvergent progress events). Find it:

```
grep -rn "data-reflection-\|emit_persistent_event\|emit_raw" src/ballast/events/ src/ballast/agents/ | head
```

Most likely something like `ThreadEventBroadcaster.emit_raw(thread_id, payload)` or a helper named `emit_thread_event(...)`. Use whatever the existing pattern is — do NOT invent a new primitive.

- [ ] **Step 2: Extend the test**

Add to `tests/patterns/hitl/test_ui_card_channel.py`:

```python
@pytest.mark.asyncio
async def test_deliver_emits_data_approval_pending_when_thread_scoped(
    fresh_repo: InMemoryApprovalCardRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside a ``progress_to_thread`` scope, ``deliver`` posts a
    persistent ``data-approval-pending`` event to that thread so the
    chat shows a 'waiting for your approval' marker."""
    emitted: list[tuple[str, dict]] = []

    async def _spy_emit(thread_id: str, payload: dict) -> None:
        emitted.append((thread_id, payload))

    # Path depends on the framework's actual emitter — adjust monkeypatch
    # target after the grep in step 1.
    monkeypatch.setattr(
        "ballast.patterns.hitl.channels.ui_card._emit_thread_marker",
        _spy_emit,
    )

    chan: UICardChannel[_Note] = UICardChannel(payload_type=_Note)
    with progress_to_thread("thread-1"):
        await chan.deliver(
            request_id="req-1", workflow_id="wf-1",
            respond_topic="hitl:req-1",
            payload=_Note(title="t", body="b"),
        )

    assert len(emitted) == 1
    thread_id, payload = emitted[0]
    assert thread_id == "thread-1"
    assert payload["type"] == "data-approval-pending"
    assert payload["data"]["card_id"] == "req-1"
    assert payload["data"]["kind"] == "note.create"


@pytest.mark.asyncio
async def test_deliver_skips_marker_without_thread_scope(
    fresh_repo: InMemoryApprovalCardRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside ``progress_to_thread``, no marker is emitted (no thread
    to post to)."""
    emitted: list[tuple[str, dict]] = []
    async def _spy_emit(thread_id: str, payload: dict) -> None:
        emitted.append((thread_id, payload))
    monkeypatch.setattr(
        "ballast.patterns.hitl.channels.ui_card._emit_thread_marker",
        _spy_emit,
    )

    chan: UICardChannel[_Note] = UICardChannel(payload_type=_Note)
    await chan.deliver(
        request_id="req-1", workflow_id="wf-1",
        respond_topic="hitl:req-1",
        payload=_Note(title="t", body="b"),
    )
    assert emitted == []
```

- [ ] **Step 3: Implement marker emission**

In `src/ballast/patterns/hitl/channels/ui_card.py`, add a helper function and a `data-approval-resolved` mirror emitter triggered by `approval_card_decided`. Sketch:

```python
async def _emit_thread_marker(thread_id: str, payload: dict) -> None:
    """Adapter to the framework's persistent thread-event emitter.

    The exact emitter API is project-internal — fill in based on the
    grep result from step 1. Likely shape::

        from ballast.events.thread_events import emit_raw
        await emit_raw(thread_id=thread_id, payload=payload)
    """
    from ballast.events.thread_events import emit_raw  # noqa: PLC0415
    await emit_raw(thread_id=thread_id, payload=payload)
```

In `UICardChannel.deliver`, after the signal send:

```python
if card.parent_thread_id is not None:
    await _emit_thread_marker(
        card.parent_thread_id,
        {
            "type": "data-approval-pending",
            "data": {"card_id": card.id, "kind": card.kind},
        },
    )
```

Plus a module-level signal listener that posts `data-approval-resolved` on resolution:

```python
async def _on_decided(_sender: Any, *, card: ApprovalCard, **_: Any) -> None:
    if card.parent_thread_id is None:
        return
    await _emit_thread_marker(
        card.parent_thread_id,
        {
            "type": "data-approval-resolved",
            "data": {
                "card_id": card.id,
                "approved": card.status == "approved",
            },
        },
    )

approval_card_decided.connect(_on_decided, weak=False)
```

- [ ] **Step 4: Run — confirm pass**

```
uv run pytest tests/patterns/hitl/test_ui_card_channel.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ballast/patterns/hitl/channels/ui_card.py tests/patterns/hitl/test_ui_card_channel.py
git commit -m "feat(hitl): emit data-approval-{pending,resolved} chat markers"
```

---

## Task 13: Notes-app `create_note_flow` child workflow

**Files:**
- Create: `examples/notes-app/backend/src/notes_app/workflows/create_note.py`
- Modify: `examples/notes-app/backend/src/notes_app/agents/notes.py` — replace `create_note` body.
- Modify: `examples/notes-app/backend/src/notes_app/agents/note_refiner.py` — `ProposedNote.__hitl_kind__`.

- [ ] **Step 1: Write the failing test**

Create `examples/notes-app/backend/tests/test_create_note_approval.py`:

```python
"""End-to-end test for the new ``create_note`` flow:

  1. Tool builds the ProposedNote draft.
  2. (Refiner may run; here it's None — no API key.)
  3. ``create_note_flow`` child workflow calls ``UICardChannel(payload_type=ProposedNote).request(...)``
     which suspends on ``Durable.recv_async``.
  4. From outside, send the approve verdict via ``Durable.send_async``.
  5. Tool returns the persisted note.

This test uses the real UICardChannel + repo + DBOS. It validates the
integration end-to-end without involving an actual LLM run.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ballast.durable import Durable
from ballast.patterns.hitl.channels.ui_card import (
    CardVerdict, register_card_kind,
)
from notes_app.agents.note_refiner import ProposedNote
from notes_app.repositories.note import InMemoryNoteRepository
from notes_app.workflows.create_note import create_note_flow


register_card_kind(ProposedNote)


async def _wait_card_pending(
    repo: "ballast.persistence.approval_card.InMemoryApprovalCardRepository",
    timeout: float = 5.0,
) -> str:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        rows = await repo.list_pending()
        if rows:
            return rows[0].id
        await asyncio.sleep(0.05)
    raise TimeoutError("no pending card surfaced")


@pytest.mark.asyncio
async def test_create_note_approve_path(
    fresh_dbos_executor: None,
    repo: InMemoryNoteRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ballast.persistence.approval_card import InMemoryApprovalCardRepository
    approvals = InMemoryApprovalCardRepository()
    monkeypatch.setattr(
        "ballast.persistence.approval_card.approval_card_repo", approvals,
    )

    draft = ProposedNote(title="grocery", body="milk, eggs")
    handle = await Durable.start_workflow(create_note_flow, draft)

    card_id = await _wait_card_pending(approvals)
    card = await approvals.get(card_id)
    assert card is not None

    verdict = CardVerdict[ProposedNote](
        decision="approve",
        modified=None,
        answered_at=datetime.now(UTC),
    )
    await Durable.send_async(
        destination_id=card.workflow_id,
        message=verdict.model_dump(mode="json"),
        topic=card.respond_topic,
    )

    note = await handle.get_result()
    assert note is not None
    assert note.title == "grocery"

    listed = await repo.list_()
    assert [n.id for n in listed] == [note.id]


@pytest.mark.asyncio
async def test_create_note_reject_path(
    fresh_dbos_executor: None,
    repo: InMemoryNoteRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ballast.persistence.approval_card import InMemoryApprovalCardRepository
    approvals = InMemoryApprovalCardRepository()
    monkeypatch.setattr(
        "ballast.persistence.approval_card.approval_card_repo", approvals,
    )

    draft = ProposedNote(title="x", body="y")
    handle = await Durable.start_workflow(create_note_flow, draft)
    card_id = await _wait_card_pending(approvals)
    card = await approvals.get(card_id)

    verdict = CardVerdict[ProposedNote](
        decision="reject",
        answered_at=datetime.now(UTC),
    )
    await Durable.send_async(
        destination_id=card.workflow_id,
        message=verdict.model_dump(mode="json"),
        topic=card.respond_topic,
    )

    note = await handle.get_result()
    assert note is None

    listed = await repo.list_()
    assert listed == []
```

- [ ] **Step 2: Add `__hitl_kind__` to `ProposedNote`**

Edit `examples/notes-app/backend/src/notes_app/agents/note_refiner.py` — add at top of `ProposedNote`:

```python
class ProposedNote(BaseModel):
    """Draft + refined shape Reflection passes around the loop."""

    __hitl_kind__ = "note.create"

    title: str
    body: str
```

- [ ] **Step 3: Implement `create_note_flow`**

Create `examples/notes-app/backend/src/notes_app/workflows/create_note.py`:

```python
"""``create_note_flow`` — child ``@Durable.workflow`` that gates note
persistence behind a UI-card approval.

Extracted from the tool body so the human-pause unit is its own DBOS
workflow (visible independently in the inspector; recoverable on
restart without re-driving the entire agent run).
"""
from __future__ import annotations

from ballast.durable import Durable
from ballast.patterns.hitl.channels.ui_card import (
    UICardChannel, register_card_kind,
)

from notes_app.agents.note_refiner import ProposedNote
from notes_app.models.note import Note
from notes_app.repositories.note import notes_repo

register_card_kind(ProposedNote)

# Channel needs the payload type to decode the verdict's `modified`
# field into a typed ProposedNote on resume. Constructed once at module
# import; safe to share across invocations (stateless besides the
# captured payload type).
_channel: UICardChannel[ProposedNote] = UICardChannel(payload_type=ProposedNote)


@Durable.workflow()
async def create_note_flow(draft: ProposedNote) -> Note | None:
    """Ask the user to approve persisting ``draft``; save on approve,
    return ``None`` on reject."""
    verdict = await _channel.request(draft)
    if verdict.decision != "approve":
        return None
    final = verdict.modified or draft
    return await notes_repo.create(title=final.title, body=final.body)
```

- [ ] **Step 4: Rewire `create_note` tool body**

Edit `examples/notes-app/backend/src/notes_app/agents/notes.py` — replace the `create_note` body:

```python
@NotesAgent.tool
async def create_note(
    ctx: RunContext[NoteToolDeps], title: str, body: str,
) -> str:
    """Save a note for the current user, gated by a UI approval card.

    Refines the draft, then suspends on ``create_note_flow``; the user
    sees the card in the side-panel "Approvals" drawer (separate from
    chat). On approve the note is persisted; on reject the save is
    cancelled. While waiting, a "Waiting for your approval →" pill
    appears in the chat thread.
    """
    from contextlib import nullcontext  # noqa: PLC0415
    from ballast.events import progress_to_thread  # noqa: PLC0415

    from notes_app.agents.note_refiner import ProposedNote, note_refiner  # noqa: PLC0415
    from notes_app.workflows.create_note import create_note_flow  # noqa: PLC0415

    draft = ProposedNote(title=title, body=body)
    scope = (
        progress_to_thread(ctx.deps.parent_thread_id)
        if ctx.deps.parent_thread_id is not None
        else nullcontext()
    )
    with scope:
        # Refinement loop unchanged — only the persistence step is
        # now gated by approval.
        if note_refiner is not None:
            from ballast.patterns import ReflectionExhausted  # noqa: PLC0415
            try:
                refined = await note_refiner.run(draft)
            except ReflectionExhausted as exc:
                refined = exc.last_draft
        else:
            refined = draft

        note = await create_note_flow(refined)

    if note is None:
        return "Note save cancelled by user."
    return f"Saved note '{note.title}'."
```

Note: previously the tool returned `Note` — now it returns `str`. Adjust the tool's declared return type accordingly (if the agent's return-type schema cares). Update the existing `test_create_note_persists_via_repo` test in `examples/notes-app/backend/tests/test_note_tools.py` to either be deleted (replaced by the new approval test) or to use a stub channel that auto-approves.

- [ ] **Step 5: Update the old test to use stub auto-approve channel OR remove**

Open `examples/notes-app/backend/tests/test_note_tools.py`. The original `test_create_note_persists_via_repo` directly invoked the tool function. Since the tool now suspends on a real channel, either:

  - **Delete** that one test (covered by the new `test_create_note_approval.py`), OR
  - **Refit** it: monkeypatch `notes_app.workflows.create_note.create_note_flow` with a fake that just calls `notes_repo.create` directly.

Simpler: delete it. The 16 other tests in that file are about tool prepare / search / delete / etc., unaffected.

- [ ] **Step 6: Run notes-app tests**

```
cd examples/notes-app/backend && uv run pytest --tb=short -q
```

Expected: all tests pass including the two new approval tests.

- [ ] **Step 7: Commit**

```bash
git add examples/notes-app/backend
git commit -m "feat(notes-app): create_note gated by UICardChannel approval"
```

---

## Task 14: Wire `Ballast.with_approval_repo` in notes-app main

**Files:**
- Modify: `examples/notes-app/backend/src/notes_app/main.py` — chain `.with_approval_repo(InMemoryApprovalCardRepository())`.

- [ ] **Step 1: Edit `main.py`**

Find the Ballast builder chain and add the line:

```python
from ballast.persistence.approval_card import InMemoryApprovalCardRepository

ballast = (
    Ballast()
    .with_judge_defaults(...)
    .with_approval_repo(InMemoryApprovalCardRepository())
    .build()
)
```

- [ ] **Step 2: Smoke run**

```
cd examples/notes-app/backend && uv run pytest --tb=short -q
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add examples/notes-app/backend/src/notes_app/main.py
git commit -m "feat(notes-app): wire InMemoryApprovalCardRepository via builder"
```

---

## Task 15: Frontend — `useApprovals` hook

**Files:**
- Create: `examples/notes-app/frontend/src/components/approvals/use-approvals.ts`

This task only adds the hook (data layer); the panel UI comes in Task 16. No test framework is established for the frontend — manual smoke later via the dev server.

- [ ] **Step 1: Create the hook**

```ts
"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export type ApprovalCard = {
  id: string;
  workflow_id: string;
  respond_topic: string;
  kind: string;
  payload: Record<string, unknown>;
  parent_thread_id: string | null;
  user_id: string | null;
  status: "pending" | "approved" | "rejected" | "timeout";
  created_at: string;
};

export type Decision = {
  decision: "approve" | "reject";
  modified?: Record<string, unknown> | null;
  feedback?: string | null;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function useApprovals() {
  const [pending, setPending] = useState<ApprovalCard[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  const fetchInitial = useCallback(async () => {
    const r = await fetch(`${API_BASE}/approvals?status=pending`);
    if (r.ok) setPending(await r.json());
  }, []);

  useEffect(() => {
    fetchInitial();

    const es = new EventSource(`${API_BASE}/approvals/stream`);
    eventSourceRef.current = es;

    es.addEventListener("card-requested", (e: MessageEvent) => {
      const card: ApprovalCard = JSON.parse(e.data);
      setPending((prev) =>
        prev.find((c) => c.id === card.id) ? prev : [card, ...prev],
      );
    });
    es.addEventListener("card-decided", (e: MessageEvent) => {
      const card: ApprovalCard = JSON.parse(e.data);
      setPending((prev) => prev.filter((c) => c.id !== card.id));
    });

    return () => {
      es.close();
    };
  }, [fetchInitial]);

  const decide = useCallback(
    async (cardId: string, decision: Decision) => {
      const r = await fetch(`${API_BASE}/approvals/${cardId}/decision`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(decision),
      });
      if (!r.ok) throw new Error(`decide failed: ${r.status}`);
      // Optimistic remove; SSE card-decided will reconcile.
      setPending((prev) => prev.filter((c) => c.id !== cardId));
    },
    [],
  );

  return { pending, decide };
}
```

- [ ] **Step 2: Commit**

```bash
git add examples/notes-app/frontend/src/components/approvals/use-approvals.ts
git commit -m "feat(notes-app/fe): useApprovals hook — fetch + SSE + decide"
```

---

## Task 16: Frontend — `<ApprovalsPanel/>` + per-kind renderers

**Files:**
- Create: `examples/notes-app/frontend/src/components/approvals/approvals-panel.tsx`
- Create: `examples/notes-app/frontend/src/components/approvals/card-renderers/note-create.tsx`

- [ ] **Step 1: Per-kind renderer**

Create `card-renderers/note-create.tsx`:

```tsx
import type { ApprovalCard } from "../use-approvals";

export function NoteCreateCard({
  card,
  onApprove,
  onReject,
}: {
  card: ApprovalCard;
  onApprove: () => void;
  onReject: () => void;
}) {
  const p = card.payload as { title: string; body: string };
  return (
    <div className="rounded border p-3 my-2 bg-white">
      <div className="text-xs text-gray-500 mb-1">Save note?</div>
      <div className="font-semibold">{p.title}</div>
      <div className="text-sm whitespace-pre-wrap mt-1">{p.body}</div>
      <div className="flex gap-2 mt-3">
        <button
          className="px-3 py-1 rounded bg-emerald-600 text-white"
          onClick={onApprove}
        >
          Approve
        </button>
        <button
          className="px-3 py-1 rounded bg-gray-200"
          onClick={onReject}
        >
          Reject
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Panel**

Create `approvals-panel.tsx`:

```tsx
"use client";

import { useApprovals } from "./use-approvals";
import { NoteCreateCard } from "./card-renderers/note-create";

const RENDERERS: Record<
  string,
  React.ComponentType<{
    card: import("./use-approvals").ApprovalCard;
    onApprove: () => void;
    onReject: () => void;
  }>
> = {
  "note.create": NoteCreateCard,
};

export function ApprovalsPanel({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { pending, decide } = useApprovals();

  return (
    <aside
      className={`fixed top-0 right-0 h-full w-96 bg-gray-50 border-l shadow-lg transform transition-transform ${
        open ? "translate-x-0" : "translate-x-full"
      }`}
    >
      <header className="p-3 flex items-center justify-between border-b">
        <div className="font-semibold">Approvals</div>
        <button onClick={onClose} className="text-sm">Close</button>
      </header>
      <div className="p-3 overflow-y-auto h-[calc(100%-3rem)]">
        {pending.length === 0 && (
          <div className="text-sm text-gray-500">Nothing pending.</div>
        )}
        {pending.map((card) => {
          const Renderer = RENDERERS[card.kind];
          if (!Renderer) {
            return (
              <div key={card.id} className="text-sm text-red-600">
                No renderer for kind {card.kind}.
              </div>
            );
          }
          return (
            <Renderer
              key={card.id}
              card={card}
              onApprove={() => decide(card.id, { decision: "approve" })}
              onReject={() => decide(card.id, { decision: "reject" })}
            />
          );
        })}
      </div>
    </aside>
  );
}

export function ApprovalsBadge({
  count,
  onClick,
}: {
  count: number;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="relative px-3 py-1 rounded bg-gray-100"
    >
      Approvals
      {count > 0 && (
        <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full px-1.5">
          {count}
        </span>
      )}
    </button>
  );
}
```

- [ ] **Step 3: Wire into app shell**

Find the app's top-level page / layout component (likely `examples/notes-app/frontend/src/app/page.tsx` or `layout.tsx`). Add:

```tsx
"use client";

import { useState } from "react";
import { ApprovalsPanel, ApprovalsBadge } from "@/components/approvals/approvals-panel";
import { useApprovals } from "@/components/approvals/use-approvals";

// inside the component:
const [drawerOpen, setDrawerOpen] = useState(false);
const { pending } = useApprovals();

// in the JSX header:
<ApprovalsBadge count={pending.length} onClick={() => setDrawerOpen(true)} />
<ApprovalsPanel open={drawerOpen} onClose={() => setDrawerOpen(false)} />
```

Calling `useApprovals()` from both the header (for the badge count) and the panel mounts two SSE connections — fix in a follow-up by lifting the state into a context provider if it bothers you. For MVP it's fine.

- [ ] **Step 4: Smoke test in the dev browser**

```
cd examples/notes-app/backend && uv run uvicorn notes_app.main:app --reload &
cd examples/notes-app/frontend && pnpm dev &
```

Open `http://localhost:3000`. Ask the assistant to "save a note about groceries". Verify:

  1. Chat shows "Waiting for your approval →" pill (data-approval-pending).
  2. Approvals badge in header increments to 1.
  3. Drawer opens, shows the note title/body + Approve/Reject buttons.
  4. Click Approve → drawer empties, chat pill flips to "Approved ✓", final tool reply "Saved note 'groceries'." appears.
  5. Repeat with Reject → tool replies "Note save cancelled by user.", no note saved.

If any step fails, debug before moving on.

- [ ] **Step 5: Commit**

```bash
git add examples/notes-app/frontend
git commit -m "feat(notes-app/fe): ApprovalsPanel side drawer + note.create renderer"
```

---

## Task 17: Chat marker data-parts

**Files:**
- Create: `examples/notes-app/frontend/src/components/data-parts/data-approval-pending.tsx`
- Create: `examples/notes-app/frontend/src/components/data-parts/data-approval-resolved.tsx`
- Register both with the assistant-ui adapter (follow the convention of existing `data-*` parts — `grep -rn "makeAssistantDataUI\|data-reflection" examples/notes-app/frontend/src/`).

- [ ] **Step 1: Find existing data-part registration pattern**

```
grep -rn "data-reflection\|makeAssistantDataUI" examples/notes-app/frontend/src/
```

Mirror the existing pattern (e.g. the Reflection progress data-parts).

- [ ] **Step 2: Pending renderer**

```tsx
import type { ToolPart } from "@assistant-ui/react";

type Payload = { card_id: string; kind: string };

export function DataApprovalPending({ data }: { data: Payload }) {
  return (
    <a
      href={`#approval-${data.card_id}`}
      className="text-sm text-blue-600 underline"
    >
      Waiting for your approval →
    </a>
  );
}
```

- [ ] **Step 3: Resolved renderer**

```tsx
type Payload = { card_id: string; approved: boolean };

export function DataApprovalResolved({ data }: { data: Payload }) {
  return (
    <span
      className={`text-sm ${
        data.approved ? "text-emerald-600" : "text-gray-500"
      }`}
    >
      {data.approved ? "Approved ✓" : "Rejected ✗"}
    </span>
  );
}
```

- [ ] **Step 4: Register both renderers**

In whichever module wires data-parts into the assistant-ui adapter, add:

```ts
import { DataApprovalPending } from "@/components/data-parts/data-approval-pending";
import { DataApprovalResolved } from "@/components/data-parts/data-approval-resolved";

const dataParts = {
  ...existing,
  "data-approval-pending":  DataApprovalPending,
  "data-approval-resolved": DataApprovalResolved,
};
```

- [ ] **Step 5: Manual smoke**

Restart dev servers if running. Repeat the smoke flow from Task 16 step 4 and confirm the chat now shows the pill (and that it flips on resolve).

- [ ] **Step 6: Commit**

```bash
git add examples/notes-app/frontend/src/components/data-parts examples/notes-app/frontend/src/<wherever data-parts are registered>
git commit -m "feat(notes-app/fe): data-approval-{pending,resolved} chat markers"
```

---

## Task 18: Final smoke — full framework + notes-app suite

- [ ] **Step 1: Run framework suite**

```
uv run pytest tests/ --tb=short -q
```

Expected: green (no regressions).

- [ ] **Step 2: Run notes-app suite**

```
cd examples/notes-app/backend && uv run pytest --tb=short -q
```

Expected: green.

- [ ] **Step 3: Manual end-to-end in browser**

Repeat Task 16 step 4 smoke flow. Verify the full UX:
  - card surfaces in panel
  - chat shows pending pill
  - approve → save + pill flips to ✓
  - reject → no save + pill flips to ✗

- [ ] **Step 4: Commit (if any tweaks)**

```bash
git status && git diff
# commit any cleanup needed
```

---

## Follow-up plan (out of scope here)

A separate plan should cover:

1. **`ThreadChannel` extraction** — extract today's helper-thread flow from `DurableHITLWorkflow` into a clean `ThreadChannel(DBOSHITLChannel)` implementation. Behavior-preserving refactor.
2. **`propose_todo` migration** — replace the `DurableHITLWorkflow` subclass with a plain `@Durable.workflow` + `ThreadChannel.request`.
3. **Delete legacy** — `ask_human`, `DurableHITLWorkflow` ABC, `HITLResponse` union (`Approved/Rejected/Modified/Timeout`), `HITLContext` model.
4. **Postgres repo + Alembic migration** — production-grade approval card storage.
5. **Card-edit form** — frontend renderer that allows tweaking `title`/`body` before approve; uses the existing `modified` field on `CardVerdict`.
6. **Multi-user RBAC** — per-card approver list, delegation.
7. **Slack channels** — `SlackMessageChannel` (DM with interactive buttons) + `SlackThreadChannel` (Slack-bot agent).
