# HITL Channels + UI Card Approval — Design

**Date:** 2026-05-25
**Status:** Approved (brainstorming) → ready for implementation plan
**Author:** Kir + Claude

## Problem

`create_note` saves the (title, body) draft directly after the
Reflection loop polishes it. We need a human approval gate before
persistence, with strict UX constraints:

  - **NOT** an inline approval card embedded in the chat (existing
    `delete_note` `requires_approval` pattern).
  - **NOT** a helper sub-thread with an approval agent (existing
    `propose_todo` pattern).
  - **YES**: a separate UI surface — a side panel/drawer with badge
    count; the chat shows a small "Waiting for your approval →" pill
    deep-linking to it.

Future channels (Slack DM, Slack thread, custom user channels) must
plug in without re-architecting the suspend/resume machinery.

## Core insight

The HITL suspend/resume primitive is already correct:

  - `Durable.recv_async(topic=f"hitl:{request_id}")` on the caller
  - `Durable.send_async(workflow_id, response, topic=...)` from
    whoever delivers the verdict

What varies across mediums (thread / UI card / Slack DM / Slack
thread) is **only the request delivery surface + the verdict wire**.
The suspend mechanism and DBOS topic convention are identical.

The right abstraction is therefore a single `HITLChannel` Protocol
where each channel owns its entire lifecycle (deliver, await, decode).
No `ask_human` function, no `DurableHITLWorkflow` ABC, no global
`HITLResponse` union — those are subsumed.

## Design

### 1. `HITLChannel` Protocol — one method

```python
# src/ballast/patterns/hitl/channels/_protocol.py
InT      = TypeVar("InT",      bound=BaseModel)
VerdictT = TypeVar("VerdictT", bound=BaseModel)

class HITLChannel(Protocol, Generic[InT, VerdictT]):
    """Owns the full request lifecycle for one human decision.

    A channel knows what payload type it accepts, how to surface the
    request (UI, chat thread, Slack, …), how to wait for the verdict,
    and how to decode the response into a typed model. The framework
    knows nothing about the medium.
    """

    async def request(
        self,
        payload: InT,
        *,
        timeout: timedelta | None = None,
    ) -> VerdictT: ...
```

That is the entire framework HITL contract. Anything else is a
recipe / base class / channel implementation.

### 2. `DBOSHITLChannel` — base for durable channels

In practice nearly every channel needs the verdict to arrive via
`Durable.recv_async` so the workflow is recoverable. A small abstract
base captures the boilerplate:

```python
# src/ballast/patterns/hitl/channels/_base.py
class DBOSHITLChannel(Generic[InT, VerdictT], ABC):
    """Channels that use DBOS topics for verdict delivery (the common
    case). Subclasses implement just deliver() + decode_verdict()."""

    async def request(
        self, payload: InT, *, timeout: timedelta | None = None,
    ) -> VerdictT:
        request_id  = Det.uuid4()
        workflow_id = DBOS.workflow_id
        topic       = f"hitl:{request_id}"
        await self.deliver(
            request_id=request_id, workflow_id=workflow_id,
            respond_topic=topic, payload=payload,
        )
        raw = await Durable.recv_async(topic=topic, timeout=timeout)
        return await self.decode_verdict(raw)

    @abstractmethod
    async def deliver(
        self, *,
        request_id: str, workflow_id: str, respond_topic: str,
        payload: InT,
    ) -> None: ...

    @abstractmethod
    async def decode_verdict(self, raw: Any) -> VerdictT: ...
```

Channels that need a non-DBOS transport (rare; future Kafka /
long-poll / etc.) implement the Protocol directly without inheriting
`DBOSHITLChannel`. Framework stays neutral.

### 3. Built-in channels

#### `ThreadChannel` — migration of helper-thread flow

```python
# src/ballast/patterns/hitl/channels/thread.py
class ThreadChannel(DBOSHITLChannel[InT, VerdictT]):
    """Opens a helper sub-thread + agent. Today's helper-thread
    code (currently in DurableHITLWorkflow / ask_human) moves here
    verbatim; no behavior change."""

    def __init__(
        self,
        *,
        helper_agent: type[BallastAgent],
        verdict_type: type[VerdictT],
        opening_message_template: str | None = None,
    ) -> None: ...

    async def deliver(self, *, request_id, workflow_id, respond_topic,
                      payload) -> None:
        # Open helper thread; persist {request_id, workflow_id,
        # respond_topic} as thread metadata; post opening assistant
        # message; register helper agent. Helper-agent tools call
        # Durable.send_async(workflow_id, verdict_dict, respond_topic)
        # when the user decides.
        ...

    async def decode_verdict(self, raw: Any) -> VerdictT:
        return TypeAdapter(self._verdict_type).validate_python(raw)
```

#### `UICardChannel` — new

```python
# src/ballast/patterns/hitl/channels/ui_card.py
class UICardChannel(DBOSHITLChannel[InT, "CardVerdict[InT]"]):
    """Persists an ApprovalCard, fires a signal so the UI panel SSE
    picks it up, then waits for POST /approvals/{id}/decision to
    push the verdict back via Durable.send_async."""

    async def deliver(self, *, request_id, workflow_id, respond_topic,
                      payload: InT) -> None:
        from ballast.persistence.approval_card import approval_card_repo  # noqa: PLC0415
        from ballast.events.context import current_parent_thread_id    # noqa: PLC0415
        from ballast.auth.context import current_user_id               # noqa: PLC0415

        card = ApprovalCard(
            id=request_id, workflow_id=workflow_id,
            respond_topic=respond_topic,
            kind=type(payload).__hitl_kind__,
            payload=payload.model_dump(mode="json"),
            parent_thread_id=current_parent_thread_id(),
            user_id=current_user_id(),
            status="pending", created_at=Det.now(),
        )
        await approval_card_repo.add(card)
        await approval_card_requested.send_async(sender=self, card=card)

    async def decode_verdict(self, raw: Any) -> "CardVerdict[InT]":
        return TypeAdapter(CardVerdict[InT]).validate_python(raw)


ui_card_channel: UICardChannel = UICardChannel()    # module singleton
```

`CardVerdict` — the standard verdict shape for card-style approvals;
caller-owned channels can ship their own:

```python
class CardVerdict(BaseModel, Generic[OutT]):
    decision: Literal["approve", "reject"]
    modified: OutT | None = None          # populated when approve-with-edits
    feedback: str | None = None
    answered_at: datetime
```

### 4. `__hitl_kind__` convention on payload models

`kind` is derived from `type(payload)`, not passed as a parameter:

```python
class ProposedNote(BaseModel):
    __hitl_kind__: ClassVar[str] = "note.create"
    title: str
    body: str
```

The card renderer keys off `kind`; the frontend registers per-kind
components (`"note.create"` → `<NoteCreateCard/>`).

### 5. `current_user_id` ContextVar

```python
# src/ballast/auth/context.py
_current_user_id: ContextVar[str | None] = ContextVar(
    "current_user_id", default=None,
)

def current_user_id() -> str | None:
    return _current_user_id.get()

@contextmanager
def acting_as(user_id: str) -> Iterator[None]:
    tok = _current_user_id.set(user_id)
    try: yield
    finally: _current_user_id.reset(tok)
```

API middleware wraps each request handler in `acting_as(user_id)`.
Notes-app sets a static dev user. Tests set per-test.

### 6. Persistence: `ApprovalCard` + `ApprovalCardRepository`

Separate from existing HITL helper-thread persistence (different
lifecycle, different queries — "list pending by user" doesn't apply
to helper threads).

```python
# src/ballast/persistence/approval_card.py
class ApprovalCard(BaseModel):
    id: str                              # == request_id
    workflow_id: str
    respond_topic: str                   # f"hitl:{id}"
    kind: str                            # "note.create", …
    payload: dict[str, Any]              # JSON dump of InT
    parent_thread_id: str | None
    user_id: str | None
    status: Literal["pending", "approved", "rejected", "timeout"]
    resolution: dict[str, Any] | None    # verdict.model_dump() on resolve
    created_at: datetime
    resolved_at: datetime | None


class ApprovalCardRepository(Protocol):
    async def add(self, card: ApprovalCard) -> None: ...
    async def get(self, card_id: str) -> ApprovalCard | None: ...
    async def list_pending(
        self, *, user_id: str | None = None, limit: int = 50,
    ) -> list[ApprovalCard]: ...
    async def resolve(
        self, card_id: str, *, verdict: BaseModel,
    ) -> ApprovalCard: ...


approval_card_repo: ApprovalCardRepository = (
    _InMemoryApprovalCardRepository()
)
# Reassigned at Ballast.build() time when an explicit repo is configured —
# same module-singleton pattern as notes_repo.
```

Implementations:
  - `InMemoryApprovalCardRepository` — tests + local dev.
  - `PostgresApprovalCardRepository` — SQLAlchemy + Alembic
    migration `0002_approval_cards.py`.

Repo enforces access on read: `list_pending(user_id=...)` and
`get(...)` filter by the caller's `current_user_id()` when configured
to do so (mirrors `ThreadRepository` per-tenant access).

### 7. REST router + SSE

```
GET    /approvals?status=pending&limit=50    → list ApprovalCard (filtered by current_user_id)
GET    /approvals/{id}                       → single (403 if not yours)
POST   /approvals/{id}/decision              → {decision: "approve"|"reject", modified?: {...}, feedback?: str}
GET    /approvals/stream  (SSE)              → live feed of approval_card_*
```

Decision handler:

1. Load card; 404 if missing, 403 if `card.user_id != current_user_id()`,
   409 if `status != pending`.
2. Validate body against the channel's verdict type (the card's `kind`
   indexes into a registry of verdict types — channels register at
   import time).
3. Build the typed verdict (e.g. `CardVerdict[ProposedNote]`).
4. `await Durable.send_async(destination_id=card.workflow_id,
   message=verdict.model_dump(), topic=card.respond_topic)`.
5. `await approval_card_repo.resolve(card.id, verdict=verdict)`.
6. Fire `approval_card_decided` signal.
7. Return resolved card.

SSE subscribes to `approval_card_requested` + `approval_card_decided`
signals; emits typed `card-*` events.

### 8. Chat marker

`UICardChannel.deliver` (when `parent_thread_id` is set) emits a
persistent thread event:

```
data-approval-pending {
  card_id: str,
  kind: str,
}
```

Assistant-ui renderer shows an inline "Waiting for your approval →"
pill linking to `/approvals#{card_id}` (opens drawer with the card
focused).

On resolution → emit `data-approval-resolved {card_id, approved: bool}`
so the inline pill flips to "Approved ✓" / "Rejected ✗".

### 9. Frontend `<ApprovalsPanel/>`

  - Side drawer toggle in app header with badge count.
  - On mount: `GET /approvals?status=pending` for initial list.
  - SSE subscription to `/approvals/stream` for live updates
    (prepend on `card-requested`, remove on `card-decided`).
  - Per-`kind` component registry. MVP: only `"note.create"`
    registered, renders title/body preview + Approve / Reject.

### 10. `create_note` integration — child `@Durable.workflow`

The tool body extracts into its own child workflow. The blocking
`request()` call lives there, not in the tool body directly.

```python
# notes_app/workflows/create_note.py
@Durable.workflow
async def create_note_flow(refined: ProposedNote) -> Note | None:
    """Owns the approval gate + save side-effect. Suspends on the
    UI-card channel's recv; returns the persisted Note or None on
    user rejection."""
    verdict = await ui_card_channel.request(refined)
    if verdict.decision != "approve":
        return None
    final = verdict.modified or refined
    return await notes_repo.create(
        title=final.title, body=final.body,
    )
```

The tool body becomes a thin shim:

```python
@NotesAgent.tool
async def create_note(ctx, title: str, body: str) -> str:
    draft = ProposedNote(title=title, body=body)
    refined = (
        draft if note_refiner is None
        else await _refine_or_fallback(draft, ctx.deps.parent_thread_id)
    )
    note = await create_note_flow(refined)        # blocks; child workflow
    return (
        f"Saved note '{note.title}'."
        if note else "Note save cancelled by user."
    )
```

Why the child workflow:

  - Approval can take hours; recovery topology is cleaner if it's
    its own named DBOS workflow (visible in inspector, restartable
    independently).
  - The DurableAgent run workflow is still blocked on the child
    (per-thread queue concurrency=1) — same suspend semantics, but
    the unit of durability is the explicit `create_note_flow` rather
    than a tool body nested inside an agent run.

### 11. `propose_todo` migration

Today: `await todo_flow.open(helper_agent=..., context=...)` via
`DurableHITLWorkflow` ABC subclass.

After: plain function + `@Durable.workflow`, no ABC:

```python
# notes_app/workflows/todo_approval.py
todo_thread_channel = ThreadChannel(
    helper_agent=NotesTodoApprovalAgent,
    verdict_type=CardVerdict[TodoProposal],   # same shape works fine
)

@Durable.workflow
async def todo_approval_flow(payload: TodoProposal) -> None:
    verdict = await todo_thread_channel.request(payload)
    if verdict.decision == "approve":
        await notes_repo.create(title=payload.title, body=payload.body)
        await _notify_parent_thread(
            payload.parent_thread_id, "Saved!",
        )


@NotesAgent.tool
async def propose_todo(ctx, title: str) -> str:
    await Durable.start_workflow(
        todo_approval_flow,
        TodoProposal(
            title=title, parent_thread_id=ctx.deps.parent_thread_id,
        ),
    )
    return "Opened a confirmation thread."
```

`DurableHITLWorkflow` ABC is deleted from the framework.

## Removed APIs

| Removed | Reason | Replacement |
|---|---|---|
| `ask_human()` function | Was orchestrating transport details (`recv_async` + decode) outside the channel; broke SRP. | `channel.request(...)` |
| `DurableHITLWorkflow` ABC | Existed only to "spawn a workflow that suspends on recv then calls a handler". | Plain `@Durable.workflow` calling `channel.request(...)`. |
| `HITLResponse` union (`Approved`/`Rejected`/`Modified`/`Timeout`) | Verdict shape is channel-specific; one-size-fits-all hierarchy is wrong shape. | Each channel ships its own verdict type (`CardVerdict`, etc.). |
| `HITLContext` | All its fields are now sourced from convention (`__hitl_kind__`) or ambient ContextVars. | `__hitl_kind__` class attr; `current_parent_thread_id()`; `current_user_id()`. |

## Error handling

  - **Decision on already-resolved card** → 409 with current
    resolution.
  - **Decision attempted by wrong user** → 403.
  - **`Durable.send_async` failure** (workflow cancelled, e.g. user
    hit Stop in chat): catch in router, mark card `timeout` with
    explanatory resolution, return 410 Gone, SSE fires `card-decided`
    so the panel removes the entry.
  - **`channel.request` timeout** → raises `TimeoutError`; child
    workflow handles by leaving the card pending (or, later, a sweep
    job flips stale cards to `timeout` and sends a synthetic verdict).
  - **Workflow crash mid-`request`**: DBOS replays from before
    `recv_async`. `channel.deliver` is invoked inside a `@Durable.step`
    inside the base class so it's memoised — card row is added once.
    On replay we re-enter `recv_async` and resume waiting.

## Testing

  - **Framework unit** — `UICardChannel.deliver` writes a card +
    fires the signal (in-memory repo, signal spy).
  - **Framework integration** — channel end-to-end: kick off a
    workflow, POST a decision via the router, assert workflow returns
    the typed verdict.
  - **Notes-app** — `test_create_note_persists_via_repo` adapts to
    a stub channel that auto-approves; assert `notes_repo.create`
    ran AND a card row was added.
  - **Notes-app** — rejection path: stub channel auto-rejects;
    assert `notes_repo.create` NOT called; tool returns cancellation
    string.
  - **API** — router tests: list (filtered), get (403/404),
    decision (200/403/404/409).
  - **Migration smoke** — alembic upgrade head + downgrade base.

## What this design deliberately does NOT do

  - **No app-level default channel.** Tools/workflows pass channels
    explicitly. (Confirmed.)
  - **No card-edit form in MVP.** `CardVerdict.modified` exists in
    the type but the panel renders Approve / Reject only. Edit form
    ships in a later iteration.
  - **No Slack / webhook channels.** Designed-for: implement
    `HITLChannel` (or `DBOSHITLChannel`) without touching `request`,
    repo, router. Not built in this iteration.
  - **No multi-approver / delegation / RBAC beyond per-user.**
    `user_id` column exists; assignment policy is out of scope.

## Files touched

**Framework — new:**

  - `src/ballast/patterns/hitl/channels/__init__.py`
  - `src/ballast/patterns/hitl/channels/_protocol.py`         (HITLChannel)
  - `src/ballast/patterns/hitl/channels/_base.py`             (DBOSHITLChannel)
  - `src/ballast/patterns/hitl/channels/thread.py`            (ThreadChannel)
  - `src/ballast/patterns/hitl/channels/ui_card.py`           (UICardChannel + CardVerdict + signals + ui_card_channel singleton)
  - `src/ballast/auth/context.py`                             (current_user_id ContextVar + acting_as)
  - `src/ballast/persistence/approval_card.py`                (model + protocol + module singleton)
  - `src/ballast/persistence/in_memory/approval_card.py`
  - `src/ballast/persistence/sql/approval_card.py`
  - `src/ballast/persistence/alembic/versions/0002_approval_cards.py`
  - `src/ballast/api/approvals/router.py`                     (REST + SSE)

**Framework — modified:**

  - `src/ballast/patterns/hitl/__init__.py`                   (drop ask_human, DurableHITLWorkflow, HITLResponse exports)
  - `src/ballast/app.py`                                      (wire approvals router + auth middleware hook)

**Framework — deleted:**

  - `src/ballast/patterns/hitl/ask.py`                        (ask_human function)
  - `src/ballast/patterns/hitl/durable.py`                    (DurableHITLWorkflow ABC)
  - `src/ballast/patterns/hitl/response.py`                   (HITLResponse union)

**Notes-app — new:**

  - `examples/notes-app/backend/src/notes_app/workflows/create_note.py` (create_note_flow)
  - `examples/notes-app/frontend/src/components/approvals-panel.tsx`
  - `examples/notes-app/frontend/src/components/data-parts/data-approval-pending.tsx`

**Notes-app — modified:**

  - `examples/notes-app/backend/src/notes_app/agents/notes.py` (create_note → flow, propose_todo → channel)
  - `examples/notes-app/backend/src/notes_app/workflows/todo_approval.py` (drop DurableHITLWorkflow subclass; plain @Durable.workflow)
  - `examples/notes-app/backend/src/notes_app/main.py` (configure approval repo via Ballast builder; install auth middleware stub)
  - Frontend header (drawer toggle + badge count).

## Open follow-ups (not blocking this iteration)

  - Card timeout sweep (background workflow flips stale pending → timeout).
  - Card-edit form + `Modified` rendering.
  - SlackMessageChannel + SlackThreadChannel.
  - Multi-approver / delegation / RBAC.
