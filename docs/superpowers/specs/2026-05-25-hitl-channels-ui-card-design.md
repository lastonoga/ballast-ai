# HITL Channels + UI Card Approval — Design

**Date:** 2026-05-25
**Status:** Approved (brainstorming) → ready for implementation plan
**Author:** Kir + Claude

## Problem

`create_note` currently saves the (title, body) draft directly after the
Reflection loop polishes it. The user wants a human approval gate
before persistence — and explicitly:

  - **NOT** an inline approval card embedded in the chat (the existing
    `delete_note` `requires_approval` pattern).
  - **NOT** a helper sub-thread with an approval agent (the existing
    `propose_todo` / `ask_human` + `ThreadChannel` pattern).
  - **YES**: a separate UI surface (side panel/drawer with badge
    count), with the chat showing a small "Waiting for your approval →"
    marker linking to it.

Future channels (Slack DM, Slack thread, webhook) must drop in without
re-architecting the suspend/resume machinery.

## Insight

The existing HITL suspend/resume primitive is already correct:

  - `Durable.recv_async(topic=f"hitl:{request_id}")` on the caller side
  - `Durable.send_async(workflow_id, response, topic=...)` from
    whoever is delivering the verdict
  - `HITLResponse` discriminated union (`Approved` / `Rejected` /
    `Modified` / `Timeout`)

What varies between in-thread / UI-card / Slack-DM / Slack-thread is
**only the request delivery surface** — the medium in which the
request is shown to the human and the wire by which the verdict comes
back. The suspension topic + response type are identical.

So the right abstraction is a **`HITLChannel` Protocol**, not a new
"approval request" pattern. `ask_human` becomes channel-pluggable;
today's helper-thread behavior is one implementation.

## Design

### 1. `HITLChannel` Protocol

```python
# src/ballast/patterns/hitl/channels/_protocol.py
from typing import Protocol
from pydantic import BaseModel

class HITLChannel(Protocol):
    """Delivers an HITL request to a human via some medium.

    Channels never touch DBOS topics directly — they receive
    ``respond_topic`` and are responsible only for surfacing the
    request (in a thread, UI card, Slack message, etc.) along with
    a way for the verdict to reach ``Durable.send_async`` with that
    topic. Suspend + verdict-unwrap live in ``ask_human``.
    """

    async def deliver(
        self,
        *,
        request_id: str,           # uuid for this request
        workflow_id: str,          # caller's DBOS workflow id (recv target)
        prompt: str,               # short, channel-rendered question
        payload: BaseModel,        # opaque to channel; rendered by kind
        context: "HITLContext",    # parent_thread_id, user_id, kind, ...
        respond_topic: str,        # f"hitl:{request_id}"; channels embed
                                   # this in their verdict pathway
    ) -> None: ...
```

`HITLContext`:

```python
class HITLContext(BaseModel):
    kind: str                          # e.g. "note.create" — for renderers/routing
    parent_thread_id: str | None       # for chat marker / deep-link back
    user_id: str | None = None         # future multi-user routing
```

### 2. `ask_human` becomes channel-agnostic

```python
# src/ballast/patterns/hitl/ask.py — REWRITTEN
async def ask_human(
    *,
    channel: HITLChannel,
    prompt: str,
    payload: BaseModel,
    context: HITLContext,
    timeout: timedelta | None = None,
) -> HITLResponse:
    request_id = Det.uuid4()
    workflow_id = DBOS.workflow_id
    await channel.deliver(
        request_id=request_id, workflow_id=workflow_id,
        prompt=prompt, payload=payload, context=context,
        respond_topic=f"hitl:{request_id}",
    )
    return await Durable.recv_async(
        topic=f"hitl:{request_id}", timeout=timeout,
    )
```

The current `ask_human(helper_agent=..., context=..., opening_message=..., ...)`
signature is removed. Today's only caller (`propose_todo`) migrates to
explicit `channel=ThreadChannel(NotesTodoApprovalAgent)`.

### 3. Channel implementations

#### `ThreadChannel` (wraps existing helper-thread flow)

```python
# src/ballast/patterns/hitl/channels/thread.py
class ThreadChannel(HITLChannel):
    def __init__(
        self,
        helper_agent: type[BallastAgent],
        *,
        opening_message_template: str | None = None,
    ) -> None: ...

    async def deliver(self, *, request_id, workflow_id, prompt, payload,
                      context, respond_topic) -> None:
        # Open helper thread, persist metadata {request_id, workflow_id,
        # respond_topic}, post the opening assistant message, register
        # the helper agent. Existing helper-thread logic moves here
        # verbatim — no behavior change.
        ...
```

`DurableHITLWorkflow.open` already does this work — `ThreadChannel.deliver`
is its body extracted into the protocol.

#### `UICardChannel` (new)

```python
# src/ballast/patterns/hitl/channels/ui_card.py
class UICardChannel(HITLChannel):
    def __init__(
        self,
        *,
        repo: ApprovalCardRepository,
        marker_emitter: ChatMarkerEmitter | None = None,
    ) -> None: ...

    async def deliver(self, *, request_id, workflow_id, prompt, payload,
                      context, respond_topic) -> None:
        card = ApprovalCard(
            id=request_id, workflow_id=workflow_id,
            respond_topic=respond_topic,
            kind=context.kind, prompt=prompt,
            payload=payload.model_dump(mode="json"),
            parent_thread_id=context.parent_thread_id,
            user_id=context.user_id,
            status="pending", created_at=Det.now(),
        )
        await self._repo.add(card)
        await approval_card_requested.send_async(sender=self, card=card)
        if context.parent_thread_id and self._marker_emitter:
            await self._marker_emitter.emit(
                thread_id=context.parent_thread_id,
                card_id=card.id, prompt=prompt,
            )
```

### 4. New persistence: `ApprovalCard` + `ApprovalCardRepository`

Storage is **separate** from existing HITL helper-thread persistence
(different lifecycle, different query patterns — "list pending by
user" doesn't apply to helper threads). Confirmed in earlier Q&A.

```python
# src/ballast/persistence/approval_card.py
class ApprovalCard(BaseModel):
    id: str                          # == request_id
    workflow_id: str
    respond_topic: str               # f"hitl:{id}"
    kind: str                        # "note.create", "note.delete", ...
    prompt: str
    payload: dict[str, Any]          # JSON dump of the BaseModel
    parent_thread_id: str | None
    user_id: str | None
    status: Literal["pending", "approved", "rejected", "timeout"]
    resolution: dict[str, Any] | None  # HITLResponse.model_dump on decide
    created_at: datetime
    resolved_at: datetime | None


class ApprovalCardRepository(Protocol):
    async def add(self, card: ApprovalCard) -> None: ...
    async def get(self, card_id: str) -> ApprovalCard | None: ...
    async def list_pending(
        self, *, user_id: str | None = None, limit: int = 50,
    ) -> list[ApprovalCard]: ...
    async def resolve(
        self, card_id: str, *, response: HITLResponse,
    ) -> ApprovalCard: ...
```

Implementations: `InMemoryApprovalCardRepository` for tests + dev, plus
a `PostgresApprovalCardRepository` (SQLAlchemy + new Alembic migration
`0002_approval_cards.py`).

### 5. REST router + SSE

```
GET    /approvals?status=pending&limit=50    → list ApprovalCard
GET    /approvals/{id}                       → single card
POST   /approvals/{id}/decision              → {approved: bool, feedback?: str}
GET    /approvals/stream  (SSE)              → live feed of approval_card_*
```

`POST /approvals/{id}/decision` body:

```json
{ "approved": true,  "feedback": null }
{ "approved": false, "feedback": "title is too vague" }
```

Handler logic:

1. Load card by id; 404 if missing, 409 if not pending.
2. Build `HITLResponse` (`ApprovedResponse` or `RejectedResponse`).
3. `await Durable.send_async(destination_id=card.workflow_id,
   message=response.model_dump(), topic=card.respond_topic)`.
4. `await repo.resolve(card.id, response=response)` — persist outcome.
5. Fire `approval_card_decided` signal.
6. Return resolved card.

SSE stream subscribes to both `approval_card_requested` and
`approval_card_decided` signals; emits a typed `card-*` event payload
matching the wire shape already used for thread events.

### 6. Chat marker

When `UICardChannel.deliver` runs with a `parent_thread_id`, emit a
persistent thread event:

```
data-approval-pending {
  card_id: str,
  kind: str,
  prompt: str,
}
```

Assistant-ui renderer for `data-approval-pending` shows an inline
"Waiting for your approval →" pill with a link to
`/approvals#{card_id}` (badge + drawer-open).

When the card resolves, emit `data-approval-resolved` (same payload +
`approved: bool`) so the inline pill flips to "Approved ✓" /
"Rejected ✗".

### 7. Frontend `<ApprovalsPanel/>`

  - Side drawer toggle button in app header with badge count
    (`pending.length`).
  - On mount: `GET /approvals?status=pending` → initial list.
  - Open SSE subscription to `/approvals/stream` → react to
    `card-requested` (prepend) / `card-decided` (remove).
  - Each card renders the payload by `kind`. For MVP only
    `"note.create"` is registered → renders title/body preview +
    Approve / Reject buttons.
  - Card actions call `POST /approvals/{id}/decision` and remove
    optimistically.

### 8. `create_note` integration

```python
@NotesAgent.tool
async def create_note(ctx, title, body):
    draft = ProposedNote(title=title, body=body)
    refined = (
        draft if note_refiner is None
        else await _refine_or_fallback(draft, ctx.deps.parent_thread_id)
    )

    verdict = await ask_human(
        channel=UICardChannel(repo=ctx.deps.approval_repo),
        prompt="Save this note?",
        payload=refined,                              # ProposedNote
        context=HITLContext(
            kind="note.create",
            parent_thread_id=ctx.deps.parent_thread_id,
        ),
    )
    if not isinstance(verdict, ApprovedResponse):
        return "Note save cancelled by user."

    return await notes_repo.create(
        title=refined.title, body=refined.body,
    )
```

`NoteToolDeps` gains `approval_repo: ApprovalCardRepository`.

### 9. `propose_todo` migration (minimal)

Today: `await todo_flow.open(helper_agent=NotesTodoApprovalAgent,
context=TodoApprovalContext(...))` (fire-and-forget via
`DurableHITLWorkflow`).

`DurableHITLWorkflow` stays — it's the fire-and-forget shape, not part
of this refactor. Its internals just delegate to `ThreadChannel.deliver`
instead of duplicating the helper-thread logic. No call-site changes
in `propose_todo`.

## Error handling

  - **`POST /approvals/{id}/decision` on already-resolved card** → 409
    Conflict with current resolution.
  - **`Durable.send_async` failure** (workflow already cancelled, e.g.
    user hit Stop in chat): catch in router; mark card `status="timeout"`
    with explanatory resolution; return 410 Gone to the panel; SSE
    emits `card-decided` so panel removes it.
  - **`ask_human` timeout** (caller-supplied) → returns
    `TimeoutResponse`; card status flips to `"timeout"` via a
    separate background sweep (out of scope this iteration — for now
    timeouts are app-policy; the card just sits as `pending`).
  - **Workflow crash mid-`ask_human`**: DBOS replays from the step
    before `recv_async`; `channel.deliver` is in a `@Durable.step()`
    inside `ask_human` so it's memoised — the card row is added once.
    On replay we re-enter `recv_async` and resume waiting.

## Testing

  - **Framework unit**: `UICardChannel.deliver` writes a card + fires
    the signal (in-memory repo, signal spy).
  - **Framework integration**: `ask_human` end-to-end against
    `UICardChannel` — kick off a workflow, post a decision via the
    router, assert workflow returns `ApprovedResponse`.
  - **Notes-app**: existing `test_create_note_persists_via_repo` adapts
    — a stub channel that auto-approves; assert `notes_repo.create`
    runs and a card row was added.
  - **Notes-app**: rejection path — stub channel auto-rejects; assert
    `notes_repo.create` NOT called; tool returns cancellation string.
  - **API**: router tests for list / decide (404 / 409 / 200).
  - **Migration smoke**: alembic upgrade head + downgrade base on a
    throwaway PG.

## What this design deliberately does NOT do

  - **No app-level default channel.** Tools pass `channel=...`
    explicitly. (Q1 answer.)
  - **No migration of `propose_todo` to cards.** (Q2 answer.)
  - **No new `HITLResponse` variants.** Approved / Rejected /
    Timeout cover the card UX.
  - **No `Modified` from cards in this iteration.** Edit-before-approve
    needs a frontend form + payload-validation contract — deferred
    until requested.
  - **No multi-user approval routing / RBAC.** `user_id` column
    exists; assignment policy is out of scope.
  - **No Slack / webhook channels.** Designed-for but not built —
    Slack channel can be added later by implementing `HITLChannel`
    without touching `ask_human`, the card repo, or the router.

## Files touched (estimate)

**Framework — new:**

  - `src/ballast/patterns/hitl/channels/__init__.py`
  - `src/ballast/patterns/hitl/channels/_protocol.py`         (HITLChannel, HITLContext)
  - `src/ballast/patterns/hitl/channels/thread.py`            (ThreadChannel)
  - `src/ballast/patterns/hitl/channels/ui_card.py`           (UICardChannel + signals)
  - `src/ballast/persistence/approval_card.py`                (model + protocol)
  - `src/ballast/persistence/in_memory/approval_card.py`
  - `src/ballast/persistence/sql/approval_card.py`
  - `src/ballast/persistence/alembic/versions/0002_approval_cards.py`
  - `src/ballast/api/approvals/router.py`                     (REST + SSE)

**Framework — modified:**

  - `src/ballast/patterns/hitl/ask.py`                        (new signature)
  - `src/ballast/patterns/hitl/__init__.py`                   (exports)
  - `src/ballast/patterns/hitl/durable.py`                    (delegate to ThreadChannel)
  - `src/ballast/app.py`                                      (wire approvals router)

**Notes-app:**

  - `examples/notes-app/backend/src/notes_app/agents/notes.py`
    (`create_note` calls `ask_human(channel=UICardChannel(...), …)`)
  - `examples/notes-app/backend/src/notes_app/agents/notes.py`
    (`NoteToolDeps.approval_repo`)
  - `examples/notes-app/backend/src/notes_app/main.py`
    (construct `ApprovalCardRepository`, inject into deps)
  - `examples/notes-app/frontend/src/components/approvals-panel.tsx` (new)
  - `examples/notes-app/frontend/src/components/data-parts/data-approval-pending.tsx` (new)
  - Header wiring + badge count.

## Open follow-ups (not blocking this iteration)

  - Approval inbox per-user filtering (UI selector + repo filter).
  - Card timeout sweep (background workflow that flips stale
    pending → timeout).
  - `Modified` response path for edit-before-approve.
  - SlackMessageChannel + SlackThreadChannel.
