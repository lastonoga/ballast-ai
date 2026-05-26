# 21. Human-in-the-loop

**Prerequisites:** [07-capabilities.md](07-capabilities.md), [09-persistence.md](09-persistence.md).

## Introduction

Some agent actions shouldn't be fully autonomous. Sending an email, charging a card, publishing content, modifying production infrastructure — these are decisions where the *agent* proposes and a *human* commits. If the agent is wrong about a tool argument and there's no human checkpoint, the wrong email goes out, the wrong card gets charged, the wrong infra gets changed.

The framework's HITL surface is built on a single principle: *escalate exceptions, not routine actions*. Asking a human to approve every tool call gives you a tedious workflow that humans will rubber-stamp. Asking a human to approve the high-risk minority gives you actual safety. The Protocols and channels are designed for the second pattern.

This chapter covers the `HITLChannel` Protocol and its three built-in implementations (UI cards, in-chat markers, side conversations), the `ApprovalCard` + `CardVerdict` data model, how `ApprovalCapability` bridges pydantic-ai's `requires_approval=True` to UI cards automatically, the durable wait mechanism that lets your workflow sleep until a human responds, and how to write your own channel (Slack, email, Telegram).

## The mental model

HITL = "workflow pauses on a signal until a human resolves it." The framework's contract:

1. The agent (or workflow) calls `await channel.request(payload, timeout=...)`.
2. The channel delivers the request through whatever surface (UI card, chat message, Slack).
3. The workflow suspends durably via DBOS `recv_async`. A crash doesn't lose the in-flight request.
4. The human acts; the channel publishes a verdict; the workflow resumes with the verdict.

The durability is what makes this safe for long waits. A workflow waiting on a human can pause for hours or days; if the server restarts in the middle, the workflow resumes from the wait point and continues to wait. The verdict eventually arrives and execution continues.

## The `HITLChannel` Protocol

```python
class HITLChannel(Protocol, Generic[InT, VerdictT]):
    async def request(
        self,
        payload: InT,
        *,
        timeout: timedelta | None = None,
    ) -> VerdictT: ...
```

One method. Takes a payload (whatever you want the human to see); returns a verdict (whatever shape your verdict has). Typing is per-channel.

The base class `DBOSHITLChannel` implements the `request` method generically — it generates an ID, delivers the request via the abstract `deliver` method, suspends via `Durable.recv_async`, and decodes the verdict via the abstract `decode_verdict` method. To write a channel, you implement `deliver` and `decode_verdict`; the suspend/resume logic is free.

## `UICardChannel` — the dashboard approval surface

The shipped channel for "user opens an approvals dashboard and clicks Approve / Reject."

```python
from ballast.patterns.hitl import UICardChannel
from pydantic import BaseModel

class PublishApprovalPayload(BaseModel):
    __hitl_kind__ = "publish_approval"   # registry key
    draft: str
    topic: str

@register_card_kind
class PublishApprovalPayload(BaseModel):
    ...

approval_channel = UICardChannel(payload_type=PublishApprovalPayload)

# Inside your workflow:
verdict = await approval_channel.request(
    PublishApprovalPayload(draft=draft, topic=topic),
    timeout=timedelta(minutes=30),
)

if verdict.decision == "approve":
    final = verdict.modified or draft   # allow human edits
    await publish(final)
else:
    log_rejected(verdict.feedback)
```

What this does end-to-end:

1. `await approval_channel.request(payload, timeout)` is called.
2. The channel persists an `ApprovalCard` to the `approval_cards` table (filtered by `current_user_id`).
3. A signal fires; subscribers to `/approvals/stream` see the new card via SSE.
4. The workflow suspends. The user-facing HTTP call returns immediately (or yields the card in the response, depending on your route).
5. The human opens the dashboard, sees the card, clicks Approve.
6. The frontend POSTs to `/approvals/{card_id}/decide` with the verdict.
7. The verdict is persisted; the workflow resumes with the typed verdict.

The `__hitl_kind__` attribute is what tells the frontend which renderer to use for the card. The frontend registry maps `"publish_approval"` → `PublishApprovalRenderer` (a React component).

## `CardVerdict` — the standardized verdict

```python
class CardVerdict(BaseModel, Generic[OutT]):
    decision: Literal["approve", "reject"]
    modified: OutT | None = None    # human edits to the payload
    feedback: str | None = None     # free-text comment
    answered_at: datetime
```

Three pieces of information: the decision, optional modifications (if the human edited the payload before approving), and free-text feedback. The `Generic[OutT]` parameter is the payload type so `verdict.modified` is typed.

## `ThreadChannel` — in-chat approval

When the agent is in the middle of a chat conversation and you want to ask the human a question without leaving the chat, use `ThreadChannel`:

```python
from ballast.patterns.hitl import ThreadChannel
from app.helper_agents import TodoApprovalAgent

todo_channel = ThreadChannel(
    helper_agent=TodoApprovalAgent,
    payload_type=TodoApprovalPayload,
    opening_message="I'd like to add the following todos. Approve, edit, or reject?",
)

verdict = await todo_channel.request(
    TodoApprovalPayload(items=[...]),
    timeout=timedelta(minutes=5),
)
```

The channel spawns a *helper sub-thread* with `TodoApprovalAgent`, which has a typed `metadata_model = TodoApprovalPayload`. The helper agent renders the request in chat; the human responds in chat; the helper agent extracts the verdict and posts it back. The main thread sees a marker that an approval was requested + the verdict when done.

This is the right surface for conversational flows where the human is *already in the chat* and you don't want to push them to a separate dashboard.

## `ConversationalChannel` — multi-turn clarification

`HelperAgent` lets you have a structured *conversation* with the human to gather clarification before continuing:

```python
helper = HelperAgent(
    name="trip_clarifier",
    metadata_model=TripClarificationPayload,
    system_prompt="Ask the user any clarifying questions about their trip plan...",
)

verdict = await helper_session_runner.run(
    helper_agent=helper,
    context=TripClarificationPayload(initial=initial_plan),
    timeout=timedelta(minutes=10),
)
# verdict.context now has the clarified data
```

The verdict shape is `HelperVerdict[ContextT]` — the human's final answer is the resolved context. Useful for flows where one approval card isn't enough and you need back-and-forth.

## `ApprovalCapability` — auto-bridging `requires_approval=True`

pydantic-ai has a native `@agent.tool(requires_approval=True)` decorator. When the model calls such a tool, pydantic-ai returns a `DeferredToolRequests` instead of executing it. By default, your code has to inspect this, collect verdicts somehow, and re-run the agent with `deferred_tool_results=...`. That's a lot of glue.

`ApprovalCapability` does it automatically:

```python
from ballast.patterns.hitl import ApprovalCapability

tool_card_map = {
    "send_email": (
        SendEmailPayload,
        lambda args: SendEmailPayload(to=args.to, subject=args.subject, body=args.body),
        email_approval_channel,
    ),
    "charge_card": (
        ChargeCardPayload,
        lambda args: ChargeCardPayload(amount=args.amount, customer_id=args.customer_id),
        charge_approval_channel,
    ),
}

agent = Agent(
    model="openai:gpt-4o",
    tools=[send_email, charge_card],
    capabilities=[ApprovalCapability(tool_card_map=tool_card_map)],
)

# Now agent.run(...) handles approvals end-to-end:
#  - If the model calls a requires_approval tool, the capability opens an approval card
#  - Waits for the human verdict
#  - If approved, re-runs the agent with the verdict so the tool fires
#  - If rejected, re-runs with a "denied" result so the model can respond
result = await agent.run(query)
```

The capability uses `wrap_run` (the most powerful hook, chapter 7) because it needs to potentially re-run the agent multiple times — once per approval round if the model calls multiple `requires_approval=True` tools.

## Durable wait via `Durable.recv_async`

Under the hood, every channel's `request` method does:

```python
topic = f"hitl:{request_id}"
await self.deliver(request_id=..., respond_topic=topic, payload=...)
raw = await Durable.recv_async(topic=topic, timeout_seconds=...)
return await self.decode_verdict(raw)
```

`Durable.recv_async` is a DBOS primitive: the workflow suspends, the suspension is durably recorded, and when a message arrives on the topic (sent via `Durable.send_async` from the decide endpoint), the workflow resumes with the message. A crash between suspend and resume → on replay, the workflow re-suspends and picks up where it was.

This is what makes "wait for a human for an hour" safe even with restarts every 10 minutes. The wait is durable; the workflow doesn't care about server lifecycle.

## Approval cards survive restarts

`UICardChannel` persists the `ApprovalCard` to the `approval_cards` table on `deliver`. The card is in the DB *before* the workflow suspends. So:

- Card persists → workflow suspends → server restarts → workflow replays, re-suspends → card still in DB → human approves → workflow resumes.

The card isn't tied to the in-memory state of any process. It's a row.

## Custom channels — Slack, email, Telegram

`DBOSHITLChannel` is the base. Implement `deliver` (push the request somewhere) and `decode_verdict` (parse the response back into your typed verdict):

```python
class SlackApprovalChannel(DBOSHITLChannel[InT, MyVerdict]):
    def __init__(self, slack_client, channel_id):
        self._slack = slack_client
        self._channel_id = channel_id

    async def deliver(self, *, request_id, workflow_id, respond_topic, payload):
        # Post a message to Slack with Approve / Reject buttons
        msg = await self._slack.chat_postMessage(
            channel=self._channel_id,
            blocks=build_approval_blocks(payload, request_id, respond_topic),
        )
        # When the user clicks, your Slack webhook handler calls
        # Durable.send_async(workflow_id, raw_verdict, topic=respond_topic)

    async def decode_verdict(self, raw: Any) -> MyVerdict:
        return MyVerdict(**raw)
```

Same shape for email (deliver via SMTP, decode from a reply parsed via webhook), Telegram (deliver via bot API, decode from an inline keyboard callback), etc. The `respond_topic` is what your webhook handler uses to wake the right workflow.

## The Accountability Gateway pattern

For high-stakes flows, combine HITL with output-level grading (chapter 23):

```python
@Durable.workflow
async def publish_with_accountability(draft, user_id):
    # 1. Auto-grade
    grade = await quality_judge.grade(draft)
    if grade.confidence == "high" and grade.passed:
        # Confident enough — auto-publish
        return await publish(draft)

    # 2. Otherwise, escalate to human
    verdict = await approval_channel.request(
        PublishApprovalPayload(draft=draft, grade=grade),
        timeout=timedelta(hours=1),
    )

    if verdict.decision == "approve":
        return await publish(verdict.modified or draft)
    else:
        return {"status": "blocked", "reason": verdict.feedback}
```

90% of cases auto-publish; the 10% the judge isn't confident about go to a human. Total human load is small; high-risk cases get human attention.

## Common mistakes

- **Asking humans about every action.** Banks teach their fraud-detection ML to escalate only the uncertain transactions. Apply the same logic — auto-handle the obvious cases, escalate the rest.
- **No timeout on `request`.** A workflow that waits forever for a human is a resource leak. Always set a `timeout` and handle the `HITLTimeoutError` (treat as "no decision," default to safest action).
- **Forgetting to `register_card_kind`** for `UICardChannel` payload types. The frontend renderer registry uses `__hitl_kind__` as the lookup key; if it's not registered, the card renders as raw JSON.
- **Putting non-serializable objects in payloads.** The payload is persisted to the DB. Use pydantic models with serializable fields — no callables, no open file handles.
- **Custom channel that doesn't go through `DBOSHITLChannel`.** You'll lose the durable-wait property. Always extend `DBOSHITLChannel` and only implement `deliver` + `decode_verdict`.

## What this chapter did NOT cover

- The exact frontend code for the assistant-ui approvals panel — see the notes-app demo.
- DBOS messaging internals — chapter 24.
- Multi-tenant scoping for cards — covered by `current_user_id` ContextVar (chapter 9).
- Card retention / cleanup policies — apps own this.

## Where to go next

→ [22-observability.md](22-observability.md) — knowing what's happening in production.
