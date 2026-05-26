# How to auto-bridge `requires_approval=True` tools to UI cards

**Problem:** You have tools that affect the world (publish a post, send an email, transfer funds). pydantic-ai supports `requires_approval=True` on these tools — but when triggered, it just returns `DeferredToolRequests`. You'd have to wire the human-approval flow yourself: collect verdicts, build `DeferredToolResults`, call `agent.run()` again to resume.

**Solution:** `ApprovalCapability` automates the loop. Tools marked `requires_approval=True` automatically open `UICardChannel` approval cards; the capability awaits the human verdict (durably, via DBOS); maps the verdict to `ToolApproved` or `ToolDenied`; re-runs the agent to continue.

## End-to-end example

### 1. Declare the tool with `requires_approval=True`

```python
from pydantic_ai import Agent

agent = Agent(model="openai:gpt-4o")

@agent.tool(requires_approval=True)
async def publish_post(title: str, body: str) -> str:
    """Publish a post. NEVER runs without human approval."""
    return await blog_client.publish(title=title, body=body)
```

### 2. Define the HITL card payload

This is what the human sees in the UI:

```python
from pydantic import BaseModel
from typing import ClassVar
from ballast.patterns.hitl import register_card_kind, CardVerdict

@register_card_kind
class PublishCard(BaseModel):
    __hitl_kind__: ClassVar[str] = "publish-post"
    title: str
    body: str


class PublishVerdict(CardVerdict[PublishCard]):
    __hitl_kind__: ClassVar[str] = "publish-post"
```

### 3. Wire `ApprovalCapability`

```python
from ballast import ApprovalCapability, UICardChannel

ui_channel = UICardChannel(payload_type=PublishCard)

def build_publish_card(tc, deps) -> PublishCard:
    args = tc.args if isinstance(tc.args, dict) else {}
    return PublishCard(title=args.get("title", ""), body=args.get("body", ""))

approval = ApprovalCapability(
    tool_card_map={
        "publish_post": (PublishCard, build_publish_card, ui_channel),
    },
)

agent = Agent(
    model=...,
    capabilities=[approval],
)
@agent.tool(requires_approval=True)
async def publish_post(title: str, body: str) -> str: ...
```

### 4. Run

```python
result = await agent.run("Publish a post about today's weather")
# Internally:
# 1. LLM emits tool call publish_post(title="...", body="...")
# 2. pydantic-ai returns DeferredToolRequests
# 3. ApprovalCapability.wrap_run sees it, opens a PublishCard
# 4. The card appears in the user's /approvals UI panel
# 5. Workflow suspends durably via Durable.recv_async
# 6. Human clicks approve / modify / reject in the UI
# 7. POST /approvals/{id}/decision wakes up the workflow
# 8. ApprovalCapability maps verdict → ToolApproved/ToolDenied
# 9. agent.run() resumes; publish_post body runs (or is denied)
# 10. Final output returned
```

## Verdict mapping

| `CardVerdict.decision` | `CardVerdict.modified` | → maps to |
|---|---|---|
| `"approve"` | `None` | `ToolApproved(override_args=None)` — runs with original args |
| `"approve"` | `PublishCard(...)` | `ToolApproved(override_args={"title": ..., "body": ...})` — runs with edited args |
| `"reject"` | (any) | `ToolDenied(message=verdict.feedback or "Denied by user.")` |

This gives users 3 actions in the UI: **approve as-is**, **edit + approve**, **reject**.

## Multiple tools, shared channel

```python
approval = ApprovalCapability(
    tool_card_map={
        "publish_post":   (PublishCard,   build_publish_card,   ui_channel),
        "send_email":     (EmailCard,     build_email_card,     ui_channel),
        "delete_resource":(DeleteCard,    build_delete_card,    ui_channel_red),  # different channel for red ops
    },
)
```

Different tools can have different channels, payloads, factories. The capability matches by `tool_name`.

## Timeout

```python
approval = ApprovalCapability(
    tool_card_map={...},
    timeout=timedelta(hours=2),
)
```

If no verdict arrives within 2 hours, the underlying `Durable.recv_async` raises `TimeoutError`. Wrap your agent runner with a try/except + treat timeout as denial.

## Unmapped tools

If `requires_approval=True` is set on a tool you forgot to add to `tool_card_map`, the capability auto-denies with a helpful error message — the agent gets `ToolDenied(message="Tool 'X' is not registered in tool_card_map; auto-denying. Either add it to ApprovalCapability.tool_card_map or remove requires_approval=True from the tool.")`. Better than silent hang.

## Notes-app example

The reference app uses this for `propose_note` flow — see `examples/notes-app/backend/src/notes_app/agents/notes.py` for a real integration with `assistant-ui` panel.

## When not to use this

- For **routine HITL** (clarification questions, multi-choice prompts): use `HelperAgent` + `ConversationalChannel` instead — those open a side-conversation rather than a card.
- For **non-tool approvals** (e.g. "approve this draft email I'm about to send via webhook"): wire `UICardChannel.request(...)` manually inside your workflow — `ApprovalCapability` is specifically for tool-call gating.

## Caveats

- The capability calls `agent.run(..., deferred_tool_results=...)` to resume. This means the agent state (message history, deps) carries through. Ensure your `deps` is picklable across the resume boundary if you're running inside a DBOS workflow.
- If the resumed agent emits ANOTHER round of deferred approvals (e.g. the original tool led to a chain that needed another approval), the capability's `while` loop handles it automatically.
- `CardVerdict.modified` must be a `BaseModel` instance (Pydantic dump-able). String-only feedback should go in `CardVerdict.feedback`.

## Related

- [add-approval-card-flow.md](add-approval-card-flow.md) — manual `UICardChannel.request(...)` flow (non-tool case)
- [customize-hitl-channel.md](customize-hitl-channel.md) — Slack/email/Telegram channel implementations
- Reference: `reference/capabilities/approval-capability.md`
- Reference: `reference/hitl/ui-card-channel.md`
- Explanation: [article-pain-points.md](../../explanation/article-pain-points.md) #21
