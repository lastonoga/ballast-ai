# How to build a Helper Agent for clarification

**Problem:** Your main agent reaches a decision point where it needs human input that's NOT a binary approval ("yes" / "no") — it needs nuanced guidance ("which of these 3 options?", "what timezone are you in?", "should I include drafts?"). A simple approval card doesn't fit; you need a brief conversation.

**Solution:** `HelperAgent` + `ConversationalChannel`. Main agent opens a side-thread, hands it off to a helper agent that chats with the user for as long as needed, then receives back a typed `HelperVerdict[ContextT]` with the user's structured response.

## Minimum

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from ballast.patterns.hitl import HelperAgent, HelperVerdict


class TripPreferences(BaseModel):
    """The typed verdict the main agent gets back."""
    destination: str
    duration_days: int
    budget_usd: int
    style: str   # "luxury" | "backpacker" | "family"


# Define a helper agent that fills in TripPreferences via chat
helper = HelperAgent[TripPreferences](
    name="trip_planner_helper",
    model="openai:gpt-4o-mini",
    system_prompt=(
        "You're helping a user plan a trip. Ask short clarifying questions "
        "until you have: destination, duration in days, budget, style. "
        "Then call the `finalize` tool with the collected info."
    ),
    verdict_type=TripPreferences,
)
```

The framework auto-registers a `finalize(verdict: TripPreferences)` tool on the helper agent — when called, the helper session ends and the verdict is returned to the calling code.

## Invoke from main agent / workflow

```python
from ballast.patterns.hitl import HITLGate

gate = HITLGate(channel=conversational_channel)

@notes_agent.tool
async def plan_trip(ctx) -> str:
    """Ask the user to plan a trip via a helper agent."""
    verdict: TripPreferences = await gate.ask_helper(
        helper_agent=helper,
        context={"existing_notes_count": await notes_repo.count_for_user(...)},
    )
    
    # Now we have typed user preferences:
    return f"Planning {verdict.duration_days}-day trip to {verdict.destination} for ${verdict.budget_usd}."
```

The main agent suspends durably while the helper-agent chats with the user in a side-thread. When the user provides enough info, the helper calls `finalize(...)` — the verdict bubbles back to the main agent's `plan_trip` tool.

## Wire the conversational channel

```python
from ballast.patterns.hitl import ConversationalChannel

conversational_channel = ConversationalChannel(
    helper_agents={"trip_planner_helper": helper},
)
```

`ConversationalChannel` knows how to dispatch helper-thread messages to the right HelperAgent. The notes-app already wires this for its TodoApprovalAgent flow.

## Frontend side

`assistant-ui` shows helper threads as a side-panel chat. The user sees:
1. Main thread: "Let me ask you a few questions about the trip..."
2. Helper panel opens: helper agent asks first question
3. User answers; helper asks next question
4. User answers; helper calls finalize → panel closes
5. Main thread: "Planning 7-day trip to Tokyo for $3000..."

All durable: if the page reloads mid-conversation, state resumes correctly.

## Pass context to the helper

The helper agent should know what main-agent is asking about:

```python
verdict = await gate.ask_helper(
    helper_agent=helper,
    context={                                # arbitrary dict, the helper sees this
        "user_existing_trips": ["Paris 2023", "Tokyo 2024"],
        "currency_preference": "USD",
        "concierge_available": False,
    },
)
```

Inside the helper, `context` is bound to `RunContext.deps` for the helper's tools / prompt template. The helper system prompt can reference it.

## Helper tools beyond `finalize`

You can give the helper agent its own tools to look things up before finalizing:

```python
helper = HelperAgent[TripPreferences](
    name="trip_planner_helper",
    model=...,
    system_prompt="...",
    verdict_type=TripPreferences,
)

@helper.tool
async def check_currency_rates(ctx, base: str = "USD") -> dict:
    """Look up current currency rates so we can suggest realistic budgets."""
    return await currency_api.rates(base=base)
```

Helper uses tools naturally in its conversation. Only `finalize` ends the session.

## Cancel a helper session

If the user wants to cancel:

```
POST /threads/{helper_thread_id}/cancel
```

Returns the user to the main thread; main agent receives `HITLDenied` (sub-class of `BallastError`). Main agent should catch and react gracefully.

## Multi-helper composition

A complex flow can use multiple helpers in sequence:

```python
@workflow_agent.tool
async def plan_business_trip(ctx) -> str:
    trip_prefs = await gate.ask_helper(helper_agent=trip_planner_helper, context={})
    
    if trip_prefs.budget_usd > 10_000:
        # Expensive trip — open expense approval helper
        approval = await gate.ask_helper(
            helper_agent=expense_approver_helper,
            context={"trip_preferences": trip_prefs.model_dump()},
        )
        if not approval.approved:
            return "Trip cancelled: budget rejected"
    
    return await book_trip(trip_prefs)
```

Each helper is a separate side-thread; main thread sees only the final result of the chain.

## Caveats

- **Helper agents are normal `Agent`s with extra wiring.** The `finalize` tool is auto-injected; otherwise they behave like any pydantic-ai agent. Use sparingly — they cost LLM tokens per user message.
- **Don't put long-running tools in the helper.** Users are watching the panel; tool that takes 30 seconds = bad UX. Move heavy lifting to the main agent post-finalize.
- **`HelperVerdict[T]` must be a pydantic BaseModel.** No raw dicts or primitives — the framework needs a schema to validate the LLM's `finalize` call.
- **State is durable BUT context dict is captured at handoff time.** If the underlying state changes during the helper session (e.g. user adds notes), the helper won't see it unless you give it a tool to refetch.

## When NOT to use this

- **Simple yes/no approvals** → use `UICardChannel` + `ApprovalCard` (much cheaper, no helper LLM needed)
- **Tool-call gating** → use `ApprovalCapability` ([require-approval-for-dangerous-tools.md](require-approval-for-dangerous-tools.md))
- **Async / batch operations** → don't block the workflow on human input; emit a thread event + let the user respond when ready

## Related

- [require-approval-for-dangerous-tools.md](require-approval-for-dangerous-tools.md) — for binary approval / tool gating
- [add-approval-card-flow.md](add-approval-card-flow.md) — manual `UICardChannel.request` for structured approvals (non-conversational)
- Reference: `reference/hitl/helper-agent.md`
- Reference: `reference/hitl/conversational-channel.md`
- Explanation: [article-pain-points.md](../../explanation/article-pain-points.md) #22
