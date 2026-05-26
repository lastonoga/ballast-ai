# 21. Human-in-the-loop

**Prerequisites:** [07-capabilities.md](07-capabilities.md), [09-persistence.md](09-persistence.md).

**What you'll learn:** the principle of exception escalation (not routine review); the `HITLChannel` Protocol and its built-in implementations; how `ApprovalCapability` automatically bridges pydantic-ai's `requires_approval=True` to UI cards; how `HelperAgent` enables structured side-conversations; how verdicts persist durably via DBOS.

## Sections

1. The HITL principle: escalate exceptions, don't review every action
2. `HITLChannel` Protocol: one method, many implementations
3. `UICardChannel` — REST + SSE approval panel
4. `ThreadChannel` — in-chat approval marker
5. `ConversationalChannel` — side-conversations via HelperAgent
6. `ApprovalCard` + `CardVerdict[T]` + `register_card_kind` + `__hitl_kind__`
7. The durable wait: `Durable.recv_async` for crash-safe HITL
8. `ApprovalCapability`: auto-bridge `@tool(requires_approval=True)` → cards
9. `HelperAgent` for structured clarifications (typed `HelperVerdict[T]`)
10. The Accountability Gateway pattern for high-stakes flows
11. Custom channels: Slack, email, Telegram
12. Where to go next

## Next

[22-observability.md](22-observability.md) — knowing what's happening in production.
