# Known bugs / upstream gaps

Issues encountered while dogfooding the framework against real LLM traffic.
Each entry: **symptom**, **root cause**, **what we did**, **upstream fix candidate**.

---

## Iter 4 round 1 — HITL `delete_note` approval

The headline finding: **pydantic-ai's AG-UI integration does not surface
`requires_approval=True` tool calls to the wire**. Five separate gaps, each
necessary to make the AG-UI path work, none of which we owned. We swapped
the entire UI ecosystem (AG-UI → Vercel AI SDK v6) instead of patching
upstream — Vercel's adapter handles approvals out of the box.

### B1. `AGUIAdapter.deferred_tool_results` returns `None`

- **Where**: `pydantic_ai.ui.ag_ui.AGUIAdapter`
- **Symptom**: even when the incoming AG-UI body carries an approval
  decision, the agent never sees a `DeferredToolResults` on the next
  `run_stream` call. The paused tool body never re-fires.
- **Root cause**: the AG-UI adapter does not override the base
  `deferred_tool_results` extractor — it returns `None` unconditionally
  whereas `VercelAIAdapter` reads the SDK v6 `approval-responded` part.
- **Workaround**: switched to `VercelAIAdapter(sdk_version=6)` which
  ships a working extractor.
- **Upstream**: needs an AG-UI extractor that maps `ToolMessage` /
  approval-decision parts into `ToolApproved` / `ToolDenied`.

### B2. `AGUIEventStream.after_stream` emits no approval / interrupt event

- **Where**: `pydantic_ai.ui.ag_ui.AGUIEventStream`
- **Symptom**: when the agent yields `DeferredToolRequests`, the AG-UI
  stream finishes cleanly with no signal that a tool needs approval.
  Frontend has nothing to bind a card to.
- **Root cause**: the event-stream hook for deferred-tool requests is
  not implemented; `RunFinishedEvent` is emitted with no `outcome`.
- **Workaround**: the Vercel adapter emits a discrete
  `tool-approval-request {approvalId, toolCallId}` chunk natively.
- **Upstream**: AG-UI event stream should emit `tool_call_start` + a
  canonical "needs approval" event when a `DeferredToolRequest` is in
  the output.

### B3. `ag_ui.core.RunFinishedEvent` has no `outcome` field

- **Where**: `ag_ui` package (current pin)
- **Symptom**: even if (B2) were fixed, the canonical AG-UI
  `RunFinishedEvent` only exposes `result` — there is no `outcome:
  {type: "interrupt", ...}` surface.
- **Consequence**: assistant-ui's `@assistant-ui/react-ag-ui` runtime
  reads `event.outcome?.type === "interrupt"` to render approvals
  (see `useAgUiRuntime` source). That branch is **unreachable** from a
  pydantic-ai backend regardless of what the adapter does.
- **Upstream**: either `ag_ui.core` needs `outcome` on
  `RunFinishedEvent`, OR assistant-ui's AG-UI runtime needs to be
  re-pointed at a different signal.

### B4. assistant-ui `makeAssistantToolUI` + AG-UI: `addResult` is the wrong path

- **Where**: `@assistant-ui/react-ag-ui`
- **Symptom**: when our approval card called
  `addResult({approved: true})`, the backend received it as a generic
  `ToolMessage` / `ToolReturnPart` — pydantic-ai parsed it as a tool
  return, not as an approval decision. Deferred tool body never
  re-executed.
- **Root cause**: there is no "approval response" routing in the AG-UI
  runtime; `addResult` is the only escape hatch and it maps to tool
  output, not approval.
- **Workaround (Vercel side)**: the Vercel runtime exposes
  `chat.addToolApprovalResponse({id, approved, reason?})` which the
  Vercel adapter then extracts into `DeferredToolResults`.

### B5. `useChatRuntime` forces `useCloudThreadListAdapter`

- **Where**: `@assistant-ui/react`
- **Symptom**: `useChatRuntime` (the "obvious" entry point for the
  Vercel AI SDK path) implicitly wires
  `useCloudThreadListAdapter`, which conflicts with our own
  `RemoteThreadListAdapter` (we own thread lifecycle on the
  framework's `/threads` endpoints, not assistant-ui Cloud).
- **Workaround**: use `useChat` from `@ai-sdk/react` directly +
  `useAISDKRuntime` from `@assistant-ui/react-ai-sdk` and feed that
  into our `useRemoteThreadListRuntime` ourselves.
- **Upstream**: `useChatRuntime` should accept `cloud: false` (or a
  custom thread-list adapter) without dragging in Cloud.

### B6. assistant-ui Vercel runtime proxies `addResult` to `addToolOutput`

- **Where**: `@assistant-ui/react-ai-sdk`
- **Symptom**: the natural-looking
  `addResult({approved: true})` on a tool-UI for a
  `requires_approval=True` tool *also* goes the wrong way here — it
  hits `chatHelpers.addToolOutput`, not the approval response path.
- **Workaround**: in the `makeAssistantToolUI` card, reach into
  `ChatHelpersContext` directly and call
  `chat.addToolApprovalResponse({id, approved, reason?})`. The Vercel
  adapter's `deferred_tool_results` extractor reads those.
- **Upstream**: `addResult` on a `requires-action`-state tool should
  route to the approval response path, not the output path.

### B7. Pre-existing vendored shadcn lint warnings

- **Where**: `examples/notes-app/frontend/src/components/assistant-ui/attachment.tsx`,
  `examples/notes-app/frontend/src/components/theme-toggle.tsx`
- **Symptom**: two `setState`-in-effect lint warnings emitted by
  Next 16's checker on files vendored from the assistant-ui shadcn
  registry.
- **Workaround**: untouched (vendored — fix belongs upstream in the
  registry source).

### B8. `prepareSendMessagesRequest` returning a `body` bypasses default merge

- **Where**: `ai` (Vercel AI SDK v6) `HttpChatTransport.sendMessages`
- **Symptom**: backend rejected requests with
  `union_tag_not_found` on `trigger` discriminator — incoming body was
  `{"tools": {}}` with no `trigger`/`id`/`messages`.
- **Root cause**: when `prepareSendMessagesRequest` returns a `body`
  field, the SDK uses it verbatim. The default
  `{id, messages, trigger, messageId, ...resolvedBody, ...options.body}`
  merge is only applied when `body` is absent from the prepared
  request (`ai/dist/index.mjs` line ~12958).
- **Workaround**: explicitly re-include `id`, `messages`, `trigger`,
  `messageId` from the callback's options when constructing the
  returned body (see `runtime-provider.tsx`).
- **Upstream/doc**: the SDK docs make this look like an override layer
  that augments the default body, when it actually replaces it. A
  callout in `PrepareSendMessagesRequest`'s JSDoc would have saved an
  hour.

### B9. Qwen on Alibaba endpoint rejects `content: null` on tool-call follow-up

- **Where**: OpenRouter → Alibaba upstream for `qwen/qwen3.6-plus`
- **Symptom**: after a successful tool-call round-trip (e.g. user
  approves `delete_note`, tool runs, agent loops back to the LLM to
  produce a final reply), the second LLM call 400s with:
  `<400> InternalError.Algo.InvalidParameter: The content field is a
  required field.`
- **Root cause**: pydantic-ai (per OpenAI spec) sends the assistant
  turn that contained the tool call as
  `{role: "assistant", content: null, tool_calls: [...]}`. The OpenAI
  schema allows `content: null` when `tool_calls` is present; Alibaba's
  Qwen endpoint requires `content` as a string. Other upstreams for
  the same model accept it.
- **Workaround**: route around the Alibaba upstream via OpenRouter
  provider routing. We expose this as an env var on the example:
  `OPENROUTER_PROVIDER_IGNORE=alibaba` (or `OPENROUTER_PROVIDER_ONLY=...`
  for a positive allowlist). `build_model_settings()` plumbs it into
  `openrouter_provider`.
- **Upstream**: Alibaba endpoint compliance with OpenAI spec, or a
  pydantic-ai shim that fills `content: ""` for tool-call-only
  assistant turns when the model is known to be Qwen-on-Alibaba.

### B10. Live browser smoke not driven for iter 4 round 1

- **Symptom**: no browser-controller available in the session that
  shipped iter 4. The full HITL round-trip was verified via curl
  against the live OpenRouter-backed backend (create → delete
  request → approval pause → approve/cancel → final reply), not by
  clicking the UI.
- **Implication**: assume the rendered approval card may have visual
  polish bugs that didn't surface in CI / curl. Re-verify in a real
  browser before declaring the iter shipped.

---

## How to add to this file

When a dogfood iteration hits an upstream gap, append a section with the
five-field shape above (where, symptom, root cause, workaround, upstream).
Don't move fixed bugs to RETRO.md — strike them through in place so the
historical "why we have this glue" trail stays readable.
