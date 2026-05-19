# Iteration 1 — frontend RETRO

Scope: stand up a runnable chat UI using **only** assistant-ui + shadcn primitives, on top of a mock `ChatModelAdapter`. No backend.

## What assistant-ui gave us for free

Vendored via `npx shadcn@latest add https://r.assistant-ui.com/<name>.json`, so the source lives in `src/components/assistant-ui/` and is fully ownable:

- **`Thread`** — viewport, auto-scroll, scroll-to-bottom button, welcome screen, suggestion pills, message list, "if running" gating.
- **`ThreadList`** — `New thread` button, item list with active highlight, hover "more" menu with **Archive** and **Delete** actions, skeleton loading state, keyboard accessible. Threads are persisted in-memory by `useLocalRuntime`'s built-in thread store.
- **`Composer`** — auto-resizing textarea (`ComposerPrimitive.Input`), Enter-to-send / Shift-Enter newline, **send / stop / cancel** action button that swaps based on run state, drag-and-drop file dropzone (`ComposerPrimitive.AttachmentDropzone`), add-attachment button, attachment chip rendering with remove / preview.
- **`EditComposer`** — inline edit of any user message, with "edit-during-streaming cancels the in-flight run and branches" semantics for free.
- **`ActionBar`** — Copy, Reload, Edit, and a "More" overflow menu on assistant + user messages.
- **`BranchPicker`** — prev / next branch navigation when an edit creates siblings.
- **Markdown renderer** (`@assistant-ui/react-markdown` + `remark-gfm`) wired into `MarkdownText`, with streaming-safe smoothing (`useSmooth`) so partial markdown does not flash.
- **`Reasoning`** — collapsible reasoning/think block rendering.
- **`ToolGroup` + `ToolFallback`** — generic collapsible "tool was called" UI with input args, output, error, and running spinner. Renders **any** tool call without registration; per-tool custom UI is opt-in via `makeAssistantToolUI`.
- **`Attachment`** primitives — preview tile, remove button, server-side upload progress state machine baked in (`PendingAttachment` → `CompleteAttachment`).
- **Suggestions** — both static (`ThreadPrimitive.Suggestions`) and runtime-driven (`thread.suggestions` populated by a `SuggestionAdapter`).
- **`AuiIf`** — declarative conditional rendering against runtime state (e.g. show welcome while empty, show stop button while running).
- All of the above are **keyboard accessible** out of the box (focus rings, Esc to cancel edit, etc).

## Streaming primitives — exact `ChatModelAdapter` contract

This is what iteration 2's backend MUST match (or wrap), regardless of whether we serve it via `useLocalRuntime` directly or front it with another runtime like the AG-UI / Data-Stream adapter:

```ts
type ChatModelAdapter = {
  run(options: ChatModelRunOptions):
    | Promise<ChatModelRunResult>
    | AsyncGenerator<ChatModelRunResult, ChatModelRunResult | void, unknown>;
};

type ChatModelRunOptions = {
  messages: readonly ThreadMessage[];   // full history, normalized
  abortSignal: AbortSignal;             // fired on stop / unmount / new run
  // plus runConfig, context, tools, ... — see @assistant-ui/react types
};

type ChatModelRunResult = {
  content: MessagePart[];               // FULL snapshot, NOT a delta
  metadata?: { timing?: MessageTiming; usage?: {...}; /* ... */ };
  status?: { type: "running" | "complete" | "incomplete"; reason?: string };
};

type MessagePart =
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "tool-call"; toolName: string; toolCallId: string;
      args: unknown; argsText?: string; result?: unknown;
      status?: { type: "running" | "complete" | ... } }
  | { type: "image"; image: string }
  | { type: "file"; data: string; mimeType: string; filename?: string };
```

Streaming semantics observed in this iteration's mock:

1. `run` is an **async generator**. Each `yield` carries the **complete** message-so-far, not a delta. The framework diffs internally to decide what to re-render and what to feed `useSmooth`.
2. The last value (`yield` or `return`) is treated as the terminal snapshot. Returning nothing after the last `yield` is fine.
3. Aborting `abortSignal` is the only way to stop a run. The composer's stop button just calls `controller.abort()`.
4. Per-field tool-arg streaming (`useToolArgsStatus`) works automatically if you yield successive `tool-call` parts whose `args` JSON grows incrementally — the framework derives `propStatus: "streaming" | "complete"` per top-level key.
5. Timing metrics (`useMessageTiming`, TTFT, tokens/sec) are computed for `useLocalRuntime` **only if** the adapter passes `metadata.timing` in the final `ChatModelRunResult`. Without it, only chunk count + duration are derivable.

## Approval / tool-call UX

- **Tool-call rendering: built-in.** `ToolGroup` + `ToolFallback` render args, result, error, and a `running` spinner for any tool the model calls. We do not need custom UI to ship iteration 3.
- **Approval cards: NOT built-in as a primitive.** assistant-ui's docs show two routes:
  1. `makeAssistantToolUI({ toolName, render: ({ args, addResult, resume, interrupt }) => ... })` — the render fn receives an `interrupt?: { type: "human"; payload: unknown }` from `context.human()` server-side, and an `addResult` / `resume` callback to send the user's decision back. This is the official "human-in-the-loop tool" pattern.
  2. Render a custom approval card inside that `render` function (buttons → `addResult(...)` / `resume(...)`).
- There is no shadcn-registry "approval card" component. **Action for iteration 4**: ship a small `<HitlApprovalCard />` in `src/components/assistant-ui/` that wraps `makeAssistantToolUI` and renders shadcn buttons; have the framework's `HITLGate` emit a tool-call with a stable name (e.g. `__hitl_approval`) so the card binds to it.

## What was awkward / not in scope

- **Next.js 16 + RSC**: the generated `ThreadList` passes inline arrow functions (`<AuiIf condition={(s) => ...}>`) to client components, which trips RSC's serialization check. Forced us to mark `app/page.tsx` `"use client"`. Acceptable for a chat-first app, but worth documenting.
- **`useLocalRuntime` thread store is in-memory only** — refreshing the browser wipes threads. The docs say for cross-session persistence you graduate to `useRemoteThreadListRuntime` (with a `RemoteThreadListAdapter`) **or** `AssistantCloud`. We can stay on `useLocalRuntime` through iteration 3 and adopt the remote thread adapter when we wire the real backend; iteration 2 should design its persistence schema with that adapter's contract in mind (`list`, `rename`, `archive`, `delete`, `initialize`, `generateTitle`).
- **Shadcn registry + Tailwind v4**: works, but the registry quietly rewrote `globals.css` with `@import "shadcn/tailwind.css"` and `@custom-variant dark`. Anyone editing the CSS by hand must know to preserve those.
- **No "configure once" wiring for both the runtime AND the thread list adapter** — `useLocalRuntime(adapter)` covers messages; thread list comes from the same hook's internal default. If iteration 2 wants server-side threads, the call site moves to `useRemoteThreadListRuntime({ adapter, runtimeHook: () => useLocalRuntime(...) })` — non-obvious nesting.
- **No built-in "system message" UI**. If the framework wants to surface tenant / agent identity in the chat, we need to plumb it through `runConfig` or render it ourselves (e.g. in the sidebar header — which is what we already did).

## Framework-side TODOs for iteration 2

The backend MUST satisfy these to plug into assistant-ui without custom client glue:

1. **Stream format**: emit a stream the chosen runtime adapter accepts. Three reasonable backend choices, ranked by least → most framework lock-in:
   - **Data-Stream** (`@assistant-ui/react-data-stream`) — newline-delimited typed events (`text-delta`, `tool-input-start`, `tool-input-delta`, `tool-output-available`, `finish`). This is assistant-ui's native protocol.
   - **AG-UI** (`@assistant-ui/react-ag-ui`) — already standardized in the framework spec; client uses `useAgUiThreadRuntime({ runtimeUrl })`.
   - **Bespoke** — implement a `ChatModelAdapter` whose `run` is an `async function*` that consumes our own JSON/SSE stream and yields full-snapshot `ChatModelRunResult`s. This is what our mock does today.
2. **Full snapshots, not deltas**, on every yield (or, if using Data-Stream, append-only deltas keyed by `id`).
3. **Tool calls** must surface as `{ type: "tool-call", toolName, toolCallId, args, result }` parts with `status.type` transitioning `running` → `complete`/`incomplete`. The toolCallId must be stable across stream chunks.
4. **HITL interrupt shape**: when our `HITLGate` blocks, the backend must surface it as a tool-call part with `interrupt: { type: "human", payload: <our approval-card payload> }` and accept the user's decision via the resume endpoint that the runtime adapter calls (`addResult` / `resume` semantics).
5. **Abort propagation**: HTTP request must honor client `AbortSignal` — disconnect cancels the run, which our framework's `Engine` must translate to a DBOS step cancellation.
6. **Thread CRUD endpoints** for iteration 3+: `list`, `create`, `rename`, `archive` (soft delete), `delete`, `getState` (returns full message history). Tenant-scoped from day one (header: `x-tenant-id`).
7. **Title generation**: optional `POST /threads/{id}/title` returning an `assistant-stream` (one `appendText` call). Keeps it within the same wire format.
8. **Streaming timing metadata** in the terminal event (`metadata.timing.{streamStartTime, firstTokenTime, totalStreamTime, tokenCount, tokensPerSecond, totalChunks, toolCallCount}`) so the `useMessageTiming` badges populate.
9. **Suggestion adapter**: if we want the welcome-screen and follow-up suggestion pills powered by the agent, expose a `POST /threads/{id}/suggestions` returning `[{ prompt: string }]`. Maps cleanly to `LocalRuntimeOptions.adapters.suggestion`.

## Commit

`feat(notes-app/frontend): iteration 1 — assistant-ui shell on shadcn (mock runtime)`

# Iteration 3 — frontend wiring

Scope: swap the mock runtime for the real backend — canonical AG-UI streaming for messages, tenant-scoped `/threads` CRUD for persistence.

## Packages installed

- `@assistant-ui/react-ag-ui` (0.0.30) — wraps an `@ag-ui/client` agent in an assistant-ui runtime. Hook is `useAgUiRuntime({ agent })` (singular — the doc-snippet name `useAgUiThreadRuntime` is stale; the package only ships `useAgUiRuntime`).
- `@ag-ui/client` (0.0.53) — provides `HttpAgent` (the POST/SSE transport) and re-exports `@ag-ui/core` types (`RunAgentInput`, event schemas).
- `assistant-stream` (0.3.14) — for `createAssistantStream` used by `generateTitle` (empty-stream no-op).
- `@assistant-ui/core` (0.2.2) — added as a direct dep so we can import `RemoteThreadListAdapter`'s sibling types (`RemoteThreadInitializeResponse`, `RemoteThreadListResponse`, `RemoteThreadMetadata`, `RemoteThreadListPageOptions`); `@assistant-ui/react` re-exports only the adapter type itself.

## What assistant-ui's AG-UI adapter accepted unchanged

Group A landed canonical AG-UI camelCase events (`runStarted`, `textMessageStart/Content/End`, `toolCallStart/Args/End`, `runFinished`, `runError`, `messageId`, `threadId`, `runId`, `toolCallId`, `delta`). `@ag-ui/client`'s SSE parser and event schemas (from `@ag-ui/core`) use exactly this shape — see `dist/index.d.ts` types `RunStartedEvent`, `TextMessageContentEvent`, etc. **Zero translation layer needed on the wire format.**

## What required a bridge

One mismatch: `HttpAgent` POSTs to ONE fixed URL, but our streaming endpoint embeds the thread id in the path (`POST /threads/{id}/messages?protocol=ag-ui`). Two-line bridge — subclass `HttpAgent`, override `run(input)`, set `this.url` from `input.threadId` before delegating to `super.run(input)`. See `src/components/runtime-provider.tsx::NotesAppAgent`. Logged as framework TODO #1 below — the backend could equivalently grow a thread-agnostic `POST /agent?protocol=ag-ui` that reads `threadId` from the JSON body, which is what the AG-UI reference deployment assumes.

## `RemoteThreadListAdapter` mapping

| Adapter method | Backend call | Notes |
| --- | --- | --- |
| `list(opts)` | `GET /threads?include_archived=…` (×2 in parallel for regular + archived) | Server may return a bare array or `{threads:[…]}` — adapter accepts both. |
| `initialize()` | `POST /threads` body `{purpose, purpose_metadata, actor_id}` | Returns `{remoteId: t.id, externalId: undefined}`. |
| `rename(id, title)` | `PATCH /threads/{id}` body `{title}` | |
| `archive(id)` / `unarchive(id)` | `POST /threads/{id}/archive` (or `/unarchive`) | 404 treated as success (already in target state). |
| `delete(id)` | `DELETE /threads/{id}` | 404 treated as success. |
| `fetch(id)` | `GET /threads/{id}` | Returns the single `RemoteThreadMetadata`. |
| `generateTitle(id, messages)` | (none) | **Decision: return `createAssistantStream(c => c.close())`** — an empty, immediately-closed stream. Iteration 3 has no agent-driven title endpoint, so the runtime's auto-title step is a no-op; users rename manually via the thread-list "more" menu (which goes through `rename` → `PATCH`). |

`status` mapping: backend's `status === "archived"` OR a non-null `archived_at` → `"archived"`; otherwise `"regular"`.

## `generateTitle` decision

Picked the empty-stream path over the "set title from first user message client-side" alternative. Reasoning: (a) one less place that needs to know about title heuristics, (b) once the backend ships `POST /threads/{id}/title` per framework TODO #7 (iter-1 RETRO), the swap is one function body — no client-side fallback to delete. The user-visible cost is that new threads start untitled until the user renames them.

## Smoke verification

- `pnpm build` clean.
- `pnpm dev` + `curl http://localhost:3000/` with **no backend running** → HTTP 200 (page mounts; runtime errors propagate to assistant-ui's standard error UI on the first network call, not a white screen).

## Framework TODOs for the next round

1. **Backend should accept thread id in the request body**, not just the path. Either expose a thread-agnostic `POST /agent?protocol=ag-ui` (matches the AG-UI reference deployment so `HttpAgent` works with a single fixed `url:` and zero subclassing), OR document the per-thread URL pattern as a first-class part of our AG-UI flavor. The current path-based shape forces every AG-UI client to subclass `HttpAgent`.
2. **`POST /threads/{id}/title`** that streams an `AssistantStream` (one `appendText` call) — fulfills iter-1 framework TODO #7, lets `generateTitle` do real work.
3. **Thread list pagination**: backend ignores our `offset` query param today; the adapter passes `opts?.after` through as `offset` in anticipation. Either honor it server-side or document that the list endpoint is unpaginated.
4. **Suggestion adapter wiring** (iter-1 TODO #9 — `POST /threads/{id}/suggestions`) — still deferred; the welcome screen currently shows static `ThreadPrimitive.Suggestions` only.
5. **HITL `AgUiInterrupt` shape**: the AG-UI runtime surfaces `unstable_getPendingInterrupts` / `unstable_submitInterruptResponses` (see `AgUiAssistantRuntime`). When `HITLGate` lands, the encoder must emit interrupts matching `AgUiInterrupt` (`{ id, reason, message?, toolCallId?, responseSchema?, expiresAt?, metadata? }`) and accept resume responses keyed by `interruptId`.
6. **Thread-list `unstable_Provider`** on the adapter: currently unused, but the same hook is how `useCloudThreadListAdapter` injects per-thread history / attachments adapters. When we want server-loaded message history on switch (rather than re-streaming), wire `GET /threads/{id}/messages` through here.

## Commit

`feat(notes-app/frontend): iteration 3 — real AG-UI runtime + RemoteThreadListAdapter`

