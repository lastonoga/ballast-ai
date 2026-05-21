"use client";

/**
 * Iteration-4 runtime provider — Vercel AI SDK transport.
 *
 * Wires assistant-ui to the notes-app FastAPI backend using:
 *   - `@assistant-ui/react-ai-sdk` (`useAISDKRuntime`) + `@ai-sdk/react`
 *     (`useChat`) over the Vercel AI SDK v6 wire format. Our backend
 *     serializes events via `pydantic_ai.ui.vercel_ai.VercelAIAdapter`,
 *     so `useChat` parses them natively (text-delta, tool-input-*,
 *     tool-approval-request, finish, etc).
 *   - `useRemoteThreadListRuntime` for cross-session thread persistence
 *     backed by `/threads` CRUD endpoints (see `thread-list-adapter.ts`).
 *   - `sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithApprovalResponses`
 *     so that clicking Approve/Cancel auto-rebroadcasts the conversation
 *     with the approval response attached — pydantic-ai resumes the
 *     paused `requires_approval=True` tool call on the next round-trip.
 *
 * URL bridge: Vercel's `DefaultChatTransport` is built for a single
 * `/api/chat` URL. Our backend exposes a per-thread streaming endpoint
 * (`POST /threads/{threadId}/messages`), so we override `api` per
 * request via `prepareSendMessagesRequest` using the current
 * thread-list-item's `remoteId` from `useAuiState`.
 *
 * Approval bridge: the assistant-ui Vercel runtime does NOT proxy
 * `chatHelpers.addToolApprovalResponse` through `addResult`, so the
 * approval card (`<DeleteNoteApproval />`) reads the helpers from
 * `ChatHelpersContext` and calls `addToolApprovalResponse` directly.
 */

import {
  AssistantRuntimeProvider,
  useAui,
  useAuiState,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import {
  useAISDKRuntime,
  AssistantChatTransport,
} from "@assistant-ui/react-ai-sdk";
import { useChat } from "@ai-sdk/react";
import {
  type ChatTransport,
  type UIMessage,
  lastAssistantMessageIsCompleteWithApprovalResponses,
} from "ai";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  type FC,
  type PropsWithChildren,
} from "react";
import { buildRemoteThreadListAdapter } from "@/lib/thread-list-adapter";
import {
  createJustInitializedSink,
  useNotesAppThreadHistoryAdapter,
} from "@/lib/thread-history-adapter";

const DEFAULT_API_URL = "http://localhost:8000";

/**
 * Minimal slice of `useChat`'s helpers the approval card needs.
 * Pulled from `@ai-sdk/react`'s `UseChatHelpers<UIMessage>` so the
 * runtime can hand it to deferred-tool consumers without leaking the
 * full chat shape.
 */
export type ChatApprovalHelpers = Pick<
  ReturnType<typeof useChat<UIMessage>>,
  "addToolApprovalResponse"
>;

const ChatHelpersContext = createContext<ChatApprovalHelpers | null>(null);

/**
 * Read the approval helpers from the nearest `<RuntimeProvider>`. Used
 * by `makeAssistantToolUI` cards that need to call
 * `addToolApprovalResponse({ id, approved })` on the underlying
 * `useChat` instance — assistant-ui's per-tool `addResult` callback only
 * triggers `addToolOutput`, not the approval response path.
 */
export const useChatApprovalHelpers = (): ChatApprovalHelpers => {
  const ctx = useContext(ChatHelpersContext);
  if (!ctx) {
    throw new Error(
      "useChatApprovalHelpers must be used inside <RuntimeProvider>",
    );
  }
  return ctx;
};

/**
 * Per-thread transport that lazily ensures the backend thread exists
 * before sending.
 *
 * Two responsibilities:
 *
 * 1. Rewrites the `api` URL to `POST /threads/{remoteId}/messages` on
 *    every send (Vercel's `DefaultChatTransport` is single-URL).
 * 2. **Lazily** awaits `aui.threadListItem().initialize()` BEFORE the
 *    first send to a draft thread. Without this, the first POST would
 *    race the initialize call and land on a phantom client id → 404.
 *
 * The prior fix (iter 3) eagerly called `initialize()` on mount when
 * `status === "new"`. That bled into every page reload: assistant-ui
 * defaults the active thread to a fresh "new" draft until the user
 * picks one from the sidebar, so each reload POSTed an empty thread to
 * the backend. The lazy approach defers the POST to the exact moment
 * the user actually sends a message.
 */
class ThreadAwareTransport extends AssistantChatTransport<UIMessage> {
  private readonly ensureRemoteId: () => Promise<string>;

  constructor(
    options: ConstructorParameters<typeof AssistantChatTransport<UIMessage>>[0],
    ensureRemoteId: () => Promise<string>,
  ) {
    super(options);
    this.ensureRemoteId = ensureRemoteId;
  }

  override async sendMessages(
    opts: Parameters<AssistantChatTransport<UIMessage>["sendMessages"]>[0],
  ): ReturnType<AssistantChatTransport<UIMessage>["sendMessages"]> {
    await this.ensureRemoteId();
    return super.sendMessages(opts);
  }
}

function buildTransport(
  apiUrl: string,
  headers: Record<string, string>,
  getRemoteId: () => string | undefined,
  ensureRemoteId: () => Promise<string>,
): ChatTransport<UIMessage> {
  return new ThreadAwareTransport(
    {
      // `api` is required by the base type but our `prepareSendMessagesRequest`
      // overrides it on every send. Keep the template visible for error logs.
      api: `${apiUrl}/threads/{threadId}/messages`,
      headers,
      prepareSendMessagesRequest: ({
        body,
        headers: h,
        id,
        messages,
        trigger,
        messageId,
      }) => {
        const remoteId = getRemoteId();
        // sendMessages awaited ensureRemoteId() before calling us; if
        // the ref still isn't populated, something is wrong with the
        // initialize flow — fail loud.
        if (!remoteId) {
          throw new Error(
            "[runtime-provider] missing thread remoteId at send time",
          );
        }
        // When `prepareSendMessagesRequest` returns a `body`, the SDK uses it
        // verbatim (see HttpChatTransport.sendMessages in `ai/dist/index.mjs`)
        // — the default merge of `{id, messages, trigger, messageId, ...body}`
        // is bypassed. Reconstruct the full v6 envelope so the backend's
        // `SubmitMessage | RegenerateMessage` discriminator (`trigger`) is
        // present; otherwise the request body is `{tools: {}}` and pydantic
        // rejects with `union_tag_not_found`.
        return {
          api: `${apiUrl}/threads/${remoteId}/messages`,
          body: {
            ...(body ?? {}),
            id,
            messages,
            trigger,
            messageId,
          },
          headers: h,
        };
      },
    },
    ensureRemoteId,
  );
}

export const RuntimeProvider: FC<PropsWithChildren> = ({ children }) => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL;

  const headers = useMemo<Record<string, string>>(() => ({}), []);

  const adapter = useMemo(
    () => buildRemoteThreadListAdapter(apiUrl, headers),
    [apiUrl, headers],
  );

  // Per-thread chat helpers (publicly: `addToolApprovalResponse`) are
  // produced inside the `runtimeHook` (one chat per thread) but consumed
  // outside it (the approval card hangs off `<ChatHelpersContext>` at the
  // RuntimeProvider level). We bridge via a ref captured by both:
  //
  //   - the per-thread runtime hook publishes the latest helpers via
  //     `useEffect` after every chat-helper identity change
  //   - the outer proxy reads `.current` on each invocation
  //
  // Why not stash on the runtime object: `useRemoteThreadListRuntime`
  // returns an OUTER runtime wrapper, distinct from the per-thread runtime
  // we attach properties to — the wrapper hides the attached property,
  // making "(runtime as any).__chatApprovalHelpers" always undefined.
  const helpersRef = useRef<ChatApprovalHelpers | null>(null);

  // Shared between the per-thread initialize() path and the per-thread
  // history adapter so that the redundant first load() right after
  // initialize() can be skipped — see the JustInitializedSink docstring
  // for why (UI duplicate prevention).
  const justInitialized = useMemo(() => createJustInitializedSink(), []);

  // Stable callback that adds a freshly-created side thread to the
  // sidebar + switches focus to it. Populated once the outer
  // ``useRemoteThreadListRuntime`` runtime exists (see below).
  // PerThreadRuntime calls it on incoming ``thread-created`` events
  // so the helper conversation appears live and becomes the active
  // thread without a page reload.
  const showNewThreadRef = useRef<((threadId: string) => void) | null>(null);

  // assistant-ui's `useRemoteThreadListRuntime` re-invokes `runtimeHook`
  // for each thread. To pass `apiUrl`/`headers` in without closing over
  // stale references, we curry them via a factory.
  const runtimeHook = useMemo(
    () =>
      function PerThreadRuntime() {
        const aui = useAui();
        const remoteId = useAuiState((s) => s.threadListItem.remoteId);
        // Client-side stable id for the thread. Survives the remoteId
        // round-trip (initialize() flips remoteId from undefined → uuid;
        // the client id never changes). We use this as `useChat`'s `id`
        // so the underlying chat instance isn't torn down mid-send when
        // remoteId resolves.
        const clientThreadId = useAuiState((s) => s.threadListItem.id);

        // `prepareSendMessagesRequest` needs the *current* remoteId; capture
        // it through a ref so the transport instance is stable across renders
        // (rebuilding the transport per render confuses `useChat`).
        const remoteIdRef = useRef<string | undefined>(remoteId);
        useEffect(() => {
          remoteIdRef.current = remoteId;
        }, [remoteId]);

        // Lazy initialize: called by the transport just before its first
        // `sendMessages`. Idempotent on the assistant-ui side (the runtime's
        // RemoteThreadListHookInstanceManager caches the in-flight init
        // promise). If `initialize()` returns a remoteId, we also write
        // it to the ref synchronously so `prepareSendMessagesRequest` —
        // which fires immediately after — sees the new value without
        // waiting for the React state-→useEffect-→ref cycle.
        const ensureRemoteIdRef = useRef<() => Promise<string>>(
          async () => {
            throw new Error("[runtime-provider] ensureRemoteId not ready");
          },
        );
        useEffect(() => {
          ensureRemoteIdRef.current = async (): Promise<string> => {
            if (remoteIdRef.current) return remoteIdRef.current;
            const result = (await aui.threadListItem().initialize()) as
              | { remoteId?: string }
              | undefined
              | void;
            const newId = result && "remoteId" in result
              ? result.remoteId
              : undefined;
            if (newId) {
              remoteIdRef.current = newId;
              // Tell the history adapter to skip its next load() for
              // this remoteId — useChat already has the optimistic
              // user message that triggered initialize() in the first
              // place, and re-importing the same row from the backend
              // would render it twice.
              justInitialized.mark(newId);
              return newId;
            }
            if (remoteIdRef.current) return remoteIdRef.current;
            throw new Error(
              "[runtime-provider] initialize() resolved without a remoteId",
            );
          };
        }, [aui]);

        // The transport must be stable across renders — `useChat` keys
        // its internal `AbstractChat` instance off the transport identity.
        // `apiUrl`/`headers` are config-time constants in practice; we
        // depend on them so the linter is happy.
        const transport = useMemo(
          () =>
            buildTransport(
              apiUrl,
              headers,
              () => remoteIdRef.current,
              () => ensureRemoteIdRef.current(),
            ),
          [],
        );

        const chat = useChat<UIMessage>({
          // Stable client-side id — does NOT swap to backend remoteId
          // mid-session, so `useChat`'s internal `AbstractChat` instance
          // survives the initialize() handshake without dropping the
          // in-flight stream that triggered initialize() in the first
          // place.
          id: clientThreadId,
          transport,
          // Auto-resend the conversation as soon as the user supplies all
          // outstanding approval responses — pydantic-ai's
          // `VercelAIAdapter.deferred_tool_results` then picks them up.
          sendAutomaticallyWhen:
            lastAssistantMessageIsCompleteWithApprovalResponses,
        });

        // Bridge the Stop button to the backend's durable workflow:
        //
        // Vercel AI SDK's `chat.stop()` only aborts the client-side
        // fetch — for a non-durable backend that's enough (server sees
        // the dropped connection and bails). Our `StateflowDurableAgent`
        // is the opposite: it survives client disconnects on purpose,
        // so aborting the fetch leaves the workflow running and the
        // user just sees nothing happen.
        //
        // Wrap `chat.stop` once so every code path that triggers it
        // (ComposerPrimitive.Cancel, programmatic cancellation, the
        // assistant-ui runtime's cancelRun, …) first POSTs
        // `/threads/{id}/cancel`. The backend kills every active
        // workflow on that thread + emits a `cancelled` event so the
        // SSE stream closes cleanly. THEN we drop the local fetch.
        useEffect(() => {
          const w = chat as unknown as {
            stop: (...a: unknown[]) => Promise<void>;
            __backendCancelWrapped?: boolean;
          };
          if (w.__backendCancelWrapped) return;
          const original = w.stop.bind(chat);
          w.stop = async (...args: unknown[]): Promise<void> => {
            const tid = remoteIdRef.current;
            if (tid) {
              try {
                await fetch(`${apiUrl}/threads/${tid}/cancel`, {
                  method: "POST",
                  headers,
                });
              } catch (err) {
                // Best-effort: a failed cancel endpoint shouldn't
                // block the local SSE abort. The workflow will
                // eventually finish or time out on its own; meanwhile
                // the user still gets immediate UI feedback.
                // eslint-disable-next-line no-console
                console.warn(
                  "[runtime-provider] backend cancel failed",
                  err,
                );
              }
            }
            return original(...args);
          };
          w.__backendCancelWrapped = true;
        }, [chat]);

        // Cross-workflow notifications via long-lived SSE on the
        // thread's event log. A separate, durable agent run (e.g.
        // ``TodoApprovalFlow.on_decision``) that needs to push a
        // message into this thread emits a ``message-added`` event
        // into the event log; this SSE delivers it live to the active
        // chat — no page reload required. Opens for the active
        // remoteId; closes on switch / unmount.
        // Capture ``chat`` in a ref so the SSE-opening effect doesn't
        // re-fire on every chat identity change (which happens on
        // every render — useChat returns a new object each time).
        // The effect only depends on ``remoteId`` now; the ref gives
        // the handler access to the LATEST chat for setMessages.
        const chatRef = useRef(chat);
        useEffect(() => {
          chatRef.current = chat;
        }, [chat]);

        useEffect(() => {
          if (!remoteId) return;
          const url = `${apiUrl}/threads/${remoteId}/events`;
          // eslint-disable-next-line no-console
          console.debug("[thread-events] opening SSE", url);
          const es = new EventSource(url);
          es.onopen = () => {
            // eslint-disable-next-line no-console
            console.debug("[thread-events] open", remoteId);
          };
          es.onmessage = (ev) => {
            // eslint-disable-next-line no-console
            console.debug("[thread-events] msg", ev.data);
            try {
              const data = JSON.parse(ev.data) as {
                kind?: string;
                payload?: {
                  id: string;
                  role: "user" | "assistant" | "system" | "tool";
                  parts: Array<Record<string, unknown>>;
                };
              };
              if (data.kind === "thread-created") {
                // A workflow on this thread spawned a side thread.
                // Pull just THIS thread into the sidebar via
                // ``runtime.threads.switchToThread`` — the runtime's
                // ``getItemById(newId)``-then-``adapter.fetch`` path
                // appends only the new entry (spread merge over
                // existing ``threadData``). We deliberately avoid a
                // full list reload here: ``getLoadThreadsPromise``
                // re-classifies threads from scratch and overwrites
                // the existing ``threadIdMap[remoteId] →
                // __LOCALID_…`` mappings produced by ``initialize()``
                // earlier in the session, orphaning the locally-keyed
                // hook instance (the user's currently-active thread
                // re-mounts empty when they navigate back).
                const newId = (
                  data as unknown as { payload?: { thread_id?: string } }
                ).payload?.thread_id;
                // eslint-disable-next-line no-console
                console.debug(
                  "[thread-events] thread-created → switchToThread",
                  newId,
                  showNewThreadRef.current ? "have-fn" : "NO-FN",
                );
                if (newId) showNewThreadRef.current?.(newId);
                return;
              }
              if (data.kind !== "message-added" || !data.payload) return;
              const newMsg = data.payload;
              // Append to useChat state — skip if it's already there
              // (an event-log replay after reconnect would otherwise
              // duplicate). Uses chat.setMessages with the functional
              // updater pattern so we don't race with concurrent
              // useChat-driven updates (streaming etc).
              const w = chatRef.current as unknown as {
                setMessages: (
                  updater: (prev: UIMessage[]) => UIMessage[],
                ) => void;
              };
              w.setMessages((prev) => {
                if (prev.some((m) => m.id === newMsg.id)) return prev;
                return [...prev, newMsg as unknown as UIMessage];
              });
            } catch (err) {
              // eslint-disable-next-line no-console
              console.warn(
                "[runtime-provider] thread-events parse failed",
                err,
              );
            }
          };
          es.onerror = (err) => {
            // eslint-disable-next-line no-console
            console.debug("[thread-events] error", err);
            // EventSource auto-reconnects with Last-Event-ID — no
            // manual handling needed. The error fires on transient
            // disconnects which the browser recovers from.
          };
          return () => {
            es.close();
          };
        }, [remoteId]);

        // History adapter MUST be built inside PerThreadRuntime (not in
        // an outer provider): ``useNotesAppThreadHistoryAdapter`` calls
        // ``useAui()`` which is only available once the
        // ``useRemoteThreadListRuntime``/``AssistantRuntimeProvider``
        // chain has set up the AUI store context — and that context IS
        // available inside the per-thread runtimeHook. Passing the
        // adapter directly to ``useAISDKRuntime`` is the canonical
        // ai-sdk/v6 wiring (the cloud variant ``useChatRuntime`` does
        // the same internally via ``useChatThreadRuntime``).
        const historyAdapter = useNotesAppThreadHistoryAdapter(
          apiUrl, headers, justInitialized,
        );
        const runtime = useAISDKRuntime(chat, {
          adapters: { history: historyAdapter },
        });

        // History persistence flows through the canonical
        // ThreadHistoryAdapter wired via <RuntimeAdapterProvider>
        // below — useAISDKRuntime/useExternalHistory triggers
        // ``historyAdapter.withFormat(...).load()`` once
        // ``threadListItem.remoteId`` resolves and imports the
        // messages via ``runtime.thread.import``. No manual
        // ``chat.setMessages`` needed; thread switching auto-fires
        // the load for the newly-active thread.

        // Publish this thread's chat helpers to the outer provider via
        // the shared ref. Clear on unmount so a stale helper from a
        // discarded thread can't fire against the wrong chat.
        useEffect(() => {
          helpersRef.current = {
            addToolApprovalResponse: chat.addToolApprovalResponse,
          };
          return () => {
            helpersRef.current = null;
          };
        }, [chat.addToolApprovalResponse]);

        return runtime;
      },
    [apiUrl, headers],
  );

  // Restore the previously-active thread across reloads. Without this,
  // assistant-ui drops the user onto a fresh draft and the history
  // adapter's load() is never triggered (it short-circuits on missing
  // remoteId on first mount; switching threads later wouldn't re-fire
  // load for the original active one). We persist the active id to
  // localStorage on every switch and use it as ``initialThreadId``.
  const initialThreadId = useMemo<string | undefined>(() => {
    if (typeof window === "undefined") return undefined;
    return window.localStorage.getItem("notes-app:active-thread") ?? undefined;
  }, []);

  const runtime = useRemoteThreadListRuntime({
    adapter,
    runtimeHook,
    initialThreadId,
  });

  // Publish a "show this new thread" callback to the per-thread closure.
  //
  // Why ``switchToThread(newId)`` and NOT ``runtime.threads.reload()``
  // or a manual ``getLoadThreadsPromise()``: assistant-ui's
  // ``classifyThreads`` rebuilds the ``threadIdMap`` from scratch with
  // ``threadIdMap[remoteId] = remoteId``. Any thread that the user
  // created locally in this session (via ``switchToNewThread`` +
  // ``initialize``) has its mapping stored as
  // ``threadIdMap[remoteId] = __LOCALID_XXX`` AND a hook instance keyed
  // by ``__LOCALID_XXX``. A list refresh OVERWRITES the LOCALID
  // mapping with ``remoteId → remoteId``; the next click in the sidebar
  // calls ``startThreadRuntime(remoteId)`` and creates a fresh EMPTY
  // hook instance — the messages-bearing LOCALID instance is orphaned.
  //
  // ``switchToThread`` takes the safe path: when ``getItemById(newId)``
  // returns undefined it calls ``adapter.fetch`` and spread-merges the
  // single new entry into ``threadData`` without touching existing
  // mappings.
  useEffect(() => {
    const threads = (runtime as unknown as {
      threads?: { switchToThread?: (id: string) => Promise<void> | void };
    }).threads;
    showNewThreadRef.current = threads && typeof threads.switchToThread === "function"
      ? (newId: string) => {
          // eslint-disable-next-line no-console
          console.debug("[show-new-thread] switchToThread", newId);
          void Promise.resolve(threads.switchToThread!(newId)).catch((err) => {
            // eslint-disable-next-line no-console
            console.warn("[show-new-thread] switchToThread failed", err);
          });
        }
      : null;
    return () => {
      showNewThreadRef.current = null;
    };
  }, [runtime]);

  // Track the current thread id and persist it. ``runtime.threads`` is
  // the thread-list API; subscribe to learn when the active thread
  // switches.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const persist = () => {
      try {
        const id =
          (runtime as unknown as {
            threads?: { getState: () => { mainThreadId?: string } };
          }).threads?.getState().mainThreadId;
        if (id) {
          window.localStorage.setItem("notes-app:active-thread", id);
        }
      } catch {
        /* best-effort */
      }
    };
    persist();
    const sub = (runtime as unknown as {
      threads?: { subscribe?: (fn: () => void) => () => void };
    }).threads?.subscribe?.(persist);
    return () => {
      if (sub) sub();
    };
  }, [runtime]);

  // Stable proxy over `helpersRef` so consumers can `useContext` once
  // and call through to whichever chat is currently mounted.
  const helpersProxy = useMemo<ChatApprovalHelpers>(
    () => ({
      addToolApprovalResponse: (...args) => {
        const live = helpersRef.current;
        if (!live) {
          throw new Error(
            "[runtime-provider] no per-thread chat helpers — runtime not ready",
          );
        }
        return live.addToolApprovalResponse(...args);
      },
    }),
    [],
  );

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatHelpersContext.Provider value={helpersProxy}>
        {children}
      </ChatHelpersContext.Provider>
    </AssistantRuntimeProvider>
  );
};
