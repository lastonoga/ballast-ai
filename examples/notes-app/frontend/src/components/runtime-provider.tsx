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

const DEFAULT_API_URL = "http://localhost:8000";
const DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001";

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
 * Build a per-thread transport that rewrites `api` to the backend's
 * `POST /threads/{remoteId}/messages` URL on every send. We extend
 * `AssistantChatTransport` (not `DefaultChatTransport`) so the
 * assistant-ui-specific extras (model context, callSettings, tools)
 * still get folded into the body upstream.
 */
function buildTransport(
  apiUrl: string,
  headers: Record<string, string>,
  getRemoteId: () => string | undefined,
): ChatTransport<UIMessage> {
  return new AssistantChatTransport<UIMessage>({
    // `api` is required by the base type but our `prepareSendMessagesRequest`
    // overrides it on every send. Keep the template visible for error logs.
    api: `${apiUrl}/threads/{threadId}/messages`,
    headers,
    prepareSendMessagesRequest: ({ body, headers: h }) => {
      const remoteId = getRemoteId();
      // The runtime guarantees a remoteId exists before sending (the
      // per-thread `initialize()` resolves it). Bail loud if it doesn't.
      if (!remoteId) {
        throw new Error(
          "[runtime-provider] missing thread remoteId at send time",
        );
      }
      return {
        api: `${apiUrl}/threads/${remoteId}/messages`,
        body: body ?? {},
        headers: h,
      };
    },
  });
}

export const RuntimeProvider: FC<PropsWithChildren> = ({ children }) => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL;
  const tenantId = process.env.NEXT_PUBLIC_TENANT_ID ?? DEFAULT_TENANT_ID;

  const headers = useMemo(
    () => ({ "X-Tenant-Id": tenantId }),
    [tenantId],
  );

  const adapter = useMemo(
    () => buildRemoteThreadListAdapter(apiUrl, headers),
    [apiUrl, headers],
  );

  // assistant-ui's `useRemoteThreadListRuntime` re-invokes `runtimeHook`
  // for each thread. To pass `apiUrl`/`headers` in without closing over
  // stale references, we curry them via a factory.
  const runtimeHook = useMemo(
    () =>
      function PerThreadRuntime() {
        const aui = useAui();
        const remoteId = useAuiState((s) => s.threadListItem.remoteId);
        const threadListItemState = useAuiState((s) => s.threadListItem);

        // `prepareSendMessagesRequest` needs the *current* remoteId; capture
        // it through a ref so the transport instance is stable across renders
        // (rebuilding the transport per render confuses `useChat`).
        const remoteIdRef = useRef<string | undefined>(remoteId);
        useEffect(() => {
          remoteIdRef.current = remoteId;
        }, [remoteId]);

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
            ),
          [],
        );

        const chat = useChat<UIMessage>({
          id: remoteId,
          transport,
          // Auto-resend the conversation as soon as the user supplies all
          // outstanding approval responses — pydantic-ai's
          // `VercelAIAdapter.deferred_tool_results` then picks them up.
          sendAutomaticallyWhen:
            lastAssistantMessageIsCompleteWithApprovalResponses,
        });

        const runtime = useAISDKRuntime(chat);

        // Eagerly fire `initialize()` for brand-new threads. Same fix as
        // iter 3: `RemoteThreadListHookInstanceManager`'s built-in
        // `unstable_on("initialize")` only fires once `messages.length > 0`,
        // so a brand-new thread's first POST would land on a phantom client
        // id the backend has never seen (→ 404). `initialize()` is
        // idempotent via the manager's internal promise cache.
        useEffect(() => {
          if (threadListItemState.status === "new") {
            void aui.threadListItem().initialize();
          }
        }, [aui, threadListItemState.status]);

        const approvalHelpers = useMemo<ChatApprovalHelpers>(
          () => ({ addToolApprovalResponse: chat.addToolApprovalResponse }),
          [chat.addToolApprovalResponse],
        );

        // Stash helpers on the runtime so the outer provider can publish
        // them via context. The outer provider mounts a fresh
        // `<ChatHelpersContext.Provider>` per thread by re-reading via
        // `useAuiState` from the runtime side, but that's circular — so
        // instead we attach them as a property on the runtime object and
        // read them in the outer wrapper.
        (runtime as unknown as {
          __chatApprovalHelpers: ChatApprovalHelpers;
        }).__chatApprovalHelpers = approvalHelpers;

        return runtime;
      },
    [apiUrl, headers],
  );

  const runtime = useRemoteThreadListRuntime({
    adapter,
    runtimeHook,
  });

  // Pull the per-thread chat helpers off the runtime. They're mutated
  // when threads switch (the runtimeHook reattaches them), so we read
  // them via a getter wrapped in a stable proxy.
  const helpersProxy = useMemo<ChatApprovalHelpers>(
    () => ({
      addToolApprovalResponse: (...args) => {
        const live = (runtime as unknown as {
          __chatApprovalHelpers?: ChatApprovalHelpers;
        }).__chatApprovalHelpers;
        if (!live) {
          throw new Error(
            "[runtime-provider] no per-thread chat helpers — runtime not ready",
          );
        }
        return live.addToolApprovalResponse(...args);
      },
    }),
    [runtime],
  );

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ChatHelpersContext.Provider value={helpersProxy}>
        {children}
      </ChatHelpersContext.Provider>
    </AssistantRuntimeProvider>
  );
};
