"use client";

/**
 * Iteration-3 runtime provider.
 *
 * Wires assistant-ui to the notes-app FastAPI backend using:
 *   - `@assistant-ui/react-ag-ui` for the streaming runtime (canonical AG-UI
 *     events: run_started / text_message_* / tool_call_* / run_finished /
 *     run_error in camelCase, exactly what our backend emits).
 *   - `useRemoteThreadListRuntime` for cross-session thread persistence backed
 *     by `/threads` CRUD endpoints (see `thread-list-adapter.ts`).
 *
 * Bridge: AG-UI's `HttpAgent` posts to a single fixed URL, but our backend
 * exposes a per-thread streaming endpoint. `ThreadAwareHttpAgent` (in
 * `src/lib/thread-aware-http-agent.ts`) encapsulates the per-run URL rewrite.
 */

import { useEffect, useMemo, type FC, type PropsWithChildren } from "react";
import {
  AssistantRuntimeProvider,
  useAui,
  useAuiState,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { buildRemoteThreadListAdapter } from "@/lib/thread-list-adapter";
import { ThreadAwareHttpAgent } from "@/lib/thread-aware-http-agent";

const DEFAULT_API_URL = "http://localhost:8000";
const DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001";

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

  const runtime = useRemoteThreadListRuntime({
    adapter,
    runtimeHook: function RuntimeHook() {
      // `useAgUiRuntime` lives inside the per-thread provider, so the agent
      // is rebuilt for each thread switch — cheap, and gives the underlying
      // HttpAgent its own AbortController per thread.
      const agent = useMemo(
        () => new ThreadAwareHttpAgent({ apiUrl, headers }),
        // Recreated when api/tenant config changes (effectively never at
        // runtime, but keeps the hook honest).
        [apiUrl, headers],
      );

      // Bridge: the canonical pattern from assistant-ui's own
      // `useAssistantTransportRuntime` (see
      // node_modules/@assistant-ui/react/dist/legacy-runtime/runtime-cores/
      // assistant-transport/useAssistantTransportRuntime.js:73) is to read
      // the per-thread `remoteId` from the thread-list-item store and feed
      // it back to the underlying transport. `useAgUiRuntime` doesn't do
      // this on its own, so we wire it explicitly.
      //
      // `useAui`/`useAuiState` are public exports of `@assistant-ui/react`
      // (re-exported from `@assistant-ui/store`); see
      // node_modules/@assistant-ui/react/dist/index.d.ts line 2.
      const aui = useAui();
      const remoteId = useAuiState((s) => s.threadListItem.remoteId);

      // `ThreadAwareHttpAgent.run()` rewrites `this.url` from
      // `input.threadId`, which AG-UI's `AgUiThreadRuntimeCore` populates
      // from `adapters.threadList.threadId`. Keep both in sync with the
      // resolved per-thread `remoteId`.
      useEffect(() => {
        agent.threadId = remoteId ?? "";
      }, [agent, remoteId]);

      // Eagerly fire `initialize()` for brand-new threads. Background:
      // `RemoteThreadListHookInstanceManager` (core/dist/.../
      // RemoteThreadListHookInstanceManager.js) listens for
      // `runtime.threads.main.unstable_on("initialize")` to auto-call
      // `aui.threadListItem().initialize()` (which POSTs /threads).
      // That event is only emitted from `ensureInitialized()` paths —
      // and `external-store-thread-runtime-core.ts:237` gates it on
      // `messages.length > 0`. So an empty new thread NEVER triggers it,
      // and the first `POST /threads/{id}/messages` lands on a phantom
      // client-generated id the backend has never seen (→ 404).
      //
      // We sidestep the broken event path by initializing as soon as the
      // per-thread provider mounts in `status === "new"`. Subsequent
      // renders are no-ops (status flips to `regular` once `remoteId`
      // resolves). `initialize()` is idempotent via the internal
      // `initPromiseRef` in the hook-instance manager.
      const threadListItemState = useAuiState((s) => s.threadListItem);
      useEffect(() => {
        if (threadListItemState.status === "new") {
          aui.threadListItem().initialize();
        }
      }, [aui, threadListItemState.status]);

      return useAgUiRuntime({
        agent,
        // Tell AG-UI's `AgUiThreadRuntimeCore` which thread it's running
        // against so it propagates `threadId` into `RunAgentInput`.
        adapters: {
          threadList: {
            threadId: remoteId ?? "",
          },
        },
      });
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
};
