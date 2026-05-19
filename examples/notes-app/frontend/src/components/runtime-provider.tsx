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

import { useMemo, type FC, type PropsWithChildren } from "react";
import {
  AssistantRuntimeProvider,
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
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [],
      );
      return useAgUiRuntime({ agent });
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
};
