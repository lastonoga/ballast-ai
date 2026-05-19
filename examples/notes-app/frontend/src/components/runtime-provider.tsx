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
 * exposes a per-thread streaming endpoint (`POST /threads/{id}/messages
 * ?protocol=ag-ui`). We subclass `HttpAgent` and override `requestInit` so
 * the URL is rebuilt from `input.threadId` on every run.
 */

import { useMemo, type FC, type PropsWithChildren } from "react";
import {
  AssistantRuntimeProvider,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { HttpAgent, type RunAgentInput } from "@ag-ui/client";
import { buildRemoteThreadListAdapter } from "@/lib/thread-list-adapter";

const DEFAULT_API_URL = "http://localhost:8000";
const DEFAULT_TENANT_ID = "00000000-0000-0000-0000-000000000001";

/**
 * HttpAgent variant whose POST URL is computed per-run from
 * `input.threadId`. Required because our streaming endpoint embeds the
 * thread id in the path.
 */
class NotesAppAgent extends HttpAgent {
  private readonly apiBase: string;

  constructor(apiBase: string, headers: Record<string, string>) {
    // `url` on the base class is unused; we override requestInit and
    // recompute the URL there. Pass a sentinel so logging stays sane.
    super({ url: `${apiBase}/threads/{id}/messages?protocol=ag-ui`, headers });
    this.apiBase = apiBase;
  }

  protected override requestInit(input: RunAgentInput): RequestInit {
    // Call the parent to pick up the JSON body + headers + AbortSignal, then
    // we'll re-issue with the correct per-thread URL via run() below.
    return super.requestInit(input);
  }

  // The base class reads `this.url` inside its run() to choose the endpoint.
  // Rather than reimplement run() against the rxjs/SSE plumbing, we mutate
  // `this.url` synchronously before super.run() consumes it.
  override run(input: RunAgentInput) {
    this.url = `${this.apiBase}/threads/${input.threadId}/messages?protocol=ag-ui`;
    return super.run(input);
  }
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

  const runtime = useRemoteThreadListRuntime({
    adapter,
    runtimeHook: function RuntimeHook() {
      // `useAgUiRuntime` lives inside the per-thread provider, so the agent
      // is rebuilt for each thread switch — cheap, and gives the underlying
      // HttpAgent its own AbortController per thread.
      const agent = useMemo(
        () => new NotesAppAgent(apiUrl, headers),
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
