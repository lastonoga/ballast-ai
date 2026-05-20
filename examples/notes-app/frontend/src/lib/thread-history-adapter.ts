/**
 * ``ThreadHistoryAdapter`` — read-only hydration for the notes-app backend.
 *
 * Backend persistence is a **flat linear message list** managed entirely
 * by ``POST /threads/{id}/messages`` (the single streaming endpoint
 * that persists the user message AND runs the agent in one shot via
 * body-vs-DB sync). The frontend doesn't need to write through this
 * adapter — useChat sends the full message array to the streaming
 * endpoint and the backend reconciles.
 *
 *   - ``load`` — GET /threads/{id}/messages, hands the rows back to
 *     ``useAISDKRuntime`` via ``runtime.thread.import`` for thread
 *     switch / page reload hydration. Skipped immediately after
 *     ``initialize()`` flips a draft thread to a real remoteId
 *     mid-session (useChat already holds the optimistic state).
 *   - ``append`` — no-op. The streaming endpoint owns persistence.
 */
import { useAui } from "@assistant-ui/react";
import type {
  ThreadHistoryAdapter,
  GenericThreadHistoryAdapter,
  MessageFormatAdapter,
  MessageFormatRepository,
} from "@assistant-ui/react";
import { useState } from "react";


type BackendMessageRow = {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  parts: Array<Record<string, unknown>>;
};

/**
 * Shared signal between the runtime-provider's ``initialize()`` path
 * and ``historyAdapter.load()`` — suppresses the redundant first
 * ``load()`` that ``useExternalHistory`` fires the instant
 * ``initialize()`` flips ``remoteId`` from undefined → real-UUID
 * mid-session. Otherwise the just-sent user message would render
 * twice (optimistic state + re-imported from backend).
 */
export type JustInitializedSink = {
  /** Mark a remoteId as freshly produced by initialize() this session. */
  mark(remoteId: string): void;
  /** Consume + return whether ``load()`` should skip for this id. */
  consume(remoteId: string): boolean;
};

export function createJustInitializedSink(): JustInitializedSink {
  const ids = new Set<string>();
  return {
    mark(remoteId) {
      ids.add(remoteId);
    },
    consume(remoteId) {
      if (!ids.has(remoteId)) return false;
      ids.delete(remoteId);
      return true;
    },
  };
}

class NotesAppThreadHistoryAdapter implements ThreadHistoryAdapter {
  constructor(
    private readonly apiUrl: string,
    private readonly headers: Record<string, string>,
    private readonly aui: ReturnType<typeof useAui>,
    private readonly justInitialized: JustInitializedSink,
  ) {}

  private get _remoteId(): string | undefined {
    return this.aui.threadListItem().getState().remoteId;
  }

  withFormat<TMessage, TStorageFormat extends Record<string, unknown>>(
    _fmt: MessageFormatAdapter<TMessage, TStorageFormat>,
  ): GenericThreadHistoryAdapter<TMessage> {
    const adapter = this;
    return {
      async load(): Promise<MessageFormatRepository<TMessage>> {
        const remoteId = adapter._remoteId;
        if (!remoteId) return { messages: [] };
        if (adapter.justInitialized.consume(remoteId)) {
          return { messages: [] };
        }
        const r = await fetch(
          `${adapter.apiUrl}/threads/${remoteId}/messages`,
          { headers: adapter.headers },
        );
        if (!r.ok) {
          throw new Error(
            `GET /threads/${remoteId}/messages failed: ${r.status}`,
          );
        }
        const rows = (await r.json()) as BackendMessageRow[];
        // Backend persists parts in Vercel UIMessage shape already
        // (assistant via VercelAIAdapter.dump_messages; user as
        // {type:"text", text:..., state:"done"}). Linear order, no
        // parent_id — useAISDKRuntime's format adapter consumes the
        // flat list directly.
        return {
          messages: rows.map((row) => ({
            parentId: null,
            message: {
              id: row.id,
              role: row.role,
              parts: row.parts,
            } as unknown as TMessage,
          })),
        };
      },

      async append(): Promise<void> {
        // No-op: persistence flows through POST /threads/{id}/messages
        // (the streaming endpoint). That route receives the full
        // useChat message array on every send and diffs it against
        // the DB (truncate-then-append on the divergent suffix), so
        // edits and regenerates collapse to one round-trip.
      },
    };
  }

  // ThreadHistoryAdapter's non-format methods are required by the type
  // even though useAISDKRuntime only goes through `withFormat`.
  async load() {
    return { headId: null, messages: [] };
  }
  async append() {
    /* no-op */
  }
}

/**
 * Hook that captures ``aui`` once and returns a stable adapter
 * instance — matches ``useAssistantCloudThreadHistoryAdapter``'s
 * shape so it can be dropped into ``RuntimeAdapterProvider``.
 */
export function useNotesAppThreadHistoryAdapter(
  apiUrl: string,
  headers: Record<string, string>,
  justInitialized: JustInitializedSink,
): ThreadHistoryAdapter {
  const aui = useAui();
  const [adapter] = useState(
    () => new NotesAppThreadHistoryAdapter(apiUrl, headers, aui, justInitialized),
  );
  return adapter;
}
