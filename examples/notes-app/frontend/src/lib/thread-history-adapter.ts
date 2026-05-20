/**
 * ``ThreadHistoryAdapter`` wired against the notes-app FastAPI backend.
 *
 * Loads persisted messages on thread mount so ``useAISDKRuntime``
 * imports them via ``runtime.thread.import``. ``append`` is a
 * deliberate no-op: the server's streaming router already persists
 * every turn via ``thread_repo.add_message`` (user pre-stream +
 * assistant on_complete), so a client-side append would duplicate.
 *
 * Modeled after assistant-ui's ``AssistantCloudThreadHistoryAdapter``:
 * the adapter captures ``aui`` once and reads the active thread's
 * ``remoteId`` dynamically inside ``load``/``append``. The adapter
 * instance is stable across renders (``useState`` initializer) so
 * ``useExternalHistory``'s ``loadedRef`` guard works correctly.
 */
import { useAui } from "@assistant-ui/react";
import type {
  ThreadHistoryAdapter,
  GenericThreadHistoryAdapter,
  MessageFormatAdapter,
  MessageFormatRepository,
  MessageFormatItem,
} from "@assistant-ui/react";
import { useState } from "react";

type BackendMessageRow = {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  parent_id: string | null;
  parts: Array<Record<string, unknown>>;
};

class NotesAppThreadHistoryAdapter implements ThreadHistoryAdapter {
  constructor(
    private readonly apiUrl: string,
    private readonly headers: Record<string, string>,
    private readonly aui: ReturnType<typeof useAui>,
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
        // {type:"text", text:..., state:"done"}). Hand them through as
        // TMessage — useAISDKRuntime's format adapter will encode/decode
        // through its own internals when needed.
        return {
          messages: rows.map((row) => ({
            parentId: row.parent_id,
            message: {
              id: row.id,
              role: row.role,
              parts: row.parts,
            } as unknown as TMessage,
          })),
        };
      },

      async append(_item: MessageFormatItem<TMessage>): Promise<void> {
        // Server-side streaming router already persisted this turn.
        // useExternalHistory still fires append for every new message
        // it observes; we accept it and do nothing to avoid duplicates.
      },
    };
  }

  // ThreadHistoryAdapter's non-format methods are required by the type
  // even though useAISDKRuntime only goes through `withFormat`. Stub
  // them so other runtimes that hit the raw API don't break.
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
): ThreadHistoryAdapter {
  const aui = useAui();
  const [adapter] = useState(
    () => new NotesAppThreadHistoryAdapter(apiUrl, headers, aui),
  );
  return adapter;
}
