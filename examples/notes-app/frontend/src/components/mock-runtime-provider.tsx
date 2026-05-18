"use client";

import type { FC, PropsWithChildren } from "react";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  type ChatModelAdapter,
} from "@assistant-ui/react";

/**
 * Iteration-1 mock ChatModelAdapter.
 *
 * Streams a deterministic echo response character-by-character so the
 * default assistant-ui streaming UI animations (smooth text, status pills,
 * stop button) are exercised end-to-end with no backend.
 *
 * Contract reminder for iteration-2 backend implementers:
 *   `run` may be an `async function*` and yield ANY number of
 *   `ChatModelRunResult` snapshots. Each snapshot is the FULL message
 *   content up to that point (NOT a delta). Yielding
 *     { content: [{ type: "text", text: "Hello" }] }
 *   then
 *     { content: [{ type: "text", text: "Hello world" }] }
 *   produces an incremental stream. assistant-ui diffs internally.
 */
const mockEchoAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const last = messages[messages.length - 1];
    const userText =
      last?.content
        ?.map((p) => (p.type === "text" ? p.text : ""))
        .join("") ?? "";

    const reply =
      `(mock runtime — iteration 1) you said: ${userText.trim() || "(nothing)"}`;

    let acc = "";
    for (const ch of reply) {
      if (abortSignal.aborted) break;
      acc += ch;
      yield { content: [{ type: "text", text: acc }] };
      await new Promise((r) => setTimeout(r, 15));
    }
  },
};

export const MockRuntimeProvider: FC<PropsWithChildren> = ({ children }) => {
  const runtime = useLocalRuntime(mockEchoAdapter);
  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
};
