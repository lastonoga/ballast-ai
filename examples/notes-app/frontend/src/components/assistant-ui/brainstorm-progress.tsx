"use client";

/**
 * Custom assistant-ui data-part renderer for ``data-brainstorm-progress``.
 *
 * Backend emits these via ``BRAINSTORM_PROGRESS.stream(broadcaster, parent_id)``
 * — same ``message_id`` reused across updates, so the UI sees ONE row
 * that mutates through the phases (diverge → converge → hitl) rather
 * than N separate messages piling up.
 *
 * Wire: backend writes a message part of shape
 *   ``{type: "data-brainstorm-progress", data: {...BrainstormProgress}, state: "done"}``
 *
 * useChat parses the Vercel AI SDK v6 chunk; assistant-ui's
 * ``convertMessage`` strips the ``data-`` prefix and dispatches to
 * the renderer registered under ``name: "brainstorm-progress"`` —
 * which is THIS component, mounted at the root of the runtime tree.
 */

import { makeAssistantDataUI } from "@assistant-ui/react";
import { Check, Loader2, X } from "lucide-react";

type BrainstormProgressData = {
  step: "diverge" | "converge" | "hitl";
  status: "running" | "ok" | "failed";
  detail?: string | null;
};

const STEP_LABEL: Record<BrainstormProgressData["step"], string> = {
  diverge: "Brainstorming ideas",
  converge: "Picking the best one",
  hitl: "Spawning approval thread",
};

export const BrainstormProgressUI = makeAssistantDataUI<BrainstormProgressData>(
  {
    name: "brainstorm-progress",
    render: (props) => {
      const data = props.data as BrainstormProgressData;
      const icon =
        data.status === "running" ? (
          <Loader2 className="size-3 animate-spin text-muted-foreground" />
        ) : data.status === "ok" ? (
          <Check className="size-3 text-green-600 dark:text-green-400" />
        ) : (
          <X className="size-3 text-red-600 dark:text-red-400" />
        );
      return (
        <div className="my-1 flex items-center gap-2 rounded-md border border-dashed border-muted-foreground/30 bg-muted/30 px-3 py-1.5 text-xs">
          {icon}
          <span className="font-medium">{STEP_LABEL[data.step]}</span>
          {data.detail ? (
            <span className="font-mono text-[10px] text-muted-foreground truncate">
              — {data.detail}
            </span>
          ) : null}
        </div>
      );
    },
  },
);
