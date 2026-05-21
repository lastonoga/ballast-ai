"use client";

/**
 * Custom assistant-ui data-part renderer for ``data-brainstorm-branch``.
 *
 * Backend emits one of these per divergent branch (label × sample_idx)
 * via ``BRAINSTORM_BRANCH.emit(broadcaster, parent_id, ..., message_id=...)``.
 * The ``message_id`` is deterministic per (parent, label, sample_idx)
 * so each branch's row mutates in place ``running → ok|failed``
 * instead of stacking — the user sees N proposers tick off in parallel
 * underneath the high-level "Brainstorming ideas" line.
 *
 * Wire: backend writes a message part of shape
 *   ``{type: "data-brainstorm-branch", data: {...BrainstormBranchData}, state: "done"}``
 */

import { makeAssistantDataUI } from "@assistant-ui/react";
import { Check, Loader2, X } from "lucide-react";

type BrainstormBranchData = {
  label: string;
  sample_idx: number;
  status: "running" | "ok" | "failed";
  pool_size?: number | null;
  error_type?: string | null;
};

export const BrainstormBranchUI = makeAssistantDataUI<BrainstormBranchData>(
  {
    name: "brainstorm-branch",
    render: (props) => {
      const data = props.data as BrainstormBranchData;
      const icon =
        data.status === "running" ? (
          <Loader2 className="size-3 animate-spin text-muted-foreground" />
        ) : data.status === "ok" ? (
          <Check className="size-3 text-green-600 dark:text-green-400" />
        ) : (
          <X className="size-3 text-red-600 dark:text-red-400" />
        );
      const tail =
        data.status === "ok" && data.pool_size != null
          ? `${data.pool_size} idea${data.pool_size === 1 ? "" : "s"}`
          : data.status === "failed"
            ? (data.error_type ?? "failed")
            : null;
      return (
        <div className="my-0.5 ml-4 flex items-center gap-2 rounded-md border border-dashed border-muted-foreground/20 bg-muted/20 px-2.5 py-1 text-xs">
          {icon}
          <span className="font-mono text-[11px]">{data.label}</span>
          {tail ? (
            <span className="font-mono text-[10px] text-muted-foreground">
              — {tail}
            </span>
          ) : null}
        </div>
      );
    },
  },
);
