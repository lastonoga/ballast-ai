"use client";

/**
 * Custom assistant-ui data-part renderers for the framework's
 * ``DivergentConvergent`` pattern progress events. Backend emits these
 * from ``ballast/patterns/divergent_convergent/events.py`` via the
 * auto-connected ``default_chat_router`` on
 * ``divergent_convergent_progress``.
 *
 * Many per run (3-9 branch events + dedup + converge), so each
 * renderer is a single compact row — not a full card. Visual grouping
 * comes from the shared muted background; the surrounding chat keeps
 * its rhythm.
 *
 * Wire shape per part::
 *
 *   { "type": "data-branch-completed",
 *     "data": { "type": "branch-completed", "label": "...",
 *               "sample_idx": 0, "pool_size": 3 },
 *     "state": "done" }
 */

import { makeAssistantDataUI } from "@assistant-ui/react";
import {
  Brain,
  Check,
  Filter,
  Loader2,
  X,
} from "lucide-react";

// ── Payload shapes (mirror the Pydantic event models) ──────────────────

type BranchEnqueuedData = {
  type: "branch-enqueued";
  label: string;
  sample_idx: number;
};

type BranchCompletedData = {
  type: "branch-completed";
  label: string;
  sample_idx: number;
  pool_size: number;
};

type BranchFailedData = {
  type: "branch-failed";
  label: string;
  sample_idx: number;
  error_type: string;
};

type DedupCompletedData = {
  type: "dedup-completed";
  input_count: number;
  output_count: number;
};

type ConvergeStartedData = {
  type: "converge-started";
  candidate_count: number;
};

// ── Shared row primitive ───────────────────────────────────────────────

function Row({
  icon,
  children,
  tone = "muted",
}: {
  icon: React.ReactNode;
  children: React.ReactNode;
  tone?: "muted" | "ok" | "fail";
}) {
  const toneClasses = {
    muted: "text-muted-foreground",
    ok: "text-emerald-700 dark:text-emerald-400",
    fail: "text-rose-700 dark:text-rose-400",
  }[tone];
  return (
    <div
      className={`my-0.5 flex items-center gap-2 px-1 py-0.5 text-xs ${toneClasses}`}
    >
      <span className="shrink-0">{icon}</span>
      <span className="flex-1 truncate">{children}</span>
    </div>
  );
}

// ── Renderers ──────────────────────────────────────────────────────────

export const BranchEnqueuedUI = makeAssistantDataUI<BranchEnqueuedData>({
  name: "branch-enqueued",
  render: (props) => {
    const data = props.data as BranchEnqueuedData;
    return (
      <Row
        icon={<Loader2 className="size-3.5 animate-spin" />}
        tone="muted"
      >
        <span className="font-medium">{data.label}</span> brainstorming…
      </Row>
    );
  },
});

export const BranchCompletedUI = makeAssistantDataUI<BranchCompletedData>({
  name: "branch-completed",
  render: (props) => {
    const data = props.data as BranchCompletedData;
    return (
      <Row icon={<Check className="size-3.5" />} tone="ok">
        <span className="font-medium">{data.label}</span> ·{" "}
        <span className="tabular-nums">{data.pool_size}</span> ideas
      </Row>
    );
  },
});

export const BranchFailedUI = makeAssistantDataUI<BranchFailedData>({
  name: "branch-failed",
  render: (props) => {
    const data = props.data as BranchFailedData;
    return (
      <Row icon={<X className="size-3.5" />} tone="fail">
        <span className="font-medium">{data.label}</span> failed ·{" "}
        <span className="font-mono text-[10px]">{data.error_type}</span>
      </Row>
    );
  },
});

export const DedupCompletedUI = makeAssistantDataUI<DedupCompletedData>({
  name: "dedup-completed",
  render: (props) => {
    const data = props.data as DedupCompletedData;
    return (
      <Row icon={<Filter className="size-3.5" />} tone="muted">
        Dedup ·{" "}
        <span className="tabular-nums">{data.input_count}</span> →{" "}
        <span className="tabular-nums font-medium text-foreground">
          {data.output_count}
        </span>{" "}
        ideas
      </Row>
    );
  },
});

export const ConvergeStartedUI = makeAssistantDataUI<ConvergeStartedData>({
  name: "converge-started",
  render: (props) => {
    const data = props.data as ConvergeStartedData;
    return (
      <Row icon={<Brain className="size-3.5" />} tone="muted">
        Picking the best of{" "}
        <span className="tabular-nums font-medium text-foreground">
          {data.candidate_count}
        </span>{" "}
        candidates…
      </Row>
    );
  },
});

/**
 * Bundle of all DivergentConvergent-pattern renderers. Mount once
 * anywhere inside ``<RuntimeProvider />``; each ``makeAssistantDataUI``
 * self-registers via a side-effect hook on mount.
 */
export function DivergentConvergentEventsUI() {
  return (
    <>
      <BranchEnqueuedUI />
      <BranchCompletedUI />
      <BranchFailedUI />
      <DedupCompletedUI />
      <ConvergeStartedUI />
    </>
  );
}
