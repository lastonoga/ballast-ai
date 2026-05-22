"use client";

/**
 * DBOS workflow / step inspector — right-hand panel.
 *
 * Polls ``GET /dbos/threads/{thread_id}/workflows`` every few seconds for
 * the active conversation, renders each workflow as an expandable card
 * with its steps, and exposes per-workflow (cancel/resume/fork) +
 * per-step (fork-from-this-step) actions wired to the backend's
 * ``POST /dbos/workflows/{id}/{op}`` endpoints.
 *
 * Why this lives in the example, not the framework: framework's
 * ``build_dbos_router`` is shape-only (no opinion on UI). Apps choose
 * how to render it — a sidecar panel here, a separate ops dashboard
 * elsewhere, etc.
 */

import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  CornerDownRight,
  GitFork,
  Lightbulb,
  Play,
  RefreshCw,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useAuiState } from "@assistant-ui/react";

import { Button } from "@/components/ui/button";

const DEFAULT_API_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 3000;

type Workflow = {
  workflow_id: string;
  status: string | null;
  name: string | null;
  class_name: string | null;
  config_name: string | null;
  queue_name: string | null;
  queue_partition_key: string | null;
  created_at: number | null;
  updated_at: number | null;
  executor_id: string | null;
  app_version: string | null;
  parent_workflow_id: string | null;
  forked_from: string | null;
  was_forked_from: boolean;
  recovery_attempts: number | null;
  output?: string;
  error?: string;
};

type Step = {
  function_id: number | null;
  function_name: string | null;
  child_workflow_id: string | null;
  started_at_epoch_ms: number | null;
  completed_at_epoch_ms: number | null;
  output: unknown;
  error: string | null;
};

const STATUS_COLOR: Record<string, string> = {
  PENDING: "bg-blue-500/10 text-blue-700 dark:text-blue-300",
  ENQUEUED: "bg-blue-500/10 text-blue-700 dark:text-blue-300",
  SUCCESS: "bg-green-500/10 text-green-700 dark:text-green-300",
  ERROR: "bg-red-500/10 text-red-700 dark:text-red-300",
  CANCELLED: "bg-gray-500/10 text-gray-700 dark:text-gray-300",
  DELAYED: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-300",
  MAX_RECOVERY_ATTEMPTS_EXCEEDED:
    "bg-red-500/10 text-red-700 dark:text-red-300",
};

function fmtTime(ms: number | null): string {
  if (ms == null) return "—";
  const d = new Date(ms);
  return d.toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function fmtDuration(start: number | null, end: number | null): string {
  if (start == null) return "—";
  const e = end ?? Date.now();
  const dur = Math.max(0, e - start);
  if (dur < 1000) return `${dur}ms`;
  if (dur < 60_000) return `${(dur / 1000).toFixed(1)}s`;
  return `${(dur / 60_000).toFixed(1)}m`;
}

function preview(value: unknown): string {
  if (value == null) return "";
  let s: string;
  try {
    s = typeof value === "string" ? value : JSON.stringify(value);
  } catch {
    s = String(value);
  }
  if (s.length > 200) s = s.slice(0, 200) + "…";
  return s;
}

function StatusBadge({ status }: { status: string | null | undefined }) {
  const s = status ?? "—";
  const cls = STATUS_COLOR[s] ?? "bg-muted text-muted-foreground";
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] font-mono uppercase ${cls}`}
    >
      {s}
    </span>
  );
}

// Hard cap on recursive descent so a pathological self-referential or
// extremely deep tree can't blow up the inspector. Real workflows in
// this codebase top out around 4 levels (BrainstormFlow →
// DivergentConvergent → queue-enqueued _diverge_one → pydantic-ai
// model_request); 6 leaves slack for future patterns.
const MAX_NESTED_DEPTH = 6;

function StepRow({
  apiUrl,
  step,
  depth,
  onForkFrom,
}: {
  apiUrl: string;
  step: Step;
  depth: number;
  onForkFrom: (functionId: number) => void;
}) {
  const [childExpanded, setChildExpanded] = useState(false);

  const status: string =
    step.error != null
      ? "ERROR"
      : step.completed_at_epoch_ms != null
      ? "SUCCESS"
      : step.started_at_epoch_ms != null
      ? "PENDING"
      : "ENQUEUED";

  const hasChild = !!step.child_workflow_id;
  const canExpand = hasChild && depth < MAX_NESTED_DEPTH;

  return (
    <div className="border-l-2 border-muted pl-3 py-2 text-xs">
      <div className="flex items-center gap-2">
        {canExpand ? (
          <button
            type="button"
            className="rounded p-0.5 hover:bg-accent"
            title={childExpanded ? "Collapse child workflow" : "Expand child workflow"}
            onClick={() => setChildExpanded((v) => !v)}
          >
            {childExpanded ? (
              <ChevronDown className="size-3" />
            ) : (
              <ChevronRight className="size-3" />
            )}
          </button>
        ) : (
          <span className="w-4" />
        )}
        <span className="font-mono text-muted-foreground tabular-nums">
          #{step.function_id ?? "?"}
        </span>
        <span className="font-mono truncate flex-1" title={step.function_name ?? ""}>
          {step.function_name ?? "<unnamed>"}
        </span>
        <StatusBadge status={status} />
        {step.function_id != null && (
          <button
            type="button"
            className="rounded p-1 hover:bg-accent"
            title={`Fork workflow from step #${step.function_id}`}
            onClick={() => onForkFrom(step.function_id!)}
          >
            <GitFork className="size-3" />
          </button>
        )}
      </div>
      <div className="mt-1 flex gap-3 text-[10px] text-muted-foreground">
        <span>started {fmtTime(step.started_at_epoch_ms)}</span>
        <span>
          dur {fmtDuration(step.started_at_epoch_ms, step.completed_at_epoch_ms)}
        </span>
      </div>
      {hasChild && (
        <div className="mt-1 flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
          <CornerDownRight className="size-3" />
          <span className="truncate" title={step.child_workflow_id!}>
            {step.child_workflow_id}
          </span>
        </div>
      )}
      {step.output != null && (
        <div className="mt-1 rounded bg-muted/50 p-1.5 font-mono text-[10px] break-all">
          {preview(step.output)}
        </div>
      )}
      {step.error && (
        <div className="mt-1 flex items-start gap-1 text-[10px] text-destructive">
          <AlertCircle className="size-3 mt-0.5 shrink-0" />
          <span className="break-all">{step.error}</span>
        </div>
      )}
      {childExpanded && canExpand && (
        <div className="mt-2 ml-2 border-l-2 border-dashed border-muted pl-2">
          <NestedSteps
            apiUrl={apiUrl}
            workflowId={step.child_workflow_id!}
            depth={depth + 1}
            onForkFrom={onForkFrom}
          />
        </div>
      )}
    </div>
  );
}

function NestedSteps({
  apiUrl,
  workflowId,
  depth,
  onForkFrom,
}: {
  apiUrl: string;
  workflowId: string;
  depth: number;
  onForkFrom: (functionId: number) => void;
}) {
  const [steps, setSteps] = useState<Step[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // One-shot fetch + slow poll. Top-level WorkflowCard already polls
  // the steps endpoint for its own workflow at POLL_INTERVAL_MS;
  // nested workflows usually finish quickly and we'd rather keep the
  // request count modest at deep recursion levels.
  useEffect(() => {
    let cancelled = false;
    const fetchSteps = async () => {
      try {
        const r = await fetch(`${apiUrl}/dbos/workflows/${workflowId}/steps`);
        if (!r.ok) throw new Error(`steps ${r.status}`);
        const data = (await r.json()) as Step[];
        if (!cancelled) {
          setSteps(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    };
    void fetchSteps();
    const t = setInterval(() => void fetchSteps(), POLL_INTERVAL_MS * 2);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [apiUrl, workflowId]);

  if (error) {
    return (
      <div className="py-1 text-[10px] text-destructive">
        child steps: {error}
      </div>
    );
  }
  if (steps === null) {
    return (
      <div className="py-1 text-[10px] text-muted-foreground italic">
        loading child steps…
      </div>
    );
  }
  if (steps.length === 0) {
    return (
      <div className="py-1 text-[10px] text-muted-foreground italic">
        no steps recorded yet
      </div>
    );
  }
  return (
    <div className="space-y-1">
      {steps.map((s, i) => (
        <StepRow
          key={s.function_id ?? `${workflowId}-${i}`}
          apiUrl={apiUrl}
          step={s}
          depth={depth}
          onForkFrom={onForkFrom}
        />
      ))}
    </div>
  );
}

function WorkflowCard({
  apiUrl,
  workflow,
  onAction,
}: {
  apiUrl: string;
  workflow: Workflow;
  onAction: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [steps, setSteps] = useState<Step[]>([]);
  const [loadingSteps, setLoadingSteps] = useState(false);

  // Lazy-load steps on first expand; refresh on each poll cycle thereafter
  // by relying on the parent's polling to re-render us (we re-fetch
  // whenever ``expanded`` is true and parent ticks).
  useEffect(() => {
    if (!expanded) return;
    let cancelled = false;
    const fetchSteps = async () => {
      try {
        setLoadingSteps(true);
        const r = await fetch(
          `${apiUrl}/dbos/workflows/${workflow.workflow_id}/steps`,
        );
        if (!r.ok) throw new Error(`steps ${r.status}`);
        const data = (await r.json()) as Step[];
        if (!cancelled) setSteps(data);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn("[dbos-inspector] fetch steps failed", err);
      } finally {
        if (!cancelled) setLoadingSteps(false);
      }
    };
    void fetchSteps();
    const t = setInterval(() => void fetchSteps(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [apiUrl, workflow.workflow_id, expanded]);

  const doAction = async (op: "cancel" | "resume") => {
    try {
      await fetch(`${apiUrl}/dbos/workflows/${workflow.workflow_id}/${op}`, {
        method: "POST",
      });
      onAction();
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(`[dbos-inspector] ${op} failed`, err);
    }
  };

  const doFork = async (startStep: number) => {
    try {
      await fetch(`${apiUrl}/dbos/workflows/${workflow.workflow_id}/fork`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start_step: startStep }),
      });
      onAction();
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[dbos-inspector] fork failed", err);
    }
  };

  const isTerminal =
    workflow.status === "SUCCESS"
    || workflow.status === "ERROR"
    || workflow.status === "CANCELLED";

  return (
    <div className="rounded-md border bg-card p-2 text-xs shadow-sm">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="rounded p-0.5 hover:bg-accent"
        >
          {expanded ? (
            <ChevronDown className="size-3.5" />
          ) : (
            <ChevronRight className="size-3.5" />
          )}
        </button>
        <span className="font-mono text-[10px] truncate flex-1" title={workflow.name ?? ""}>
          {workflow.name ?? "<unnamed>"}
        </span>
        <StatusBadge status={workflow.status} />
      </div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
        <span>created {fmtTime(workflow.created_at)}</span>
        <span>updated {fmtTime(workflow.updated_at)}</span>
        {workflow.recovery_attempts != null && workflow.recovery_attempts > 0 && (
          <span className="text-yellow-600 dark:text-yellow-400">
            recovery×{workflow.recovery_attempts}
          </span>
        )}
      </div>
      <div className="mt-1 font-mono text-[9px] text-muted-foreground/70 truncate" title={workflow.workflow_id}>
        {workflow.workflow_id}
      </div>
      {workflow.forked_from && (
        <div className="mt-1 font-mono text-[9px] text-muted-foreground">
          ↳ forked from {workflow.forked_from}
        </div>
      )}
      {workflow.error && (
        <div className="mt-1 flex items-start gap-1 text-[10px] text-destructive">
          <AlertCircle className="size-3 mt-0.5 shrink-0" />
          <span className="break-all">{workflow.error}</span>
        </div>
      )}

      <div className="mt-1.5 flex gap-1">
        {!isTerminal && (
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-6 px-2 text-[10px]"
            onClick={() => void doAction("cancel")}
          >
            <X className="size-3" /> Cancel
          </Button>
        )}
        {isTerminal && (
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-6 px-2 text-[10px]"
            onClick={() => void doAction("resume")}
          >
            <Play className="size-3" /> Resume
          </Button>
        )}
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="h-6 px-2 text-[10px]"
          onClick={() => void doFork(1)}
          title="Fork from step 1 (full restart)"
        >
          <RefreshCw className="size-3" /> Restart
        </Button>
      </div>

      {expanded && (
        <div className="mt-2 space-y-1">
          {loadingSteps && steps.length === 0 ? (
            <div className="text-[10px] text-muted-foreground italic">
              loading steps…
            </div>
          ) : steps.length === 0 ? (
            <div className="text-[10px] text-muted-foreground italic">
              no steps recorded yet
            </div>
          ) : (
            steps.map((s, i) => (
              <StepRow
                key={s.function_id ?? `${workflow.workflow_id}-${i}`}
                apiUrl={apiUrl}
                step={s}
                depth={0}
                onForkFrom={(fid) => void doFork(fid)}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

export function DbosInspector() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL;
  const remoteId = useAuiState((s) => s.threadListItem.remoteId);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  // Poll workflows for the active thread. ``remoteId`` is undefined for
  // drafts (thread not yet initialized) — render an empty state instead
  // of hammering the backend with __LOCALID_ ids.
  //
  // We pass TWO prefixes: the framework's ``agent-run:`` (chat-turn
  // workflows) AND the app's ``brainstorm:`` (divergent-convergent
  // runs started via POST /workflows/brainstorm-flow). Each top-level
  // workflow's nested execution tree (queued divergent samples, child
  // synthesizer steps, HITL helpers) becomes visible by clicking the
  // ChevronRight on any step whose ``child_workflow_id`` is set.
  useEffect(() => {
    if (!remoteId) {
      setWorkflows([]);
      return;
    }
    let cancelled = false;
    const prefixes = [
      `agent-run:${remoteId}:`,
      `brainstorm:${remoteId}:`,
    ];
    const qs = new URLSearchParams();
    qs.set("limit", "50");
    for (const p of prefixes) qs.append("prefix", p);
    const url = `${apiUrl}/dbos/threads/${remoteId}/workflows?${qs.toString()}`;
    const fetchWorkflows = async () => {
      try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`workflows ${r.status}`);
        const data = (await r.json()) as Workflow[];
        if (!cancelled) {
          setWorkflows(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    };
    void fetchWorkflows();
    const t = setInterval(() => void fetchWorkflows(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [apiUrl, remoteId, tick]);

  const refresh = () => setTick((v) => v + 1);

  const startBrainstorm = async () => {
    if (!remoteId) return;
    const topic = window.prompt(
      "Brainstorm topic (optional — empty uses default)",
      "",
    );
    if (topic === null) return; // user cancelled
    try {
      const r = await fetch(`${apiUrl}/workflows/brainstorm-flow`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: topic || "Идеи для todo на эту неделю",
          parent_thread_id: remoteId,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
      // Workflow is fire-and-forget. Helper thread will appear in the
      // sidebar via the parent's SSE thread-created event; the inspector
      // will pick the new workflow up on its next poll tick.
      refresh();
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[brainstorm] start failed", err);
      window.alert(`Brainstorm failed: ${err instanceof Error ? err.message : err}`);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b px-3 py-2">
        <div className="flex flex-col">
          <span className="text-sm font-semibold">DBOS Workflows</span>
          <span className="text-[10px] text-muted-foreground">
            {remoteId ? `thread ${remoteId.slice(0, 8)}…` : "no active thread"}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-7 gap-1 px-2 text-[11px]"
            onClick={() => void startBrainstorm()}
            disabled={!remoteId}
            title="Run divergent-convergent brainstorm → HITL approval"
          >
            <Lightbulb className="size-3.5" /> Brainstorm
          </Button>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-7 w-7 p-0"
            onClick={refresh}
            title="Refresh now"
          >
            <RefreshCw className="size-3.5" />
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {error && (
          <div className="rounded border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
            {error}
          </div>
        )}
        {!remoteId && (
          <div className="text-xs text-muted-foreground italic px-1 py-4">
            Send a message to start a workflow.
          </div>
        )}
        {remoteId && workflows.length === 0 && !error && (
          <div className="text-xs text-muted-foreground italic px-1 py-4">
            No workflows for this thread yet.
          </div>
        )}
        {workflows.map((wf) => (
          <WorkflowCard
            key={wf.workflow_id}
            apiUrl={apiUrl}
            workflow={wf}
            onAction={refresh}
          />
        ))}
      </div>
    </div>
  );
}
