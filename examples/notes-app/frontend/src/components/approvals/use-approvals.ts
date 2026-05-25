"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export type ApprovalCard = {
  id: string;
  workflow_id: string;
  respond_topic: string;
  kind: string;
  payload: Record<string, unknown>;
  parent_thread_id: string | null;
  user_id: string | null;
  status: "pending" | "approved" | "rejected" | "timeout";
  created_at: string;
};

export type Decision = {
  decision: "approve" | "reject";
  modified?: Record<string, unknown> | null;
  feedback?: string | null;
};

const DEFAULT_API_URL = "http://localhost:8000";
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL;

export function useApprovals() {
  const [pending, setPending] = useState<ApprovalCard[]>([]);
  const eventSourceRef = useRef<EventSource | null>(null);

  const fetchInitial = useCallback(async () => {
    const r = await fetch(`${API_BASE}/approvals?status=pending`);
    if (r.ok) setPending(await r.json());
  }, []);

  useEffect(() => {
    fetchInitial();

    const es = new EventSource(`${API_BASE}/approvals/stream`);
    eventSourceRef.current = es;

    es.addEventListener("card-requested", (e: MessageEvent) => {
      const card: ApprovalCard = JSON.parse(e.data);
      setPending((prev) =>
        prev.find((c) => c.id === card.id) ? prev : [card, ...prev],
      );
    });
    es.addEventListener("card-decided", (e: MessageEvent) => {
      const card: ApprovalCard = JSON.parse(e.data);
      setPending((prev) => prev.filter((c) => c.id !== card.id));
    });

    return () => {
      es.close();
    };
  }, [fetchInitial]);

  const decide = useCallback(
    async (cardId: string, decision: Decision) => {
      const r = await fetch(`${API_BASE}/approvals/${cardId}/decision`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(decision),
      });
      if (!r.ok) throw new Error(`decide failed: ${r.status}`);
      // Optimistic remove; SSE card-decided will reconcile.
      setPending((prev) => prev.filter((c) => c.id !== cardId));
    },
    [],
  );

  return { pending, decide };
}
