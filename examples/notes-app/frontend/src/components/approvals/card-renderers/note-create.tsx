import type { ApprovalCard } from "../use-approvals";

export function NoteCreateCard({
  card,
  onApprove,
  onReject,
}: {
  card: ApprovalCard;
  onApprove: () => void;
  onReject: () => void;
}) {
  const p = card.payload as { title: string; body: string };
  return (
    <div className="rounded border p-3 my-2 bg-white">
      <div className="text-xs text-gray-500 mb-1">Save note?</div>
      <div className="font-semibold">{p.title}</div>
      <div className="text-sm whitespace-pre-wrap mt-1">{p.body}</div>
      <div className="flex gap-2 mt-3">
        <button
          className="px-3 py-1 rounded bg-emerald-600 text-white"
          onClick={onApprove}
        >
          Approve
        </button>
        <button
          className="px-3 py-1 rounded bg-gray-200"
          onClick={onReject}
        >
          Reject
        </button>
      </div>
    </div>
  );
}
