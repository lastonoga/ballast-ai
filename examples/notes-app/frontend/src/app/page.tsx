"use client";

import { AssistantSidebar } from "@/components/assistant-ui/assistant-sidebar";
import { DeleteNoteApproval } from "@/components/assistant-ui/delete-note-approval";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import { DbosInspector } from "@/components/dbos-inspector";
import { DebugToggle } from "@/components/debug-toggle";
import { RuntimeProvider } from "@/components/runtime-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";

export default function Home() {
  return (
    <RuntimeProvider>
      {/* `makeAssistantToolUI` self-registers via a side-effect hook; mounting
          this anywhere inside the runtime provider is enough to take over
          rendering for the `delete_note` tool call. */}
      <DeleteNoteApproval />
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <ResizablePanelGroup orientation="horizontal">
          <ResizablePanel id="main" defaultSize={67}>
            <AssistantSidebar>
              <div className="flex h-full flex-col">
                <header className="flex items-center justify-between border-b px-4 py-3">
                  <div className="flex flex-col">
                    <span className="text-sm font-semibold">Notes</span>
                    <span className="text-xs text-muted-foreground">
                      iteration 3 — live backend
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    <DebugToggle />
                    <ThemeToggle />
                  </div>
                </header>
                <div className="flex-1 overflow-y-auto p-3">
                  <ThreadList />
                </div>
              </div>
            </AssistantSidebar>
          </ResizablePanel>
          <ResizableHandle withHandle />
          <ResizablePanel id="dbos" defaultSize={33}>
            <DbosInspector />
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </RuntimeProvider>
  );
}
