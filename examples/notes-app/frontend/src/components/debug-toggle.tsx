"use client";

import { BugIcon, BugOffIcon } from "lucide-react";
import type { FC } from "react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useDebugMode } from "@/lib/use-debug-mode";

/**
 * Header button that toggles ``useDebugMode``.
 *
 * When OFF (default), assistant reasoning chains and tool-call cards are
 * hidden — chat stays focused on the human-facing reply. Flip ON to
 * inspect what the agent actually did (used heavily during dogfood
 * iterations).
 */
export const DebugToggle: FC = () => {
  const enabled = useDebugMode((s) => s.enabled);
  const toggle = useDebugMode((s) => s.toggle);
  const Icon = enabled ? BugIcon : BugOffIcon;
  const label = enabled
    ? "Debug ON — reasoning + tool calls visible"
    : "Debug OFF — clean chat";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-pressed={enabled}
          aria-label={label}
          onClick={toggle}
        >
          <Icon className="size-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
};
