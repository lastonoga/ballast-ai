/**
 * Debug-mode store: gates internal-detail UI (reasoning chains, tool-call
 * cards, raw tool args/results) behind a toggle. Persisted in
 * ``localStorage`` so the choice survives reloads but stays per-browser.
 *
 * Default: ``false`` (clean chat). Toggle from the page header.
 */
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

type DebugState = {
  enabled: boolean;
  toggle: () => void;
  set: (next: boolean) => void;
};

export const useDebugMode = create<DebugState>()(
  persist(
    (set) => ({
      enabled: false,
      toggle: () => set((s) => ({ enabled: !s.enabled })),
      set: (next) => set({ enabled: next }),
    }),
    {
      name: "notes-app.debug-mode",
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
