// The pipeline tabs feature's decisions: the three tabs in pipeline order,
// which layout each takes, when a step reads as done, and the remembered tab
// and unit. Pure, like editing.ts - the panel hands the stored strings in.

import type { ConceptState } from "./api";
import { MAIN_VIEWS } from "./concept";

export type Tab = "concept" | "views" | "animate";
/** In pipeline order: make the character, draw its views, animate it. */
export const TABS: readonly Tab[] = ["concept", "views", "animate"];

export const TAB_LABELS: Record<Tab, string> = {
  concept: "Concept",
  views: "Views",
  animate: "Animate",
};

/** A remembered tab, or Concept for nothing / a tab that no longer exists
 * (a studio from before this feature remembered "pipeline" or "board"). */
export function readTab(stored: string | null): Tab {
  return TABS.includes(stored as Tab) ? (stored as Tab) : "concept";
}

/** Which steps are done, for the ✓ on each tab. Concept: the base has a
 * pick. Views: every main view (back, left, top) has one. Animate never
 * finishes - there is always another animation. */
export function stepsDone(state: Pick<ConceptState, "base" | "views"> | null): Record<Tab, boolean> {
  const concept = !!state && state.base.pick !== null;
  const views = concept && MAIN_VIEWS.every((view) => {
    const row = state!.views[view];
    return !!row && row.pick !== null && row.pick !== undefined;
  });
  return { concept, views, animate: false };
}

/** A remembered unit, or "" for nothing / one that is not a slug. */
export function readUnit(stored: string | null): string {
  return stored && /^[a-z0-9][a-z0-9_]*$/.test(stored) ? stored : "";
}
