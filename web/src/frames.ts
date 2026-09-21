// Which of a job's PNG sets the editor opens, and what each frame is called
// on the timeline. Pure: no DOM, no Piskel, no fetch - the tests run it in
// Node.

import type { FrameEntry, PicksState } from "./api";

/**
 * The sets the editor can open, best first. `post` is the finished 64x64
 * frames (the ones `publish` ships); `clip` the generated frames, 1:1 with
 * the source; `frames` the source itself (a filmed clip's frames or the rig's
 * projections). `render` and `strip` are NOT here: they hold one N-cell strip
 * per file, not one PNG per frame - slicing them needs strip/layout.json,
 * which is a later feature.
 */
export const OPENABLE_SETS = ["post", "clip", "frames"] as const;
export type OpenableSet = (typeof OPENABLE_SETS)[number];

/** `knight_strike_7.png` -> 7; `palette.png`, `knight_strike_7_hd.png` -> null. */
export function frameNumber(file: string): number | null {
  const match = /_(\d+)\.png$/i.exec(file);
  return match ? Number(match[1]) : null;
}

/**
 * `post/` also holds `palette.png` (the snap palette) and, after `post
 * --parts`, `<stem>_<n>_hd.png` / `<stem>_<n>_labels.png` diagnostics. The
 * frames are the `<stem>_<n>.png` files, in the order of n - the same rule
 * `publish` applies. Other sets are taken as listed.
 */
export function editableEntries(set: string, entries: FrameEntry[]): FrameEntry[] {
  if (set !== "post") return entries.slice();
  return entries
    .filter((entry) => frameNumber(entry.file) !== null)
    .sort((a, b) => (frameNumber(a.file) ?? 0) - (frameNumber(b.file) ?? 0));
}

export interface ChosenSet {
  set: OpenableSet;
  entries: FrameEntry[];
}

/** The best set the job has, or null when it has none the editor can open. */
export function chooseFrameSet(stages: Record<string, FrameEntry[]>): ChosenSet | null {
  for (const set of OPENABLE_SETS) {
    const entries = editableEntries(set, stages[set] ?? []);
    if (entries.length) return { set, entries };
  }
  return null;
}

/**
 * The phase tag for each frame of `set`, from the review state.
 *
 * `frames` and `clip` are indexed like the source: frame i's tag is
 * `frames[i].phase`. `post` is one frame per pick, in ascending pick order
 * (what `strip` and `post` build), so post frame k is source frame
 * `picks[k]` and carries that frame's tag. No picks, or a set of a different
 * length than the picks: no tags, rather than wrong ones.
 */
export function phaseLabels(set: string, count: number, picks: PicksState | null): (string | null)[] {
  const none = Array.from({ length: count }, () => null);
  if (!picks) return none;
  const byIndex = new Map(picks.frames.map((frame) => [frame.index, frame.phase]));
  if (set === "frames" || set === "clip") {
    return none.map((_, i) => byIndex.get(i) ?? null);
  }
  if (set === "post") {
    const order = picks.picks.slice().sort((a, b) => a - b);
    if (order.length !== count) return none;
    return order.map((source) => byIndex.get(source) ?? null);
  }
  return none;
}
