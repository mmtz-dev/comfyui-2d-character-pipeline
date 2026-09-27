// The edit bar's decisions: how wide the panel is while it sits beside
// Piskel, and when leaving an edit must ask first. Pure: no DOM, no Piskel,
// no storage - the panel reads localStorage and hands the strings in, so the
// tests run it in Node.

/** Wide enough for the Animate tab's grid (five actions x eight directions)
 * beside Piskel; the owner's drag overrides it. */
export const EDIT_WIDTH_DEFAULT = 440;
export const EDIT_WIDTH_MIN = 240;
/** Piskel keeps at least a fifth of the window, whatever the drag says. */
export const EDIT_WIDTH_MAX_SHARE = 0.8;

/** A width for the split panel: 240 px .. 80% of the window, whole pixels.
 * A window too narrow for both bounds gets the minimum. */
export function clampEditWidth(width: number, viewport: number): number {
  if (!Number.isFinite(width)) return EDIT_WIDTH_DEFAULT;
  const max = Math.max(EDIT_WIDTH_MIN, viewport * EDIT_WIDTH_MAX_SHARE);
  return Math.round(Math.min(max, Math.max(EDIT_WIDTH_MIN, width)));
}

/** The remembered width, or the default for nothing / garbage. The window
 * bound is applied where the window is known (CSS clamp, the drag). */
export function readEditWidth(stored: string | null): number {
  const width = Number(stored);
  return stored && Number.isFinite(width) && width >= EDIT_WIDTH_MIN ? Math.round(width) : EDIT_WIDTH_DEFAULT;
}

/** Leaving an edit (Done, or opening something else) asks only when there is
 * unsaved paint on the canvas. */
export function mustConfirmLeave(editing: unknown, dirty: boolean): boolean {
  return editing !== null && editing !== undefined && dirty;
}

export function discardQuestion(label: string): string {
  return `Discard unsaved changes to ${label}?`;
}
