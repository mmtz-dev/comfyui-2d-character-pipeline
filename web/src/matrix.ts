// The action matrix feature's pure half: what the Matrix tab selects, what a
// press sends, and every sentence it shows. No DOM, no fetch - the tests run
// it in Node; matrix-view.ts draws it.
//
// The tab is about what COULD run, so the grid is always the whole five
// actions by eight facings and a cell the stage refuses carries the reason it
// refused. `GET /matrix/{unit}` recomputes that verdict from the curation
// record on every read, so nothing here decides whether a cell may render -
// it reads `runnable` and repeats the stage's own words.

import type { MatrixCell, MatrixState, RunRecord } from "./api";
import { STUDIO_ACTIONS, cellOrder } from "./board";
import { isActive } from "./concept";

/** The rows: the studio's five actions and their pipeline verbs. Re-exported
 * from ./board rather than written again - one source for the table the
 * board, the sheet and the job names all use (config.STUDIO_ACTIONS). */
export const ACTIONS = STUDIO_ACTIONS;

/** The columns: the side elevation's eight facings, in compass order. v1 is
 * side only; the top-down pitch is #285 and would add a second row of eight. */
export const SIDE_DIRECTIONS: readonly string[] = ["s0", "s45", "s90", "s135", "s180", "s225", "s270", "s315"];

/** Where the pipeline starts and the matrix's own default: the side view
 * game-02 ships (config.DEFAULT_DIRECTION). */
export const DEFAULT_DIRECTION = "s0";

/**
 * What the owner has ticked. `allDirs` is not a third opinion: it is true
 * exactly when all eight are selected, and it exists so `argsFor` can send
 * `--all-dirs` instead of eight names - the argv in `GET /runs` then reads
 * the way the owner would have typed the command.
 */
export interface Selection {
  actions: string[];
  directions: string[];
  allDirs: boolean;
}

/** Every action at s0 alone - the stage's own default (`plan(dirs=None)`),
 * and the cheap corner of a grid whose far corner is forty clips. */
export function defaultSelection(state?: MatrixState | null): Selection {
  const actions = state?.actions?.length ? state.actions.slice() : ACTIONS.map((row) => row.action);
  return { actions: inActionOrder(actions), directions: [DEFAULT_DIRECTION], allDirs: false };
}

function inActionOrder(actions: string[]): string[] {
  return ACTIONS.map((row) => row.action).filter((action) => actions.includes(action));
}

function inDirectionOrder(directions: string[]): string[] {
  return SIDE_DIRECTIONS.filter((direction) => directions.includes(direction));
}

/** The verb the job name uses: attack -> strike, the rest their own word. */
export function verbOf(action: string): string {
  return ACTIONS.find((row) => row.action === action)?.verb ?? action;
}

export function keyOf(action: string, direction: string): string {
  return `${action}/${direction}`;
}

/** Tick or untick one action. Never mutates - the tab holds one Selection and
 * replaces it, so a re-render always draws a value nobody edited underneath. */
export function toggleAction(selection: Selection, action: string): Selection {
  const on = selection.actions.includes(action);
  const actions = inActionOrder(
    on ? selection.actions.filter((name) => name !== action) : [...selection.actions, action]);
  return { ...selection, actions };
}

/** Tick or untick one direction. Ticking the eighth is the same selection the
 * "all 8 directions" toggle makes, so `allDirs` follows from the count rather
 * than being remembered separately and drifting out of step with it. */
export function toggleDirection(selection: Selection, direction: string): Selection {
  const on = selection.directions.includes(direction);
  const directions = inDirectionOrder(
    on ? selection.directions.filter((name) => name !== direction) : [...selection.directions, direction]);
  return { ...selection, directions, allDirs: directions.length === SIDE_DIRECTIONS.length };
}

/** The toggle: on is all eight, off is back to s0 alone. Off does not restore
 * an earlier hand-picked set, because the honest opposite of "every facing"
 * is the default the stage would have run, not a list nobody can see. */
export function setAllDirs(selection: Selection, on: boolean): Selection {
  return on
    ? { ...selection, directions: [...SIDE_DIRECTIONS], allDirs: true }
    : { ...selection, directions: [DEFAULT_DIRECTION], allDirs: false };
}

/** The cells a press would ask for, in `curation.cell_order` order -
 * DIRECTION-MAJOR, so a cancelled run leaves a complete direction. */
export function selectedKeys(selection: Selection): string[] {
  const keys: string[] = [];
  for (const action of selection.actions) {
    for (const direction of selection.directions) keys.push(keyOf(action, direction));
  }
  return keys.sort((a, b) => cellOrder(a) - cellOrder(b));
}

export function cellOf(state: MatrixState | null, key: string): MatrixCell | null {
  return state?.cells[key] ?? null;
}

/** The selected cells the stage would actually render: the ones whose
 * prerequisite verdict came back ok. This is the number the #277 cap counts
 * and the number the cost line multiplies. */
export function runnableKeys(selection: Selection, state: MatrixState | null): string[] {
  return selectedKeys(selection).filter((key) => cellOf(state, key)?.runnable === true);
}

/** The selected cells that are blocked - a cell with no row at all counts as
 * blocked, because a cell the API does not name is one nothing can render. */
export function blockedKeys(selection: Selection, state: MatrixState | null): string[] {
  return selectedKeys(selection).filter((key) => cellOf(state, key)?.runnable !== true);
}

/**
 * The `POST /jobs` args for a press: `{"stage": "matrix", "args": ...}`, the
 * same route every other stage launches through.
 *
 * `actions` is omitted when all five are ticked and `all_dirs: true` replaces
 * eight names, so `anim matrix knight --all-dirs` is what `GET /runs` shows
 * rather than a forty-character list the owner would never have typed.
 */
export function argsFor(unit: string, selection: Selection): Record<string, unknown> {
  const args: Record<string, unknown> = { unit };
  if (selection.actions.length !== ACTIONS.length) args.actions = selection.actions.join(",");
  if (selection.allDirs) args.all_dirs = true;
  else args.dirs = selection.directions.join(",");
  return args;
}

export interface MatrixGridCell {
  key: string;
  action: string;
  direction: string;
  /** null only when the API did not name the cell - it names all forty. */
  cell: MatrixCell | null;
  selected: boolean;
}

export interface MatrixGridRow {
  action: string;
  verb: string;
  cells: MatrixGridCell[];
}

export interface MatrixGrid {
  directions: readonly string[];
  rows: MatrixGridRow[];
}

/** Rows the five actions, columns the eight facings - ALWAYS the whole grid,
 * selected or not, blocked or not. The board shows what was rendered; this
 * tab shows what could be, which is why nothing is filtered out of it. */
export function gridOf(state: MatrixState | null, selection: Selection): MatrixGrid {
  const directions = state?.directions?.length ? state.directions : SIDE_DIRECTIONS;
  const rows = ACTIONS.map(({ action, verb }) => ({
    action,
    verb,
    cells: directions.map((direction) => ({
      key: keyOf(action, direction),
      action,
      direction,
      cell: cellOf(state, keyOf(action, direction)),
      selected: selection.actions.includes(action) && selection.directions.includes(direction),
    })),
  }));
  return { directions, rows };
}

/**
 * The cell's class: what it looks like and, for a blocked one, the `.locked`
 * the Concept tab already styles - constraint 6 asks the two tabs to refuse
 * in the same visual language. A blocked cell still shows its reason as text
 * and as a title; the class only greys it.
 */
export function cellClass(cell: MatrixCell | null, selected: boolean): string {
  const state = !cell ? "missing"
    : !cell.runnable ? "locked"
    : cell.status === "failed" ? "failed"
    : cell.status === "done" || cell.status === "kept" ? "done"
    : "ready";
  return selected ? `${state} selected` : state;
}

/** How long one cell's frame is on screen in the inline player. */
export function frameAt(elapsedMs: number, fps: number, count: number): number {
  if (count <= 0) return 0;
  const rate = fps > 0 ? fps : 12;
  const n = Math.floor((elapsedMs * rate) / 1000) % count;
  return n < 0 ? n + count : n;
}

/** The #277 hold as the tab says it, or null when the selection is inside the
 * cap. The stage refuses this selection before any GPU call, so saying it
 * here means the UI never sends a request that is already known to fail. */
export function holdReason(selection: Selection, state: MatrixState | null): string | null {
  const cap = state?.max_cells ?? 0;
  const n = runnableKeys(selection, state).length;
  if (!cap || n <= cap) return null;
  return `${n} cells selected and the matrix is held to ${cap} until the motion exaggeration `
    + `feature (#277) lands - sprite-studio.md orders #277 before #278 so the matrix renders `
    + `once. Tick fewer actions or directions, or raise the cap with ANIM_MATRIX_MAX_CELLS.`;
}

/**
 * The one sentence that keeps Generate disabled, or null. In this order,
 * because that is the order the owner can act on them: the card is busy (wait),
 * nothing is ticked (tick something), everything ticked is blocked (go and
 * accept a view), the selection is over the cap (tick less).
 */
export function generateBlocked(selection: Selection, state: MatrixState | null,
                                busy: RunRecord | null): string | null {
  if (busy && isActive(busy)) {
    return `the GPU is busy with ${busy.stage} ${busy.id} - one run at a time on the card`;
  }
  if (!state) return "pick a unit first";
  if (!selection.actions.length || !selection.directions.length) {
    return "nothing is selected - tick at least one action and one direction";
  }
  const blocked = blockedKeys(selection, state);
  if (blocked.length === selectedKeys(selection).length) {
    const first = cellOf(state, blocked[0]!);
    return `every selected cell is blocked - ${first?.reason ?? "no cell of this unit can render yet"}`;
  }
  return holdReason(selection, state);
}

/** "5 actions × 1 direction = 5 cells, 4 to render, 1 blocked". */
export function selectionText(selection: Selection, state: MatrixState | null): string {
  const cells = selectedKeys(selection).length;
  const runnable = runnableKeys(selection, state).length;
  const blocked = cells - runnable;
  const parts = [
    `${plural(selection.actions.length, "action")} × ${plural(selection.directions.length, "direction")}`
    + ` = ${plural(cells, "cell")}`,
    `${runnable} to render`,
  ];
  if (blocked) parts.push(`${blocked} blocked`);
  return parts.join(", ");
}

/** What the press costs, before it is pressed: how many cells, the one seed
 * they all share, the warm GPU estimate from the measured numbers, and the
 * chain each cell runs. An estimate on the record - never a timeout. */
export function costText(selection: Selection, state: MatrixState | null): string {
  const cells = runnableKeys(selection, state).length;
  const per = state?.estimate_seconds ?? 0;
  const seed = state?.seed ?? null;
  return [
    plural(cells, "cell"),
    seed === null ? "one seed (derived from the base on the first run)" : `one seed (${seed})`,
    per && cells ? `~${duration(cells * per)} of GPU, warm` : "no GPU cost - nothing to render",
    "rig → clip → pick --phases → post per cell",
  ].join(" · ");
}

/** The banner over the grid: how much of the unit can render at all, how far
 * and the first reason standing in the way. It is the
 * answer to "why is everything grey", which is otherwise a silent tab. */
export function blockedText(state: MatrixState | null): string {
  if (!state) {
    return "Pick a unit. Every action × direction cell of it is a square here, and a square "
      + "that cannot be generated says why.";
  }
  const cells = Object.values(state.cells);
  const blocked = cells.filter((cell) => !cell.runnable);
  if (!blocked.length) return `all ${plural(cells.length, "cell")} can be generated`;
  const reason = blocked[0]!.reason ?? "blocked";
  if (!state.selectable) return `nothing can be generated yet · ${reason}`;
  return `${state.selectable} of ${cells.length} cells can be generated · `
    + `${blocked.length} blocked, first: ${reason}`;
}

/** A row's or a column's own verdict: the first reason every cell in it
 * shares, or null when at least one of them can render. What disables the
 * checkbox, with the reason as its title - a tick nothing can act on is
 * worse than no tick. */
export function groupReason(cells: (MatrixCell | null)[]): string | null {
  if (cells.some((cell) => cell?.runnable)) return null;
  for (const cell of cells) {
    if (cell?.reason) return cell.reason;
  }
  return "no cell of this row can be generated yet";
}

export function actionReason(state: MatrixState | null, action: string): string | null {
  if (!state) return null;
  return groupReason(SIDE_DIRECTIONS.map((direction) => cellOf(state, keyOf(action, direction))));
}

export function directionReason(state: MatrixState | null, direction: string): string | null {
  if (!state) return null;
  return groupReason(ACTIONS.map((row) => cellOf(state, keyOf(row.action, direction))));
}

/** "loop does not close: 0.31 > 0.12" - information, not a failure. `pick`
 * computes it for the three looping actions and nothing acts on it; showing
 * it is how the owner learns a cycle will stutter before a sheet does. */
export function closureText(cell: MatrixCell | null): string | null {
  const closure = cell?.closure;
  if (!closure || closure.ok === null || closure.distance === null) return null;
  return closure.ok
    ? `loop closes: ${closure.distance} ≤ ${closure.tolerance}`
    : `loop does not close: ${closure.distance} > ${closure.tolerance}`;
}

export interface PhaseChip {
  tag: string;
  /** Which frames of the strip carry it - a phase is a span, not a frame. */
  frames: number[];
}

/** The phase tags of a finished cell, one chip per tag in the order they
 * first appear. `phases` is one entry PER FRAME (attack picks twelve frames
 * across five phases), so the chips are the distinct tags and each one knows
 * which frames it covers; an untagged frame is left out rather than shown as
 * a blank chip. */
export function phaseChips(cell: MatrixCell | null): PhaseChip[] {
  const chips: PhaseChip[] = [];
  for (const [index, tag] of (cell?.phases ?? []).entries()) {
    if (!tag) continue;
    const seen = chips.find((chip) => chip.tag === tag);
    if (seen) seen.frames.push(index);
    else chips.push({ tag, frames: [index] });
  }
  return chips;
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function duration(seconds: number): string {
  return seconds < 90 ? `${Math.round(seconds)} s` : `${Math.round(seconds / 60)} min`;
}
