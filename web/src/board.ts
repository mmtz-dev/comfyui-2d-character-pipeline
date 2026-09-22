// The curation board's pure parts: the grid out of the board state, the
// reducer the view folds edits into, and the header lines. No DOM, no
// fetch - the tests run it in Node; board-view.ts draws it.

import type { BoardCell, BoardState, BoardView, CurationMode, FrameMark, GateState, JobRow, TurnaroundState, ViewState, Waiting } from "./api";

/** The board's rows: the studio's five actions, and the pipeline verb of each
 * (attack <-> strike; the rest are the same word) - config.STUDIO_ACTIONS. */
export const STUDIO_ACTIONS: ReadonlyArray<{ action: string; verb: string }> = [
  { action: "attack", verb: "strike" },
  { action: "idle", verb: "idle" },
  { action: "walk", verb: "walk" },
  { action: "run", verb: "run" },
  { action: "jump", verb: "jump" },
];

/** The 16 direction cells in the API's order (config.DIRECTIONS). */
export const DIRECTIONS: readonly string[] = ["side", "top"].flatMap((elevation) =>
  [0, 45, 90, 135, 180, 225, 270, 315].map((yaw) => `${elevation[0]}${yaw}`),
);

export const MODES: readonly CurationMode[] = ["gates", "every", "end"];

export const MODE_HELP: Record<CurationMode, string> = {
  gates: "gates: clips wait for the base to be accepted, and a direction outside s0 for the turnaround; review in a batch",
  every: "every step: as gates, and a cell's clip waits until every earlier cell is accepted (s0's actions first)",
  end: "end only: nothing blocks; review everything at the end",
};

export function studioAction(action: string): string | null {
  const key = action.trim().toLowerCase();
  const hit = STUDIO_ACTIONS.find((row) => row.action === key || row.verb === key);
  return hit ? hit.action : null;
}

export function cellKey(action: string, direction: string): string | null {
  const name = studioAction(action);
  const dir = direction.trim().toLowerCase();
  return name && DIRECTIONS.includes(dir) ? `${name}/${dir}` : null;
}

/**
 * The cell a `GET /jobs` row belongs to, or null - the API's `cell_of_row`,
 * so the panel can say which cell a job is before the board is fetched.
 * Bake-off cells, base and concept jobs and other units' jobs are not cells.
 */
export function cellOfJob(unit: string, job: JobRow): string | null {
  if (job.slug !== unit || job.name.startsWith("bakeoff_")) return null;
  return cellKey(job.action, job.direction || "s0");
}

/** `{cell key: job name}` for one unit, from the jobs list: the canonical
 * name (`<unit>_<verb>[_<dir>]`) wins a cell over a variant. */
export function cellsFromJobs(unit: string, jobs: JobRow[]): Record<string, string> {
  const cells: Record<string, string> = {};
  for (const job of jobs) {
    const key = cellOfJob(unit, job);
    if (!key) continue;
    const [action, direction] = key.split("/") as [string, string];
    const verb = STUDIO_ACTIONS.find((row) => row.action === action)!.verb;
    const canonical = direction === "s0" ? `${unit}_${verb}` : `${unit}_${verb}_${direction}`;
    const current = cells[key];
    if (!current || job.name === canonical) cells[key] = job.name;
  }
  return cells;
}

export interface GridRow {
  action: string;
  verb: string;
  cells: (BoardCell | null)[];
}

export interface Grid {
  directions: string[];
  rows: GridRow[];
}

/**
 * Rows = the five studio actions in order, columns = the directions the unit
 * has a job in (s0 first, the API's order), a null where there is no job.
 * An action with no job in any direction is still a row - the board shows
 * what is missing as much as what is there.
 */
export function gridOf(board: Pick<BoardState, "cells" | "directions">): Grid {
  const directions = DIRECTIONS.filter((dir) => board.directions.includes(dir));
  const rows = STUDIO_ACTIONS.map(({ action, verb }) => ({
    action,
    verb,
    cells: directions.map((dir) => board.cells[`${action}/${dir}`] ?? null),
  }));
  return { directions, rows };
}

/** Click cycles a frame: none -> accept -> reject -> none. */
export function cycleMark(mark: FrameMark | null): FrameMark | null {
  if (mark === null) return "accept";
  if (mark === "accept") return "reject";
  return null;
}

export type BoardEvent =
  | { type: "cell"; cell: BoardCell }
  | { type: "mark"; key: string; index: number; mark: FrameMark | null }
  | { type: "gates"; gates: Record<string, GateState> }
  | { type: "view"; view: string; state: ViewState }
  | { type: "mode"; mode: CurationMode }
  | { type: "board"; board: BoardState };

/**
 * Fold one edit into the board without waiting for the next GET, so a click
 * shows at once; the view then re-fetches and replaces the whole state
 * (`board`). Never mutates. A `mark` on a locked (accepted) cell is ignored -
 * the API would refuse it too.
 */
export function reduceBoard(board: BoardState, event: BoardEvent): BoardState {
  switch (event.type) {
    case "board":
      return event.board;
    case "cell":
      return withCell(board, event.cell);
    case "gates":
      return recount({ ...board, exists: true, gates: { ...board.gates, ...event.gates } });
    case "view": {
      const current = board.views?.[event.view];
      if (!current) return board;
      return { ...board, exists: true, views: { ...board.views, [event.view]: { ...current, state: event.state } } };
    }
    case "mode":
      return recount({ ...board, mode: event.mode });
    case "mark": {
      const cell = board.cells[event.key];
      if (!cell || cell.status === "accepted") return board;
      const frames = { ...cell.frames };
      if (event.mark === null) delete frames[String(event.index)];
      else frames[String(event.index)] = event.mark;
      const strip = cell.strip.map((frame) => (frame.index === event.index ? { ...frame, mark: event.mark } : frame));
      return withCell(board, { ...cell, frames, strip });
    }
  }
}

/** A cell or a gate written = the record now exists on disk (`exists`). */
function withCell(board: BoardState, cell: BoardCell): BoardState {
  return recount({ ...board, exists: true, cells: { ...board.cells, [cell.key]: cell } });
}

function recount(board: BoardState): BoardState {
  return { ...board, waiting: waitingOn(board) };
}

/**
 * What the owner has to decide - the API's `waiting_on`, computed again here
 * so an optimistic edit updates the header. Gates first (not in `end` mode),
 * then the rejected and the pending cells that have frames; in `every` mode
 * only the first such cell, the rest are queued behind it.
 */
export function waitingOn(board: Pick<BoardState, "mode" | "gates" | "cells">): Waiting[] {
  const out: Waiting[] = [];
  if (board.mode !== "end") {
    if (board.gates.base !== "accepted") {
      out.push({ kind: "gate", key: "base", why: "accept the base (the concept pick) before any cell renders" });
    }
    if (board.gates.turnaround !== "accepted") {
      out.push({ kind: "gate", key: "turnaround", why: "accept the turnaround before any cell outside s0 renders" });
    }
  }
  const ordered = Object.values(board.cells).sort((a, b) => cellOrder(a.key) - cellOrder(b.key));
  for (const cell of ordered) {
    if (cell.status === "rejected") {
      out.push({ kind: "cell", key: cell.key, why: `rejected - regenerate ${cell.job ?? cell.key}` });
    } else if (cell.status === "pending" && cell.set) {
      out.push({ kind: "cell", key: cell.key, why: `review ${cell.job ?? cell.key}` });
    } else {
      continue;
    }
    if (board.mode === "every") break;
  }
  return out;
}

/** Direction-major, the API's `cell_order`: every action of s0, then s45, ... */
export function cellOrder(key: string): number {
  const [action, direction] = key.split("/") as [string, string];
  const dir = DIRECTIONS.indexOf(direction);
  const act = STUDIO_ACTIONS.findIndex((row) => row.action === action);
  return (dir < 0 ? 99 : dir) * 100 + (act < 0 ? 99 : act);
}

/** The header's one line. */
export function waitingText(waiting: Waiting[]): string {
  if (!waiting.length) return "nothing waiting on you";
  const items = waiting.map((w) => (w.kind === "gate" ? `${w.key} gate` : w.key));
  return `waiting on you: ${items.join(", ")}`;
}

export function gatesText(gates: Record<string, GateState>, mode: CurationMode): string {
  if (mode === "end") return "gates off (end-only review)";
  return `base ${gates.base ?? "pending"} · turnaround ${gates.turnaround ?? "pending"}`;
}

/** The seven views `concept --views` can render, in the table's order
 * (config.CONCEPT_VIEWS); "front" is the base itself. */
export const VIEWS: readonly string[] = ["back", "left", "right", "back_right", "back_left", "front_left", "front_right"];

/** One tile per view of the base the board knows (the multi-view reference
 * feature): the views step under the base gate, in the table's order. */
export interface ViewTile {
  view: string;
  yaw: number;
  state: ViewState;
  url: string | null;
  provisional: boolean;
  /** Click cycles: pending -> accepted -> rejected -> pending. */
  next: ViewState;
}

export function nextViewState(state: ViewState): ViewState {
  if (state === "pending") return "accepted";
  if (state === "accepted") return "rejected";
  return "pending";
}

export function viewTiles(board: Pick<BoardState, "views">): ViewTile[] {
  const views = board.views ?? {};
  return VIEWS.filter((name) => views[name]).map((name) => {
    const row = views[name] as BoardView;
    return { view: name, yaw: row.yaw, state: row.state, url: row.url, provisional: row.provisional, next: nextViewState(row.state) };
  });
}

/** The one line over the views row. */
export function viewsText(board: Pick<BoardState, "views">): string {
  const tiles = viewTiles(board);
  if (!tiles.length) return "no views of the base yet - anim concept --unit <unit> --views renders back, left, right";
  const accepted = tiles.filter((t) => t.state === "accepted").length;
  const provisional = tiles.filter((t) => t.provisional).length;
  const parts = [`${accepted} / ${tiles.length} views accepted`];
  if (provisional) parts.push(`${provisional} provisional (picked for a rerun - your --pick-view supersedes it)`);
  const rejected = tiles.filter((t) => t.state === "rejected").map((t) => t.view);
  if (rejected.length) parts.push(`rejected: ${rejected.join(", ")} (the turnaround falls back past them)`);
  return parts.join(" · ");
}

/** One entry per planned direction of the turnaround, in compass order (the
 * plan's order, s0 first), whether or not it rendered - the thing the
 * `turnaround` gate accepts (the turnaround feature). */
export interface TurnaroundTile {
  direction: string;
  job: string;
  status: string;
  still: string | null;
  error: string | null;
  /** The reference the still used: "back", "front (fallback)" ...; null before it ran. */
  reference: string | null;
  referenceUrl: string | null;
}

export function referenceText(ref: { view: string | null; fallback: boolean; provisional: boolean } | null | undefined): string | null {
  if (!ref || !ref.view) return null;
  const notes = [ref.fallback ? "fallback" : null, ref.provisional ? "provisional" : null].filter(Boolean);
  return notes.length ? `${ref.view} (${notes.join(", ")})` : ref.view;
}

export function turnaroundTiles(state: Pick<TurnaroundState, "plan" | "directions">): TurnaroundTile[] {
  return state.plan.map((direction) => {
    const row = state.directions[direction];
    return {
      direction,
      job: row?.job ?? "",
      status: row?.status ?? "not run",
      still: row?.still ?? null,
      error: row?.error ?? null,
      reference: referenceText(row?.reference),
      referenceUrl: row?.reference?.url ?? null,
    };
  });
}

/** The header's one line about the turnaround: how many stills, the seed, the
 * seconds, and the spread of the silhouette so a broken facing reads as a number. */
export function turnaroundText(state: Pick<TurnaroundState, "done" | "of" | "seed" | "directions" | "identity"> | null): string {
  if (!state) return "no turnaround yet - anim turnaround <unit> renders the eight idle stills";
  const done = Object.values(state.directions).filter((d) => d.status === "done");
  const seconds = done.map((d) => d.seconds ?? 0).reduce((a, b) => a + b, 0);
  const per = done.length ? Math.round(seconds / done.length) : 0;
  const failed = Object.values(state.directions).filter((d) => d.status === "failed").map((d) => d.direction);
  const parts = [`${state.done} / ${state.of} facings`, `seed ${state.seed ?? "?"}`];
  if (done.length) parts.push(`${per} s per facing`);
  const h = state.identity?.height?.spread;
  const w = state.identity?.width?.spread;
  if (typeof h === "number" && typeof w === "number") parts.push(`silhouette spread ${h} px tall, ${w} px wide`);
  if (failed.length) parts.push(`failed: ${failed.join(", ")}`);
  return parts.join(" · ");
}

/** The counts under the grid. */
export function tally(board: Pick<BoardState, "cells">): { cells: number; accepted: number; rejected: number; pending: number; marked: number } {
  const cells = Object.values(board.cells);
  return {
    cells: cells.length,
    accepted: cells.filter((c) => c.status === "accepted").length,
    rejected: cells.filter((c) => c.status === "rejected").length,
    pending: cells.filter((c) => c.status === "pending").length,
    marked: cells.reduce((n, c) => n + Object.keys(c.frames).length, 0),
  };
}
