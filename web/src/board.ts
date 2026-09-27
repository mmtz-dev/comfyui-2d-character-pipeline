// The curation board's pure parts: the grid out of the board state, the
// reducer the view folds edits into, and the header lines. No DOM, no
// fetch - the tests run it in Node; board-view.ts draws it.

import type { BoardCell, BoardState, BoardView, CurationMode, FrameMark, GateState, JobRow, ViewState, Waiting } from "./api";

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
  gates: "gates: clips wait for the base to be accepted, and a matrix direction for its view; review in a batch",
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

/** The views step as waiting entries - the API's `waiting_views`: the slots
 * nobody has drawn at all first, then the ones that need a decision (an
 * attempt waiting for a pick, a picked view waiting for a verdict). */
export function waitingViews(unit: string, views: Record<string, BoardView>): Waiting[] {
  const tiles = viewTiles({ unit, views });
  const because = (tile: ViewTile) => (tile.wantedBy.length ? `; ${tile.wantedBy.join(", ")} wants it` : "");
  const undrawn = tiles.filter((t) => t.missing && !t.attempts)
    .map((t) => ({ kind: "view" as const, key: t.view, why: `no ${t.view} view of the base yet - render it on the Concept tab (\`anim concept --unit ${unit} --views ${t.view}\`)${because(t)}` }));
  const decide = tiles.filter((t) => (t.missing && t.attempts) || (!t.missing && t.state === "pending"))
    .map((t) => ({
      kind: "view" as const, key: t.view,
      why: t.missing
        ? `${t.attempts} attempt${t.attempts === 1 ? "" : "s"} at the ${t.view} view and none picked - pick one on the Concept tab (\`anim concept --unit ${unit} --pick-view ${t.view}=NN\`)${because(t)}`
        : `judge the ${t.view} view of the base - accept it or reject it`,
    }));
  return [...undrawn, ...decide];
}

/**
 * What the owner has to decide - the API's `waiting_on`, computed again here
 * so an optimistic edit updates the header. In pipeline order: the gates (not
 * in `end` mode), the views of the base and its part palettes, then the
 * rejected and the pending cells that have frames; in
 * `every` mode only the first such cell, the rest are queued behind it.
 */
export function waitingOn(board: Pick<BoardState, "mode" | "gates" | "cells"> & Partial<BoardState>): Waiting[] {
  const out: Waiting[] = [];
  const unit = board.unit ?? "<unit>";
  if (board.mode !== "end") {
    if (board.gates.base !== "accepted") {
      out.push({ kind: "gate", key: "base", why: "accept the base (the concept pick) before any cell renders" });
    }
    // Only when the API sent the view slots: one from before the eight-slot
    // ring sends none, and guessing at them would make an optimistic line
    // disagree with the GET that follows it.
    if (board.exists && board.views && Object.keys(board.views).length) {
      out.push(...waitingViews(unit, board.views));
      if (board.base_parts === false) {
        const job = board.base_parts_job ?? `${unit}_base`;
        out.push({ kind: "parts", key: job, why: `no part palettes yet - \`anim post ${job} --parts\` measures them off the accepted base (CPU, no GPU); every direction locks its colours to them` });
      }
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

/** The gates line. A gate with nothing behind it says so instead of printing
 * a state the owner should not act on. */
export function gatesText(gates: Record<string, GateState>, mode: CurationMode,
                          blocked: Record<string, string | null> = {}): string {
  if (mode === "end") return "gates off (end-only review)";
  const one = (gate: string) => {
    const state = gates[gate] ?? "pending";
    return state !== "accepted" && blocked[gate] ? `${gate} has nothing to accept` : `${gate} ${state}`;
  };
  return GATES.map(one).join(" · ");
}

/** The curation gates (curation.GATES). The views of the base, accepted one
 * by one, are the rest of the human gate; there is no second gate. */
export const GATES: readonly string[] = ["base"];

/** The eight views `concept --views` can render, in the table's order
 * (config.CONCEPT_VIEWS); "front" is the base itself and is never one. */
export const VIEWS: readonly string[] = ["back", "left", "right", "top", "back_right", "back_left", "front_left", "front_right"];

/**
 * One tile per view slot (the multi-view reference feature): ALL EIGHT, in
 * the table's order, whether or not the base carries the file - a slot with
 * no file is the ring's third state and the one that usually matters, since
 * a direction of the matrix whose view is missing is blocked. `how` is the line a
 * missing slot shows: what to run to fill it.
 */
export interface ViewTile {
  view: string;
  yaw: number;
  state: ViewState;
  url: string | null;
  provisional: boolean;
  /** The base carries no file for this view: nothing to judge yet. */
  missing: boolean;
  /** Candidates `concept --views` has drawn for it. Missing with attempts is
   * a pick waiting to be made; missing with none is GPU time not yet spent. */
  attempts: number;
  /** The matrix directions that ask for this view first ("s0" wants "right"). */
  wantedBy: string[];
  /** How to fill a missing slot; null when the slot has a file. */
  how: string | null;
  /** The mirrored view feature: the partner this slot could be derived from
   * (null for back and top, which sit on the symmetry plane), and whether the
   * picture on the base already IS that flip. The ring marks the second. */
  mirrorOf: string | null;
  mirrored: boolean;
  mirroredFrom: string | null;
  /** Click cycles: pending -> accepted -> rejected -> pending. A missing
   * slot has nothing to cycle, and the view renders no button for it. */
  next: ViewState;
}

export function nextViewState(state: ViewState): ViewState {
  if (state === "pending") return "accepted";
  if (state === "accepted") return "rejected";
  return "pending";
}

export function viewTiles(board: Pick<BoardState, "views"> & Partial<BoardState>): ViewTile[] {
  const views = board.views ?? {};
  const unit = board.unit ?? "<unit>";
  return VIEWS.map((name) => {
    const row: BoardView | undefined = views[name];
    // No row at all is the same gap as a row with no file: an API from
    // before the eight-slot ring simply left the empty slots out.
    const missing = !row || row.missing === true || (row.missing === undefined && !row.url);
    const attempts = row?.attempts ?? 0;
    return {
      view: name,
      yaw: row?.yaw ?? 0,
      state: row?.state ?? "pending",
      url: row?.url ?? null,
      provisional: row?.provisional ?? false,
      missing,
      attempts,
      wantedBy: row?.wanted_by ?? [],
      mirrorOf: row?.mirror_of ?? null,
      mirrored: !missing && (row?.mirrored ?? false),
      mirroredFrom: row?.mirrored_from ?? null,
      how: !missing ? null
        : attempts ? `anim concept --unit ${unit} --pick-view ${name}=NN`
        : `anim concept --unit ${unit} --views ${name}`,
      next: nextViewState(row?.state ?? "pending"),
    };
  });
}

/** The one line over the views ring: how many slots are filled, how many of
 * those are accepted, and - the point of the line - which are missing. */
export function viewsText(board: Pick<BoardState, "views"> & Partial<BoardState>): string {
  const tiles = viewTiles(board);
  const have = tiles.filter((t) => !t.missing);
  const unpicked = tiles.filter((t) => t.missing && t.attempts).map((t) => t.view);
  const missing = tiles.filter((t) => t.missing && !t.attempts).map((t) => t.view);
  const accepted = have.filter((t) => t.state === "accepted").length;
  const parts = [`${have.length} / ${tiles.length} views on the base, ${accepted} accepted`];
  if (unpicked.length) parts.push(`drawn, not picked: ${unpicked.join(", ")} (pick one on the Concept tab)`);
  if (missing.length) parts.push(`missing: ${missing.join(", ")} (the Concept tab renders them)`);
  const provisional = have.filter((t) => t.provisional).length;
  if (provisional) parts.push(`${provisional} provisional (picked for a rerun - your --pick-view supersedes it)`);
  const rejected = have.filter((t) => t.state === "rejected").map((t) => t.view);
  if (rejected.length) parts.push(`rejected: ${rejected.join(", ")} (the matrix refuses the facings that want them)`);
  // The mirrored view feature: named on the line because a flip is a different
  // KIND of view - the same pixels as its partner, the other hand - and the
  // owner reading "6 / 8 views" deserves to know two of them cost no GPU.
  const mirrored = have.filter((t) => t.mirrored)
    .map((t) => `${t.view} ← ${t.mirroredFrom ?? "its partner"}`);
  if (mirrored.length) parts.push(`mirrored: ${mirrored.join(", ")} (flipped, not generated)`);
  return parts.join(" · ");
}

/** One button per gate, decided here rather than in the DOM: a gate whose
 * subject does not exist yet (`gate_blocked`, the API's `gate_blockers`) is
 * not acceptable, so the button is disabled and carries the reason. An
 * ACCEPTED gate stays clickable even when blocked - reopening it is how the
 * owner undoes a judgement that was made over an empty set. */
export interface GateButton {
  gate: string;
  state: GateState;
  label: string;
  title: string;
  blocked: string | null;
  disabled: boolean;
}

export function gateButtons(board: Pick<BoardState, "gates" | "mode"> & Partial<BoardState>): GateButton[] {
  return GATES.map((gate) => {
    const state = board.gates[gate] ?? "pending";
    const blocked = board.gate_blocked?.[gate] ?? null;
    const wouldAccept = state !== "accepted";
    const disabled = board.mode === "end" || (wouldAccept && blocked !== null);
    return {
      gate,
      state,
      label: blocked && wouldAccept ? `${gate}: nothing to accept` : `${gate}: ${state}`,
      title: blocked && wouldAccept ? blocked
        : state === "accepted" ? `${gate} accepted - click to reopen${blocked ? ` (${blocked})` : ""}`
        : `accept the ${gate} gate`,
      blocked,
      disabled,
    };
  });
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

/** The part-palette job the matrix needs, when it is missing.
 *
 * `post --parts` is a CPU job (SAM over the accepted base, ~45 s) and it is the
 * one thing standing between a unit with accepted views and its matrix cells,
 * which all lock their colours to it. The Pipeline tab's `post` sends only
 * {job, source}, so pressing it would post the sprite and skip the
 * measurement. Null when the base already has them - there is nothing to offer.
 */
export interface PartsOffer {
  job: string;
  label: string;
  title: string;
  confirm: string;
}

export function partsOffer(
  board: Pick<BoardState, "unit" | "gates"> & Partial<BoardState>,
): PartsOffer | null {
  if (!board.unit) return null;
  // `undefined` is an API that predates the field, and guessing "missing" there
  // would offer the job to every unit including ones that have it.
  if (board.base_parts !== false) return null;
  if (board.gates.base !== "accepted") return null;   // nothing to measure yet
  const job = board.base_parts_job ?? `${board.unit}_base`;
  return {
    job,
    label: "measure part palettes",
    title: `anim post ${job} --parts: SAM over the accepted base, once, to record what colour `
      + `each part is.\nA CPU job - it does not touch the card - and the matrix renders `
      + `no cell without it.`,
    confirm: `measure ${board.unit}'s part palettes from its accepted base (${job}).\n`
      + `This runs anim post ${job} --parts: about 45 s on the CPU, no GPU. Every matrix `
      + `cell then locks its colours to the result, which is what keeps the same colours at `
      + `every facing.\n\nContinue?`,
  };
}

/** The args `POST /jobs {stage: "post"}` takes for it. */
export function partsArgs(offer: Pick<PartsOffer, "job">): Record<string, unknown> {
  return { job: offer.job, parts: true };
}
