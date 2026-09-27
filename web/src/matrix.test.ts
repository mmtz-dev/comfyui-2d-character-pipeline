import { describe, expect, it } from "vitest";
import type { MatrixCell, MatrixState, RunRecord } from "./api";
import { cellOrder } from "./board";
import {
  ACTIONS,
  SIDE_DIRECTIONS,
  actionReason,
  argsFor,
  blockedKeys,
  blockedText,
  cellClass,
  closureText,
  costText,
  defaultSelection,
  directionReason,
  frameAt,
  generateBlocked,
  gridOf,
  holdReason,
  phaseChips,
  runnableKeys,
  selectedKeys,
  selectionText,
  setAllDirs,
  toggleAction,
  toggleDirection,
  type Selection,
} from "./matrix";

// The two things that block a cell in this fixture: the UAL ships no whole
// jump clip, and s45 has no accepted view of the base. Everything else is
// runnable, so a test can tell "blocked" from "not selected".
const NO_CLIP = "the quaternius rig ships no whole jump clip (the UAL has Jump_Start / Jump_Loop / Jump_Land) - config.STUDIO_ACTIONS";
const NO_VIEW = "s45: no accepted view for it - wanted back_right (not on the base), back (pending); the matrix does not render a cell from the front-view fallback";

const cell = (key: string, extra: Partial<MatrixCell> = {}): MatrixCell => {
  const [action, direction] = key.split("/") as [string, string];
  const verb = action === "attack" ? "strike" : action;
  const blocked = action === "jump" ? NO_CLIP : direction === "s45" ? NO_VIEW : null;
  return {
    key, action, verb, direction, yaw: Number(direction.slice(1)),
    job: direction === "s0" ? `knight_${verb}` : `knight_${verb}_${direction}`,
    clip: action === "jump" ? null : `Clip_${verb}`,
    status: blocked ? "blocked" : "not run",
    runnable: !blocked, reason: blocked,
    view: blocked ? null : "back", wanted: ["back_right", "back"],
    frames: 0, strip: [], phases: [], fps: 12,
    loop: action === "idle" || action === "walk" || action === "run",
    closure: null, seconds: null, gpu_seconds: null, seed: null, error: null, curation: null,
    ...extra,
  };
};

// What GET /matrix/knight answers: every planned cell of the whole grid,
// whether or not anything has run - the tab greys on first load.
const matrix = (extra: Record<string, Partial<MatrixCell>> = {}): MatrixState => {
  const cells: Record<string, MatrixCell> = {};
  for (const direction of SIDE_DIRECTIONS) {
    for (const { action } of ACTIONS) {
      const key = `${action}/${direction}`;
      cells[key] = cell(key, extra[key] ?? {});
    }
  }
  return {
    unit: "knight", schema: "anim.matrix/1", elevation: "side", seed: 5417,
    rig: { name: "quaternius", glb: "/assets/rigs/ual.glb" },
    actions: ACTIONS.map((row) => row.action), directions: [...SIDE_DIRECTIONS],
    cells,
    max_cells: 8, selectable: Object.values(cells).filter((c) => c.runnable).length,
    estimate_seconds: 85,
    summary: null, updated: "2026-09-22T20:00:00+00:00", path: "/work/knight_matrix.json",
  };
};

const run = (extra: Partial<RunRecord> = {}): RunRecord => ({
  id: "run-7", stage: "clip", argv: ["anim", "clip", "knight_strike"], gpu: true, status: "running",
  percent: 40, started: "2026-09-22T20:00:00+00:00", finished: null, returncode: null, pid: 9,
  log_lines: 3, url: "/runs/run-7", events: "/runs/run-7/events", ...extra,
});

describe("the selection", () => {
  it("starts as every action at s0 alone - the stage's own default", () => {
    const selection = defaultSelection(matrix());
    expect(selection.actions).toEqual(["attack", "idle", "walk", "run", "jump"]);
    expect(selection.directions).toEqual(["s0"]);
    expect(selection.allDirs).toBe(false);
    expect(defaultSelection(null).directions).toEqual(["s0"]);
  });

  it("toggles an action without mutating, and keeps the table's order", () => {
    const before = defaultSelection(matrix());
    const without = toggleAction(before, "idle");
    expect(without.actions).toEqual(["attack", "walk", "run", "jump"]);
    expect(before.actions).toEqual(["attack", "idle", "walk", "run", "jump"]);
    expect(toggleAction(without, "idle").actions).toEqual(["attack", "idle", "walk", "run", "jump"]);
  });

  it("toggles a direction without mutating, in compass order", () => {
    const before = defaultSelection(matrix());
    const two = toggleDirection(before, "s270");
    expect(two.directions).toEqual(["s0", "s270"]);
    expect(before.directions).toEqual(["s0"]);
    expect(toggleDirection(two, "s0").directions).toEqual(["s270"]);
  });

  it("the all-8 toggle is the eight, and off is s0 again", () => {
    const all = setAllDirs(defaultSelection(matrix()), true);
    expect(all.directions).toEqual([...SIDE_DIRECTIONS]);
    expect(all.allDirs).toBe(true);
    expect(setAllDirs(all, false)).toEqual({ actions: all.actions, directions: ["s0"], allDirs: false });
  });

  it("ticking the eighth direction by hand is the same as the toggle", () => {
    let selection = defaultSelection(matrix());
    for (const direction of SIDE_DIRECTIONS.slice(1)) selection = toggleDirection(selection, direction);
    expect(selection.allDirs).toBe(true);
    expect(toggleDirection(selection, "s45").allDirs).toBe(false);
  });

  it("names the selected cells direction-major, like curation.cell_order", () => {
    const selection: Selection = { actions: ["walk", "attack"], directions: ["s90", "s0"], allDirs: false };
    expect(selectedKeys(selection)).toEqual(["attack/s0", "walk/s0", "attack/s90", "walk/s90"]);
    const sorted = selectedKeys(selection).slice().sort((a, b) => cellOrder(a) - cellOrder(b));
    expect(selectedKeys(selection)).toEqual(sorted);
  });

  it("splits the selection into what would render and what is blocked", () => {
    const state = matrix();
    const selection = setAllDirs(defaultSelection(state), true);
    expect(selectedKeys(selection)).toHaveLength(40);
    // jump has no clip on this rig (5 of the 8), s45 has no accepted view (4),
    // and jump/s45 is both - so 40 - 8 - 5 + 1 blocked.
    expect(blockedKeys(selection, state)).toHaveLength(12);
    expect(runnableKeys(selection, state)).toHaveLength(28);
    expect(runnableKeys(selection, state)).not.toContain("jump/s0");
    expect(runnableKeys(selection, null)).toEqual([]);
  });
});

describe("argsFor", () => {
  it("reads the way the owner would have typed the command", () => {
    const every = defaultSelection(matrix());
    expect(argsFor("knight", every)).toEqual({ unit: "knight", dirs: "s0" });
    expect(argsFor("knight", toggleAction(every, "jump")))
      .toEqual({ unit: "knight", actions: "attack,idle,walk,run", dirs: "s0" });
    expect(argsFor("knight", setAllDirs(every, true))).toEqual({ unit: "knight", all_dirs: true });
    expect(argsFor("knight", { actions: ["attack"], directions: ["s0", "s90"], allDirs: false }))
      .toEqual({ unit: "knight", actions: "attack", dirs: "s0,s90" });
  });
});

describe("the grid", () => {
  it("is always five actions by eight facings, with the selected ones flagged", () => {
    const state = matrix();
    const grid = gridOf(state, defaultSelection(state));
    expect(grid.directions).toEqual([...SIDE_DIRECTIONS]);
    expect(grid.rows.map((row) => row.action)).toEqual(["attack", "idle", "walk", "run", "jump"]);
    expect(grid.rows[0]!.verb).toBe("strike");
    expect(grid.rows.flatMap((row) => row.cells)).toHaveLength(40);
    expect(grid.rows.flatMap((row) => row.cells).filter((entry) => entry.cell === null)).toEqual([]);
    expect(grid.rows[0]!.cells[0]).toMatchObject({ key: "attack/s0", direction: "s0", selected: true });
    expect(grid.rows[0]!.cells[1]!.selected).toBe(false);
    expect(gridOf(null, defaultSelection(null)).rows[0]!.cells[0]!.cell).toBeNull();
  });

  it("greys a blocked cell with the Concept tab's own class", () => {
    const state = matrix();
    expect(cellClass(state.cells["jump/s0"]!, false)).toBe("locked");
    expect(cellClass(state.cells["attack/s0"]!, true)).toBe("ready selected");
    expect(cellClass({ ...state.cells["attack/s0"]!, status: "done" }, false)).toBe("done");
    expect(cellClass({ ...state.cells["attack/s0"]!, status: "kept" }, false)).toBe("done");
    expect(cellClass({ ...state.cells["attack/s0"]!, status: "failed" }, false)).toBe("failed");
    expect(cellClass(null, false)).toBe("missing");
  });

  it("disables a row or a column no cell of which can render, with its reason", () => {
    const state = matrix();
    expect(actionReason(state, "jump")).toBe(NO_CLIP);
    expect(actionReason(state, "attack")).toBeNull();
    expect(directionReason(state, "s45")).toBe(NO_VIEW);
    expect(directionReason(state, "s0")).toBeNull();
    expect(directionReason(null, "s0")).toBeNull();
  });
});

describe("the #277 hold", () => {
  it("names the issue above the cap and says nothing at it", () => {
    const state = matrix();
    const seven: Selection = { actions: ["attack"], directions: [...SIDE_DIRECTIONS], allDirs: true };
    expect(runnableKeys(seven, state)).toHaveLength(7);
    expect(holdReason(seven, state)).toBeNull();
    const fourteen = toggleAction(seven, "idle");
    expect(runnableKeys(fourteen, state)).toHaveLength(14);
    expect(holdReason(fourteen, state)).toContain("#277");
    expect(holdReason(fourteen, state)).toContain("held to 8");
    expect(holdReason(fourteen, { ...state, max_cells: 0 })).toBeNull();
  });
});

describe("generateBlocked", () => {
  const state = matrix();
  it("gives the busy reason first, then the empty selection, then all-blocked", () => {
    const empty: Selection = { actions: [], directions: [], allDirs: false };
    const jumpOnly: Selection = { actions: ["jump"], directions: ["s0"], allDirs: false };
    expect(generateBlocked(empty, state, run())).toContain("GPU is busy with clip run-7");
    expect(generateBlocked(empty, state, null)).toContain("nothing is selected");
    expect(generateBlocked(jumpOnly, state, null)).toContain(NO_CLIP);
    expect(generateBlocked(defaultSelection(state), state, null)).toBeNull();
  });
  it("lets a finished run through, and asks for a unit before anything else", () => {
    expect(generateBlocked(defaultSelection(state), state, run({ status: "done" }))).toBeNull();
    expect(generateBlocked(defaultSelection(state), null, null)).toBe("pick a unit first");
  });
  it("falls through to the cap when nothing else stands in the way", () => {
    const wide: Selection = { actions: ["attack", "idle"], directions: [...SIDE_DIRECTIONS], allDirs: true };
    expect(generateBlocked(wide, state, null)).toContain("#277");
  });
});

describe("the lines the tab shows", () => {
  it("count the selection and what it costs", () => {
    const state = matrix();
    const selection = defaultSelection(state);
    expect(selectionText(selection, state)).toBe("5 actions × 1 direction = 5 cells, 4 to render, 1 blocked");
    const cost = costText(selection, state);
    expect(cost).toContain("4 cells");
    expect(cost).toContain("one seed (5417)");
    expect(cost).toContain("~6 min of GPU, warm");
    expect(cost).toContain("rig → clip → pick --phases → post per cell");
    expect(costText(selection, { ...state, seed: null })).toContain("derived from the base on the first run");
    expect(costText({ actions: ["jump"], directions: ["s0"], allDirs: false }, state))
      .toContain("no GPU cost");
  });

  it("say why the grid is grey", () => {
    const state = matrix();
    expect(blockedText(null)).toContain("Pick a unit");
    const text = blockedText(state);
    expect(text).toContain("28 of 40 cells can be generated");
    expect(text).not.toContain("turnaround");
    expect(text).toContain("12 blocked, first:");
    const open = { ...state, cells: Object.fromEntries(Object.entries(state.cells).map(([key, row]) =>
      [key, { ...row, runnable: true, reason: null }])), selectable: 40 };
    expect(blockedText(open)).toBe("all 40 cells can be generated");
    const shut = { ...state, selectable: 0 };
    expect(blockedText(shut)).toContain("nothing can be generated yet");
  });
});

describe("a finished cell", () => {
  const done = cell("attack/s0", {
    status: "done", frames: 12, fps: 12,
    strip: Array.from({ length: 12 }, (_, i) => `/jobs/knight_strike/frames/post/${i}.png`),
    phases: ["start", "start", "wind-up", "wind-up", "action", "action", "action",
             "wind-down", "wind-down", "end", "end", null],
    gpu_seconds: 18.2, seconds: 81.4, seed: 5417,
  });

  it("chips the distinct phase tags and remembers which frames they cover", () => {
    const chips = phaseChips(done);
    expect(chips.map((chip) => chip.tag)).toEqual(["start", "wind-up", "action", "wind-down", "end"]);
    expect(chips[2]!.frames).toEqual([4, 5, 6]);
    expect(phaseChips(null)).toEqual([]);
    expect(phaseChips(cell("idle/s0"))).toEqual([]);
  });

  it("says whether a loop closes, and nothing when it cannot be measured", () => {
    const loop = (extra: Partial<NonNullable<MatrixCell["closure"]>>) =>
      cell("idle/s0", { status: "done", closure: { first: 0, last: 5, metric: "pose", distance: 0.31, tolerance: 0.12, ok: false, ...extra } });
    expect(closureText(loop({}))).toBe("loop does not close: 0.31 > 0.12");
    expect(closureText(loop({ distance: 0.05, ok: true }))).toBe("loop closes: 0.05 ≤ 0.12");
    expect(closureText(loop({ metric: "none", distance: null, ok: null }))).toBeNull();
    expect(closureText(done)).toBeNull();
    expect(closureText(null)).toBeNull();
  });

  it("wraps the inline player round the frame count at the cell's fps", () => {
    expect(frameAt(0, 12, 5)).toBe(0);
    expect(frameAt(500, 12, 5)).toBe(1);
    expect(frameAt(1000, 12, 5)).toBe(2);
    expect(frameAt(5000, 12, 5)).toBe(0);
    expect(frameAt(0, 12, 0)).toBe(0);
    expect(frameAt(1000, 0, 5)).toBe(2);
  });
});
