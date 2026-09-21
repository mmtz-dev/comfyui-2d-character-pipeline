import { describe, expect, it } from "vitest";
import type { BoardCell, BoardState, JobRow } from "./api";
import {
  cellKey,
  cellOfJob,
  cellOrder,
  cellsFromJobs,
  cycleMark,
  gatesText,
  gridOf,
  reduceBoard,
  tally,
  waitingOn,
  waitingText,
} from "./board";

const job = (name: string, slug: string, action: string, direction = "s0"): JobRow => ({
  name, slug, action, direction, frames: 13, stages: ["extract", "pose"],
  rerender_requested: false, updated: "2026-09-21T12:00:00+00:00", url: `/jobs/${name}`,
});

const cell = (key: string, job: string, extra: Partial<BoardCell> = {}): BoardCell => {
  const [action, direction] = key.split("/") as [string, string];
  return {
    key, action, direction, verb: action === "attack" ? "strike" : action, job,
    status: "pending", frames: {}, seed: null, regenerated_from: null, decided_at: null,
    stages: ["extract", "pose", "pick"], stage: "render", current_seed: null,
    rerender_requested: false, set: "frames", picks: [0, 4, 9],
    strip: [0, 4, 9].map((index) => ({ index, url: `/jobs/${job}/frames/frames/${index}.png`, mark: null })),
    ...extra,
  };
};

// What GET /curation/knight answers for the test suite's three cells.
const board: BoardState = {
  unit: "knight", mode: "gates", origin: null, exists: false, path: "/work/knight_curation.json",
  gates: { base: "accepted", turnaround: "accepted" },
  actions: ["attack", "idle", "walk", "run", "jump"],
  directions: ["s0", "s45"],
  cells: {
    "attack/s0": cell("attack/s0", "knight_strike"),
    "idle/s0": cell("idle/s0", "knight_idle"),
    "walk/s45": cell("walk/s45", "knight_walk_s45", { stage: "clip" }),
  },
  waiting: [],
};

describe("cells from the jobs list", () => {
  it("names a job's cell by the studio action and the direction", () => {
    expect(cellOfJob("knight", job("knight_strike", "knight", "strike"))).toBe("attack/s0");
    expect(cellOfJob("knight", job("knight_walk_s45", "knight", "walk", "s45"))).toBe("walk/s45");
    expect(cellOfJob("knight", job("orc_strike", "orc", "strike"))).toBeNull();
    expect(cellOfJob("knight", job("knight_base", "knight", ""))).toBeNull();
    expect(cellOfJob("knight", job("bakeoff_knight_attack_V-kaykit", "knight", "strike"))).toBeNull();
    expect(cellKey("attack", "sideways")).toBeNull();
  });
  it("prefers the canonical job name when two jobs share a cell", () => {
    const cells = cellsFromJobs("knight", [
      job("knight_strike_phases", "knight", "strike"),
      job("knight_strike", "knight", "strike"),
      job("knight_idle", "knight", "idle"),
      job("orc_idle", "orc", "idle"),
    ]);
    expect(cells).toEqual({ "attack/s0": "knight_strike", "idle/s0": "knight_idle" });
  });
});

describe("gridOf", () => {
  it("is five action rows by the unit's directions, null where there is no job", () => {
    const grid = gridOf(board);
    expect(grid.directions).toEqual(["s0", "s45"]);
    expect(grid.rows.map((r) => r.action)).toEqual(["attack", "idle", "walk", "run", "jump"]);
    expect(grid.rows[0]!.cells.map((c) => c?.job ?? null)).toEqual(["knight_strike", null]);
    expect(grid.rows[2]!.cells.map((c) => c?.job ?? null)).toEqual([null, "knight_walk_s45"]);
    expect(grid.rows[4]!.cells).toEqual([null, null]);
  });
  it("keeps the API's direction order whatever order the state lists", () => {
    expect(gridOf({ cells: {}, directions: ["t0", "s45", "s0"] }).directions).toEqual(["s0", "s45", "t0"]);
  });
});

describe("cycleMark", () => {
  it("goes none -> accept -> reject -> none", () => {
    expect(cycleMark(null)).toBe("accept");
    expect(cycleMark("accept")).toBe("reject");
    expect(cycleMark("reject")).toBeNull();
  });
});

describe("reduceBoard", () => {
  it("marks a frame in the cell and its strip, without mutating", () => {
    const next = reduceBoard(board, { type: "mark", key: "attack/s0", index: 4, mark: "reject" });
    expect(next.cells["attack/s0"]!.frames).toEqual({ "4": "reject" });
    expect(next.cells["attack/s0"]!.strip[1]!.mark).toBe("reject");
    expect(board.cells["attack/s0"]!.frames).toEqual({});
    const cleared = reduceBoard(next, { type: "mark", key: "attack/s0", index: 4, mark: null });
    expect(cleared.cells["attack/s0"]!.frames).toEqual({});
  });
  it("says the record exists once a cell or a gate was written", () => {
    expect(board.exists).toBe(false);
    expect(reduceBoard(board, { type: "cell", cell: board.cells["idle/s0"]! }).exists).toBe(true);
    expect(reduceBoard(board, { type: "gates", gates: { base: "pending" } }).exists).toBe(true);
    expect(reduceBoard(board, { type: "mode", mode: "end" }).exists).toBe(false);
  });
  it("ignores a mark on an accepted cell - it is locked", () => {
    const locked = reduceBoard(board, { type: "cell", cell: { ...board.cells["attack/s0"]!, status: "accepted" } });
    const same = reduceBoard(locked, { type: "mark", key: "attack/s0", index: 0, mark: "reject" });
    expect(same).toBe(locked);
  });
  it("recomputes what is waiting after a cell, a gate or the mode changes", () => {
    expect(waitingOn(board).map((w) => w.key)).toEqual(["attack/s0", "idle/s0", "walk/s45"]);
    const accepted = reduceBoard(board, { type: "cell", cell: { ...board.cells["attack/s0"]!, status: "accepted" } });
    expect(accepted.waiting.map((w) => w.key)).toEqual(["idle/s0", "walk/s45"]);
    const gated = reduceBoard(accepted, { type: "gates", gates: { turnaround: "pending" } });
    expect(gated.waiting[0]).toEqual({ kind: "gate", key: "turnaround", why: expect.stringContaining("turnaround") });
    const end = reduceBoard(gated, { type: "mode", mode: "end" });
    expect(end.waiting.map((w) => w.kind)).toEqual(["cell", "cell"]);
    const every = reduceBoard(gated, { type: "mode", mode: "every" });
    expect(every.waiting.map((w) => w.key)).toEqual(["turnaround", "idle/s0"]);
  });
  it("a rejected cell waits for a regenerate; a cell without frames waits for nothing", () => {
    const rejected = reduceBoard(board, { type: "cell", cell: { ...board.cells["idle/s0"]!, status: "rejected" } });
    expect(rejected.waiting.find((w) => w.key === "idle/s0")?.why).toContain("regenerate knight_idle");
    const empty = reduceBoard(board, { type: "cell", cell: { ...board.cells["idle/s0"]!, set: null, strip: [] } });
    expect(empty.waiting.map((w) => w.key)).toEqual(["attack/s0", "walk/s45"]);
  });
});

describe("the header lines", () => {
  it("order cells direction-major, like the API", () => {
    const keys = ["walk/s45", "jump/s0", "attack/s0", "attack/t0"].sort((a, b) => cellOrder(a) - cellOrder(b));
    expect(keys).toEqual(["attack/s0", "jump/s0", "walk/s45", "attack/t0"]);
  });
  it("say what is waiting and how the gates stand", () => {
    expect(waitingText([])).toBe("nothing waiting on you");
    expect(waitingText([{ kind: "gate", key: "base", why: "" }, { kind: "cell", key: "idle/s0", why: "" }]))
      .toBe("waiting on you: base gate, idle/s0");
    expect(gatesText({ base: "accepted", turnaround: "pending" }, "gates")).toBe("base accepted · turnaround pending");
    expect(gatesText({ base: "pending", turnaround: "pending" }, "end")).toContain("off");
  });
  it("count the cells", () => {
    const marked = reduceBoard(board, { type: "mark", key: "idle/s0", index: 0, mark: "accept" });
    expect(tally(marked)).toEqual({ cells: 3, accepted: 0, rejected: 0, pending: 3, marked: 1 });
  });
});
