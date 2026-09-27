import { describe, expect, it } from "vitest";
import type { BoardCell, BoardState, BoardView, JobRow } from "./api";
import {
  GATES,
  VIEWS,
  cellKey,
  cellOfJob,
  cellOrder,
  cellsFromJobs,
  cycleMark,
  gateButtons,
  gatesText,
  gridOf,
  partsArgs,
  partsOffer,
  nextViewState,
  reduceBoard,
  tally,
  viewTiles,
  viewsText,
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
  gates: { base: "accepted" },
  views: {},
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
    const gated = reduceBoard(accepted, { type: "gates", gates: { base: "pending" } });
    expect(gated.waiting[0]).toEqual({ kind: "gate", key: "base", why: expect.stringContaining("base") });
    const end = reduceBoard(gated, { type: "mode", mode: "end" });
    expect(end.waiting.map((w) => w.kind)).toEqual(["cell", "cell"]);
    const every = reduceBoard(gated, { type: "mode", mode: "every" });
    expect(every.waiting.map((w) => w.key)).toEqual(["base", "idle/s0"]);
    // An older record's `turnaround` key is not a gate the board asks about.
    const old = reduceBoard(accepted, { type: "gates", gates: { turnaround: "pending" } });
    expect(old.waiting.map((w) => w.kind)).not.toContain("gate");
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
    expect(GATES).toEqual(["base"]);
    expect(gatesText({ base: "accepted" }, "gates")).toBe("base accepted");
    expect(gatesText({ base: "accepted", turnaround: "pending" }, "gates")).toBe("base accepted");
    expect(gatesText({ base: "pending" }, "end")).toContain("off");
  });
  it("count the cells", () => {
    const marked = reduceBoard(board, { type: "mark", key: "idle/s0", index: 0, mark: "accept" });
    expect(tally(marked)).toEqual({ cells: 3, accepted: 0, rejected: 0, pending: 3, marked: 1 });
  });
});

describe("the views of the base under the base gate", () => {
  const view = (name: string, state: "pending" | "accepted" | "rejected", provisional = false, file: string | null = `view_${name}.png`) => ({
    view: name, yaw: { back: 90, left: 180, right: 0 }[name] ?? 45, state, file, candidate: 1, seed: 7, provisional,
    missing: file === null, attempts: 1, wanted_by: { back: ["s90"], right: ["s0"], left: ["s180"] }[name] ?? [],
    url: file ? `/jobs/knight_base/frames/render/1.png` : null, picked: "2026-09-22T10:00:00+00:00",
  });
  // Three slots the base carries, one the record names but the base no longer
  // has (front_left), and four it never had: the ring's three states.
  const withViews: BoardState = {
    ...board,
    views: {
      right: view("right", "pending", true), back: view("back", "accepted"),
      left: view("left", "rejected"), front_left: view("front_left", "rejected", false, null),
    },
  };
  it("lists all eight slots in the table's order, with the next state on click", () => {
    const tiles = viewTiles(withViews);
    expect(tiles.map((t) => t.view)).toEqual([...VIEWS]);
    expect(tiles[0]).toEqual({ view: "back", yaw: 90, state: "accepted", url: "/jobs/knight_base/frames/render/1.png", provisional: false, missing: false, attempts: 1, wantedBy: ["s90"], how: null, next: "rejected", mirrorOf: null, mirrored: false, mirroredFrom: null });
    expect(tiles[2]).toMatchObject({ view: "right", state: "pending", provisional: true, missing: false, wantedBy: ["s0"], next: "accepted" });
    expect(tiles[3]).toMatchObject({ view: "top", missing: true, attempts: 0, url: null, how: "anim concept --unit knight --views top" });
    expect(tiles[6]).toMatchObject({ view: "front_left", state: "rejected", missing: true, url: null, how: "anim concept --unit knight --pick-view front_left=NN" });
    expect(viewTiles(board).every((t) => t.missing)).toBe(true);
    expect(viewTiles(board)).toHaveLength(8);
    expect(nextViewState("rejected")).toBe("pending");
  });
  it("says how many slots are filled, which are missing, and how they stand", () => {
    expect(viewsText(board)).toContain("0 / 8 views on the base");
    const text = viewsText(withViews);
    expect(text).toContain("3 / 8 views on the base, 1 accepted");
    expect(text).toContain("drawn, not picked: front_left");
    expect(text).toContain("missing: top, back_right, back_left, front_right");
    expect(text).toContain("1 provisional");
    expect(text).toContain("rejected: left");
  });
  it("marks a view that is its partner flipped, and never offers the plane a partner", () => {
    // The mirrored view feature. `mirror_of` is the partner a slot COULD be
    // derived from; `mirrored` that the picture on the base already is one.
    const mirrored: BoardState = {
      ...withViews,
      views: {
        ...withViews.views,
        left: { ...view("left", "accepted"), mirror_of: "right" },
        right: { ...view("right", "accepted"), mirror_of: "left", mirrored: true, mirrored_from: "left" },
        back: { ...view("back", "accepted"), mirror_of: null },
      },
    };
    const tiles = viewTiles(mirrored);
    const of = (name: string) => tiles.find((t) => t.view === name)!;
    expect(of("right")).toMatchObject({ mirrored: true, mirroredFrom: "left", mirrorOf: "left" });
    expect(of("left")).toMatchObject({ mirrored: false, mirroredFrom: null, mirrorOf: "right" });
    expect(of("back").mirrorOf).toBeNull();
    expect(of("top").mirrorOf).toBeNull();
    // A slot the base does not carry cannot be "mirrored" whatever it claims:
    // the flip is a file, and there is no file.
    expect(of("front_left")).toMatchObject({ missing: true, mirrored: false });
    const text = viewsText(mirrored);
    expect(text).toContain("mirrored: right ← left (flipped, not generated)");
    expect(viewsText(withViews)).not.toContain("mirrored:");
    // An API from before the feature sends none of the three keys, and the ring
    // simply has nothing to mark.
    expect(viewTiles(withViews).every((t) => !t.mirrored && t.mirrorOf === null)).toBe(true);
  });
  it("folds a view click in without waiting for the GET", () => {
    const next = reduceBoard(withViews, { type: "view", view: "right", state: "accepted" });
    expect(next.views.right!.state).toBe("accepted");
    expect(next.views.right!.provisional).toBe(true);
    expect(next.exists).toBe(true);
    expect(withViews.views.right!.state).toBe("pending");
    // A name that is not one of the eight slots changes nothing.
    expect(reduceBoard(withViews, { type: "view", view: "underneath", state: "accepted" })).toBe(withViews);
  });
});

// A unit straight out of `concept --pick`: a record, three of the eight views
// on the base, no cells. What GET /curation/hooded_archer
// answers, which is where the four board defects showed (2026-09-23).
const slot = (name: string, attempts = 0, file: string | null = null, wantedBy: string[] = []): BoardView => ({
  view: name, yaw: 0, elevation: name === "top" ? "top" : "side", state: "pending",
  missing: file === null, attempts, wanted_by: wantedBy, file,
  url: file ? `/jobs/archer_base/frames/render/0.png` : null,
  candidate: file ? 1 : null, seed: file ? 7 : null, provisional: false, picked: null,
});

// hooded_archer: one attempt at back, left and top, none of them picked onto
// the base, and the other five never drawn.
const freshViews: Record<string, BoardView> = Object.fromEntries(VIEWS.map((name) => [name, slot(name)]));
freshViews.right = slot("right", 0, null, ["s0"]);
for (const name of ["back", "left", "top"]) freshViews[name] = slot(name, 1);

const fresh: BoardState = {
  unit: "archer", mode: "gates", origin: "concept --pick", exists: true,
  path: "/work/archer_curation.json",
  gates: { base: "accepted" },
  gate_blocked: { base: null },
  views: freshViews,
  actions: ["attack", "idle", "walk", "run", "jump"],
  directions: ["s0", "s45", "s90", "s135", "s180", "s225", "s270", "s315"],
  cells: {},
  waiting: [],
};

describe("a unit with nothing on it yet", () => {
  it("draws the whole grid: five action rows by eight direction columns, every cell empty", () => {
    const grid = gridOf(fresh);
    expect(grid.directions).toEqual(["s0", "s45", "s90", "s135", "s180", "s225", "s270", "s315"]);
    expect(grid.rows.map((r) => r.action)).toEqual(["attack", "idle", "walk", "run", "jump"]);
    expect(grid.rows.every((r) => r.cells.length === 8 && r.cells.every((c) => c === null))).toBe(true);
  });

  it("waits on the missing views, then the ones nobody judged", () => {
    const waiting = waitingOn(fresh);
    expect(waiting.length).toBeGreaterThan(0);
    expect(waitingText(waiting)).not.toBe("nothing waiting on you");
    const views = waiting.filter((w) => w.kind === "view");
    expect(views.map((w) => w.key)).toEqual(["right", "back_right", "back_left", "front_left", "front_right", "back", "left", "top"]);
    expect(views[0]!.why).toContain("anim concept --unit archer --views right");
    expect(views[0]!.why).toContain("s0 wants it");
    expect(views[5]!.why).toContain("1 attempt at the back view and none picked");
    expect(views[5]!.why).toContain("--pick-view back=NN");
    // Picked onto the base, so what is waiting is a verdict, not a pick.
    const picked = { ...fresh, views: { ...freshViews, back: { ...slot("back", 1, "view_back.png"), state: "pending" as const } } };
    expect(waitingOn(picked).find((w) => w.key === "back")!.why).toBe("judge the back view of the base - accept it or reject it");
    expect(waiting.some((w) => w.kind === "gate")).toBe(false);
  });

  it("stops naming the views once they are accepted, and says nothing in end mode", () => {
    const done = Object.fromEntries(VIEWS.map((name) => [name, { ...slot(name, 1, "view.png"), state: "accepted" as const }]));
    expect(waitingOn({ ...fresh, views: done })).toEqual([]);
    expect(waitingOn({ ...fresh, mode: "end" })).toEqual([]);
    // A board from an API that sends no view slots is left exactly as it was.
    expect(waitingOn({ ...fresh, views: undefined as unknown as BoardState["views"] })).toEqual([]);
  });

  it("names the missing part palettes after the views", () => {
    const waiting = waitingOn({ ...fresh, base_parts: false, base_parts_job: "archer_base" });
    expect(waiting[waiting.length - 1]).toMatchObject({ kind: "parts", key: "archer_base" });
    expect(waiting[waiting.length - 1]!.why).toContain("anim post archer_base --parts");
    expect(waitingOn({ ...fresh, base_parts: true }).some((w) => w.kind === "parts")).toBe(false);
  });

  it("offers one gate button, the base", () => {
    const buttons = gateButtons(fresh);
    expect(buttons).toHaveLength(1);
    expect(buttons[0]).toMatchObject({ gate: "base", state: "accepted", disabled: false, blocked: null, label: "base: accepted" });
    expect(buttons[0]!.title).toContain("click to reopen");
    // An older record's `turnaround` key gets no button.
    expect(gateButtons({ ...fresh, gates: { base: "accepted", turnaround: "pending" } })).toHaveLength(1);
    expect(gateButtons({ ...fresh, mode: "end" }).every((g) => g.disabled)).toBe(true);
  });

  it("shows all eight view slots, and says how to fill the missing ones", () => {
    const tiles = viewTiles(fresh);
    expect(tiles).toHaveLength(8);
    expect(tiles.every((t) => t.missing)).toBe(true);
    expect(tiles.filter((t) => t.attempts).map((t) => t.view)).toEqual(["back", "left", "top"]);
    const right = tiles.find((t) => t.view === "right")!;
    expect(right).toMatchObject({ missing: true, attempts: 0, url: null, wantedBy: ["s0"], how: "anim concept --unit archer --views right" });
    expect(tiles[0]!.how).toBe("anim concept --unit archer --pick-view back=NN");
    expect(viewsText(fresh)).toContain("0 / 8 views on the base, 0 accepted");
    expect(viewsText(fresh)).toContain("drawn, not picked: back, left, top");
    expect(viewsText(fresh)).toContain("missing: right, back_right, back_left, front_left, front_right");
  });
});

// The part palettes every matrix cell locks to. `post --parts` had no button
// anywhere, and the Pipeline tab's `post` sends no --parts, so a unit could
// only get past this from a terminal.
describe("the part palette offer", () => {
  const noParts = { ...fresh, base_parts: false, base_parts_job: "archer_base" };

  it("offers the job when the base has no measured palettes", () => {
    const offer = partsOffer(noParts)!;
    expect(offer.job).toBe("archer_base");
    expect(offer.label).toBe("measure part palettes");
    expect(offer.title).toContain("anim post archer_base --parts");
    expect(offer.confirm).toContain("no GPU");
    expect(offer.title).toContain("the matrix renders no cell without it");
    expect(partsArgs(offer)).toEqual({ job: "archer_base", parts: true });
  });

  it("offers nothing when the base already has them", () => {
    expect(partsOffer({ ...fresh, base_parts: true })).toBeNull();
  });

  it("offers nothing when the field is absent, rather than guessing", () => {
    // An API process older than this build sends no `base_parts`; guessing
    // "missing" would show the button to every unit, including ones that have
    // the palettes already.
    expect(partsOffer(fresh)).toBeNull();
  });

  it("offers nothing before the base is accepted - there is nothing to measure", () => {
    expect(partsOffer({ ...noParts, gates: { base: "pending" } })).toBeNull();
  });

  it("falls back to <unit>_base when the job name is not sent", () => {
    const offer = partsOffer({ ...fresh, base_parts: false })!;
    expect(offer.job).toBe("archer_base");
  });
});
