import { describe, expect, it } from "vitest";
import type { RunCells, RunRecord } from "./api";
import {
  describeCells,
  describeRun,
  parseRunEvent,
  parseRunEvents,
  reduceRunEvent,
  runTone,
  succeeded,
  viewOf,
} from "./sse";

const record: RunRecord = {
  id: "20260921-120000-pick-ab12", stage: "pick", argv: ["--phases", "knight_strike"], gpu: false,
  status: "queued", percent: null, started: "2026-09-21T12:00:00+00:00", finished: null,
  returncode: null, pid: null, log_lines: 0, url: "/runs/x", events: "/runs/x/events",
};

// What anim/api/runs.py writes, verbatim (test_sse_yields_log_progress_and_done).
const STREAM = [
  ": keepalive",
  'event: log\ndata: {"line": "[1/2] started pick x"}',
  'event: progress\ndata: {"percent": 50}',
  'event: log\ndata: {"line": "[2/2] finished"}',
  'event: progress\ndata: {"percent": 100}',
  'event: done\ndata: {"status": "done", "returncode": 0, "percent": 100, "cells": null}',
].join("\n\n") + "\n\n";

// A batch stage: the [n/m] bar reaches 100 over the cells that FINISHED, and
// the tally that arrives with `done` says every one of them failed. This is
// the 2026-09-24 eight-direction batch run, as anim/api/runs.py now writes it.
const ALL_FAILED = [
  'event: log\ndata: {"line": "  s0: FAILED after 0.4 s - ValueError: no part table for unit"}',
  'event: progress\ndata: {"percent": 100}',
  'event: log\ndata: {"line": "outcome: done=0 failed=8 blocked=0 of=8"}',
  'event: progress\ndata: {"percent": 0}',
  'event: done\ndata: {"status": "failed", "returncode": 4, "percent": 0, "cells": {"done": 0, "failed": 8, "blocked": 0, "of": 8}}',
].join("\n\n") + "\n\n";

describe("parseRunEvent", () => {
  it("reads the three event kinds and nothing else", () => {
    expect(parseRunEvent("log", '{"line": "pick: [0, 4]"}')).toEqual({ type: "log", line: "pick: [0, 4]" });
    expect(parseRunEvent("progress", '{"percent": 43}')).toEqual({ type: "progress", percent: 43 });
    expect(parseRunEvent("done", '{"status": "failed", "returncode": 3, "percent": 100}'))
      .toEqual({ type: "done", status: "failed", returncode: 3, percent: 100, cells: null });
    expect(parseRunEvent("done", '{"status": "sideways"}')).toBeNull();
    expect(parseRunEvent("ping", "{}")).toBeNull();
    expect(parseRunEvent("log", "not json")).toBeNull();
  });
  it("clamps a percent and drops a non-number", () => {
    expect(parseRunEvent("progress", '{"percent": 140}')).toEqual({ type: "progress", percent: 100 });
    expect(parseRunEvent("progress", '{"percent": null}')).toEqual({ type: "progress", percent: null });
  });
});

describe("parseRunEvents + reduceRunEvent", () => {
  it("folds the API's stream into the final view, keepalives ignored", () => {
    const events = parseRunEvents(STREAM);
    expect(events.map((e) => e.type)).toEqual(["log", "progress", "log", "progress", "done"]);
    const view = events.reduce(reduceRunEvent, viewOf(record));
    expect(view.log).toEqual(["[1/2] started pick x", "[2/2] finished"]);
    expect(view.status).toBe("done");
    expect(view.percent).toBe(100);
    expect(view.returncode).toBe(0);
  });
  it("marks a queued run running at its first line, and never mutates", () => {
    const start = viewOf(record);
    const next = reduceRunEvent(start, { type: "log", line: "hello" });
    expect(next.status).toBe("running");
    expect(start.status).toBe("queued");
    expect(start.log).toEqual([]);
  });
  it("keeps the last percent when done carries none", () => {
    const view = reduceRunEvent({ ...viewOf(record), percent: 80 }, { type: "done", status: "lost", returncode: null, percent: null, cells: null });
    expect(view.percent).toBe(80);
    expect(view.status).toBe("lost");
  });
});

// The bug this file grew for: a batch whose eight directions all failed
// exited cleanly, because going on past a bad cell is deliberate, and arrived
// here as `done` at 100%. anim/outcome.py sends the tally along; nothing below
// may let a run that accomplished nothing read as a full success.
describe("a run whose every cell failed", () => {
  const tally: RunCells = { done: 0, failed: 8, blocked: 0, of: 8 };

  it("does not end done at 100%", () => {
    const view = parseRunEvents(ALL_FAILED).reduce(reduceRunEvent, viewOf({ ...record, stage: "matrix" }));
    expect(view.status).toBe("failed");
    expect(view.status).not.toBe("done");
    expect(view.percent).toBe(0);
    expect(view.percent).not.toBe(100);
    expect(view.cells).toEqual(tally);
    expect(succeeded(view)).toBe(false);
  });

  it("says what went wrong instead of an exit code", () => {
    const view = { ...viewOf({ ...record, stage: "matrix" }), status: "failed" as const, returncode: 4, percent: 0, cells: tally };
    expect(describeRun(view)).toBe("matrix: failed - 0 of 8 cells done, 8 failed");
    expect(runTone("failed")).toBe("bad");
  });

  it("reads a tally off the wire only when every counter is there", () => {
    expect(parseRunEvent("done", '{"status": "failed", "returncode": 4, "percent": 0, "cells": {"done": 0, "failed": 8, "blocked": 0, "of": 8}}'))
      .toEqual({ type: "done", status: "failed", returncode: 4, percent: 0, cells: tally });
    for (const cells of ['{"done": 0, "of": 8}', '"eight"', "null", '{"done": null, "failed": 8, "blocked": 0, "of": 8}']) {
      expect(parseRunEvent("done", `{"status": "failed", "returncode": 4, "percent": 0, "cells": ${cells}}`))
        .toEqual({ type: "done", status: "failed", returncode: 4, percent: 0, cells: null });
    }
  });
});

describe("a partial run", () => {
  const view = (cells: RunCells, status: "partial" | "done" = "partial") => ({
    ...viewOf({ ...record, stage: "matrix" }),
    status,
    returncode: status === "done" ? 0 : 3,
    percent: Math.round((100 * cells.done) / cells.of),
    cells,
  });

  it("is told apart from a full success by status, bar and line", () => {
    const partial = view({ done: 5, failed: 3, blocked: 0, of: 8 });
    const full = view({ done: 8, failed: 0, blocked: 0, of: 8 }, "done");
    expect(partial.status).not.toBe(full.status);
    expect(partial.percent).toBe(63);
    expect(full.percent).toBe(100);
    expect(succeeded(partial)).toBe(false);
    expect(succeeded(full)).toBe(true);
    expect(describeRun(partial)).toBe("matrix: partial - 5 of 8 cells done, 3 failed");
    expect(describeRun(full)).toBe("matrix: done - 8 of 8 cells done");
    // A warning, not a loss: five cells are on disk and the owner re-runs
    // three, which is neither what `done` nor what `failed` would mean.
    expect(runTone("partial")).toBe("warn");
    expect(runTone("partial")).not.toBe(runTone("done"));
    expect(runTone("partial")).not.toBe(runTone("failed"));
  });

  it("names the blocked and the never-run cells too", () => {
    expect(describeCells({ done: 2, failed: 0, blocked: 6, of: 8 }))
      .toBe("2 of 8 cells done, 6 blocked");
    expect(describeCells({ done: 3, failed: 0, blocked: 0, of: 8 }))
      .toBe("3 of 8 cells done, 5 not run");
    expect(describeCells({ done: 1, failed: 0, blocked: 0, of: 1 })).toBe("1 of 1 cell done");
    expect(describeCells(null)).toBeNull();
    expect(describeCells({ done: 0, failed: 0, blocked: 0, of: 0 })).toBeNull();
  });

  it("leaves a stage that counts no cells reading exactly as before", () => {
    const view = { ...viewOf(record), status: "done" as const, percent: 100 };
    expect(view.cells).toBeNull();
    expect(describeRun(view)).toBe("pick: done");
    expect(describeRun({ ...view, status: "failed", returncode: 3 })).toBe("pick: failed (exit 3)");
  });
});

describe("describeRun", () => {
  it("says what the panel prints under the bar", () => {
    expect(describeRun(viewOf(record))).toBe("pick: queued");
    expect(describeRun({ ...viewOf(record), status: "running", percent: 43 })).toBe("pick: running 43%");
    expect(describeRun({ ...viewOf(record), status: "failed", returncode: 3 })).toBe("pick: failed (exit 3)");
    expect(describeRun({ ...viewOf(record), status: "lost" })).toContain("lost");
  });
});
