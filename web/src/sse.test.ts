import { describe, expect, it } from "vitest";
import type { RunRecord } from "./api";
import { describeRun, parseRunEvent, parseRunEvents, reduceRunEvent, viewOf } from "./sse";

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
  'event: done\ndata: {"status": "done", "returncode": 0, "percent": 100}',
].join("\n\n") + "\n\n";

describe("parseRunEvent", () => {
  it("reads the three event kinds and nothing else", () => {
    expect(parseRunEvent("log", '{"line": "pick: [0, 4]"}')).toEqual({ type: "log", line: "pick: [0, 4]" });
    expect(parseRunEvent("progress", '{"percent": 43}')).toEqual({ type: "progress", percent: 43 });
    expect(parseRunEvent("done", '{"status": "failed", "returncode": 3, "percent": 100}'))
      .toEqual({ type: "done", status: "failed", returncode: 3, percent: 100 });
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
    const view = reduceRunEvent({ ...viewOf(record), percent: 80 }, { type: "done", status: "lost", returncode: null, percent: null });
    expect(view.percent).toBe(80);
    expect(view.status).toBe("lost");
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
