// A run as the panel sees it: `GET /runs/{id}` for the starting point, then
// `GET /runs/{id}/events` (text/event-stream) for every line after. The
// reducer is pure and tested; `followRun` is the thin EventSource around it.
//
// A BATCH STAGE\'S RUN CARRIES A TALLY. `matrix` and `bakeoff`
// catch each cell\'s failure and go on to the next, so the process can exit
// cleanly having accomplished nothing; the API answers that with the status
// `partial` and a `cells` count (anim/outcome.py). Everything here that used
// to ask "is the status done?" has to mean "did the work succeed?", which is
// why `describeRun` spends the tally rather than printing the word `done`.

import type { RunCells, RunRecord, RunStatus } from "./api";

export interface RunView {
  id: string;
  stage: string;
  gpu: boolean;
  status: RunStatus;
  percent: number | null;
  returncode: number | null;
  cells: RunCells | null;
  log: string[];
}

export type RunEvent =
  | { type: "log"; line: string }
  | { type: "progress"; percent: number | null }
  | { type: "done"; status: RunStatus; returncode: number | null; percent: number | null; cells: RunCells | null };

export const ACTIVE: readonly RunStatus[] = ["queued", "running"];

/** Statuses a finished run can hold. `done` is the only full success; the
 * other four are each a different kind of not-success, and none of them may
 * be rendered as one. */
export const FINISHED: readonly RunStatus[] = ["done", "partial", "failed", "lost", "cancelled"];

export function viewOf(run: RunRecord): RunView {
  return {
    id: run.id,
    stage: run.stage,
    gpu: run.gpu,
    status: run.status,
    percent: run.percent,
    returncode: run.returncode,
    cells: run.cells ?? null,
    log: run.log ? run.log.slice() : [],
  };
}

/** One event folded into the view. Never mutates its input. */
export function reduceRunEvent(view: RunView, event: RunEvent): RunView {
  switch (event.type) {
    case "log":
      return { ...view, log: [...view.log, event.line], status: view.status === "queued" ? "running" : view.status };
    case "progress":
      return { ...view, percent: event.percent };
    case "done":
      return {
        ...view,
        status: event.status,
        returncode: event.returncode,
        percent: event.percent ?? view.percent,
        // A tally only ever arrives with the end of the run, and a `done`
        // without one leaves what the record started with rather than
        // clearing it - a reconnect replays the stream from the beginning.
        cells: event.cells ?? view.cells,
      };
  }
}

/**
 * `event: log\ndata: {"line": "..."}` -> the event, or null for a keepalive
 * comment or anything the API does not send. The API writes one JSON object
 * per `data:` line, so a block has at most one.
 */
export function parseRunEvent(name: string, data: string): RunEvent | null {
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(data) as Record<string, unknown>;
  } catch {
    return null;
  }
  switch (name) {
    case "log":
      return typeof payload.line === "string" ? { type: "log", line: payload.line } : null;
    case "progress":
      return { type: "progress", percent: asPercent(payload.percent) };
    case "done": {
      const status = payload.status as RunStatus;
      if (!FINISHED.includes(status)) return null;
      const returncode = typeof payload.returncode === "number" ? payload.returncode : null;
      return { type: "done", status, returncode, percent: asPercent(payload.percent), cells: asCells(payload.cells) };
    }
    default:
      return null;
  }
}

function asPercent(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
}

/** A tally off the wire, or null. Every counter must be a number: a half-read
 * tally would make `describeRun` say "0 of 0", which is worse than saying
 * nothing and letting the status speak. */
function asCells(value: unknown): RunCells | null {
  if (typeof value !== "object" || value === null) return null;
  const row = value as Record<string, unknown>;
  const counts = ["done", "failed", "blocked", "of"].map((key) => row[key]);
  if (!counts.every((n) => typeof n === "number" && Number.isFinite(n))) return null;
  const [done, failed, blocked, of] = counts as [number, number, number, number];
  return { done, failed, blocked, of };
}

/** The whole stream as text (what a test or a replay hands over) -> events. */
export function parseRunEvents(stream: string): RunEvent[] {
  const events: RunEvent[] = [];
  for (const block of stream.split(/\n\n+/)) {
    let name = "message";
    let data: string | null = null;
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) name = line.slice(6).trim();
      else if (line.startsWith("data:")) data = (data === null ? "" : data + "\n") + line.slice(5).trim();
    }
    if (data === null) continue;
    const event = parseRunEvent(name, data);
    if (event) events.push(event);
  }
  return events;
}

/**
 * `0 of 8 cells done, 8 failed` - a batch stage\'s tally in words, or null when
 * the run counted no cells. It is the sentence the panel could not say before
 * the tally existed, when a run that produced nothing read `done`.
 */
export function describeCells(cells: RunCells | null): string | null {
  if (!cells || cells.of <= 0) return null;
  const parts = [`${cells.done} of ${cells.of} cell${cells.of === 1 ? "" : "s"} done`];
  if (cells.failed) parts.push(`${cells.failed} failed`);
  if (cells.blocked) parts.push(`${cells.blocked} blocked`);
  const missing = cells.of - cells.done - cells.failed - cells.blocked;
  if (missing > 0) parts.push(`${missing} not run`);
  return parts.join(", ");
}

/** The one-line summary the panel prints under the bar. */
export function describeRun(view: RunView): string {
  const where = view.percent === null ? "" : ` ${view.percent}%`;
  const tally = describeCells(view.cells);
  switch (view.status) {
    case "queued":
      return `${view.stage}: queued`;
    case "running":
      return `${view.stage}: running${where}`;
    case "done":
      // A full success still names its cells when it has them: "8 of 8" is
      // what makes the partial and failed lines below read as the exception.
      return tally ? `${view.stage}: done - ${tally}` : `${view.stage}: done`;
    case "partial":
      return `${view.stage}: partial - ${tally ?? `exit ${view.returncode ?? "?"}`}`;
    case "failed":
      // The tally comes first when there is one: "8 of 8 cells failed" says
      // what went wrong, where "exit 4" only says that something did.
      return tally
        ? `${view.stage}: failed - ${tally}`
        : `${view.stage}: failed (exit ${view.returncode ?? "?"})`;
    case "lost":
      return `${view.stage}: lost (the API restarted while it ran)`;
    case "cancelled":
      return tally ? `${view.stage}: cancelled - ${tally}` : `${view.stage}: cancelled`;
  }
}

/** Did the run accomplish what it was started for? `done` and nothing else -
 * the question every caller that used to compare against `"done"` is really
 * asking, now that a run can finish without succeeding. */
export function succeeded(view: RunView): boolean {
  return view.status === "done";
}

/**
 * The `studio-status` class a run\'s line wears, for every surface that shows
 * one (the panel, the Board, the Matrix tab). It lives here rather than in
 * each view so the three cannot drift into colouring `partial` differently -
 * which is the whole hazard of adding a status to the vocabulary.
 *
 * `partial` is a WARNING: some cells are on disk and some are not, and the
 * owner re-runs that differently from a clean run and from a total loss.
 */
export function runTone(status: RunStatus): "ok" | "warn" | "bad" | "" {
  switch (status) {
    case "done":
      return "ok";
    case "failed":
    case "lost":
      return "bad";
    case "partial":
    case "cancelled":
      return "warn";
    default:
      return "";
  }
}

/**
 * Follow a run until it ends. `onChange` gets every new view; the returned
 * function stops following (the stream is closed either way when `done`
 * arrives). The browser's EventSource reconnects on its own if the
 * connection drops, and the API replays the whole log on reconnect - so the
 * view is rebuilt from `initial` on every open, never appended twice.
 */
export function followRun(initial: RunView, onChange: (view: RunView) => void): () => void {
  let view = initial;
  const source = new EventSource(`/runs/${encodeURIComponent(initial.id)}/events`);
  const push = (event: RunEvent | null) => {
    if (!event) return;
    view = reduceRunEvent(view, event);
    onChange(view);
    if (event.type === "done") source.close();
  };
  source.onopen = () => {
    view = { ...initial, log: [] };
    onChange(view);
  };
  for (const name of ["log", "progress", "done"]) {
    source.addEventListener(name, (message) => push(parseRunEvent(name, (message as MessageEvent).data as string)));
  }
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) onChange(view);
  };
  return () => source.close();
}
