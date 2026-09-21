// A run as the panel sees it: `GET /runs/{id}` for the starting point, then
// `GET /runs/{id}/events` (text/event-stream) for every line after. The
// reducer is pure and tested; `followRun` is the thin EventSource around it.

import type { RunRecord, RunStatus } from "./api";

export interface RunView {
  id: string;
  stage: string;
  gpu: boolean;
  status: RunStatus;
  percent: number | null;
  returncode: number | null;
  log: string[];
}

export type RunEvent =
  | { type: "log"; line: string }
  | { type: "progress"; percent: number | null }
  | { type: "done"; status: RunStatus; returncode: number | null; percent: number | null };

export const ACTIVE: readonly RunStatus[] = ["queued", "running"];

export function viewOf(run: RunRecord): RunView {
  return {
    id: run.id,
    stage: run.stage,
    gpu: run.gpu,
    status: run.status,
    percent: run.percent,
    returncode: run.returncode,
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
      return { ...view, status: event.status, returncode: event.returncode, percent: event.percent ?? view.percent };
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
      const status = payload.status;
      if (status !== "done" && status !== "failed" && status !== "lost") return null;
      const returncode = typeof payload.returncode === "number" ? payload.returncode : null;
      return { type: "done", status, returncode, percent: asPercent(payload.percent) };
    }
    default:
      return null;
  }
}

function asPercent(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null;
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

/** The one-line summary the panel prints under the bar. */
export function describeRun(view: RunView): string {
  const where = view.percent === null ? "" : ` ${view.percent}%`;
  switch (view.status) {
    case "queued":
      return `${view.stage}: queued`;
    case "running":
      return `${view.stage}: running${where}`;
    case "done":
      return `${view.stage}: done`;
    case "failed":
      return `${view.stage}: failed (exit ${view.returncode ?? "?"})`;
    case "lost":
      return `${view.stage}: lost (the API restarted while it ran)`;
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
