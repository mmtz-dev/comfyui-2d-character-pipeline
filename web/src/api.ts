// The pipeline API (anim/api/app.py) as typed calls. Same origin: the API
// serves this page from web/dist at `/`, and `vite` proxies these routes to
// it while developing. Nothing here caches or retries - a job is its files
// on disk, so the answer is always the current one.

export interface JobRow {
  name: string;
  slug: string;
  action: string;
  direction: string;
  frames: number | null;
  stages: string[];
  rerender_requested: boolean;
  updated: string;
  url: string;
}

export interface JobDetail extends JobRow {
  manifest: Record<string, unknown> & { fps?: number };
  picks: number[] | null;
  frame_sets: Record<string, number>;
}

export interface FrameEntry {
  index: number;
  file: string;
  url: string;
}

export interface FramesListing {
  job: string;
  stages: Record<string, FrameEntry[]>;
}

export interface PickFrame {
  index: number;
  file: string;
  picked: boolean;
  order: number | null;
  phase: string | null;
  skeleton: boolean;
}

export interface PicksState {
  job: string;
  action: string;
  slug: string;
  fps: number | null;
  n: number;
  picks: number[];
  auto: number[];
  edited: boolean;
  rerender_requested: boolean;
  phases: { tags?: string[]; plan?: Record<string, unknown> } | null;
  frames: PickFrame[];
}

export type RunStatus = "queued" | "running" | "done" | "failed" | "lost";

export interface RunRecord {
  id: string;
  stage: string;
  argv: string[];
  gpu: boolean;
  status: RunStatus;
  percent: number | null;
  started: string;
  finished: string | null;
  returncode: number | null;
  pid: number | null;
  log_lines: number;
  url: string;
  events: string;
  log?: string[];
}

export interface Health {
  ok: boolean;
  version: string;
  work_dir: string;
  comfyui_url: string;
  gpu_run: string | null;
  cpu_runs: string[];
}

/** `work/project.json` merged over the defaults; the palette feature's fields
 * are whatever it adds, so the type is open. */
export type Settings = Record<string, unknown> & {
  schema?: string;
  palette?: unknown;
  frame_size?: [number, number];
};

export interface SavedFrame {
  job: string;
  stage: string;
  index: number;
  file: string;
  size: [number, number];
  alpha_rounded: number;
  bytes: number;
}

// --- the curation board feature ---------------------------------------------
export type CurationMode = "gates" | "every" | "end";
export type GateState = "pending" | "accepted";
export type CellStatus = "pending" | "accepted" | "rejected";
export type FrameMark = "accept" | "reject";

export interface StripFrame {
  /** The source frame number - what picks.json and the marks use. */
  index: number;
  url: string;
  mark: FrameMark | null;
}

export interface BoardCell {
  key: string;
  action: string;
  direction: string;
  verb: string | null;
  job: string | null;
  status: CellStatus;
  frames: Record<string, FrameMark>;
  seed: number | null;
  regenerated_from: number | null;
  decided_at: string | null;
  run?: string | null;
  stages: string[];
  /** The GPU stage a regenerate runs: `clip` or `render`. */
  stage: string | null;
  current_seed: number | null;
  rerender_requested: boolean;
  set: string | null;
  picks: number[];
  strip: StripFrame[];
  stale?: boolean;
}

export interface Waiting {
  kind: "gate" | "cell";
  key: string;
  why: string;
}

export interface BoardState {
  unit: string;
  mode: CurationMode;
  origin: string | null;
  exists: boolean;
  path: string;
  gates: Record<string, GateState>;
  actions: string[];
  directions: string[];
  cells: Record<string, BoardCell>;
  waiting: Waiting[];
}

export interface CurationUnit {
  unit: string;
  cells: number;
  record: boolean;
}

export class ApiError extends Error {
  constructor(public status: number, message: string, public body: unknown = null) {
    super(message);
  }
}

async function detail(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const body = JSON.parse(text) as Record<string, unknown>;
    const message = body.detail ?? body.error ?? text;
    return typeof message === "string" ? message : JSON.stringify(message);
  } catch {
    return text || response.statusText;
  }
}

export async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return (await response.json()) as T;
}

export const listJobs = () => getJSON<JobRow[]>("/jobs");
export const getJob = (name: string) => getJSON<JobDetail>(`/jobs/${encodeURIComponent(name)}`);
export const listFrames = (name: string) => getJSON<FramesListing>(`/jobs/${encodeURIComponent(name)}/frames`);
export const getRun = (id: string) => getJSON<RunRecord>(`/runs/${encodeURIComponent(id)}`);
export const listRuns = () => getJSON<RunRecord[]>("/runs");
export const getSettings = () => getJSON<Settings>("/settings");
export const getHealth = () => getJSON<Health>("/health");

/** The picks, or null when the job has no extracted frames (the API's 409). */
export async function getPicks(name: string): Promise<PicksState | null> {
  try {
    return await getJSON<PicksState>(`/jobs/${encodeURIComponent(name)}/picks`);
  } catch (error) {
    if (error instanceof ApiError && error.status === 409) return null;
    throw error;
  }
}

export interface RunRefusal {
  status: number;
  error: string;
  run?: RunRecord;
  runs?: RunRecord[];
}

/** `POST /jobs`: the run (202), or the refusal (400 / 409) as a value - a
 * busy GPU is an answer the panel shows, not an exception. */
export async function startRun(
  stage: string,
  args: Record<string, unknown>,
): Promise<{ run: RunRecord } | { refused: RunRefusal }> {
  const response = await fetch("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ stage, args }),
  });
  const body = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (response.status === 202) return { run: body as unknown as RunRecord };
  const error = (body.error ?? body.detail ?? response.statusText) as string;
  return {
    refused: {
      status: response.status,
      error: typeof error === "string" ? error : JSON.stringify(error),
      run: body.run as RunRecord | undefined,
      runs: body.runs as RunRecord[] | undefined,
    },
  };
}

/** `PUT /jobs/{job}/frames/{set}/{n}.png` with the PNG as the body. */
export async function putFrame(job: string, set: string, index: number, png: Blob): Promise<SavedFrame> {
  const response = await fetch(
    `/jobs/${encodeURIComponent(job)}/frames/${encodeURIComponent(set)}/${index}.png`,
    { method: "PUT", headers: { "Content-Type": "image/png" }, body: png },
  );
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return (await response.json()) as SavedFrame;
}

async function sendJSON<T>(method: string, path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return (await response.json()) as T;
}

// --- the curation board feature ---------------------------------------------
export const listCuration = () => getJSON<{ units: CurationUnit[]; mode: CurationMode }>("/curation");
export const getBoard = (unit: string) => getJSON<BoardState>(`/curation/${encodeURIComponent(unit)}`);

function cellPath(unit: string, action: string, direction: string): string {
  return `/curation/${encodeURIComponent(unit)}/cells/${encodeURIComponent(action)}/${encodeURIComponent(direction)}`;
}

/** `PUT .../cells/{action}/{dir}`: a status and/or frame marks (null clears one). */
export const putCell = (
  unit: string,
  action: string,
  direction: string,
  body: { status?: CellStatus; frames?: Record<string, FrameMark | null> },
) => sendJSON<BoardCell>("PUT", cellPath(unit, action, direction), body);

export const putGate = (unit: string, gate: string, state: GateState) =>
  sendJSON<BoardState>("PUT", `/curation/${encodeURIComponent(unit)}/gates/${encodeURIComponent(gate)}`, { state });

/** `PUT /settings` is the whole document: send back what GET gave, changed. */
export const putSettings = (settings: Settings) => sendJSON<Settings>("PUT", "/settings", settings);

/** `POST .../regenerate`: the cell with its run (202), or the refusal - a busy
 * GPU (409) or a pending gate (423) - as a value, like `startRun`. */
export async function regenerateCell(
  unit: string,
  action: string,
  direction: string,
): Promise<{ cell: BoardCell; run: RunRecord } | { refused: RunRefusal }> {
  const response = await fetch(`${cellPath(unit, action, direction)}/regenerate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: "{}",
  });
  const body = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (response.status === 202) {
    const cell = body as unknown as BoardCell & { run: RunRecord };
    return { cell, run: cell.run };
  }
  const error = (body.error ?? body.detail ?? response.statusText) as string;
  return {
    refused: {
      status: response.status,
      error: typeof error === "string" ? error : JSON.stringify(error),
      run: body.run as RunRecord | undefined,
      runs: body.runs as RunRecord[] | undefined,
    },
  };
}
