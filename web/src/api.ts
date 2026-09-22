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

export type RunStatus = "queued" | "running" | "done" | "failed" | "lost" | "cancelled";

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
  cancelled?: boolean;
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

export type ViewState = "pending" | "accepted" | "rejected";

/** One view of the base (the multi-view reference feature): the views step
 * under the base gate. `url` is the file on the base job, null when the
 * record names a view the base no longer carries. */
export interface BoardView {
  view: string;
  yaw: number;
  state: ViewState;
  file: string | null;
  url: string | null;
  candidate: number | null;
  seed: number | null;
  provisional: boolean;
  picked: string | null;
}

export interface BoardState {
  unit: string;
  mode: CurationMode;
  origin: string | null;
  exists: boolean;
  path: string;
  gates: Record<string, GateState>;
  views: Record<string, BoardView>;
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

// --- the turnaround feature -------------------------------------------------
export type TurnaroundStatus = "done" | "failed" | "running" | "not run";

export interface TurnaroundDirection {
  direction: string;
  job: string;
  status: TurnaroundStatus;
  error: string | null;
  seconds: number | null;
  gpu_seconds: number | null;
  /** `/jobs/{job}/frames/post/0.png` once the direction is done. */
  still: string | null;
  raw: string | null;
  control: string | null;
  skeleton: string | null;
  depth: string | null;
  hidden_face: string[] | null;
  silhouette: { width: number; height: number; pixels: number; colours: number } | null;
  prompt?: string | null;
  /** The reference this still handed Wan (the multi-view reference feature):
   * which view of the base, whether it fell back to the front, and the URL
   * of the image `clip` actually sent. */
  reference?: TurnaroundReference | null;
}

export interface TurnaroundReference {
  view: string | null;
  kind: string | null;
  file: string | null;
  fallback: boolean;
  warning: string | null;
  provisional: boolean;
  state: string | null;
  url: string | null;
}

/** `GET /turnaround/{unit}`: what `anim turnaround` recorded, per direction. */
export interface TurnaroundState {
  unit: string;
  elevation: string;
  seed: number | null;
  question: string;
  updated: string;
  plan: string[];
  directions: Record<string, TurnaroundDirection>;
  done: number;
  of: number;
  page: string;
  identity: { directions: string[]; done: string[]; height: { spread: number } | null; width: { spread: number } | null } & Record<string, unknown>;
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

/** The unit's turnaround, or null when none has run yet (the API's 404). */
export async function getTurnaround(unit: string): Promise<TurnaroundState | null> {
  try {
    return await getJSON<TurnaroundState>(`/turnaround/${encodeURIComponent(unit)}`);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}

export const putGate = (unit: string, gate: string, state: GateState) =>
  sendJSON<BoardState>("PUT", `/curation/${encodeURIComponent(unit)}/gates/${encodeURIComponent(gate)}`, { state });

/** `PUT .../views/{view}`: the views step under the base gate. */
export const putView = (unit: string, view: string, state: ViewState) =>
  sendJSON<BoardState>("PUT", `/curation/${encodeURIComponent(unit)}/views/${encodeURIComponent(view)}`, { state });

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

// --- the concept generation feature ------------------------------------------
export type AttemptSource = "txt2img" | "upload" | "existing" | "img2img" | "edit" | "views";
export type ConceptOp = "txt2img" | "views" | "img2img" | "edit";
export type ModelFamily = "zimage" | "qwen-edit" | "wan";

export interface LoraChoice {
  file: string;
  strength: number;
}

/** One attempt - a base candidate or a view attempt - as job.json / views.json record it. */
export interface Attempt {
  index: number;
  file: string;
  source: AttemptSource;
  prompt: string | null;
  description?: string | null;
  seed: number | null;
  model: string | null;
  loras: LoraChoice[];
  parent: { target: string; index: number } | null;
  strength?: number | null;
  instruction?: string | null;
  workflow?: string | null;
  seconds: number | null;
  sha256?: string | null;
  created?: string;
  from?: string;
  name?: string;
  /** Added by the API: the image, and whether there is one yet. */
  url: string | null;
  rendered: boolean;
}

export interface ConceptView {
  view: string;
  yaw: number;
  elevation: "side" | "top";
  group: "main" | "advanced";
  prompt: string;
  default_prompt: string;
  candidates: Attempt[];
  pick: number | null;
  state: "pending" | "accepted" | "rejected";
  accepted_file: string | null;
}

export interface ConceptState {
  unit: string;
  job: string;
  exists: boolean;
  cell: [number, number];
  /** Why the views may not be drawn now, or null. */
  locked: string | null;
  gates: Record<string, string>;
  defaults: Partial<Record<ModelFamily, { model: string | null; loras: LoraChoice[] }>>;
  last_seed: number | null;
  parts_phrase: string | null;
  base: {
    description: string | null;
    prompt: string | null;
    sketch: unknown;
    candidates: Attempt[];
    pick: number | null;
    accepted: boolean;
    front_url: string | null;
  };
  views: Record<string, ConceptView>;
  views_stale: boolean;
  page: string;
  uploaded?: Attempt;
}

export interface ConceptRequest {
  op: ConceptOp;
  prompt?: string;
  views?: string[] | string;
  view_prompt?: string;
  ref?: string;
  strength?: number;
  instruction?: string;
  count?: number;
  seed?: number;
  model?: string;
  loras?: (string | LoraChoice)[];
  lora_strength?: number;
  fresh?: boolean;
  force?: boolean;
}

export interface RegistryRow {
  file: string;
  family: ModelFamily;
  role: "checkpoint" | "lora" | "controlnet" | "te" | "vae";
  label: string;
  trigger: string | null;
  default_strength: number | null;
  locked: boolean;
  repo: string | null;
  license: string | null;
  installed: boolean | null;
}

export interface ModelsListing {
  registry: string;
  probed: boolean;
  families: Record<ModelFamily, Record<RegistryRow["role"], RegistryRow[]>>;
  ops: Record<ConceptOp, ModelFamily>;
  workflow_defaults: Record<ModelFamily, string>;
  default_lora: Record<ModelFamily, LoraChoice[]>;
}

export const getConcept = (unit: string) => getJSON<ConceptState>(`/concept/${encodeURIComponent(unit)}`);
export const getModels = (probe = false) => getJSON<ModelsListing>(`/models${probe ? "?probe=1" : ""}`);

/** `POST /concept/{unit}/generate`: the run (202), or the refusal (400 / 409 /
 * 423) as a value, like `startRun` - a busy GPU or a pending gate is an
 * answer the tab shows. */
export async function generateConcept(
  unit: string,
  body: ConceptRequest,
): Promise<{ run: RunRecord } | { refused: RunRefusal }> {
  const response = await fetch(`/concept/${encodeURIComponent(unit)}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (response.status === 202) return { run: payload as unknown as RunRecord };
  const error = (payload.error ?? payload.detail ?? response.statusText) as string;
  return {
    refused: {
      status: response.status,
      error: typeof error === "string" ? error : JSON.stringify(error),
      run: payload.run as RunRecord | undefined,
      runs: payload.runs as RunRecord[] | undefined,
    },
  };
}

/** `PUT /concept/{unit}/upload` with the PNG as the body. */
export async function uploadConcept(unit: string, png: Blob, name: string): Promise<ConceptState> {
  const response = await fetch(`/concept/${encodeURIComponent(unit)}/upload`, {
    method: "PUT",
    headers: { "Content-Type": "image/png", "X-Filename": name.replace(/[^\w.-]+/g, "_") || "upload.png" },
    body: png,
  });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return (await response.json()) as ConceptState;
}

export const pickConcept = (unit: string, index: number, parts?: string, avoid?: string) =>
  sendJSON<ConceptState>("POST", `/concept/${encodeURIComponent(unit)}/pick`, { index, parts, avoid });

export const pickConceptView = (unit: string, view: string, index: number) =>
  sendJSON<ConceptState>("POST", `/concept/${encodeURIComponent(unit)}/views/${encodeURIComponent(view)}/pick`, { index });

export const putConceptDefaults = (unit: string, family: ModelFamily, model: string | null, loras: LoraChoice[]) =>
  sendJSON<ConceptState>("PUT", `/concept/${encodeURIComponent(unit)}/defaults`, { family, model, loras });

/** `POST /runs/{id}/cancel`: the run as it stands (202), or the refusal. */
export async function cancelRun(id: string): Promise<{ run: RunRecord } | { refused: RunRefusal }> {
  const response = await fetch(`/runs/${encodeURIComponent(id)}/cancel`, {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (response.status === 202) return { run: payload as unknown as RunRecord };
  const error = (payload.error ?? payload.detail ?? response.statusText) as string;
  return { refused: { status: response.status, error: typeof error === "string" ? error : JSON.stringify(error) } };
}
