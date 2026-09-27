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

/** `partial` is the batch stages\' own: the process finished and some of the
 * cells it planned did not. See anim/outcome.py - a run whose every cell
 * failed used to arrive here as `done` at 100%. */
export type RunStatus = "queued" | "running" | "done" | "partial" | "failed" | "lost" | "cancelled";

/** What `matrix` and `bakeoff` accomplished, in cells. Null for
 * every other stage: a single-shot stage has no cells to count, and no tally
 * is not a tally of zero. Older records (before 2026-09-24) carry none. */
export interface RunCells {
  done: number;
  failed: number;
  blocked: number;
  of: number;
}

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
  cells?: RunCells | null;
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
  kind: "gate" | "view" | "parts" | "cell";
  key: string;
  why: string;
}

export type ViewState = "pending" | "accepted" | "rejected";

/** One view of the base (the multi-view reference feature): the views step
 * under the base gate. The API sends all eight slots of `CONCEPT_VIEWS`, so
 * `missing` (the base carries no file for it) is the ring's third state
 * beside accepted and pending. `url` is the file on the base job, null when
 * the view is missing. `wanted_by` names the matrix directions that ask
 * for this view first - s0 wants `right`. */
export interface BoardView {
  view: string;
  yaw: number;
  elevation?: string;
  state: ViewState;
  /** Optional: an API from before the eight-slot ring sends only what exists. */
  missing?: boolean;
  /** How many candidates `concept --views` has drawn for this view. A slot
   * can be missing from the base and still have attempts waiting for a pick. */
  attempts?: number;
  wanted_by?: string[];
  /** The mirrored view feature; optional for the same reason as `missing` -
   * an API from before it sends neither. `mirror_of` is the partner this slot
   * could be flipped from, `mirrored` that it already was. */
  mirror_of?: string | null;
  mirrored?: boolean;
  mirrored_from?: string | null;
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
  /** Why a gate cannot be accepted right now, per gate, or null when it can.
   * Optional because the long-lived API process may predate this build. */
  gate_blocked?: Record<string, string | null>;
  /** Whether `<unit>_base/post/parts.json` exists - the measured part palettes
   * every direction locks its colours to. Optional for the same reason as
   * `gate_blocked`: the long-lived API process may predate this build. */
  base_parts?: boolean;
  /** The job whose `post --parts` measures them. */
  base_parts_job?: string;
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

// --- the action matrix feature ------------------------------------------------
/** What one cell's record says it is. `kept` is done with the owner's own
 * hand-edited picks left alone; `not run` is planned and never started. */
export type MatrixCellStatus = "done" | "failed" | "blocked" | "kept" | "not run";

/** Does the last pick lead back into the first (`pick`'s `loop_closure`)?
 * Information, not a failure: idle / walk / run are loops and Wan does not
 * know it. `metric: "none"` means neither poses nor frames could answer. */
export interface LoopClosure {
  first: number;
  last: number;
  metric: "pose" | "pixel" | "none";
  distance: number | null;
  tolerance: number;
  ok: boolean | null;
}

/**
 * One (action x direction) cell of `GET /matrix/{unit}`, as
 * `matrix.listing()` builds it. Every planned cell has a row whether or not
 * anything has run, so the tab can grey what it cannot generate on first
 * load: `runnable` and `reason` are recomputed from the curation record on
 * every GET, never read back from the state file.
 */
export interface MatrixCell {
  key: string;
  action: string;
  /** The PIPELINE verb - attack's job is `<unit>_strike[_<dir>]`. */
  verb: string;
  direction: string;
  yaw: number;
  job: string;
  /** The rig's clip for the action, null when that rig ships none (jump). */
  clip: string | null;
  status: MatrixCellStatus;
  runnable: boolean;
  /** The whole sentence a blocked cell shows - never a silent grey. */
  reason: string | null;
  /** The accepted view of the base this cell would render against. */
  view: string | null;
  /** The views the direction wants, best first (`matrix_reference_views`). */
  wanted: string[];
  frames: number;
  /** `/jobs/<job>/frames/post/<i>.png`, one per finished frame. */
  strip: string[];
  /** The phase tag of each frame of `strip`, null where there is none. */
  phases: (string | null)[];
  fps: number;
  loop: boolean;
  closure: LoopClosure | null;
  seconds: number | null;
  gpu_seconds: number | null;
  seed: number | null;
  error: string | null;
  /** The cell's status on the Board, so the tab needs no second fetch. */
  curation: CellStatus | null;
}

/** The counts of the last run's plan; null before anything has run. */
export interface MatrixSummary {
  done: number;
  failed: number;
  blocked: number;
  of: number;
  gpu_seconds: number;
  seconds: number;
}

/** `GET /matrix/{unit}`: the whole five-by-eight grid with its live verdicts.
 * There is no 404 for "no matrix yet" - only a name that is not a unit slug. */
export interface MatrixState {
  unit: string;
  schema: string;
  elevation: string;
  /** The unit's one seed, once a run has derived it. */
  seed: number | null;
  rig: { name: string; glb: string };
  actions: string[];
  directions: string[];
  cells: Record<string, MatrixCell>;
  /** The #277 hold: the widest selection the stage will accept (0 = no cap). */
  max_cells: number;
  selectable: number;
  /** Seconds one cell costs, warm - multiply by the selection. */
  estimate_seconds: number;
  summary: MatrixSummary | null;
  updated: string | null;
  path: string;
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

/** The action matrix. Always answers for a unit slug - a unit nothing has
 * run for still gets a row per planned cell, blocked with the reason. */
export const getMatrix = (unit: string) => getJSON<MatrixState>(`/matrix/${encodeURIComponent(unit)}`);

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
export type AttemptSource = "txt2img" | "upload" | "existing" | "img2img" | "edit" | "views" | "animate" | "mirror";
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
  /** What was done to it after it was generated (the retouch feature). */
  edited?: {
    how: "hand" | "inpaint" | "img2img" | "edit";
    at: string;
    prompt?: string;
    strength?: number;
    instruction?: string;
    seed?: number;
    seconds?: number;
  } | null;
  /** Hidden from the strips (the record and, unless deleted, the file stay). */
  hidden: boolean;
  /** When the PNG was deleted, if it was; the row is its record. */
  file_deleted: string | null;
  /** Added by the API: the image, and whether there is one yet. */
  url: string | null;
  rendered: boolean;
  /** The animate tab feature: the owner's pose sentence (the rest of
   * `prompt` is the wrapper), what the attempt was drawn from - the images in
   * the order Qwen-Image-Edit saw them - the skeleton frame, if any, and for a
   * mirrored attempt the partner frame it is the flip of. */
  pose_prompt?: string | null;
  draw_from?: DrawFrom | null;
  images?: AnimateImage[];
  skeleton?: AnimateSkeleton | null;
  mirror?: { direction: string; frame: string; ref: string; sha256: string; how: string } | null;
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
  hidden_count: number;
  /** The accepted attempt was edited after it was accepted: the copy the
   * matrix reads is the older picture until it is accepted again. */
  pick_stale: boolean;
  state: "pending" | "accepted" | "rejected";
  accepted_file: string | null;
  /** The mirrored view feature. `mirror_of` is the view this one COULD be
   * derived from by flipping it (null for back, top and front, which look
   * straight down the symmetry plane); `mirrored` / `mirrored_from` say the
   * picture the base carries already IS that flip. Two different facts. */
  mirror_of: string | null;
  mirrored: boolean;
  mirrored_from: string | null;
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
    hidden_count: number;
    pick_stale: boolean;
    accepted: boolean;
    front_url: string | null;
  };
  views: Record<string, ConceptView>;
  views_stale: boolean;
  /** Seconds one WARM view render takes, measured from this unit's own
   * records (or every unit's, or the config constant). The first render of a
   * cold run costs COLD_SECONDS more while Qwen-Image-Edit loads. */
  estimate_seconds: number;
  page: string;
  uploaded?: Attempt;
  /** The one line a mirror press answers with when the unit carries handed
   * gear: the flip swaps the sword hand. Absent on every other response. */
  warning?: string | null;
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
  in_place?: boolean;
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
/** The mirrored view feature: the same route, deriving the view from its
 * accepted partner instead of naming one of its own attempts. No GPU. */
export const mirrorConceptView = (unit: string, view: string) =>
  sendJSON<ConceptState>("POST", `/concept/${encodeURIComponent(unit)}/views/${encodeURIComponent(view)}/pick`, { mirror: true });

/** `PUT .../attempts/{target}/{index}/image`: the editor's Save (the retouch
 * feature). The body is the PNG; transparency is flattened onto white. */
export async function putConceptImage(unit: string, target: string, index: number, png: Blob): Promise<ConceptState> {
  const response = await fetch(`${attemptPath(unit, target, index)}/image`, {
    method: "PUT", headers: { "Content-Type": "image/png" }, body: png,
  });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return (await response.json()) as ConceptState;
}

/** `POST .../attempts/{target}/{index}/inpaint`: the canvas as the editor left
 * it (erased = redraw) -> a run, or the refusal as a value. */
export async function inpaintAttempt(
  unit: string, target: string, index: number, canvas: Blob,
  options: { prompt?: string; strength?: number; seed?: number; model?: string } = {},
): Promise<{ run: RunRecord } | { refused: RunRefusal }> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(options)) {
    if (value !== undefined && value !== null && value !== "") query.set(key, String(value));
  }
  const suffix = query.toString() ? `?${query}` : "";
  const response = await fetch(`${attemptPath(unit, target, index)}/inpaint${suffix}`, {
    method: "POST", headers: { "Content-Type": "image/png" }, body: canvas,
  });
  const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
  if (response.status === 202) return { run: payload as unknown as RunRecord };
  const error = (payload.error ?? payload.detail ?? response.statusText) as string;
  return {
    refused: {
      status: response.status,
      error: typeof error === "string" ? error : JSON.stringify(error),
      run: payload.run as RunRecord | undefined,
    },
  };
}

function attemptPath(unit: string, target: string, index: number): string {
  return `/concept/${encodeURIComponent(unit)}/attempts/${encodeURIComponent(target)}/${index}`;
}

/** `PUT .../attempts/{target}/{index}`: hide one attempt, or show it again. */
export const hideAttempt = (unit: string, target: string, index: number, hidden: boolean) =>
  sendJSON<ConceptState>("PUT", attemptPath(unit, target, index), { hidden });

/** `DELETE .../attempts/{target}/{index}`: unlink a hidden attempt's PNG. */
export async function deleteAttemptFile(unit: string, target: string, index: number): Promise<ConceptState> {
  const response = await fetch(attemptPath(unit, target, index), {
    method: "DELETE",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return (await response.json()) as ConceptState;
}

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

// --- the animate tab feature -------------------------------------------------
/** view = the direction's accepted view; keys = the approved keyframes either
 * side of the frame; both = the view and the two keys. */
export type DrawFrom = "view" | "keys" | "both";
export type FrameStatus = "empty" | "generating" | "review" | "approved" | "rejected";
export type Batch = "all" | "keys" | "rejected" | "inbetweens";

export interface AnimateImage {
  role: "view" | "before" | "after" | "skeleton";
  ref: string;
  path: string;
  sha256: string;
}

export interface AnimateSkeleton {
  frame: number;
  default_frame: number;
  frames: number;
  fps: number;
  time: number;
  clip: string;
  rig: string;
  file?: string;
}

export interface FrameNeighbour {
  id: string;
  position: number;
  label: string;
  approved: boolean;
}

export interface AnimateFrame {
  id: string;
  /** 1-based, what "Frame N" says. */
  position: number;
  label: string;
  key: boolean;
  phase: string | null;
  /** Where in the rig clip the frame samples, 0..1 (the template's instant). */
  at: number;
  status: FrameStatus;
  current: number | null;
  verdict: { state: "approved" | "rejected"; index: number; sha256: string; at: string } | null;
  prompt: string;
  default_prompt: string;
  draw_from: DrawFrom;
  before: FrameNeighbour | null;
  after: FrameNeighbour | null;
  default_skeleton_frame: number | null;
  skeleton_frame: number | null;
  candidates: Attempt[];
  hidden_count: number;
  game_url: string | null;
}

export interface CompassCell {
  direction: string;
  view: string;
  yaw: number;
  url: string | null;
  blocked: string | null;
  mirror_of: string | null;
  approved: number;
  frames: number;
  exists: boolean;
}

export interface AnimateState {
  unit: string;
  action: string;
  verb: string;
  direction: string;
  view: string;
  job: string;
  exists: boolean;
  loop: boolean;
  fps: number;
  height: number;
  max_frames: number;
  phases: string[];
  draw_from_options: DrawFrom[];
  max_images: number;
  compass: CompassCell[];
  actions: { action: string; verb: string; frames: number; loop: boolean; clip: string | null }[];
  /** Why frames may not be drawn now (the base gate), or null. */
  locked: string | null;
  reference: { view: string; sha256: string; url: string | null } | null;
  reference_error: string | null;
  rig: { rig: string; clip: string; duration: number; fps: number; frames: number; times: number[] } | null;
  rig_error: string | null;
  /** `/animate/.../skeleton/{frame}.png` - replace `{frame}`. */
  skeleton_url: string;
  frames: AnimateFrame[];
  counts: Record<FrameStatus | "total", number>;
  next_step: { step: "keys" | "inbetweens" | "done"; targets: number[]; waiting: number[] };
  mirror: {
    partner: string | null;
    partner_view: string | null;
    available: boolean;
    partner_pictures: number;
    last: { from: string; at: string; frames: number; warning: string } | null;
    warning: string | null;
  };
  output: { job: string; at: string | null; written: boolean; stale: boolean; ready: boolean; error: string | null };
  defaults: ConceptState["defaults"];
  last_seed: number | null;
  /** Present on the responses that did something worth a line. */
  warning?: string | null;
  mirrored?: number;
  approved?: string[];
  added?: string;
  written?: { job: string; frames: number; board: string };
  edited?: Attempt;
}

export interface AnimateRequest {
  frame?: string;
  batch?: Batch;
  prompt?: string;
  draw_from?: DrawFrom;
  skeleton?: boolean;
  skeleton_frame?: number;
  count?: number;
  seed?: number;
  model?: string;
  force?: boolean;
}

function animatePath(unit: string, action: string, direction: string): string {
  return `/animate/${encodeURIComponent(unit)}/${encodeURIComponent(action)}/${encodeURIComponent(direction)}`;
}

async function runOrRefusal(response: Response): Promise<{ run: RunRecord } | { refused: RunRefusal }> {
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

export const getAnimate = (unit: string, action: string, direction: string) =>
  getJSON<AnimateState>(animatePath(unit, action, direction));

/** `POST /animate/.../generate`: the run (202), or the refusal as a value. */
export async function generateAnimate(
  unit: string, action: string, direction: string, body: AnimateRequest,
): Promise<{ run: RunRecord } | { refused: RunRefusal }> {
  const response = await fetch(`${animatePath(unit, action, direction)}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  return runOrRefusal(response);
}

export const addAnimateFrame = (unit: string, action: string, direction: string, after: number) =>
  sendJSON<AnimateState>("POST", `${animatePath(unit, action, direction)}/frames`, { after });

export const putAnimateFrame = (unit: string, action: string, direction: string, frame: string,
  body: { key?: boolean; phase?: string | null; current?: number }) =>
  sendJSON<AnimateState>("PUT", `${animatePath(unit, action, direction)}/frames/${encodeURIComponent(frame)}`, body);

export async function removeAnimateFrame(unit: string, action: string, direction: string, frame: string): Promise<AnimateState> {
  const response = await fetch(`${animatePath(unit, action, direction)}/frames/${encodeURIComponent(frame)}`, {
    method: "DELETE", headers: { Accept: "application/json" },
  });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return (await response.json()) as AnimateState;
}

export const judgeAnimateFrame = (unit: string, action: string, direction: string, frame: string,
  verdict: "approve" | "reject", index?: number) =>
  sendJSON<AnimateState>("POST", `${animatePath(unit, action, direction)}/frames/${encodeURIComponent(frame)}/${verdict}`,
    index === undefined ? {} : { index });

export const approveReviewed = (unit: string, action: string, direction: string) =>
  sendJSON<AnimateState>("POST", `${animatePath(unit, action, direction)}/approve-reviewed`, {});

export const mirrorAnimate = (unit: string, action: string, direction: string) =>
  sendJSON<AnimateState>("POST", `${animatePath(unit, action, direction)}/mirror`, {});

export const outputAnimate = (unit: string, action: string, direction: string, replaceJob = false) =>
  sendJSON<AnimateState>("POST", `${animatePath(unit, action, direction)}/output`, { replace_job: replaceJob });

function animateAttemptPath(unit: string, action: string, direction: string, frame: string, index: number): string {
  return `${animatePath(unit, action, direction)}/attempts/${encodeURIComponent(frame)}/${index}`;
}

export const hideAnimateAttempt = (unit: string, action: string, direction: string, frame: string, index: number, hidden: boolean) =>
  sendJSON<AnimateState>("PUT", animateAttemptPath(unit, action, direction, frame, index), { hidden });

/** The editor's Save over one frame attempt. */
export async function putAnimateImage(unit: string, action: string, direction: string, frame: string, index: number, png: Blob): Promise<AnimateState> {
  const response = await fetch(`${animateAttemptPath(unit, action, direction, frame, index)}/image`, {
    method: "PUT", headers: { "Content-Type": "image/png" }, body: png,
  });
  if (!response.ok) throw new ApiError(response.status, await detail(response));
  return (await response.json()) as AnimateState;
}

/** The retouch feature's masked redraw on a frame attempt -> a run, or the refusal. */
export async function inpaintAnimate(
  unit: string, action: string, direction: string, frame: string, index: number, canvas: Blob,
  options: { prompt?: string; strength?: number; seed?: number } = {},
): Promise<{ run: RunRecord } | { refused: RunRefusal }> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(options)) {
    if (value !== undefined && value !== null && value !== "") query.set(key, String(value));
  }
  const suffix = query.toString() ? `?${query}` : "";
  const response = await fetch(`${animateAttemptPath(unit, action, direction, frame, index)}/inpaint${suffix}`, {
    method: "POST", headers: { "Content-Type": "image/png" }, body: canvas,
  });
  return runOrRefusal(response);
}
