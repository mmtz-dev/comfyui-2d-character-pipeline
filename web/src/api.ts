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

/** What `bakeoff` and an animate batch or sweep accomplished, in cells. Null for
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

/** `GET /units`: every unit the studio knows - what the unit bar lists. */
export const listUnits = async () => (await getJSON<{ units: string[] }>("/units")).units;

// --- the concept generation feature ------------------------------------------
export type AttemptSource = "txt2img" | "upload" | "existing" | "img2img" | "edit" | "views" | "animate" | "mirror" | "copy";
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
  /** The view keys feature: a copy of the direction's view, which view and its hash. */
  copy?: { view: string; sha256: string } | null;
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
  /** Whether `post --parts` has measured the part palettes off the accepted
   * base, and the job it runs on (the Views tab's "measure part palettes"). */
  base_parts: boolean;
  base_parts_job: string;
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
export type DrawFrom = "view" | "keys" | "both" | "copy";
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
  /** The view keys feature: a keyframe whose picture is, by default, a copy of the view. */
  view_key: boolean;
  /** Its picture is a copy of a view that has since been re-accepted. */
  stale: boolean;
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
  written?: { job: string; frames: number };
  edited?: Attempt;
}

/** One animation as a cell of the Animate tab's grid (the pipeline tabs
 * feature). `none`: no picture yet; `keys`: a keyframe is not approved;
 * `keys_ok`: every keyframe is approved, the in-betweens may be drawn;
 * `done`: every frame is approved. A cell not started counts the template. */
export type CellState = "none" | "keys" | "keys_ok" | "done";

export interface GridCell {
  state: CellState;
  frames: number;
  approved: number;
  keys: number;
  keys_approved: number;
  pictures: number;
  exists: boolean;
  mirrored: boolean;
  /** Why this direction may not be drawn (its view), or null. */
  blocked: string | null;
}

export interface AnimateGrid {
  unit: string;
  actions: string[];
  directions: Omit<CompassCell, "approved" | "frames" | "exists">[];
  /** `cells[action][direction]`. */
  cells: Record<string, Record<string, GridCell>>;
  locked: string | null;
  max_cells: number;
  states: CellState[];
}

export type SweepBatch = "keys" | "inbetweens";

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

export const getAnimateGrid = (unit: string) => getJSON<AnimateGrid>(`/animate/${encodeURIComponent(unit)}`);

/** `POST /animate/{unit}/sweep`: ONE run over several animations (the
 * keyframe sweep feature), or the refusal as a value. */
export async function sweepAnimate(
  unit: string, cells: { action: string; direction: string }[], batch: SweepBatch, model: string | null,
): Promise<{ run: RunRecord } | { refused: RunRefusal }> {
  const response = await fetch(`/animate/${encodeURIComponent(unit)}/sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ cells, batch, ...(model ? { model } : {}) }),
  });
  return runOrRefusal(response);
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
