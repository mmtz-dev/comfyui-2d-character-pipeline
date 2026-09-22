// The concept generation feature's pure half: what the Concept tab derives
// from `GET /concept/{unit}` and `GET /models`, and the request each button
// builds. No DOM, so it is testable; concept-view.ts draws it.

import type {
  Attempt,
  ConceptOp,
  ConceptRequest,
  ConceptState,
  ConceptView,
  LoraChoice,
  ModelFamily,
  ModelsListing,
  RegistryRow,
  RunRecord,
} from "./api";

/** The buttons above the views: Front is the accepted base itself (never
 * rendered), Side is the left profile (the right is mirrored in a sheet),
 * and "all" is the main group in one press. */
export const MAIN_VIEWS: readonly string[] = ["back", "left", "top"];
export const VIEW_LABELS: Record<string, string> = {
  front: "Front",
  back: "Back",
  left: "Side",
  top: "Top-down",
  right: "Right",
  back_right: "Back-right",
  back_left: "Back-left",
  front_left: "Front-left",
  front_right: "Front-right",
};
export const ACTIVE_STATUSES: readonly string[] = ["queued", "running"];
/** The strength slider's band for classic img2img: below 0.3 nothing moves,
 * above 0.9 the composition is gone and it is txt2img with extra steps. */
export const STRENGTH_MIN = 0.3;
export const STRENGTH_MAX = 0.9;
export const STRENGTH_DEFAULT = 0.55;
export const COUNT_MAX = 4;

export interface ViewGroups {
  main: ConceptView[];
  advanced: ConceptView[];
}

/** The views in their two groups, in the API's (compass) order. */
export function viewGroups(state: ConceptState): ViewGroups {
  const views = Object.values(state.views);
  return {
    main: views.filter((view) => view.group === "main"),
    advanced: views.filter((view) => view.group !== "main"),
  };
}

export function viewLabel(view: string): string {
  return VIEW_LABELS[view] ?? view;
}

/** Which family a generate op draws its model and LoRAs from. */
export function familyOf(op: ConceptOp, listing: ModelsListing | null): ModelFamily {
  return listing?.ops[op] ?? (op === "views" || op === "edit" ? "qwen-edit" : "zimage");
}

/** The seed a press sends: the last one when locked, none (random) otherwise. */
export function seedFor(lock: boolean, lastSeed: number | null): number | undefined {
  return lock && lastSeed !== null ? lastSeed : undefined;
}

/** What the settings row holds for one press. */
export interface GenerateForm {
  count: number;
  lockSeed: boolean;
  lastSeed: number | null;
  /** null = the workflow's own (or the unit's default). */
  model: string | null;
  /** null = untouched (the unit's default, else the pixel LoRA); [] = none. */
  loras: LoraChoice[] | null;
}

export interface GenerateAsk {
  op: ConceptOp;
  prompt?: string;
  views?: string[];
  viewPrompt?: string;
  ref?: string;
  strength?: number;
  instruction?: string;
  force?: boolean;
}

function clampCount(count: number): number {
  return Math.max(1, Math.min(COUNT_MAX, Math.round(count) || 1));
}

/** The `POST /concept/{unit}/generate` body for one press. Throws on an
 * ask that would be a 400, so the tab can say so before the round trip. */
export function generateBody(ask: GenerateAsk, form: GenerateForm): ConceptRequest {
  const body: ConceptRequest = { op: ask.op, count: clampCount(form.count) };
  const seed = seedFor(form.lockSeed, form.lastSeed);
  if (seed !== undefined) body.seed = seed;
  if (form.model) body.model = form.model;
  if (form.loras !== null) {
    body.loras = form.loras.filter((lora) => lora.file && lora.strength > 0);
    if (form.loras.length && !body.loras.length) body.lora_strength = 0;
  }
  if (ask.force) body.force = true;
  switch (ask.op) {
    case "txt2img": {
      const prompt = (ask.prompt ?? "").trim();
      if (!prompt) throw new Error("the base needs a prompt: who the character is");
      body.prompt = prompt;
      return body;
    }
    case "views": {
      const views = (ask.views ?? []).filter(Boolean);
      if (!views.length) throw new Error("pick at least one view");
      body.views = views;
      const viewPrompt = (ask.viewPrompt ?? "").trim();
      if (viewPrompt) {
        if (views.length !== 1) throw new Error("a view prompt goes with one view");
        body.view_prompt = viewPrompt;
      }
      return body;
    }
    case "img2img": {
      if (!ask.ref) throw new Error("img2img needs an attempt to start from");
      body.ref = ask.ref;
      const strength = ask.strength ?? STRENGTH_DEFAULT;
      if (!(strength > 0 && strength <= 1)) throw new Error("strength is between 0 and 1");
      body.strength = Math.round(strength * 100) / 100;
      const prompt = (ask.prompt ?? "").trim();
      if (prompt) body.prompt = prompt;
      return body;
    }
    case "edit": {
      if (!ask.ref) throw new Error("an edit needs an attempt to start from");
      const instruction = (ask.instruction ?? "").trim();
      if (!instruction) throw new Error("an edit needs an instruction: what should change");
      body.ref = ask.ref;
      body.instruction = instruction;
      return body;
    }
  }
}

/** `base:03` / `left:01` - how the API names an attempt. */
export function refOf(target: string, attempt: Attempt): string {
  return `${target}:${String(attempt.index).padStart(2, "0")}`;
}

/** A unit slug from whatever the owner typed: "Cloaked Assassin" ->
 * "cloaked_assassin". The rule is the pipeline's own (`export.UNIT_RE`,
 * `jobs_api.JOB_NAME`): lower case, starting with a letter or a digit, then
 * letters, digits, `-` and `_`. Anything else becomes `_`, runs collapse, and
 * the ends are trimmed; "" means there was nothing usable in it. Slugifying
 * rather than refusing, because a unit's name is the owner's words and the
 * slug is the pipeline's filename for them. */
export function unitSlug(typed: string): string {
  return typed
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/_{2,}/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "");
}

/** Why this attempt may not be hidden or deleted, or null. */
export function lockedReason(state: ConceptState, target: string, attempt: Attempt): string | null {
  const accepted = target === "base" ? state.base.pick : state.views[target]?.pick ?? null;
  if (accepted === attempt.index) {
    return target === "base"
      ? "the accepted base - every later stage reads it; accept another first"
      : "this view's accepted attempt - the turnaround reads it; accept another first";
  }
  return null;
}

export type ConceptAction =
  | { type: "pick"; index: number }
  | { type: "pickView"; view: string; index: number }
  | { type: "viewPrompt"; view: string; prompt: string };

/** The click folded into the state before the API answers, so the outline
 * moves at once. Never mutates its input. */
export function reduceConcept(state: ConceptState, action: ConceptAction): ConceptState {
  switch (action.type) {
    case "pick":
      return {
        ...state,
        locked: null,
        gates: { ...state.gates, base: "accepted", turnaround: "pending" },
        base: { ...state.base, pick: action.index, accepted: true },
        views: Object.fromEntries(Object.entries(state.views).map(([name, view]) =>
          [name, { ...view, candidates: [], pick: null, state: "pending" as const, accepted_file: null }])),
        views_stale: false,
      };
    case "pickView": {
      const view = state.views[action.view];
      if (!view) return state;
      return {
        ...state,
        views: { ...state.views, [action.view]: { ...view, pick: action.index, state: "accepted" } },
      };
    }
    case "viewPrompt": {
      const view = state.views[action.view];
      if (!view) return state;
      return { ...state, views: { ...state.views, [action.view]: { ...view, prompt: action.prompt } } };
    }
  }
}

/** The attempts a strip shows: what was kept, or everything when the "show
 * hidden" toggle is on. */
export function visibleAttempts(attempts: Attempt[], showHidden: boolean): Attempt[] {
  return showHidden ? attempts : attempts.filter((attempt) => !attempt.hidden);
}

export function hiddenCount(attempts: Attempt[]): number {
  return attempts.filter((attempt) => attempt.hidden).length;
}

/** The line under a tile: where it came from and with what. */
export function attemptCaption(attempt: Attempt): string {
  const parts: string[] = [];
  const origin = attempt.parent
    ? `${attempt.source} of ${attempt.parent.target}:${String(attempt.parent.index).padStart(2, "0")}`
    : attempt.source;
  parts.push(origin);
  if (attempt.source === "img2img" && typeof attempt.strength === "number") parts.push(`× ${attempt.strength}`);
  if (attempt.seed !== null) parts.push(`seed ${attempt.seed}`);
  if (!attempt.rendered) parts.push("waiting for the GPU");
  else if (typeof attempt.seconds === "number") parts.push(`${attempt.seconds.toFixed(1)} s`);
  if (attempt.edited) parts.push(`edited (${attempt.edited.how})`);
  if (attempt.file_deleted) parts.push("file deleted");
  else if (attempt.hidden) parts.push("hidden");
  return parts.join(" · ");
}

/** What the tab says while a run holds the GPU. */
export function busyText(run: RunRecord | null): string {
  if (!run) return "";
  return `GPU: ${run.stage} ${run.id}${run.percent !== null ? ` ${run.percent}%` : ""} - the generate buttons wait for it`;
}

export function isActive(run: RunRecord | null): boolean {
  return run !== null && ACTIVE_STATUSES.includes(run.status);
}

/** The checkpoints and LoRAs the picker offers for a family, the LoRAs with
 * their registry default strength; a row ComfyUI cannot see (probed) is
 * still offered but flagged, so the owner learns before the run does. */
export function choicesFor(listing: ModelsListing | null, family: ModelFamily): { checkpoints: RegistryRow[]; loras: RegistryRow[] } {
  const rows = listing?.families[family];
  return { checkpoints: rows?.checkpoint ?? [], loras: rows?.lora ?? [] };
}

export function rowLabel(row: RegistryRow): string {
  const flags: string[] = [];
  if (row.installed === false) flags.push("not installed");
  if (!row.locked) flags.push("unlocked");
  return flags.length ? `${row.label} (${flags.join(", ")})` : row.label;
}

/** A file ComfyUI told us it does not have cannot be chosen: it would be a
 * 400 on the whole graph at `/prompt`, after the inputs went up. `null`
 * (unprobed) stays choosable - the stage checks again before it posts. */
export function choosable(row: RegistryRow): boolean {
  return row.installed !== false;
}

/** The LoRAs a family can actually offer right now. */
export function offerableLoras(listing: ModelsListing | null, family: ModelFamily): RegistryRow[] {
  return choicesFor(listing, family).loras.filter(choosable);
}

/** The form's starting LoRA list for a family: the unit's default when it
 * has one, else the workflow's (the pixel LoRA for Z-Image, none for Qwen). */
export function initialLoras(state: ConceptState | null, listing: ModelsListing | null, family: ModelFamily): LoraChoice[] {
  const saved = state?.defaults[family];
  if (saved?.loras) return saved.loras.map((lora) => ({ ...lora }));
  return (listing?.default_lora[family] ?? []).map((lora) => ({ ...lora }));
}

/** The LoRAs of a form that the GPU service does not have - what a press
 * would be refused for, named before it is pressed. */
export function notInstalled(listing: ModelsListing | null, family: ModelFamily,
                             model: string | null, loras: LoraChoice[]): string[] {
  const rows = choicesFor(listing, family);
  const missing: string[] = [];
  const check = (file: string, from: RegistryRow[]) => {
    const row = from.find((r) => r.file === file);
    if (row && row.installed === false) missing.push(row.label);
  };
  if (model) check(model, rows.checkpoints);
  for (const lora of loras) check(lora.file, rows.loras);
  return missing;
}

export function initialModel(state: ConceptState | null, family: ModelFamily): string | null {
  return state?.defaults[family]?.model ?? null;
}

/** Whether the form differs from the unit's saved default (the "save" button lights up). */
export function sameAsDefault(state: ConceptState | null, family: ModelFamily, model: string | null, loras: LoraChoice[]): boolean {
  const saved = state?.defaults[family];
  if (!saved) return false;
  if ((saved.model ?? null) !== (model ?? null)) return false;
  if (saved.loras.length !== loras.length) return false;
  return saved.loras.every((lora, i) => lora.file === loras[i]!.file && lora.strength === loras[i]!.strength);
}
