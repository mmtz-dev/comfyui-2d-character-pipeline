// The animate tab feature's pure half: what the Animate tab derives from
// `GET /animate/{unit}` and `GET /animate/{unit}/{action}/{direction}` - the
// grid of every animation and what a sweep over ticked cells would draw, the
// frame row's statuses, Generate all's two steps, why a draw-from is refused,
// the request each button builds, and the player's frame list. No DOM, so it
// is testable; animate-view.ts draws it.

import type {
  AnimateFrame,
  AnimateGrid,
  AnimateRequest,
  AnimateState,
  Attempt,
  Batch,
  CellState,
  DrawFrom,
  FrameStatus,
  GridCell,
  SweepBatch,
} from "./api";
import { COUNT_MAX, seedFor } from "./concept";

/** The view keys feature: why a copied frame carries a "stale" mark. */
export const STALE_COPY = "a copy of an earlier view - the view has been re-accepted since. Copy it again, draw it, or keep it; nothing is replaced for you.";

export const STATUS_TEXT: Record<FrameStatus, string> = {
  empty: "empty",
  generating: "generating…",
  review: "to review",
  approved: "approved",
  rejected: "rejected",
};

// --- the grid and the sweep (the pipeline tabs feature) ---------------------
export const CELL_TEXT: Record<CellState, string> = {
  none: "not started",
  keys: "keys to review",
  keys_ok: "keys approved",
  done: "done",
};

/** A grid cell's short mark: nothing, K? (keys to review), K✓, ✓ or ⇋
 * (done, and mirrored from its partner), – when its direction is blocked. */
export function cellMark(cell: GridCell): string {
  if (cell.blocked && cell.state === "none") return "–";
  if (cell.state === "done") return cell.mirrored ? "⇋" : "✓";
  if (cell.state === "keys_ok") return "K✓";
  if (cell.state === "keys") return "K?";
  return "";
}

export function cellTitle(action: string, direction: string, view: string, cell: GridCell): string {
  const lines = [`${action} · ${direction} (${viewLabel(view)})`];
  if (cell.blocked) lines.push(`blocked - ${cell.blocked}`);
  else lines.push(CELL_TEXT[cell.state]);
  if (cell.exists) lines.push(`${cell.approved}/${cell.frames} frames approved, ${cell.keys_approved}/${cell.keys} keys`);
  lines.push("click to open · shift-click to tick for a sweep");
  return lines.join("\n");
}

export const cellKey = (action: string, direction: string) => `${action}:${direction}`;

/** Tick or untick one cell; a tick past the cap is refused with the reason. */
export function toggleTick(ticked: readonly string[], key: string, max: number
): { ticked: string[]; refused: string | null } {
  if (ticked.includes(key)) return { ticked: ticked.filter((k) => k !== key), refused: null };
  if (max && ticked.length >= max) {
    return { ticked: [...ticked], refused: `one sweep takes at most ${max} animations - untick one first` };
  }
  return { ticked: [...ticked, key], refused: null };
}

export interface SweepPlan {
  /** The ticked cells that would be drawn, in grid order. */
  cells: { action: string; direction: string }[];
  /** Roughly how many renders (one attempt per frame). */
  renders: number;
  /** Ticked cells the sweep would skip, and why - the keyframe gate among them. */
  blocked: { key: string; why: string }[];
}

/** What one sweep press over the ticked cells would draw. `keys`: every
 * keyframe not yet approved. `inbetweens`: the in-betweens of cells whose
 * keys are ALL approved - the keyframe gate; a cell short of it is blocked,
 * never drawn. Mirrors `anim.stages.animate.sweep`. */
export function sweepPlan(grid: Pick<AnimateGrid, "actions" | "directions" | "cells">,
                          ticked: readonly string[], batch: SweepBatch): SweepPlan {
  const plan: SweepPlan = { cells: [], renders: 0, blocked: [] };
  for (const action of grid.actions) {
    for (const { direction } of grid.directions) {
      const key = cellKey(action, direction);
      if (!ticked.includes(key)) continue;
      const cell = grid.cells[action]?.[direction];
      if (!cell) continue;
      if (cell.blocked) {
        plan.blocked.push({ key, why: cell.blocked });
        continue;
      }
      if (batch === "inbetweens" && cell.keys_approved < cell.keys) {
        plan.blocked.push({ key, why: `${cell.keys - cell.keys_approved} keyframe(s) not approved yet` });
        continue;
      }
      const renders = batch === "keys"
        ? cell.keys - cell.keys_approved
        : (cell.frames - cell.keys) - (cell.approved - cell.keys_approved);
      plan.cells.push({ action, direction });
      plan.renders += Math.max(0, renders);
    }
  }
  return plan;
}

/** The confirm line a sweep press asks before it books the card. */
export function sweepConfirm(unit: string, plan: SweepPlan, batch: SweepBatch): string {
  const what = batch === "keys" ? "keyframes" : "in-betweens";
  const lines = [`Generate the ${what} of ${plan.cells.length} animation(s) of ${unit} as one run?`,
    "", plan.cells.map((c) => cellKey(c.action, c.direction)).join(", "),
    `About ${plan.renders} render(s), one GPU job, one Cancel.`];
  if (plan.blocked.length) {
    lines.push("", `Skipped (${plan.blocked.length}): ${plan.blocked.map((b) => `${b.key} - ${b.why}`).join("; ")}`);
  }
  return lines.join("\n");
}

/** "back_left" -> "back left". */
export function viewLabel(view: string): string {
  return view.replace(/_/g, " ");
}

/** The attempt the frame shows, or null. */
export function currentAttempt(frame: AnimateFrame): Attempt | null {
  if (frame.current === null) return null;
  return frame.candidates.find((a) => a.index === frame.current) ?? null;
}

/** The picture of the frame's current attempt, cache-busted by its hash. */
export function frameImage(frame: AnimateFrame): string | null {
  const attempt = currentAttempt(frame);
  if (!attempt?.url || attempt.file_deleted) return null;
  return `${attempt.url}?t=${(attempt.sha256 ?? String(attempt.seconds ?? "")).slice(0, 12)}`;
}

/** "3 / 12 approved" */
export function approvedLine(frames: AnimateFrame[]): string {
  const approved = frames.filter((f) => f.status === "approved").length;
  return `${approved} / ${frames.length} approved`;
}

// --- Generate all: two steps with approval between -----------------------------
export interface NextStep {
  step: "keys" | "inbetweens" | "done";
  /** Positions (0-based) the press would draw: empty or rejected frames of the step's kind. */
  targets: number[];
  /** Positions of that kind still waiting on the owner. */
  waiting: number[];
}

/** The same rule as `animate.next_step`: while any keyframe is unapproved the
 * press draws the empty / rejected keyframes; once all are approved, the
 * empty / rejected in-betweens. Frames to review are never drawn again. */
export function nextStep(frames: AnimateFrame[]): NextStep {
  const keys = frames.map((f, i) => (f.key ? i : -1)).filter((i) => i >= 0);
  const unapproved = keys.filter((i) => frames[i]!.status !== "approved");
  const pool = unapproved.length ? keys : frames.map((f, i) => (f.key ? -1 : i)).filter((i) => i >= 0);
  const targets = pool.filter((i) => ["empty", "rejected"].includes(frames[i]!.status));
  const waiting = pool.filter((i) => ["review", "generating"].includes(frames[i]!.status));
  let step: NextStep["step"] = unapproved.length ? "keys" : "inbetweens";
  if (step === "inbetweens" && !targets.length && !waiting.length && frames.every((f) => f.status === "approved")) step = "done";
  return { step, targets, waiting };
}

export interface StepButton {
  label: string;
  /** The batch the press sends, or null when there is nothing to press. */
  batch: Batch | null;
  title: string;
}

/** Generate all's label names the next step, so the owner can see where the
 * row is: keyframes first, then (after approving them) the in-betweens. */
export function generateAllButton(frames: AnimateFrame[]): StepButton {
  const step = nextStep(frames);
  const n = step.targets.length;
  if (step.step === "done") {
    return { label: `All ${frames.length} frames approved`, batch: null, title: "every frame is approved" };
  }
  if (step.step === "keys") {
    if (n) {
      return { label: `Generate keyframes (${n})`, batch: "all",
        title: "step 1 of 2: draw every keyframe that is empty or rejected, then stop so you can approve them" };
    }
    return { label: "Approve the keyframes to go on", batch: null,
      title: `${step.waiting.length} keyframe(s) wait for your review; the in-betweens are drawn from approved keyframes` };
  }
  if (n) {
    return { label: `Fill the in-betweens (${n})`, batch: "all",
      title: "step 2 of 2: every keyframe is approved - draw each empty or rejected in-between from the keyframes either side" };
  }
  return { label: "Approve the in-betweens", batch: null, title: `${step.waiting.length} in-between(s) wait for your review` };
}

/** Positions the other toolbar batches would draw. */
export function batchTargets(frames: AnimateFrame[], batch: Batch): number[] {
  switch (batch) {
    case "all":
      return nextStep(frames).targets;
    case "keys":
      return frames.map((f, i) => (f.key && !["approved", "generating"].includes(f.status) ? i : -1)).filter((i) => i >= 0);
    case "rejected":
      return frames.map((f, i) => (f.status === "rejected" ? i : -1)).filter((i) => i >= 0);
    case "inbetweens":
      return frames.map((f, i) => (!f.key && ["empty", "rejected"].includes(f.status) ? i : -1)).filter((i) => i >= 0);
  }
}

// --- draw from -------------------------------------------------------------------
/** The images a draw-from sends, in order, as roles. */
export function imagesFor(frame: AnimateFrame, drawFrom: DrawFrom, skeleton: boolean): string[] {
  if (drawFrom === "copy") return ["view"];
  const roles: string[] = [];
  if (drawFrom === "view" || drawFrom === "both") roles.push("view");
  if (drawFrom === "keys" || drawFrom === "both") {
    roles.push("before");
    if (!(frame.before && frame.after && frame.before.id === frame.after.id)) roles.push("after");
  }
  if (skeleton) roles.push("skeleton");
  return roles;
}

/** Why drawing `frame` this way would be refused, or null - the stage's own
 * rules, said before the round trip. */
export function drawFromBlocked(state: AnimateState, frame: AnimateFrame, drawFrom: DrawFrom,
                                skeleton: boolean): string | null {
  if (state.locked) return state.locked;
  if (drawFrom === "copy" && !frame.key) return `${frame.label} is an in-between; only a keyframe can be a copy of the view`;
  if (drawFrom !== "keys" && !state.reference) {
    return state.reference_error ?? `no accepted ${viewLabel(state.view)} view to draw from`;
  }
  if (drawFrom === "keys" || drawFrom === "both") {
    for (const [side, key] of [["before", frame.before], ["after", frame.after]] as const) {
      if (!key) return `${frame.label} has no keyframe ${side} it - mark one, or draw from the view`;
      if (!key.approved) return `${key.label} (frame ${key.position}) is not approved yet - approve it first, or draw from the view`;
    }
  }
  const count = imagesFor(frame, drawFrom, skeleton).length;
  if (count > state.max_images) {
    return `${count} images; Qwen-Image-Edit takes ${state.max_images} - untick the skeleton, or draw from the keys or the view alone`;
  }
  return null;
}

/** "image1 the left view, image2 Wind-up, image3 Action, image4 skeleton" */
export function drawFromText(state: AnimateState, frame: AnimateFrame, drawFrom: DrawFrom, skeleton: boolean): string {
  if (drawFrom === "copy") return `the ${viewLabel(state.view)} view itself, copied as this frame's picture - no GPU`;
  return imagesFor(frame, drawFrom, skeleton).map((role, i) => {
    const what = role === "view" ? `the ${viewLabel(state.view)} view`
      : role === "before" ? `the approved ${frame.before?.label ?? "keyframe before"}`
        : role === "after" ? `the approved ${frame.after?.label ?? "keyframe after"}` : "the skeleton";
    return `image${i + 1}: ${what}`;
  }).join(", ");
}

// --- the pose slider ----------------------------------------------------------
export function skeletonUrl(state: AnimateState, frame: number): string {
  return state.skeleton_url.replace("{frame}", String(frame));
}

export function clampFrame(state: AnimateState, frame: number): number {
  const last = Math.max(0, (state.rig?.frames ?? 1) - 1);
  return Math.max(0, Math.min(last, Math.round(Number.isFinite(frame) ? frame : 0)));
}

/** Where the slider starts: the frame's last chosen instant, else the template's. */
export function startFrame(frame: AnimateFrame): number {
  return frame.skeleton_frame ?? frame.default_skeleton_frame ?? 0;
}

/** "clip frame 7 / 36 · 0.29 s (template)" */
export function sliderText(state: AnimateState, frame: AnimateFrame, value: number): string {
  const rig = state.rig;
  if (!rig) return "no rig clip for this action - no skeleton";
  const time = rig.times[value] ?? value / rig.fps;
  const own = frame.default_skeleton_frame === value ? " (template)" : "";
  return `clip frame ${value} / ${rig.frames - 1} · ${time.toFixed(2)} s${own}`;
}

// --- the request -------------------------------------------------------------------
export interface FrameAsk {
  frame: AnimateFrame;
  prompt: string;
  drawFrom: DrawFrom;
  skeleton: boolean;
  /** The slider's value; sent only when it is not where the frame would start anyway. */
  skeletonFrame: number | null;
  model: string | null;
  lockSeed: boolean;
  lastSeed: number | null;
  count?: number;
}

/** The `POST /animate/.../generate` body for "Generate this frame"; throws on
 * an ask that would be a 400, so the tab says so before the round trip. */
export function frameBody(ask: FrameAsk): AnimateRequest {
  const prompt = ask.prompt.trim();
  if (!prompt) throw new Error("a frame needs a prompt: what the pose is");
  const body: AnimateRequest = { frame: ask.frame.id, prompt, draw_from: ask.drawFrom };
  if (!ask.skeleton) body.skeleton = false;
  else if (ask.skeletonFrame !== null && ask.skeletonFrame !== ask.frame.default_skeleton_frame) body.skeleton_frame = ask.skeletonFrame;
  else body.skeleton = true;
  const seed = seedFor(ask.lockSeed, ask.lastSeed);
  if (seed !== undefined) body.seed = seed;
  if (ask.model) body.model = ask.model;
  if (ask.count && ask.count > 1) body.count = Math.min(COUNT_MAX, Math.round(ask.count));
  return body;
}

/** A batch press: each frame keeps its own prompt, draw-from and instant. */
export function batchBody(batch: Batch, model: string | null): AnimateRequest {
  const body: AnimateRequest = { batch };
  if (model) body.model = model;
  return body;
}

// --- the click folded in before the API answers ------------------------------------
export type AnimateAction =
  | { type: "judge"; frame: string; verdict: "approved" | "rejected"; index: number }
  | { type: "select"; frame: string; index: number }
  | { type: "generating"; frames: string[] };

export function reduceAnimate(state: AnimateState, action: AnimateAction): AnimateState {
  const frames = state.frames.map((frame): AnimateFrame => {
    switch (action.type) {
      case "judge":
        if (frame.id !== action.frame) return frame;
        return { ...frame, current: action.index, status: action.verdict,
          verdict: { state: action.verdict, index: action.index, sha256: "", at: "" } };
      case "select": {
        if (frame.id !== action.frame) return frame;
        const judged = frame.verdict && frame.verdict.index === action.index;
        return { ...frame, current: action.index, status: judged ? frame.verdict!.state : "review" };
      }
      case "generating":
        return action.frames.includes(frame.id) ? { ...frame, status: "generating" } : frame;
    }
  });
  return { ...state, frames };
}

// --- the player ------------------------------------------------------------------
export interface PlayFrame {
  id: string;
  label: string;
  position: number;
  full: string;
  game: string | null;
}

/** The frames the player cycles through: each frame's current picture; with
 * `approvedOnly`, only approved frames. A frame with no picture is skipped. */
export function playList(frames: AnimateFrame[], approvedOnly: boolean): PlayFrame[] {
  const out: PlayFrame[] = [];
  for (const frame of frames) {
    if (approvedOnly && frame.status !== "approved") continue;
    const full = frameImage(frame);
    if (!full) continue;
    out.push({ id: frame.id, label: frame.label, position: frame.position, full, game: frame.game_url });
  }
  return out;
}

/** The next index to show, or null when a one-shot play has reached the end. */
export function nextPlayIndex(index: number, length: number, loop: boolean): number | null {
  if (length <= 0) return null;
  const next = index + 1;
  if (next < length) return next;
  return loop ? 0 : null;
}

export function playInterval(fps: number): number {
  return Math.round(1000 / Math.max(1, Math.min(60, fps || 12)));
}

// --- mirror -------------------------------------------------------------------------
export interface MirrorOffer {
  label: string;
  disabled: boolean;
  title: string;
}

/** "Mirror from left" - only for the three pairs, and only a press. */
export function mirrorOffer(state: AnimateState): MirrorOffer | null {
  const m = state.mirror;
  if (!m.partner) return null;
  const from = `${viewLabel(m.partner_view ?? m.partner)} (${m.partner})`;
  if (!m.available) {
    return { label: `Mirror from ${from}`, disabled: true, title: `${from} has no frames with a picture yet - nothing to flip` };
  }
  return {
    label: `Mirror from ${from}`,
    disabled: false,
    title: `flip ${m.partner_pictures} frame(s) of ${from} into ${state.direction} - no GPU; every flipped frame comes back to review, never approved.\n${m.warning ?? ""}`,
  };
}

/** A frame's caption under its tile: "from keys + skeleton 14 · seed 5". */
export function attemptOrigin(attempt: Attempt): string {
  if (attempt.source === "copy") return `copy of the ${viewLabel(attempt.copy?.view ?? "")} view`;
  if (attempt.source === "mirror") return `mirror of ${attempt.mirror?.direction ?? "partner"} ${attempt.mirror?.ref ?? ""}`.trim();
  const parts: string[] = [];
  if (attempt.draw_from) parts.push(`from ${attempt.draw_from}`);
  if (attempt.skeleton) parts.push(`skeleton ${attempt.skeleton.frame}`);
  if (attempt.edited) parts.push(`edited (${attempt.edited.how})`);
  return parts.join(" + ");
}
