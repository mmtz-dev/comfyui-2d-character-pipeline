// The animate tab feature's pure half: what the Animate tab derives from
// `GET /animate/{unit}/{action}/{direction}` - the compass, the frame row's
// statuses, Generate all's two steps, why a draw-from is refused, the request
// each button builds, and the player's frame list. No DOM, so it is testable;
// animate-view.ts draws it.

import type {
  AnimateFrame,
  AnimateRequest,
  AnimateState,
  Attempt,
  Batch,
  CompassCell,
  DrawFrom,
  FrameStatus,
} from "./api";
import { COUNT_MAX, seedFor } from "./concept";

export const STATUS_TEXT: Record<FrameStatus, string> = {
  empty: "empty",
  generating: "generating…",
  review: "to review",
  approved: "approved",
  rejected: "rejected",
};

/** The compass as the mockup draws it: a 3 × 3 grid, the character in the
 * middle, seen from above - back row away from the viewer, front row toward. */
export const COMPASS_LAYOUT: readonly (string | null)[] = [
  "s135", "s90", "s45",
  "s180", null, "s0",
  "s225", "s270", "s315",
];

export function compassGrid(state: Pick<AnimateState, "compass">): (CompassCell | null)[] {
  return COMPASS_LAYOUT.map((direction) =>
    direction === null ? null : state.compass.find((c) => c.direction === direction) ?? null);
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
  if ((drawFrom === "view" || drawFrom === "both") && !state.reference) {
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
  if (attempt.source === "mirror") return `mirror of ${attempt.mirror?.direction ?? "partner"} ${attempt.mirror?.ref ?? ""}`.trim();
  const parts: string[] = [];
  if (attempt.draw_from) parts.push(`from ${attempt.draw_from}`);
  if (attempt.skeleton) parts.push(`skeleton ${attempt.skeleton.frame}`);
  if (attempt.edited) parts.push(`edited (${attempt.edited.how})`);
  return parts.join(" + ");
}
