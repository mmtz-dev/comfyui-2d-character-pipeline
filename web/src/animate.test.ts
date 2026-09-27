import { describe, expect, it } from "vitest";
import type { AnimateFrame, AnimateState, Attempt, FrameStatus } from "./api";
import {
  approvedLine,
  attemptOrigin,
  batchBody,
  batchTargets,
  clampFrame,
  compassGrid,
  drawFromBlocked,
  drawFromText,
  frameBody,
  frameImage,
  generateAllButton,
  imagesFor,
  mirrorOffer,
  nextPlayIndex,
  nextStep,
  playInterval,
  playList,
  reduceAnimate,
  skeletonUrl,
  sliderText,
  startFrame,
} from "./animate";

function attempt(index: number, extra: Partial<Attempt> = {}): Attempt {
  return {
    index, file: `f01_${String(index).padStart(2, "0")}.png`, source: "animate", prompt: "p", seed: 1,
    model: null, loras: [], parent: null, seconds: 10, sha256: `sha${index}abcdefghijk`, hidden: false,
    file_deleted: null, url: `/jobs/j/frames/animate/${index}.png`, rendered: true, ...extra,
  };
}

/** An attack-shaped row: 12 frames, keys at 0/2/6/9/11, in the given statuses. */
function frames(statuses: FrameStatus[], keys = [0, 2, 6, 9, 11]): AnimateFrame[] {
  const phases = ["start", "wind-up", "action", "wind-down", "end"];
  const rows: AnimateFrame[] = statuses.map((status, i) => {
    const key = keys.includes(i);
    const pictured = status !== "empty" && status !== "generating";
    return {
      id: `f${String(i + 1).padStart(2, "0")}`, position: i + 1,
      label: key ? phases[keys.indexOf(i)]!.replace(/^./, (c) => c.toUpperCase()) : `Frame ${i + 1}`,
      key, phase: key ? phases[keys.indexOf(i)]! : null, at: i / 11, status,
      current: pictured ? 0 : null,
      verdict: status === "approved" || status === "rejected" ? { state: status, index: 0, sha256: "x", at: "t" } : null,
      prompt: "pose", default_prompt: "pose", draw_from: key ? "view" : "keys",
      before: null, after: null, default_skeleton_frame: i * 3, skeleton_frame: null,
      candidates: pictured ? [attempt(0)] : [], hidden_count: 0,
      game_url: pictured ? `/animate/u/attack/s180/game/f${i + 1}.png` : null,
    };
  });
  // neighbours, the way the API fills them
  rows.forEach((frame, i) => {
    const before = [...keys].reverse().find((k) => k < i);
    const after = keys.find((k) => k > i);
    const n = (k: number | undefined) => k === undefined ? null
      : { id: rows[k]!.id, position: k + 1, label: rows[k]!.label, approved: rows[k]!.status === "approved" };
    frame.before = n(before);
    frame.after = n(after);
  });
  return rows;
}

function state(rows: AnimateFrame[], extra: Partial<AnimateState> = {}): AnimateState {
  return {
    unit: "hooded_archer", action: "attack", verb: "strike", direction: "s180", view: "left",
    job: "hooded_archer_animate_attack_s180", exists: true, loop: false, fps: 12, height: 48,
    max_frames: 24, phases: ["start", "wind-up", "action", "wind-down", "end"],
    draw_from_options: ["view", "keys", "both"], max_images: 3,
    compass: ["s0", "s45", "s90", "s135", "s180", "s225", "s270", "s315"].map((direction) => ({
      direction, view: direction, yaw: 0, url: null, blocked: null, mirror_of: null, approved: 0, frames: 0, exists: false,
    })),
    actions: [], locked: null, reference: { view: "left", sha256: "abc", url: "/x.png" }, reference_error: null,
    rig: { rig: "quaternius", clip: "Sword_Attack", duration: 1.5, fps: 24, frames: 37, times: Array.from({ length: 37 }, (_, i) => i / 24) },
    rig_error: null, skeleton_url: "/animate/u/attack/s180/skeleton/{frame}.png",
    frames: rows, counts: { empty: 0, generating: 0, review: 0, approved: 0, rejected: 0, total: rows.length },
    next_step: { step: "keys", targets: [], waiting: [] },
    mirror: { partner: "s0", partner_view: "right", available: true, partner_pictures: 12, last: null, warning: "the flip swaps handedness" },
    output: { job: "hooded_archer_strike_s180", at: null, written: false, stale: false, ready: false, error: null },
    defaults: {}, last_seed: 41, ...extra,
  };
}

const E: FrameStatus = "empty";
const A: FrameStatus = "approved";
const R: FrameStatus = "review";
const X: FrameStatus = "rejected";

describe("the compass", () => {
  it("lays the eight side views out as seen from above", () => {
    const grid = compassGrid(state(frames(Array(12).fill(E))));
    expect(grid.map((c) => c?.direction ?? null)).toEqual(["s135", "s90", "s45", "s180", null, "s0", "s225", "s270", "s315"]);
  });
});

describe("Generate all: two steps with approval between", () => {
  it("draws the empty and rejected keyframes first, and not the to-review ones", () => {
    const row = frames([E, E, R, E, E, E, X, E, E, E, E, E]);
    expect(nextStep(row)).toEqual({ step: "keys", targets: [0, 6, 9, 11], waiting: [2] });
    const button = generateAllButton(row);
    expect(button.label).toBe("Generate keyframes (4)");
    expect(button.batch).toBe("all");
  });

  it("stops while keyframes wait for review, then fills the in-betweens once all are approved", () => {
    const waiting = frames([A, E, R, E, E, E, A, E, E, A, E, A]);
    expect(generateAllButton(waiting)).toMatchObject({ label: "Approve the keyframes to go on", batch: null });
    const ready = frames([A, E, A, X, E, E, A, R, E, A, E, A]);
    expect(nextStep(ready)).toEqual({ step: "inbetweens", targets: [1, 3, 4, 5, 8, 10], waiting: [7] });
    expect(generateAllButton(ready)).toMatchObject({ label: "Fill the in-betweens (6)", batch: "all" });
    expect(generateAllButton(frames(Array(12).fill(A)))).toMatchObject({ label: "All 12 frames approved", batch: null });
  });

  it("the other presses", () => {
    const row = frames([A, E, R, X, E, E, X, E, E, E, E, A]);
    expect(batchTargets(row, "keys")).toEqual([2, 6, 9]);
    expect(batchTargets(row, "rejected")).toEqual([3, 6]);
    expect(batchTargets(row, "inbetweens")).toEqual([1, 3, 4, 5, 7, 8, 10]);
    expect(batchBody("rejected", "q.safetensors")).toEqual({ batch: "rejected", model: "q.safetensors" });
    expect(approvedLine(row)).toBe("2 / 12 approved");
  });
});

describe("draw from", () => {
  it("an in-between is refused until both neighbouring keyframes are approved", () => {
    const row = frames([A, E, R, E, E, E, E, E, E, E, E, E]);
    const s = state(row);
    expect(drawFromBlocked(s, row[1]!, "keys", false)).toMatch(/Wind-up \(frame 3\) is not approved yet/);
    expect(drawFromBlocked(s, row[1]!, "view", true)).toBeNull();
    const ok = frames([A, E, A, E, E, E, E, E, E, E, E, E]);
    expect(drawFromBlocked(state(ok), ok[1]!, "keys", true)).toBeNull();
    expect(imagesFor(ok[1]!, "keys", true)).toEqual(["before", "after", "skeleton"]);
    expect(drawFromText(state(ok), ok[1]!, "keys", false)).toBe("image1: the approved Start, image2: the approved Wind-up");
  });

  it("both + skeleton is four images and Qwen-Image-Edit takes three", () => {
    const row = frames([A, E, A, E, E, E, E, E, E, E, E, E]);
    expect(imagesFor(row[1]!, "both", true)).toEqual(["view", "before", "after", "skeleton"]);
    expect(drawFromBlocked(state(row), row[1]!, "both", true)).toMatch(/4 images; Qwen-Image-Edit takes 3/);
    expect(drawFromBlocked(state(row), row[1]!, "both", false)).toBeNull();
  });

  it("a view that is not accepted, or a locked unit, blocks the view", () => {
    const row = frames(Array(12).fill(E));
    const noView = state(row, { reference: null, reference_error: "the left view is pending" });
    expect(drawFromBlocked(noView, row[2]!, "view", false)).toBe("the left view is pending");
    expect(drawFromBlocked(state(row, { locked: "no base" }), row[2]!, "view", false)).toBe("no base");
  });
});

describe("the request", () => {
  it("sends the slider only when it moved off the template", () => {
    const row = frames(Array(12).fill(E));
    const frame = row[2]!;
    const base = { frame, prompt: " winding up ", drawFrom: "view" as const, skeleton: true, model: null, lockSeed: false, lastSeed: 41 };
    expect(frameBody({ ...base, skeletonFrame: frame.default_skeleton_frame })).toEqual({
      frame: "f03", prompt: "winding up", draw_from: "view", skeleton: true });
    expect(frameBody({ ...base, skeletonFrame: 20, lockSeed: true, model: "q" })).toEqual({
      frame: "f03", prompt: "winding up", draw_from: "view", skeleton_frame: 20, seed: 41, model: "q" });
    expect(frameBody({ ...base, skeleton: false, skeletonFrame: 20 })).toMatchObject({ skeleton: false });
    expect(() => frameBody({ ...base, prompt: "  ", skeletonFrame: null })).toThrow(/what the pose is/);
  });

  it("the pose slider", () => {
    const row = frames(Array(12).fill(E));
    const s = state(row);
    expect(clampFrame(s, 99)).toBe(36);
    expect(clampFrame(s, -3)).toBe(0);
    expect(skeletonUrl(s, 7)).toBe("/animate/u/attack/s180/skeleton/7.png");
    expect(startFrame(row[3]!)).toBe(9);
    expect(startFrame({ ...row[3]!, skeleton_frame: 4 })).toBe(4);
    expect(sliderText(s, row[3]!, 9)).toBe("clip frame 9 / 36 · 0.38 s (template)");
    expect(sliderText(state(row, { rig: null }), row[3]!, 9)).toMatch(/no rig clip/);
  });
});

describe("clicks folded in before the API answers", () => {
  it("approve, reject and swapping in another attempt", () => {
    const row = frames([R, E, E, E, E, E, E, E, E, E, E, E]);
    const s = state(row);
    const approved = reduceAnimate(s, { type: "judge", frame: "f01", verdict: "approved", index: 0 });
    expect(approved.frames[0]!.status).toBe("approved");
    expect(s.frames[0]!.status).toBe("review");
    const other = reduceAnimate(approved, { type: "select", frame: "f01", index: 3 });
    expect(other.frames[0]!.status).toBe("review");
    const back = reduceAnimate(other, { type: "select", frame: "f01", index: 0 });
    expect(back.frames[0]!.status).toBe("approved");
    const busy = reduceAnimate(s, { type: "generating", frames: ["f02", "f03"] });
    expect(busy.frames.map((f) => f.status).slice(0, 4)).toEqual(["review", "generating", "generating", "empty"]);
  });
});

describe("the player", () => {
  it("plays the approved frames, or every frame with a picture", () => {
    const row = frames([A, R, A, E, E, E, E, E, E, E, E, E]);
    expect(playList(row, true).map((f) => f.id)).toEqual(["f01", "f03"]);
    expect(playList(row, false).map((f) => f.id)).toEqual(["f01", "f02", "f03"]);
    expect(frameImage(row[0]!)).toBe("/jobs/j/frames/animate/0.png?t=sha0abcdefgh");
    expect(playList(row, true)[0]!.game).toBe("/animate/u/attack/s180/game/f1.png");
  });

  it("loops or stops", () => {
    expect(nextPlayIndex(0, 3, true)).toBe(1);
    expect(nextPlayIndex(2, 3, true)).toBe(0);
    expect(nextPlayIndex(2, 3, false)).toBeNull();
    expect(nextPlayIndex(-1, 0, true)).toBeNull();
    expect(playInterval(12)).toBe(83);
    expect(playInterval(0)).toBe(83);
  });
});

describe("mirror", () => {
  it("is offered for the pairs only, and says it comes back to review", () => {
    const row = frames(Array(12).fill(E));
    const offer = mirrorOffer(state(row))!;
    expect(offer.label).toBe("Mirror from right (s0)");
    expect(offer.disabled).toBe(false);
    expect(offer.title).toMatch(/review, never approved/);
    expect(mirrorOffer(state(row, { mirror: { ...state(row).mirror, available: false, partner_pictures: 0 } }))!.disabled).toBe(true);
    expect(mirrorOffer(state(row, { direction: "s90", mirror: { partner: null, partner_view: null, available: false, partner_pictures: 0, last: null, warning: null } }))).toBeNull();
  });

  it("an attempt says where it came from", () => {
    expect(attemptOrigin(attempt(1, { source: "mirror", mirror: { direction: "s0", frame: "f03", ref: "f03:02", sha256: "x", how: "flip horizontally" } })))
      .toBe("mirror of s0 f03:02");
    expect(attemptOrigin(attempt(1, { draw_from: "keys", skeleton: { frame: 14, default_frame: 12, frames: 37, fps: 24, time: 0.5, clip: "c", rig: "r" } })))
      .toBe("from keys + skeleton 14");
  });
});
