import { describe, expect, it } from "vitest";
import type { FrameEntry, PicksState } from "./api";
import { chooseFrameSet, editableEntries, frameNumber, phaseLabels } from "./frames";

const entry = (index: number, file: string, set = "post"): FrameEntry => ({
  index,
  file,
  url: `/jobs/knight_strike/frames/${set}/${index}.png`,
});

// What the API lists for a real post/ after `post --parts`: lexicographic,
// with the palette and the diagnostics in the same directory.
const POST_LISTING = [
  "knight_strike_0.png", "knight_strike_0_hd.png", "knight_strike_0_labels.png",
  "knight_strike_1.png", "knight_strike_10.png", "knight_strike_11.png",
  "knight_strike_2.png", "knight_strike_3.png", "palette.png",
].map((file, index) => entry(index, file));

describe("frameNumber", () => {
  it("reads the frame number publish uses", () => {
    expect(frameNumber("knight_strike_7.png")).toBe(7);
    expect(frameNumber("knight_strike_11.png")).toBe(11);
  });
  it("is null for the palette and the part diagnostics", () => {
    expect(frameNumber("palette.png")).toBeNull();
    expect(frameNumber("knight_strike_7_hd.png")).toBeNull();
    expect(frameNumber("knight_strike_7_labels.png")).toBeNull();
  });
});

describe("editableEntries", () => {
  it("keeps only the post frames, in numeric order, with their API index", () => {
    const entries = editableEntries("post", POST_LISTING);
    expect(entries.map((e) => e.file)).toEqual([
      "knight_strike_0.png", "knight_strike_1.png", "knight_strike_2.png",
      "knight_strike_3.png", "knight_strike_10.png", "knight_strike_11.png",
    ]);
    expect(entries.map((e) => e.index)).toEqual([0, 3, 6, 7, 4, 5]);
    expect(entries[4]!.url).toBe("/jobs/knight_strike/frames/post/4.png");
  });
  it("takes other sets as listed", () => {
    const clip = [entry(0, "0001.png", "clip"), entry(1, "0002.png", "clip")];
    expect(editableEntries("clip", clip)).toEqual(clip);
  });
});

describe("chooseFrameSet", () => {
  it("prefers post, then clip, then frames; never render or strip", () => {
    const frames = [entry(0, "0001.png", "frames")];
    const clip = [entry(0, "0001.png", "clip")];
    const render = [entry(0, "pose_pass.png", "render")];
    expect(chooseFrameSet({ frames, clip, render, post: POST_LISTING })?.set).toBe("post");
    expect(chooseFrameSet({ frames, clip, render })?.set).toBe("clip");
    expect(chooseFrameSet({ frames, render })?.set).toBe("frames");
    expect(chooseFrameSet({ render, strip: render })).toBeNull();
    expect(chooseFrameSet({})).toBeNull();
  });
  it("skips a post/ that holds only the palette", () => {
    const frames = [entry(0, "0001.png", "frames")];
    expect(chooseFrameSet({ post: [entry(0, "palette.png")], frames })?.set).toBe("frames");
  });
});

function picks(picked: number[], tags: string[], total = 6): PicksState {
  const byPick = new Map(picked.map((p, k) => [p, tags[k] ?? null]));
  return {
    job: "knight_strike", action: "strike", slug: "knight", fps: 12, n: picked.length,
    picks: picked, auto: picked, edited: false, rerender_requested: false,
    phases: { tags },
    frames: Array.from({ length: total }, (_, i) => ({
      index: i, file: `${String(i + 1).padStart(4, "0")}.png`, picked: byPick.has(i),
      order: picked.indexOf(i) === -1 ? null : picked.indexOf(i), phase: byPick.get(i) ?? null, skeleton: true,
    })),
  };
}

describe("phaseLabels", () => {
  const state = picks([0, 2, 5], ["start", "action", "end"]);
  it("labels source-indexed sets per frame", () => {
    expect(phaseLabels("frames", 6, state)).toEqual(["start", null, "action", null, null, "end"]);
    expect(phaseLabels("clip", 6, state)).toEqual(["start", null, "action", null, null, "end"]);
  });
  it("labels post frames through the picks, in ascending pick order", () => {
    expect(phaseLabels("post", 3, state)).toEqual(["start", "action", "end"]);
    expect(phaseLabels("post", 3, picks([5, 0, 2], ["end", "start", "action"]))).toEqual(["start", "action", "end"]);
  });
  it("gives no tags rather than wrong ones", () => {
    expect(phaseLabels("post", 4, state)).toEqual([null, null, null, null]);
    expect(phaseLabels("post", 3, null)).toEqual([null, null, null]);
    expect(phaseLabels("render", 1, state)).toEqual([null]);
    expect(phaseLabels("frames", 2, null)).toEqual([null, null]);
  });
});
