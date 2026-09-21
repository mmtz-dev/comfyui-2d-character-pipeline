import { describe, expect, it } from "vitest";
import { normaliseColour, projectPalette } from "./palette";

describe("normaliseColour", () => {
  it("accepts hex in the usual spellings and rgb triples", () => {
    expect(normaliseColour("#F0C070")).toBe("#f0c070");
    expect(normaliseColour("f0c070")).toBe("#f0c070");
    expect(normaliseColour("#abc")).toBe("#aabbcc");
    expect(normaliseColour([240, 192, 112])).toBe("#f0c070");
  });
  it("rejects what is not a colour", () => {
    expect(normaliseColour("gold")).toBeNull();
    expect(normaliseColour("#12345")).toBeNull();
    expect(normaliseColour([300, 0, 0])).toBeNull();
    expect(normaliseColour(12)).toBeNull();
  });
});

describe("projectPalette", () => {
  it("says 'no project palette' for today's default settings", () => {
    const answer = projectPalette({ schema: "anim.project/1", palette: null, frame_size: [64, 64] });
    expect(answer.palette).toBeNull();
    expect("reason" in answer && answer.reason).toBe("no project palette");
  });
  it("is 'no project palette' for a path string (the pre-#282 shape)", () => {
    const answer = projectPalette({ palette: "/assets_src/palette_candidates/grim32/grim32.gpl" });
    expect(answer.palette).toBeNull();
    expect("reason" in answer && answer.reason).toMatch(/^no project palette/);
    expect("reason" in answer && answer.reason).toContain("grim32.gpl");
  });
  it("reads the palette feature's object: {path, name, colours, source}", () => {
    const answer = projectPalette({
      schema: "anim.project/1",
      palette: { path: "/assets_src/grim32.gpl", name: "grim32", colours: ["#1a1c2c", "#5D275D", "#b13e53"], source: "file" },
      frame_size: [64, 64],
    });
    expect(answer.palette).toEqual({ name: "grim32", colours: ["#1a1c2c", "#5d275d", "#b13e53"] });
  });
  it("reads colours from an object palette in the US spelling, de-duplicated", () => {
    const answer = projectPalette({ palette: { name: "grim32", colors: ["#111", "#222222", "#222222", "nope"] } });
    expect(answer.palette).toEqual({ name: "grim32", colours: ["#111111", "#222222"] });
  });
  it("names an object palette 'project' when it has no name", () => {
    expect(projectPalette({ palette: { colours: ["#000000"] } }).palette?.name).toBe("project");
  });
  it("reads colours listed beside the path", () => {
    const answer = projectPalette({ palette: "/x/grim32.gpl", palette_colours: [[1, 2, 3], "#ffffff"] });
    expect(answer.palette).toEqual({ name: "grim32", colours: ["#010203", "#ffffff"] });
  });
  it("copes with no settings at all", () => {
    expect(projectPalette(null).palette).toBeNull();
    expect(projectPalette({ palette: { colors: [] } }).palette).toBeNull();
  });
});
