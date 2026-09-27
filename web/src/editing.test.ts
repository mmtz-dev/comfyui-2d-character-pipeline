import { describe, expect, it } from "vitest";
import { clampEditWidth, discardQuestion, mustConfirmLeave, readEditWidth } from "./editing";

describe("clampEditWidth", () => {
  it("keeps a width inside 240 px .. 80% of the window", () => {
    expect(clampEditWidth(500, 1600)).toBe(500);
    expect(clampEditWidth(100, 1600)).toBe(240);
    expect(clampEditWidth(1500, 1600)).toBe(1280);
  });
  it("rounds to whole pixels", () => {
    expect(clampEditWidth(412.6, 1600)).toBe(413);
  });
  it("gives the minimum when the window is too narrow for both bounds", () => {
    expect(clampEditWidth(400, 250)).toBe(240);
  });
  it("falls back to the default for a width that is not a number", () => {
    expect(clampEditWidth(Number.NaN, 1600)).toBe(440);
  });
});

describe("readEditWidth", () => {
  it("is 440 when nothing is remembered", () => {
    expect(readEditWidth(null)).toBe(440);
    expect(readEditWidth("")).toBe(440);
  });
  it("reads a remembered width", () => {
    expect(readEditWidth("512")).toBe(512);
  });
  it("ignores garbage and widths under the minimum", () => {
    expect(readEditWidth("wide")).toBe(440);
    expect(readEditWidth("12")).toBe(440);
  });
});

describe("the unsaved guard", () => {
  it("asks only when something is open and dirty", () => {
    expect(mustConfirmLeave({ label: "x" }, true)).toBe(true);
    expect(mustConfirmLeave({ label: "x" }, false)).toBe(false);
    expect(mustConfirmLeave(null, true)).toBe(false);
  });
  it("names what would be lost", () => {
    expect(discardQuestion("knight base:02")).toBe("Discard unsaved changes to knight base:02?");
  });
});
