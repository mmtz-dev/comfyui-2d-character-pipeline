import { describe, expect, it } from "vitest";
import type { ConceptState, ConceptView } from "./api";
import { layoutFor, readTab, readUnit, stepsDone } from "./tabs";

function concept(pick: number | null, views: Record<string, number | null>): Pick<ConceptState, "base" | "views"> {
  return {
    base: { description: null, prompt: null, sketch: null, candidates: [], pick, hidden_count: 0,
      pick_stale: false, accepted: pick !== null, front_url: null },
    views: Object.fromEntries(Object.entries(views).map(([view, p]) => [view, { view, pick: p } as unknown as ConceptView])),
  };
}

describe("the three tabs", () => {
  it("remembers a tab, and falls back to Concept for the retired ones", () => {
    expect(readTab("animate")).toBe("animate");
    expect(readTab("board")).toBe("concept");
    expect(readTab("pipeline")).toBe("concept");
    expect(readTab(null)).toBe("concept");
  });

  it("is wide on Concept and split beside Piskel everywhere else", () => {
    expect(layoutFor("concept", false)).toBe("wide");
    expect(layoutFor("concept", true)).toBe("split");
    expect(layoutFor("views", false)).toBe("split");
    expect(layoutFor("animate", false)).toBe("split");
  });

  it("ticks Concept once the base is picked, and Views once back, left and top are", () => {
    expect(stepsDone(null)).toEqual({ concept: false, views: false, animate: false });
    expect(stepsDone(concept(null, {}))).toEqual({ concept: false, views: false, animate: false });
    expect(stepsDone(concept(2, { back: 0, left: 1 }))).toEqual({ concept: true, views: false, animate: false });
    expect(stepsDone(concept(2, { back: 0, left: 1, top: null }))).toEqual({ concept: true, views: false, animate: false });
    expect(stepsDone(concept(2, { back: 0, left: 1, top: 3 }))).toEqual({ concept: true, views: true, animate: false });
  });

  it("remembers only a unit slug", () => {
    expect(readUnit("hooded_archer")).toBe("hooded_archer");
    expect(readUnit("Not a slug")).toBe("");
    expect(readUnit(null)).toBe("");
  });
});
