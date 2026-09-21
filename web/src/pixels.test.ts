import { describe, expect, it } from "vitest";
import { binariseAlpha } from "./pixels";

describe("binariseAlpha", () => {
  it("rounds alpha to 0/255 and zeroes transparent pixels, like imaging.binarise_alpha", () => {
    const rgba = new Uint8ClampedArray([
      10, 20, 30, 255, // opaque: kept
      10, 20, 30, 200, // soft, above: opaque
      10, 20, 30, 127, // soft, below: cleared to (0,0,0,0)
      10, 20, 30, 0, // transparent with colour under it: cleared
    ]);
    const report = binariseAlpha(rgba);
    expect(Array.from(rgba)).toEqual([10, 20, 30, 255, 10, 20, 30, 255, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(report).toEqual({ rounded: 2, transparent: 2 });
  });
  it("is a no-op on a frame that already keeps the contract", () => {
    const rgba = new Uint8ClampedArray([1, 2, 3, 255, 0, 0, 0, 0]);
    expect(binariseAlpha(rgba)).toEqual({ rounded: 0, transparent: 1 });
    expect(Array.from(rgba)).toEqual([1, 2, 3, 255, 0, 0, 0, 0]);
  });
});
