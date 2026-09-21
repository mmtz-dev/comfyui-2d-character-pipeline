// The frame contract's alpha rule, applied before a frame leaves the editor:
// alpha is 0 or 255, nothing between, and a transparent pixel is (0,0,0,0)
// (anim/imaging.py `binarise_alpha`, same threshold). Piskel rounds alpha on
// import already; this is the same rule on the way out, so what the API
// receives is what the API would have made of it.

export const ALPHA_THRESHOLD = 128;

export interface AlphaReport {
  /** Pixels whose alpha was neither 0 nor 255 before rounding. */
  rounded: number;
  /** Pixels that ended transparent. */
  transparent: number;
}

/** In place, on RGBA bytes (an ImageData's `data`). */
export function binariseAlpha(rgba: Uint8ClampedArray | Uint8Array, threshold = ALPHA_THRESHOLD): AlphaReport {
  let rounded = 0;
  let transparent = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    const alpha = rgba[i + 3]!;
    if (alpha !== 0 && alpha !== 255) rounded += 1;
    if (alpha >= threshold) {
      rgba[i + 3] = 255;
    } else {
      rgba[i] = 0;
      rgba[i + 1] = 0;
      rgba[i + 2] = 0;
      rgba[i + 3] = 0;
      transparent += 1;
    }
  }
  return { rounded, transparent };
}
