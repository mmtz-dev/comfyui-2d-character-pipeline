# Can the anim-pipeline trace a Wan sprite frame to vector, then pixelate it at a chosen size, and would that beat what we have?

## Short answer
Yes, it can be built (vtracer or storytold's `vectorcraft-trace` to trace, any hard-edge rasteriser to sample), but it will not beat `--downscale grid`. The video's method is just nearest-neighbour resampling, and a trace adds no information to a frame that is already pixel art; on the wanderer it gave the same picture with slightly more frame-to-frame change. Re-render the figure smaller (about 300 HD px tall) so the art grid lands at 1:1 or an integer ratio with the canvas; keep `--fit band` only as a fallback.

## Findings

### 1. What the video recommends
- Scaling a high-res image down normally blurs it because the software averages colours between pixels; the fix is the Nearest Neighbor resampling option, which copies exact pixel values. [transcript 00:28-00:58](/home/workstation/Repos/apps/tube-digest/Transcriptions/Pixel_Art_Animation/How_to_turn_Vectors_and_Photos_into_Pixel_Art_Quick_Tips_2026-10-11.txt)
- Vectors / flat art: Image Size to 10%, resampling = Nearest Neighbor, then touch up by hand. The speaker notes his "vector" is really a high-res PNG exported from vector shapes, with few flat colours (best case). [transcript 00:17-01:11](/home/workstation/Repos/apps/tube-digest/Transcriptions/Pixel_Art_Animation/How_to_turn_Vectors_and_Photos_into_Pixel_Art_Quick_Tips_2026-10-11.txt)
- Photos: reduce colours first (Desaturate optionally, then Posterize to 2-5 levels), then the same 10% Nearest Neighbor scale, then re-add hue to shadows, midtones and highlights with Curves / Color Balance and fix by hand in a pixel editor. [transcript 01:15-01:51](/home/workstation/Repos/apps/tube-digest/Transcriptions/Pixel_Art_Animation/How_to_turn_Vectors_and_Photos_into_Pixel_Art_Quick_Tips_2026-10-11.txt)
- The video never traces anything to vector. "Vector" means a flat-colour image exported at high resolution. The only real step is nearest-neighbour downscale (plus posterize for photos). [summary](/home/workstation/Repos/apps/tube-digest/Summaries/Pixel_Art_Animation/How_to_turn_Vectors_and_Photos_into_Pixel_Art_Quick_Tips_2026-10-11.md)

### 2. Who can do each step
All storytold code is Rust (MIT OR Apache-2.0), so it needs a Rust toolchain; `cargo`/`rustc` are not installed on this host (checked with `which`), so using it means a Rust build container. [reuse sweep](/home/workstation/Repos/storytold/2026-10-10-storytold-reuse-sweep.md) says the same: Craft apps are Rust, realistic reuse is calling their `*-cli` or porting an algorithm.

| Step | storytold | Already have / alternatives |
| --- | --- | --- |
| Trace raster to SVG | `vectorcraft/crates/trace/src/lib.rs`: `trace(&Raster, &TraceParams)`, `trace_within`, `preset`; quantise (`quantize.rs::quantize`), denoise (`denoise`), crack-edge contours (`contour.rs::trace_mask`), Douglas-Peucker then Bezier fit. Driven by `vectorcraft-cli run --in FILE --cmd imageTrace.make --export out.svg` (`vectorcraft/apps/vectorcraft-cli/src/main.rs:5,79-84`; command and params named in `vectorcraft/crates/mcp/src/prompts.rs:271-310`) | `vtracer` (pip, tried here, see section 3); `potrace` and `autotrace` are not installed; ImageMagick `convert`/`magick` is installed but does not trace |
| Rasterise a vector with hard edges | `vectorcraft/crates/render/src/lib.rs:124-140`: `AntiAlias::None` ("a pixel is painted when a path covers at least half of it"); exposed to commands via `antiAlias` (`vectorcraft/crates/engine/src/cmd/menucmds.rs:381-394`); export at a scale via `--scale N` (CLI usage, `main.rs:79`) | `cairosvg` rendered fine here (anti-aliased, then sampled point-wise, which is the same result); ImageMagick can also render SVG |
| Nearest-neighbour resize | `photocraft/crates/algo/src/resample.rs`: `Resample::Nearest` (line 12), `resize_surface` (line 135). `vectorcraft/crates/trace/src/mosaic.rs::mosaic` is an *average* per tile, not nearest, so it blurs like `box` | PIL `Image.NEAREST`; `gridkit.sample` already takes the dominant colour per art pixel (`anim/gridkit.py:242`), which is better than nearest because it votes inside the cell's central half |
| Posterize / colour reduce | `photocraft/crates/algo/src/quantize.rs`: `build_palette` (line 263; median cut + k-means, perceptual, adaptive) and `quantize` (line 353; none / Floyd-Steinberg / ordered / noise dither). Also `vectorcraft-trace` quantise (limited palette) | `anim/palettekit.py::reduce_colours` (line 443, Oklab with Lr lightness) and the unit-palette snap already in `post` |
| Find the art grid | none | `anim/gridkit.py::measure` (line 187) |

- Gaps: no Python binding to any storytold code (Rust only, so a CLI call or a container); no storytold feature for "measure the art grid"; `vectorcraft-cli`'s trace-then-export path takes a placed image through the document engine, which is heavier than a single trace call (I did not build or run it). [vectorcraft CLI](/home/workstation/Repos/storytold/vectorcraft/apps/vectorcraft-cli/src/main.rs)
- Nothing in photocraft-corpus or the other repos in the sweep adds a pixel-art-specific step (I looked at the photocraft `algo` crate list: `pixelate.rs` is Color Halftone / Crystallize / Facet etc., not pixel-grid snapping). [photocraft algo](/home/workstation/Repos/storytold/photocraft/crates/algo/src/pixelate.rs)

### 3. Will it work on our frames? (small local test, CPU only, 2026-10-11)
Setup: first 8 frames of `work/wanderer_walk_cmp_grid/clip/frames` (512x768 RGB, whole frame, white background, not matted), `gridkit.measure` over all 22 frames gave pitch 5.65 again. Target = the `--fit band` size (true 84 art px squeezed to 53, ratio 1.585). Scratch in `/tmp/vt`, script `/tmp/vt/exp.py`.
- A (current): `gridkit.sample` on the band-fitted grid (what `--fit band` does).
- B: `vtracer` on the raw frame, rasterise the SVG, point-sample at the band cell centres (the video's vector method).
- C: `gridkit.sample` at true size, nearest-neighbour up 8x, trace that, rasterise, point-sample at the band ratio.
- D: C's input without the trace, nearest-neighbour straight to the band size.
Observations:
- All four give nearly the same figure at about 53 art px tall. B (raw frame traced) loses small details and tints the background to a near-white (254) and rounds some edges; C and D are close to A. No variant recovers the 31 rows that `--fit band` drops, since squeezing 84 art pixels into 53 discards information whichever way it is done. (Visual comparison of montages `/tmp/vt/out/{A,B,C}.png`; my own reading, one sprite, 4 frames viewed.)
- Tracing the raw frame: vtracer's default settings quantise colour (about 92 colours on frame 0 versus 404 for A's unreduced whole frame), which removes compression noise, but it also merges the near colours that the wanderer's shading depends on. `gridkit.sample` already removes the noise by voting colour bins of width 8 (`VOTE_BIN`, `anim/gridkit.py:67`) inside each cell's central half, with no merge across cells.
- Tracing the grid-sampled image (C) is the better order because the staircase is already clean, but then the trace only re-encodes what `sample` produced; spline mode rounds corners, and a staircase is exactly what pixel art should keep. Polygon mode would keep it but equals nearest-neighbour. So the measured grid helps, but it makes the vector step redundant rather than useful.
- Caveat: the pip `vtracer` 0.6.15 wheel on this host's Python 3.14 segfaulted on every call that passed an option (also with `convert_pixels_to_svg`); only `convert_image_to_svg_py(in, out)` with defaults worked. Settings like `mode='polygon'` or `filter_speckle=0` were therefore not tested. Not tested: `vectorcraft-trace` (no Rust toolchain), matted frames, the full 22-frame clip at the 64x64 canvas.

### 4. Animation and flicker
- Mean absolute difference between consecutive output frames (first 8 frames, whole frame, includes real walk motion): A 4.58, B 4.85, C 4.80, D 4.85 (units: 0-255 per channel). Tracing therefore did not reduce flicker and was slightly worse than the current route. Differences are small and include true motion, so treat this as "no benefit shown" rather than a precise ranking.
- Reasons it should not help: the tracer decides region boundaries and merged colours per frame independently, so a one-art-pixel shift in a boundary or a colour merge flips cells between frames; and the per-frame grid pitch spread the project already saw (5.65 to 6.54 across frames, `report.json.grid.per_frame`) comes from the model's frames, not from the sampling, so the vector does not remove it. Fixing frame-to-frame jitter is the job of the planned stabiliser on the sampled pixels. [grid report](/home/workstation/Repos/game-dev/framework-ladder/tools/anim-pipeline/work/wanderer_walk_cmp_grid/post/report.json)

### 5. Comparison and recommendation
- The project's own design already says why squeezing blurs and why true size was chosen: one measured grid per job, one colour per art pixel, true size by default, and a refusal that says to re-render smaller. [gridkit.py docstring](/home/workstation/Repos/game-dev/framework-ladder/tools/anim-pipeline/anim/gridkit.py), [README, "The pixel grid downscale"](/home/workstation/Repos/game-dev/framework-ladder/tools/anim-pipeline/README.md)
- `--fit band` keeps a grid but makes each output pixel stand for 1.58 art pixels, so detail is lost. [README](/home/workstation/Repos/game-dev/framework-ladder/tools/anim-pipeline/README.md)
- Vector route: scaling the vector does not change that, since the resolution of the art (84 art px tall) is the limit, not the format. At a non-integer ratio the final nearest-neighbour step picks cells unevenly, the same as `--fit band`. Cost: a Rust toolchain or a vtracer wheel, extra per-frame tracing time, and flicker risk. Benefit: none measured.
- Re-render smaller: the figure then comes out of Wan at the target pitch (84 art px becomes about 53-60 if framed ~35% smaller, or exactly 1:1 with the canvas), so `sample` is lossless; only the reframing is needed, which is the project's current plan (about 300 HD px tall).
- Recommendation: re-render smaller. If a different size is wanted later, resize with an integer ratio of art pixels (2:1 or 3:2 after choosing a 2x2 vote) rather than tracing.
- Small test if the owner still wants the vector idea checked: take the matted 22-frame clip, run (a) `grid --fit band`, (b) vtracer polygon mode + hard-edge sample (needs a Python version where vtracer options do not crash, or `vectorcraft-cli` with `AntiAlias::None`), (c) a re-render at 300 HD px with `grid --fit true`. Score each on (1) the number of art pixels that differ from the true-size sample resampled the same way, (2) mean frame-to-frame difference after the stabiliser, (3) eye, belt buckle and coat seams kept, judged on an 8x crop as in the README.

## Sources
- /home/workstation/Repos/apps/tube-digest/Transcriptions/Pixel_Art_Animation/How_to_turn_Vectors_and_Photos_into_Pixel_Art_Quick_Tips_2026-10-11.txt
- /home/workstation/Repos/apps/tube-digest/Summaries/Pixel_Art_Animation/How_to_turn_Vectors_and_Photos_into_Pixel_Art_Quick_Tips_2026-10-11.md
- /home/workstation/Repos/storytold/2026-10-10-storytold-reuse-sweep.md
- /home/workstation/Repos/storytold/vectorcraft/crates/trace/src/lib.rs
- /home/workstation/Repos/storytold/vectorcraft/crates/trace/src/mosaic.rs
- /home/workstation/Repos/storytold/vectorcraft/crates/trace/src/quantize.rs
- /home/workstation/Repos/storytold/vectorcraft/crates/render/src/lib.rs
- /home/workstation/Repos/storytold/vectorcraft/crates/engine/src/cmd/menucmds.rs
- /home/workstation/Repos/storytold/vectorcraft/crates/mcp/src/prompts.rs
- /home/workstation/Repos/storytold/vectorcraft/apps/vectorcraft-cli/src/main.rs
- /home/workstation/Repos/storytold/photocraft/crates/algo/src/resample.rs
- /home/workstation/Repos/storytold/photocraft/crates/algo/src/quantize.rs
- /home/workstation/Repos/storytold/photocraft/crates/algo/src/pixelate.rs
- /home/workstation/Repos/game-dev/framework-ladder/tools/anim-pipeline/anim/gridkit.py
- /home/workstation/Repos/game-dev/framework-ladder/tools/anim-pipeline/anim/palettekit.py
- /home/workstation/Repos/game-dev/framework-ladder/tools/anim-pipeline/README.md
- /home/workstation/Repos/game-dev/framework-ladder/tools/anim-pipeline/work/wanderer_walk_cmp_grid/post/report.json
- /tmp/vt/exp.py and /tmp/vt/out/{A,B,C}.png (local experiment, scratch, not kept in the repo)

## Blocked
Nothing was blocked. Not read or not tested: `vectorcraft-cli` and `vectorcraft-trace` were read but not built or run (no Rust toolchain on host); `DEVLOG.md` and `anim/stages/post.py` were skimmed only through README/gridkit docstrings; vtracer's option parameters crashed on Python 3.14, so only default settings were tried; vtracer behaviour is from the local run, not from its docs.
