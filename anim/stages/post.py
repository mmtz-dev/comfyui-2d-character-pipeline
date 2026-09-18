"""Stage: `anim post` - generated strip -> 64x64 frames (deterministic)

The palette-locking chain. Entirely outside ComfyUI, entirely deterministic:
this is the half of the pipeline that has to produce the same bytes twice.

    anim post knight_strike --palette /assets_src/palette_candidates/grim32/grim32.gpl

    cut alpha (binary)  ->  slice cells  ->  grid repair  ->  pixel downscale
      ->  fit  ->  palette snap  ->  re-apply mask  ->  recanvas 64x64  ->  verify

STEP BY STEP, AND WHY
---------------------
**Cut alpha, once, on the whole strip.** `rembg` with `isnet-anime` and
`post_process_mask=True`, then hard-thresholded to {0, 255}. Binary, not soft:
a 64px sprite with a feathered edge has a halo the palette cannot express, and
ComfyUI's BiRefNet node exposes no threshold - which is exactly why background
removal is on this side of the wire. (`rembg`'s CLI extra is not installed in
this image, so the Python API is used; `post_process_mask=True` is the `-ppm`
in the plan, and the alpha-matting knobs are moot because the mask is
thresholded anyway.) `--matte chroma` swaps in a deterministic flat-background
key with no model download - that is what the tests and the fixture strip use,
and it is the right choice for any synthetic input: measured 2026-09-13,
`isnet-anime` returns an EMPTY mask on a strip of flat drawn shapes (it has no
anime character to find) while returning a clean 78%-opaque binary matte on a
real photograph. On an actual generation, `rembg` is the better cut.

**Slice by layout.json.** The cell rects `strip` recorded, not a fresh guess.

**Grid repair BEFORE the downscale, not after.** The plan's order was pixeloe
then pixel-art-fixer; measured on 2026-09-13, that is backwards.
`pixelfixer process` looks for the block grid of an up-scaled pixel image and
rebuilds it at one pixel per block - so on a 51x54 image that pixeloe had
already reduced to native scale it "found" a 5.77px grid and collapsed it to
9x10. Run first, on the full-resolution cell, it does the job it was built for:
a 304px cell of generated art really does have an 8-ish pixel block grid. It is
guarded either way - an output smaller than the target height is rejected and
the report says so, so the chain cannot silently lose the sprite.

**pixeloe for the downscale.** Contrast-aware and grid-aware, which is the
difference between a readable 53px knight and mush. The target height is the
unit's band on the art bible's scale ladder, so a hound comes out 37px and the
boss 63px instead of every unit being the same size.

**Flatten, then snap.** Pillow's `quantize()` falls back to octree on RGBA and
will not honour a fixed palette, so the snap is ImageMagick's
`-dither None -remap palette.png` on flat RGB. Flattening onto a chroma key the
palette cannot represent (not onto white, not onto a palette colour) means no
background pixel can fuse into the silhouette; the binary mask goes back on
afterwards and the key never survives into the output.

**Recanvas 64x64 bottom-center**, the paste rule from
`assets_src/extract_barrow_assets.py`, then **verify** - dimensions, binary
alpha, every opaque colour a palette member. A frame that fails is not written.

TWO SOURCES, ONE CHAIN (the clip render feature)
-------------------------------------------------
`--input <strip.png>` (or the default `render/<final>`) is the strip route: the
generated N x 1 strip is matted once and sliced by `strip/layout.json`.
`--input clip` is the clip route: the picked frames are read one by one from
`clip/frames/<frame + 1>.png` (what `anim clip` generated, 1:1 with the rig
frames) and each is matted on its own. From the cell onward the chain is the
same function, so the two routes differ only in where the pixels came from;
`report.json` says which (`route`), and `publish` picks the matching
provenance file from it.

**The part palette lock (`--parts`, the part palette lock feature).** Off by
default, and with it off the chain above produces exactly the bytes it did
before the feature existed. On, each HD cell is masked into the unit's parts
(`config.UNIT_PARTS`: sword, shield, helmet, arms, torso, legs for the knight)
by SAM 2.1 prompted from the rig's joints and prop outlines, the masks ride
the same trim/downscale as the pixels, and after the unit-level snap every
pixel of a part is re-snapped to *that part's* palette. The palettes come
from the unit's accepted base sprite: `post --parts` on a `base` job is the
EXTRACT mode (no action -> nothing to lock to) and writes `post/parts.json`,
which `publish` carries into the sprite's sidecar as `generation.parts`; every
action job then LOCKS to it (`--parts-from` overrides the lookup). Per frame,
per part, `report.json` records pixels, colours and `drift` - how many pixels
were NOT in the part's palette before the lock - which is the number the
second gate reads. See anim/partkit.py for the why of each choice.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from anim import config, imaging, partkit, posekit
from anim.job import Job
from anim.util import have, read_json, tool_version, write_json_in_place
from anim.util import run as run_tool

NAME = "post"
HELP = "generated strip -> 64x64 frames (deterministic)"

POST_SCHEMA = "anim.post/1"
REMBG_MODEL = "isnet-anime"
# A border pixel further than this from the sampled background colour is subject.
CHROMA_TOLERANCE = 40


# ---------------------------------------------------------------------------
# Matting
# ---------------------------------------------------------------------------
def matte_rembg(img: Image.Image) -> Image.Image:
    """Background removal with isnet-anime. Downloads its ONNX on first use."""
    from rembg import new_session, remove

    cut = remove(img.convert("RGBA"), session=new_session(REMBG_MODEL),
                 post_process_mask=True)
    return imaging.binarise_alpha(cut)


def matte_chroma(img: Image.Image, tolerance: int = CHROMA_TOLERANCE) -> Image.Image:
    """Key out a flat background sampled from the image border.

    Deterministic, model-free, and correct for exactly the input the workflows
    ask for ("plain flat white background, no ground shadow, no floor"). The
    background colour is the median of the one-pixel border ring rather than a
    hardcoded white, so a strip generated on light grey still keys.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    distance = np.abs(rgb - background).sum(axis=2)
    alpha = np.where(distance > tolerance, 255, 0).astype(np.uint8)
    out = np.dstack([np.asarray(img.convert("RGB")), alpha])
    out[alpha == 0] = (0, 0, 0, 0)
    return Image.fromarray(out, mode="RGBA")


def matte_keep(img: Image.Image) -> Image.Image:
    """Trust the alpha already in the image (or treat it as fully opaque)."""
    return imaging.binarise_alpha(img)


MATTES: dict[str, Callable[[Image.Image], Image.Image]] = {
    "rembg": matte_rembg,
    "chroma": matte_chroma,
    "none": matte_keep,
}


# ---------------------------------------------------------------------------
# The per-cell chain
# ---------------------------------------------------------------------------
GRID_ASPECT_TOLERANCE = 0.15


def grid_repair(img: Image.Image, target_height: int) -> tuple[Image.Image, str]:
    """pixel-art-fixer's grid detection + reconstruction. Guarded; never fatal.

    Returns (image, note). The note goes into report.json so a rejected repair
    is visible rather than invisible.
    """
    if not have("pixelfixer"):
        return img, "skipped (pixelfixer not on PATH)"
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.png"
        dst = Path(tmp) / "out.png"
        img.convert("RGB").save(src)
        proc = run_tool(["pixelfixer", "process", str(src), str(dst), "full"],
                   check=False, timeout=300)
        if proc.returncode != 0 or not dst.exists():
            return img, f"skipped (pixelfixer exit {proc.returncode})"
        with Image.open(dst) as repaired:
            out = repaired.convert("RGB").copy()
    if out.height < target_height or out.height > img.height:
        return img, (f"rejected ({img.width}x{img.height} -> "
                     f"{out.width}x{out.height}; below the {target_height}px "
                     f"target)")
    # A grid found with different pitches on the two axes is a grid that was
    # not there. knight_base's first render (2026-09-15): 413x695 -> 36x81, an
    # 11.5px x-pitch against an 8.6px y-pitch, and the knight came out 23px
    # wide. The height guard above cannot see that; the aspect ratio can.
    aspect_in = img.width / img.height
    aspect_out = out.width / out.height
    if abs(aspect_out - aspect_in) > GRID_ASPECT_TOLERANCE * aspect_in:
        return img, (f"rejected ({img.width}x{img.height} -> "
                     f"{out.width}x{out.height}; aspect {aspect_out:.2f} vs "
                     f"{aspect_in:.2f} - unequal x/y pitch)")
    return out, f"{img.width}x{img.height} -> {out.width}x{out.height}"


# pixeloe's outline-expansion pass strides by `(patch_size // 4) * 2`, so any
# patch size below 4 gives it a stride of zero and torch raises deep inside
# im2col. Measured 2026-09-13; nothing in pixeloe guards it.
MIN_PATCH = 4


def pixel_downscale(img: Image.Image,
                    target_height: int) -> tuple[Image.Image, str]:
    """pixeloe contrast-aware downscale to roughly `target_height` pixels tall.

    `patch_size` is the block the contrast pass reasons over, so it wants to be
    the real downscale factor - the sprite's height over the target height - and
    never below MIN_PATCH. Falls back to a plain nearest resize if pixeloe still
    refuses, with the reason recorded in report.json rather than swallowed.
    """
    from pixeloe.legacy.pixelize import pixelize

    bgr = np.asarray(img.convert("RGB"))[:, :, ::-1]
    patch = max(MIN_PATCH, round(img.height / max(1, target_height)))
    try:
        out = pixelize(bgr, mode="contrast", target_size=target_height,
                       patch_size=patch, no_upscale=True)
    except Exception as exc:  # pragma: no cover - the guard above should hold
        return (imaging.scale_to_height(img.convert("RGB"), target_height),
                f"nearest fallback (pixeloe: {type(exc).__name__}: {exc})")
    return (Image.fromarray(np.asarray(out)[:, :, ::-1], mode="RGB"),
            f"pixeloe patch {patch}")


def box_downscale(img: Image.Image, target_height: int) -> tuple[Image.Image, str]:
    """Plain area-average to exactly `target_height` tall, aspect kept.

    The default since 2026-09-15. On the first real render (knight_base,
    512x768 Z-Image output with an ~8px block grid) pixeloe's contrast pass
    produced a noisier sprite than a box filter, and its `target_size` does
    not mean the output height (53 asked, 68 given), so the chain then
    nearest-resampled 68 -> 53 and lost rows unevenly. A box filter followed
    by the palette snap keeps the render's outline and shading reads intact,
    lands on the target height in one step, and is exactly reproducible.
    Comparison sheet: game02-anchor-gate.html on the visuals server.
    """
    width = max(1, round(img.width * target_height / max(1, img.height)))
    return img.convert("RGB").resize((width, target_height), Image.BOX), "box"


DOWNSCALES: dict[str, Callable[[Image.Image, int], tuple[Image.Image, str]]] = {
    "box": box_downscale,
    "pixeloe": pixel_downscale,
}
DEFAULT_DOWNSCALE = "box"


def palette_snap(img: Image.Image, palette_png: Path) -> Image.Image:
    """`magick -dither None -remap`. RGB in, RGB out, every pixel a member."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.png"
        dst = Path(tmp) / "out.png"
        img.convert("RGB").save(src)
        run_tool(["magick", str(src), "-dither", "None", "-remap", str(palette_png),
             str(dst)], timeout=300)
        with Image.open(dst) as snapped:
            return snapped.convert("RGB").copy()


def process_cell(cell: Image.Image, *, target_height: int, palette_png: Path,
                 chroma: tuple[int, int, int], use_pixelfixer: bool = True,
                 canvas: tuple[int, int] = None,
                 downscale: str = DEFAULT_DOWNSCALE,
                 part_masks: partkit.PartMasks | None = None,
                 part_palettes: dict[str, Any] | None = None,
                 parts_out: dict[str, Any] | None = None
                 ) -> tuple[Image.Image, list[str]]:
    """One generated cell -> one finished 64x64 frame. Pure; no disk state.

    `part_masks` (the part palette lock feature) is the cell's HD label map;
    it is trimmed and downscaled alongside the pixels and, after the palette
    snap, either LOCKS each part to `part_palettes` or - when no palettes are
    given - EXTRACTS the colours each part shows. What happened lands in
    `parts_out` (an out-param so the return shape every caller relies on stays
    `(frame, steps)`). With `part_masks` None, nothing below changes.
    """
    canvas = tuple(canvas or config.CANVAS)
    if downscale not in DOWNSCALES:
        raise ValueError(f"unknown downscale {downscale!r}; known: "
                         f"{sorted(DOWNSCALES)}")
    steps: list[str] = []

    if imaging.opaque_box(cell) is None:
        raise ValueError("cell is fully transparent after matting - nothing to "
                         "publish. Check the background prompt or --matte.")
    cell, dropped = imaging.keep_main_blobs(cell)
    if dropped:
        steps.append(f"dropped {dropped} stray island(s)")
    trim_box = imaging.opaque_box(cell) if part_masks is not None else None
    trimmed = imaging.trim(cell)
    alpha = trimmed.getchannel("A")
    steps.append(f"trim to {trimmed.width}x{trimmed.height}")
    # Bleed the edge colours outward before flattening, so the downscale
    # averages an edge pixel with the sprite and not with the chroma key (see
    # imaging.bleed_edges). The key still fills everything beyond the bleed.
    passes = 2 * math.ceil(trimmed.height / max(1, target_height)) + 2
    bled, grown = imaging.bleed_edges(trimmed, passes)
    flat = Image.new("RGB", trimmed.size, chroma)
    flat.paste(bled.convert("RGB"), (0, 0), grown)
    steps.append(f"bleed edges {passes}px")

    if use_pixelfixer:
        flat, note = grid_repair(flat, target_height)
        steps.append(f"pixelfixer {note}")
    else:
        steps.append("pixelfixer off")

    small, note = DOWNSCALES[downscale](flat, target_height)
    steps.append(f"{note} -> {small.width}x{small.height} "
                 f"(target {target_height}px tall)")

    fitted = imaging.scale_to_height(small, min(target_height, canvas[1]))
    if fitted.width > canvas[0]:
        ratio = canvas[0] / fitted.width
        fitted = fitted.resize((canvas[0], max(1, round(fitted.height * ratio))),
                               Image.NEAREST)
    steps.append(f"fit -> {fitted.width}x{fitted.height}")

    snapped = palette_snap(fitted, palette_png)
    steps.append("magick -dither None -remap")

    mask = alpha.resize(snapped.size, Image.NEAREST)
    rgba = snapped.convert("RGBA")
    rgba.putalpha(mask)
    rgba = imaging.binarise_alpha(rgba)
    steps.append("re-apply binary mask")

    if part_masks is not None:
        small = partkit.downscale_labels(part_masks.labels, rgba.size, trim_box)
        if part_palettes is not None:
            rgba, part_report = partkit.lock(rgba, small, part_masks.names,
                                             part_palettes)
            locked = sum(1 for r in part_report.values() if r.get("locked"))
            steps.append(f"part palette lock ({locked}/{len(part_masks.names)} "
                         f"part(s) locked)")
        else:
            palettes = partkit.extract_palettes(rgba, small, part_masks.names)
            part_report = {
                name: {"pixels": int(((small == i) & (np.asarray(rgba)[:, :, 3] > 0)).sum()),
                       "colours": [list(c) for c in palettes[name]]}
                for i, name in enumerate(part_masks.names)}
            steps.append("part palettes extracted")
        if parts_out is not None:
            # Only the sprite's own pixels carry a label on the canvas map.
            small = np.where(np.asarray(rgba)[:, :, 3] > 0, small, -1)
            placed = np.full((canvas[1], canvas[0]), -1, dtype=np.int16)
            left = (canvas[0] - rgba.width) // 2
            top = canvas[1] - rgba.height
            h = min(rgba.height, canvas[1] - max(top, 0))
            w = min(rgba.width, canvas[0] - max(left, 0))
            placed[max(top, 0):max(top, 0) + h, max(left, 0):max(left, 0) + w] = \
                small[:h, :w]
            parts_out.update({"report": part_report, "labels": placed,
                              "found": dict(part_masks.found),
                              "scores": dict(part_masks.scores),
                              "reasons": dict(part_masks.reasons)})

    framed = imaging.recanvas(rgba, canvas, "bottom-center")
    steps.append(f"recanvas {canvas[0]}x{canvas[1]} bottom-center")
    return framed, steps


# ---------------------------------------------------------------------------
# The part palette lock feature - where the palettes come from, and the masks
# ---------------------------------------------------------------------------
def load_part_palettes(slug: str, explicit: Path | None = None, *,
                       assets_dir: Path | None = None,
                       work_dir: Path | None = None
                       ) -> tuple[dict[str, list[list[int]]], str]:
    """(part -> colours, where they came from), in precedence order:

      1. `--parts-from FILE` - a `post/parts.json` or a unit sidecar;
      2. the published unit sprite's sidecar, `<assets>/2d/units/<slug>.json`
         `generation.parts.palettes` - the accepted sprite is the source of truth;
      3. `work/<slug>_base/post/parts.json` - the last `post --parts` on the base.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    else:
        base = Path(assets_dir) if assets_dir is not None else config.ASSETS_DIR
        candidates.append(base / "2d" / "units" / f"{slug}.json")
        candidates.append(Job.open(f"{slug}_base", work_dir).post_dir / "parts.json")
    for path in candidates:
        if not path.exists():
            continue
        data = read_json(path)
        palettes = (data.get("palettes") if data.get("schema") == partkit.PARTS_SCHEMA
                    else ((data.get("generation") or {}).get("parts") or {}).get("palettes"))
        if palettes:
            return {k: [list(c) for c in v] for k, v in palettes.items()}, str(path)
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"no part palettes for {slug!r} (tried {tried}). Run "
        f"`anim post {slug}_base --parts` on the accepted base sprite first, "
        f"publish it, or pass --parts-from.")


def _mask_cells(job: Job, *, slug: str, cells: list[Cell],
                segmenter: partkit.Segmenter | None
                ) -> tuple[list[partkit.PartMasks], list[str]]:
    """SAM masks for every cell (strip or clip route), prompted from keypoints.json."""
    table = config.parts_for_unit(slug)
    if not table:
        raise ValueError(f"no part table for unit {slug!r} - add a row to "
                         f"config.UNIT_PARTS before locking its palette")
    if not job.keypoints_path.exists():
        raise FileNotFoundError(f"no {job.keypoints_path} - the part prompts come "
                                f"from the rig's joints (run `anim rig`/`pose`)")
    payload = read_json(job.keypoints_path)
    # The canned base pose carries no `names`; resolve them from the rig like
    # posekit.series_from does.
    names = list(payload.get("names") or posekit.rig_names(payload["rig"]))
    source_size = tuple(payload["source_size"])
    thr = float(payload.get("kpt_thr", config.POSE_KPT_THR))
    by_index = {f.get("index", i): f for i, f in enumerate(payload["frames"])}
    segmenter = segmenter or partkit.Sam2Segmenter()

    masks: list[partkit.PartMasks] = []
    for cell_meta, cell, hd_rgb in cells:
        entry = by_index.get(cell_meta["frame"]) or {}
        prompts = partkit.prompts_for(table, entry, names, source_size,
                                      cell.size, thr)
        rgb = np.asarray(hd_rgb.convert("RGB"))
        alpha = np.asarray(cell)[:, :, 3]
        results = segmenter.segment(rgb, prompts)
        masks.append(partkit.resolve(table, prompts, results, alpha,
                                     min_score=config.PART_MIN_SCORE))
    return masks, [p.name for p in table]


def _save_part_images(job: Job, stem: str, index: int, *, hd_rgb: Image.Image,
                      masks: partkit.PartMasks, parts_out: dict[str, Any]) -> None:
    """The HD overlay and the 64px label map, for the report and the review page."""
    out = job.post_dir / "parts"
    out.mkdir(parents=True, exist_ok=True)
    partkit.overlay(hd_rgb, masks.labels, masks.names).save(out / f"{stem}_{index}_hd.png")
    labels = parts_out.get("labels")
    if labels is not None:
        arr = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
        for i in range(len(masks.names)):
            arr[labels == i] = (*partkit.PART_COLOURS[i % len(partkit.PART_COLOURS)], 255)
        Image.fromarray(arr, "RGBA").save(out / f"{stem}_{index}_labels.png")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
CLIP_SOURCE = "clip"

# One cell of work: its layout meta, the matted RGBA cell, and the un-matted
# HD RGB crop the part palette lock feature prompts SAM with and draws its
# overlay on.
Cell = tuple[dict[str, Any], Image.Image, Image.Image]


def is_clip_source(source: Path | str | None) -> bool:
    """`--input clip` selects the clip route; anything else is a strip path."""
    return source is not None and str(source) == CLIP_SOURCE


def strip_cells(job: Job, manifest: dict[str, Any], source: Path | None,
                cut: Callable[[Image.Image], Image.Image]) -> tuple[list[Cell], str]:
    """The strip route: matte the whole strip once, slice by layout.json."""
    layout = read_json(job.layout_path) if job.layout_path.exists() else None
    if layout is None:
        raise FileNotFoundError(f"no {job.layout_path} - run `anim strip` first")
    strip_path = Path(source) if source else (
        job.render_dir / (manifest.get("render", {}).get("final")
                          or "identity_pass.png"))
    if not strip_path.exists():
        raise FileNotFoundError(
            f"no generated strip at {strip_path} - run `anim render` first "
            f"(or pass --input)")
    with Image.open(strip_path) as raw:
        strip_img = raw.convert("RGBA")
    matted = cut(strip_img)
    cells: list[Cell] = []
    for cell_meta in layout["cells"]:
        x, y, width, height = cell_meta["rect"]
        box = (x, y, x + width, y + height)
        cells.append(({"index": cell_meta["index"], "frame": cell_meta["frame"]},
                      matted.crop(box), strip_img.crop(box)))
    return cells, str(strip_path)


def clip_cells(job: Job, cut: Callable[[Image.Image], Image.Image]
               ) -> tuple[list[Cell], str]:
    """The clip route: one generated frame per pick, each matted on its own."""
    if not job.clip_frames():
        raise FileNotFoundError(f"no generated frames in {job.clip_frames_dir} - "
                                f"run `anim clip` first")
    if not job.picks_path.exists():
        raise FileNotFoundError(f"no {job.picks_path} - run `anim pick` first")
    picks = list(read_json(job.picks_path).get("picks") or [])
    if not picks:
        raise ValueError(f"{job.picks_path} has no picks")
    cells: list[Cell] = []
    for position, frame_index in enumerate(picks):
        path = job.clip_frames_dir / f"{frame_index + 1:04d}.png"
        if not path.exists():
            raise FileNotFoundError(f"pick {frame_index} has no generated frame at "
                                    f"{path}; re-run `anim clip`")
        with Image.open(path) as raw:
            hd = raw.convert("RGBA")
        cells.append(({"index": position, "frame": frame_index,
                       "source": path.name}, cut(hd), hd))
    return cells, str(job.clip_frames_dir)


def post(job_name: str, *, palette: Path | None = None, matte: str = "rembg",
         source: Path | str | None = None, use_pixelfixer: bool = True,
         downscale: str = DEFAULT_DOWNSCALE,
         parts: bool | None = None, parts_from: Path | None = None,
         segmenter: partkit.Segmenter | None = None,
         assets_dir: Path | None = None,
         work_dir: Path | None = None) -> dict[str, Any]:
    job = Job.open(job_name, work_dir)
    manifest = job.read_manifest()
    if matte not in MATTES:
        raise ValueError(f"unknown matte {matte!r}; known: {sorted(MATTES)}")
    route = "clip" if is_clip_source(source) else "strip"
    if route == "clip":
        cells, source_label = clip_cells(job, MATTES[matte])
    else:
        cells, source_label = strip_cells(job, manifest, source, MATTES[matte])

    palette_path = Path(palette) if palette else config.PALETTE
    colours = imaging.read_palette(palette_path)
    chroma = imaging.chroma_key_for(colours)
    job.post_dir.mkdir(parents=True, exist_ok=True)
    palette_png = imaging.palette_png(colours, job.post_dir / "palette.png")

    slug = manifest.get("slug", "unit")
    action = manifest.get("action") or ""
    # `base` has no action, so its stem is just the slug - and its frames must
    # not come out as "knight__0.png".
    stem = f"{slug}_{action}" if action else slug
    target_height = config.height_for_unit(slug)

    for stale in job.post_dir.glob(f"{stem}_*.png"):
        stale.unlink()

    # The part palette lock feature. `parts` None -> the unit's config default.
    lock_on = parts if parts is not None else config.part_lock_default(slug)
    part_masks: list[partkit.PartMasks] | None = None
    part_names: list[str] = []
    part_palettes: dict[str, list[list[int]]] | None = None
    palettes_from: str | None = None
    if lock_on:
        part_masks, part_names = _mask_cells(job, slug=slug, cells=cells,
                                             segmenter=segmenter)
        if action:   # an action locks to the base sprite; the base extracts
            part_palettes, palettes_from = load_part_palettes(
                slug, parts_from, assets_dir=assets_dir, work_dir=work_dir)
        for stale in (job.post_dir / "parts").glob(f"{stem}_*.png"):
            stale.unlink()

    records: list[dict[str, Any]] = []
    problems_total = 0
    for cell_index, (cell_meta, cell, hd_rgb) in enumerate(cells):
        parts_out: dict[str, Any] = {}
        frame, steps = process_cell(cell, target_height=target_height,
                                    palette_png=palette_png, chroma=chroma,
                                    use_pixelfixer=use_pixelfixer,
                                    downscale=downscale,
                                    part_masks=(part_masks[cell_index]
                                                if part_masks else None),
                                    part_palettes=part_palettes,
                                    parts_out=parts_out)
        problems = imaging.verify_frame(frame, colours)
        problems_total += len(problems)
        name = f"{stem}_{cell_meta['index']}.png"
        if not problems:
            frame.save(job.post_dir / name)
        record = {
            "index": cell_meta["index"],
            "frame": cell_meta["frame"],
            "file": name,
            "size": list(frame.size),
            "colours": len(imaging.opaque_colours(frame)),
            "alpha_binary": imaging.alpha_is_binary(frame),
            "steps": steps,
            "problems": problems,
        }
        if part_masks:
            masks = part_masks[cell_index]
            record["parts"] = {
                name_: {**parts_out["report"].get(name_, {}),
                        "found": masks.found.get(name_, False),
                        "score": masks.scores.get(name_),
                        **({"reason": masks.reasons[name_]}
                           if name_ in masks.reasons else {})}
                for name_ in part_names}
            _save_part_images(job, stem, cell_meta["index"], hd_rgb=hd_rgb,
                              masks=masks, parts_out=parts_out)
        records.append(record)

    report = {
        "schema": POST_SCHEMA,
        "job": job.name,
        "slug": slug,
        "action": action,
        "route": route,
        "source": source_label,
        "matte": matte,
        "downscale": downscale,
        "target_height": target_height,
        "palette": {"path": str(palette_path), "colours": len(colours),
                    "chroma_key": list(chroma)},
        "tools": {
            "magick": tool_version("magick", "-version"),
            "pixelfixer": tool_version("pixelfixer", "--help"),
            "pixeloe": ("pixeloe.legacy.pixelize (contrast)"
                        if downscale == "pixeloe" else None),
            "rembg_model": REMBG_MODEL if matte == "rembg" else None,
        },
        "frames": records,
    }
    if part_masks:
        if part_palettes is None:
            # EXTRACT: the base sprite's part palettes, the union over its cells
            # (one, for a base job). Written where `load_part_palettes` looks
            # and where `publish` copies it into the sprite's sidecar.
            extracted: dict[str, list[list[int]]] = {n: [] for n in part_names}
            pixels: dict[str, int] = {n: 0 for n in part_names}
            for record in records:
                for name_, info in (record.get("parts") or {}).items():
                    pixels[name_] += int(info.get("pixels") or 0)
                    for colour in info.get("colours") or []:
                        if colour not in extracted[name_]:
                            extracted[name_].append(colour)
            # A part the base barely shows is not a palette. Locking a raised
            # shield to the three pixels of shield the standing sprite has
            # flattens it (knight_strike, 2026-09-17); better to leave that
            # part unlocked, say so, and let the owner fix the base or the
            # file by hand.
            too_small = {n: pixels[n] for n in part_names
                         if 0 < pixels[n] < config.PART_MIN_PALETTE_PIXELS}
            part_palettes = {n: (cols if n not in too_small and cols else [])
                             for n, cols in extracted.items()}
            write_json_in_place(job.post_dir / "parts.json", {
                "schema": partkit.PARTS_SCHEMA, "slug": slug, "job": job.name,
                "model": partkit.model_entry(), "parts": part_names,
                "palettes": part_palettes,
                "pixels": pixels,
                "rejected": {n: {"pixels": px, "colours": extracted[n],
                                 "reason": f"under {config.PART_MIN_PALETTE_PIXELS} "
                                           f"px on the base sprite"}
                             for n, px in too_small.items()},
                "found": {n: any((r.get("parts") or {}).get(n, {}).get("found")
                                 for r in records) for n in part_names},
                "_note": ("Hand-editable: `anim post <job> --parts-from` this "
                          "file, or publish the base and every action reads "
                          "the sprite's sidecar. Colours are [r, g, b] members "
                          "of the unit palette."),
            })
        report["parts"] = {
            "schema": partkit.PARTS_SCHEMA,
            "mode": "lock" if action else "extract",
            "model": partkit.model_entry(),
            "min_score": config.PART_MIN_SCORE,
            "parts": part_names,
            "palettes": part_palettes,
            "palettes_from": palettes_from,
        }
    write_json_in_place(job.report_path, report)
    if problems_total:
        bad = [f"{r['file']}: {'; '.join(r['problems'])}"
               for r in records if r["problems"]]
        raise RuntimeError("post-chain verification failed:\n  - "
                           + "\n  - ".join(bad)
                           + f"\nSee {job.report_path}")
    job.update_manifest(post={"frames": len(records), "palette": str(palette_path),
                              "matte": matte, "downscale": downscale,
                              "route": route, "parts": bool(part_masks)})
    return report


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", help="job name, e.g. knight_strike")
    parser.add_argument("--palette", default=None,
                        help=f"palette .gpl or one-row .png (default "
                             f"{config.PALETTE})")
    parser.add_argument("--matte", default="rembg", choices=sorted(MATTES),
                        help="rembg = isnet-anime, the real cut (downloads a "
                             "model on first use); chroma = deterministic "
                             "flat-background key, correct for the workflows' "
                             "plain-white prompts and the only one that works "
                             "on a drawn fixture strip; none = trust the alpha "
                             "already there")
    parser.add_argument("--input", dest="source", default=None,
                        help="strip to slice (default render/<final>), or the "
                             "word `clip` to take the picked frames from "
                             "clip/frames/ (the clip render feature)")
    parser.add_argument("--no-pixelfixer", dest="pixelfixer",
                        action="store_false",
                        help="skip the grid-repair step")
    parser.add_argument("--downscale", default=DEFAULT_DOWNSCALE,
                        choices=sorted(DOWNSCALES),
                        help="box = area-average straight to the target height "
                             "(default); pixeloe = contrast-aware pixelize, "
                             "noisier on real renders (anchor gate, 2026-09-15)")
    parser.add_argument("--parts", dest="parts", action="store_true",
                        default=None,
                        help="the part palette lock feature: mask each frame "
                             "into the unit's parts (SAM 2.1, prompted from "
                             "the rig) and snap every part to its palette on "
                             "the base sprite; on a base job, extract those "
                             "palettes instead. Default: config.PART_LOCK_DEFAULT")
    parser.add_argument("--no-parts", dest="parts", action="store_false",
                        help="force the lock off")
    parser.add_argument("--parts-from", dest="parts_from", default=None,
                        metavar="FILE",
                        help="part palettes to lock to (a post/parts.json or a "
                             "unit sidecar); default: the published sprite's "
                             "sidecar, then work/<slug>_base/post/parts.json")
    parser.set_defaults(pixelfixer=True)


def run_stage(args: argparse.Namespace) -> int:
    report = post(args.job, palette=args.palette, matte=args.matte,
                  source=args.source, use_pixelfixer=args.pixelfixer,
                  downscale=args.downscale, parts=args.parts,
                  parts_from=Path(args.parts_from) if args.parts_from else None)
    print(f"post: {len(report['frames'])} frame(s) at "
          f"{config.CANVAS[0]}x{config.CANVAS[1]}, "
          f"{report['palette']['colours']}-colour palette")
    for record in report["frames"]:
        print(f"  {record['file']}  {record['colours']} colour(s)")
        for name, info in (record.get("parts") or {}).items():
            state = ("locked" if info.get("locked") else
                     "extracted" if "drift" not in info and info.get("found") else
                     f"skipped ({info.get('reason', 'not found')})")
            drift = f", drift {info['drift']}" if info.get("drift") is not None else ""
            print(f"      {name:<8} {info.get('pixels', 0):4d} px, "
                  f"{len(info.get('colours') or [])} colour(s){drift}  {state}")
    if report.get("parts"):
        p = report["parts"]
        where = (f"locked to {p['palettes_from']}" if p["mode"] == "lock"
                 else f"palettes -> {Job.open(args.job).post_dir / 'parts.json'}")
        print(f"  parts: {p['mode']} ({', '.join(p['parts'])}); {where}")
    return 0


run = run_stage
