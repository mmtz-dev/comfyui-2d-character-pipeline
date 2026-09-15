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
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from anim import config, imaging
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
                 canvas: tuple[int, int] = None) -> tuple[Image.Image, list[str]]:
    """One generated cell -> one finished 64x64 frame. Pure; no disk state."""
    canvas = tuple(canvas or config.CANVAS)
    steps: list[str] = []

    if imaging.opaque_box(cell) is None:
        raise ValueError("cell is fully transparent after matting - nothing to "
                         "publish. Check the background prompt or --matte.")
    trimmed = imaging.trim(cell)
    alpha = trimmed.getchannel("A")
    flat = Image.new("RGB", trimmed.size, chroma)
    flat.paste(trimmed.convert("RGB"), (0, 0), trimmed)
    steps.append(f"trim to {trimmed.width}x{trimmed.height}")

    if use_pixelfixer:
        flat, note = grid_repair(flat, target_height)
        steps.append(f"pixelfixer {note}")
    else:
        steps.append("pixelfixer off")

    small, note = pixel_downscale(flat, target_height)
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

    framed = imaging.recanvas(rgba, canvas, "bottom-center")
    steps.append(f"recanvas {canvas[0]}x{canvas[1]} bottom-center")
    return framed, steps


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def post(job_name: str, *, palette: Path | None = None, matte: str = "rembg",
         source: Path | None = None, use_pixelfixer: bool = True,
         work_dir: Path | None = None) -> dict[str, Any]:
    job = Job.open(job_name, work_dir)
    manifest = job.read_manifest()
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

    palette_path = Path(palette) if palette else config.PALETTE
    colours = imaging.read_palette(palette_path)
    chroma = imaging.chroma_key_for(colours)
    job.post_dir.mkdir(parents=True, exist_ok=True)
    palette_png = imaging.palette_png(colours, job.post_dir / "palette.png")

    if matte not in MATTES:
        raise ValueError(f"unknown matte {matte!r}; known: {sorted(MATTES)}")
    with Image.open(strip_path) as raw:
        strip_img = raw.convert("RGBA")
    cut = MATTES[matte](strip_img)

    slug = manifest.get("slug", "unit")
    action = manifest.get("action") or ""
    # `base` has no action, so its stem is just the slug - and its frames must
    # not come out as "knight__0.png".
    stem = f"{slug}_{action}" if action else slug
    target_height = config.height_for_unit(slug)

    for stale in job.post_dir.glob(f"{stem}_*.png"):
        stale.unlink()

    records: list[dict[str, Any]] = []
    problems_total = 0
    for cell_meta in layout["cells"]:
        x, y, width, height = cell_meta["rect"]
        cell = cut.crop((x, y, x + width, y + height))
        frame, steps = process_cell(cell, target_height=target_height,
                                    palette_png=palette_png, chroma=chroma,
                                    use_pixelfixer=use_pixelfixer)
        problems = imaging.verify_frame(frame, colours)
        problems_total += len(problems)
        name = f"{stem}_{cell_meta['index']}.png"
        if not problems:
            frame.save(job.post_dir / name)
        records.append({
            "index": cell_meta["index"],
            "frame": cell_meta["frame"],
            "file": name,
            "size": list(frame.size),
            "colours": len(imaging.opaque_colours(frame)),
            "alpha_binary": imaging.alpha_is_binary(frame),
            "steps": steps,
            "problems": problems,
        })

    report = {
        "schema": POST_SCHEMA,
        "job": job.name,
        "slug": slug,
        "action": action,
        "source": str(strip_path),
        "matte": matte,
        "target_height": target_height,
        "palette": {"path": str(palette_path), "colours": len(colours),
                    "chroma_key": list(chroma)},
        "tools": {
            "magick": tool_version("magick", "-version"),
            "pixelfixer": tool_version("pixelfixer", "--help"),
            "pixeloe": "pixeloe.legacy.pixelize (contrast)",
            "rembg_model": REMBG_MODEL if matte == "rembg" else None,
        },
        "frames": records,
    }
    write_json_in_place(job.report_path, report)
    if problems_total:
        bad = [f"{r['file']}: {'; '.join(r['problems'])}"
               for r in records if r["problems"]]
        raise RuntimeError("post-chain verification failed:\n  - "
                           + "\n  - ".join(bad)
                           + f"\nSee {job.report_path}")
    job.update_manifest(post={"frames": len(records), "palette": str(palette_path),
                              "matte": matte})
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
                        help="strip to slice (default render/identity_pass.png)")
    parser.add_argument("--no-pixelfixer", dest="pixelfixer",
                        action="store_false",
                        help="skip the grid-repair step")
    parser.set_defaults(pixelfixer=True)


def run_stage(args: argparse.Namespace) -> int:
    report = post(args.job, palette=args.palette, matte=args.matte,
                  source=args.source, use_pixelfixer=args.pixelfixer)
    print(f"post: {len(report['frames'])} frame(s) at "
          f"{config.CANVAS[0]}x{config.CANVAS[1]}, "
          f"{report['palette']['colours']}-colour palette")
    for record in report["frames"]:
        print(f"  {record['file']}  {record['colours']} colour(s)")
    return 0


run = run_stage
