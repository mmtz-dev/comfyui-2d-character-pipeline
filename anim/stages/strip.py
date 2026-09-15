"""Stage: `anim strip` - picks -> one N x 1 strip

Compose the picked skeleton PNGs left to right into ONE strip, and prepare the
reference sprite that gives the generated poses their identity.

    anim strip knight_strike

    strip/skeleton_strip.png   N cells wide, 1 tall  (4 picks -> 1216 x 320)
    strip/reference_512.png    the unit's 64px sprite, nearest x8, on white
    strip/reference_strip.png  the same sprite in the same N x 1 cell grid
    strip/layout.json          every cell's rect, and which frame is in it

WHY ONE STRIP AND NOT N IMAGES
------------------------------
One `/prompt` per pose would give four unrelated characters. The strip is the
consistency mechanism: the model sees all N poses at once and has to keep one
character across them.

WHY IT IS NEVER SQUARE
----------------------
Qwen-Image-Edit degrades on square aspect ratios (upstream issue #243), so the
strip is checked, not merely intended, to be non-square - and both dimensions
must be multiples of 16 because EmptySD3LatentImage has step 16 and a stray
1215-wide strip would fail inside ComfyUI rather than here.

WHY THE REFERENCE IS PRE-UPSCALED BY 8
--------------------------------------
The unit's sprite is 64px. Handed to a VL encoder at 64px it is a smudge. x8
nearest to 512 is the standard trick: the encoder sees the shapes, and because
the scale is an integer and the filter is nearest, not one new colour is
invented on the way. `identity_pass.json`'s ImageScale node then re-scales
512 -> 512, which is a no-op, so the pre-upscale cannot fight it.

`reference_strip.png` is the same sprite laid into the SAME cell grid, feet on
the same row as the skeleton's feet. That is an optional second identity hint
for the edit pass; the largest whole-number scale that fits the cell is used, so
it stays crisp where the 512 version has to be letterboxed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from anim import config, imaging
from anim.job import Job
from anim.util import read_json, write_json_in_place

NAME = "strip"
HELP = "picks -> one N x 1 skeleton strip"

LAYOUT_SCHEMA = "anim.layout/1"
REFERENCE_UPSCALE = 8
# NOTES.md finding 6: LoadImage drops alpha, so whatever is behind the sprite
# becomes visible context for the VL encoder. White, to match the prompts'
# "plain flat white background".
REFERENCE_BG = (255, 255, 255)


def check_strip_size(width: int, height: int) -> None:
    """Fail here rather than inside ComfyUI twenty minutes later."""
    if width == height:
        raise ValueError(
            f"strip is square ({width}x{height}). Qwen-Image-Edit degrades on "
            f"square aspect ratios (upstream issue #243) - change the cell size "
            f"or the pick count.")
    step = config.LATENT_STEP
    if width % step or height % step:
        raise ValueError(
            f"strip is {width}x{height}; EmptySD3LatentImage has step {step}, "
            f"so both dimensions must be multiples of {step}.")


def compose(images: list[Image.Image], cell: tuple[int, int],
            background=(0, 0, 0)) -> Image.Image:
    """Lay images left to right into equal cells. Each is fitted, never stretched."""
    width, height = cell[0] * len(images), cell[1]
    check_strip_size(width, height)
    mode = "RGB" if len(background) == 3 else "RGBA"
    canvas = Image.new(mode, (width, height), background)
    for index, image in enumerate(images):
        tile = image.convert(mode)
        if tile.size != tuple(cell):
            tile = imaging.fit_into_cell(
                image.convert("RGBA"), cell,
                background=background if mode == "RGB" else None).convert(mode)
        canvas.paste(tile, (index * cell[0], 0))
    return canvas


def reference_sprite(slug: str, override: Path | None = None,
                     assets_dir: Path | None = None) -> Path:
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"no reference sprite at {path}")
        return path
    base = Path(assets_dir) if assets_dir is not None else config.ASSETS_DIR
    path = base / "2d" / "units" / f"{slug}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"no reference sprite at {path}. Generate the unit's standing "
            f"sprite first with `anim base --unit {slug}`, or pass "
            f"--reference.")
    return path


def build_reference(path: Path, cells: int, cell: tuple[int, int],
                    out_dir: Path) -> dict[str, Any]:
    with Image.open(path) as raw:
        sprite = raw.convert("RGBA")

    big = imaging.upscale_nearest(sprite, REFERENCE_UPSCALE)
    flat = Image.new("RGB", big.size, REFERENCE_BG)
    flat.paste(big, (0, 0), big)
    ref_path = out_dir / "reference_512.png"
    flat.save(ref_path)

    scale = imaging.largest_integer_scale(sprite, cell)
    tile = imaging.upscale_nearest(sprite, scale)
    strip = compose([tile] * cells, cell, background=REFERENCE_BG)
    strip_path = out_dir / "reference_strip.png"
    strip.save(strip_path)

    return {
        "file": ref_path.name,
        "source": str(path),
        "source_size": list(sprite.size),
        "size": list(flat.size),
        "upscale": REFERENCE_UPSCALE,
        "strip": {"file": strip_path.name, "size": list(strip.size),
                  "cell_upscale": scale},
    }


def strip(job_name: str, *, cell: tuple[int, int] | None = None,
          reference: Path | None = None, work_dir: Path | None = None,
          assets_dir: Path | None = None) -> dict[str, Any]:
    job = Job.open(job_name, work_dir)
    manifest = job.read_manifest()
    if not job.picks_path.exists():
        raise FileNotFoundError(f"no {job.picks_path} - run `anim pick` first")
    picks = list(read_json(job.picks_path).get("picks") or [])
    if not picks:
        raise ValueError(f"{job.picks_path} has no picks")

    cell = tuple(cell or manifest.get("pose", {}).get("cell") or config.STRIP_CELL)
    job.strip_dir.mkdir(parents=True, exist_ok=True)

    tiles: list[Image.Image] = []
    cells: list[dict[str, Any]] = []
    for position, frame_index in enumerate(picks):
        skel = job.pose_dir / f"skel_{frame_index + 1:04d}.png"
        if not skel.exists():
            raise FileNotFoundError(
                f"pick {frame_index} has no skeleton at {skel} - no person was "
                f"detected in that frame. Drop it in `anim review`, or re-run "
                f"`anim pose`.")
        with Image.open(skel) as img:
            tiles.append(img.convert("RGB"))
        cells.append({"index": position, "frame": frame_index,
                      "source": skel.name,
                      "rect": [position * cell[0], 0, cell[0], cell[1]]})

    sheet = compose(tiles, cell)
    skeleton_path = job.strip_dir / "skeleton_strip.png"
    sheet.save(skeleton_path)

    slug = manifest.get("slug", "")
    ref_meta = build_reference(reference_sprite(slug, reference, assets_dir),
                               len(picks), cell, job.strip_dir)

    layout = {
        "schema": LAYOUT_SCHEMA,
        "job": job.name,
        "strip": {"file": skeleton_path.name, "size": list(sheet.size),
                  "cells": len(picks), "cell": list(cell)},
        "cells": cells,
        "reference": ref_meta,
    }
    write_json_in_place(job.layout_path, layout)
    job.update_manifest(strip={"cells": len(picks), "size": list(sheet.size),
                               "cell": list(cell)})
    return layout


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", help="job name, e.g. knight_strike")
    parser.add_argument("--cell-size", default=None, metavar="WxH",
                        help=f"cell size (default "
                             f"{config.STRIP_CELL[0]}x{config.STRIP_CELL[1]})")
    parser.add_argument("--reference", default=None,
                        help="reference sprite PNG (default "
                             "<assets>/2d/units/<slug>.png)")


def _parse_cell(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    try:
        width, height = (int(part) for part in text.lower().split("x", 1))
    except ValueError as exc:
        raise SystemExit(f"--cell-size must look like 304x320, got {text!r}") from exc
    return width, height


def run_stage(args: argparse.Namespace) -> int:
    layout = strip(args.job, cell=_parse_cell(args.cell_size),
                   reference=Path(args.reference) if args.reference else None)
    size = layout["strip"]["size"]
    print(f"strip: {layout['strip']['cells']} cell(s) -> {size[0]}x{size[1]} "
          f"({layout['strip']['file']})")
    print(f"  frames {[c['frame'] for c in layout['cells']]}")
    print(f"  reference {layout['reference']['file']} "
          f"(x{layout['reference']['upscale']} nearest)")
    return 0


run = run_stage
