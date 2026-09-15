#!/usr/bin/env python3
"""Compose N skeleton PNGs into one non-square strip. A stand-in for `anim strip`.

the 3D-intermediate spike feature (Feature 9)

`anim strip` is still a scaffold (`anim/stages/strip.py` prints "not implemented" and exits 2),
and the spike must not be blocked on Feature 3 landing. This does the one thing the spike needs,
with the same geometry contract, so the two are interchangeable:

  * 4 cells of 304x320 -> 1216x320 (``anim.config.STRIP_CELLS`` / ``STRIP_CELL``)
  * NEVER square: Qwen-Image-Edit degrades on a square aspect (upstream issue #243)
  * both dimensions divisible by 16, for EmptySD3LatentImage

When `anim strip` is implemented, delete this and call that instead -- and check the cell
geometry still matches, because `anim post` slices by it.

    docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \\
      run --rm --entrypoint python anim spike3d/make_strip.py \\
        spike3d/work/strike/skeletons spike3d/work/strike/strip.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

DEFAULT_CELL = (304, 320)
DEFAULT_CELLS = 4


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="make_strip.py", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="directory of numbered skeleton PNGs, or a file list")
    p.add_argument("dst", type=Path)
    p.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    p.add_argument("--cell-width", type=int, default=DEFAULT_CELL[0])
    p.add_argument("--cell-height", type=int, default=DEFAULT_CELL[1])
    args = p.parse_args(argv)

    if not args.src.exists():
        print(f"no such directory or file list: {args.src}", file=sys.stderr)
        return 2
    if args.src.is_dir():
        paths = sorted(args.src.glob("*.png"))
    else:
        paths = [Path(line.strip()) for line in args.src.read_text().splitlines() if line.strip()]

    if not paths:
        print(f"no PNGs under {args.src}", file=sys.stderr)
        return 2
    if len(paths) != args.cells:
        print(
            f"{len(paths)} frame(s) but --cells {args.cells}. Pick exactly {args.cells} key "
            "poses (the art bible says 2-6 per action) and re-run.",
            file=sys.stderr,
        )
        return 2

    cw, ch = args.cell_width, args.cell_height
    width = cw * args.cells
    if width == ch:
        print(f"refusing a square {width}x{ch} strip - Qwen-Image-Edit issue #243", file=sys.stderr)
        return 2
    for name, value in (("strip width", width), ("cell height", ch)):
        if value % 16:
            print(f"{name} {value} is not divisible by 16 (latent step)", file=sys.stderr)
            return 2

    strip = Image.new("RGB", (width, ch), (0, 0, 0))
    for i, path in enumerate(paths):
        cell = Image.open(path).convert("RGB")
        if cell.size != (cw, ch):
            # Contain, not stretch: a squashed rig is a different pose.
            cell.thumbnail((cw, ch), Image.NEAREST)
            pad = Image.new("RGB", (cw, ch), (0, 0, 0))
            pad.paste(cell, ((cw - cell.width) // 2, (ch - cell.height) // 2))
            cell = pad
        strip.paste(cell, (i * cw, 0))

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    strip.save(args.dst)
    print(f"{args.dst}  {width}x{ch}  cell {cw}x{ch} x{args.cells}")
    for i, path in enumerate(paths):
        print(f"  cell {i}: {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
