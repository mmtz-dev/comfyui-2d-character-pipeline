#!/usr/bin/env python3
"""Mixamo .dae -> COCO-18 joint lists as JSON, using mixamotoopenpose's maths only.

the 3D-intermediate spike feature (Feature 9)

Two problems this solves, both of which would otherwise eat a session:

**1. mixamotoopenpose never writes its joint list to disk.** ``convert_dae_to_openpose()``
returns ``[{"candidate": [[x,y,c] x18], "subset": [[...]]}, ...]`` in memory and the CLI goes
straight on to rasterise it. We want the numbers, not its picture -- see RESEARCH.md §3 for why
its picture is the wrong picture.

**2. You cannot just import it.** ``mixamo_to_openpose.py`` builds an ``argparse`` parser at
module level, calls ``parse_args()`` at module level, and dispatches ``convert_dae(...)`` at
module level. A plain ``import`` would parse OUR argv and then either crash or silently run a
conversion. So this loads the source and executes it with those three module-level pieces
removed:

  * ``args = vars(ap.parse_args())``      -> ``args = {}``
  * everything from ``if os.path.isfile(args["input"]):`` to EOF  -> dropped

Both anchors are asserted before the surgery. If upstream is ever re-pinned and the anchors move,
this fails loudly with the line it could not find, rather than executing a half-patched module.

Output is the same JSON ``render_openpose.py`` reads, so the two chain together:

    python3 spike3d/dae_to_joints.py mixamo/slash.dae work/strike/joints.json \\
        --rotation-y 90 --max-frames 4 --width 512 --height 512
    python3 spike3d/render_openpose.py work/strike/joints.json work/strike/skeletons \\
        --width 304 --height 320

``--rotation-y 90`` is what turns Mixamo's front-facing clip into the profile view the art bible
requires (party faces right).
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor" / "mixamotoopenpose" / "mixamo_to_openpose.py"

ANCHOR_ARGS = 'args = vars(ap.parse_args())'
ANCHOR_DISPATCH = 'if os.path.isfile(args["input"]):'


def load_upstream(path: Path = VENDOR) -> types.ModuleType:
    """Execute mixamotoopenpose with its module-level CLI removed."""
    if not path.exists():
        raise SystemExit(
            f"vendored mixamotoopenpose not found at {path}\n"
            "Re-vendor it (MIT):\n"
            "  git clone https://github.com/Astropulse/mixamotoopenpose "
            f"{path.parent.parent}/mixamotoopenpose-src\n"
            "and copy mixamo_to_openpose.py + LICENSE into place. See vendor/.../PIN.txt."
        )

    source = path.read_text(encoding="utf-8")

    for anchor in (ANCHOR_ARGS, ANCHOR_DISPATCH):
        if anchor not in source:
            raise SystemExit(
                f"{path}: expected line not found:\n    {anchor}\n"
                "mixamotoopenpose has changed shape since it was pinned "
                "(66ec14dfd01e81f77da3966d314edffd6a61a9e6). Re-read its tail and update the "
                "anchors in dae_to_joints.py rather than patching around it."
            )

    source = source.replace(ANCHOR_ARGS, "args = {}  # CLI removed by dae_to_joints.py", 1)
    source = source[: source.index(ANCHOR_DISPATCH)]

    module = types.ModuleType("mixamo_to_openpose_lib")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)

    for required in ("convert_dae_to_openpose", "reduce_frames", "openpose_keypoints"):
        if not hasattr(module, required):
            raise SystemExit(f"{path}: loaded, but {required}() is missing")
    return module


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="dae_to_joints.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("dae", type=Path, help="a Mixamo Collada export (download WITHOUT skin, "
                                          "and do NOT reduce keyframes in Mixamo)")
    p.add_argument("out", type=Path, help="joints JSON for render_openpose.py")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--scale", type=float, default=2.0, help="mixamotoopenpose pose scale")
    p.add_argument("--rotation-x", type=int, default=0)
    p.add_argument("--rotation-y", type=int, default=90,
                   help="90 gives the profile view the art bible wants (default)")
    p.add_argument("--rotation-z", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0,
                   help="reduce to N evenly-spaced key poses (0 = keep all); the art bible "
                        "budget is 2-6 per action")
    args = p.parse_args(argv)

    if not args.dae.exists():
        print(f"no such file: {args.dae}", file=sys.stderr)
        return 2
    if args.dae.suffix.lower() != ".dae":
        print(f"{args.dae}: mixamotoopenpose reads Collada .dae only, not "
              f"{args.dae.suffix or 'a directory'}. Re-export from Mixamo as Collada.",
              file=sys.stderr)
        return 2

    m2o = load_upstream()
    print(f"loaded {VENDOR.name} (CLI stripped); keypoints = {len(m2o.openpose_keypoints)}")

    frames = m2o.convert_dae_to_openpose(
        str(args.dae),
        image_size=(args.width, args.height),
        rotation_angles=(args.rotation_x, args.rotation_y, args.rotation_z),
        scale_factor=args.scale,
    )
    print(f"{len(frames)} frame(s) parsed from {args.dae.name}")

    if args.max_frames > 0:
        frames = m2o.reduce_frames(frames, args.max_frames)
        print(f"reduced to {len(frames)} evenly-spaced key pose(s)")

    missing = sum(1 for f in frames for i in f["subset"][0] if int(i) < 0)
    if missing:
        print(f"note: {missing} joint slot(s) across all frames are absent from the rig "
              "(Mixamo rigs usually have no eye/ear bones; mixamotoopenpose synthesises them "
              "from the head bone, so this is only a problem if it is a limb).", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frames, indent=1), encoding="utf-8")
    print(f"joints -> {args.out}")
    print(f"next:  python3 spike3d/render_openpose.py {args.out} <dir> --width 304 --height 320")
    return 0


if __name__ == "__main__":
    sys.exit(main())
