#!/usr/bin/env python3
"""GLB -> OBJ, for Mixamo's auto-rigger. Pure trimesh (MIT) - no Blender needed.

the 3D-intermediate spike feature (Feature 9)

Mixamo's auto-rigger accepts FBX, OBJ or ZIP, and TripoSG emits GLB, so something has to sit
between them. That something is usually Blender, and it does not need to be: `trimesh` is
already a TripoSG dependency and reads GLB and writes OBJ in four lines. Blender only becomes
necessary on the CC0/BVH route (RESEARCH.md §2.3), where clips must be retargeted onto a
mixamorig-named skeleton.

    docker --context desktop-linux compose -f tools/anim-pipeline/spike3d/compose.yml \\
      run --rm --entrypoint python triposg /opt/spike3d/glb_to_obj.py in.glb out.obj

(This one has no GPU in it, so it is fine on Docker Desktop.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def load_single_mesh(path: Path) -> trimesh.Trimesh:
    """GLB files are scenes; Mixamo wants one mesh."""
    loaded = trimesh.load(str(path), force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise SystemExit(f"{path}: did not resolve to a single mesh (got {type(loaded).__name__})")
    return loaded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="glb_to_obj.py", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument(
        "--rotate-x",
        type=float,
        default=0.0,
        help="degrees about X. Mixamo expects Y-up, standing upright; if the preview shows the "
        "knight lying down, 90 or -90 fixes it. Check the preview before guessing.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="uniform scale. Mixamo is tolerant of size but not of a mesh a thousand units tall.",
    )
    p.add_argument(
        "--center",
        action="store_true",
        help="move the mesh so its base sits at the origin - what the auto-rigger's marker "
        "placement UI expects",
    )
    args = p.parse_args(argv)

    if not args.src.exists():
        print(f"no such file: {args.src}", file=sys.stderr)
        return 2

    mesh = load_single_mesh(args.src)
    print(f"in : {args.src}  {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"extents {np.round(mesh.extents, 3).tolist()}")

    if args.rotate_x:
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(args.rotate_x), [1, 0, 0])
        )
    if args.scale != 1.0:
        mesh.apply_scale(args.scale)
    if args.center:
        lo, hi = mesh.bounds
        mesh.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.dst))
    print(f"out: {args.dst}  extents {np.round(mesh.extents, 3).tolist()}  "
          f"watertight={mesh.is_watertight}")
    if not mesh.is_watertight:
        print("  note: not watertight. Mixamo's auto-rigger wants a clean humanoid mesh; "
              "holes are the usual cause of a failed rig.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
