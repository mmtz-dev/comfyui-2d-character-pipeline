#!/usr/bin/env python3
"""TripoSG image -> 3D mesh, with the non-commercial background remover designed out.

the 3D-intermediate spike feature (Feature 9)
=============================================

This is a replacement for upstream's ``scripts/inference_triposg.py``. It differs in exactly
one way that matters, and the reason is a licence:

Upstream unconditionally does::

    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)

``briaai/RMBG-1.4`` is ``license: other`` / ``bria-rmbg-1.4``, and its model card says: *"The
model is released under a Creative Commons license for non-commercial use. … Commercial use is
subject to a commercial agreement with BRIA."* game-02 ships commercially, so that model may not
be in the chain -- even as a preprocessing step, even though only the mesh is kept.

It does not have to be. ``scripts/image_process.py::load_image`` only reaches for ``rmbg_net``
when the input has no usable alpha channel::

    elif num_channels == 4:
        b, g, r, alpha = cv2.split(img)
        if not is_valid_alpha(alpha):
            alpha = None          # -> the branch that calls rmbg()
        else:
            alpha_gpu = ...       # -> the branch that does not

So an RGBA PNG with a genuinely binary alpha -- which is precisely what this pipeline already
guarantees (``anim.config.ALPHA_VALUES == (0, 255)``) -- never touches BRIA's model.

This script therefore **requires** such an input and refuses anything else, loudly. It does not
fall back, because a silent fallback is how a non-commercial model ends up in a shipped game.
``briarmbg`` is never imported and RMBG-1.4 is never downloaded.

Usage (native daemon; see compose.yml and README.md)::

    docker --context docker-default compose -f tools/anim-pipeline/spike3d/compose.yml \\
      run --rm triposg \\
        --image-input /work/in/knight_apose.png \\
        --output-path /work/out/knight.glb \\
        --obj /work/out/knight.obj \\
        --faces 20000 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# TripoSG's own modules; PYTHONPATH is set in the Dockerfile.
try:
    import torch
    import trimesh
    from huggingface_hub import snapshot_download
    from PIL import Image

    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from image_process import prepare_image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"TripoSG imports failed ({exc}).\n"
        "This script runs inside spike3d-triposg:local, not on the host:\n"
        "  docker --context docker-default compose "
        "-f tools/anim-pipeline/spike3d/compose.yml run --rm triposg --help"
    ) from exc

# Deliberately NOT imported: `from briarmbg import BriaRMBG`. See the module docstring.

TRIPOSG_REPO = "VAST-AI/TripoSG"
ALPHA_BINS = 20
ALPHA_MIN_RATIO = 0.01


def check_alpha_keeps_rmbg_out(path: Path) -> tuple[int, int, float, float]:
    """Reimplements TripoSG's ``is_valid_alpha`` gate, so we fail here instead of silently
    handing the image to BRIA's model.

    Upstream: 20 histogram bins over 0..255; the alpha is "valid" when at least 1 % of pixels
    fall in the lowest bin (fully transparent) AND at least 1 % in the highest (fully opaque).

    Returns ``(width, height, transparent_fraction, opaque_fraction)``. Raises on anything that
    would take the RMBG branch.
    """
    img = Image.open(path)
    if img.mode != "RGBA":
        raise SystemExit(
            f"{path} is mode {img.mode!r}, not RGBA.\n"
            "TripoSG would fall back to briaai/RMBG-1.4, which is non-commercial and must not\n"
            "enter this pipeline. Cut the background first with the CPU chain's own remover:\n"
            "  rembg -m isnet-anime -dc -ppm  (MIT; already in the anim-pipeline image)"
        )

    alpha = np.array(img.split()[-1], dtype=np.uint8)
    h, w = alpha.shape
    hist, _ = np.histogram(alpha, bins=ALPHA_BINS, range=(0, 256))
    total = float(h * w)
    transparent = hist[0] / total
    opaque = hist[-1] / total

    if transparent < ALPHA_MIN_RATIO or opaque < ALPHA_MIN_RATIO:
        raise SystemExit(
            f"{path}: alpha channel is not usable as a mask "
            f"(fully-transparent {transparent:.3%}, fully-opaque {opaque:.3%}; "
            f"each must be >= {ALPHA_MIN_RATIO:.0%}).\n"
            "TripoSG's is_valid_alpha() would reject it and fall back to briaai/RMBG-1.4,\n"
            "which is non-commercial. Re-cut the mask so alpha is strictly 0 or 255."
        )

    mid = int(((alpha > 0) & (alpha < 255)).sum())
    if mid:
        print(
            f"  note: {mid} pixel(s) have partial alpha. The gate passes, so RMBG stays out, "
            "but 64px sprites want alpha strictly in {0, 255}.",
            file=sys.stderr,
        )
    return w, h, transparent, opaque


@torch.no_grad()
def generate(
    image_path: Path,
    weights_dir: Path,
    seed: int,
    steps: int,
    guidance: float,
    faces: int,
) -> trimesh.Trimesh:
    # rmbg_net is None and stays None: the RGBA gate above guarantees it is never dereferenced.
    img_pil = prepare_image(str(image_path), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None)

    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(str(weights_dir)).to(
        "cuda", torch.float16
    )
    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).samples[0]

    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if faces > 0 and mesh.faces.shape[0] > faces:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=faces)
        cur = ms.current_mesh()
        mesh = trimesh.Trimesh(vertices=cur.vertex_matrix(), faces=cur.face_matrix())

    return mesh


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_triposg.py",
        description=(
            "TripoSG single-image 3D, with briaai/RMBG-1.4 (non-commercial) designed out. "
            "Requires an RGBA input with binary alpha."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image-input", type=Path, required=True, help="RGBA PNG, binary alpha")
    p.add_argument("--output-path", type=Path, default=Path("/work/out/mesh.glb"))
    p.add_argument(
        "--obj",
        type=Path,
        default=None,
        help="also write an OBJ here - Mixamo's auto-rigger takes FBX/OBJ/ZIP, not GLB",
    )
    p.add_argument("--seed", type=int, default=42, help="recorded in the sidecar; keep it fixed")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=7.0)
    p.add_argument(
        "--faces",
        type=int,
        default=20000,
        help="decimate to this many faces; Mixamo's auto-rigger prefers a light, clean mesh",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=Path(os.environ.get("TRIPOSG_WEIGHTS", "/weights/TripoSG")),
    )
    p.add_argument("--skip-download", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.image_input.exists():
        print(f"no such image: {args.image_input}", file=sys.stderr)
        return 2

    w, h, transparent, opaque = check_alpha_keeps_rmbg_out(args.image_input)
    print(
        f"input {args.image_input}  {w}x{h} RGBA  "
        f"transparent {transparent:.1%} / opaque {opaque:.1%}  -> RMBG-1.4 not needed"
    )

    if not args.skip_download:
        print(f"fetching {TRIPOSG_REPO} (MIT, ~8 GB) into {args.weights} ...")
        snapshot_download(repo_id=TRIPOSG_REPO, local_dir=str(args.weights))
    elif not args.weights.exists():
        print(f"--skip-download given but {args.weights} does not exist", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print(
            "CUDA is not available in this container. TripoSG hard-codes device='cuda'.\n"
            "This service must run on the NATIVE daemon:\n"
            "  sudo systemctl start docker   # owner, real TTY\n"
            "  docker --context docker-default compose "
            "-f tools/anim-pipeline/spike3d/compose.yml run --rm triposg ...",
            file=sys.stderr,
        )
        return 2
    print(f"gpu: {torch.cuda.get_device_name(0)}")

    mesh = generate(
        args.image_input,
        args.weights,
        seed=args.seed,
        steps=args.num_inference_steps,
        guidance=args.guidance_scale,
        faces=args.faces,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(args.output_path))
    print(f"mesh -> {args.output_path}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

    if args.obj:
        args.obj.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(args.obj))
        print(f"obj  -> {args.obj}  (upload this one to Mixamo)")

    print(f"seed={args.seed} steps={args.num_inference_steps} guidance={args.guidance_scale} "
          f"faces={args.faces}  # record these in DEVLOG.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
