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

Usage (rootless Podman + CDI; see compose.yml and README.md)::

    tools/anim-pipeline/spike3d/triposg.sh \\
        --image-input /work/in/knight_apose.png \\
        --output-path /work/out/knight.glb \\
        --obj /work/out/knight.obj \\
        --faces 20000 --seed 42

GPU sharing: the run books its memory from the shared GPU gatekeeper (~/Repos/gpu-gatekeeper,
job profile ``anim-triposg``) after the weights are on disk and before the model loads, waits
in its queue while the card is full, and releases it when the run ends. ``--dry-run`` checks
the input, the GPU and the lease without loading the model (and never marks the lease loaded,
so it teaches the gatekeeper nothing).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from PIL import Image

# The shared GPU gatekeeper's client, synced from ~/Repos/gpu-gatekeeper (scripts/sync-client.sh);
# it sits next to this file, which puts it on sys.path.
import gpu_lease

# TripoSG's own modules (PYTHONPATH is set in the Dockerfile) load lazily in _triposg(), so
# --dry-run and the tests need only numpy, Pillow and torch.
_IMPORT_HINT = (
    "This script runs inside localhost/spike3d-triposg:local, not on the host:\n"
    "  tools/anim-pipeline/spike3d/triposg.sh --help"
)


def _triposg():
    try:
        import trimesh
        from huggingface_hub import snapshot_download

        from triposg.pipelines.pipeline_triposg import TripoSGPipeline
        from image_process import prepare_image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"TripoSG imports failed ({exc}).\n{_IMPORT_HINT}") from exc
    return trimesh, snapshot_download, TripoSGPipeline, prepare_image

# Deliberately NOT imported: `from briarmbg import BriaRMBG`. See the module docstring.

TRIPOSG_REPO = "VAST-AI/TripoSG"
GPU_PROFILE = "anim-triposg"  # seed size in gpu-gatekeeper's config/gatekeeper.toml
GPU_HOLDER = "anim-pipeline spike3d triposg"
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


def generate(
    image_path: Path,
    weights_dir: Path,
    seed: int,
    steps: int,
    guidance: float,
    faces: int,
    on_loaded=lambda: None,
):
    """Returns a trimesh.Trimesh. ``on_loaded`` is called once the model is on the GPU."""
    import torch

    trimesh, _, TripoSGPipeline, prepare_image = _triposg()
    # rmbg_net is None and stays None: the RGBA gate above guarantees it is never dereferenced.
    img_pil = prepare_image(str(image_path), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=None)

    with torch.no_grad():
        pipe = TripoSGPipeline.from_pretrained(str(weights_dir)).to("cuda", torch.float16)
        on_loaded()
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
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="check the input, the GPU and the gatekeeper lease; no weights, no model, no mesh",
    )
    return p


def book_gpu() -> "gpu_lease.Lease":
    """Book this run's GPU memory; blocks in the gatekeeper's queue while the card is full.
    No torch cap: the profile's size is a seed, not a measured peak yet."""
    url = os.environ.get("GPU_GATEKEEPER_URL") or gpu_lease.DEFAULT_URL
    print(f"GPU: booking {GPU_PROFILE} from the GPU gatekeeper ({url})", flush=True)
    lease = gpu_lease.acquire(GPU_PROFILE, holder=GPU_HOLDER, url=url, cap_torch=False)
    if lease.guarded:
        print(f"GPU: lease {lease.id} granted, {lease.gb} GB", flush=True)
    else:
        print("GPU: gatekeeper unreachable, running without a lease", flush=True)
    return lease


def dry_run_gpu(lease: "gpu_lease.Lease") -> None:
    """Touch the GPU the way a real run starts, without the model: a small allocation.
    The lease is NOT marked loaded: a loaded run is how the gatekeeper learns the profile's
    size, and a dry run's few hundred MB would teach it a size far too small."""
    import torch

    x = torch.ones((1024, 1024, 64), device="cuda", dtype=torch.float16)  # 128 MiB
    print(f"dry run: {x.numel() * 2 / 2**20:.0f} MiB on {torch.cuda.get_device_name(0)} "
          f"under lease {lease.id}", flush=True)
    del x


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

    if args.dry_run:
        print("dry run: weights skipped")
    elif not args.skip_download:
        _, snapshot_download, _, _ = _triposg()
        print(f"fetching {TRIPOSG_REPO} (MIT, ~8 GB) into {args.weights} ...")
        snapshot_download(repo_id=TRIPOSG_REPO, local_dir=str(args.weights))
    elif not args.weights.exists():
        print(f"--skip-download given but {args.weights} does not exist", file=sys.stderr)
        return 2

    import torch

    if not torch.cuda.is_available():
        print(
            "CUDA is not available in this container. TripoSG hard-codes device='cuda'.\n"
            "Run it through the rootless Podman + CDI wrapper, which hands it the GPU:\n"
            "  tools/anim-pipeline/spike3d/triposg.sh ...",
            file=sys.stderr,
        )
        return 2
    print(f"gpu: {torch.cuda.get_device_name(0)}")

    # Booked after the weights are on disk (a download never holds the GPU), released on exit.
    with book_gpu() as lease:
        if args.dry_run:
            dry_run_gpu(lease)
            return 0
        mesh = generate(
            args.image_input,
            args.weights,
            seed=args.seed,
            steps=args.num_inference_steps,
            guidance=args.guidance_scale,
            faces=args.faces,
            on_loaded=lease.mark_loaded,
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
