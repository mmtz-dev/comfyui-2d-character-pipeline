"""Stage: `anim run` - the whole chain, one command

    anim run capture/knight_strike.mp4 --unit knight --action strike

    extract -> pose -> pick -> strip      always (CPU only)
    [ GPU reachable? ]
    render -> post -> publish             only if the answer is yes

STOPS CLEANLY WHEN THE GPU IS OFF. The native Docker daemon is disabled by
default and only the owner starts it, so "ComfyUI is not running" is the normal
case, not an error - the CPU half of the chain is still worth having done, and
re-running `anim run` later picks up from the files it left behind. The stage
says exactly which two commands are missing and exits 0. `--require-gpu` turns
that into a failure, for a caller that wants to know.

`review` IS NOT IN THE CHAIN. It is a page the owner reads and clicks (served
by the pipeline API on :8720), not a step that finishes on its own. The URL is
printed instead, and re-running from `strip` after a click is one command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anim import comfy, config
from anim.stages import extract as extract_stage
from anim.stages import pick as pick_stage
from anim.stages import pose as pose_stage
from anim.stages import post as post_stage
from anim.stages import publish as publish_stage
from anim.stages import render as render_stage
from anim.stages import strip as strip_stage

NAME = "run"
HELP = "extract -> pose -> pick -> strip, then render -> post -> publish"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("clip", help="clip in capture/, or a path to one")
    parser.add_argument("--unit", "--slug", dest="slug", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--fps", type=int, default=config.EXTRACT_FPS)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("-n", type=int, default=None,
                        help="key poses (default: the art bible's budget)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--palette", default=None)
    parser.add_argument("--matte", default="rembg", choices=sorted(post_stage.MATTES))
    parser.add_argument("--reference", default=None)
    parser.add_argument("--job", default=None)
    parser.add_argument("--force", action="store_true",
                        help="re-pick even if picks.json was hand-edited")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail instead of stopping politely when ComfyUI "
                             "is unreachable")
    parser.add_argument("--no-audit", dest="audit", action="store_false")
    parser.add_argument("--no-pixelfixer", dest="pixelfixer", action="store_false")
    parser.set_defaults(audit=True, pixelfixer=True)


def run_stage(args: argparse.Namespace) -> int:
    job = extract_stage.extract(Path(args.clip), action=args.action,
                                slug=args.slug, fps=args.fps, crop=args.crop,
                                job_name=args.job)
    manifest = job.read_manifest()
    print(f"[1/7] extract: {manifest['frames']} frame(s) at {manifest['fps']} fps")

    pose_stage.pose(job.name)
    print(f"[2/7] pose: {job.read_manifest()['pose']['detected']} skeleton(s)")

    picks = pick_stage.pick(job.name, n=args.n, force=args.force)
    print(f"[3/7] pick: {picks['picks']}")

    layout = strip_stage.strip(
        job.name, reference=Path(args.reference) if args.reference else None)
    size = layout["strip"]["size"]
    print(f"[4/7] strip: {size[0]}x{size[1]}")
    print(f"      review these picks at http://{config.LAN_HOST}:"
          f"{config.API_PORT}/jobs/{job.name}/review  (anim review {job.name})")

    client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
    if not client.reachable():
        message = (
            f"\nSTOPPED BEFORE RENDER - the GPU image service is not answering "
            f"at {client.base_url}.\n"
            f"Everything up to the strip is done and on disk in {job.root}.\n\n"
            f"The owner starts it (needs a real TTY for the sudo password):\n"
            f"    sudo systemctl start docker\n"
            f"    docker --context docker-default compose -f "
            f"tools/anim-pipeline/gpu/compose.yml up -d\n\n"
            f"Then finish this job with:\n"
            f"    anim render {job.name} && anim post {job.name} && "
            f"anim publish {job.name}\n")
        if args.require_gpu:
            print(message)
            return 1
        print(message)
        return 0

    provenance = render_stage.render(job.name, seed=args.seed, client=client)
    print(f"[5/7] render: seed {provenance['seed']}")

    report = post_stage.post(job.name, palette=args.palette, matte=args.matte,
                             use_pixelfixer=args.pixelfixer)
    print(f"[6/7] post: {len(report['frames'])} frame(s)")

    result = publish_stage.publish(job.name, run_audit=args.audit)
    print(f"[7/7] publish: {', '.join(result['frames'])} + {result['sidecar']}")
    return 0


run = run_stage
