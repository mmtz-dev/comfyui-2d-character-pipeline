"""Stage: `anim base` - canned standing skeleton -> <slug>.png

A unit's standing sprite, for a unit that has no sprite yet.

    anim base --unit knight --prompt "a plate-armoured knight with a kite shield"

Every other stage re-skins a pose onto a reference sprite that already exists.
This one has nothing to take identity from, so it is the POSE stack alone -
Z-Image plus the Fun ControlNet, one non-square portrait cell (512x768), with
the art-bible description as the only style input. Once the result has been
nudged by hand it becomes `image1` for every action that unit ever gets, which
is what keeps sixteen units from drifting into sixteen art styles.

The standing skeleton is canned (`anim/data/stand_pose.json`): joint
coordinates, SYNTHESISED rather than traced, rendered here through rtmlib's own
`draw_skeleton` exactly as a filmed frame would be. No photograph, no licence,
no model download - and one pose that every unit shares, so the whole roster
stands the same way.

Or it comes from a rig (the rig motion input feature):

    anim base --unit knight --rig rigs/kaykit-adventurers/Knight.glb \
        --clip Idle --frame 0

which takes one sampled frame of a clip, projected at the portrait cell with
the same facing detection and per-rig scale `anim rig` uses, so the standing
sprite and every action strip of that unit come from one skeleton at one scale.
`job.json` then carries the rig as `motion_source`, exactly as `anim rig` does.

It then re-uses the same `post` and `publish` chain, with `contract.frames`
[1, 1] and the file named `<slug>.png` instead of `<slug>_<action>_<n>.png`.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from PIL import Image

from anim import comfy, config, models_lock, posekit, rigkit
from anim.job import Job, new_manifest
from anim.stages import post as post_stage
from anim.stages import publish as publish_stage
from anim.stages import render as render_stage
from anim.stages import rig as rig_stage
from anim.stages.pick import PICKS_SCHEMA
from anim.stages.strip import LAYOUT_SCHEMA
from anim.util import read_json, write_bytes_in_place, write_json_in_place

NAME = "base"
HELP = "canned standing skeleton -> <slug>.png"

STAND_POSE = Path(__file__).resolve().parent.parent / "data" / "stand_pose.json"


def stand_payload(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or STAND_POSE)


def rig_payload(glb: Path, *, clip: str, frame: int, cell: tuple[int, int],
                fps: int | None = None, mirror: bool = False
                ) -> tuple[dict[str, Any], dict[str, Any]]:
    """One frame of a rig clip as a one-frame keypoints payload, plus its
    motion_source block. Same facing detection and per-rig scale as `anim rig`,
    at the portrait cell."""
    glb = Path(glb)
    fps = fps if fps is not None else config.EXTRACT_FPS
    provenance = rig_stage.rig_provenance(glb)
    loaded = rigkit.load_rig(glb)
    jmap = rigkit.detect_map(loaded.names)
    ref = rigkit.reference_frame(loaded, jmap, mirror=mirror)
    clip_obj = loaded.clip(clip)
    times = rigkit.sample_times(clip_obj.duration, fps)
    if not 0 <= frame < len(times):
        raise ValueError(f"--frame {frame} is outside 0..{len(times) - 1} "
                         f"({clip} is {clip_obj.duration:.3f} s at {fps} fps)")
    payload = rigkit.clip_keypoints(loaded, clip, fps=fps, cell=cell, mirror=mirror,
                                    frame=ref, times=[float(times[frame])])
    payload["frames"][0]["file"] = "(rig)"
    payload["_note"] = (f"Frame {frame} of {clip} from {glb.name}, projected at "
                        f"the portrait cell by anim.rigkit - see anim rig.")
    source = rig_stage.motion_source(glb, provenance, clip_obj, payload, jmap, mirror)
    source["frame"] = frame
    return payload, source


def prepare(slug: str, *, cell: tuple[int, int] | None = None,
            pose_path: Path | None = None, rig: Path | None = None,
            clip: str = "Idle", frame: int = 0, mirror: bool = False,
            work_dir: Path | None = None) -> Job:
    """Lay out work/<slug>_base/ so `post` and `publish` can be re-used verbatim.

    Writes the job manifest, renders the standing skeleton at the portrait cell,
    and fakes the one-cell picks/layout the later stages read. Doing it this way
    rather than special-casing `post` means the base sprite goes through exactly
    the same palette snap and verification as every animation frame.

    The skeleton is the canned one unless `rig` names a GLB, in which case it is
    `frame` of `clip` sampled from that rig.
    """
    cell = tuple(cell or config.BASE_CELL)
    job = Job.open(f"{slug}_base", work_dir)
    job.pose_dir.mkdir(parents=True, exist_ok=True)
    job.strip_dir.mkdir(parents=True, exist_ok=True)

    source: dict[str, Any] | None = None
    if rig is not None:
        payload, source = rig_payload(Path(rig), clip=clip, frame=frame,
                                      cell=cell, mirror=mirror)
    else:
        payload = stand_payload(pose_path)
    entry = payload["frames"][0]
    canvas = posekit.draw(entry["keypoints"], entry["scores"],
                          tuple(payload["source_size"]), cell,
                          kpt_thr=float(payload.get("kpt_thr", 0.4)))
    skeleton = job.pose_dir / "skel_0001.png"
    Image.fromarray(canvas[:, :, ::-1]).save(skeleton)
    write_json_in_place(job.keypoints_path, payload)

    strip_path = job.strip_dir / "skeleton_strip.png"
    write_bytes_in_place(strip_path, skeleton.read_bytes())

    if source is not None:
        clip_label = f"{Path(rig).name}::{source['clip']}#{frame}"
        manifest = new_manifest(
            clip=clip_label, clip_sha256=source["file_sha256"], fps=0, frames=1,
            action="", slug=slug, frame_size=list(cell), crop=None,
            source_fps=None,
            created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        manifest["motion_source"] = source
        manifest["pose"] = {"rig": payload["rig"], "model": "rig",
                            "mode": payload["mode"], "cell": list(cell),
                            "detected": 1}
    else:
        manifest = new_manifest(
            clip="(canned standing skeleton)", clip_sha256="", fps=0, frames=1,
            action="", slug=slug, frame_size=list(cell), crop=None,
            source_fps=None,
            created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        manifest["pose"] = {"rig": payload["rig"], "model": "canned",
                            "mode": "canned", "cell": list(cell), "detected": 1}
    job.write_manifest(manifest)

    write_json_in_place(job.picks_path, {
        "schema": PICKS_SCHEMA, "n": 1, "action": "",
        "dedup": config.PICK_DEDUP, "picks": [0], "auto": [0], "signals": {}})
    write_json_in_place(job.layout_path, {
        "schema": LAYOUT_SCHEMA, "job": job.name,
        "strip": {"file": strip_path.name, "size": list(cell), "cells": 1,
                  "cell": list(cell)},
        "cells": [{"index": 0, "frame": 0, "source": skeleton.name,
                   "rect": [0, 0, cell[0], cell[1]]}],
        "reference": None,
    })
    return job


def render_base(job: Job, *, prompt: str | None = None, seed: int | None = None,
                strength: float | None = None,
                client: comfy.ComfyClient | None = None,
                workflow_dir: Path | None = None, lock_path: Path | None = None,
                timeout_s: float = 900.0) -> dict[str, Any]:
    """One `/prompt` job against base_sprite.json. No identity pass - see above."""
    manifest = job.read_manifest()
    layout = read_json(job.layout_path)
    width, height = layout["strip"]["size"]
    seed = (render_stage.derive_seed(manifest, [0]) if seed is None else int(seed))

    client = client or comfy.ComfyClient()
    try:
        stats = client.system_stats()
    except comfy.ComfyError as exc:
        raise comfy.ComfyUnreachable(
            f"the GPU image service is not answering at {client.base_url}: "
            f"{exc}") from exc

    job.render_dir.mkdir(parents=True, exist_ok=True)
    strip_path = job.strip_dir / layout["strip"]["file"]

    workflow = comfy.load_workflow("base_sprite", workflow_dir)
    comfy.patch(workflow, "standing skeleton",
                image=client.upload_image(strip_path))
    comfy.patch(workflow, "cell size", width=width, height=height)
    comfy.patch(workflow, "seed", seed=seed)
    comfy.patch(workflow, "output prefix",
                filename_prefix=f"anim_pipeline/{job.name}/base_sprite")
    if strength is not None:
        comfy.patch(workflow, "control scale", strength=float(strength))
    if prompt:
        comfy.patch(workflow, "positive prompt", text=prompt)

    png, record = render_stage.submit(client, workflow, timeout_s=timeout_s)
    out = write_bytes_in_place(job.render_dir / "base_sprite.png", png)
    record |= {"name": "base_sprite", "workflow": "base_sprite.json",
               "seed": seed, "file": out.name,
               "models": [m.as_sidecar_entry()
                          for m in models_lock.for_workflow(workflow, lock_path)]}
    provenance = render_stage.build_provenance(client.base_url, stats, seed,
                                               [record])
    write_json_in_place(job.provenance_path, provenance)
    job.update_manifest(render={"seed": seed, "passes": ["base_sprite"],
                                "final": out.name})
    return provenance


def base(slug: str, *, prompt: str | None = None, seed: int | None = None,
         strength: float | None = None, palette: Path | None = None,
         matte: str = "rembg", client: comfy.ComfyClient | None = None,
         assets_dir: Path | None = None, run_audit: bool = True,
         audit_script: Path | None = None, use_pixelfixer: bool = True,
         rig: Path | None = None, clip: str = "Idle", frame: int = 0,
         mirror: bool = False, work_dir: Path | None = None) -> dict[str, Any]:
    job = prepare(slug, rig=rig, clip=clip, frame=frame, mirror=mirror,
                  work_dir=work_dir)
    render_base(job, prompt=prompt, seed=seed, strength=strength, client=client)
    post_stage.post(job.name, palette=palette, matte=matte,
                    use_pixelfixer=use_pixelfixer, work_dir=work_dir)
    return publish_stage.publish(job.name, assets_dir=assets_dir,
                                 run_audit=run_audit, audit_script=audit_script,
                                 work_dir=work_dir)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unit", "--slug", dest="slug", required=True,
                        help="unit slug, e.g. knight")
    parser.add_argument("--prompt", default=None,
                        help="the unit's description from the art bible")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strength", type=float, default=None,
                        help="ControlNet control scale (workflow default 0.7)")
    parser.add_argument("--palette", default=None)
    parser.add_argument("--matte", default="rembg", choices=sorted(post_stage.MATTES))
    parser.add_argument("--no-audit", dest="audit", action="store_false")
    parser.add_argument("--no-pixelfixer", dest="pixelfixer", action="store_false")
    parser.add_argument("--skeleton-only", action="store_true",
                        help="render the standing skeleton and stop - no GPU, "
                             "useful for eyeballing the rig")
    parser.add_argument("--rig", default=None, metavar="GLB",
                        help="take the standing pose from a rig clip instead of "
                             "the canned skeleton (the rig motion input feature)")
    parser.add_argument("--clip", default="Idle",
                        help="clip in the rig to take the pose from (default Idle)")
    parser.add_argument("--frame", type=int, default=0,
                        help="sampled frame of that clip, at the extract fps "
                             "(default 0)")
    parser.add_argument("--mirror", action="store_true",
                        help="flip the rig's detected facing")
    parser.set_defaults(audit=True, pixelfixer=True)


def run_stage(args: argparse.Namespace) -> int:
    rig_path = Path(args.rig) if args.rig else None
    if args.skeleton_only:
        job = prepare(args.slug, rig=rig_path, clip=args.clip, frame=args.frame,
                      mirror=args.mirror)
        what = (f"{job.read_manifest()['clip']}" if rig_path
                else "canned standing skeleton")
        print(f"base: {what} -> {job.strip_dir / 'skeleton_strip.png'}")
        print(f"  job.json -> {job.manifest_path}  (stopped before render)")
        return 0
    client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
    result = base(args.slug, prompt=args.prompt, seed=args.seed,
                  strength=args.strength, palette=args.palette, matte=args.matte,
                  client=client, run_audit=args.audit,
                  use_pixelfixer=args.pixelfixer, rig=rig_path, clip=args.clip,
                  frame=args.frame, mirror=args.mirror)
    print(f"base: wrote {', '.join(result['frames'])} + {result['sidecar']} "
          f"to {result['dir']}")
    return 0


run = run_stage
