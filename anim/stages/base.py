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
`--yaw` / `--elevation` (the camera direction feature) view the rig from any
cell of `config.DIRECTIONS`; the job is then `<slug>_base_<direction>`, so the
default `<slug>_base` stays the s0 standing sprite every later stage looks for.

It then re-uses the same `post` and `publish` chain, with `contract.frames`
[1, 1] and the file named `<slug>.png` instead of `<slug>_<action>_<n>.png`.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Sequence

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
# The front base pose feature. `anim base` renders STAND_POSE through the
# ControlNet, so that one stays the side profile the art bible asks for. A
# concept base is not rendered from its skeleton at all - `concept --pick`
# copies a txt2img picture in and the skeleton is only ever read for the part
# prompts - and CONCEPT_TXT2IMG_PROMPT asks for a FRONT view, so pointing SAM
# at a side skeleton put every prompt in one narrow column (hooded_archer,
# 2026-09-24: arms/torso/legs all scored below the contamination gate and
# every direction then failed on the missing palettes).
STAND_POSE_FRONT = (Path(__file__).resolve().parent.parent
                    / "data" / "stand_pose_front.json")


def stand_payload(path: Path | None = None) -> dict[str, Any]:
    return read_json(path or STAND_POSE)


def _write_pose(job: Job, payload: dict[str, Any], cell: tuple[int, int]) -> Path:
    """Draw the skeleton, write keypoints.json, mirror it into strip/.

    The pose half of `prepare`, on its own so `repose` can redo it for a base
    job that already exists without disturbing the rest of its manifest.
    """
    entry = payload["frames"][0]
    canvas = posekit.draw(entry["keypoints"], entry["scores"],
                          tuple(payload["source_size"]), cell,
                          kpt_thr=float(payload.get("kpt_thr", 0.4)),
                          props=entry.get("props"))
    skeleton = job.pose_dir / "skel_0001.png"
    Image.fromarray(canvas[:, :, ::-1]).save(skeleton)
    write_json_in_place(job.keypoints_path, payload)
    return skeleton


def rig_payload(glb: Path, *, clip: str, frame: int, cell: tuple[int, int],
                fps: int | None = None, mirror: bool = False,
                props: Sequence[str] = (), depth: str | bool = "full",
                yaw: int = config.DEFAULT_YAW,
                elevation: str = config.DEFAULT_ELEVATION
                ) -> tuple[dict[str, Any], dict[str, Any], rigkit.DepthClip | None]:
    """One frame of a rig clip as a one-frame keypoints payload, its
    motion_source block, and its depth frame (None for a bare rig or
    `depth="none"`; a `rigkit.DEPTH_MODES` value). Same facing detection,
    per-rig scale, props and direction cell as `anim rig`, at the portrait
    cell."""
    glb = Path(glb)
    fps = fps if fps is not None else config.EXTRACT_FPS
    provenance = rig_stage.rig_provenance(glb)
    loaded = rigkit.load_rig(glb)
    jmap = rigkit.detect_map(loaded.names)
    ref = rigkit.reference_frame(loaded, jmap, mirror=mirror, props=props,
                                 yaw=yaw, elevation=elevation)
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
    mode = rigkit.depth_mode(depth)
    wants_depth = mode == "props" and bool(props) or mode == "full" and loaded.skinned
    depth_clip = rigkit.clip_depth(loaded, clip, fps=fps, cell=cell, frame=ref,
                                   times=[float(times[frame])], mode=mode) \
        if wants_depth else None
    source = rig_stage.motion_source(glb, provenance, clip_obj, payload, jmap,
                                     mirror, depth=depth_clip)
    source["frame"] = frame
    return payload, source, depth_clip


def prepare(slug: str, *, cell: tuple[int, int] | None = None,
            pose_path: Path | None = None, rig: Path | None = None,
            clip: str = "Idle", frame: int = 0, mirror: bool = False,
            props: Sequence[str] | None = None, depth: str | bool = "full",
            yaw: int = config.DEFAULT_YAW, elevation: str = config.DEFAULT_ELEVATION,
            work_dir: Path | None = None) -> Job:
    """Lay out work/<slug>_base/ so `post` and `publish` can be re-used verbatim.

    Writes the job manifest, renders the standing skeleton at the portrait cell,
    and fakes the one-cell picks/layout the later stages read. Doing it this way
    rather than special-casing `post` means the base sprite goes through exactly
    the same palette snap and verification as every animation frame.

    The skeleton is the canned one unless `rig` names a GLB, in which case it is
    `frame` of `clip` sampled from that rig, viewed from the `yaw` / `elevation`
    cell (the camera direction feature; only a rig can be turned, so the canned
    skeleton refuses any cell but the default). A non-default cell suffixes the
    job name: `<slug>_base_<direction>`.
    """
    cell = tuple(cell or config.BASE_CELL)
    if rig is None and not config.is_default_direction(yaw, elevation):
        raise ValueError(
            f"the canned standing skeleton is the {config.DEFAULT_DIRECTION} "
            f"profile only; a direction ({config.direction_name(yaw, elevation)}) "
            f"needs --rig")
    job = Job.open(rig_stage.default_job_name(f"{slug}_base", yaw, elevation), work_dir)
    job.pose_dir.mkdir(parents=True, exist_ok=True)
    job.strip_dir.mkdir(parents=True, exist_ok=True)

    source: dict[str, Any] | None = None
    depth_clip: rigkit.DepthClip | None = None
    if rig is not None:
        props = config.props_for_unit(slug) if props is None else tuple(props)
        payload, source, depth_clip = rig_payload(
            Path(rig), clip=clip, frame=frame, cell=cell, mirror=mirror,
            props=props, depth=depth, yaw=yaw, elevation=elevation)
    else:
        payload = stand_payload(pose_path)
    skeleton = _write_pose(job, payload, cell)
    if depth_clip is not None:
        rig_stage.write_depths(depth_clip, job.depth_dir)
    elif job.depth_dir.is_dir():
        for stale in job.depth_dir.glob("*.png"):
            stale.unlink()

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


def repose(slug: str, *, pose_path: Path | None = None,
           work_dir: Path | None = None) -> dict[str, Any]:
    """Re-lay the canned skeleton on a base job that already exists.

    The front base pose feature, for the units picked before it landed. Only
    the pose artifacts and the manifest's `pose` block are rewritten - the
    render, the concept block and the accepted views are left exactly as they
    are, which `prepare` cannot promise because it builds the manifest from
    scratch. Re-picking would do it too, but `concept --pick` clears the base's
    views on the way through, and a unit whose eight views are accepted should
    not have to earn them twice to get its part prompts fixed.
    """
    job = Job.open(f"{slug}_base", work_dir)
    if not job.manifest_path.exists():
        raise FileNotFoundError(
            f"{slug} has no base job; `anim concept --unit {slug} --pick NN` first")
    manifest = job.read_manifest()
    pose = dict(manifest.get("pose") or {})
    if pose.get("model") == "rig":
        raise ValueError(
            f"{job.name} was posed from a rig ({manifest.get('clip')}), not the "
            f"canned skeleton; re-run `anim base --unit {slug} --rig ...` to "
            f"change its pose")
    cell = tuple(pose.get("cell") or manifest.get("frame_size") or config.BASE_CELL)
    payload = stand_payload(pose_path or STAND_POSE_FRONT)
    job.pose_dir.mkdir(parents=True, exist_ok=True)
    job.strip_dir.mkdir(parents=True, exist_ok=True)
    skeleton = _write_pose(job, payload, cell)
    write_bytes_in_place(job.strip_dir / "skeleton_strip.png", skeleton.read_bytes())
    job.update_manifest(pose={"rig": payload["rig"], "model": "canned",
                              "mode": payload.get("mode", "canned"),
                              "cell": list(cell), "detected": 1})
    return {"job": job.name, "pose": str(pose_path or STAND_POSE_FRONT),
            "cell": list(cell), "keypoints": str(job.keypoints_path)}


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
         mirror: bool = False, props: Sequence[str] | None = None,
         parts: bool | None = None,
         yaw: int = config.DEFAULT_YAW, elevation: str = config.DEFAULT_ELEVATION,
         work_dir: Path | None = None) -> dict[str, Any]:
    job = prepare(slug, rig=rig, clip=clip, frame=frame, mirror=mirror,
                  props=props, yaw=yaw, elevation=elevation, work_dir=work_dir)
    render_base(job, prompt=prompt, seed=seed, strength=strength, client=client)
    # `parts` on a base job EXTRACTS the unit's part palettes (there is no
    # action to lock to); publish then carries them in the sprite's sidecar.
    post_stage.post(job.name, palette=palette, matte=matte,
                    use_pixelfixer=use_pixelfixer, parts=parts,
                    assets_dir=assets_dir, work_dir=work_dir)
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
    parser.add_argument("--repose", action="store_true",
                        help="re-lay the canned skeleton on an existing base "
                             "job and stop - no GPU. The front base pose "
                             "feature: fixes the part prompts of a unit picked "
                             "before the front pose landed, without clearing "
                             "its accepted views the way a re-pick would")
    parser.add_argument("--pose", dest="pose", default=None, metavar="FILE",
                        help="keypoints to lay down instead of the default "
                             f"({STAND_POSE_FRONT.name} for --repose, "
                             f"{STAND_POSE.name} otherwise)")
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
    parser.add_argument("--props", default=None,
                        help="props drawn onto the rig skeleton (see anim rig "
                             "--props); default: the unit's kit")
    rig_stage.add_direction_arguments(parser)
    parser.add_argument("--parts", dest="parts", action="store_true",
                        default=None,
                        help="extract the unit's part palettes from the "
                             "finished sprite (the part palette lock feature) "
                             "and publish them in its sidecar")
    parser.set_defaults(audit=True, pixelfixer=True)


def run_stage(args: argparse.Namespace) -> int:
    if args.repose:
        result = repose(args.slug,
                        pose_path=Path(args.pose) if args.pose else None)
        print(f"base: {Path(result['pose']).name} -> {result['keypoints']}")
        print(f"  {result['job']} re-posed at {result['cell'][0]}x"
              f"{result['cell'][1]}  (render and views untouched)")
        return 0
    rig_path = Path(args.rig) if args.rig else None
    rig_stage.check_direction("anim base", yaw=args.yaw, elevation=args.elevation,
                              mirror=args.mirror)
    if rig_path is None and not config.is_default_direction(args.yaw, args.elevation):
        raise SystemExit("anim base: --yaw/--elevation need --rig; the canned "
                         "standing skeleton is the s0 profile only")
    if args.skeleton_only:
        job = prepare(args.slug, rig=rig_path, clip=args.clip, frame=args.frame,
                      mirror=args.mirror, props=rig_stage.parse_props(args.props),
                      pose_path=Path(args.pose) if args.pose else None,
                      yaw=args.yaw, elevation=args.elevation)
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
                  frame=args.frame, mirror=args.mirror,
                  props=rig_stage.parse_props(args.props), parts=args.parts,
                  yaw=args.yaw, elevation=args.elevation)
    print(f"base: wrote {', '.join(result['frames'])} + {result['sidecar']} "
          f"to {result['dir']}")
    return 0


run = run_stage
