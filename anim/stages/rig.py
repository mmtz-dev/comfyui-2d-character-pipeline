"""Stage: `anim rig` - a rig clip -> frames + keypoints, in place of extract+pose

The rig motion input feature. A rigged, animated glTF character (the CC0 KayKit
Knight, first) is the motion source instead of a filmed clip:

    anim rig rigs/kaykit-adventurers/Knight.glb --list
    anim rig rigs/kaykit-adventurers/Knight.glb --clip Hit_A \\
        --unit knight --action hit [--fps 12] [--job knight_hit] [--mirror] \\
        [--props sword,shield | --props none] [--depth full|props|none] \\
        [--yaw 0|45|..|315] [--elevation side|top]

It writes exactly what `anim extract` followed by `anim pose` would have:

    work/<job>/job.json              + motion_source {kind: rig, pack, file, clip, ...}
    work/<job>/frames/%04d.png       a dark-ground preview of each sampled pose
    work/<job>/pose/keypoints.json   the same anim.keypoints/1 schema `pose` writes
    work/<job>/pose/skel_%04d.png    drawn by the SAME renderer `pose` uses

plus one thing a filmed clip never has (the rig props feature):

    work/<job>/depth/%04d.png        8-bit grayscale depth of the posed mesh +
                                     props, same index and cell as skel_%04d.png;
                                     near = 255, far = DEPTH_FAR_GREY, background 0;
                                     one range per clip, in job.json.motion_source
                                     .depth_range = [far, near]; `--depth props`
                                     draws the prop solids only (motion_source
                                     .depth_mode). See anim.rigkit (DEPTH).
                                     `anim clip` reads it as a control channel;
                                     every other stage ignores it.

so `anim pick`, `anim review`, `anim strip` and everything after them run
unchanged and never learn where the motion came from - except through the
manifest, which `publish` copies into the sidecar so the audit can see that the
MOTION is a CC0 rig clip while the PIXELS are a local generation.

THE SKELETON PNGs ARE NOT DRAWN HERE. `anim.stages.pose.render_skeletons`
draws them, through `anim.posekit.draw`, through rtmlib's own `draw_skeleton`.
One renderer for filmed and rigged motion, or the ControlNet sees two styles
and the anchor-gate comparison measures the renderer instead of the pose. The
frames/ previews are a different thing: a picture for the review page, drawn
with Pillow, deliberately not in DWPose's colours so nobody feeds one to the
GPU by mistake.

PROPS (the skeleton props feature). The unit's kit - `config.UNIT_PROPS`, or
`--props` - is computed in 3D on the rig's hand and forearm bones and written
into keypoints.json per frame, so the skeleton PNGs carry a blade and a shield
outline and the ControlNet is told which hand is which (anim.rigkit, PROPS).
The manifest's `motion_source.props` records what was drawn.

DIRECTION (the camera direction feature). `--yaw` and `--elevation` pick one
of the sixteen cells in `config.DIRECTIONS`; the rig is rotated into that view
before the projection (anim.rigkit, DIRECTION), so the skeletons, keypoints and
depth frames of every cell come from the same code. The default cell `s0` is
the profile every job had before, byte for byte, and keeps its job name; any
other cell is written into `motion_source` (`direction`, `yaw`, `elevation`)
and suffixed onto the default job name - `knight_hit_t270` - so the sixteen
views of one action never overwrite each other. `--mirror` is the flipped s0
profile and is refused with any other cell: `--yaw 180` is that picture.

PROVENANCE. A rig directory carries `rig.json` (pack, author, url, licence,
per-file sha256). The stage refuses a rig without one, and refuses one whose
recorded hash does not match the file: motion is an input like any other, and
an input without a licence on record cannot reach `assets/`.

Geometry (facing detection, per-rig scale, the joint map) is `anim.rigkit`.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from anim import config, rigkit
from anim.job import Job, new_manifest
from anim.stages import pose as pose_stage
from anim.util import read_json, sha256_file, write_json_in_place

NAME = "rig"
HELP = "rig clip -> frames + keypoints (replaces extract + pose)"

RIG_SIDECAR = "rig.json"
PREVIEW_BG = (21, 23, 28)
PREVIEW_FLOOR = (52, 56, 66)
PREVIEW_LIMB = (167, 180, 200)
PREVIEW_JOINT = (240, 192, 112)
PREVIEW_FACE = (120, 170, 220)
PREVIEW_PROP = (214, 214, 214)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def rig_provenance(glb: Path) -> dict[str, Any]:
    """The rig.json next to the GLB, checked against the file's sha256."""
    glb = Path(glb)
    sidecar = glb.parent / RIG_SIDECAR
    if not sidecar.exists():
        raise FileNotFoundError(
            f"no {RIG_SIDECAR} next to {glb}. Every rig directory records its "
            f"pack, author, url, licence and per-file sha256 there (see "
            f"rigs/README.md); a rig without one is an input without a licence.")
    record = read_json(sidecar)
    digest = sha256_file(glb)
    expected = (record.get("files") or {}).get(glb.name)
    if expected is None:
        raise ValueError(f"{sidecar} does not list {glb.name}; add it with its "
                         f"sha256 ({digest})")
    if expected != digest:
        raise ValueError(f"{glb.name} hashes to {digest} but {sidecar} records "
                         f"{expected} - the file changed; re-verify its licence "
                         f"and update the sidecar")
    for key in ("pack", "license", "url"):
        if not record.get(key):
            raise ValueError(f"{sidecar} is missing {key!r}")
    return {"record": record, "sha256": digest, "sidecar": sidecar}


def motion_source(glb: Path, provenance: dict[str, Any], clip: rigkit.Clip,
                  payload: dict[str, Any], jmap: rigkit.JointMap,
                  mirror: bool, depth: rigkit.DepthClip | None = None
                  ) -> dict[str, Any]:
    record = provenance["record"]
    motion = payload["motion"]
    depth_block: dict[str, Any] = {"depth": depth is not None,
                                   "depth_mode": depth.mode if depth is not None else "none"}
    if depth is not None:
        depth_block["depth_range"] = [round(depth.far, 4), round(depth.near, 4)]
        depth_block["depth_far_grey"] = rigkit.DEPTH_FAR_GREY
    return {
        "kind": "rig",
        "pack": record["pack"],
        "author": record.get("author"),
        "file": str(glb),
        "file_sha256": provenance["sha256"],
        "clip": clip.name,
        "clip_duration": round(clip.duration, 4),
        "license": record["license"],
        "url": record["url"],
        "rig_map": jmap.name,
        "fps": motion["fps"],
        "mirror": bool(mirror),
        "direction": motion["direction"],
        "yaw": motion["yaw"],
        "elevation": motion["elevation"],
        "forward": motion["forward"],
        "height": motion["height"],
        "scale_px_per_unit": motion["scale_px_per_unit"],
        "props": list(motion.get("props") or []),
        **depth_block,
    }


# ---------------------------------------------------------------------------
# The depth frames (the rig props feature)
# ---------------------------------------------------------------------------
def write_depths(depth: rigkit.DepthClip, depth_dir: Path,
                 files: Sequence[str] | None = None) -> list[Path]:
    """One depth/%04d.png per frame, grey-mapped over the clip's range.
    Stale PNGs are cleared first, like frames/ and pose/."""
    depth_dir = Path(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)
    for stale in depth_dir.glob("*.png"):
        stale.unlink()
    written: list[Path] = []
    for index, grey in enumerate(depth.frames()):
        name = files[index] if files else f"{index + 1:04d}.png"
        path = depth_dir / name
        Image.fromarray(np.asarray(grey, dtype=np.uint8), mode="L").save(path)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# The preview frames
# ---------------------------------------------------------------------------
def draw_preview(keypoints: list[list[float]], names: list[str],
                 cell: tuple[int, int], baseline: float | None = None,
                 props: list[dict[str, Any]] | None = None) -> Image.Image:
    """A stick figure on a dark ground for the review page. Not for the GPU."""
    img = Image.new("RGB", tuple(cell), PREVIEW_BG)
    draw = ImageDraw.Draw(img)
    if baseline is not None:
        draw.line([(0, baseline), (cell[0], baseline)], fill=PREVIEW_FLOOR, width=1)
    index = {n: i for i, n in enumerate(names)}
    width = max(2, round(min(cell) * 0.012))
    for prop in props or ():
        for polygon in prop.get("polygons") or ():
            draw.polygon([tuple(pt) for pt in polygon], outline=PREVIEW_PROP, width=1)
    for a, b in rigkit.LIMBS:
        if a in index and b in index:
            colour = PREVIEW_FACE if a in rigkit.FACE_JOINTS or b in rigkit.FACE_JOINTS \
                else PREVIEW_LIMB
            draw.line([tuple(keypoints[index[a]]), tuple(keypoints[index[b]])],
                      fill=colour, width=width)
    r = max(2, width)
    for name, (x, y) in zip(names, keypoints):
        colour = PREVIEW_FACE if name in rigkit.FACE_JOINTS else PREVIEW_JOINT
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)
    return img


def write_previews(payload: dict[str, Any], frames_dir: Path) -> list[Path]:
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()
    cell = tuple(payload["source_size"])
    motion = payload.get("motion") or {}
    # A top-down cell has no floor row to draw (the camera direction feature's pivot
    # rule, anim.rigkit DIRECTION): the line would cut through the figure.
    baseline = motion.get("baseline_row") if motion.get("elevation", "side") != "top" \
        else None
    written: list[Path] = []
    for entry in payload["frames"]:
        img = draw_preview(entry["keypoints"], payload["names"], cell, baseline,
                           props=entry.get("props"))
        path = frames_dir / entry["file"]
        img.save(path)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def default_job_name(stem: str, yaw: int = config.DEFAULT_YAW,
                     elevation: str = config.DEFAULT_ELEVATION) -> str:
    """`<stem>` in the default cell, `<stem>_<direction>` in any other - so
    every job name from before the camera direction feature is unchanged."""
    direction = config.direction_name(yaw, elevation)
    return stem if direction == config.DEFAULT_DIRECTION else f"{stem}_{direction}"


def rig(glb: Path, *, clip: str, action: str, slug: str, fps: int | None = None,
        mirror: bool = False, job_name: str | None = None,
        cell: tuple[int, int] | None = None,
        props: Sequence[str] | None = None, depth: str | bool = "full",
        yaw: int = config.DEFAULT_YAW, elevation: str = config.DEFAULT_ELEVATION,
        times: Sequence[float] | None = None,
        work_dir: Path | None = None) -> Job:
    """Sample one clip into work/<job>/ as if extract + pose had run.

    `props` None means the unit's kit from `config.UNIT_PROPS`; pass () for none.
    `depth` is a `rigkit.DEPTH_MODES` value (or the older bool): `full` writes
    depth/%04d.png of body + props when the rig has a skinned mesh, `props`
    the prop solids alone, `none` nothing. `yaw` / `elevation` pick the cell
    of `config.DIRECTIONS` (the camera direction feature). `times` (seconds
    into the clip) replaces the even `fps` sampling with explicit sample
    instants - the frame route bake-off feature's rig-space tween samples a
    clip at exactly its frame budget; the CLI has no flag for it. `fps` is
    still recorded as the playback rate.
    """
    glb = Path(glb)
    fps = fps if fps is not None else config.EXTRACT_FPS
    cell = tuple(cell or config.STRIP_CELL)
    props = config.props_for_unit(slug) if props is None else tuple(props)
    job_name = job_name or default_job_name(f"{slug}_{action}", yaw, elevation)
    provenance = rig_provenance(glb)

    loaded = rigkit.load_rig(glb)
    jmap = rigkit.detect_map(loaded.names)
    frame = rigkit.reference_frame(loaded, jmap, mirror=mirror, props=props,
                                   yaw=yaw, elevation=elevation)
    sample = [float(t) for t in times] if times is not None else None
    payload = rigkit.clip_keypoints(loaded, clip, fps=fps, cell=cell,
                                    mirror=mirror, frame=frame, times=sample)
    clip_obj = loaded.clip(clip)
    mode = rigkit.depth_mode(depth)
    wants_depth = mode == "props" and bool(props) or mode == "full" and loaded.skinned
    depth_clip = rigkit.clip_depth(loaded, clip, fps=fps, cell=cell, frame=frame,
                                   times=sample, mode=mode) if wants_depth else None

    job = Job.open(job_name, work_dir)
    job.root.mkdir(parents=True, exist_ok=True)
    write_previews(payload, job.frames_dir)
    job.pose_dir.mkdir(parents=True, exist_ok=True)
    write_json_in_place(job.keypoints_path, payload)
    skeletons = pose_stage.render_skeletons(payload, job.pose_dir, cell)
    if depth_clip is not None:
        write_depths(depth_clip, job.depth_dir)
    elif job.depth_dir.is_dir():
        for stale in job.depth_dir.glob("*.png"):
            stale.unlink()

    manifest = new_manifest(
        clip=f"{glb.name}::{clip_obj.name}", clip_sha256=provenance["sha256"],
        fps=fps, frames=len(payload["frames"]), action=action, slug=slug,
        frame_size=list(cell), crop=None, source_fps=None,
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    manifest["motion_source"] = motion_source(glb, provenance, clip_obj, payload,
                                              jmap, mirror, depth=depth_clip)
    manifest["pose"] = {"rig": payload["rig"], "model": "rig",
                        "mode": jmap.name, "cell": list(cell),
                        "detected": len(skeletons)}
    job.write_manifest(manifest)
    return job


def list_clips(glb: Path) -> list[dict[str, Any]]:
    return rigkit.list_clips(rigkit.load_rig(Path(glb)))


def parse_props(text: str | None) -> tuple[str, ...] | None:
    """`--props`: None (unit default), 'none' -> (), else the listed kinds."""
    if text is None:
        return None
    kinds = tuple(k.strip() for k in text.split(",") if k.strip())
    if kinds == ("none",):
        return ()
    unknown = [k for k in kinds if k not in rigkit.PROP_KINDS]
    if unknown:
        raise SystemExit(f"anim rig: unknown prop(s) {unknown}; known: "
                         f"{list(rigkit.PROP_KINDS)} or none")
    return kinds


def check_direction(prog: str, *, yaw: int, elevation: str, mirror: bool) -> str:
    """The direction name for a stage's flags, or SystemExit for a pair the
    table refuses or for `--mirror` on any cell but the default."""
    try:
        name = config.direction_name(yaw, elevation)
    except ValueError as exc:
        raise SystemExit(f"{prog}: {exc}") from None
    if mirror and name != config.DEFAULT_DIRECTION:
        raise SystemExit(
            f"{prog}: --mirror cannot be combined with --yaw/--elevation "
            f"({name}); --yaw 180 is the flipped profile, and a direction "
            f"already says which way the figure faces")
    return name


def add_direction_arguments(parser: argparse.ArgumentParser) -> None:
    """`--yaw` / `--elevation`, shared by `rig` and `base`."""
    parser.add_argument("--yaw", type=int, choices=config.DIRECTION_YAWS,
                        default=config.DEFAULT_YAW, metavar="DEG",
                        help="the figure's turn about its up axis, one of "
                             f"{list(config.DIRECTION_YAWS)}: 0 faces right "
                             "(the default profile), 90 away, 180 left, 270 "
                             "the viewer (the camera direction feature)")
    parser.add_argument("--elevation", choices=config.DIRECTION_ELEVATIONS,
                        default=config.DEFAULT_ELEVATION,
                        help="'side' (default) or 'top', straight down")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("glb", help="rig file, e.g. rigs/kaykit-adventurers/Knight.glb")
    parser.add_argument("--list", action="store_true",
                        help="print the clip names and durations and stop")
    parser.add_argument("--clip", default=None,
                        help="clip name inside the rig, e.g. Hit_A")
    parser.add_argument("--unit", "--slug", dest="slug", default=None,
                        help="unit slug, e.g. knight")
    parser.add_argument("--action", default=None,
                        help="action name from the art bible's vocabulary")
    parser.add_argument("--fps", type=int, default=config.EXTRACT_FPS,
                        help=f"sampling rate (default {config.EXTRACT_FPS})")
    parser.add_argument("--mirror", action="store_true",
                        help="flip the detected facing (the default puts the "
                             "rig's forward on +x: the party faces right)")
    parser.add_argument("--job", default=None,
                        help="job name (default <slug>_<action>, plus _<direction> "
                             "for any cell but s0)")
    add_direction_arguments(parser)
    parser.add_argument("--props", default=None,
                        help="props drawn onto the skeleton, comma-separated "
                             f"from {list(rigkit.PROP_KINDS)}, or 'none' "
                             "(default: the unit's kit in config.UNIT_PROPS)")
    parser.add_argument("--depth", choices=rigkit.DEPTH_MODES, default="full",
                        help="depth/%%04d.png (the rig props feature): 'full' = "
                             "the posed mesh + props (default), 'props' = the "
                             "prop solids only, 'none' = skip it")
    parser.add_argument("--no-depth", dest="depth", action="store_const", const="none",
                        help="alias of --depth none")
    parser.add_argument("--width", type=int, default=config.STRIP_CELL[0],
                        help="skeleton cell width; must match the strip cell")
    parser.add_argument("--height", type=int, default=config.STRIP_CELL[1])


def run_stage(args: argparse.Namespace) -> int:
    if args.list:
        clips = list_clips(Path(args.glb))
        print(f"{len(clips)} clip(s) in {args.glb}:")
        for c in clips:
            print(f"  {c['name']:40s} {c['duration']:6.3f} s  "
                  f"{c['keyframes']:3d} keys  {c['nodes']:2d} nodes")
        return 0
    missing = [flag for flag, value in (("--clip", args.clip),
                                        ("--unit", args.slug),
                                        ("--action", args.action)) if not value]
    if missing:
        raise SystemExit(f"anim rig: {', '.join(missing)} required (or --list)")
    check_direction("anim rig", yaw=args.yaw, elevation=args.elevation,
                    mirror=args.mirror)
    job = rig(Path(args.glb), clip=args.clip, action=args.action, slug=args.slug,
              fps=args.fps, mirror=args.mirror, job_name=args.job,
              props=parse_props(args.props), depth=args.depth,
              yaw=args.yaw, elevation=args.elevation,
              cell=(args.width, args.height))
    manifest = job.read_manifest()
    src = manifest["motion_source"]
    print(f"rig: {manifest['frames']} frame(s) at {manifest['fps']} fps from "
          f"{src['pack']} / {src['clip']} ({src['clip_duration']} s) -> {job.root}")
    print(f"  facing {src['forward']}{' (mirrored)' if src['mirror'] else ''}, "
          f"direction {src['direction']} (yaw {src['yaw']}, {src['elevation']}), "
          f"scale {src['scale_px_per_unit']} px/unit, map {src['rig_map']}")
    print(f"  {manifest['pose']['detected']} skeleton(s) at "
          f"{manifest['pose']['cell'][0]}x{manifest['pose']['cell'][1]} -> "
          f"{job.pose_dir}")
    if src.get("depth"):
        print(f"  {len(job.depths())} depth frame(s) ({src['depth_mode']}), range "
              f"{src['depth_range']} -> {job.depth_dir}")
    print(f"  next: anim pick {job.name} && anim strip {job.name}")
    return 0


run = run_stage
