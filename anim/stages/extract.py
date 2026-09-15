"""Stage: `anim extract` - clip -> frames

ffmpeg-decode the filmed clip to PNG frames at a capped fps, honour the phone's
rotation metadata, optionally crop to the person, and write job.json.

    anim extract capture/knight_strike.mp4 --unit knight --action strike

The cap matters: a phone films at 30 or 60 fps and a key-pose picker does not
get better answers from four near-identical frames per hundredth of a second.
12 is both plenty and the rate the game plays frames back at
(render/unit_sprites.gd DEFAULT_FPS). The cap is a cap, not a target - a 10 fps
clip stays at 10 rather than being frame-doubled up to 12.

Rotation is honoured by NOT disabling it: ffmpeg applies a video stream's
display matrix by default, and a phone clip is almost always stored landscape
with a 90 degree matrix. `-noautorotate` would produce sideways frames, so it is
never passed, and the rotation ffprobe reported is recorded in job.json so a
sideways result is diagnosable instead of mysterious.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anim import config
from anim.job import Job, new_manifest
from anim.util import run as run_tool
from anim.util import sha256_file

NAME = "extract"
HELP = "clip -> frames"

# A pixel this far from the background estimate is the subject. Generous
# because the art bible's filming rules promise a plain background, and a
# too-tight threshold silently crops a limb off.
CROP_DIFF_THRESHOLD = 24
# Grown by this fraction of the subject's own size on every side. The pose model
# wants the whole body plus air; a box tight to the silhouette costs ankles.
CROP_MARGIN = 0.12
# Below this the "subject" is noise and cropping is refused.
CROP_MIN_AREA = 0.01


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------
def probe(clip: Path) -> dict[str, Any]:
    """Width, height, rotation, fps and duration of a clip's first video stream."""
    proc = run_tool(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames:stream_side_data=rotation",
                "-show_entries", "format=duration",
                "-of", "json", str(clip)])
    payload = json.loads(proc.stdout or "{}")
    streams = payload.get("streams") or [{}]
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or "0/0"
    try:
        num, _, den = rate.partition("/")
        fps = float(num) / float(den) if float(den or 0) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    rotation = 0
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "rotation": rotation,
        "duration": float((payload.get("format") or {}).get("duration") or 0.0),
    }


def effective_fps(source_fps: float, cap: int) -> int:
    """The decode rate: the cap, or the source rate when it is already slower."""
    if source_fps <= 0:
        return cap
    return max(1, min(cap, int(round(source_fps))))


def decode(clip: Path, frames_dir: Path, fps: int) -> list[Path]:
    """ffmpeg -> frames_dir/%04d.png. Clears the directory first.

    Clearing is not tidiness: re-running `extract` on a shorter clip would
    otherwise leave the tail of the previous run's frames behind, and every
    later stage counts files.
    """
    frames_dir = Path(frames_dir)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    run_tool(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(clip),
         "-vf", f"fps={fps}",
         str(frames_dir / "%04d.png")])
    return sorted(frames_dir.glob("*.png"))


def subject_box(frames: list[Path], *, threshold: int = CROP_DIFF_THRESHOLD,
                margin: float = CROP_MARGIN) -> tuple[int, int, int, int] | None:
    """A single crop box containing the person in every frame, or None.

    ONE box for the whole clip, not one per frame: a per-frame crop would make
    the subject jitter against the frame edge, and the pose normaliser would
    then see motion that is really the camera. The box is the union of what
    moved plus what differs from the background, grown by a margin.

    The background estimate is the per-pixel median across frames, which is
    exactly right for the art bible's filming rules (locked camera, plain wall,
    one actor): whatever is there most of the time is the wall.

    Returns None when nothing convincing was found, and the caller keeps the
    full frame - a wrong crop is far more expensive than no crop.
    """
    if len(frames) < 2:
        return None
    stack = np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.int16)
                      for p in frames])
    background = np.median(stack, axis=0)
    mask = (np.abs(stack - background) > threshold).any(axis=0)
    height, width = mask.shape
    if mask.sum() < CROP_MIN_AREA * height * width:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    pad_x = int((right - left) * margin)
    pad_y = int((bottom - top) * margin)
    return (max(0, left - pad_x), max(0, top - pad_y),
            min(width, right + pad_x), min(height, bottom + pad_y))


def crop_frames(frames: list[Path], box: tuple[int, int, int, int]) -> None:
    """Crop every frame to the same box, in place."""
    for path in frames:
        with Image.open(path) as img:
            img.convert("RGB").crop(box).save(path)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def extract(clip: Path, *, action: str, slug: str, fps: int | None = None,
            crop: bool = False, job_name: str | None = None,
            work_dir: Path | None = None) -> Job:
    clip = Path(clip)
    if not clip.exists():
        # A bare name is resolved against capture/ so the owner can type the
        # filename they dropped in the watched folder.
        candidate = config.CAPTURE_DIR / clip.name
        if candidate.exists():
            clip = candidate
        else:
            raise FileNotFoundError(f"no clip at {clip} (or {candidate})")

    job = Job.open(job_name or f"{slug}_{action}", work_dir)
    job.root.mkdir(parents=True, exist_ok=True)

    info = probe(clip)
    rate = effective_fps(info["fps"], fps if fps is not None else config.EXTRACT_FPS)
    frames = decode(clip, job.frames_dir, rate)
    if not frames:
        raise RuntimeError(f"ffmpeg produced no frames from {clip}")

    box = None
    if crop:
        box = subject_box(frames)
        if box is not None:
            crop_frames(frames, box)

    with Image.open(frames[0]) as first:
        size = [first.width, first.height]

    manifest = new_manifest(
        clip=str(clip), clip_sha256=sha256_file(clip), fps=rate,
        frames=len(frames), action=action, slug=slug, frame_size=size,
        crop=list(box) if box else None, source_fps=info["fps"] or None,
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    manifest["source_rotation"] = info["rotation"]
    manifest["source_size"] = [info["width"], info["height"]]
    job.write_manifest(manifest)
    return job


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("clip", help="clip in capture/, or a path to one")
    parser.add_argument("--unit", "--slug", dest="slug", required=True,
                        help="unit slug, e.g. knight")
    parser.add_argument("--action", required=True,
                        help="action name from the art bible's vocabulary, "
                             "e.g. strike")
    parser.add_argument("--fps", type=int, default=config.EXTRACT_FPS,
                        help=f"decode rate CAP (default {config.EXTRACT_FPS}); "
                             f"a slower clip is not sped up")
    parser.add_argument("--crop", action="store_true",
                        help="crop every frame to one person-centred box "
                             "(off by default: a wrong crop costs ankles)")
    parser.add_argument("--job", default=None,
                        help="job name (default <slug>_<action>)")


def run_stage(args: argparse.Namespace) -> int:
    job = extract(Path(args.clip), action=args.action, slug=args.slug,
                  fps=args.fps, crop=args.crop, job_name=args.job)
    manifest = job.read_manifest()
    print(f"extract: {manifest['frames']} frame(s) at {manifest['fps']} fps "
          f"-> {job.frames_dir}")
    if manifest.get("crop"):
        print(f"  cropped to {manifest['crop']}")
    if manifest.get("source_rotation"):
        print(f"  source rotation {manifest['source_rotation']} deg applied")
    print(f"  job.json -> {job.manifest_path}")
    return 0


run = run_stage
