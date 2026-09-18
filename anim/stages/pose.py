"""Stage: `anim pose` - frames -> keypoints.json + skeleton PNGs

Run rtmlib (DWPose / RTMW) over onnxruntime on the CPU for every extracted
frame, then render each pose with rtmlib's OWN OpenPose renderer at the
generation cell size.

    anim pose knight_strike

Two decisions are not negotiable here:

**The renderer is always rtmlib's `draw_skeleton(..., openpose_skeleton=True)`.**
Z-Image's Fun ControlNet was trained on DWPose output and rejects hand-authored
rigs (gpu/workflows/NOTES.md, upstream discussion #28). We choose the canvas
size and the line width - which are geometry - and nothing else.

**Inference happens once per clip and is cached.** `pick`, `review` and `strip`
all re-read pose/keypoints.json; none of them may re-infer. On a CPU that is the
difference between a second and a minute every time the owner changes their mind
about which poses to keep.

The ONNX weights (~200 MB) download on first use into $XDG_CACHE_HOME/rtmlib,
which the Dockerfile points at the bind-mounted /work/.cache, so they are
fetched once per machine rather than once per container.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anim import config, posekit
from anim.job import Job
from anim.util import write_json_in_place

NAME = "pose"
HELP = "frames -> keypoints.json + skeleton PNGs"

MODELS = {"wholebody": "Wholebody", "body": "Body", "bodywithfeet": "BodyWithFeet"}


def _detector(model: str, mode: str):
    """Build the rtmlib estimator. Imported late so `--help` needs no onnxruntime."""
    import rtmlib

    cls_name = MODELS.get(model.lower())
    if cls_name is None:
        raise ValueError(f"unknown pose model {model!r}; known: {sorted(MODELS)}")
    cls = getattr(rtmlib, cls_name)
    # to_openpose=True is what makes the keypoints come out in the OpenPose
    # joint order the renderer and every ControlNet expect.
    return cls(mode=mode, to_openpose=True, backend="onnxruntime", device="cpu")


def best_person(keypoints: np.ndarray, scores: np.ndarray) -> int | None:
    """Index of the person to keep: the one with the most confident skeleton.

    The art bible's filming rules say one actor, but a clip filmed at a public
    place (or the CC0 fixture) will find bystanders. Summed confidence favours a
    fully-visible near subject over a half-occluded distant one, which is the
    right bias for "the person who filled the frame on purpose".
    """
    if len(keypoints) == 0:
        return None
    return int(np.argmax(scores.sum(axis=1)))


def estimate(frames: list[Path], *, model: str = None, mode: str = None,
             kpt_thr: float = None) -> dict[str, Any]:
    """Run inference over every frame and return a keypoints.json payload."""
    model = model or config.POSE_MODEL
    mode = mode or config.POSE_MODE
    kpt_thr = config.POSE_KPT_THR if kpt_thr is None else kpt_thr
    estimator = _detector(model, mode)

    entries: list[dict[str, Any]] = []
    rig = ""
    source_size = [0, 0]
    for index, path in enumerate(frames):
        image = np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1]  # BGR
        source_size = [image.shape[1], image.shape[0]]
        keypoints, scores = estimator(image)
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        who = best_person(keypoints, scores)
        if who is None:
            entries.append({"index": index, "file": path.name,
                            "keypoints": None, "scores": None, "person": None})
            continue
        rig = rig or posekit.rig_for(keypoints.shape[1])
        entries.append({
            "index": index,
            "file": path.name,
            "keypoints": [[round(float(x), 2), round(float(y), 2)]
                          for x, y in keypoints[who]],
            "scores": [round(float(s), 4) for s in scores[who]],
            "person": {"score": round(float(scores[who].mean()), 4)},
        })
    if not rig:
        raise RuntimeError(
            "no person detected in any frame. Check the art bible's filming "
            "rules: full body in frame, plain background, even light.")
    return {
        "schema": posekit.KEYPOINTS_SCHEMA,
        "rig": rig,
        "names": posekit.rig_names(rig),
        "model": model,
        "mode": mode,
        "kpt_thr": kpt_thr,
        "source_size": source_size,
        "frames": entries,
    }


def render_skeletons(payload: dict[str, Any], out_dir: Path,
                     cell: tuple[int, int] = None) -> list[Path]:
    """One skel_NNNN.png per frame that had a person, at the generation cell size."""
    cell = tuple(cell or config.STRIP_CELL)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("skel_*.png"):
        stale.unlink()
    source = tuple(payload["source_size"])
    written: list[Path] = []
    for entry in payload["frames"]:
        if not entry.get("keypoints"):
            continue
        canvas = posekit.draw(entry["keypoints"], entry["scores"], source, cell,
                              kpt_thr=float(payload.get("kpt_thr", 0.4)),
                              props=entry.get("props"))
        path = out_dir / f"skel_{entry['index'] + 1:04d}.png"
        Image.fromarray(canvas[:, :, ::-1]).save(path)  # BGR -> RGB
        written.append(path)
    return written


def pose(job_name: str, *, model: str = None, mode: str = None,
         cell: tuple[int, int] = None, work_dir: Path | None = None,
         force: bool = False) -> Job:
    job = Job.open(job_name, work_dir)
    frames = job.frames()
    if not frames:
        raise FileNotFoundError(
            f"no frames in {job.frames_dir} - run `anim extract` first")
    job.pose_dir.mkdir(parents=True, exist_ok=True)

    if job.keypoints_path.exists() and not force:
        from anim.util import read_json
        payload = read_json(job.keypoints_path)
        cached = len(payload.get("frames") or [])
        if cached != len(frames):
            raise RuntimeError(
                f"{job.keypoints_path} has {cached} frame(s) but "
                f"{len(frames)} are on disk - the clip changed. Re-run with "
                f"--force.")
    else:
        payload = estimate(frames, model=model, mode=mode)
        write_json_in_place(job.keypoints_path, payload)

    skeletons = render_skeletons(payload, job.pose_dir, cell)
    job.update_manifest(pose={"rig": payload["rig"], "model": payload["model"],
                              "mode": payload["mode"],
                              "cell": list(cell or config.STRIP_CELL),
                              "detected": len(skeletons)})
    return job


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", help="job name, e.g. knight_strike")
    parser.add_argument("--model", default=config.POSE_MODEL,
                        choices=sorted(MODELS),
                        help=f"rtmlib family (default {config.POSE_MODEL}; "
                             f"wholebody IS DWPose)")
    parser.add_argument("--mode", default=config.POSE_MODE,
                        choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--width", type=int, default=config.STRIP_CELL[0],
                        help="skeleton render width; must match the strip cell")
    parser.add_argument("--height", type=int, default=config.STRIP_CELL[1])
    parser.add_argument("--force", action="store_true",
                        help="re-infer even though keypoints.json exists")


def run_stage(args: argparse.Namespace) -> int:
    job = pose(args.job, model=args.model, mode=args.mode,
               cell=(args.width, args.height), force=args.force)
    payload = job.read_manifest()["pose"]
    print(f"pose: {payload['detected']} skeleton(s) at "
          f"{payload['cell'][0]}x{payload['cell'][1]} "
          f"({payload['model']}/{payload['rig']}) -> {job.pose_dir}")
    return 0


run = run_stage
