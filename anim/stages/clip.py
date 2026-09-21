"""Stage: `anim clip` - every rig frame -> one generated clip (a GPU stage)

The clip render feature. Instead of generating 2-6 picked poses as one 4x1
strip with a per-frame image model, render the WHOLE clip as a short video
conditioned on the unit's reference image, then let `pick` and `post` take the
key poses out of frames that already share one identity:

    anim rig rigs/quaternius/UAL1_Standard.glb --clip Sword_Attack \\
        --unit knight --action strike --job knightq_strike_clip \\
        --width 512 --height 768
    anim clip knightq_strike_clip [--seed N] [--control auto|pose|depth|blend]
    anim pick knightq_strike_clip
    anim post knightq_strike_clip --input clip
    anim publish knightq_strike_clip

    clip/control/0001.png ...  the control frames as sent (skeleton, depth, or both)
    clip/control.mp4           the same frames as the video LoadVideo reads
    clip/frames/0001.png ...   the generated frames, 1:1 with frames/ (RGB)
    clip/clip.json             control mode, frame map, size, seed, seconds, models
    clip/provenance.json       same shape as render/provenance.json

WHY A VIDEO MODEL. The anchor gate (2026-09-15) published every action as a
pose pass: the identity pass flattened poses, so cross-action identity was
prompt + seed only, and the knight's shield and armour wandered between
strike, hit and death. A reference-conditioned video model denoises all the
frames together against one reference latent, which is the mechanism the
2026-09-16 survey found for cross-frame identity (research doc S4.1). The
strip pose pass stays as the seconds-long preview.

WHY WAN 2.2 FUN-CONTROL 5B. Owner's choice (2026-09-17): ~10 GB bf16, fits the
24 GB A5000 with no quantisation, and it is what the official
`video_wan2_2_5B_fun_control` template loads through a plain UNETLoader. The
node is `Wan22FunControlToVideo`: `ref_image` is the identity, `control_video`
is the pose. A14B is the fallback if 5B fails the gate.

THE CONTROL VIDEO. Every frame's `pose/skel_%04d.png` (2 px sticks with the
unit's props drawn on, exactly what the strip route sends), and - when the rig
props feature has written `depth/%04d.png` beside them - the depth frame too.
Fun-Control has ONE control input, so "pose + depth" means one composited
video (`blend`: the skeleton drawn over the depth), not two channels. The
frames go up as an mp4 because core `LoadVideo` is the only batch input the
service has; it is encoded lossless so a 2 px line stays a 2 px line. Padded
to 4n+1 frames by holding the last one - Wan's latent holds 4 frames per step.

THE SEED IS DERIVED from the clip hash, action, slug and control mode, so a
re-run reproduces; the picks cannot be part of it because the clip is rendered
before `pick` runs. `--seed` re-rolls.

WHAT IT DOES NOT DO. It does not matte, pick or downscale: `post --input clip`
mattes the picked frames with the same `--matte` choices the strip route has,
and everything after that is unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anim import comfy, config, models_lock
from anim.job import Job
from anim.stages import render as render_stage
from anim.util import have, read_json, sha256_bytes, write_bytes_in_place, write_json_in_place
from anim.util import run as run_tool

NAME = "clip"
HELP = "every rig frame -> one generated clip (Wan 2.2 Fun-Control; a GPU stage)"

CLIP_SCHEMA = "anim.clip/1"
WORKFLOW = "clip_render"
CONTROL_MODES = ("auto", "pose", "depth", "blend")
# A skeleton pixel is anything that is not the black ground the renderer uses.
SKELETON_THRESHOLD = 16


# ---------------------------------------------------------------------------
# Control video assembly (pure functions; the tests live here)
# ---------------------------------------------------------------------------
def padded_length(frames: int, step: int = config.CLIP_FRAME_STEP) -> int:
    """The smallest 4n+1 >= frames. Wan's latent takes `step` frames per slot."""
    if frames < 1:
        raise ValueError("a clip needs at least one frame")
    return ((frames - 1 + step - 1) // step) * step + 1


def frame_map(frames: int, length: int) -> list[int]:
    """Control-video position -> rig frame index. The tail holds the last frame."""
    return [min(i, frames - 1) for i in range(length)]


def resolve_control(mode: str, have_depth: bool) -> str:
    """`auto` is blend when depth exists, else pose; the others are explicit."""
    if mode not in CONTROL_MODES:
        raise ValueError(f"unknown control mode {mode!r}; known: {CONTROL_MODES}")
    if mode == "auto":
        return "blend" if have_depth else "pose"
    if mode in ("depth", "blend") and not have_depth:
        raise FileNotFoundError(
            f"--control {mode} needs depth/ frames from `anim rig` (the rig "
            f"props feature) and this job has none")
    return mode


def compose_control(skeleton: Image.Image | None, depth: Image.Image | None,
                    mode: str, depth_gain: float = 1.0) -> Image.Image:
    """One control frame. `blend` draws the skeleton's pixels over the depth.

    `depth_gain` scales the depth's brightness before the skeleton goes on
    top: at 1.0 the mannequin's silhouette is as strong as the sticks and the
    model paints a mannequin (measured 2026-09-17, death lost its helmet);
    lower values keep the props' placement as a hint and let the reference
    decide the body.
    """
    if mode == "pose":
        assert skeleton is not None
        return skeleton.convert("RGB")
    assert depth is not None
    grey = np.asarray(depth.convert("L"), dtype=np.float32)
    if depth_gain != 1.0:
        grey = np.clip(grey * float(depth_gain), 0, 255)
    grey8 = grey.astype(np.uint8)
    if mode == "depth":
        return Image.fromarray(np.stack([grey8] * 3, axis=2), mode="RGB")
    assert skeleton is not None
    if depth.size != skeleton.size:
        raise ValueError(f"depth frame is {depth.size} but the skeleton is "
                         f"{skeleton.size}; the rig props feature writes them at "
                         f"the same cell")
    base = np.stack([grey8] * 3, axis=2).copy()
    skel = np.asarray(skeleton.convert("RGB"), dtype=np.uint8)
    drawn = skel.max(axis=2) > SKELETON_THRESHOLD
    base[drawn] = skel[drawn]
    return Image.fromarray(base, mode="RGB")


def check_cell(width: int, height: int) -> None:
    step = config.LATENT_STEP
    if width % step or height % step:
        raise ValueError(f"clip cell is {width}x{height}; Wan22FunControlToVideo "
                         f"has step {step} on both dimensions")


def build_control_frames(job: Job, mode: str, length: int,
                         cell: tuple[int, int],
                         depth_gain: float = 1.0) -> tuple[list[Path], list[int]]:
    """Write clip/control/%04d.png - one per control-video position."""
    skeletons = job.skeletons()
    depths = job.depths()
    frames = len(skeletons)
    if frames == 0:
        raise FileNotFoundError(f"no skeletons in {job.pose_dir} - run `anim rig` "
                                f"(or `anim pose`) first")
    if mode != "pose" and len(depths) != frames:
        raise ValueError(f"{len(depths)} depth frame(s) for {frames} skeleton(s); "
                         f"they must be 1:1")
    control_dir = job.clip_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    for stale in control_dir.glob("*.png"):
        stale.unlink()
    mapping = frame_map(frames, length)
    written: list[Path] = []
    for position, frame_index in enumerate(mapping):
        with Image.open(skeletons[frame_index]) as skel_img:
            skel = skel_img.convert("RGB")
        depth = None
        if mode != "pose":
            with Image.open(depths[frame_index]) as depth_img:
                depth = depth_img.convert("L")
        frame = compose_control(skel, depth, mode, depth_gain)
        if frame.size != tuple(cell):
            raise ValueError(f"{skeletons[frame_index].name} is {frame.size}, the "
                             f"clip cell is {cell}; re-run `anim rig` with "
                             f"--width {cell[0]} --height {cell[1]}")
        path = control_dir / f"{position + 1:04d}.png"
        frame.save(path)
        written.append(path)
    return written, mapping


def encode_control_video(frames_dir: Path, dest: Path, fps: int) -> Path:
    """Lossless H.264 (qp 0, 4:4:4) so the control lines survive the codec.

    `LoadVideo` decodes through PyAV; lossless 4:4:4 H.264 is a plain ffmpeg
    decode, which PyAV wraps. Verified on the service 2026-09-17 (NOTES.md).
    """
    if not have("ffmpeg"):
        raise RuntimeError("ffmpeg is not on PATH; the control video cannot be encoded")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_tool(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(max(1, fps)),
              "-i", str(frames_dir / "%04d.png"),
              "-c:v", "libx264", "-preset", "veryfast", "-qp", "0",
              "-pix_fmt", "yuv444p", "-movflags", "+faststart", str(dest)],
             timeout=600)
    return dest


def derive_seed(manifest: dict[str, Any], control: str) -> int:
    """Stable for (clip, action, unit, control mode). Picks are not known yet."""
    material = "|".join([manifest.get("clip_sha256", ""), manifest.get("action", ""),
                         manifest.get("slug", ""), f"clip:{control}"])
    return int(sha256_bytes(material.encode("utf-8"))[:13], 16) % render_stage.SEED_MAX


def find_reference(job: Job, manifest: dict[str, Any],
                   override: Path | None = None,
                   assets_dir: Path | None = None) -> tuple[Path, str]:
    """The identity image: `--reference`, else the unit's base render, else the
    published 64 px sprite upscaled x8 on white (the strip route's reference).
    """
    if override is not None:
        return Path(override), "override"
    slug = manifest.get("slug", "")
    base_job = Job.open(f"{slug}_base", job.root.parent)
    if base_job.manifest_path.exists():
        base_manifest = base_job.read_manifest()
        final = (base_manifest.get("render") or {}).get("final") or "base_sprite.png"
        candidate = base_job.render_dir / final
        if candidate.exists():
            return candidate, "base render"
    from anim.stages import strip as strip_stage

    sprite = strip_stage.reference_sprite(slug, None, assets_dir)
    job.clip_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(sprite) as img:
        up = img.convert("RGBA").resize((img.width * strip_stage.REFERENCE_UPSCALE,
                                         img.height * strip_stage.REFERENCE_UPSCALE),
                                        Image.NEAREST)
    flat = Image.new("RGB", up.size, strip_stage.REFERENCE_BG)
    flat.paste(up, (0, 0), up)
    path = job.clip_dir / "reference_sprite_x8.png"
    flat.save(path)
    return path, "sprite x8"


def fit_reference(source: Path, cell: tuple[int, int], dest: Path) -> Path:
    """Letterbox the reference onto the clip cell's aspect, on white.

    `Wan22FunControlToVideo` resizes `ref_image` to (width, height) with
    `common_upscale(..., "center")`, which CROPS a mismatched aspect - a
    512x768 base render sent to a 768x768 death clip would lose the knight's
    head and feet. Padding here means the node's resize is a no-op.
    """
    from anim.stages import strip as strip_stage

    with Image.open(source) as img:
        src = img.convert("RGB")
    width, height = cell
    scale = min(width / src.width, height / src.height)
    fitted = src.resize((max(1, round(src.width * scale)),
                         max(1, round(src.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), strip_stage.REFERENCE_BG)
    canvas.paste(fitted, ((width - fitted.width) // 2, height - fitted.height))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return dest


# ---------------------------------------------------------------------------
# The drive
# ---------------------------------------------------------------------------
def submit_batch(client: comfy.ComfyClient, workflow: dict[str, Any], *,
                 timeout_s: float) -> tuple[list[bytes], dict[str, Any]]:
    """Run the workflow and return EVERY output image, in batch order."""
    prompt_id = client.post_prompt(workflow)
    started = time.monotonic()
    entry = client.wait(prompt_id, timeout_s=timeout_s)
    elapsed = execution_seconds(entry) or (time.monotonic() - started)
    # Only the SaveImage output. `LoadVideo` reports its input preview under
    # `images` too (type "input") - measured 2026-09-17: 22 entries for a
    # 21-frame clip - and that is not a frame.
    images = sorted((i for i in client.outputs(entry) if i.type == "output"),
                    key=lambda i: i.filename)
    if not images:
        raise comfy.ComfyError(f"job {prompt_id} produced no images")
    pngs = [client.view(image) for image in images]
    return pngs, {
        "prompt_id": prompt_id,
        "workflow_sha256": comfy.workflow_sha256(workflow),
        "output": {"filename": images[0].filename, "subfolder": images[0].subfolder,
                   "count": len(images)},
        "embedded_prompt": render_stage.prompt_chunk(pngs[0]) is not None,
        "seconds": round(elapsed, 1),
    }


def execution_seconds(entry: dict[str, Any]) -> float | None:
    """ComfyUI's own clock: execution_start -> execution_success in /history.

    The service stamps both in the status messages (ms since the epoch), which
    is a cleaner number than a wall clock that includes queue time and polling.
    """
    stamps: dict[str, int] = {}
    for message in (entry.get("status") or {}).get("messages") or []:
        if isinstance(message, list) and len(message) == 2 and isinstance(message[1], dict):
            stamps[message[0]] = message[1].get("timestamp") or 0
    if stamps.get("execution_start") and stamps.get("execution_success"):
        return (stamps["execution_success"] - stamps["execution_start"]) / 1000.0
    return None


def clip(job_name: str, *, seed: int | None = None, control: str = "auto",
         depth_gain: float = 1.0,
         prompt: str | None = None, negative: str | None = None,
         steps: int | None = None, cfg: float | None = None,
         reference: Path | None = None, client: comfy.ComfyClient | None = None,
         workflow_dir: Path | None = None, lock_path: Path | None = None,
         work_dir: Path | None = None, assets_dir: Path | None = None,
         timeout_s: float = 1800.0) -> dict[str, Any]:
    job = Job.open(job_name, work_dir)
    manifest = job.read_manifest()
    skeletons = job.skeletons()
    if not skeletons:
        raise FileNotFoundError(f"no skeletons in {job.pose_dir} - run `anim rig` "
                                f"(or `anim pose`) first")
    # The skeleton PNGs ARE the control frames, so their size is the clip's
    # size - whatever `rig --width/--height` (or `pose`) drew them at.
    with Image.open(skeletons[0]) as first:
        cell = first.size
    check_cell(*cell)
    mode = resolve_control(control, bool(job.depths()))
    frames = len(skeletons)
    length = padded_length(frames)
    fps = int(manifest.get("fps") or config.EXTRACT_FPS)
    seed = derive_seed(manifest, mode) if seed is None else int(seed)

    client = client or comfy.ComfyClient()
    try:
        stats = client.system_stats()
    except comfy.ComfyError as exc:
        raise comfy.ComfyUnreachable(
            f"the GPU image service is not answering at {client.base_url}: {exc}\n"
            f"Start it with: sudo systemctl start docker  (owner, in a TTY), then\n"
            f"  docker --context docker-default compose -f "
            f"tools/anim-pipeline/gpu/compose.yml up -d") from exc

    job.clip_dir.mkdir(parents=True, exist_ok=True)
    control_frames, mapping = build_control_frames(job, mode, length, cell, depth_gain)
    video = encode_control_video(control_frames[0].parent, job.clip_dir / "control.mp4", fps)
    reference_source, reference_kind = find_reference(job, manifest, reference, assets_dir)
    reference_path = fit_reference(reference_source, cell, job.clip_dir / "reference.png")

    workflow = comfy.load_workflow(WORKFLOW, workflow_dir)
    # Both go to the input ROOT under job-unique names: LoadImage and LoadVideo
    # list only the root, and two jobs must not overwrite each other's inputs.
    comfy.patch(workflow, "reference image",
                image=client.upload_image(reference_path,
                                          name=f"anim_{job.name}_reference.png"))
    comfy.patch(workflow, "control video",
                file=client.upload_image(video, name=f"anim_{job.name}_control.mp4"))
    comfy.patch(workflow, "clip size", width=cell[0], height=cell[1], length=length)
    comfy.patch(workflow, "seed", seed=seed)
    comfy.patch(workflow, "output prefix",
                filename_prefix=f"anim_pipeline/{job.name}/clip/frame")
    comfy.patch(workflow, "positive prompt",
                text=prompt or config.clip_prompt(manifest.get("action") or ""))
    if negative:
        comfy.patch(workflow, "negative prompt", text=negative)
    if steps is not None:
        comfy.patch(workflow, "seed", steps=int(steps))
    if cfg is not None:
        comfy.patch(workflow, "seed", cfg=float(cfg))

    pngs, run_record = submit_batch(client, workflow, timeout_s=timeout_s)
    if len(pngs) != length:
        raise comfy.ComfyError(f"the service returned {len(pngs)} frame(s) for a "
                               f"{length}-frame clip")

    # One generated frame per RIG frame, named like frames/ so picks index them.
    # The padding tail is dropped: those positions all held the last rig frame.
    job.clip_frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in job.clip_frames_dir.glob("*.png"):
        stale.unlink()
    written: list[str] = []
    for position, frame_index in enumerate(mapping):
        if position != frame_index:
            continue
        path = write_bytes_in_place(job.clip_frames_dir / f"{frame_index + 1:04d}.png",
                                    pngs[position])
        written.append(path.name)

    models = [m.as_sidecar_entry() for m in models_lock.for_workflow(workflow, lock_path)]
    run_record |= {"name": "clip_render", "workflow": f"{WORKFLOW}.json", "seed": seed,
                   "file": f"frames/{written[0]}" if written else None, "models": models}
    provenance = render_stage.build_provenance(client.base_url, stats, seed, [run_record])
    provenance["final"] = "frames"
    write_json_in_place(job.clip_provenance_path, provenance)

    record = {
        "schema": CLIP_SCHEMA,
        "job": job.name,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "workflow": f"{WORKFLOW}.json",
        "workflow_sha256": run_record["workflow_sha256"],
        "control": mode,
        "depth_gain": float(depth_gain) if mode != "pose" else None,
        "control_video": video.name,
        "reference": {"file": str(reference_source), "kind": reference_kind,
                      "sent": reference_path.name},
        "cell": list(cell),
        "fps": fps,
        "frames": frames,
        "length": length,
        "frame_map": mapping,
        "seed": seed,
        "prompt": workflow[comfy.find_node(workflow, "positive prompt")]["inputs"]["text"],
        "sampler": {k: workflow[comfy.find_node(workflow, "seed")]["inputs"][k]
                    for k in ("steps", "cfg", "sampler_name", "scheduler")},
        "seconds": run_record["seconds"],
        "models": models,
        "written": written,
    }
    write_json_in_place(job.clip_path, record)
    # `clip` in the manifest is the CLIP FILE `extract`/`rig` recorded; this
    # stage's block is `clip_render`, like `render` has its own.
    job.update_manifest(clip_render={"seed": seed, "control": mode,
                                     "frames": len(written), "length": length,
                                     "cell": list(cell),
                                     "seconds": run_record["seconds"]})
    return record


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", help="job name, e.g. knightq_strike_clip")
    parser.add_argument("--force", action="store_true",
                        help="render even when the unit's curation gate is "
                             "pending (the curation board feature)")
    parser.add_argument("--seed", type=int, default=None,
                        help="explicit seed (default: derived from the clip hash, "
                             "action, unit and control mode, so a re-run reproduces)")
    parser.add_argument("--control", default="auto", choices=CONTROL_MODES,
                        help="what the control video carries: pose = the skeleton "
                             "PNGs; depth = the rig props feature's depth frames; "
                             "blend = skeleton drawn over depth; auto = blend when "
                             "depth/ exists, else pose (default)")
    parser.add_argument("--depth-gain", type=float, default=1.0,
                        help="brightness multiplier on the depth frames for the "
                             "depth/blend modes (1.0 = as rendered; lower keeps "
                             "the props as a hint and lets the reference own the "
                             "body)")
    parser.add_argument("--prompt", default=None,
                        help="positive prompt (default: config.clip_prompt(action))")
    parser.add_argument("--negative", default=None,
                        help="negative prompt (default: Wan's standard negative)")
    parser.add_argument("--steps", type=int, default=None, help="sampler steps (workflow default 20)")
    parser.add_argument("--cfg", type=float, default=None, help="CFG scale (workflow default 5.0)")
    parser.add_argument("--reference", default=None,
                        help="identity image (default: work/<unit>_base/render/"
                             "base_sprite.png, else the published sprite x8)")
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="seconds to wait for the /prompt job")


def run_stage(args: argparse.Namespace) -> int:
    client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
    record = clip(args.job, seed=args.seed, control=args.control,
                  depth_gain=args.depth_gain, prompt=args.prompt,
                  negative=args.negative, steps=args.steps, cfg=args.cfg,
                  reference=Path(args.reference) if args.reference else None,
                  client=client, timeout_s=args.timeout)
    print(f"clip: {record['frames']} frame(s) ({record['length']} sent, "
          f"{record['control']} control) at {record['cell'][0]}x{record['cell'][1]}, "
          f"seed {record['seed']}, {record['seconds']} s")
    print(f"  reference: {record['reference']['file']} ({record['reference']['kind']}, "
          f"fitted to the cell as clip/{record['reference']['sent']})")
    print(f"  frames -> {Job.open(args.job).clip_frames_dir}")
    print(f"  next: anim pick {args.job} && anim post {args.job} --input clip")
    return 0


run = run_stage
