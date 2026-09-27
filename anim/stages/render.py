"""Stage: `anim render` - strip -> generated strip (the ONLY GPU stage)

POST `gpu/workflows/pose_pass.json`, then `identity_pass.json`, to the GPU image
service; poll `/history`; save both strips and a provenance record.

    anim render knight_strike --seed 861234901

    render/pose_pass.png       Z-Image + Fun ControlNet: the skeleton, posed
    render/identity_pass.png   Qwen-Image-Edit: that pose, as THIS unit
    render/provenance.json     seed, workflow hashes, model hashes, versions

TWO JOBS, NOT ONE GRAPH. Z-Image (int8, ~6 GB) plus Qwen-Image-Edit (INT4,
~12 GB) plus two text encoders does not comfortably fit in 24 GB together.
Splitting lets ComfyUI's own model manager evict between them, and means a bad
identity pass can be re-rolled without paying for the pose pass again.

NODES ARE PATCHED BY `_meta.title`, NEVER BY ID. Re-exporting a workflow
renumbers nodes; swapping the reference sprite for the skeleton by accident
would produce a beautifully posed skeleton instead of a character.

THE SEED IS DERIVED, NOT RANDOM. With no `--seed`, it comes from the clip hash,
the action and the picks, so re-running the same job reproduces the same frames
- which is the promise the sidecar makes. Pass `--seed` to re-roll.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from anim import comfy, config, models_lock
from anim.job import Job
from anim.util import read_json, sha256_bytes, write_bytes_in_place, write_json_in_place

NAME = "render"
HELP = "strip -> generated strip (the only GPU stage)"

PROVENANCE_SCHEMA = "anim.provenance/1"
# ComfyUI's seed widget is a 64-bit field; keep well inside it.
SEED_MAX = 2 ** 53


def derive_seed(manifest: dict[str, Any], picks: list[int]) -> int:
    """A stable seed for this exact (clip, action, picks) triple."""
    material = "|".join([
        manifest.get("clip_sha256", ""),
        manifest.get("action", ""),
        manifest.get("slug", ""),
        ",".join(str(p) for p in picks),
    ])
    return int(sha256_bytes(material.encode("utf-8"))[:13], 16) % SEED_MAX


def prompt_chunk(png: bytes) -> dict[str, Any] | None:
    """The graph ComfyUI embedded in the PNG's `prompt` text chunk, if any.

    Core SaveImage writes it, which makes the output file self-describing. We
    keep it in provenance.json too, because a PNG that has been through the
    post-chain no longer carries its own metadata.
    """
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(png)) as img:
            raw = (img.info or {}).get("prompt")
    except Exception:  # pragma: no cover - a non-PNG body is already an error
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def submit(client: comfy.ComfyClient, workflow: dict[str, Any], *,
           timeout_s: float = 900.0) -> tuple[bytes, dict[str, Any]]:
    """Run one workflow to completion and return (png bytes, run record)."""
    prompt_id = client.post_prompt(workflow)
    entry = client.wait(prompt_id, timeout_s=timeout_s)
    images = client.outputs(entry)
    if not images:
        raise comfy.ComfyError(f"job {prompt_id} produced no images")
    png = client.view(images[0])
    return png, {
        "prompt_id": prompt_id,
        "workflow_sha256": comfy.workflow_sha256(workflow),
        "output": {"filename": images[0].filename,
                   "subfolder": images[0].subfolder},
        "embedded_prompt": prompt_chunk(png) is not None,
    }


def render(job_name: str, *, seed: int | None = None, strength: float | None = None,
           pose_prompt: str | None = None, identity_prompt: str | None = None,
           lora_strength: float | None = None, client: comfy.ComfyClient | None = None,
           workflow_dir: Path | None = None, lock_path: Path | None = None,
           work_dir: Path | None = None, timeout_s: float = 900.0) -> dict[str, Any]:
    job = Job.open(job_name, work_dir)
    manifest = job.read_manifest()
    if not job.layout_path.exists():
        raise FileNotFoundError(f"no {job.layout_path} - run `anim strip` first")
    layout = read_json(job.layout_path)
    picks = [cell["frame"] for cell in layout["cells"]]
    width, height = layout["strip"]["size"]
    seed = derive_seed(manifest, picks) if seed is None else int(seed)

    client = client or comfy.ComfyClient()
    job.render_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {}
    try:
        stats = client.system_stats()
    except comfy.ComfyError as exc:
        raise comfy.unreachable(client.base_url, exc) from exc

    skeleton = job.strip_dir / layout["strip"]["file"]
    reference = job.strip_dir / layout["reference"]["file"]

    # --- pass 1: pose ------------------------------------------------------
    pose_wf = comfy.load_workflow("pose_pass", workflow_dir)
    comfy.patch(pose_wf, "skeleton strip", image=client.upload_image(skeleton))
    comfy.patch(pose_wf, "strip size", width=width, height=height)
    comfy.patch(pose_wf, "seed", seed=seed)
    comfy.patch(pose_wf, "output prefix",
                filename_prefix=f"anim_pipeline/{job.name}/pose_pass")
    if strength is not None:
        comfy.patch(pose_wf, "control scale", strength=float(strength))
    if pose_prompt:
        comfy.patch(pose_wf, "positive prompt", text=pose_prompt)
    pose_png, pose_run = submit(client, pose_wf, timeout_s=timeout_s)
    pose_path = write_bytes_in_place(job.render_dir / "pose_pass.png", pose_png)
    pose_run |= {"name": "pose_pass", "workflow": "pose_pass.json", "seed": seed,
                 "file": pose_path.name,
                 "models": [m.as_sidecar_entry()
                            for m in models_lock.for_workflow(pose_wf, lock_path)]}

    # --- pass 2: identity --------------------------------------------------
    identity_wf = comfy.load_workflow("identity_pass", workflow_dir)
    comfy.patch(identity_wf, "image1", image=client.upload_image(reference))
    comfy.patch(identity_wf, "image2", image=client.upload_image(pose_path))
    # Two images, not three. Measured on knight_strike (2026-09-15): with the
    # skeleton as image2 and the pose pass as image3, Qwen-Image-Edit dropped
    # the strip layout (two zoomed figures, wrong character); with the pose
    # pass as image2 and no image3 it kept all four cells in place and
    # re-skinned them. The skeleton's job was done by the pose pass.
    pose_node = comfy.find_node(identity_wf, "image2")
    for node in identity_wf.values():
        if node["class_type"] == "VAEEncode":
            node["inputs"]["pixels"] = [pose_node, 0]  # the latent is still the pose pass
    comfy.unlink(identity_wf, "image3")
    comfy.patch(identity_wf, "seed", seed=seed)
    comfy.patch(identity_wf, "output prefix",
                filename_prefix=f"anim_pipeline/{job.name}/identity_pass")
    if lora_strength:
        comfy.patch(identity_wf, "Pixel-Art LoRA",
                    strength_model=float(lora_strength))
    else:
        # The LoRA file is optional and (as of 2026-09-15) not installable -
        # see gpu/README.md "Known issues" - and ComfyUI rejects a workflow
        # naming a missing file even at strength 0. So the node is only in
        # the graph when it is actually asked for.
        comfy.bypass(identity_wf, "Pixel-Art LoRA", "model")
    if identity_prompt:
        comfy.patch(identity_wf, "positive prompt", prompt=identity_prompt)
    identity_png, identity_run = submit(client, identity_wf, timeout_s=timeout_s)
    identity_path = write_bytes_in_place(job.render_dir / "identity_pass.png",
                                         identity_png)
    identity_run |= {"name": "identity_pass", "workflow": "identity_pass.json",
                     "seed": seed, "file": identity_path.name,
                     "models": [m.as_sidecar_entry()
                                for m in models_lock.for_workflow(identity_wf,
                                                                  lock_path)]}

    provenance = build_provenance(client.base_url, stats, seed,
                                  [pose_run, identity_run])
    write_json_in_place(job.provenance_path, provenance)
    job.update_manifest(render={"seed": seed, "passes": ["pose_pass", "identity_pass"],
                                "final": identity_path.name})
    return provenance


def build_provenance(url: str, stats: dict[str, Any], seed: int,
                     passes: list[dict[str, Any]]) -> dict[str, Any]:
    """The record that makes a frame re-creatable. Merged model list included."""
    merged: list[dict[str, Any]] = []
    for run_record in passes:
        for model in run_record.get("models") or []:
            if model not in merged:
                merged.append(model)
    system = stats.get("system") or {}
    devices = stats.get("devices") or []
    return {
        "schema": PROVENANCE_SCHEMA,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "comfyui": {
            "url": url,
            "version": system.get("comfyui_version") or system.get("version"),
            "python": system.get("python_version"),
            "devices": [d.get("name") for d in devices],
        },
        "seed": seed,
        "passes": passes,
        "models": merged,
        # The final image is the LAST pass - `post` slices that one.
        "final": passes[-1]["file"] if passes else None,
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", help="job name, e.g. knight_strike")
    parser.add_argument("--force", action="store_true",
                        help="render even when the unit's curation gate is "
                             "pending (the base gate)")
    parser.add_argument("--seed", type=int, default=None,
                        help="explicit seed (default: derived from the clip "
                             "hash, action and picks, so a re-run reproduces)")
    parser.add_argument("--strength", type=float, default=None,
                        help="ControlNet control scale on the pose pass "
                             "(workflow default 0.7)")
    parser.add_argument("--lora-strength", type=float, default=None,
                        help="Pixel-Art LoRA strength on the identity pass "
                             "(workflow default 0.0 = bypassed)")
    parser.add_argument("--pose-prompt", default=None)
    parser.add_argument("--identity-prompt", default=None)
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait for one /prompt job")


def run_stage(args: argparse.Namespace) -> int:
    client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
    provenance = render(args.job, seed=args.seed, strength=args.strength,
                        lora_strength=args.lora_strength,
                        pose_prompt=args.pose_prompt,
                        identity_prompt=args.identity_prompt,
                        client=client, timeout_s=args.timeout)
    print(f"render: seed {provenance['seed']} on "
          f"ComfyUI {provenance['comfyui']['version']} "
          f"({', '.join(filter(None, provenance['comfyui']['devices'])) or 'no device reported'})")
    for run_record in provenance["passes"]:
        print(f"  {run_record['name']}: {run_record['file']} "
              f"(workflow {run_record['workflow_sha256'][:12]})")
    return 0


run = run_stage
