"""Stage: `anim concept` - text or sketch -> N candidate base characters -> pick

The character concept feature (sprite-studio.md S2 "Base character"): the first
thing the studio asks for is *who* the character is, and the answer is a
front-facing base image the owner chooses from several - generated ones and,
side by side, the art that already exists.

    anim concept --unit knight --prompt "a plate-armoured knight with a kite shield"
    anim concept --unit knight --prompt "..." --sketch scans/knight.png --count 6
    anim concept --unit knight --prompt "..." --existing            # the published sprite as 00
    anim concept --unit knight --pick 02                             # or --pick existing

    work/<slug>_concept/
      job.json               prompt, seed per candidate, LoRA + strength, workflow hash
      sketch.png             the control image actually sent (white strokes on black)
      candidates/NN.png      one per candidate; 00 is the existing art when registered

NO SKELETON. `base` is the pose stack - a skeleton the Fun ControlNet traces -
because a unit's standing sprite has to match the rig's proportions. A concept
is earlier than that: there is no pose to hold yet, only a description, so this
is Z-Image-Turbo text-to-image with the pixel-art style LoRA
(`config.CONCEPT_LORA`, tarn59's, Apache-2.0) on the same int8 model, and the
optional sketch goes through the same Union ControlNet as a SCRIBBLE - white
strokes on black, which is the condition the Union model reads from the image
alone (it has no mode input). With no sketch the ControlNet nodes are not in
the submitted graph at all, so the provenance hash and the model list describe
what the image actually saw.

ONE `/prompt` PER CANDIDATE, `batch_size` 1. Each candidate has its own seed
(base seed + index), so any one of them can be re-made alone from the sidecar
without regenerating its neighbours - the reproducibility rule every other
stage keeps.

WHERE THE PICK LANDS. Every later stage reads the unit's reference from the
base job: `clip` takes `work/<slug>_base/render/<render.final>` as the identity
image, `post --parts` extracts the part palettes there, `post` + `publish` turn
it into the 64 px sprite. So `--pick` writes the chosen candidate into exactly
that slot - `render/concept_pick.png`, with `render.final` pointing at it and a
provenance record from the concept job - after laying the base job out with
`base.prepare` at the candidate's size. The base contract does not change;
`anim base` is untouched and `anim post <slug>_base` / `anim publish
<slug>_base` run verbatim on the pick. An earlier base render's files stay on
disk (only the manifest is rewritten), and the concept job keeps every
candidate, so a pick can be changed by picking again.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from anim import comfy, config, imaging, models_lock
from anim.job import Job
from anim.stages import base as base_stage
from anim.stages import render as render_stage
from anim.stages.clip import execution_seconds
from anim.util import (sha256_bytes, sha256_file, write_bytes_in_place,
                       write_json_in_place)

NAME = "concept"
HELP = "text or sketch -> N candidate base characters -> --pick one"

CONCEPT_SCHEMA = "anim.concept/1"
WORKFLOW = "concept_pass"
PICK_FILE = "concept_pick.png"
EXISTING = "existing"
# The white the base pass and the strip reference put behind a figure.
BACKGROUND = (255, 255, 255)


def job_name(slug: str) -> str:
    return f"{slug}_concept"


def derive_seed(slug: str, prompt: str, sketch_sha256: str | None = None) -> int:
    """Stable for (unit, prompt, sketch): the same ask reproduces the same set."""
    material = "|".join([slug, prompt, sketch_sha256 or "", "concept"])
    return int(sha256_bytes(material.encode("utf-8"))[:13], 16) % render_stage.SEED_MAX


# ---------------------------------------------------------------------------
# Inputs: the sketch and the existing art
# ---------------------------------------------------------------------------
def sketch_control(source: Path, cell: tuple[int, int], dest: Path) -> Path:
    """The owner's sketch as the Union model's scribble condition.

    Greyscale, strokes white on black - a pencil sketch is dark on light, so a
    light-majority image is inverted - and letterboxed into the cell on black
    so the strokes land where the figure should. No thresholding: a soft line
    is still a line to the model, and a hard edge would add jaggies that the
    model then draws.
    """
    with Image.open(source) as raw:
        if raw.mode in ("RGBA", "LA", "P"):
            # Transparent pixels are paper, not ink.
            rgba = raw.convert("RGBA")
            flat = Image.new("RGBA", rgba.size, BACKGROUND + (255,))
            flat.paste(rgba, (0, 0), rgba)
            grey = flat.convert("L")
        else:
            grey = raw.convert("L")
    histogram = grey.histogram()
    light = sum(histogram[128:])
    if light > grey.width * grey.height / 2:
        grey = ImageOps.invert(grey)
    scale = min(cell[0] / grey.width, cell[1] / grey.height)
    fitted = grey.resize((max(1, round(grey.width * scale)),
                          max(1, round(grey.height * scale))), Image.LANCZOS)
    canvas = Image.new("L", cell, 0)
    canvas.paste(fitted, ((cell[0] - fitted.width) // 2, cell[1] - fitted.height))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(dest)
    return dest


def existing_art(slug: str, existing: Path | str | None,
                 assets_dir: Path | None = None) -> tuple[Path, Image.Image] | None:
    """The art the generated candidates are compared against, as an image.

    `existing` is a path, or "auto" for the unit's published sprite
    (`assets/2d/units/<slug>.png`, the same lookup `strip` uses) shown at the
    strip reference's x8 nearest upscale so 64 px art is not a dot next to a
    768 px render. None means no existing candidate.
    """
    if existing is None:
        return None
    from anim.stages import strip as strip_stage

    if str(existing) == "auto":
        path = strip_stage.reference_sprite(slug, None, assets_dir)
        with Image.open(path) as raw:
            image = imaging.upscale_nearest(raw.convert("RGBA"),
                                            strip_stage.REFERENCE_UPSCALE)
        return path, image
    path = Path(existing)
    if not path.exists():
        raise FileNotFoundError(f"no existing art at {path}")
    with Image.open(path) as raw:
        image = raw.convert("RGBA")
    return path, image


def candidate_from_art(image: Image.Image, cell: tuple[int, int]) -> Image.Image:
    """An existing image in the candidates' format: bottom-centred on white in
    the cell, scaled DOWN only if it overflows, never up (nearest, so pixel art
    stays pixel art)."""
    return imaging.fit_into_cell(image, cell, background=BACKGROUND)


# ---------------------------------------------------------------------------
# Lay out the job
# ---------------------------------------------------------------------------
def prepare(slug: str, *, description: str, count: int | None = None,
            seed: int | None = None, sketch: Path | None = None,
            existing: Path | str | None = None,
            lora_strength: float | None = None,
            sketch_strength: float | None = None,
            cell: tuple[int, int] | None = None,
            assets_dir: Path | None = None, work_dir: Path | None = None) -> Job:
    """Write work/<slug>_concept/: job.json, the control image, candidate 00.

    Nothing here touches the GPU; `render_concept` fills the candidates in.
    """
    if not description or not description.strip():
        raise ValueError("--prompt is required: the description is the only "
                         "thing the model is told about the unit")
    cell = tuple(cell or config.CONCEPT_CELL)
    if cell[0] % config.LATENT_STEP or cell[1] % config.LATENT_STEP:
        raise ValueError(f"cell {cell} is not a multiple of {config.LATENT_STEP}")
    count = config.CONCEPT_COUNT if count is None else int(count)
    if count < 1:
        raise ValueError("--count must be at least 1")
    job = Job.open(job_name(slug), work_dir)
    job.candidates_dir.mkdir(parents=True, exist_ok=True)
    for stale in job.candidates_dir.glob("*.png"):
        stale.unlink()

    prompt = config.concept_prompt(description)
    sketch_block: dict[str, Any] | None = None
    sketch_sha: str | None = None
    if sketch is not None:
        sketch = Path(sketch)
        if not sketch.exists():
            raise FileNotFoundError(f"no sketch at {sketch}")
        control = sketch_control(sketch, cell, job.root / "sketch.png")
        sketch_sha = sha256_file(sketch)
        sketch_block = {
            "source": str(sketch), "source_sha256": sketch_sha,
            "file": control.name, "control": "scribble",
            "strength": float(config.CONCEPT_SKETCH_STRENGTH
                              if sketch_strength is None else sketch_strength),
        }
    elif (job.root / "sketch.png").exists():
        (job.root / "sketch.png").unlink()

    seed = derive_seed(slug, prompt, sketch_sha) if seed is None else int(seed)
    strength = float(config.CONCEPT_LORA["strength"]
                     if lora_strength is None else lora_strength)

    candidates: list[dict[str, Any]] = []
    art = existing_art(slug, existing, assets_dir)
    if art is not None:
        source, image = art
        fitted = candidate_from_art(image, cell)
        path = job.candidates_dir / "00.png"
        fitted.save(path)
        candidates.append({"index": 0, "file": path.name, "source": EXISTING,
                           "from": str(source), "seed": None, "seconds": None})
    first = len(candidates)
    for offset in range(count):
        index = first + offset
        candidates.append({"index": index, "file": f"{index:02d}.png",
                           "source": "generated", "seed": seed + offset,
                           "seconds": None})

    manifest = {
        "schema": CONCEPT_SCHEMA,
        "slug": slug,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "description": description.strip(),
        "prompt": prompt,
        "seed": seed,
        "count": count,
        "cell": list(cell),
        "lora": {"repo": config.CONCEPT_LORA["repo"],
                 "file": config.CONCEPT_LORA["file"],
                 "trigger": config.CONCEPT_LORA["trigger"],
                 "strength": strength},
        "sketch": sketch_block,
        "workflow": f"{WORKFLOW}.json",
        "workflow_sha256": None,
        "comfyui": None,
        "models": [],
        "candidates": candidates,
        "pick": None,
    }
    job.write_manifest(manifest)
    return job


# ---------------------------------------------------------------------------
# The GPU: one /prompt per candidate
# ---------------------------------------------------------------------------
def build_workflow(manifest: dict[str, Any], *, seed: int, sketch_name: str | None,
                   job: Job, workflow_dir: Path | None = None) -> dict[str, Any]:
    """concept_pass.json with this candidate's inputs patched in.

    Pure apart from the file read, so a test can see exactly what goes over
    the wire: the prompt, the seed, the LoRA file + strength, the cell, and -
    only with a sketch - the control image and its strength.
    """
    workflow = comfy.load_workflow(WORKFLOW, workflow_dir)
    width, height = manifest["cell"]
    comfy.patch(workflow, "positive prompt", text=manifest["prompt"])
    comfy.patch(workflow, "cell size", width=int(width), height=int(height))
    comfy.patch(workflow, "seed", seed=int(seed))
    comfy.patch(workflow, "output prefix",
                filename_prefix=f"anim_pipeline/{job.name}/{WORKFLOW}")
    lora = manifest["lora"]
    if lora["strength"] > 0:
        comfy.patch(workflow, "Pixel-Art LoRA", lora_name=lora["file"],
                    strength_model=float(lora["strength"]))
    else:
        comfy.bypass(workflow, "Pixel-Art LoRA", "model")
    sketch = manifest.get("sketch")
    if sketch and sketch_name:
        comfy.patch(workflow, "sketch", image=sketch_name)
        comfy.patch(workflow, "control scale", strength=float(sketch["strength"]))
    else:
        # No sketch: the ControlNet apply node goes, and with it the Union
        # model loader and the LoadImage that only it consumed.
        comfy.bypass(workflow, "control scale", "model")
        comfy.prune_orphans(workflow)
    return workflow


def render_concept(job: Job, *, client: comfy.ComfyClient | None = None,
                   workflow_dir: Path | None = None, lock_path: Path | None = None,
                   timeout_s: float = 900.0) -> dict[str, Any]:
    manifest = job.read_manifest()
    client = client or comfy.ComfyClient()
    try:
        stats = client.system_stats()
    except comfy.ComfyError as exc:
        raise comfy.ComfyUnreachable(
            f"the GPU image service is not answering at {client.base_url}: "
            f"{exc}") from exc

    sketch_name: str | None = None
    if manifest.get("sketch"):
        # The input ROOT under a job-unique name, like `clip`'s uploads.
        sketch_name = client.upload_image(job.root / manifest["sketch"]["file"],
                                          name=f"anim_{job.name}_sketch.png")

    models: list[dict[str, Any]] = []
    workflow_sha: str | None = None
    for candidate in manifest["candidates"]:
        if candidate["source"] != "generated":
            continue
        workflow = build_workflow(manifest, seed=candidate["seed"],
                                  sketch_name=sketch_name, job=job,
                                  workflow_dir=workflow_dir)
        started = time.monotonic()
        prompt_id = client.post_prompt(workflow)
        entry = client.wait(prompt_id, timeout_s=timeout_s)
        images = [i for i in client.outputs(entry) if i.type == "output"]
        if not images:
            raise comfy.ComfyError(f"job {prompt_id} produced no images")
        png = client.view(images[0])
        path = write_bytes_in_place(job.candidates_dir / candidate["file"], png)
        elapsed = execution_seconds(entry) or (time.monotonic() - started)
        candidate |= {"prompt_id": prompt_id, "seconds": round(elapsed, 1),
                      "file": path.name,
                      # The graph AS SENT, seed included - what `render`
                      # records per pass and what the pick's provenance copies.
                      "workflow_sha256": comfy.workflow_sha256(workflow),
                      "embedded_prompt": render_stage.prompt_chunk(png) is not None}
        if workflow_sha is None:
            # Same graph for every candidate but the seed; the job-level hash
            # is the first one with its seed zeroed - the SET's, not a member's.
            unseeded = comfy.patch(copy.deepcopy(workflow), "seed", seed=0)
            workflow_sha = comfy.workflow_sha256(unseeded)
            models = [m.as_sidecar_entry()
                      for m in models_lock.for_workflow(workflow, lock_path)]

    comfyui = render_stage.build_provenance(client.base_url, stats, manifest["seed"],
                                            [])["comfyui"]
    manifest |= {"workflow_sha256": workflow_sha, "comfyui": comfyui, "models": models}
    job.write_manifest(manifest)
    return manifest


def concept(slug: str, *, description: str, count: int | None = None,
            seed: int | None = None, sketch: Path | None = None,
            existing: Path | str | None = None, lora_strength: float | None = None,
            sketch_strength: float | None = None,
            client: comfy.ComfyClient | None = None,
            workflow_dir: Path | None = None, lock_path: Path | None = None,
            assets_dir: Path | None = None, work_dir: Path | None = None,
            timeout_s: float = 900.0) -> dict[str, Any]:
    job = prepare(slug, description=description, count=count, seed=seed,
                  sketch=sketch, existing=existing, lora_strength=lora_strength,
                  sketch_strength=sketch_strength, assets_dir=assets_dir,
                  work_dir=work_dir)
    return render_concept(job, client=client, workflow_dir=workflow_dir,
                          lock_path=lock_path, timeout_s=timeout_s)


# ---------------------------------------------------------------------------
# The pick
# ---------------------------------------------------------------------------
def resolve_pick(manifest: dict[str, Any], choice: str | int) -> dict[str, Any]:
    candidates = manifest.get("candidates") or []
    if str(choice).strip().lower() == EXISTING:
        for candidate in candidates:
            if candidate["source"] == EXISTING:
                return candidate
        raise ValueError("no existing-art candidate in this concept job; run "
                         "`anim concept --existing [PATH]` to register one as 00")
    try:
        index = int(choice)
    except (TypeError, ValueError):
        raise ValueError(f"--pick takes a candidate number or 'existing', "
                         f"not {choice!r}") from None
    for candidate in candidates:
        if candidate["index"] == index:
            return candidate
    raise ValueError(f"no candidate {index:02d}; this job has "
                     f"{[c['index'] for c in candidates]}")


def pick(slug: str, choice: str | int, *, work_dir: Path | None = None
         ) -> dict[str, Any]:
    """Make candidate `choice` the unit's base render (see the module docstring)."""
    job = Job.open(job_name(slug), work_dir)
    manifest = job.read_manifest()
    candidate = resolve_pick(manifest, choice)
    source = job.candidates_dir / candidate["file"]
    if not source.exists():
        raise FileNotFoundError(f"candidate {candidate['index']:02d} has no file at "
                                f"{source}; run `anim concept` again")
    png = source.read_bytes()
    with Image.open(source) as img:
        size = img.size

    base_job = Job.open(f"{slug}_base", work_dir)
    replaced: str | None = None
    if base_job.manifest_path.exists():
        previous = base_job.read_manifest()
        final = (previous.get("render") or {}).get("final")
        if final and final != PICK_FILE and (base_job.render_dir / final).exists():
            replaced = final
    base_stage.prepare(slug, cell=size, work_dir=work_dir)
    out = write_bytes_in_place(base_job.render_dir / PICK_FILE, png)

    generated = candidate["source"] == "generated"
    passes: list[dict[str, Any]] = []
    if generated:
        passes.append({
            "name": "concept_pass", "workflow": manifest["workflow"],
            "workflow_sha256": candidate.get("workflow_sha256"),
            "prompt_id": candidate.get("prompt_id"), "seed": candidate["seed"],
            "file": out.name, "seconds": candidate.get("seconds"),
            "embedded_prompt": candidate.get("embedded_prompt"),
            "models": list(manifest.get("models") or [])})
    provenance = {
        "schema": render_stage.PROVENANCE_SCHEMA,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "comfyui": manifest.get("comfyui") or {"url": None, "version": None,
                                               "python": None, "devices": []},
        "seed": candidate.get("seed"),
        "passes": passes,
        "models": list(manifest.get("models") or []) if generated else [],
        "final": out.name,
        # Where this render came from - the concept job and the candidate -
        # and, for the existing art, the file it was.
        "concept": {"job": job.name, "candidate": candidate["index"],
                    "file": candidate["file"], "source": candidate["source"],
                    "from": candidate.get("from"),
                    "prompt": manifest["prompt"] if generated else None,
                    "lora": manifest["lora"] if generated else None,
                    "sketch": manifest.get("sketch") if generated else None},
    }
    write_json_in_place(base_job.provenance_path, provenance)
    base_job.update_manifest(
        render={"seed": candidate.get("seed"),
                "passes": [p["name"] for p in passes], "final": out.name},
        concept=provenance["concept"])

    manifest["pick"] = {"index": candidate["index"], "file": candidate["file"],
                        "source": candidate["source"], "base_job": base_job.name,
                        "picked": provenance["created"]}
    job.write_manifest(manifest)
    return {"candidate": candidate, "base_job": base_job.name,
            "file": str(out), "replaced": replaced}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unit", "--slug", dest="slug", required=True,
                        help="unit slug, e.g. knight")
    parser.add_argument("--prompt", default=None,
                        help="the unit's description; wrapped in config."
                             "CONCEPT_PROMPT with the LoRA's trigger phrase")
    parser.add_argument("--sketch", default=None, metavar="PNG",
                        help="a sketch of the character, sent as a scribble "
                             "condition to the Fun ControlNet Union")
    parser.add_argument("--count", type=int, default=None,
                        help=f"candidates to generate (default "
                             f"{config.CONCEPT_COUNT})")
    parser.add_argument("--seed", type=int, default=None,
                        help="base seed; candidate i uses seed + i (default: "
                             "derived from the unit, prompt and sketch)")
    parser.add_argument("--existing", nargs="?", const="auto", default=None,
                        metavar="PNG",
                        help="register the existing art as candidate 00: a "
                             "path, or with no value the unit's published "
                             "sprite (assets/2d/units/<slug>.png, shown x8)")
    parser.add_argument("--lora-strength", type=float, default=None,
                        help=f"Pixel-Art LoRA strength (default "
                             f"{config.CONCEPT_LORA['strength']}; 0 drops the node)")
    parser.add_argument("--sketch-strength", type=float, default=None,
                        help=f"ControlNet strength for the sketch (default "
                             f"{config.CONCEPT_SKETCH_STRENGTH})")
    parser.add_argument("--pick", default=None, metavar="NN|existing",
                        help="make that candidate the unit's base render "
                             "(work/<slug>_base/render/concept_pick.png) and stop")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait for one /prompt job")


def run_stage(args: argparse.Namespace) -> int:
    if args.pick is not None:
        result = pick(args.slug, args.pick)
        candidate = result["candidate"]
        print(f"concept: candidate {candidate['index']:02d} ({candidate['source']}"
              f"{', seed ' + str(candidate['seed']) if candidate['seed'] is not None else ''}) "
              f"-> {result['file']}")
        if result["replaced"]:
            print(f"  replaced the base render {result['replaced']} as "
                  f"{result['base_job']}'s render.final (the file is still there)")
        print(f"  next: anim post {result['base_job']} [--parts], then "
              f"anim publish {result['base_job']}")
        return 0
    if not args.prompt:
        raise SystemExit("anim concept: --prompt is required (or --pick to choose)")
    client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
    manifest = concept(args.slug, description=args.prompt, count=args.count,
                       seed=args.seed,
                       sketch=Path(args.sketch) if args.sketch else None,
                       existing=args.existing, lora_strength=args.lora_strength,
                       sketch_strength=args.sketch_strength, client=client,
                       timeout_s=args.timeout)
    job = Job.open(job_name(args.slug))
    print(f"concept: {len(manifest['candidates'])} candidate(s) in {job.candidates_dir} "
          f"(ComfyUI {manifest['comfyui']['version']}, seed {manifest['seed']}, "
          f"LoRA {manifest['lora']['strength']}"
          f"{', sketch ' + manifest['sketch']['file'] if manifest.get('sketch') else ''})")
    for candidate in manifest["candidates"]:
        seconds = candidate.get("seconds")
        print(f"  {candidate['file']}  {candidate['source']:9s}  "
              f"seed {candidate['seed'] if candidate['seed'] is not None else '-':<18} "
              f"{f'{seconds:.1f} s' if seconds is not None else ''}")
    print(f"  then: anim concept --unit {args.slug} --pick NN")
    return 0


run = run_stage
