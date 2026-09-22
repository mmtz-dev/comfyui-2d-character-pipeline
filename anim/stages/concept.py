"""Stage: `anim concept` - text or sketch -> N candidate base characters -> pick

The character concept feature (sprite-studio.md S2 "Base character"): the first
thing the studio asks for is *who* the character is, and the answer is a
front-facing base image the owner chooses from several - generated ones and,
side by side, the art that already exists.

    anim concept --unit knight --prompt "a plate-armoured knight with a kite shield"
    anim concept --unit knight --prompt "..." --sketch scans/knight.png --count 6
    anim concept --unit knight --prompt "..." --existing            # the published sprite as 00
    anim concept --unit knight --pick 02                             # or --pick existing
    anim concept --unit knight --pick 02 --parts "closed great helm, kite shield, longsword, dark cloak" \
                                         --avoid "bare head, uncovered face"
    anim concept --unit knight --parts "..." [--avoid "..."]        # set the phrases on the base without a new pick
    anim concept --unit knight --views                                # back, left, right of the accepted base
    anim concept --unit knight --views all --count 2                  # + the four obliques, 2 candidates each
    anim concept --unit knight --pick-view back=01 left=00 right=02   # the views the turnaround will use

    work/<slug>_concept/
      job.json               prompt, seed per candidate, LoRA + strength, workflow hash
      sketch.png             the control image actually sent (white strokes on black)
      candidates/NN.png      one per candidate; 00 is the existing art when registered
      views/<view>_NN.png    the multi-view reference feature: the base from another side
      views/views.json       per view: prompt, seeds, seconds, the pick
    work/_views/<slug>/index.html   the views page (the base beside each view's candidates)

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

THE PARTS PHRASE (the multi-view reference feature, #298). A front-view
reference cannot say what the back of a closed helm looks like, and Wan
invented bare heads on the turned cells of the first identity gate. So the
pick records, in the base's `concept` block, a `parts_phrase` - the parts
that must survive every turn, "closed great helm, kite shield, longsword,
dark cloak" - and an `avoid_phrase` for the negative ("bare head, uncovered
face"). Both are explicit strings the owner gives with `--parts` / `--avoid`
(alone, or with `--pick`); with no `--parts` the phrase is the whole
description the front was rendered from, which is honest and a little long.
`config.clip_prompt(action, direction, parts)` puts them in every clip of
the unit; a base with no parts phrase renders the old prompt word for word.

THE VIEWS (`--views`). The base is the one image every stage trusts, so the
other sides are drawn FROM it: Qwen-Image-Edit 2509 (the identity edit path
already in the container - `gpu/workflows/view_pass.json`, one image in,
`TextEncodeQwenImageEditPlus` with the base as image1 and the base's own
latent for the size) with a per-view sentence and the parts phrase. N
candidates per view (`--count`, default `config.CONCEPT_VIEW_COUNT`), each
with its own seed (base seed + index) so one can be re-made alone, into
`views/<view>_NN.png` and `views/views.json`; the page puts the base beside
each view's candidates. The alternative - Z-Image + the Fun ControlNet with
a rig-rendered depth of the mannequin at that yaw - would hold the pose but
carry identity in the prompt only, which is the failure being fixed; it is
the fallback if the edit path turns out not to keep the character.

`--pick-view back=01 ...` copies the chosen candidates into the base job as
`render/view_<view>.png` and records them in the base's `concept.views`
(file, candidate, seed, sha256, `provisional` false) and in the curation
record's `views` step (accepted); `anim turnaround` then hands each direction
the nearest view (`config.TURNAROUND_REFERENCE_BY_DIR`). `--pick` of a new
base clears the views: they were drawn from the old one.

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
from typing import Any, Callable, Sequence

from PIL import Image, ImageOps

from anim import comfy, config, curation, imaging, models_lock
from anim.job import Job
from anim.stages import base as base_stage
from anim.stages import render as render_stage
from anim.stages.clip import execution_seconds
from anim.util import (read_json, sha256_bytes, sha256_file, write_bytes_in_place,
                       write_json_in_place, write_text_in_place)

NAME = "concept"
HELP = "text or sketch -> N candidate base characters -> --pick one"

CONCEPT_SCHEMA = "anim.concept/1"
VIEWS_SCHEMA = "anim.concept-views/1"
WORKFLOW = "concept_pass"
VIEW_WORKFLOW = "view_pass"
PICK_FILE = "concept_pick.png"
VIEW_FILE = "view_{view}.png"
EXISTING = "existing"
VIEWS_PAGE_ROOT = "_views"
VIEWS_VISUALS_TARGET = "~/.cache/claude/visuals/sprite-studio/views"
VIEWS_VISUALS_URL = "http://192.168.1.158:8090/sprite-studio/views/"
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


def parts_phrases(parts: str | None, avoid: str | None,
                  description: str | None) -> tuple[str, str]:
    """(parts phrase, avoid phrase) as stored: the owner's strings, cleaned;
    with no `--parts` the whole description the front was rendered from."""
    clean = lambda text: " ".join(str(text or "").split()).strip().rstrip(",.")  # noqa: E731
    return clean(parts) or clean(description), clean(avoid)


def pick(slug: str, choice: str | int, *, parts: str | None = None,
         avoid: str | None = None, work_dir: Path | None = None
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
    # The multi-view reference feature: the parts that must survive every
    # turn, and what the negative names; the views of an earlier base are
    # cleared - they were drawn from a different character.
    parts_phrase, avoid_phrase = parts_phrases(parts, avoid, manifest.get("description"))
    provenance["concept"] |= {"parts_phrase": parts_phrase, "avoid_phrase": avoid_phrase,
                              "views": {}}
    for stale in base_job.render_dir.glob(VIEW_FILE.format(view="*")):
        stale.unlink()
    write_json_in_place(base_job.provenance_path, provenance)
    base_job.update_manifest(
        render={"seed": candidate.get("seed"),
                "passes": [p["name"] for p in passes], "final": out.name},
        concept=provenance["concept"])

    manifest["pick"] = {"index": candidate["index"], "file": candidate["file"],
                        "source": candidate["source"], "base_job": base_job.name,
                        "picked": provenance["created"]}
    job.write_manifest(manifest)
    # The curation board feature: choosing the base IS the base gate. The
    # turnaround, drawn from this base, goes back to pending - and so do the
    # views (the multi-view reference feature).
    gates = curation.accept_base_from_concept(slug, work_dir)["gates"]
    return {"candidate": candidate, "base_job": base_job.name,
            "file": str(out), "replaced": replaced, "gates": gates,
            "parts_phrase": parts_phrase, "avoid_phrase": avoid_phrase}


def set_phrases(slug: str, *, parts: str | None = None, avoid: str | None = None,
                work_dir: Path | None = None) -> dict[str, Any]:
    """`--parts` / `--avoid` without a pick: the phrases onto the base's
    concept block as it is, nothing else touched (no gate moves, the views
    stay)."""
    base_job = Job.open(f"{slug}_base", work_dir)
    if not base_job.manifest_path.exists():
        raise FileNotFoundError(f"{slug} has no base job; `anim concept --unit {slug} "
                                f"--pick NN` first")
    manifest = base_job.read_manifest()
    block = dict(manifest.get("concept") or {})
    description = block.get("prompt") or ""
    parts_phrase, avoid_phrase = parts_phrases(
        parts if parts is not None else block.get("parts_phrase"),
        avoid if avoid is not None else block.get("avoid_phrase"), description)
    block |= {"parts_phrase": parts_phrase, "avoid_phrase": avoid_phrase}
    block.setdefault("views", {})
    base_job.update_manifest(concept=block)
    if base_job.provenance_path.exists():
        provenance = read_json(base_job.provenance_path)
        provenance["concept"] = {**(provenance.get("concept") or {}), **block}
        write_json_in_place(base_job.provenance_path, provenance)
    return block


def base_phrases(slug: str, work_dir: Path | None = None) -> tuple[str | None, str | None]:
    """(parts phrase, avoid phrase) of the unit's base, or (None, None) for
    a base without them - the old prompt, word for word."""
    base_job = Job.open(f"{slug}_base", work_dir)
    if not base_job.manifest_path.exists():
        return None, None
    block = base_job.read_manifest().get("concept") or {}
    parts = (block.get("parts_phrase") or "").strip() or None
    return parts, ((block.get("avoid_phrase") or "").strip() or None)


# ---------------------------------------------------------------------------
# The views: the base from the other sides (the multi-view reference feature)
# ---------------------------------------------------------------------------
def views_dir(slug: str, work_dir: Path | None = None) -> Path:
    return Job.open(job_name(slug), work_dir).root / "views"


def views_path(slug: str, work_dir: Path | None = None) -> Path:
    return views_dir(slug, work_dir) / "views.json"


def views_page_dir(slug: str, work_dir: Path | None = None) -> Path:
    base = Path(work_dir) if work_dir is not None else config.WORK_DIR
    return base / VIEWS_PAGE_ROOT / slug


def base_render(slug: str, work_dir: Path | None = None) -> tuple[Job, Path, dict[str, Any]]:
    """The accepted base's render (what `clip` reads as the identity), its
    job and manifest; refused when there is no base to draw the views from."""
    base_job = Job.open(f"{slug}_base", work_dir)
    if not base_job.manifest_path.exists():
        raise FileNotFoundError(f"{slug} has no base job; pick a concept first "
                                f"(`anim concept --unit {slug} --pick NN`)")
    manifest = base_job.read_manifest()
    final = (manifest.get("render") or {}).get("final") or "base_sprite.png"
    path = base_job.render_dir / final
    if not path.exists():
        raise FileNotFoundError(f"{slug}'s base render {path} is missing")
    return base_job, path, manifest


def derive_view_seed(slug: str, base_sha256: str, parts: str | None) -> int:
    """Stable for (unit, base render, parts phrase): a new base is a new set."""
    material = "|".join([slug, base_sha256, parts or "", "views"])
    return int(sha256_bytes(material.encode("utf-8"))[:13], 16) % render_stage.SEED_MAX


def build_view_workflow(prompt: str, *, seed: int, base_name: str, slug: str,
                        workflow_dir: Path | None = None) -> dict[str, Any]:
    """view_pass.json with this candidate's inputs patched in: the base as
    image1 (and the latent), the view's prompt, the seed, the output prefix."""
    workflow = comfy.load_workflow(VIEW_WORKFLOW, workflow_dir)
    comfy.patch(workflow, "base image", image=base_name)
    comfy.patch(workflow, "positive prompt", prompt=prompt)
    comfy.patch(workflow, "seed", seed=int(seed))
    comfy.patch(workflow, "output prefix",
                filename_prefix=f"anim_pipeline/{job_name(slug)}/views/{VIEW_WORKFLOW}")
    return workflow


def render_views(slug: str, *, views: str | Sequence[str] | None = None,
          count: int | None = None, seed: int | None = None,
          client: comfy.ComfyClient | None = None, workflow_dir: Path | None = None,
          lock_path: Path | None = None, work_dir: Path | None = None,
          timeout_s: float = 900.0, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Render `views` of the accepted base, N candidates each, through the
    edit path; write views/views.json and the page. Views not asked for this
    time keep their files and records; a view asked for again is redone."""
    names = config.concept_view_names(views)
    count = config.CONCEPT_VIEW_COUNT if count is None else int(count)
    if count < 1:
        raise ValueError("--count must be at least 1")
    base_job, base_path, base_manifest = base_render(slug, work_dir)
    base_sha = sha256_file(base_path)
    parts, _avoid = base_phrases(slug, work_dir)
    seed = derive_view_seed(slug, base_sha, parts) if seed is None else int(seed)

    client = client or comfy.ComfyClient()
    try:
        stats = client.system_stats()
    except comfy.ComfyError as exc:
        raise comfy.ComfyUnreachable(
            f"the GPU image service is not answering at {client.base_url}: {exc}") from exc

    out_dir = views_dir(slug, work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = views_path(slug, work_dir)
    record = read_json(path) if path.exists() else {}
    if record.get("base_sha256") not in (None, base_sha):
        # A new base: every old view was drawn from another character.
        for stale in out_dir.glob("*.png"):
            stale.unlink()
        record = {}
    rows: dict[str, Any] = dict(record.get("views") or {})
    base_name = client.upload_image(base_path, name=f"anim_{job_name(slug)}_view_base.png")
    models: list[dict[str, Any]] = []
    workflow_sha: str | None = None
    for offset, view in enumerate(names):
        prompt = config.concept_view_prompt(view, parts)
        for stale in out_dir.glob(f"{view}_[0-9][0-9].png"):
            stale.unlink()
        candidates: list[dict[str, Any]] = []
        for index in range(count):
            cand_seed = seed + offset * 100 + index
            workflow = build_view_workflow(prompt, seed=cand_seed, base_name=base_name,
                                           slug=slug, workflow_dir=workflow_dir)
            started = time.monotonic()
            log(f"concept views: {view} {index:02d} (seed {cand_seed}) ...")
            prompt_id = client.post_prompt(workflow)
            entry = client.wait(prompt_id, timeout_s=timeout_s)
            images = [i for i in client.outputs(entry) if i.type == "output"]
            if not images:
                raise comfy.ComfyError(f"job {prompt_id} produced no images")
            png = client.view(images[0])
            file = write_bytes_in_place(out_dir / f"{view}_{index:02d}.png", png)
            elapsed = execution_seconds(entry) or (time.monotonic() - started)
            log(f"  -> {file.name}  {elapsed:.1f} s")
            candidates.append({"index": index, "file": file.name, "seed": cand_seed,
                               "prompt_id": prompt_id, "seconds": round(elapsed, 1),
                               "sha256": sha256_bytes(png),
                               "workflow_sha256": comfy.workflow_sha256(workflow),
                               "embedded_prompt": render_stage.prompt_chunk(png) is not None})
            if workflow_sha is None:
                unseeded = comfy.patch(copy.deepcopy(workflow), "seed", seed=0)
                workflow_sha = comfy.workflow_sha256(unseeded)
                models = [m.as_sidecar_entry()
                          for m in models_lock.for_workflow(workflow, lock_path)]
        # A view rendered again is a new set: its old pick (if any) is gone
        # with its files; the base keeps what --pick-view copied there.
        rows[view] = {"view": view, "yaw": config.CONCEPT_VIEWS[view]["yaw"],
                      "prompt": prompt, "candidates": candidates, "pick": None}
    record = {
        "schema": VIEWS_SCHEMA, "slug": slug,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "base": {"job": base_job.name, "file": str(base_path), "sha256": base_sha,
                 "candidate": (base_manifest.get("concept") or {}).get("candidate")},
        "base_sha256": base_sha, "parts_phrase": parts, "seed": seed, "count": count,
        "workflow": f"{VIEW_WORKFLOW}.json", "workflow_sha256": workflow_sha or record.get("workflow_sha256"),
        "comfyui": render_stage.build_provenance(client.base_url, stats, seed, [])["comfyui"],
        "models": models or record.get("models") or [],
        "views": {v: rows[v] for v in config.CONCEPT_VIEWS if v in rows},
    }
    write_json_in_place(path, record)
    record["page"] = str(build_views_page(record, views_page_dir(slug, work_dir),
                                          base_path=base_path, out_dir=out_dir))
    return record


def parse_view_picks(specs: Sequence[str]) -> dict[str, int]:
    """`["back=01", "left=0"]` -> {"back": 1, "left": 0}; refuses a bad spec."""
    picks: dict[str, int] = {}
    for spec in specs:
        view, sep, number = str(spec).partition("=")
        view = view.strip().lower()
        if not sep or view not in config.CONCEPT_VIEWS or not number.strip().isdigit():
            raise ValueError(f"--pick-view takes <view>=NN with a view from "
                             f"{list(config.CONCEPT_VIEWS)}, not {spec!r}")
        picks[view] = int(number)
    if not picks:
        raise ValueError("--pick-view needs at least one <view>=NN")
    return picks


def pick_view(slug: str, picks: dict[str, int], *, provisional: bool = False,
              work_dir: Path | None = None) -> dict[str, Any]:
    """Copy the chosen candidates into the base as render/view_<view>.png,
    record them in the base's `concept.views` and in the curation record's
    `views` step (accepted; `provisional` records a pick made for a rerun
    that the owner's own --pick-view supersedes, and leaves the step pending)."""
    path = views_path(slug, work_dir)
    if not path.exists():
        raise FileNotFoundError(f"{slug} has no views yet; run `anim concept --unit {slug} "
                                f"--views` first")
    record = read_json(path)
    base_job, base_path, base_manifest = base_render(slug, work_dir)
    if record.get("base_sha256") != sha256_file(base_path):
        raise ValueError(f"{slug}'s views were drawn from an earlier base; run "
                         f"`anim concept --unit {slug} --views` again")
    block = dict(base_manifest.get("concept") or {})
    stored = dict(block.get("views") or {})
    written: dict[str, Any] = {}
    for view, index in picks.items():
        row = (record.get("views") or {}).get(view)
        if row is None:
            raise ValueError(f"no view {view!r} in {path}; rendered: "
                             f"{list(record.get('views') or {})}")
        candidate = next((c for c in row["candidates"] if c["index"] == index), None)
        if candidate is None:
            raise ValueError(f"view {view} has no candidate {index:02d}; it has "
                             f"{[c['index'] for c in row['candidates']]}")
        source = views_dir(slug, work_dir) / candidate["file"]
        if not source.exists():
            raise FileNotFoundError(f"{source} is missing; run `anim concept --views` again")
        png = source.read_bytes()
        out = write_bytes_in_place(base_job.render_dir / VIEW_FILE.format(view=view), png)
        entry = {"view": view, "yaw": row["yaw"], "file": out.name, "candidate": index,
                 "source": candidate["file"], "seed": candidate["seed"],
                 "sha256": sha256_bytes(png), "prompt": row["prompt"],
                 "workflow_sha256": candidate.get("workflow_sha256"),
                 "provisional": bool(provisional),
                 "picked": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        stored[view] = entry
        written[view] = entry
        row["pick"] = {"index": index, "file": candidate["file"], "sha256": entry["sha256"],
                       "provisional": bool(provisional), "picked": entry["picked"]}
    block["views"] = {v: stored[v] for v in config.CONCEPT_VIEWS if v in stored}
    base_job.update_manifest(concept=block)
    if base_job.provenance_path.exists():
        provenance = read_json(base_job.provenance_path)
        provenance["concept"] = {**(provenance.get("concept") or {}), "views": block["views"]}
        write_json_in_place(base_job.provenance_path, provenance)
    write_json_in_place(path, record)
    states = curation.set_views(slug, {v: ("pending" if provisional else "accepted")
                                       for v in written}, work_dir)
    page = build_views_page(record, views_page_dir(slug, work_dir), base_path=base_path,
                            out_dir=views_dir(slug, work_dir))
    return {"base_job": base_job.name, "views": written, "curation": states, "page": str(page)}


# ---------------------------------------------------------------------------
# The views page
# ---------------------------------------------------------------------------
_VIEWS_STYLE = """
:root { color-scheme: dark; }
body { background: #14161b; color: #d8dee9; font: 14px/1.45 -apple-system, BlinkMacSystemFont,
       "Segoe UI", sans-serif; margin: 0; padding: 24px; max-width: 1800px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 8px; color: #9fb0c9; border-top: 1px solid #2a2f3a; padding-top: 12px; }
p.caption, .note { color: #8b96a5; }
code { color: #cdd; }
.row { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
figure { margin: 0; background: #1c1f26; border: 1px solid #2a2f3a; border-radius: 6px; padding: 8px; }
figure.base { border-color: #4a5a7a; }
figure.picked { border-color: #8fd18f; }
figure img { width: 256px; height: 256px; object-fit: contain; background: #fff; display: block;
             image-rendering: pixelated; }
figcaption { color: #8b96a5; font-size: 11px; margin-top: 6px; max-width: 256px; }
.ok { color: #8fd18f; } .warn { color: #e0b060; }
footer { margin-top: 32px; color: #6b7686; font-size: 12px; }
"""


def build_views_page(record: dict[str, Any], page_dir: Path, *, base_path: Path,
                     out_dir: Path) -> Path:
    """The base beside each view's candidates; every write truncates in place."""
    page_dir = Path(page_dir)
    img_dir = page_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    write_bytes_in_place(img_dir / "base.png", base_path.read_bytes())
    slug = record["slug"]
    sections = []
    for view, row in (record.get("views") or {}).items():
        figures = [f'<figure class="base"><img src="img/base.png" alt="the base">'
                   f'<figcaption>the base (front), <code>{Path(base_path).name}</code></figcaption></figure>']
        pick = row.get("pick") or {}
        for c in row["candidates"]:
            src = out_dir / c["file"]
            if src.exists():
                write_bytes_in_place(img_dir / c["file"], src.read_bytes())
            picked = pick.get("index") == c["index"]
            tag = (' <span class="ok">picked</span>' + (' <span class="warn">(provisional)</span>'
                                                       if pick.get("provisional") else "")) if picked else ""
            figures.append(
                f'<figure class="{"picked" if picked else ""}" id="{view}_{c["index"]:02d}">'
                f'<img src="img/{c["file"]}" alt="{view} {c["index"]:02d}">'
                f'<figcaption><b>{view} {c["index"]:02d}</b>{tag} · seed {c["seed"]} · {c.get("seconds")} s</figcaption></figure>')
        sections.append(f'<h2 id="{view}">{view} — yaw {row["yaw"]}</h2>'
                        f'<p class="caption">{row["prompt"]}</p><div class="row">{"".join(figures)}</div>')
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>the views — {slug}</title><style>{_VIEWS_STYLE}</style></head>
<body>
<h1>The views — {slug}, {len(record.get('views') or {})} of {len(config.CONCEPT_VIEWS)} views</h1>
<p class="caption">{record.get('created', '')} · Qwen-Image-Edit 2509 (<code>{record.get('workflow')}</code>) with the base as image1 ·
seed <b>{record.get('seed')}</b> (+100 per view, +1 per candidate) · parts phrase: <i>{record.get('parts_phrase') or '—'}</i> ·
base <code>{Path(record['base']['file']).name}</code> sha256 {str(record.get('base_sha256') or '')[:12]}.
Choose with <code>anim concept --unit {slug} --pick-view back=NN left=NN right=NN</code>; the turnaround then hands each direction its nearest view
(<code>config.TURNAROUND_REFERENCE_BY_DIR</code>). A view marked <span class="warn">provisional</span> was picked for a rerun and your <code>--pick-view</code> supersedes it.</p>
{''.join(sections)}
<footer>Generated by <code>anim concept --unit {slug} --views</code>; files in <code>work/{slug}_concept/views/</code>, record <code>views.json</code>.</footer>
</body>
</html>
"""
    return write_text_in_place(page_dir / "index.html", html)


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
    parser.add_argument("--parts", default=None, metavar="PHRASE",
                        help="the parts every clip must keep, e.g. \"closed great helm, "
                             "kite shield, longsword, dark cloak\" - stored on the base "
                             "with --pick, or alone (default at --pick: the description)")
    parser.add_argument("--avoid", default=None, metavar="PHRASE",
                        help="what the clip negative names for this unit, e.g. \"bare "
                             "head, uncovered face\"")
    parser.add_argument("--views", nargs="?", const="", default=None, metavar="LIST",
                        help="render views of the accepted base through Qwen-Image-Edit: "
                             "no value = back,left,right; a comma list; or 'all' "
                             f"({', '.join(config.CONCEPT_VIEWS)}); --count per view "
                             f"(default {config.CONCEPT_VIEW_COUNT})")
    parser.add_argument("--pick-view", dest="pick_view", nargs="+", default=None,
                        metavar="VIEW=NN",
                        help="copy those view candidates into the base "
                             "(render/view_<view>.png) for the turnaround to use")
    parser.add_argument("--provisional", action="store_true",
                        help="with --pick-view: record the pick as provisional (made "
                             "for a rerun; the owner's --pick-view supersedes it)")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait for one /prompt job")


def run_stage(args: argparse.Namespace) -> int:
    if args.pick is not None:
        result = pick(args.slug, args.pick, parts=args.parts, avoid=args.avoid)
        candidate = result["candidate"]
        print(f"concept: candidate {candidate['index']:02d} ({candidate['source']}"
              f"{', seed ' + str(candidate['seed']) if candidate['seed'] is not None else ''}) "
              f"-> {result['file']}")
        if result["replaced"]:
            print(f"  replaced the base render {result['replaced']} as "
                  f"{result['base_job']}'s render.final (the file is still there)")
        print(f"  parts phrase: {result['parts_phrase']!r}"
              f"{'; avoid: ' + repr(result['avoid_phrase']) if result['avoid_phrase'] else ''}"
              f"  (--parts / --avoid change them; the views are cleared)")
        print(f"  base gate accepted for {args.slug} (turnaround "
              f"{result['gates']['turnaround']}) - work/{args.slug}_curation.json")
        print(f"  next: anim post {result['base_job']} [--parts], then "
              f"anim publish {result['base_job']}")
        print(f"  the turnaround is drawn from the base: anim concept --unit {args.slug} --views, "
              f"--pick-view, then anim turnaround {args.slug} --force re-renders all eight facings")
        return 0
    if args.pick_view is not None:
        try:
            picks = parse_view_picks(args.pick_view)
            result = pick_view(args.slug, picks, provisional=args.provisional)
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        for view, entry in result["views"].items():
            print(f"concept: view {view} <- {entry['source']} (seed {entry['seed']}) -> "
                  f"work/{result['base_job']}/render/{entry['file']}"
                  f"{'  [provisional]' if entry['provisional'] else ''}")
        print(f"  curation views: " + ", ".join(f"{v}={s}" for v, s in result["curation"].items()))
        print(f"  page   -> {result['page']}  (cp -r {Path(result['page']).parent}/. {VIEWS_VISUALS_TARGET}/)")
        print(f"  then: anim turnaround {args.slug} --force")
        return 0
    if args.views is not None:
        client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
        try:
            record = render_views(args.slug, views=args.views, count=args.count,
                                  seed=args.seed, client=client, timeout_s=args.timeout)
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        print(f"concept: {len(record['views'])} view(s) of {args.slug}'s base, seed {record['seed']} "
              f"(ComfyUI {record['comfyui']['version']})")
        for view, row in record["views"].items():
            for c in row["candidates"]:
                print(f"  {c['file']:<22} seed {c['seed']:<18} {c['seconds']:.1f} s")
        print(f"  page   -> {record['page']}")
        print(f"  publish: mkdir -p {VIEWS_VISUALS_TARGET} && cp -r {Path(record['page']).parent}/. "
              f"{VIEWS_VISUALS_TARGET}/")
        print(f"  then:    {VIEWS_VISUALS_URL}")
        print(f"  choose:  anim concept --unit {args.slug} --pick-view "
              + " ".join(f"{v}=NN" for v in record["views"]))
        return 0
    if args.parts is not None or args.avoid is not None:
        try:
            block = set_phrases(args.slug, parts=args.parts, avoid=args.avoid)
        except FileNotFoundError as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        print(f"concept: {args.slug}'s base now carries parts phrase {block['parts_phrase']!r}"
              f"{'; avoid ' + repr(block['avoid_phrase']) if block['avoid_phrase'] else ''}")
        print(f"  every clip of {args.slug} names them from now on; anim turnaround {args.slug} "
              f"--force re-renders the eight facings with them")
        return 0
    if not args.prompt:
        raise SystemExit("anim concept: --prompt is required (or --pick, --views, "
                         "--pick-view, --parts to work on the base)")
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
