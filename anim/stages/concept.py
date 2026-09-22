"""Stage: `anim concept` - text or sketch -> N candidate base characters -> pick

The character concept feature (sprite-studio.md S2 "Base character"): the first
thing the studio asks for is *who* the character is, and the answer is a
front-facing base image the owner chooses from several - generated ones and,
side by side, the art that already exists.

    anim concept --unit knight --prompt "a plate-armoured knight with a kite shield"
    anim concept --unit knight --prompt "..." --sketch scans/knight.png --count 6
    anim concept --unit knight --prompt "..." --existing            # the published sprite as a candidate
    anim concept --unit knight --upload scans/knight_base.png        # the owner's own art as a candidate
    anim concept --unit knight --pick 02                             # or --pick existing
    anim concept --unit knight --pick 02 --parts "closed great helm, kite shield, longsword, dark cloak" \\
                                         --avoid "bare head, uncovered face"
    anim concept --unit knight --parts "..." [--avoid "..."]        # set the phrases on the base without a new pick
    anim concept --unit knight --views                                # back, left, right, top of the accepted base
    anim concept --unit knight --views all --count 2                  # + the four obliques, 2 candidates each
    anim concept --unit knight --views left --view-prompt "..."       # one view again, with the owner's sentence
    anim concept --unit knight --img2img base:02 --strength 0.5       # rework one attempt, keep its composition
    anim concept --unit knight --edit left:01 --instruction "make the helmet red"
    anim concept --unit knight --prompt "..." --model FILE --lora FILE:0.7 --lora FILE2:0.4
    anim concept --unit knight --pick-view back=01 left=00 top=02     # the views the turnaround will use

    work/<slug>_concept/
      job.json               every base attempt: prompt, seed, model, LoRAs, source, parent
      sketch.png             the control image actually sent (white strokes on black)
      candidates/NN.png      one per base attempt; never deleted (--fresh starts over)
      uploads/<stamp>.png    the owner's uploads as given, before fitting
      views/<view>_NN.png    the multi-view reference feature: the base from another side
      views/views.json       per view: prompt, every attempt, the pick
    work/_views/<slug>/index.html   the views page (the base beside each view's attempts)

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

ONE `/prompt` PER ATTEMPT, `batch_size` 1. Each attempt has its own seed, so any
one of them can be re-made alone from its row without regenerating its
neighbours - the reproducibility rule every other stage keeps.

EVERY ATTEMPT IS KEPT (the concept generation feature, schema `anim.concept/2`).
A run APPENDS to the job: the owner presses Generate again and the earlier
attempts are still there to compare against; `--fresh` is the old wipe. An
attempt row says where it came from - `txt2img`, `upload`, `existing`,
`img2img` (a `parent` attempt redrawn at `strength`, classic denoise < 1 on the
Z-Image stack, `gpu/workflows/img2img_pass.json`) or `edit` (a `parent`
attempt through Qwen-Image-Edit with an `instruction`) - and what it was made
with: its own prompt, seed, checkpoint and LoRA list. A view attempt is the
same row under `views/views.json`; `base:02` / `left:01` name one either way.

THE MODEL AND THE LoRAs are chosen per call (`--model`, `--lora FILE[:STRENGTH]`
repeatable) or per unit (`job.json["defaults"][family]`, the studio's "save as
unit default"), and only from `gpu/models.registry.tsv` for the family the
op uses - a Z-Image graph cannot load a Qwen checkpoint, and the registry says
no before ComfyUI would. The seed is random unless `--seed` is given (the
studio's "lock seed" resends the last one).

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
attempts per view (`--count`), each with its own seed, into
`views/<view>_NN.png` and `views/views.json`; a view asked for again gets
MORE attempts, with `--view-prompt` in place of the config sentence when the
owner has a better one. The top-down view (#285) is one of them. The
alternative - Z-Image + the Fun ControlNet with a rig-rendered depth of the
mannequin at that yaw - would hold the pose but carry identity in the prompt
only, which is the failure being fixed; it is the fallback if the edit path
turns out not to keep the character.

`--pick-view back=01 ...` copies the chosen attempts into the base job as
`render/view_<view>.png` and records them in the base's `concept.views`
(file, candidate, seed, sha256, `provisional` false) and in the curation
record's `views` step (accepted); `anim turnaround` then hands each direction
the nearest view (`config.TURNAROUND_REFERENCE_BY_DIR`). `--pick` of a new
base clears the views: they were drawn from the old one. The views wait
behind the base gate (`curation.views_blocked`): a view of a base nobody has
accepted is GPU time spent on a character that may not survive.

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
attempt, so a pick can be changed by picking again.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import secrets
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

CONCEPT_SCHEMA = "anim.concept/2"
VIEWS_SCHEMA = "anim.concept-views/2"
WORKFLOW = "concept_pass"
IMG2IMG_WORKFLOW = "img2img_pass"
VIEW_WORKFLOW = "view_pass"
PICK_FILE = "concept_pick.png"
VIEW_FILE = "view_{view}.png"
EXISTING = "existing"
BASE_TARGET = "base"
# Where an attempt came from. The GPU ones are the three that carry a
# workflow; `upload` and `existing` are the owner's files fitted to the cell.
SOURCES = ("txt2img", "upload", "existing", "img2img", "edit")
GPU_SOURCES = ("txt2img", "img2img", "edit")
VIEWS_PAGE_ROOT = "_views"
VIEWS_VISUALS_TARGET = "~/.cache/claude/visuals/sprite-studio/views"
VIEWS_VISUALS_URL = "http://192.168.1.158:8090/sprite-studio/views/"
# The white the base pass and the strip reference put behind a figure.
BACKGROUND = (255, 255, 255)


def job_name(slug: str) -> str:
    return f"{slug}_concept"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def random_seed() -> int:
    """A fresh seed: pressing Generate again must not redraw the same set."""
    return secrets.randbelow(render_stage.SEED_MAX)


def derive_seed(slug: str, prompt: str, sketch_sha256: str | None = None) -> int:
    """Stable for (unit, prompt, sketch) - what a `--fresh` run without
    `--seed` uses, so the first set of a unit reproduces."""
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
# The model and the LoRAs of one call
# ---------------------------------------------------------------------------
def family_of(source: str) -> str:
    """`txt2img` / `img2img` -> zimage; `views` / `edit` -> qwen-edit."""
    try:
        return config.CONCEPT_FAMILIES[source]
    except KeyError:
        raise ValueError(f"no model family for {source!r}") from None


def resolve_generation(family: str, *, model: str | None = None,
                       loras: Sequence[str | dict[str, Any]] | None = None,
                       lora_strength: float | None = None,
                       defaults: dict[str, Any] | None = None,
                       registry_path: Path | None = None,
                       lock_path: Path | None = None) -> dict[str, Any]:
    """`{"model": file, "loras": [{"file", "strength", "trigger"}]}` for one
    call: the flags first, then the unit's defaults for that family, then the
    workflow's own checkpoint and (for Z-Image) the pixel LoRA at
    `--lora-strength`. Explicit choices are checked against the registry."""
    unit_defaults = dict((defaults or {}).get(family) or {})
    if model is None:
        model = unit_defaults.get("model")
    resolved_model = models_lock.resolve_model(
        family, model, default=config.CONCEPT_MODELS[family],
        path=registry_path, lock_path=lock_path)
    if loras:
        resolved = models_lock.resolve_loras(family, list(loras), path=registry_path,
                                             lock_path=lock_path)
    elif lora_strength is not None:
        resolved = ([] if family != "zimage" or lora_strength <= 0 else
                    [{"file": config.CONCEPT_LORA["file"], "strength": float(lora_strength),
                      "trigger": config.CONCEPT_LORA["trigger"]}])
    elif unit_defaults.get("loras") is not None:
        resolved = models_lock.resolve_loras(family, list(unit_defaults["loras"]),
                                             path=registry_path, lock_path=lock_path)
    elif family == "zimage":
        resolved = [{"file": config.CONCEPT_LORA["file"],
                     "strength": float(config.CONCEPT_LORA["strength"]),
                     "trigger": config.CONCEPT_LORA["trigger"]}]
    else:
        resolved = []
    return {"model": resolved_model, "loras": resolved}


def set_defaults(slug: str, family: str, *, model: str | None,
                 loras: Sequence[str | dict[str, Any]] | None,
                 registry_path: Path | None = None, lock_path: Path | None = None,
                 work_dir: Path | None = None) -> dict[str, Any]:
    """The unit's default checkpoint + LoRAs for one family, validated and
    written to job.json["defaults"]; None for the model means the workflow's
    own. Creates the concept job when the unit has none yet."""
    if family not in models_lock.FAMILIES:
        raise ValueError(f"family {family!r} is not one of {models_lock.FAMILIES}")
    job = _open_job(slug, work_dir)
    manifest = job.read_manifest()
    resolved_model = models_lock.resolve_model(
        family, model, default="", path=registry_path, lock_path=lock_path) or None
    resolved_loras = models_lock.resolve_loras(family, list(loras or []),
                                               path=registry_path, lock_path=lock_path)
    defaults = dict(manifest.get("defaults") or {})
    defaults[family] = {"model": resolved_model,
                        "loras": [{"file": r["file"], "strength": r["strength"]}
                                  for r in resolved_loras]}
    manifest["defaults"] = defaults
    manifest["updated"] = _now()
    job.write_manifest(manifest)
    return defaults


# ---------------------------------------------------------------------------
# The job and its attempts
# ---------------------------------------------------------------------------
def empty_manifest(slug: str, cell: tuple[int, int]) -> dict[str, Any]:
    return {
        "schema": CONCEPT_SCHEMA, "slug": slug, "created": _now(), "updated": _now(),
        "description": None, "prompt": None, "seed": None, "last_seed": None,
        "count": 0, "cell": list(cell), "lora": None, "sketch": None,
        "workflow": f"{WORKFLOW}.json", "workflow_sha256": None, "comfyui": None,
        "models": [], "defaults": {}, "next_index": 0, "candidates": [], "pick": None,
    }


def migrate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """`anim.concept/1` -> `/2` in memory: `generated` becomes `txt2img`, the
    manifest-level prompt and LoRA move onto each row, `defaults` and
    `next_index` appear. A `/2` manifest comes back unchanged."""
    if manifest.get("schema") == CONCEPT_SCHEMA:
        manifest.setdefault("defaults", {})
        manifest.setdefault("next_index", _next_index(manifest.get("candidates") or []))
        return manifest
    lora = manifest.get("lora") or {}
    rows = []
    for row in manifest.get("candidates") or []:
        row = dict(row)
        source = row.get("source")
        row["source"] = "txt2img" if source == "generated" else source
        row.setdefault("prompt", manifest.get("prompt") if row["source"] == "txt2img" else None)
        row.setdefault("description", manifest.get("description")
                       if row["source"] == "txt2img" else None)
        row.setdefault("model", config.CONCEPT_MODELS["zimage"]
                       if row["source"] == "txt2img" else None)
        row.setdefault("loras", ([{"file": lora.get("file"), "strength": lora.get("strength"),
                                   "trigger": lora.get("trigger")}]
                                 if row["source"] == "txt2img" and lora.get("file")
                                 and (lora.get("strength") or 0) > 0 else []))
        row.setdefault("parent", None)
        row.setdefault("workflow", manifest.get("workflow")
                       if row["source"] == "txt2img" else None)
        row.setdefault("created", manifest.get("created"))
        rows.append(row)
    manifest = {**empty_manifest(manifest.get("slug", ""), tuple(manifest.get("cell") or config.CONCEPT_CELL)),
                **manifest, "schema": CONCEPT_SCHEMA, "candidates": rows}
    manifest.setdefault("last_seed", manifest.get("seed"))
    manifest["defaults"] = manifest.get("defaults") or {}
    manifest["next_index"] = _next_index(rows)
    return manifest


def _next_index(rows: Sequence[dict[str, Any]]) -> int:
    return (max(int(r["index"]) for r in rows) + 1) if rows else 0


def _open_job(slug: str, work_dir: Path | None, *, cell: tuple[int, int] | None = None) -> Job:
    """The concept job with a `/2` manifest on disk, created if missing."""
    job = Job.open(job_name(slug), work_dir)
    job.candidates_dir.mkdir(parents=True, exist_ok=True)
    if job.manifest_path.exists():
        before = job.read_manifest()
        if before.get("schema") == CONCEPT_SCHEMA and "defaults" in before:
            return job
        manifest = migrate_manifest(before)
    else:
        manifest = empty_manifest(slug, tuple(cell or config.CONCEPT_CELL))
    job.write_manifest(manifest)
    return job


def parse_ref(ref: str) -> tuple[str, int]:
    """`base:03` -> ("base", 3); `left:01` -> ("left", 1). Refuses the rest."""
    target, sep, number = str(ref or "").strip().lower().partition(":")
    if not sep or not number.strip().isdigit() or \
            (target != BASE_TARGET and target not in config.CONCEPT_VIEWS):
        raise ValueError(f"an attempt is named base:NN or <view>:NN with a view from "
                         f"{list(config.CONCEPT_VIEWS)}, not {ref!r}")
    return target, int(number)


def attempt_file(target: str, index: int) -> str:
    return f"{index:02d}.png" if target == BASE_TARGET else f"{target}_{index:02d}.png"


class Attempts:
    """The attempt rows of one target and where their files live: the base's
    in job.json + candidates/, a view's in views/views.json + views/. One
    shape for both so the ops (txt2img, img2img, edit, views) plan and render
    the same way whichever they land in."""

    def __init__(self, slug: str, target: str, work_dir: Path | None = None, *,
                 record: dict[str, Any] | None = None) -> None:
        self.slug = slug
        self.target = target
        self.work_dir = work_dir
        self.job = _open_job(slug, work_dir)
        if target == BASE_TARGET:
            self.dir = self.job.candidates_dir
            self.record = record if record is not None else self.job.read_manifest()
        else:
            if target not in config.CONCEPT_VIEWS:
                raise ValueError(f"no view {target!r}; one of {list(config.CONCEPT_VIEWS)}")
            self.dir = views_dir(slug, work_dir)
            self.dir.mkdir(parents=True, exist_ok=True)
            path = views_path(slug, work_dir)
            if record is not None:
                self.record = record      # a views record shared across views
            else:
                self.record = migrate_views(read_json(path)) if path.exists() else empty_views(slug)
            self.record["views"].setdefault(target, view_row(target, None))

    @property
    def rows(self) -> list[dict[str, Any]]:
        if self.target == BASE_TARGET:
            return self.record["candidates"]
        return self.record["views"][self.target]["candidates"]

    def get(self, index: int) -> dict[str, Any]:
        for row in self.rows:
            if int(row["index"]) == index:
                return row
        raise ValueError(f"no attempt {self.target}:{index:02d}; it has "
                         f"{[r['index'] for r in self.rows]}")

    def path(self, row: dict[str, Any]) -> Path:
        return self.dir / row["file"]

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        index = _next_index(self.rows)
        row = {"index": index, "file": attempt_file(self.target, index), **row}
        row.setdefault("created", _now())
        self.rows.append(row)
        if self.target == BASE_TARGET:
            self.record["next_index"] = index + 1
        return row

    def drop_unrendered(self) -> list[dict[str, Any]]:
        """Forget the GPU attempts that never produced a file - what a failed
        run planned. Their settings are the ones that failed, so keeping them
        would make the next run repeat the failure and the record claim
        attempts that do not exist. Returns the rows dropped."""
        dropped = [r for r in self.rows
                   if r.get("seconds") is None and r.get("source") in (*GPU_SOURCES, "views")
                   and not self.path(r).exists()]
        if not dropped:
            return []
        keep = [r for r in self.rows if r not in dropped]
        if self.target == BASE_TARGET:
            self.record["candidates"] = keep
            self.record["next_index"] = _next_index(keep)
        else:
            self.record["views"][self.target]["candidates"] = keep
        return dropped

    def save(self) -> None:
        self.record["updated"] = _now()
        if self.target == BASE_TARGET:
            self.job.write_manifest(self.record)
        else:
            write_json_in_place(views_path(self.slug, self.work_dir), self.record)


def attempt_row(source: str, *, prompt: str | None, seed: int | None,
                generation: dict[str, Any] | None, parent: tuple[str, int] | None = None,
                description: str | None = None, workflow: str | None = None,
                **extra: Any) -> dict[str, Any]:
    """One attempt as recorded - what it was made from and with, before the
    GPU fills in the result (`seconds`, `prompt_id`, `sha256`...)."""
    generation = generation or {"model": None, "loras": []}
    return {
        "source": source, "prompt": prompt, "description": description, "seed": seed,
        "model": generation["model"],
        "loras": [{"file": r["file"], "strength": r["strength"]} for r in generation["loras"]],
        "parent": ({"target": parent[0], "index": parent[1]} if parent else None),
        "workflow": f"{workflow}.json" if workflow else None,
        "workflow_sha256": None, "prompt_id": None, "seconds": None, "sha256": None,
        **extra,
    }


# ---------------------------------------------------------------------------
# Lay out the job
# ---------------------------------------------------------------------------
def prepare(slug: str, *, description: str, count: int | None = None,
            seed: int | None = None, sketch: Path | None = None,
            existing: Path | str | None = None,
            lora_strength: float | None = None,
            sketch_strength: float | None = None,
            cell: tuple[int, int] | None = None,
            model: str | None = None, loras: Sequence[str | dict[str, Any]] | None = None,
            fresh: bool = False, registry_path: Path | None = None,
            lock_path: Path | None = None,
            assets_dir: Path | None = None, work_dir: Path | None = None) -> Job:
    """Plan `count` txt2img attempts of the base (and the existing art, when
    asked) onto work/<slug>_concept/: job.json, the control image, the rows.
    Appends to what is there; `fresh` wipes first. Nothing here touches the
    GPU; `render_pending` fills the files in.
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
    if fresh:
        job = Job.open(job_name(slug), work_dir)
        if job.manifest_path.exists():
            job.manifest_path.unlink()
        if job.candidates_dir.is_dir():
            for stale in job.candidates_dir.glob("*.png"):
                stale.unlink()
    store = Attempts(slug, BASE_TARGET, work_dir)
    manifest = store.record
    if fresh or not manifest.get("candidates"):
        manifest["cell"] = list(cell)
    elif tuple(manifest.get("cell") or cell) != cell:
        raise ValueError(f"the job's cell is {manifest['cell']}, not {list(cell)}; "
                         f"--fresh to start over at the new size")
    job = store.job

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

    if seed is None:
        seed = derive_seed(slug, prompt, sketch_sha) if fresh else random_seed()
    seed = int(seed)
    generation = resolve_generation("zimage", model=model, loras=loras,
                                    lora_strength=lora_strength,
                                    defaults=manifest.get("defaults"),
                                    registry_path=registry_path, lock_path=lock_path)
    art = existing_art(slug, existing, assets_dir)
    if art is not None:
        source, image = art
        row = store.append(attempt_row(EXISTING, prompt=None, seed=None, generation=None,
                                       **{"from": str(source)}))
        candidate_from_art(image, cell).save(store.path(row))
    for offset in range(count):
        store.append(attempt_row("txt2img", prompt=prompt, seed=seed + offset,
                                 generation=generation, description=description.strip(),
                                 workflow=WORKFLOW, sketch=sketch_block))

    pixel = next((r for r in generation["loras"]
                  if r["file"] == config.CONCEPT_LORA["file"]), None)
    manifest |= {
        "description": description.strip(), "prompt": prompt,
        "seed": seed, "last_seed": seed, "count": count,
        "lora": {"repo": config.CONCEPT_LORA["repo"], "file": config.CONCEPT_LORA["file"],
                 "trigger": config.CONCEPT_LORA["trigger"],
                 "strength": float(pixel["strength"]) if pixel else 0.0},
        "sketch": sketch_block, "workflow": f"{WORKFLOW}.json",
    }
    store.save()
    return job


def register_upload(slug: str, source: Path | bytes, *, name: str | None = None,
                    cell: tuple[int, int] | None = None,
                    work_dir: Path | None = None) -> dict[str, Any]:
    """The owner's own art as a base attempt (`source: upload`): the file as
    given under uploads/, fitted into the cell as candidates/NN.png. No GPU,
    no prompt needed. Returns the row."""
    store = Attempts(slug, BASE_TARGET, work_dir)
    cell = tuple(cell or store.record.get("cell") or config.CONCEPT_CELL)
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        given = name or "upload.png"
    else:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"no file at {source}")
        data = source.read_bytes()
        given = name or source.name
    from io import BytesIO
    try:
        with Image.open(BytesIO(data)) as raw:
            raw.load()
            image = raw.convert("RGBA")
    except Exception as exc:  # noqa: BLE001 - PIL raises a zoo
        raise ValueError(f"{given} is not an image PIL can read: {exc}") from None
    uploads = store.job.root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    kept = write_bytes_in_place(uploads / f"{stamp}_{Path(given).stem}.png", data)
    row = store.append(attempt_row("upload", prompt=None, seed=None, generation=None,
                                   **{"from": str(kept), "name": given}))
    candidate_from_art(image, cell).save(store.path(row))
    row["sha256"] = sha256_file(store.path(row))
    store.save()
    return row


def plan_rework(slug: str, ref: str, *, op: str, count: int | None = None,
                seed: int | None = None, strength: float | None = None,
                instruction: str | None = None, prompt: str | None = None,
                model: str | None = None, loras: Sequence[str | dict[str, Any]] | None = None,
                lora_strength: float | None = None, registry_path: Path | None = None,
                lock_path: Path | None = None, work_dir: Path | None = None
                ) -> tuple[Attempts, list[dict[str, Any]]]:
    """`img2img` or `edit` of the attempt `ref` names: `count` new rows on the
    same target with `parent` = ref. img2img keeps the parent's prompt unless
    `prompt` is given (a view's parent prompt is the view sentence); an edit's
    prompt is the instruction wrapped by `config.concept_edit_prompt`."""
    if op not in ("img2img", "edit"):
        raise ValueError(f"op is img2img or edit, not {op!r}")
    target, index = parse_ref(ref)
    store = Attempts(slug, target, work_dir)
    parent = store.get(index)
    if not store.path(parent).exists():
        raise FileNotFoundError(f"{ref} has no file at {store.path(parent)}")
    count = 1 if count is None else int(count)
    if count < 1:
        raise ValueError("--count must be at least 1")
    seed = random_seed() if seed is None else int(seed)
    family = family_of(op)
    defaults = store.job.read_manifest().get("defaults")
    generation = resolve_generation(family, model=model, loras=loras,
                                    lora_strength=lora_strength, defaults=defaults,
                                    registry_path=registry_path, lock_path=lock_path)
    parts, _avoid = base_phrases(slug, work_dir)
    description = store.job.read_manifest().get("description") or "character"
    if op == "img2img":
        strength = float(config.CONCEPT_IMG2IMG_STRENGTH if strength is None else strength)
        if not 0 < strength <= 1:
            raise ValueError(f"--strength is in (0, 1], not {strength}")
        if prompt and prompt.strip():
            text = prompt.strip()
        elif target == BASE_TARGET:
            # The parent's own prompt when it has a Z-Image one; a redraw of
            # an upload is asked for as the unit's description.
            text = (parent.get("prompt") if parent.get("source") in ("txt2img", "img2img")
                    else None) or config.concept_prompt(description)
        else:
            # A view's prompt is written for Qwen ("from image1..."); Z-Image
            # gets the description seen from that side instead.
            text = config.concept_view_img2img_prompt(description, target, parts)
        extra = {"strength": strength, "instruction": None}
        workflow = IMG2IMG_WORKFLOW
    else:
        text = config.concept_edit_prompt(instruction or "", parts)
        extra = {"strength": None, "instruction": " ".join(str(instruction).split())}
        workflow = VIEW_WORKFLOW
    planned = [store.append(attempt_row(op, prompt=text, seed=seed + i, generation=generation,
                                        parent=(target, index), workflow=workflow, **extra))
               for i in range(count)]
    if target == BASE_TARGET:
        store.record["last_seed"] = seed
    else:
        store.record["seed"] = seed
    store.save()
    return store, planned


# ---------------------------------------------------------------------------
# The GPU: one /prompt per attempt
# ---------------------------------------------------------------------------
def build_workflow(manifest: dict[str, Any], *, seed: int, sketch_name: str | None,
                   job: Job, workflow_dir: Path | None = None,
                   attempt: dict[str, Any] | None = None) -> dict[str, Any]:
    """concept_pass.json with this attempt's inputs patched in.

    Pure apart from the file read, so a test can see exactly what goes over
    the wire: the prompt, the seed, the checkpoint, the LoRA files +
    strengths, the cell, and - only with a sketch - the control image and
    its strength. `attempt` carries the row's own prompt/model/LoRAs; without
    one the manifest's last-used values apply (the `/1` call shape).
    """
    workflow = comfy.load_workflow(WORKFLOW, workflow_dir)
    width, height = manifest["cell"]
    attempt = attempt or {}
    prompt = attempt.get("prompt") or manifest["prompt"]
    comfy.patch(workflow, "positive prompt", text=prompt)
    comfy.patch(workflow, "cell size", width=int(width), height=int(height))
    comfy.patch(workflow, "seed", seed=int(seed))
    comfy.patch(workflow, "output prefix",
                filename_prefix=f"anim_pipeline/{job.name}/{WORKFLOW}")
    if attempt.get("model"):
        comfy.patch_checkpoint(workflow, attempt["model"])
    if "loras" in attempt:
        loras = attempt["loras"]
    else:
        lora = manifest["lora"]
        loras = [{"file": lora["file"], "strength": lora["strength"]}] \
            if lora and lora["strength"] > 0 else []
    comfy.chain_loras(workflow, "Pixel-Art LoRA", loras)
    sketch = attempt.get("sketch", manifest.get("sketch"))
    if sketch and sketch_name:
        comfy.patch(workflow, "sketch", image=sketch_name)
        comfy.patch(workflow, "control scale", strength=float(sketch["strength"]))
    else:
        # No sketch: the ControlNet apply node goes, and with it the Union
        # model loader and the LoadImage that only it consumed.
        comfy.bypass(workflow, "control scale", "model")
        comfy.prune_orphans(workflow)
    return workflow


def build_img2img_workflow(attempt: dict[str, Any], *, source_name: str, job: Job,
                           workflow_dir: Path | None = None) -> dict[str, Any]:
    """img2img_pass.json: the parent attempt as the latent, redrawn at the
    row's `strength` (KSampler denoise) with its prompt, checkpoint, LoRAs."""
    workflow = comfy.load_workflow(IMG2IMG_WORKFLOW, workflow_dir)
    comfy.patch(workflow, "source image", image=source_name)
    comfy.patch(workflow, "positive prompt", text=attempt["prompt"])
    comfy.patch(workflow, "seed", seed=int(attempt["seed"]), denoise=float(attempt["strength"]))
    comfy.patch(workflow, "output prefix",
                filename_prefix=f"anim_pipeline/{job.name}/{IMG2IMG_WORKFLOW}")
    if attempt.get("model"):
        comfy.patch_checkpoint(workflow, attempt["model"])
    comfy.chain_loras(workflow, "Pixel-Art LoRA", attempt.get("loras") or [])
    return workflow


def build_view_workflow(prompt: str, *, seed: int, base_name: str, slug: str,
                        workflow_dir: Path | None = None,
                        model: str | None = None,
                        loras: Sequence[dict[str, Any]] | None = None,
                        prefix: str = f"views/{VIEW_WORKFLOW}") -> dict[str, Any]:
    """view_pass.json with this attempt's inputs patched in: the source as
    image1 (and the latent), the prompt, the seed, the checkpoint, the LoRAs
    (the optional node is bypassed with none), the output prefix."""
    workflow = comfy.load_workflow(VIEW_WORKFLOW, workflow_dir)
    comfy.patch(workflow, "base image", image=base_name)
    comfy.patch(workflow, "positive prompt", prompt=prompt)
    comfy.patch(workflow, "seed", seed=int(seed))
    comfy.patch(workflow, "output prefix",
                filename_prefix=f"anim_pipeline/{job_name(slug)}/{prefix}")
    if model:
        comfy.patch_checkpoint(workflow, model)
    comfy.chain_loras(workflow, "optional LoRA", list(loras or []))
    return workflow


def _client(client: comfy.ComfyClient | None) -> tuple[comfy.ComfyClient, dict[str, Any]]:
    client = client or comfy.ComfyClient()
    try:
        stats = client.system_stats()
    except comfy.ComfyError as exc:
        raise comfy.ComfyUnreachable(
            f"the GPU image service is not answering at {client.base_url}: "
            f"{exc}") from exc
    return client, stats


def check_installed(client: comfy.ComfyClient, family: str, *, model: str | None,
                    loras: Sequence[dict[str, Any]] | None) -> None:
    """Refuse a checkpoint or LoRA the GPU service does not have, BEFORE the
    first `/prompt`.

    ComfyUI validates every combo input against the loader's file list, so a
    file that is registered but not installed - the Qwen pixel LoRA is exactly
    that, see the KNOWN DISCREPANCY in `gpu/models.lock` - comes back as a 400
    on the whole graph naming only a node id, after the inputs have been
    uploaded. The registry says what may be asked for; this says what is
    actually there. A node class ComfyUI does not have answers with an empty
    list and is not judged - the graph's own validation is the backstop.
    """
    missing: list[str] = []
    checkpoint = comfy.LOADER_INPUTS.get((family, "checkpoint"))
    if model and checkpoint:
        available = client.combo_values(*checkpoint)
        if available and model not in available:
            missing.append(f"checkpoint {model!r}")
    lora_input = comfy.LOADER_INPUTS.get((family, "lora"))
    if loras and lora_input:
        available = client.combo_values(*lora_input)
        if available:
            missing.extend(f"LoRA {row['file']!r}" for row in loras
                           if row.get("file") and row["file"] not in available)
    if not missing:
        return
    raise comfy.ComfyError(
        f"the GPU service at {client.base_url} does not have the "
        f"{' and the '.join(missing)} this run asks for. Install it "
        f"(`gpu/download_models.sh --only <part of the filename>`) or choose "
        f"another in the Concept tab's picker - `gpu/models.registry.tsv` lists "
        f"what may be asked for, `gpu/models.lock` whether it installs "
        f"(the Qwen pixel LoRA is registered and deliberately NOT installed)")


def _run_one(client: comfy.ComfyClient, workflow: dict[str, Any], *, timeout_s: float
             ) -> tuple[str, dict[str, Any], bytes, float]:
    started = time.monotonic()
    prompt_id = client.post_prompt(workflow)
    entry = client.wait(prompt_id, timeout_s=timeout_s)
    images = [i for i in client.outputs(entry) if i.type == "output"]
    if not images:
        raise comfy.ComfyError(f"job {prompt_id} produced no images")
    png = client.view(images[0])
    return prompt_id, entry, png, execution_seconds(entry) or (time.monotonic() - started)


def render_pending(store: Attempts, *, client: comfy.ComfyClient | None = None,
                   workflow_dir: Path | None = None, lock_path: Path | None = None,
                   timeout_s: float = 900.0, log: Callable[[str], None] = lambda _: None
                   ) -> list[dict[str, Any]]:
    """Render every attempt of the store that has no result yet, one /prompt
    each, by its source: txt2img (the concept pass), img2img (the parent
    redrawn), edit (the parent through the edit path). The view of a base
    (`views`) is `render_views`, which plans rows of source `edit`-like shape
    with the base as the source image. Returns the rows rendered."""
    pending = [r for r in store.rows if r.get("seconds") is None
               and r.get("source") in GPU_SOURCES]
    if not pending:
        return []
    client, stats = _client(client)
    for family in {family_of(row["source"]) for row in pending}:
        rows = [r for r in pending if family_of(r["source"]) == family]
        check_installed(client, family,
                        model=next((r["model"] for r in rows if r.get("model")), None),
                        loras=[lora for r in rows for lora in (r.get("loras") or [])])
    manifest = store.job.read_manifest()
    sketch_name: str | None = None
    uploaded: dict[str, str] = {}

    def upload(path: Path, name: str) -> str:
        if name not in uploaded:
            uploaded[name] = client.upload_image(path, name=name)
        return uploaded[name]

    done: list[dict[str, Any]] = []
    try:
        for row in pending:
            source = row["source"]
            if source == "txt2img":
                if row.get("sketch") and sketch_name is None:
                    # The input ROOT under a job-unique name, like `clip`'s uploads.
                    sketch_name = upload(store.job.root / row["sketch"]["file"],
                                         f"anim_{store.job.name}_sketch.png")
                workflow = build_workflow(manifest, seed=row["seed"], sketch_name=sketch_name,
                                          job=store.job, workflow_dir=workflow_dir, attempt=row)
            else:
                parent = row["parent"]
                parent_store = store if parent["target"] == store.target else \
                    Attempts(store.slug, parent["target"], store.work_dir)
                parent_path = parent_store.path(parent_store.get(int(parent["index"])))
                name = upload(parent_path,
                              f"anim_{store.job.name}_src_{parent['target']}_{int(parent['index']):02d}.png")
                if source == "img2img":
                    workflow = build_img2img_workflow(row, source_name=name, job=store.job,
                                                      workflow_dir=workflow_dir)
                else:
                    workflow = build_view_workflow(
                        row["prompt"], seed=row["seed"], base_name=name, slug=store.slug,
                        workflow_dir=workflow_dir, model=row.get("model"), loras=row.get("loras"),
                        prefix=f"edit/{VIEW_WORKFLOW}")
            log(f"concept {source}: {store.target}:{row['index']:02d} (seed {row['seed']}) ...")
            prompt_id, entry, png, elapsed = _run_one(client, workflow, timeout_s=timeout_s)
            path = write_bytes_in_place(store.path(row), png)
            log(f"  -> {path.name}  {elapsed:.1f} s")
            row |= {"prompt_id": prompt_id, "seconds": round(elapsed, 1), "file": path.name,
                    "sha256": sha256_bytes(png),
                    # The graph AS SENT, seed included - what `render` records per
                    # pass and what the pick's provenance copies.
                    "workflow_sha256": comfy.workflow_sha256(workflow),
                    "models": [m.as_sidecar_entry()
                               for m in models_lock.for_workflow(workflow, lock_path)],
                    "embedded_prompt": render_stage.prompt_chunk(png) is not None}
            done.append(row)
            if store.target == BASE_TARGET and source == "txt2img" \
                    and not store.record.get("workflow_sha256"):
                # Same graph for every candidate but the seed; the job-level hash
                # is the first one with its seed zeroed - the SET's, not a member's.
                unseeded = comfy.patch(copy.deepcopy(workflow), "seed", seed=0)
                store.record["workflow_sha256"] = comfy.workflow_sha256(unseeded)
    except BaseException:
        # A failed run's planned rows carry the settings that failed; keep only
        # what actually rendered, so a rerun starts clean and the record never
        # claims an attempt that has no file.
        store.drop_unrendered()
        store.save()
        raise
    comfyui = render_stage.build_provenance(client.base_url, stats,
                                            store.record.get("seed") or 0, [])["comfyui"]
    seen: list[dict[str, Any]] = list(store.record.get("models") or [])
    for row in done:
        for m in row["models"]:
            if m not in seen:
                seen.append(m)
    store.record |= {"comfyui": comfyui, "models": seen}
    store.save()
    return done


def render_concept(job: Job, *, client: comfy.ComfyClient | None = None,
                   workflow_dir: Path | None = None, lock_path: Path | None = None,
                   timeout_s: float = 900.0, log: Callable[[str], None] = lambda _: None
                   ) -> dict[str, Any]:
    """Render the base attempts still waiting for the GPU; the manifest after."""
    slug = job.read_manifest()["slug"]
    store = Attempts(slug, BASE_TARGET, job.root.parent)
    render_pending(store, client=client, workflow_dir=workflow_dir, lock_path=lock_path,
                   timeout_s=timeout_s, log=log)
    return job.read_manifest()


def concept(slug: str, *, description: str, count: int | None = None,
            seed: int | None = None, sketch: Path | None = None,
            existing: Path | str | None = None, lora_strength: float | None = None,
            sketch_strength: float | None = None,
            model: str | None = None, loras: Sequence[str | dict[str, Any]] | None = None,
            fresh: bool = False, registry_path: Path | None = None,
            client: comfy.ComfyClient | None = None,
            workflow_dir: Path | None = None, lock_path: Path | None = None,
            assets_dir: Path | None = None, work_dir: Path | None = None,
            timeout_s: float = 900.0, log: Callable[[str], None] = lambda _: None
            ) -> dict[str, Any]:
    job = prepare(slug, description=description, count=count, seed=seed,
                  sketch=sketch, existing=existing, lora_strength=lora_strength,
                  sketch_strength=sketch_strength, model=model, loras=loras, fresh=fresh,
                  registry_path=registry_path, lock_path=lock_path,
                  assets_dir=assets_dir, work_dir=work_dir)
    store = Attempts(slug, BASE_TARGET, work_dir)
    render_pending(store, client=client, workflow_dir=workflow_dir, lock_path=lock_path,
                   timeout_s=timeout_s, log=log)
    return job.read_manifest()


def rework(slug: str, ref: str, *, op: str, client: comfy.ComfyClient | None = None,
           workflow_dir: Path | None = None, lock_path: Path | None = None,
           work_dir: Path | None = None, timeout_s: float = 900.0,
           log: Callable[[str], None] = lambda _: None, **plan: Any) -> dict[str, Any]:
    """`--img2img REF` / `--edit REF`: plan, render, and (for a view) refresh
    the page. Returns `{"target", "attempts": [rows]}`."""
    store, planned = plan_rework(slug, ref, op=op, lock_path=lock_path, work_dir=work_dir, **plan)
    render_pending(store, client=client, workflow_dir=workflow_dir, lock_path=lock_path,
                   timeout_s=timeout_s, log=log)
    out: dict[str, Any] = {"target": store.target,
                           "attempts": [store.get(int(r["index"])) for r in planned]}
    if store.target != BASE_TARGET:
        _base_job, base_path, _m = base_render(slug, work_dir)
        out["page"] = str(build_views_page(store.record, views_page_dir(slug, work_dir),
                                           base_path=base_path, out_dir=store.dir))
    return out


# ---------------------------------------------------------------------------
# The pick
# ---------------------------------------------------------------------------
def resolve_pick(manifest: dict[str, Any], choice: str | int) -> dict[str, Any]:
    candidates = manifest.get("candidates") or []
    if str(choice).strip().lower() == EXISTING:
        for candidate in reversed(candidates):
            if candidate["source"] == EXISTING:
                return candidate
        raise ValueError("no existing-art candidate in this concept job; run "
                         "`anim concept --existing [PATH]` to register one")
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
    if not job.manifest_path.exists():
        raise FileNotFoundError(f"{slug} has no concept job; `anim concept --unit {slug} "
                                f"--prompt ...` or --upload first")
    manifest = migrate_manifest(job.read_manifest())
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

    generated = candidate["source"] in GPU_SOURCES
    models = list(candidate.get("models") or manifest.get("models") or []) if generated else []
    passes: list[dict[str, Any]] = []
    if generated:
        passes.append({
            "name": (candidate.get("workflow") or manifest["workflow"]).replace(".json", ""),
            "workflow": candidate.get("workflow") or manifest["workflow"],
            "workflow_sha256": candidate.get("workflow_sha256"),
            "prompt_id": candidate.get("prompt_id"), "seed": candidate["seed"],
            "file": out.name, "seconds": candidate.get("seconds"),
            "embedded_prompt": candidate.get("embedded_prompt"),
            "models": models})
    provenance = {
        "schema": render_stage.PROVENANCE_SCHEMA,
        "created": _now(),
        "comfyui": manifest.get("comfyui") or {"url": None, "version": None,
                                               "python": None, "devices": []},
        "seed": candidate.get("seed"),
        "passes": passes,
        "models": models,
        "final": out.name,
        # Where this render came from - the concept job and the attempt -
        # and, for the owner's files, the file it was.
        "concept": {"job": job.name, "candidate": candidate["index"],
                    "file": candidate["file"], "source": candidate["source"],
                    "from": candidate.get("from"),
                    "prompt": candidate.get("prompt") if generated else None,
                    "model": candidate.get("model") if generated else None,
                    "loras": list(candidate.get("loras") or []) if generated else None,
                    "parent": candidate.get("parent"),
                    "lora": manifest.get("lora") if candidate["source"] == "txt2img" else None,
                    "sketch": candidate.get("sketch") if candidate["source"] == "txt2img" else None},
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
    manifest["updated"] = _now()
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


def empty_views(slug: str) -> dict[str, Any]:
    return {"schema": VIEWS_SCHEMA, "slug": slug, "created": _now(), "updated": _now(),
            "base": None, "base_sha256": None, "parts_phrase": None, "seed": None,
            "count": 0, "workflow": f"{VIEW_WORKFLOW}.json", "workflow_sha256": None,
            "comfyui": None, "models": [], "views": {}}


def view_row(view: str, prompt: str | None) -> dict[str, Any]:
    spec = config.CONCEPT_VIEWS[view]
    return {"view": view, "yaw": spec["yaw"], "elevation": spec.get("elevation", "side"),
            "prompt": prompt, "candidates": [], "pick": None}


def migrate_views(record: dict[str, Any]) -> dict[str, Any]:
    """`anim.concept-views/1` -> `/2`: the rows gain `source: views`, their
    prompt, model and an empty LoRA list; `elevation` appears."""
    if record.get("schema") == VIEWS_SCHEMA:
        return record
    views: dict[str, Any] = {}
    for view, row in (record.get("views") or {}).items():
        if view not in config.CONCEPT_VIEWS:
            continue
        fresh = view_row(view, row.get("prompt"))
        fresh["pick"] = row.get("pick")
        for c in row.get("candidates") or []:
            c = dict(c)
            c.setdefault("source", "views")
            c.setdefault("prompt", row.get("prompt"))
            c.setdefault("model", config.CONCEPT_MODELS["qwen-edit"])
            c.setdefault("loras", [])
            c.setdefault("parent", None)
            c.setdefault("workflow", record.get("workflow"))
            c.setdefault("created", record.get("created"))
            fresh["candidates"].append(c)
        views[view] = fresh
    return {**empty_views(record.get("slug", "")), **record, "schema": VIEWS_SCHEMA,
            "views": views}


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
    """Stable for (unit, base render, parts phrase): what the first set of
    views of a base uses without `--seed`; later sets are random."""
    material = "|".join([slug, base_sha256, parts or "", "views"])
    return int(sha256_bytes(material.encode("utf-8"))[:13], 16) % render_stage.SEED_MAX


def render_views(slug: str, *, views: str | Sequence[str] | None = None,
                 count: int | None = None, seed: int | None = None,
                 view_prompt: str | None = None, model: str | None = None,
                 loras: Sequence[str | dict[str, Any]] | None = None,
                 registry_path: Path | None = None,
                 client: comfy.ComfyClient | None = None, workflow_dir: Path | None = None,
                 lock_path: Path | None = None, work_dir: Path | None = None,
                 timeout_s: float = 900.0, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Render `views` of the accepted base, N attempts each, through the
    edit path; write views/views.json and the page. A view asked for again
    gets MORE attempts (its earlier ones and its pick stay); a new base
    clears everything, since every old view was drawn from another
    character. `view_prompt` replaces the config sentence for a single view."""
    names = config.concept_view_names(views)
    count = config.CONCEPT_VIEW_COUNT if count is None else int(count)
    if count < 1:
        raise ValueError("--count must be at least 1")
    if view_prompt is not None and len(names) != 1:
        raise ValueError("--view-prompt goes with exactly one --views name")
    base_job, base_path, base_manifest = base_render(slug, work_dir)
    base_sha = sha256_file(base_path)
    parts, _avoid = base_phrases(slug, work_dir)

    out_dir = views_dir(slug, work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = views_path(slug, work_dir)
    record = migrate_views(read_json(path)) if path.exists() else empty_views(slug)
    if record.get("base_sha256") not in (None, base_sha):
        # A new base: every old view was drawn from another character.
        for stale in out_dir.glob("*.png"):
            stale.unlink()
        record = empty_views(slug)
    first_set = record.get("base_sha256") is None
    if seed is None:
        seed = derive_view_seed(slug, base_sha, parts) if first_set else random_seed()
    seed = int(seed)
    job = _open_job(slug, work_dir)
    generation = resolve_generation("qwen-edit", model=model, loras=loras,
                                    defaults=job.read_manifest().get("defaults"),
                                    registry_path=registry_path, lock_path=lock_path)
    record |= {
        "base": {"job": base_job.name, "file": str(base_path), "sha256": base_sha,
                 "candidate": (base_manifest.get("concept") or {}).get("candidate")},
        "base_sha256": base_sha, "parts_phrase": parts, "seed": seed, "count": count,
    }
    planned: list[tuple[str, dict[str, Any]]] = []
    offset = 0
    for view in names:
        row = record["views"].setdefault(view, view_row(view, None))
        prompt = (view_prompt.strip() if view_prompt and view_prompt.strip()
                  else config.concept_view_prompt(view, parts))
        row["prompt"] = prompt
        store = Attempts(slug, view, work_dir, record=record)
        for _ in range(count):
            planned.append((view, store.append(attempt_row(
                "views", prompt=prompt, seed=seed + offset, generation=generation,
                workflow=VIEW_WORKFLOW))))
            offset += 1
    write_json_in_place(path, record)

    client, stats = _client(client)
    try:
        check_installed(client, "qwen-edit", model=generation["model"],
                        loras=generation["loras"])
    except BaseException:
        _drop_planned(record, planned)
        write_json_in_place(path, record)
        raise
    base_name = client.upload_image(base_path, name=f"anim_{job_name(slug)}_view_base.png")
    models: list[dict[str, Any]] = list(record.get("models") or [])
    try:
        for view, row in planned:
            workflow = build_view_workflow(row["prompt"], seed=row["seed"], base_name=base_name,
                                           slug=slug, workflow_dir=workflow_dir,
                                           model=row["model"], loras=row["loras"])
            log(f"concept views: {view} {row['index']:02d} (seed {row['seed']}) ...")
            prompt_id, entry, png, elapsed = _run_one(client, workflow, timeout_s=timeout_s)
            file = write_bytes_in_place(out_dir / row["file"], png)
            log(f"  -> {file.name}  {elapsed:.1f} s")
            row_models = [m.as_sidecar_entry() for m in models_lock.for_workflow(workflow, lock_path)]
            row |= {"prompt_id": prompt_id, "seconds": round(elapsed, 1), "sha256": sha256_bytes(png),
                    "workflow_sha256": comfy.workflow_sha256(workflow), "models": row_models,
                    "embedded_prompt": render_stage.prompt_chunk(png) is not None}
            for m in row_models:
                if m not in models:
                    models.append(m)
            if not record.get("workflow_sha256"):
                unseeded = comfy.patch(copy.deepcopy(workflow), "seed", seed=0)
                record["workflow_sha256"] = comfy.workflow_sha256(unseeded)
            write_json_in_place(path, record)
    except BaseException:
        # The rows this run planned and did not render carry the settings that
        # failed; drop them so a rerun starts clean (what rendered is kept).
        _drop_planned(record, planned)
        write_json_in_place(path, record)
        raise
    record |= {
        "updated": _now(),
        "comfyui": render_stage.build_provenance(client.base_url, stats, seed, [])["comfyui"],
        "models": models,
        "views": {v: record["views"][v] for v in config.CONCEPT_VIEWS if v in record["views"]},
    }
    write_json_in_place(path, record)
    record["page"] = str(build_views_page(record, views_page_dir(slug, work_dir),
                                          base_path=base_path, out_dir=out_dir))
    record["rendered"] = [{"view": v, "index": r["index"]} for v, r in planned]
    return record


def _drop_planned(record: dict[str, Any],
                  planned: Sequence[tuple[str, dict[str, Any]]]) -> None:
    """Forget the planned view attempts that never rendered (a failed run)."""
    for view, row in planned:
        rows = (record.get("views", {}).get(view) or {}).get("candidates") or []
        if row.get("seconds") is None and row in rows:
            rows.remove(row)


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
    """Copy the chosen attempts into the base as render/view_<view>.png,
    record them in the base's `concept.views` and in the curation record's
    `views` step (accepted; `provisional` records a pick made for a rerun
    that the owner's own --pick-view supersedes, and leaves the step pending)."""
    path = views_path(slug, work_dir)
    if not path.exists():
        raise FileNotFoundError(f"{slug} has no views yet; run `anim concept --unit {slug} "
                                f"--views` first")
    record = migrate_views(read_json(path))
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
        if candidate.get("seconds") is None and candidate.get("source") in (*GPU_SOURCES, "views"):
            raise ValueError(f"view {view} candidate {index:02d} has not been rendered yet")
        source = views_dir(slug, work_dir) / candidate["file"]
        if not source.exists():
            raise FileNotFoundError(f"{source} is missing; run `anim concept --views` again")
        png = source.read_bytes()
        out = write_bytes_in_place(base_job.render_dir / VIEW_FILE.format(view=view), png)
        entry = {"view": view, "yaw": row["yaw"], "file": out.name, "candidate": index,
                 "source": candidate["file"], "seed": candidate["seed"],
                 "sha256": sha256_bytes(png),
                 "prompt": candidate.get("prompt") or row.get("prompt"),
                 "workflow_sha256": candidate.get("workflow_sha256"),
                 "provisional": bool(provisional), "picked": _now()}
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
    record["updated"] = _now()
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
    """The base beside each view's attempts; every write truncates in place.
    A printout of views.json - the studio's Concept tab is the live UI."""
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
            origin = c.get("source") or "views"
            if c.get("parent"):
                origin += f" of {c['parent']['target']}:{int(c['parent']['index']):02d}"
            figures.append(
                f'<figure class="{"picked" if picked else ""}" id="{view}_{c["index"]:02d}">'
                f'<img src="img/{c["file"]}" alt="{view} {c["index"]:02d}">'
                f'<figcaption><b>{view} {c["index"]:02d}</b>{tag} · {origin} · seed {c["seed"]} · '
                f'{c.get("seconds")} s</figcaption></figure>')
        sections.append(f'<h2 id="{view}">{view} — yaw {row["yaw"]}</h2>'
                        f'<p class="caption">{row.get("prompt") or ""}</p><div class="row">{"".join(figures)}</div>')
    base = record.get("base") or {}
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>the views — {slug}</title><style>{_VIEWS_STYLE}</style></head>
<body>
<h1>The views — {slug}, {len(record.get('views') or {})} of {len(config.CONCEPT_VIEWS)} views</h1>
<p class="caption">{record.get('updated') or record.get('created', '')} · Qwen-Image-Edit 2509 (<code>{record.get('workflow')}</code>) with the base as image1 ·
last seed <b>{record.get('seed')}</b> · parts phrase: <i>{record.get('parts_phrase') or '—'}</i> ·
base <code>{Path(base.get('file') or base_path).name}</code> sha256 {str(record.get('base_sha256') or '')[:12]}.
Choose with <code>anim concept --unit {slug} --pick-view back=NN left=NN top=NN</code> or the studio's Concept tab; the turnaround then hands each direction its nearest view
(<code>config.TURNAROUND_REFERENCE_BY_DIR</code>). A view marked <span class="warn">provisional</span> was picked for a rerun and your <code>--pick-view</code> supersedes it.</p>
{''.join(sections)}
<footer>Generated by <code>anim concept --unit {slug} --views</code>; files in <code>work/{slug}_concept/views/</code>, record <code>views.json</code>. Every attempt is kept.</footer>
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
                             "CONCEPT_PROMPT with the LoRA's trigger phrase (with "
                             "--img2img: the prompt for the rework instead of the parent's)")
    parser.add_argument("--sketch", default=None, metavar="PNG",
                        help="a sketch of the character, sent as a scribble "
                             "condition to the Fun ControlNet Union")
    parser.add_argument("--count", type=int, default=None,
                        help=f"attempts to generate (default {config.CONCEPT_COUNT} for "
                             f"the base, {config.CONCEPT_VIEW_COUNT} per view, 1 for a rework)")
    parser.add_argument("--seed", type=int, default=None,
                        help="first seed; attempt i uses seed + i (default: random; "
                             "a --fresh set derives it from the unit, prompt and sketch)")
    parser.add_argument("--existing", nargs="?", const="auto", default=None,
                        metavar="PNG",
                        help="register the existing art as a candidate: a "
                             "path, or with no value the unit's published "
                             "sprite (assets/2d/units/<slug>.png, shown x8)")
    parser.add_argument("--upload", default=None, metavar="PNG",
                        help="register your own image as a base candidate (no GPU, "
                             "no --prompt needed); the studio's Upload button")
    parser.add_argument("--fresh", action="store_true",
                        help="wipe the unit's earlier attempts first (the default appends)")
    parser.add_argument("--model", default=None, metavar="FILE",
                        help="the checkpoint, from gpu/models.registry.tsv for the op's "
                             "family (default: the unit's default, else the workflow's)")
    parser.add_argument("--lora", action="append", default=None, metavar="FILE[:STRENGTH]",
                        help="a LoRA to stack, repeatable, from the registry for the op's "
                             "family (default: the unit's default, else the pixel LoRA "
                             "for Z-Image and none for Qwen-Image-Edit)")
    parser.add_argument("--lora-strength", type=float, default=None,
                        help=f"Pixel-Art LoRA strength when no --lora is given (default "
                             f"{config.CONCEPT_LORA['strength']}; 0 drops the node)")
    parser.add_argument("--sketch-strength", type=float, default=None,
                        help=f"ControlNet strength for the sketch (default "
                             f"{config.CONCEPT_SKETCH_STRENGTH})")
    parser.add_argument("--img2img", default=None, metavar="REF",
                        help="rework the attempt REF (base:NN or <view>:NN) with classic "
                             "img2img at --strength, keeping its composition")
    parser.add_argument("--strength", type=float, default=None,
                        help=f"with --img2img: how much is redrawn, 0-1 (default "
                             f"{config.CONCEPT_IMG2IMG_STRENGTH})")
    parser.add_argument("--edit", default=None, metavar="REF",
                        help="rework the attempt REF through Qwen-Image-Edit with "
                             "--instruction (\"make the helmet red\")")
    parser.add_argument("--instruction", default=None, metavar="TEXT",
                        help="with --edit: what should change")
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
                             "no value = back,left,right,top; a comma list; or 'all' "
                             f"({', '.join(config.CONCEPT_VIEWS)}); --count per view "
                             f"(default {config.CONCEPT_VIEW_COUNT}); appends to the view")
    parser.add_argument("--view-prompt", dest="view_prompt", default=None, metavar="TEXT",
                        help="with a single --views name: your sentence for that view in "
                             "place of the config one")
    parser.add_argument("--pick-view", dest="pick_view", nargs="+", default=None,
                        metavar="VIEW=NN",
                        help="copy those view candidates into the base "
                             "(render/view_<view>.png) for the turnaround to use")
    parser.add_argument("--provisional", action="store_true",
                        help="with --pick-view: record the pick as provisional (made "
                             "for a rerun; the owner's --pick-view supersedes it)")
    parser.add_argument("--force", action="store_true",
                        help="render views of a base whose gate is still pending")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait for one /prompt job")


def _print_attempts(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        seconds = row.get("seconds")
        parent = row.get("parent")
        origin = row["source"] + (f" of {parent['target']}:{int(parent['index']):02d}" if parent else "")
        print(f"  {row['file']:<18} {origin:<22} "
              f"seed {row['seed'] if row['seed'] is not None else '-':<18} "
              f"{f'{seconds:.1f} s' if seconds is not None else ''}")


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
    if args.upload is not None:
        try:
            row = register_upload(args.slug, Path(args.upload))
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        print(f"concept: {args.upload} -> work/{job_name(args.slug)}/candidates/{row['file']} "
              f"(upload, fitted to the cell)")
        print(f"  then: anim concept --unit {args.slug} --pick {row['index']:02d}")
        return 0
    if args.img2img is not None or args.edit is not None:
        if args.img2img is not None and args.edit is not None:
            raise SystemExit("anim concept: give --img2img or --edit, not both")
        op = "img2img" if args.img2img is not None else "edit"
        client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
        try:
            result = rework(args.slug, args.img2img or args.edit, op=op, count=args.count,
                            seed=args.seed, strength=args.strength,
                            instruction=args.instruction, prompt=args.prompt,
                            model=args.model, loras=args.lora,
                            lora_strength=args.lora_strength, client=client,
                            timeout_s=args.timeout, log=print)
        except (ValueError, FileNotFoundError, models_lock.RegistryError) as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        print(f"concept: {len(result['attempts'])} {op} attempt(s) of "
              f"{args.img2img or args.edit} -> {result['target']}")
        _print_attempts(result["attempts"])
        if result["target"] == BASE_TARGET:
            print(f"  then: anim concept --unit {args.slug} --pick NN")
        else:
            print(f"  page   -> {result['page']}")
            print(f"  then: anim concept --unit {args.slug} --pick-view {result['target']}=NN")
        return 0
    if args.views is not None:
        client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
        try:
            record = render_views(args.slug, views=args.views, count=args.count,
                                  seed=args.seed, view_prompt=args.view_prompt,
                                  model=args.model, loras=args.lora, client=client,
                                  timeout_s=args.timeout)
        except (ValueError, FileNotFoundError, models_lock.RegistryError) as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        rendered = record["rendered"]
        print(f"concept: {len(rendered)} view attempt(s) of {args.slug}'s base, seed {record['seed']} "
              f"(ComfyUI {record['comfyui']['version']})")
        for item in rendered:
            row = next(c for c in record["views"][item["view"]]["candidates"]
                       if c["index"] == item["index"])
            print(f"  {row['file']:<22} seed {row['seed']:<18} {row['seconds']:.1f} s")
        print(f"  page   -> {record['page']}")
        print(f"  publish: mkdir -p {VIEWS_VISUALS_TARGET} && cp -r {Path(record['page']).parent}/. "
              f"{VIEWS_VISUALS_TARGET}/")
        print(f"  then:    {VIEWS_VISUALS_URL}")
        print(f"  choose:  anim concept --unit {args.slug} --pick-view "
              + " ".join(f"{v}=NN" for v in {i['view'] for i in rendered}))
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
        raise SystemExit("anim concept: --prompt is required (or --pick, --views, --upload, "
                         "--img2img, --edit, --pick-view, --parts to work on the base)")
    client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
    try:
        manifest = concept(args.slug, description=args.prompt, count=args.count,
                           seed=args.seed,
                           sketch=Path(args.sketch) if args.sketch else None,
                           existing=args.existing, lora_strength=args.lora_strength,
                           sketch_strength=args.sketch_strength, model=args.model,
                           loras=args.lora, fresh=args.fresh, client=client,
                           timeout_s=args.timeout, log=print)
    except (ValueError, FileNotFoundError, models_lock.RegistryError) as exc:
        raise SystemExit(f"anim concept: {exc}") from None
    job = Job.open(job_name(args.slug))
    print(f"concept: {len(manifest['candidates'])} candidate(s) in {job.candidates_dir} "
          f"(ComfyUI {manifest['comfyui']['version']}, seed {manifest['seed']}, "
          f"LoRA {manifest['lora']['strength']}"
          f"{', sketch ' + manifest['sketch']['file'] if manifest.get('sketch') else ''})")
    _print_attempts(manifest["candidates"])
    print(f"  then: anim concept --unit {args.slug} --pick NN")
    return 0


run = run_stage
