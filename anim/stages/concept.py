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
    anim concept --unit knight --pick-view back=01 left=00 top=02     # the views the matrix will use

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
record's `views` step (accepted); `anim matrix` then hands each direction
the nearest accepted view (`config.REFERENCE_BY_DIR`). `--pick` of a new
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
INPAINT_WORKFLOW = "inpaint_pass"
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


class AttemptStore:
    """The machinery every strip of attempts shares, whatever record it lives
    in: numbering, hiding, the two-step delete, dropping what a failed run
    planned. `Attempts` (the concept job: the base and the views) and the
    animate tab feature's `animate.FrameAttempts` (one frame of an animation)
    are the two stores; the ops - img2img, edit, in place, the
    editor's Save - take a store and do not care which.

    A subclass sets `slug`, `target`, `work_dir`, `job`, `dir`, `record` and
    `cli` (the command line a refusal names), and says where its rows live
    (`rows`, `_set_rows`), what its files are called (`file_for`), which
    attempt is accepted (`accepted_index`) and how it is written (`save`)."""

    # The sources whose file comes from the GPU: a row of one of these with no
    # `seconds` and no file is what a failed run planned.
    RENDERED_SOURCES: tuple[str, ...] = (*GPU_SOURCES, "views")
    # What the progress lines call this kind of store ("concept edit: ...").
    label = "concept"

    slug: str
    target: str
    work_dir: Path | None
    job: Job
    dir: Path
    record: dict[str, Any]
    cli: str

    @property
    def rows(self) -> list[dict[str, Any]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _set_rows(self, rows: list[dict[str, Any]]) -> None:  # pragma: no cover
        raise NotImplementedError

    def file_for(self, index: int) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def accepted_index(self) -> int | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def save(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def defaults(self) -> dict[str, Any]:
        """The unit's model defaults per family (`--model` / `--lora` fall back
        to them): the concept job's, which is where "save as unit default"
        writes them."""
        return dict(_open_job(self.slug, self.work_dir).read_manifest().get("defaults") or {})

    def ref(self, index: int) -> str:
        return f"{self.target}:{int(index):02d}"

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
        row = {"index": index, "file": self.file_for(index), **row}
        row.setdefault("created", _now())
        self.rows.append(row)
        return row

    def visible_rows(self) -> list[dict[str, Any]]:
        """The attempts a reader shows by default - what the owner kept."""
        return [r for r in self.rows if not r.get("hidden")]

    def set_hidden(self, index: int, hidden: bool = True) -> dict[str, Any]:
        """Hide an attempt from the strips and the page, or show it again.

        The row stays in the list either way: `_next_index` is `max(index) + 1`
        and `attempt_file` derives the filename from the index, so a removed
        row would let the next attempt reuse its number AND its filename, over
        a PNG still on disk. Hiding is a mark on the record, the way the
        curation board rejects a frame without deleting it.
        """
        row = self.get(index)
        if hidden and self.accepted_index() == index:
            raise ValueError(
                f"{self.target}:{index:02d} is the accepted one - every later stage "
                f"reads it; accept another attempt first, then hide this")
        if hidden:
            row["hidden"] = True
        else:
            row.pop("hidden", None)
        return row

    def delete_file(self, index: int) -> dict[str, Any]:
        """Unlink a hidden attempt's PNG, keeping the row as a tombstone.

        The row carries the prompt, seed, model, LoRAs and sha256 the image was
        made with, and any later attempt made FROM it names it as its `parent`;
        dropping it would break those and free its number for reuse. So the
        file goes and the record of it stays. Hidden first, deliberately: the
        deletion is the second step of a two-step.
        """
        row = self.get(index)
        if self.accepted_index() == index:
            raise ValueError(
                f"{self.target}:{index:02d} is the accepted one - accept another "
                f"attempt first, then delete this one's file")
        if not row.get("hidden"):
            raise ValueError(f"{self.target}:{index:02d} is not hidden; hide it first "
                             f"(`{self.cli} --hide {self.target}:{index:02d}`)")
        path = self.path(row)
        if path.exists():
            path.unlink()
        row["file_deleted"] = _now()
        return row

    def drop_unrendered(self) -> list[dict[str, Any]]:
        """Forget the GPU attempts that never produced a file - what a failed
        run planned. Their settings are the ones that failed, so keeping them
        would make the next run repeat the failure and the record claim
        attempts that do not exist. Returns the rows dropped."""
        dropped = [r for r in self.rows
                   if r.get("seconds") is None and r.get("source") in self.RENDERED_SOURCES
                   and not r.get("file_deleted") and not self.path(r).exists()]
        if not dropped:
            return []
        self._set_rows([r for r in self.rows if r not in dropped])
        return dropped


class Attempts(AttemptStore):
    """The attempt rows of one concept target and where their files live: the
    base's in job.json + candidates/, a view's in views/views.json + views/.
    One shape for both so the ops (txt2img, img2img, edit, views) plan and
    render the same way whichever they land in."""

    def __init__(self, slug: str, target: str, work_dir: Path | None = None, *,
                 record: dict[str, Any] | None = None) -> None:
        self.slug = slug
        self.target = target
        self.work_dir = work_dir
        self.cli = f"anim concept --unit {slug}"
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

    def _set_rows(self, rows: list[dict[str, Any]]) -> None:
        if self.target == BASE_TARGET:
            self.record["candidates"] = rows
            self.record["next_index"] = _next_index(rows)
        else:
            self.record["views"][self.target]["candidates"] = rows

    def file_for(self, index: int) -> str:
        return attempt_file(self.target, index)

    def defaults(self) -> dict[str, Any]:
        return dict(self.job.read_manifest().get("defaults") or {})

    def append(self, row: dict[str, Any]) -> dict[str, Any]:
        row = super().append(row)
        if self.target == BASE_TARGET:
            self.record["next_index"] = int(row["index"]) + 1
        return row

    def accepted_index(self) -> int | None:
        """The attempt this target has accepted, if any: the base's `--pick`
        or the view's `--pick-view`. It is what every later stage reads, so
        it is the one attempt that may not be hidden or deleted."""
        pick = (self.record.get("pick") if self.target == BASE_TARGET
                else self.record["views"][self.target].get("pick")) or {}
        index = pick.get("index")
        return None if index is None else int(index)

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
    description = store.job.read_manifest().get("description") or "character"
    parts, _avoid = base_phrases(slug, work_dir)

    def img2img_text(parent: dict[str, Any]) -> str:
        if target == BASE_TARGET:
            # The parent's own prompt when it has a Z-Image one; a redraw of
            # an upload is asked for as the unit's description.
            return (parent.get("prompt") if parent.get("source") in ("txt2img", "img2img")
                    else None) or config.concept_prompt(description)
        # A view's prompt is written for Qwen ("from image1..."); Z-Image
        # gets the description seen from that side instead.
        return config.concept_view_img2img_prompt(description, target, parts)

    planned = plan_rework_in(store, index, op=op, count=count, seed=seed, strength=strength,
                             instruction=instruction, prompt=prompt, img2img_text=img2img_text,
                             parts=parts, model=model, loras=loras, lora_strength=lora_strength,
                             registry_path=registry_path, lock_path=lock_path)
    if target == BASE_TARGET:
        store.record["last_seed"] = planned[0]["seed"]
    else:
        store.record["seed"] = planned[0]["seed"]
    store.save()
    return store, planned


def plan_rework_in(store: AttemptStore, index: int, *, op: str, count: int | None = None,
                   seed: int | None = None, strength: float | None = None,
                   instruction: str | None = None, prompt: str | None = None,
                   img2img_text: Callable[[dict[str, Any]], str],
                   parts: str | None = None,
                   model: str | None = None,
                   loras: Sequence[str | dict[str, Any]] | None = None,
                   lora_strength: float | None = None, registry_path: Path | None = None,
                   lock_path: Path | None = None) -> list[dict[str, Any]]:
    """`count` img2img / edit rows on `store` with `parent` = that attempt,
    appended but not saved. The store-agnostic half of `plan_rework`: the
    caller says what an img2img is asked for when the owner gives no prompt
    (`img2img_text(parent)`), because that is the one thing a base, a view
    and an animate frame answer differently."""
    if op not in ("img2img", "edit"):
        raise ValueError(f"op is img2img or edit, not {op!r}")
    parent = store.get(index)
    if not store.path(parent).exists():
        raise FileNotFoundError(f"{store.ref(index)} has no file at {store.path(parent)}")
    count = 1 if count is None else int(count)
    if count < 1:
        raise ValueError("--count must be at least 1")
    seed = random_seed() if seed is None else int(seed)
    family = family_of(op)
    generation = resolve_generation(family, model=model, loras=loras,
                                    lora_strength=lora_strength, defaults=store.defaults(),
                                    registry_path=registry_path, lock_path=lock_path)
    if op == "img2img":
        strength = float(config.CONCEPT_IMG2IMG_STRENGTH if strength is None else strength)
        if not 0 < strength <= 1:
            raise ValueError(f"--strength is in (0, 1], not {strength}")
        text = prompt.strip() if prompt and prompt.strip() else img2img_text(parent)
        extra = {"strength": strength, "instruction": None}
        workflow = IMG2IMG_WORKFLOW
    else:
        text = config.concept_edit_prompt(instruction or "", parts)
        extra = {"strength": None, "instruction": " ".join(str(instruction).split())}
        workflow = VIEW_WORKFLOW
    return [store.append(attempt_row(op, prompt=text, seed=seed + i, generation=generation,
                                     parent=(store.target, index), workflow=workflow, **extra))
            for i in range(count)]


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
                        prefix: str = f"views/{VIEW_WORKFLOW}",
                        job: str | None = None) -> dict[str, Any]:
    """view_pass.json with this attempt's inputs patched in: the source as
    image1 (and the latent), the prompt, the seed, the checkpoint, the LoRAs
    (the optional node is bypassed with none), the output prefix (under
    `job`, default the unit's concept job)."""
    workflow = comfy.load_workflow(VIEW_WORKFLOW, workflow_dir)
    comfy.patch(workflow, "base image", image=base_name)
    comfy.patch(workflow, "positive prompt", prompt=prompt)
    comfy.patch(workflow, "seed", seed=int(seed))
    comfy.patch(workflow, "output prefix",
                filename_prefix=f"anim_pipeline/{job or job_name(slug)}/{prefix}")
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
                        prefix=f"edit/{VIEW_WORKFLOW}", job=store.job.name)
            log(f"{store.label} {source}: {store.target}:{row['index']:02d} (seed {row['seed']}) ...")
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


def rework_in_place(slug: str, ref: str, *, op: str, instruction: str | None = None,
                    prompt: str | None = None, strength: float | None = None,
                    seed: int | None = None, model: str | None = None,
                    loras: Sequence[str | dict[str, Any]] | None = None,
                    lora_strength: float | None = None, registry_path: Path | None = None,
                    client: comfy.ComfyClient | None = None,
                    workflow_dir: Path | None = None, lock_path: Path | None = None,
                    work_dir: Path | None = None, timeout_s: float = 900.0,
                    log: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """The retouch feature: run `img2img` or `edit` and write the result over
    the attempt it was made from, instead of appending a new one. One image,
    no `parent` - the row keeps its generation record and gains an `edited`
    block saying what was done to it."""
    if op not in ("img2img", "edit"):
        raise ValueError(f"op is img2img or edit, not {op!r}")
    target, index = parse_ref(ref)
    store = Attempts(slug, target, work_dir)
    parts, _avoid = base_phrases(slug, work_dir)
    description = store.job.read_manifest().get("description") or "character"

    def img2img_text(row: dict[str, Any]) -> str:
        if target == BASE_TARGET:
            return (row.get("prompt") if row.get("source") in ("txt2img", "img2img") else None) \
                or config.concept_prompt(description)
        return config.concept_view_img2img_prompt(description, target, parts)

    row = rework_in_place_in(store, index, op=op, instruction=instruction, prompt=prompt,
                             strength=strength, seed=seed, model=model, loras=loras,
                             lora_strength=lora_strength, img2img_text=img2img_text,
                             parts=parts, registry_path=registry_path, client=client,
                             workflow_dir=workflow_dir, lock_path=lock_path,
                             timeout_s=timeout_s, log=log)
    if target != BASE_TARGET:
        _refresh_views_page(slug, store, work_dir)
    return row


def rework_in_place_in(store: AttemptStore, index: int, *, op: str,
                       instruction: str | None = None, prompt: str | None = None,
                       strength: float | None = None, seed: int | None = None,
                       model: str | None = None,
                       loras: Sequence[str | dict[str, Any]] | None = None,
                       lora_strength: float | None = None,
                       img2img_text: Callable[[dict[str, Any]], str],
                       parts: str | None = None, registry_path: Path | None = None,
                       client: comfy.ComfyClient | None = None,
                       workflow_dir: Path | None = None, lock_path: Path | None = None,
                       timeout_s: float = 900.0,
                       log: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """The store-agnostic half of `rework_in_place`: run the op on the GPU
    and write the result over that attempt, saving the store."""
    if op not in ("img2img", "edit"):
        raise ValueError(f"op is img2img or edit, not {op!r}")
    row = _editable(store, index)
    seed = random_seed() if seed is None else int(seed)
    family = family_of(op)
    generation = resolve_generation(family, model=model, loras=loras,
                                    lora_strength=lora_strength, defaults=store.defaults(),
                                    registry_path=registry_path, lock_path=lock_path)
    if op == "img2img":
        strength = float(config.CONCEPT_IMG2IMG_STRENGTH if strength is None else strength)
        if not 0 < strength <= 1:
            raise ValueError(f"--strength is in (0, 1], not {strength}")
        text = prompt.strip() if prompt and prompt.strip() else img2img_text(row)
    else:
        text = config.concept_edit_prompt(instruction or "", parts)

    client, stats = _client(client)
    check_installed(client, family, model=generation["model"], loras=generation["loras"])
    source = client.upload_image(
        store.path(row), name=f"anim_{store.job.name}_src_{store.target}_{index:02d}.png")
    if op == "img2img":
        transient = {"prompt": text, "seed": seed, "strength": strength,
                     "model": generation["model"], "loras": generation["loras"]}
        workflow = build_img2img_workflow(transient, source_name=source, job=store.job,
                                          workflow_dir=workflow_dir)
    else:
        workflow = build_view_workflow(text, seed=seed, base_name=source, slug=store.slug,
                                       workflow_dir=workflow_dir, model=generation["model"],
                                       loras=generation["loras"], prefix=f"edit/{VIEW_WORKFLOW}",
                                       job=store.job.name)
    log(f"{store.label} {op} in place: {store.target}:{index:02d} (seed {seed}) ...")
    prompt_id, entry, png, elapsed = _run_one(client, workflow, timeout_s=timeout_s)
    path = write_bytes_in_place(store.path(row), png)
    log(f"  -> {path.name}  {elapsed:.1f} s")
    _record_edit(store, row, op, prompt=text, seed=seed, strength=strength,
                 instruction=(" ".join(str(instruction).split()) if op == "edit" else None),
                 model=generation["model"],
                 loras=[{"file": r["file"], "strength": r["strength"]} for r in generation["loras"]],
                 prompt_id=prompt_id, seconds=round(elapsed, 1),
                 workflow=f"{IMG2IMG_WORKFLOW if op == 'img2img' else VIEW_WORKFLOW}.json",
                 workflow_sha256=comfy.workflow_sha256(workflow))
    # The bytes came straight from the model, so the text chunks are there.
    row["embedded_prompt"] = render_stage.prompt_chunk(png) is not None
    store.save()
    return row


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


def hide_attempt(slug: str, ref: str, *, hidden: bool = True,
                 work_dir: Path | None = None) -> dict[str, Any]:
    """`--hide` / `--show` one attempt (see `Attempts.set_hidden`)."""
    target, index = parse_ref(ref)
    store = Attempts(slug, target, work_dir)
    row = store.set_hidden(index, hidden)
    store.save()
    if target != BASE_TARGET:
        _refresh_views_page(slug, store, work_dir)
    return row


def delete_attempt_file(slug: str, ref: str, work_dir: Path | None = None) -> dict[str, Any]:
    """`--delete-file` on a hidden attempt (see `Attempts.delete_file`)."""
    target, index = parse_ref(ref)
    store = Attempts(slug, target, work_dir)
    row = store.delete_file(index)
    store.save()
    if target != BASE_TARGET:
        _refresh_views_page(slug, store, work_dir)
    return row


def _refresh_views_page(slug: str, store: Attempts, work_dir: Path | None) -> str | None:
    """Rewrite the views printout after a change to a view's attempts. Quiet
    when there is no base to draw it from - the record is still written."""
    try:
        _base_job, base_path, _manifest = base_render(slug, work_dir)
    except FileNotFoundError:
        return None
    return str(build_views_page(store.record, views_page_dir(slug, work_dir),
                                base_path=base_path, out_dir=store.dir))


# ---------------------------------------------------------------------------
# The retouch feature: change an attempt where it sits
# ---------------------------------------------------------------------------
# An attempt is normally immutable - a press appends and the record says what
# the model drew. Retouching breaks that on purpose, so it is explicit about
# it: the file is overwritten, `sha256` is recomputed, `embedded_prompt` goes
# (Pillow's re-save drops ComfyUI's text chunks), and an `edited` block says
# when and how. The generation record - seed, model, LoRAs, prompt - stays as
# the history of what the pixels grew out of. A hidden or file-deleted attempt
# is refused: retouching is for what you are looking at.
def _editable(store: "AttemptStore", index: int) -> dict[str, Any]:
    row = store.get(index)
    if row.get("file_deleted"):
        raise ValueError(f"{store.target}:{index:02d}'s file was deleted; there is nothing "
                         f"to edit")
    if row.get("hidden"):
        raise ValueError(f"{store.target}:{index:02d} is hidden; show it first "
                         f"(`{store.cli} --show {store.target}:{index:02d}`)")
    if not store.path(row).exists():
        raise FileNotFoundError(f"{store.path(row)} is missing")
    return row


def flatten_on_white(data: bytes) -> tuple[Image.Image, Image.Image | None]:
    """`(the image on white, its transparency as a mask)` - the two halves of
    a retouched canvas.

    What comes back from the editor is RGBA with binary alpha: what you
    painted is opaque, what you rubbed out is (0,0,0,0). A concept image is
    opaque on white, so on a save the hole becomes white; on an inpaint the
    same hole is the region to redraw. Mask is None when nothing was erased.
    """
    from io import BytesIO

    try:
        with Image.open(BytesIO(data)) as raw:
            raw.load()
            rgba = raw.convert("RGBA")
    except Exception as exc:  # noqa: BLE001 - PIL raises a zoo
        raise ValueError(f"not an image PIL can read: {exc}") from None
    alpha = rgba.getchannel("A")
    flat = Image.new("RGB", rgba.size, BACKGROUND)
    flat.paste(rgba, (0, 0), rgba)
    erased = [count for value, count in enumerate(alpha.histogram()) if value < 128 and count]
    return flat, (alpha if erased else None)


def _record_edit(store: "AttemptStore", row: dict[str, Any], how: str, **extra: Any) -> dict[str, Any]:
    """Re-hash the file and note what was done to it."""
    path = store.path(row)
    row["sha256"] = sha256_file(path)
    row["embedded_prompt"] = False
    row["edited"] = {"how": how, "at": _now(),
                     **{k: v for k, v in extra.items() if v is not None}}
    return row


def replace_attempt(slug: str, ref: str, data: bytes, *, how: str = "hand",
                    work_dir: Path | None = None) -> dict[str, Any]:
    """Write `data` over the attempt `ref` names (the editor's Save).

    The PNG keeps the attempt's size - a different size is a different asset,
    the rule the frame editor already holds to - and any transparency is
    flattened onto white, because that is what every other concept image is
    and transparency here means "mask", not "hole".
    """
    target, index = parse_ref(ref)
    store = Attempts(slug, target, work_dir)
    row = replace_in(store, index, data, how=how)
    if target != BASE_TARGET:
        _refresh_views_page(slug, store, work_dir)
    return row


def replace_in(store: AttemptStore, index: int, data: bytes, *, how: str = "hand"
               ) -> dict[str, Any]:
    """The store-agnostic half of `replace_attempt`: check the size, flatten,
    write over the attempt, record the edit, save the store."""
    row = _editable(store, index)
    path = store.path(row)
    flat, _mask = flatten_on_white(data)
    with Image.open(path) as current:
        size = current.size
    if flat.size != size:
        raise ValueError(f"{row['file']} is {size[0]}x{size[1]}; the image sent is "
                         f"{flat.size[0]}x{flat.size[1]} - an attempt keeps its size")
    from io import BytesIO

    out = BytesIO()
    flat.save(out, format="PNG", optimize=True)
    write_bytes_in_place(path, out.getvalue())
    _record_edit(store, row, how)
    store.save()
    return row


def stash_canvas(slug: str, ref: str, data: bytes, work_dir: Path | None = None) -> Path:
    """Put the editor's canvas under the job so a run can read it as a file.

    The API takes the canvas as a body but the redraw is a GPU stage, and a
    stage is the CLI in a subprocess - so the bytes land here first and the
    run is handed the path. Checked before it is written: the attempt must be
    one that can be edited, and something must be erased.
    """
    target, index = parse_ref(ref)
    return stash_canvas_in(Attempts(slug, target, work_dir), index, data)


def stash_canvas_in(store: AttemptStore, index: int, data: bytes) -> Path:
    """The store-agnostic half of `stash_canvas` (the animate tab feature
    stashes a frame's canvas the same way)."""
    _editable(store, index)
    _source, mask = flatten_on_white(data)
    if mask is None:
        raise ValueError("nothing is erased on this canvas - rub out the area the model "
                         "should redraw, then inpaint")
    masks = store.job.root / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return write_bytes_in_place(masks / f"{stamp}_{store.target}_{index:02d}_canvas.png", data)


def build_inpaint_workflow(attempt: dict[str, Any], *, canvas_name: str, prompt: str,
                           seed: int, strength: float, job: Job,
                           workflow_dir: Path | None = None) -> dict[str, Any]:
    """inpaint_pass.json: the canvas as the latent, its transparency as the
    noise mask, so only the erased region is denoised."""
    workflow = comfy.load_workflow(INPAINT_WORKFLOW, workflow_dir)
    comfy.patch(workflow, "source image", image=canvas_name)
    comfy.patch(workflow, "positive prompt", text=prompt)
    comfy.patch(workflow, "seed", seed=int(seed), denoise=float(strength))
    comfy.patch(workflow, "output prefix",
                filename_prefix=f"anim_pipeline/{job.name}/{INPAINT_WORKFLOW}")
    if attempt.get("model"):
        comfy.patch_checkpoint(workflow, attempt["model"])
    comfy.chain_loras(workflow, "Pixel-Art LoRA", attempt.get("loras") or [])
    return workflow


def inpaint_attempt(slug: str, ref: str, canvas: bytes | Path, *, prompt: str | None = None,
                    strength: float | None = None, seed: int | None = None,
                    model: str | None = None,
                    loras: Sequence[str | dict[str, Any]] | None = None,
                    registry_path: Path | None = None,
                    client: comfy.ComfyClient | None = None,
                    workflow_dir: Path | None = None, lock_path: Path | None = None,
                    work_dir: Path | None = None, timeout_s: float = 900.0,
                    log: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """Redraw the erased region of `canvas` into the attempt `ref` names.

    `canvas` is the attempt as the editor left it: what you painted is the
    source (so a rough shape guides the model) and what you rubbed out is the
    mask. Everything outside the mask is pasted back from the canvas
    afterwards, so a decode that shifts the whole frame cannot change a pixel
    you did not erase.
    """
    target, index = parse_ref(ref)
    store = Attempts(slug, target, work_dir)
    row = inpaint_in(store, index, canvas, prompt=prompt, strength=strength, seed=seed,
                     model=model, loras=loras, registry_path=registry_path, client=client,
                     workflow_dir=workflow_dir, lock_path=lock_path, timeout_s=timeout_s,
                     log=log)
    if target != BASE_TARGET:
        _refresh_views_page(slug, store, work_dir)
    return row


def inpaint_in(store: AttemptStore, index: int, canvas: bytes | Path, *,
               prompt: str | None = None, default_prompt: str | None = None,
               strength: float | None = None, seed: int | None = None,
               model: str | None = None,
               loras: Sequence[str | dict[str, Any]] | None = None,
               registry_path: Path | None = None,
               client: comfy.ComfyClient | None = None,
               workflow_dir: Path | None = None, lock_path: Path | None = None,
               timeout_s: float = 900.0,
               log: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """The store-agnostic half of `inpaint_attempt`: any strip of attempts -
    a concept target or an animate frame - redrawn where the canvas is
    erased. `default_prompt` is what Z-Image is asked for when the owner
    types nothing and the attempt carries no prompt of its own that suits."""
    row = _editable(store, index)
    target = store.target
    data = Path(canvas).read_bytes() if isinstance(canvas, (str, Path)) else bytes(canvas)
    source, mask = flatten_on_white(data)
    if mask is None:
        raise ValueError("nothing is erased on this canvas - rub out the area the model "
                         "should redraw, then inpaint")
    path = store.path(row)
    with Image.open(path) as current:
        size = current.size
    if source.size != size:
        raise ValueError(f"{row['file']} is {size[0]}x{size[1]}; the canvas is "
                         f"{source.size[0]}x{source.size[1]} - an attempt keeps its size")
    strength = float(config.CONCEPT_INPAINT_STRENGTH if strength is None else strength)
    if not 0 < strength <= 1:
        raise ValueError(f"--strength is in (0, 1], not {strength}")
    seed = random_seed() if seed is None else int(seed)
    concept_manifest = _open_job(store.slug, store.work_dir).read_manifest()
    text = (prompt or "").strip() or default_prompt or row.get("prompt") or config.concept_prompt(
        concept_manifest.get("description") or "character")
    generation = resolve_generation(
        "zimage", model=model, loras=loras,
        defaults=concept_manifest.get("defaults"),
        registry_path=registry_path, lock_path=lock_path)

    client, stats = _client(client)
    check_installed(client, "zimage", model=generation["model"], loras=generation["loras"])
    # The canvas goes up as RGBA: LoadImage reads its colours as the image and
    # its alpha as the mask, so one upload carries both.
    from io import BytesIO

    rgba = source.convert("RGBA")
    rgba.putalpha(mask)
    buffer = BytesIO()
    rgba.save(buffer, format="PNG")
    upload = store.job.root / "masks"
    upload.mkdir(parents=True, exist_ok=True)
    kept = write_bytes_in_place(
        upload / f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}_{target}_{index:02d}.png",
        buffer.getvalue())
    canvas_name = client.upload_image(
        kept, name=f"anim_{store.job.name}_inpaint_{target}_{index:02d}.png")
    attempt = {"model": generation["model"], "loras": generation["loras"]}
    workflow = build_inpaint_workflow(attempt, canvas_name=canvas_name, prompt=text, seed=seed,
                                      strength=strength, job=store.job,
                                      workflow_dir=workflow_dir)
    log(f"{store.label} inpaint: {target}:{index:02d} (seed {seed}, strength {strength}) ...")
    prompt_id, entry, png, elapsed = _run_one(client, workflow, timeout_s=timeout_s)
    # Composite back: only the erased region may change.
    from io import BytesIO as _BytesIO

    with Image.open(_BytesIO(png)) as raw:
        raw.load()
        drawn = raw.convert("RGB")
    if drawn.size != size:
        drawn = drawn.resize(size, Image.LANCZOS)
    merged = source.copy()
    merged.paste(drawn, (0, 0), ImageOps.invert(mask))
    out = _BytesIO()
    merged.save(out, format="PNG", optimize=True)
    write_bytes_in_place(path, out.getvalue())
    log(f"  -> {path.name}  {elapsed:.1f} s")
    _record_edit(store, row, "inpaint", prompt=text, strength=strength, seed=seed,
                 model=generation["model"],
                 loras=[{"file": r["file"], "strength": r["strength"]} for r in generation["loras"]],
                 prompt_id=prompt_id, seconds=round(elapsed, 1),
                 workflow=f"{INPAINT_WORKFLOW}.json",
                 workflow_sha256=comfy.workflow_sha256(workflow),
                 mask=str(kept), comfyui=stats.get("system", {}).get("comfyui_version"))
    store.save()
    return row


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
    if candidate.get("file_deleted"):
        raise ValueError(f"candidate {candidate['index']:02d}'s file was deleted on "
                         f"{candidate['file_deleted']}; pick another")
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
    # The front base pose feature: a concept base is never rendered from its
    # skeleton, but `post --parts` prompts SAM from it, and what was picked is
    # a front view - so lay down the front skeleton, not the side profile.
    base_stage.prepare(slug, cell=size, pose_path=base_stage.STAND_POSE_FRONT,
                       work_dir=work_dir)
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

    # Accepting an attempt is the opposite of hiding it.
    candidate.pop("hidden", None)
    manifest["pick"] = {"index": candidate["index"], "file": candidate["file"],
                        "source": candidate["source"], "base_job": base_job.name,
                        # The bytes that were copied: the retouch feature reads
                        # this back to say the accepted attempt has been edited
                        # since, and the copy is the older picture.
                        "sha256": sha256_bytes(png),
                        "picked": provenance["created"]}
    manifest["updated"] = _now()
    job.write_manifest(manifest)
    # The curation board feature: choosing the base IS the base gate. The
    # views, drawn from the old base, are cleared (the multi-view reference
    # feature).
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
        if candidate.get("file_deleted"):
            raise ValueError(f"view {view} candidate {index:02d}'s file was deleted "
                             f"on {candidate['file_deleted']}; pick another")
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
        written[view] = entry
        candidate.pop("hidden", None)
        row["pick"] = {"index": index, "file": candidate["file"], "sha256": entry["sha256"],
                       "provisional": bool(provisional), "picked": entry["picked"]}
    states = _record_views(slug, base_job, base_manifest, written,
                           state="pending" if provisional else "accepted", work_dir=work_dir)
    page = _save_views(slug, record, base_path=base_path, work_dir=work_dir)
    return {"base_job": base_job.name, "views": written, "curation": states, "page": str(page)}


def _record_views(slug: str, base_job: Job, base_manifest: dict[str, Any],
                  entries: dict[str, dict[str, Any]], *, state: str,
                  work_dir: Path | None = None) -> dict[str, str]:
    """Merge accepted view entries into the base and the curation record.

    The one place a view becomes the thing later stages read: the base's
    `concept.views` block (in the table's compass order, whatever order they
    were accepted in), the base's provenance sidecar beside it, and the
    curation record's `views` step. `pick_view` and `mirror_view` both land
    here on purpose - a second copy of this would be a second answer to
    "what does the base carry", and `curation.base_views` reads only one.
    """
    block = dict(base_manifest.get("concept") or {})
    stored = dict(block.get("views") or {}) | entries
    block["views"] = {v: stored[v] for v in config.CONCEPT_VIEWS if v in stored}
    base_job.update_manifest(concept=block)
    if base_job.provenance_path.exists():
        provenance = read_json(base_job.provenance_path)
        provenance["concept"] = {**(provenance.get("concept") or {}), "views": block["views"]}
        write_json_in_place(base_job.provenance_path, provenance)
    return curation.set_views(slug, {v: state for v in entries}, work_dir)


def _save_views(slug: str, record: dict[str, Any], *, base_path: Path,
                work_dir: Path | None = None) -> Path:
    """Save views.json and redraw the static views page over it.

    The rows are put back into the table's compass order on the way out, so a
    view accepted (or mirrored) out of order does not leave views.json and the
    page reading back-left, right, back - `render_views` does the same at the
    end of its run.
    """
    record["updated"] = _now()
    record["views"] = {v: record["views"][v] for v in config.CONCEPT_VIEWS
                       if v in (record.get("views") or {})}
    write_json_in_place(views_path(slug, work_dir), record)
    return build_views_page(record, views_page_dir(slug, work_dir), base_path=base_path,
                            out_dir=views_dir(slug, work_dir))


# ---------------------------------------------------------------------------
# The mirrored view feature: a view derived by flipping its partner
# ---------------------------------------------------------------------------
# Words that name a thing a character HOLDS or wears on one side. A horizontal
# flip is the same pixels, so a mirrored view is more consistent with its
# partner than a second Qwen pass would be - with one exception: the sword
# moves from the right hand to the left. The list is what turns that from a
# silent fact into one printed line. It is deliberately a word list and not a
# parser: `config.UNIT_PROPS` is authoritative for the units whose kit is
# settled, and for every other unit the only description of the gear is the
# owner's own parts phrase, so "longsword" has to match "sword".
HANDED_GEAR_WORDS: tuple[str, ...] = (
    "sword", "shield", "bow", "quiver", "axe", "spear", "lance", "staff",
    "wand", "dagger", "torch", "hammer", "mace", "banner", "scabbard")


def handed_gear(slug: str, work_dir: Path | None = None) -> list[str]:
    """The asymmetric kit this unit is known to carry - `config.UNIT_PROPS`
    first (the rig draws these onto the skeleton, so they are the ones the
    pipeline is sure about), then anything the base's parts phrase names."""
    named = list(config.props_for_unit(slug))
    parts, _avoid = base_phrases(slug, work_dir)
    lowered = (parts or "").lower()
    for word in HANDED_GEAR_WORDS:
        if word in lowered and word not in named:
            named.append(word)
    return named


def mirror_view(slug: str, views: str | Sequence[str], *, work_dir: Path | None = None,
                log: Callable[[str], None] = print) -> dict[str, Any]:
    """Derive views from their accepted partners by flipping them, no GPU.

    Three of the eight views are the same picture the other way round
    (`config.CONCEPT_MIRROR_PAIRS`, and the compass reasoning beside it):
    right/left, back_right/back_left, front_right/front_left. Drawing the
    second of a pair costs a Qwen-Image-Edit round trip and gives a picture
    that is only ROUGHLY the partner reversed; flipping gives one that is
    exactly it, which is the stronger answer to "is this one character in
    eight facings".

    THE PARTNER MUST BE ACCEPTED, not merely drawn. This is the matrix's
    strictness (`matrix.prerequisite`) rather than `reference_for`'s: a
    mirrored view is what a whole facing of the matrix will be rendered
    against, and deriving it from an attempt nobody has judged would launder
    an unjudged image into an accepted one. `hooded_archer` is the case -
    four views drawn, none picked - and it must refuse there.

    WHAT IS RECORDED. The same `concept.views` entry a `--pick-view` writes,
    plus `mirrored: true` and a `mirror` block naming the source view, the
    file it was flipped from and that file's sha256. The entry's own `sha256`
    is of the flipped bytes, so the view is exactly as traceable as a
    generated one: the pair (source sha256, "flip horizontally") reproduces it
    byte for byte, which no seed can promise.

    NOTHING IS MIRRORED BEHIND THE OWNER'S BACK. A matrix cell never is;
    an Animate cell is only when the owner presses "Mirror from <partner>"
    for that one animation (`animate.mirror`, the animate tab feature), and
    the flipped frames come back to review, never approved. A left-handed
    swing read as a right-handed one is a bug nobody would see until the
    sheet shipped, so the choice stays a person's.
    """
    names = [str(v).strip().lower() for v in
             ([views] if isinstance(views, str) else list(views)) if str(v).strip()]
    if not names:
        raise ValueError("--mirror-view needs at least one view name")
    for view in names:
        if view not in config.CONCEPT_VIEWS:
            raise ValueError(f"no view {view!r}; one of {list(config.CONCEPT_VIEWS)}")
        if config.mirror_of(view) is None:
            raise ValueError(
                f"the {view} view has no mirror partner - it looks straight down the "
                f"symmetry plane, so flipping it would give the same facing back. "
                f"Mirrorable: {', '.join(config.mirrorable_views())}; render {view} "
                f"with `anim concept --unit {slug} --views {view}`")

    base_job, base_path, base_manifest = base_render(slug, work_dir)
    # `base_views` is the one reader of "what the base carries, with each
    # view's curation state, minus the ones whose file is gone" - the same
    # function the Views tab and the Animate tab's gate use.
    have = curation.base_views(slug, work_dir)
    path = views_path(slug, work_dir)
    record = migrate_views(read_json(path)) if path.exists() else empty_views(slug)
    written: dict[str, Any] = {}
    for view in names:
        partner = config.mirror_of(view)
        entry = have.get(partner)
        if entry is None:
            raise ValueError(
                f"{slug}'s {partner} view is not on the base, so there is nothing to "
                f"flip into {view}. Draw it (`anim concept --unit {slug} --views "
                f"{partner}`) and accept one attempt (`--pick-view {partner}=NN`) first")
        if entry.get("state") != "accepted":
            raise ValueError(
                f"{slug}'s {partner} view is {entry.get('state')}, not accepted - a "
                f"mirrored view inherits its partner's judgement, so the partner has to "
                f"have one. Accept it first (`anim concept --unit {slug} --pick-view "
                f"{partner}=NN`, or the Board's views ring)")
        if (entry.get("mirror") or {}).get("view") == view:
            raise ValueError(
                f"{slug}'s {partner} view is itself the flip of {view}; flipping it back "
                f"would only re-derive the picture {view} already is")
        source = Path(entry["path"])
        source_sha = sha256_file(source)
        png = _flip_horizontally(source)
        out = write_bytes_in_place(base_job.render_dir / VIEW_FILE.format(view=view), png)
        picked = _now()
        written[view] = {
            "view": view, "yaw": config.CONCEPT_VIEWS[view]["yaw"], "file": out.name,
            # No candidate and no seed of its own: nothing was sampled. The
            # partner's are in the `mirror` block, where they say whose they are.
            "candidate": None, "source": entry.get("file"), "seed": None,
            "sha256": sha256_bytes(png), "prompt": entry.get("prompt"),
            "workflow_sha256": entry.get("workflow_sha256"),
            "provisional": False, "picked": picked,
            "mirrored": True,
            "mirror": {"view": partner, "file": entry.get("file"), "sha256": source_sha,
                       "candidate": entry.get("candidate"), "seed": entry.get("seed"),
                       "how": "flip horizontally", "at": picked},
        }
        # views.json learns about it too, so the record of the views and the
        # base cannot disagree about which are accepted. There is no candidate
        # index to point at - the picture is not one of this view's attempts -
        # so the pick carries `index: None` and says where it came from.
        row = record.setdefault("views", {}).setdefault(view, view_row(view, None))
        row["pick"] = {"index": None, "mirrored": True, "from": partner,
                       "file": entry.get("file"), "sha256": written[view]["sha256"],
                       "provisional": False, "picked": picked}
        log(f"concept: view {view} <- {partner} flipped horizontally -> "
            f"work/{base_job.name}/render/{out.name}")

    states = _record_views(slug, base_job, base_manifest, written, state="accepted",
                           work_dir=work_dir)
    page = _save_views(slug, record, base_path=base_path, work_dir=work_dir)
    # ONE warning per invocation, whatever it mirrored. The flip is right for
    # the silhouette and wrong for the sword hand, and the owner decides which
    # matters; saying it once is information, saying it per view is nagging.
    gear = handed_gear(slug, work_dir)
    warning: str | None = None
    if gear:
        warning = (f"the flip swaps handedness: {slug} carries {', '.join(gear)}, so "
                   f"{', '.join(names)} shows them on the other side. Everything else is "
                   f"the same pixels; render the view instead if the hand matters.")
        log(f"concept: warning - {warning}")
    return {"base_job": base_job.name, "views": written, "curation": states,
            "page": str(page), "warning": warning}


def _flip_horizontally(source: Path) -> bytes:
    """The partner's file, mirrored left-to-right, as PNG bytes. The image
    mode is kept (the views are RGB off Qwen, but an uploaded RGBA base must
    not lose its alpha here)."""
    from io import BytesIO

    out = BytesIO()
    with Image.open(source) as img:
        ImageOps.mirror(img).save(out, format="PNG")
    return out.getvalue()


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
            if c.get("hidden"):
                continue      # the printout is what the owner kept
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
        # The mirrored view feature: this view's accepted picture is not one of
        # the attempts below - it is the partner's, flipped - so the heading
        # says so rather than leaving every tile unmarked and unexplained.
        mirrored = (f' · <span class="ok">accepted: the {pick["from"]} view, flipped</span>'
                    if pick.get("mirrored") else "")
        sections.append(f'<h2 id="{view}">{view} — yaw {row["yaw"]}{mirrored}</h2>'
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
Choose with <code>anim concept --unit {slug} --pick-view back=NN left=NN top=NN</code> or the studio's Views tab; the Animate tab then draws each direction from its own accepted view
(<code>config.REFERENCE_BY_DIR</code>). A view marked <span class="warn">provisional</span> was picked for a rerun and your <code>--pick-view</code> supersedes it.</p>
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
    parser.add_argument("--replace", default=None, metavar="REF",
                        help="the retouch feature: write --image over the attempt REF "
                             "(the editor's Save); transparency is flattened onto white")
    parser.add_argument("--image", default=None, metavar="PNG",
                        help="with --replace: the image to write, the attempt's own size")
    parser.add_argument("--inpaint", default=None, metavar="REF",
                        help="redraw the erased region of --mask into the attempt REF; "
                             "everything you did not erase is pasted back untouched")
    parser.add_argument("--mask", default=None, metavar="PNG",
                        help="with --inpaint: the attempt as you left it in the editor - "
                             "what you painted is the source, what you rubbed out is redrawn")
    parser.add_argument("--in-place", dest="in_place", action="store_true",
                        help="with --img2img / --edit: write the result over that attempt "
                             "instead of appending a new one")
    parser.add_argument("--hide", default=None, metavar="REF",
                        help="hide the attempt REF (base:NN or <view>:NN) from the strips "
                             "and the views page; the file and the record stay")
    parser.add_argument("--show", default=None, metavar="REF",
                        help="show a hidden attempt again")
    parser.add_argument("--delete-file", dest="delete_file", default=None, metavar="REF",
                        help="delete a HIDDEN attempt's PNG; the row stays with its prompt, "
                             "seed and parent (its number is never reused)")
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
                             "no value = back,left,right,top; a comma list; a group "
                             f"({', '.join(config.CONCEPT_VIEW_GROUPS)}); or 'all' "
                             f"({', '.join(config.CONCEPT_VIEWS)}); --count per view "
                             f"(default {config.CONCEPT_VIEW_COUNT}); appends to the view")
    parser.add_argument("--view-prompt", dest="view_prompt", default=None, metavar="TEXT",
                        help="with a single --views name: your sentence for that view in "
                             "place of the config one")
    parser.add_argument("--pick-view", dest="pick_view", nargs="+", default=None,
                        metavar="VIEW=NN",
                        help="copy those view candidates into the base "
                             "(render/view_<view>.png) for the matrix to use")
    parser.add_argument("--mirror-view", dest="mirror_view", nargs="+", default=None,
                        metavar="VIEW",
                        help="the mirrored view feature: derive those views by flipping "
                             "their accepted partner instead of spending the GPU "
                             f"({', '.join(config.mirrorable_views())}; back, top and "
                             f"front have no partner)")
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
        # A hidden attempt is marked, not dropped: the CLI shows the whole
        # record, the tab and the page show what the owner kept.
        mark = ("  [file deleted]" if row.get("file_deleted")
                else "  [hidden]" if row.get("hidden") else "")
        print(f"  {row['file']:<18} {origin:<22} "
              f"seed {row['seed'] if row['seed'] is not None else '-':<18} "
              f"{f'{seconds:.1f} s' if seconds is not None else ''}{mark}")


def run_stage(args: argparse.Namespace) -> int:
    for flag, call in (("hide", lambda ref: hide_attempt(args.slug, ref, hidden=True)),
                       ("show", lambda ref: hide_attempt(args.slug, ref, hidden=False)),
                       ("delete_file", lambda ref: delete_attempt_file(args.slug, ref))):
        ref = getattr(args, flag)
        if ref is None:
            continue
        try:
            row = call(ref)
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        what = {"hide": "hidden", "show": "shown", "delete_file": "file deleted"}[flag]
        print(f"concept: {ref} {what}"
              f"{' (' + row['file'] + ')' if flag != 'show' else ''}")
        if flag == "hide":
            print(f"  it stays in the record; --show brings it back, --delete-file {ref} "
                  f"removes the PNG")
        return 0
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
        print(f"  base gate accepted for {args.slug} - work/{args.slug}_curation.json")
        print(f"  next: anim post {result['base_job']} [--parts], then "
              f"anim publish {result['base_job']}")
        print(f"  the views are drawn from the base: anim concept --unit {args.slug} --views, "
              f"then --pick-view and accept each one on the Board")
        return 0
    if args.pick_view is not None and args.mirror_view is not None:
        raise SystemExit("anim concept: give --pick-view or --mirror-view, not both "
                         "(a view is either one of its own attempts or its partner "
                         "flipped)")
    if args.mirror_view is not None:
        try:
            result = mirror_view(args.slug, args.mirror_view)
        except (ValueError, FileNotFoundError, OSError) as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        print(f"  curation views: " + ", ".join(f"{v}={s}" for v, s in result["curation"].items()))
        print(f"  no GPU was used: the flip is the partner's own pixels, sha256 recorded")
        print(f"  page   -> {result['page']}")
        print(f"  then: accept it on the Board (anim curate {args.slug} --accept-view ...)")
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
    if args.replace is not None:
        if not args.image:
            raise SystemExit("anim concept: --replace needs --image PNG")
        try:
            row = replace_attempt(args.slug, args.replace, Path(args.image).read_bytes())
        except (ValueError, FileNotFoundError, OSError) as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        print(f"concept: {args.image} -> {args.replace} ({row['file']}), edited by hand")
        print(f"  its record keeps the seed and model it was generated from; sha256 redone")
        if args.replace.partition(":")[0] == BASE_TARGET:
            print(f"  if this is the accepted base, `--pick {row['index']:02d}` again to push "
                  f"it through to work/{args.slug}_base/render/")
        return 0
    if args.inpaint is not None:
        if not args.mask:
            raise SystemExit("anim concept: --inpaint needs --mask PNG (the attempt with the "
                             "area to redraw erased)")
        client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
        try:
            row = inpaint_attempt(args.slug, args.inpaint, Path(args.mask),
                                  prompt=args.prompt, strength=args.strength, seed=args.seed,
                                  model=args.model, loras=args.lora, client=client,
                                  timeout_s=args.timeout, log=print)
        except (ValueError, FileNotFoundError, OSError, models_lock.RegistryError) as exc:
            raise SystemExit(f"anim concept: {exc}") from None
        edit = row["edited"]
        print(f"concept: {args.inpaint} inpainted in place ({row['file']}, seed {edit['seed']}, "
              f"strength {edit['strength']}, {edit['seconds']:.1f} s)")
        print(f"  outside the erased area not one pixel changed")
        return 0
    if args.img2img is not None or args.edit is not None:
        if args.img2img is not None and args.edit is not None:
            raise SystemExit("anim concept: give --img2img or --edit, not both")
        op = "img2img" if args.img2img is not None else "edit"
        client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
        if args.in_place:
            try:
                row = rework_in_place(args.slug, args.img2img or args.edit, op=op,
                                      instruction=args.instruction, prompt=args.prompt,
                                      strength=args.strength, seed=args.seed, model=args.model,
                                      loras=args.lora, lora_strength=args.lora_strength,
                                      client=client, timeout_s=args.timeout, log=print)
            except (ValueError, FileNotFoundError, models_lock.RegistryError) as exc:
                raise SystemExit(f"anim concept: {exc}") from None
            print(f"concept: {args.img2img or args.edit} {op}ed in place ({row['file']}, "
                  f"{row['edited']['seconds']:.1f} s)")
            return 0
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
        print(f"  every clip of {args.slug} names them from now on")
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
