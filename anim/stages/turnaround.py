"""Stage: `anim turnaround` - one idle still per direction, the identity gate (a GPU stage)

The turnaround feature (#275, the fourth feature of the sprite studio epic).
From the unit's accepted base, one standing still in every facing, each with
its own part palettes - and the question the owner answers on the page before
a single action clip is spent: is this one character in eight facings?

    anim turnaround knight                       # the eight side facings s0 .. s315
    anim turnaround knight --dirs s90,s270       # a subset
    anim turnaround knight --force               # every direction again (after `concept --pick`)
    anim turnaround knight --page-only           # rebuild the page from what is on disk; no GPU
    anim turnaround knight --elevation top       # the top-down cells t0 .. t315 (#285; accepted, not v1)

    work/<unit>_idle_<dir>/                 an ordinary rig -> clip -> pick -> post --parts job per direction
    work/<unit>_idle_<dir>/post/parts.json  THAT direction's part palettes - what the matrix locks to
    work/<unit>_turnaround.json             the plan, one record per direction, the identity table
    work/_turnaround/<unit>/index.html      the gate page (+ img/); cp to the visuals dir

THE ROUTE is the bake-off's ruling (2026-09-21): route V, Quaternius UAL, no
drift lever. Per direction: `rig` samples frame 0 of the UAL's `Idle_Loop`
(the neutral stand, the pose the concept pick shows; `config.STUDIO_ACTIONS`)
into the clip cell at that yaw - `TURNAROUND_CLIP_FRAMES` (5) times at t=0,
so `clip` sends Wan 2.2 Fun-Control a 4n+1 video of one held pose (n=1: the
shortest clip with a temporal window; a 1-frame video is legal but leaves the
video model nothing to denoise against) with the base as the reference,
skeleton over full-body depth at gain 0.35, exactly the bake-off's
V-quaternius cell. The still is the MIDDLE generated frame (`pick` hand-set
to `TURNAROUND_STILL_INDEX`): the first frame of a Wan clip is the one the
VAE encodes on its own and the one that leans on the reference most, the
last is where the padding would sit. The alternative - DWPose on the five
generated frames against the rig's keypoints, keep the closest - costs a
model load per direction for five frames that share one control pose; it is
the `pick` the action matrix will need, not this stage. Then `post --input
clip --parts`: the still is matted, downscaled, snapped and every part LOCKED
to the base's part palettes, and the colours each part shows after the lock
are written as that direction's `post/parts.json` (`anim.parts/1`, the file
`post --parts-from` reads) - so a back-view clip later locks to what the
back-view still showed, not to a front view that never saw the cloak.

THE IDENTITY GATE. Accepting the page flips the curation record's
`turnaround` gate (`anim curate <unit> --accept-gate turnaround`, or the
Board), which is what lets a cell outside s0 render. The page carries the
numbers and no verdict: per part, the pixel count in every direction and its
spread, the colours after the lock, the drift the lock had to move, and
"not found" where SAM 2.1 saw no such part; per direction, the silhouette's
height and width in the 64 px frame - so a broken direction stands out as a
number before it stands out as a picture. The owner rules.

RESUMABLE, PER DIRECTION. A direction whose job has its still, its report
and its parts.json is skipped; `--force` redoes it; `--dirs` names a subset.
One seed for the whole turnaround (`--seed`, else derived from the unit, the
rig clip and the base render's hash - so a new base is a new seed), recorded
in the state file and reused by a resume. Seconds per stage per direction
and the service's VRAM after the GPU stage are in the state file and on the
page. A direction that fails - a timed-out job, a matte with nothing in it -
is recorded with its error and the run goes on; one bad direction must not
cost the other seven.

ONE REFERENCE PER VIEW (the multi-view reference feature, #298). The first
identity gate (2026-09-22) failed on the turned heads: Wan had one front
view and invented every other side. Now each direction hands `clip` the
base's view that already shows that side - `config.TURNAROUND_REFERENCE_BY_DIR`:
s0 the right profile, s180 the left, s90 the back, s270 the front (the base
itself), the obliques their exact view when `concept --pick-view` put one
on the base, else the nearest of back / front. A view the base does not
carry, or one the owner rejected on the Board, falls back down that list to
the front with a WARNING in the log, the record and on the page; the record
(`cells.<dir>.reference`, `references`) and `GET /turnaround/{unit}` say
which reference every still used, and the page shows it under the still.
The prompt lever rides along inside `clip`: with a parts phrase on the base
the prompt names the parts and what this view shows of the head, and every
turned-away cell hides all of its face points (rigkit).

RERUN AFTER A NEW BASE. `concept --pick` reopens the turnaround gate; the
one line is `anim turnaround <unit> --force`. The state file records the
base render's sha256, and a run without `--force` against a changed base
says so rather than mixing stills of two knights.

WHAT IT DOES NOT DO. It renders no top-down cell for v1 (`--elevation top`
is accepted and planned for #285), touches no other job, and changes
nothing in `rig`, `clip`, `pick` or `post` - every direction is those stages
called as a library, and every job can be re-run by hand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from anim import comfy, config, curation, imaging, partkit
from anim.job import Job
from anim.stages import bakeoff as bakeoff_stage
from anim.stages import clip as clip_stage
from anim.stages import pick as pick_stage
from anim.stages import post as post_stage
from anim.stages import rig as rig_stage
from anim.util import read_json, sha256_bytes, sha256_file, write_bytes_in_place, \
    write_json_in_place, write_text_in_place

NAME = "turnaround"
HELP = "one idle still per direction from the base -> the identity gate page (a GPU stage)"

TURNAROUND_SCHEMA = "anim.turnaround/1"
STATE_SUFFIX = "_turnaround.json"
PAGE_ROOT = "_turnaround"
VISUALS_TARGET = "~/.cache/claude/visuals/sprite-studio/turnaround"
VISUALS_URL = "http://192.168.1.158:8090/sprite-studio/turnaround/"
QUESTION = "One character in 8 facings?"
RIGS = bakeoff_stage.RIGS


# ---------------------------------------------------------------------------
# The plan (pure)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    name: str          # the direction cell, e.g. s90
    yaw: int
    elevation: str
    job: str


def job_name(unit: str, direction: str) -> str:
    """`<unit>_idle_<dir>` for every direction, s0 included - the name the
    matrix and the export look a cell up by (export accepts both spellings
    of the default cell)."""
    return f"{unit}_{config.TURNAROUND_ACTION}_{direction}"


def directions_for(elevation: str = config.DEFAULT_ELEVATION) -> list[str]:
    """The eight cells of one elevation, in compass order."""
    if elevation not in config.DIRECTION_ELEVATIONS:
        raise ValueError(f"unknown elevation {elevation!r}; one of "
                         f"{list(config.DIRECTION_ELEVATIONS)}")
    return [name for name, (_yaw, el) in config.DIRECTIONS.items() if el == elevation]


def plan(unit: str, dirs: Sequence[str] | None = None,
         elevation: str = config.DEFAULT_ELEVATION) -> list[Cell]:
    """The cells to render, in compass order; `dirs` a subset of the
    elevation's eight (a cell of the other elevation is refused)."""
    wanted = [d.strip().lower() for d in dirs if d.strip()] if dirs else None
    cells: list[Cell] = []
    for name in directions_for(elevation):
        if wanted is not None and name not in wanted:
            continue
        yaw, el = config.direction(name)
        cells.append(Cell(name=name, yaw=yaw, elevation=el, job=job_name(unit, name)))
    if wanted is not None:
        unknown = sorted(set(wanted) - {c.name for c in cells})
        if unknown:
            raise ValueError(f"unknown direction(s) {unknown} for elevation {elevation!r}; "
                             f"one of {directions_for(elevation)}")
    return cells


def rig_for(rig_name: str = config.TURNAROUND_RIG, glb: Path | None = None) -> tuple[str, Path, str]:
    """(rig name, GLB path, idle clip) - the bake-off's rig table and the
    studio's idle row, or the overrides."""
    if rig_name not in RIGS:
        raise ValueError(f"unknown rig {rig_name!r}; one of {sorted(RIGS)}")
    path = Path(glb) if glb is not None else Path(RIGS[rig_name]["glb"])
    clip_name = config.STUDIO_ACTIONS[config.TURNAROUND_ACTION]["clips"].get(rig_name)
    if not clip_name:
        raise ValueError(f"rig {rig_name!r} names no idle clip in config.STUDIO_ACTIONS")
    return rig_name, path, clip_name


def state_path(unit: str, work_dir: Path | None = None) -> Path:
    base = Path(work_dir) if work_dir is not None else config.WORK_DIR
    return base / f"{unit}{STATE_SUFFIX}"


def page_dir(unit: str, work_dir: Path | None = None) -> Path:
    base = Path(work_dir) if work_dir is not None else config.WORK_DIR
    return base / PAGE_ROOT / unit


def derive_seed(unit: str, clip_name: str, reference_sha256: str) -> int:
    """One seed for the eight stills; a new base render is a new seed."""
    material = "|".join([unit, "turnaround", clip_name, reference_sha256])
    return int(sha256_bytes(material.encode("utf-8"))[:13], 16) % clip_stage.render_stage.SEED_MAX


def base_reference(unit: str, override: Path | None, work_dir: Path | None,
                   assets_dir: Path | None) -> tuple[Path, str]:
    """The identity image `clip` would use for this unit, and how it found it."""
    probe = Job.open(job_name(unit, config.DEFAULT_DIRECTION), work_dir)
    return clip_stage.find_reference(probe, {"slug": unit}, override, assets_dir)


def base_views(unit: str, work_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """The views the base carries (`concept.views` of work/<unit>_base) that
    still have their file, with the curation state of each."""
    base_job = Job.open(f"{unit}_base", work_dir)
    if not base_job.manifest_path.exists():
        return {}
    stored = (base_job.read_manifest().get("concept") or {}).get("views") or {}
    record = curation.read(unit, work_dir) or {}
    states = record.get("views") or {}
    out: dict[str, dict[str, Any]] = {}
    for view, entry in stored.items():
        path = base_job.render_dir / str((entry or {}).get("file") or "")
        if not (entry or {}).get("file") or not path.exists():
            continue
        out[view] = {**entry, "path": path, "state": states.get(view, "pending")}
    return out


def reference_for(direction: str, front: tuple[Path, str], views: dict[str, dict[str, Any]]
                  ) -> dict[str, Any]:
    """Which image this direction hands Wan: the first view down
    `config.TURNAROUND_REFERENCE_BY_DIR` that the base carries and the owner
    has not rejected, else the front (`front` = (path, kind) from
    `base_reference`) with a warning naming what was wanted. Pure."""
    wanted = config.turnaround_reference_views(direction)
    for view in wanted:
        if view == config.FRONT_VIEW:
            break
        entry = views.get(view)
        if entry is None or entry.get("state") == "rejected":
            continue
        path = Path(entry["path"])
        return {"view": view, "kind": f"view {view}", "file": str(path),
                "sha256": sha256_file(path), "wanted": list(wanted),
                "state": entry.get("state"), "provisional": bool(entry.get("provisional")),
                "fallback": False, "warning": None}
    path, kind = front
    missing = [v for v in wanted if v != config.FRONT_VIEW]
    warning = None
    if missing:
        why = ", ".join(
            f"{v} ({'rejected' if (views.get(v) or {}).get('state') == 'rejected' else 'not on the base'})"
            for v in missing)
        warning = (f"{direction}: no view for it - wanted {why}; using the front "
                   f"(`anim concept --views` then `--pick-view` to add one)")
    return {"view": config.FRONT_VIEW, "kind": kind, "file": str(path),
            "sha256": sha256_file(path) if path.exists() else None, "wanted": list(wanted),
            "state": None, "provisional": False, "fallback": bool(missing), "warning": warning}


def references_for(unit: str, cells: Sequence[str], front: tuple[Path, str],
                   work_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    views = base_views(unit, work_dir)
    return {d: reference_for(d, front, views) for d in cells}


def still_file(job: Job) -> Path | None:
    """The direction's finished still: post/<unit>_idle_0.png."""
    manifest = job.read_manifest() if job.manifest_path.exists() else {}
    stem = f"{manifest.get('slug', 'unit')}_{manifest.get('action', config.TURNAROUND_ACTION)}"
    path = job.post_dir / f"{stem}_0.png"
    return path if path.exists() else None


def is_done(job: Job) -> bool:
    """A direction is done when its still, its report and its parts.json exist."""
    return job.manifest_path.exists() and job.report_path.exists() \
        and (job.post_dir / "parts.json").exists() and still_file(job) is not None


# ---------------------------------------------------------------------------
# The direction's part palettes and its numbers (pure over report.json)
# ---------------------------------------------------------------------------
def direction_parts(job: Job, report: dict[str, Any], direction: str,
                    locked_to: str | None) -> dict[str, Any]:
    """`post/parts.json` for one direction from the still's part report: the
    colours each part shows AFTER the lock (a locked part: members of the
    base's palette for it; an unlocked one: the snapped colours it came out
    with), its pixels, found / locked - the file `post --parts-from` reads.
    A part under `config.PART_MIN_PALETTE_PIXELS` pixels is recorded as
    rejected, exactly as the base's extract does."""
    part_names = list((report.get("parts") or {}).get("parts") or [])
    frames = report.get("frames") or []
    palettes: dict[str, list[list[int]]] = {n: [] for n in part_names}
    pixels: dict[str, int] = {n: 0 for n in part_names}
    found: dict[str, bool] = {n: False for n in part_names}
    locked: dict[str, bool] = {n: False for n in part_names}
    drift: dict[str, int | None] = {n: None for n in part_names}
    for frame in frames:
        for name, info in (frame.get("parts") or {}).items():
            if name not in palettes:
                continue
            pixels[name] += int(info.get("pixels") or 0)
            found[name] = found[name] or bool(info.get("found"))
            locked[name] = locked[name] or bool(info.get("locked"))
            if info.get("drift") is not None:
                drift[name] = int(info["drift"]) + (drift[name] or 0)
            for colour in info.get("colours") or []:
                if colour not in palettes[name]:
                    palettes[name].append([int(v) for v in colour])
    too_small = {n: pixels[n] for n in part_names
                 if 0 < pixels[n] < config.PART_MIN_PALETTE_PIXELS}
    record = {
        "schema": partkit.PARTS_SCHEMA, "slug": report.get("slug"), "job": job.name,
        "direction": direction, "model": partkit.model_entry(), "parts": part_names,
        "palettes": {n: (cols if n not in too_small else []) for n, cols in palettes.items()},
        "pixels": pixels, "found": found, "locked": locked, "drift": drift,
        "locked_to": locked_to,
        "rejected": {n: {"pixels": px, "colours": palettes[n],
                         "reason": f"under {config.PART_MIN_PALETTE_PIXELS} px on the still"}
                     for n, px in too_small.items()},
        "_note": ("The turnaround feature: this direction's part palettes - the "
                  "colours each part shows on the accepted still after the lock to "
                  "the base. `anim post <job> --parts-from` this file for a clip of "
                  "the same direction. Colours are [r, g, b] members of the palette."),
    }
    write_json_in_place(job.post_dir / "parts.json", record)
    return record


def silhouette_of(still: Path) -> dict[str, Any]:
    """Width, height and opaque pixels of the still's opaque box."""
    with Image.open(still) as img:
        rgba = img.convert("RGBA")
    box = imaging.opaque_box(rgba)
    alpha = np.asarray(rgba)[:, :, 3]
    if box is None:
        return {"width": 0, "height": 0, "pixels": 0, "colours": 0}
    x0, y0, x1, y1 = box
    return {"width": int(x1 - x0), "height": int(y1 - y0),
            "pixels": int((alpha > 0).sum()),
            "colours": len(imaging.opaque_colours(rgba))}


def direction_metrics(job: Job, report: dict[str, Any], parts: dict[str, Any]) -> dict[str, Any]:
    frames = report.get("frames") or []
    good = [f for f in frames if not f.get("problems")]
    still = still_file(job)
    return {
        "frames": len(good), "frames_attempted": len(frames),
        "silhouette": silhouette_of(still) if still else None,
        "parts": {name: {"pixels": parts["pixels"].get(name, 0),
                         "found": parts["found"].get(name, False),
                         "locked": parts["locked"].get(name, False),
                         "drift": parts["drift"].get(name),
                         "colours": len(parts["palettes"].get(name) or [])}
                  for name in parts["parts"]},
    }


def identity_table(records: dict[str, dict[str, Any]], order: Sequence[str]) -> dict[str, Any]:
    """The numbers across directions - no verdict. Per part: the pixel count
    in each direction, its spread (max - min over the directions the part was
    found in), the colours after the lock and the drift; per direction, the
    silhouette. A direction that did not finish is a None column."""
    done = {d: records[d] for d in order if (records.get(d) or {}).get("status") == "done"
            and records[d].get("metrics")}
    part_names: list[str] = []
    for r in done.values():
        for n in r["metrics"]["parts"]:
            if n not in part_names:
                part_names.append(n)
    parts: dict[str, Any] = {}
    for name in part_names:
        pixels = {d: (r["metrics"]["parts"].get(name) or {}).get("pixels") for d, r in done.items()}
        found = [d for d, r in done.items() if (r["metrics"]["parts"].get(name) or {}).get("found")]
        counts = [pixels[d] for d in found]
        parts[name] = {
            "pixels": {d: pixels.get(d) for d in order},
            "found": found,
            "missing": [d for d in done if d not in found],
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
            "mean": round(sum(counts) / len(counts)) if counts else None,
            "spread": (max(counts) - min(counts)) if counts else None,
            "colours": {d: (r["metrics"]["parts"].get(name) or {}).get("colours") for d, r in done.items()},
            "drift": {d: (r["metrics"]["parts"].get(name) or {}).get("drift") for d, r in done.items()},
            "locked": {d: (r["metrics"]["parts"].get(name) or {}).get("locked") for d, r in done.items()},
        }
    silhouette = {d: r["metrics"].get("silhouette") for d, r in done.items()}
    heights = [s["height"] for s in silhouette.values() if s]
    widths = [s["width"] for s in silhouette.values() if s]
    return {
        "directions": list(order), "done": list(done),
        "parts": parts,
        "silhouette": {d: silhouette.get(d) for d in order},
        "height": {"min": min(heights), "max": max(heights), "spread": max(heights) - min(heights)} if heights else None,
        "width": {"min": min(widths), "max": max(widths), "spread": max(widths) - min(widths)} if widths else None,
    }


# ---------------------------------------------------------------------------
# One direction
# ---------------------------------------------------------------------------
def _hand_set_picks(job: Job, picks: list[int], *, verb: str) -> None:
    """`pick` as usual (so picks.json has every field the board, the export
    and `post` read), then the still's frame in force with the picker's
    choice kept beside it - the same hand-set a review click makes."""
    record = pick_stage.pick(job.name, n=len(picks), action=verb, force=True,
                             phases=False, work_dir=job.root.parent)
    record["picks"] = list(picks)
    write_json_in_place(job.picks_path, record)
    job.update_manifest(picks=list(picks))


def run_direction(cell: Cell, *, unit: str, glb: Path, clip_name: str, seed: int,
                  matte: str, parts_from: Path | None, client: comfy.ComfyClient,
                  segmenter: partkit.Segmenter | None, reference: dict[str, Any],
                  work_dir: Path | None, assets_dir: Path | None,
                  workflow_dir: Path | None, lock_path: Path | None,
                  timeout_s: float) -> dict[str, Any]:
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    job = rig_stage.rig(glb, clip=clip_name, action=config.TURNAROUND_ACTION, slug=unit,
                        job_name=cell.job, cell=tuple(config.TURNAROUND_CELL), depth="full",
                        yaw=cell.yaw, elevation=cell.elevation,
                        times=[0.0] * config.TURNAROUND_CLIP_FRAMES, work_dir=work_dir)
    timings["rig"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    # A view goes to `clip` as its reference override; the front is left to
    # `clip.find_reference`, so the clip record says "base render" for it
    # unless the owner passed --reference for the whole run.
    override = (Path(reference["file"]) if reference.get("override")
                or reference["view"] != config.FRONT_VIEW else None)
    record = clip_stage.clip(job.name, seed=seed, control=config.TURNAROUND_CONTROL,
                             depth_gain=config.TURNAROUND_DEPTH_GAIN, reference=override,
                             client=client, workflow_dir=workflow_dir, lock_path=lock_path,
                             work_dir=work_dir, assets_dir=assets_dir, timeout_s=timeout_s)
    timings["gpu"] = round(time.monotonic() - t0, 1)
    vram = bakeoff_stage._vram_used_gb(client)

    _hand_set_picks(job, [config.TURNAROUND_STILL_INDEX], verb=config.TURNAROUND_ACTION)

    t0 = time.monotonic()
    post_error: str | None = None
    try:
        post_stage.post(job.name, matte=matte, source=post_stage.CLIP_SOURCE, parts=True,
                        parts_from=parts_from, segmenter=segmenter,
                        assets_dir=assets_dir, work_dir=work_dir)
    except RuntimeError as exc:      # verification failures: the report is on disk
        post_error = str(exc).splitlines()[0]
    timings["post"] = round(time.monotonic() - t0, 1)
    report = read_json(job.report_path) if job.report_path.exists() else {}
    locked_to = ((report.get("parts") or {}).get("palettes_from"))
    parts = direction_parts(job, report, cell.name, locked_to)
    if still_file(job) is None:
        raise RuntimeError(post_error or f"post wrote no still for {cell.name}")
    return {"job": job.name, "timings": timings, "gpu_seconds": round(float(record.get("seconds") or 0.0), 1),
            "vram_used_gb": vram, "seed": seed, "still_frame": config.TURNAROUND_STILL_INDEX,
            "clip_frames": int(record["frames"]), "control": record["control"],
            "depth_gain": record["depth_gain"], "prompt": record.get("prompt"),
            "negative": record.get("negative"), "parts_phrase": record.get("parts_phrase"),
            "reference": {k: v for k, v in reference.items() if k != "path"},
            "hidden_face": list((read_json(job.keypoints_path)["frames"][0].get("hidden") or [])),
            "post_error": post_error, "locked_to": locked_to,
            "metrics": direction_metrics(job, report, parts)}


def run_cell(cell: Cell, **kw: Any) -> dict[str, Any]:
    """One direction end to end; the record goes into the state file and the
    job's job.json (`turnaround` block) so a re-run can tell it is done."""
    started = time.monotonic()
    record: dict[str, Any] = {**asdict(cell), "status": "running", "started": _now()}
    try:
        record.update(run_direction(cell, **kw))
        record["status"] = "done"
    except Exception as exc:  # noqa: BLE001 - one bad direction must not cost the page
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["seconds"] = round(time.monotonic() - started, 1)
    job = Job.open(cell.job, kw.get("work_dir"))
    if job.manifest_path.exists():
        job.update_manifest(turnaround={"status": record["status"], "direction": cell.name,
                                        "seconds": record["seconds"],
                                        "error": record.get("error")})
    return record


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------
def turnaround(unit: str, *, dirs: Sequence[str] | None = None,
               elevation: str = config.DEFAULT_ELEVATION, seed: int | None = None,
               force: bool = False, page_only: bool = False, matte: str = "rembg",
               parts_from: Path | None = None, rig_name: str = config.TURNAROUND_RIG,
               glb: Path | None = None, clip_name: str | None = None,
               reference: Path | None = None, client: comfy.ComfyClient | None = None,
               segmenter: partkit.Segmenter | None = None, work_dir: Path | None = None,
               assets_dir: Path | None = None, workflow_dir: Path | None = None,
               lock_path: Path | None = None, timeout_s: float | None = None,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    planned = plan(unit, dirs, elevation)
    order = directions_for(elevation)
    rig_name, glb, default_clip = rig_for(rig_name, glb)
    clip_name = clip_name or default_clip
    timeout_s = config.TURNAROUND_TIMEOUT_S if timeout_s is None else float(timeout_s)

    path = state_path(unit, work_dir)
    state = read_json(path) if path.exists() else {}
    records: dict[str, Any] = dict(state.get("cells") or {})
    references: dict[str, dict[str, Any]] = dict(state.get("references") or {})
    if page_only and state:
        seed = int(state.get("seed", seed or 0))
        matte = str(state.get("matte", matte))
        parts_from = Path(state["parts_from"]) if state.get("parts_from") else parts_from
        reference_row = state.get("reference") or {}
    else:
        ref_path, ref_kind = base_reference(unit, reference, work_dir, assets_dir)
        reference_row = {"file": str(ref_path), "kind": ref_kind,
                         "sha256": sha256_file(ref_path) if ref_path.exists() else None}
        # The multi-view reference feature: one reference per direction. An
        # explicit --reference is the front for every cell, views included -
        # it is the owner's override.
        planned_refs = references_for(unit, [c.name for c in planned], (ref_path, ref_kind),
                                      work_dir) if reference is None else {
            c.name: {"view": config.FRONT_VIEW, "kind": ref_kind, "file": str(ref_path),
                     "sha256": reference_row["sha256"], "wanted": [config.FRONT_VIEW],
                     "state": None, "provisional": False, "fallback": False,
                     "warning": None, "override": True} for c in planned}
        for name, row in planned_refs.items():
            if row.get("warning"):
                log(f"turnaround: WARNING {row['warning']}")
        if seed is None:
            seed = int(state["seed"]) if state.get("seed") and not force \
                else derive_seed(unit, clip_name, reference_row["sha256"] or "")
        previous = (state.get("reference") or {}).get("sha256")
        if previous and previous != reference_row["sha256"] and not force:
            raise ValueError(
                f"{unit}: the base render changed since this turnaround was rendered "
                f"({previous[:12]} -> {str(reference_row['sha256'])[:12]}); run "
                f"`anim turnaround {unit} --force` to re-render every direction "
                f"from the new base rather than mix two")

    if not page_only:
        client = client or comfy.ComfyClient()
        segmenter = segmenter or partkit.Sam2Segmenter()
        for cell in planned:
            job = Job.open(cell.job, work_dir)
            if is_done(job) and not force:
                log(f"turnaround: {cell.name} done -> skipped (--force redoes it)")
                records.setdefault(cell.name, {**asdict(cell), "status": "done"})
                continue
            ref = planned_refs[cell.name]
            log(f"turnaround: {cell.name} (yaw {cell.yaw}, {cell.elevation}; {clip_name} "
                f"frame 0 x{config.TURNAROUND_CLIP_FRAMES}; reference {ref['view']}"
                f"{' - fallback' if ref['fallback'] else ''}) ...")
            record = run_cell(cell, unit=unit, glb=glb, clip_name=clip_name, seed=seed,
                              matte=matte, parts_from=parts_from, client=client,
                              segmenter=segmenter, reference=ref, work_dir=work_dir,
                              assets_dir=assets_dir, workflow_dir=workflow_dir,
                              lock_path=lock_path, timeout_s=timeout_s)
            record.setdefault("reference", {k: v for k, v in ref.items() if k != "path"})
            records[cell.name] = record
            references[cell.name] = record["reference"]
            if record["status"] == "done":
                m = record["metrics"]
                sil = m.get("silhouette") or {}
                found = sum(1 for p in m["parts"].values() if p["found"])
                log(f"  {cell.name}: still {sil.get('width')}x{sil.get('height')} px, "
                    f"{found}/{len(m['parts'])} part(s) found, gpu {record['gpu_seconds']} s, "
                    f"total {record['seconds']} s"
                    + (f", post: {record['post_error']}" if record.get("post_error") else ""))
            else:
                log(f"  {cell.name}: FAILED after {record['seconds']} s - {record['error']}")
            write_json_in_place(path, _state(unit, elevation, rig_name, glb, clip_name, seed,
                                             matte, parts_from, reference_row, order, records,
                                             references))

    state = _state(unit, elevation, rig_name, glb, clip_name, seed, matte, parts_from,
                   reference_row, order, records, references)
    write_json_in_place(path, state)
    page = build_page(state, page_dir(unit, work_dir), work_dir=work_dir)
    state["page"] = str(page)
    return state


def _state(unit: str, elevation: str, rig_name: str, glb: Path, clip_name: str,
           seed: int | None, matte: str, parts_from: Path | None,
           reference: dict[str, Any], order: Sequence[str],
           records: dict[str, Any], references: dict[str, Any] | None = None
           ) -> dict[str, Any]:
    references = references or {}
    return {
        "schema": TURNAROUND_SCHEMA, "unit": unit, "action": config.TURNAROUND_ACTION,
        "elevation": elevation, "rig": {"name": rig_name, "glb": str(glb), "clip": clip_name},
        "clip_frames": config.TURNAROUND_CLIP_FRAMES, "still_frame": config.TURNAROUND_STILL_INDEX,
        "control": config.TURNAROUND_CONTROL, "depth_gain": config.TURNAROUND_DEPTH_GAIN,
        "cell": list(config.TURNAROUND_CELL), "seed": seed, "matte": matte,
        "parts_from": str(parts_from) if parts_from else None, "reference": reference,
        # The multi-view reference feature: the reference each direction used.
        "references": {d: references[d] for d in order if d in references},
        "reference_map": {d: list(config.turnaround_reference_views(d)) for d in order},
        "question": QUESTION, "updated": _now(), "plan": list(order),
        "cells": {d: records[d] for d in order if d in records},
        "identity": identity_table(records, order),
    }


# ---------------------------------------------------------------------------
# What the API answers: GET /turnaround/{unit}
# ---------------------------------------------------------------------------
def listing(unit: str, work_dir: Path | None = None) -> dict[str, Any] | None:
    """The state file plus, per direction, its status and the URLs of the
    files the page shows (through the jobs API's frame sets), or None when
    the unit has no turnaround yet."""
    path = state_path(unit, work_dir)
    if not path.exists():
        return None
    state = read_json(path)
    still = int(state.get("still_frame", config.TURNAROUND_STILL_INDEX))
    rows: dict[str, Any] = {}
    for direction in state.get("plan") or []:
        record = (state.get("cells") or {}).get(direction) or {}
        job = Job.open(job_name(unit, direction), work_dir)
        done = record.get("status") == "done" and still_file(job) is not None
        rows[direction] = {
            "direction": direction, "job": job.name,
            "status": record.get("status") or "not run",
            "error": record.get("error"), "seconds": record.get("seconds"),
            "gpu_seconds": record.get("gpu_seconds"), "timings": record.get("timings"),
            "still": f"/jobs/{job.name}/frames/post/0.png" if done else None,
            "raw": f"/jobs/{job.name}/frames/clip/{still}.png" if job.clip_frames() else None,
            "control": f"/jobs/{job.name}/frames/control/{still}.png"
            if (job.clip_dir / "control").is_dir() else None,
            "skeleton": f"/jobs/{job.name}/frames/pose/0.png" if job.skeletons() else None,
            "depth": f"/jobs/{job.name}/frames/depth/0.png" if job.depths() else None,
            "parts": (record.get("metrics") or {}).get("parts"),
            "silhouette": (record.get("metrics") or {}).get("silhouette"),
            "hidden_face": record.get("hidden_face"),
            "prompt": record.get("prompt"),
            "reference": _reference_row(record.get("reference") or
                                        (state.get("references") or {}).get(direction), job),
        }
    done_count = sum(1 for r in rows.values() if r["status"] == "done")
    return {**state, "path": str(path), "page": str(page_dir(unit, work_dir) / "index.html"),
            "directions": rows, "done": done_count, "of": len(rows)}


def _reference_row(ref: dict[str, Any] | None, job: Job) -> dict[str, Any] | None:
    """The reference a direction used, with the URL of the image `clip`
    actually sent (fitted to the cell), for the API and the Board."""
    if not ref:
        return None
    sent = job.clip_dir / "reference.png"
    return {"view": ref.get("view"), "kind": ref.get("kind"), "file": ref.get("file"),
            "fallback": bool(ref.get("fallback")), "warning": ref.get("warning"),
            "provisional": bool(ref.get("provisional")), "state": ref.get("state"),
            "url": f"/jobs/{job.name}/frames/reference/0.png" if sent.exists() else None}


# ---------------------------------------------------------------------------
# The gate page
# ---------------------------------------------------------------------------
_STYLE = """
:root { color-scheme: dark; }
body { background: #14161b; color: #d8dee9; font: 14px/1.45 -apple-system, BlinkMacSystemFont,
       "Segoe UI", sans-serif; margin: 0; padding: 24px; max-width: 1800px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 36px 0 8px; color: #9fb0c9; border-top: 1px solid #2a2f3a;
     padding-top: 14px; }
.question { font-size: 22px; color: #f0d27a; margin: 14px 0 6px; }
p.caption, .note { color: #8b96a5; }
code { color: #cdd; }
.ring { display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px; align-items: start; }
.dir { background: #1c1f26; border: 1px solid #2a2f3a; border-radius: 6px; padding: 8px; }
.dir.failed { border-color: #a04040; }
.dir .name { font-weight: 600; color: #e8ecf1; font-size: 15px; }
.dir .facing { color: #8b96a5; font-size: 12px; margin-bottom: 6px; }
.dir img.px { width: 100%; max-width: 256px; image-rendering: pixelated; background: #2a2d36;
              border: 1px solid #333; display: block; }
.dir .hd { display: flex; gap: 4px; margin-top: 6px; }
.dir .hd div { flex: 1; }
.dir .hd img { width: 100%; background: #fff; border: 1px solid #333; display: block; }
.dir .hd span { display: block; font-size: 10px; color: #6b7686; text-align: center; }
.dir .miss { border: 1px dashed #444; color: #666; font-size: 11px; padding: 40px 0;
             text-align: center; }
.dir .secs { color: #8b96a5; font-size: 11px; margin-top: 6px; }
.dir .refrow { display: flex; gap: 6px; align-items: center; margin-top: 6px; font-size: 11px;
               color: #8b96a5; }
.dir .refrow img { width: 40px; height: 60px; object-fit: contain; background: #fff;
                   border: 1px solid #333; }
.dir .refrow.fallback { color: #e0b060; }
.dir table { font-size: 11px; margin: 6px 0 0; width: 100%; }
.side { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
.side figure { margin: 0; }
.side figcaption { color: #8b96a5; font-size: 11px; margin-top: 4px; }
.turntable { width: 256px; height: 256px; image-rendering: pixelated; background-repeat: no-repeat;
             background-size: 2048px 256px; background-color: #2a2d36; border: 1px solid #333;
             animation: turn 2s steps(8) infinite; }
@keyframes turn { from { background-position: 0 0; } to { background-position: -2048px 0; } }
img.ref { height: 256px; background: #fff; border: 1px solid #333; }
table { border-collapse: collapse; font-size: 12px; margin: 8px 0 14px; }
th { color: #999; text-align: left; padding: 3px 8px; border-bottom: 1px solid #333;
     font-weight: 600; }
td { padding: 3px 8px; border-bottom: 1px solid #222; vertical-align: top; }
td.n { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: #8fd18f; } .bad { color: #f08080; } .warn { color: #e0b060; }
.toggle { font-size: 12px; color: #8b96a5; margin: 6px 0; }
body.parts .dir img.hd-raw { display: none; }
body:not(.parts) .dir img.hd-ov { display: none; }
footer { margin-top: 32px; color: #6b7686; font-size: 12px; }
"""

FACINGS: dict[str, str] = {
    "s0": "screen right — the profile game-02 ships", "s45": "away-right", "s90": "away — the back",
    "s135": "away-left", "s180": "screen left", "s225": "toward-left",
    "s270": "the viewer — the front", "s315": "toward-right",
    "t0": "east", "t45": "north-east", "t90": "north", "t135": "north-west",
    "t180": "west", "t225": "south-west", "t270": "south", "t315": "south-east",
}


def _save(img: Image.Image, dest: Path) -> Path:
    from io import BytesIO

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return write_bytes_in_place(dest, buffer.getvalue())


def _copy(src: Path, dest: Path) -> str | None:
    if not src.exists():
        return None
    write_bytes_in_place(dest, src.read_bytes())
    return dest.name


def collect_images(state: dict[str, Any], img_dir: Path, *, work_dir: Path | None
                   ) -> dict[str, Any]:
    """The still, the raw clip frame, the control frame and the part overlay
    of every finished direction into the page's img/, plus the reference
    and the 8-frame turntable sheet. Every write truncates in place."""
    unit = state["unit"]
    still_index = int(state.get("still_frame", config.TURNAROUND_STILL_INDEX))
    out: dict[str, Any] = {"dirs": {}, "reference": None, "turntable": None}
    ref = (state.get("reference") or {}).get("file")
    if ref and Path(ref).exists():
        out["reference"] = _copy(Path(ref), img_dir / "reference.png")
    frames: list[Image.Image] = []
    for direction in state["plan"]:
        job = Job.open(job_name(unit, direction), work_dir)
        row: dict[str, str | None] = {"still": None, "raw": None, "control": None, "parts": None}
        still = still_file(job)
        if still is not None:
            row["still"] = _copy(still, img_dir / f"{direction}_still.png")
            with Image.open(still) as img:
                frames.append(img.convert("RGBA").copy())
        else:
            frames.append(Image.new("RGBA", tuple(config.CANVAS), (0, 0, 0, 0)))
        row["raw"] = _copy(job.clip_frames_dir / f"{still_index + 1:04d}.png",
                           img_dir / f"{direction}_raw.png")
        row["control"] = _copy(job.clip_dir / "control" / f"{still_index + 1:04d}.png",
                               img_dir / f"{direction}_control.png")
        row["reference"] = _copy(job.clip_dir / "reference.png", img_dir / f"{direction}_ref.png")
        manifest = job.read_manifest() if job.manifest_path.exists() else {}
        stem = f"{manifest.get('slug', unit)}_{manifest.get('action', config.TURNAROUND_ACTION)}"
        row["parts"] = _copy(job.post_dir / "parts" / f"{stem}_0_hd.png",
                             img_dir / f"{direction}_parts.png")
        out["dirs"][direction] = row
    if any(f.getbbox() for f in frames):
        canvas = tuple(config.CANVAS)
        sheet = Image.new("RGBA", (canvas[0] * len(frames), canvas[1]), (0, 0, 0, 0))
        for i, frame in enumerate(frames):
            sheet.paste(frame, (i * canvas[0], 0))
        out["turntable"] = _save(sheet, img_dir / "turntable.png").name
    return out


def _fmt(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{value}{suffix}"


def _dir_html(direction: str, record: dict[str, Any], images: dict[str, Any]) -> str:
    status = record.get("status", "not run")
    facing = FACINGS.get(direction, direction)
    if status != "done":
        return (f'<div class="dir failed" id="{direction}"><div class="name">{direction}</div>'
                f'<div class="facing">{facing}</div><div class="miss">{status}'
                f'{"<br>" + record["error"] if record.get("error") else ""}</div></div>')
    m = record.get("metrics") or {}
    t = record.get("timings") or {}
    sil = m.get("silhouette") or {}
    rows = "".join(
        f'<tr><td>{name}</td>'
        + (f'<td class="n">{p["pixels"]}</td><td class="n">{p["colours"]}</td>'
           f'<td class="n">{_fmt(p["drift"])}</td>'
           f'<td class="{"ok" if p["locked"] else "warn"}">{"locked" if p["locked"] else "free"}</td>'
           if p["found"] else '<td colspan="4" class="bad">not found</td>')
        + '</tr>'
        for name, p in (m.get("parts") or {}).items())
    raw = (f'<img class="hd hd-raw" src="img/{images["raw"]}" alt="{direction} raw">'
           if images.get("raw") else '<div class="miss">—</div>')
    if images.get("parts"):
        raw += f'<img class="hd hd-ov" src="img/{images["parts"]}" alt="{direction} parts">'
    control = (f'<img class="hd" src="img/{images["control"]}" alt="{direction} control">'
               if images.get("control") else '<div class="miss">—</div>')
    still = (f'<img class="px" src="img/{images["still"]}" alt="{direction} still">'
             if images.get("still") else '<div class="miss">no still</div>')
    hidden = record.get("hidden_face") or []
    ref = record.get("reference") or {}
    ref_img = (f'<img src="img/{images["reference"]}" alt="{direction} reference">'
               if images.get("reference") else "")
    ref_text = (f'reference: <b>{ref.get("view", "?")}</b>'
                + (' <span class="warn">fallback</span>' if ref.get("fallback") else "")
                + (' (provisional)' if ref.get("provisional") else "")
                + (f' — {ref.get("state")}' if ref.get("state") else ""))
    ref_row = (f'<div class="refrow{" fallback" if ref.get("fallback") else ""}" '
               f'title="{ref.get("warning") or ref.get("file") or ""}">{ref_img}<span>{ref_text}</span></div>'
               if ref else "")
    return f"""
<div class="dir" id="{direction}">
  <div class="name">{direction}</div>
  <div class="facing">yaw {record.get('yaw')} — {facing}</div>
  {still}
  {ref_row}
  <div class="hd"><div>{raw}<span>raw frame {record.get('still_frame')}</span></div>
                  <div>{control}<span>control (skeleton + depth)</span></div></div>
  <div class="secs">{sil.get('width')}×{sil.get('height')} px, {sil.get('pixels')} opaque, {sil.get('colours')} colours
    · rig {t.get('rig')} s · GPU {record.get('gpu_seconds')} s ({t.get('gpu')} s wall) · post {t.get('post')} s
    · {_fmt(record.get('vram_used_gb'), ' GB')}
    {'· face hidden: ' + ', '.join(hidden) if hidden else '· face drawn'}
    {'· <span class="warn">' + record['post_error'] + '</span>' if record.get('post_error') else ''}</div>
  <table><tr><th>part</th><th>px</th><th>colours</th><th>drift</th><th></th></tr>{rows}</table>
</div>"""


def _references_html(state: dict[str, Any]) -> str:
    """Per direction: the view wanted, the view used, fallback or not; and
    the prompt lever - the parts phrase the clips named, if any."""
    refs = state.get("references") or {}
    rows = []
    for d in state.get("plan") or []:
        r = refs.get(d) or (state.get("cells", {}).get(d) or {}).get("reference") or {}
        wanted = " → ".join((state.get("reference_map") or {}).get(d) or
                            config.turnaround_reference_views(d))
        if not r:
            rows.append(f"<tr><td><b>{d}</b></td><td>{wanted}</td><td>—</td><td></td></tr>")
            continue
        used = (f'<b>{r.get("view")}</b>' + (' <span class="warn">fallback to the front</span>'
                                             if r.get("fallback") else "")
                + (' <span class="warn">(provisional pick)</span>' if r.get("provisional") else ""))
        note = r.get("warning") or f'{r.get("kind")} · sha256 {str(r.get("sha256") or "")[:12]}'
        rows.append(f"<tr><td><b>{d}</b></td><td>{wanted}</td><td>{used}</td><td>{note}</td></tr>")
    cells = state.get("cells") or {}
    phrase = next((c.get("parts_phrase") for c in cells.values() if c.get("parts_phrase")), None)
    faces = ", ".join(f"{d}: {'all' if len(c.get('hidden_face') or []) == 5 else ', '.join(c.get('hidden_face') or []) or 'drawn'}"
                      for d, c in cells.items() if c.get("status") == "done")
    return (f'<table><tr><th>direction</th><th>wanted (best first)</th><th>used</th><th></th></tr>'
            f'{"".join(rows)}</table>'
            f'<p class="note"><b>parts phrase</b> in every prompt: <i>{phrase or "none - the base carries no parts phrase; the old prompt, word for word"}</i>. '
            f'<b>face points</b> hidden: {faces or "—"}.</p>')


def _identity_html(identity: dict[str, Any]) -> str:
    order = identity.get("directions") or []
    head = "".join(f"<th>{d}</th>" for d in order)
    rows = []
    for name, p in (identity.get("parts") or {}).items():
        cells = "".join(
            f'<td class="n">{_fmt(p["pixels"].get(d))}</td>' if d in p["found"]
            else f'<td class="{"bad" if d in p["missing"] else ""}">'
                 f'{"not found" if d in p["missing"] else "—"}</td>'
            for d in order)
        rows.append(f"<tr><td><b>{name}</b></td>{cells}<td class=\"n\">{_fmt(p['min'])}–{_fmt(p['max'])}</td>"
                    f"<td class=\"n\">{_fmt(p['spread'])}</td><td class=\"n\">{len(p['found'])} / {len(identity.get('done') or [])}</td>"
                    f"<td class=\"n\">{sum(v for v in p['drift'].values() if v)}</td></tr>")
    sil = identity.get("silhouette") or {}
    h = identity.get("height") or {}
    w = identity.get("width") or {}
    sil_rows = (
        "<tr><td><b>height</b></td>" + "".join(f'<td class="n">{_fmt((sil.get(d) or {}).get("height"))}</td>' for d in order)
        + f'<td class="n">{_fmt(h.get("min"))}–{_fmt(h.get("max"))}</td><td class="n">{_fmt(h.get("spread"))}</td><td></td><td></td></tr>'
        "<tr><td><b>width</b></td>" + "".join(f'<td class="n">{_fmt((sil.get(d) or {}).get("width"))}</td>' for d in order)
        + f'<td class="n">{_fmt(w.get("min"))}–{_fmt(w.get("max"))}</td><td class="n">{_fmt(w.get("spread"))}</td><td></td><td></td></tr>'
        "<tr><td><b>opaque px</b></td>" + "".join(f'<td class="n">{_fmt((sil.get(d) or {}).get("pixels"))}</td>' for d in order)
        + "<td></td><td></td><td></td><td></td></tr>")
    return (f"<table><tr><th>part · pixels on the 64 px still</th>{head}<th>min–max</th><th>spread</th>"
            f"<th>found</th><th>drift px</th></tr>{''.join(rows)}{sil_rows}</table>")


def build_page(state: dict[str, Any], out_dir: Path, *, work_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir)
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(state, img_dir, work_dir=work_dir)
    cells = "".join(_dir_html(d, state["cells"].get(d) or {"status": "not run"},
                              images["dirs"].get(d) or {})
                    for d in state["plan"])
    done = [d for d in state["plan"] if (state["cells"].get(d) or {}).get("status") == "done"]
    ref = state.get("reference") or {}
    rig = state.get("rig") or {}
    turntable = (f'<figure><div class="turntable" style="background-image:url(img/{images["turntable"]})"></div>'
                 f'<figcaption>the eight stills as a turntable, 4 fps (blank = not rendered)</figcaption></figure>'
                 if images.get("turntable") else "")
    reference = (f'<figure><img class="ref" src="img/{images["reference"]}" alt="the base">'
                 f'<figcaption>the reference: {ref.get("kind")} — <code>{Path(ref.get("file", "")).name}</code>, '
                 f'sha256 {str(ref.get("sha256") or "")[:12]}</figcaption></figure>'
                 if images.get("reference") else "")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>the turnaround — {state['unit']}</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>The turnaround — {state['unit']}, {len(done)} of {len(state['plan'])} {state.get('elevation')} facings</h1>
<div class="question">{state.get('question', QUESTION)}</div>
<p class="caption">{state.get('updated', '')} · seed <b>{state['seed']}</b> in every direction ·
rig <code>{rig.get('name')}</code> / <code>{rig.get('clip')}</code> frame 0, held for {state.get('clip_frames')} frames, the still is frame {state.get('still_frame')} ·
control <code>{state.get('control')}</code> at depth gain {state.get('depth_gain')} · cell {state.get('cell')} · matte <code>{state['matte']}</code> ·
part lock <code>--parts-from {state.get('parts_from') or 'the base'}</code> ·
<a href="http://192.168.1.158:8700/work_packages/275" style="color:#7aa2d6">#275</a>.
The ruling is the owner's: accept with <code>anim curate {state['unit']} --accept-gate turnaround</code> or the Board; a new base means <code>anim turnaround {state['unit']} --force</code>.</p>
<div class="side">{reference}{turntable}</div>
<h2>The reference each still used — the multi-view reference feature (#298)</h2>
{_references_html(state)}
<div class="toggle"><label><input type="checkbox" id="parts"> show the part masks (SAM 2.1 overlays) on the raw frames</label></div>
<h2>The eight facings, compass order (yaw counter-clockwise seen from above)</h2>
<div class="ring">{cells}</div>
<h2>The identity numbers — no verdict</h2>
{_identity_html(state.get('identity') or {})}
<p class="note"><b>pixels</b>: the part's pixels on the 64 px still, from the same SAM 2.1 masks the part palette lock used, prompted from the rig; a part hidden by the pose (a shield behind the body at s45) is honestly <b>not found</b>.
<b>spread</b>: max − min over the directions the part was found in — a part that changes size between facings more than the view explains is the number to look at.
<b>drift</b>: pixels the lock had to move to a member of that part's palette on the base — 0 means the render already agreed with the base.
<b>free</b>: the base has no palette for that part (under {config.PART_MIN_PALETTE_PIXELS} px, or not found there), so the still keeps its snapped colours and this direction's <code>parts.json</code> records them.
<b>height / width</b>: the still's opaque box in the 64 px frame; the scale ladder puts the {state['unit']} at {config.height_for_unit(state['unit'])} px tall.</p>
<footer>Generated by <code>anim turnaround {state['unit']}</code>. Every direction is
<code>rig --yaw Y --elevation {state.get('elevation')} → clip → pick → post --input clip --parts</code> called as library functions;
the jobs are <code>work/{state['unit']}_idle_&lt;dir&gt;/</code>, the record <code>work/{state['unit']}_turnaround.json</code>.</footer>
<script>
document.getElementById('parts').addEventListener('change', e => document.body.classList.toggle('parts', e.target.checked));
</script>
</body>
</html>
"""
    return write_text_in_place(out_dir / "index.html", html)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("unit", help="unit slug, e.g. knight")
    parser.add_argument("--dirs", default=None,
                        help="comma-separated subset of the elevation's eight cells, "
                             "e.g. s90,s270 (default: all eight)")
    parser.add_argument("--elevation", choices=config.DIRECTION_ELEVATIONS,
                        default=config.DEFAULT_ELEVATION,
                        help="'side' (default, v1) or 'top' (the top-down pitch, #285)")
    parser.add_argument("--seed", type=int, default=None,
                        help="one seed for every direction (default: the state file's, "
                             "else derived from the unit, the clip and the base render)")
    parser.add_argument("--force", action="store_true",
                        help="redo directions whose still exists (the one line after "
                             "`concept --pick`); also skips the curation gate")
    parser.add_argument("--page-only", action="store_true",
                        help="rebuild the page from what is on disk; no GPU")
    parser.add_argument("--rig", default=config.TURNAROUND_RIG, choices=sorted(RIGS),
                        help=f"the rig (default {config.TURNAROUND_RIG}, the bake-off ruling)")
    parser.add_argument("--glb", default=None, metavar="FILE",
                        help="the rig file (default: the rig's vendored GLB)")
    parser.add_argument("--clip", default=None,
                        help="the idle clip (default: the rig's idle row in config.STUDIO_ACTIONS)")
    parser.add_argument("--matte", default="rembg", choices=sorted(post_stage.MATTES))
    parser.add_argument("--parts-from", dest="parts_from", default=None, metavar="FILE",
                        help="part palettes to lock to (default: the unit's sidecar, "
                             "then work/<unit>_base/post/parts.json - see anim post)")
    parser.add_argument("--reference", default=None,
                        help="identity image (default: work/<unit>_base/render/<final>)")
    parser.add_argument("--timeout", type=float, default=None,
                        help=f"seconds to wait for one direction's GPU job (default "
                             f"{config.TURNAROUND_TIMEOUT_S:.0f}); past it the direction "
                             f"is recorded as failed and the run goes on")


def run_stage(args: argparse.Namespace) -> int:
    client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
    dirs = [d for d in args.dirs.split(",")] if args.dirs else None
    try:
        state = turnaround(args.unit, dirs=dirs, elevation=args.elevation, seed=args.seed,
                           force=args.force, page_only=args.page_only, matte=args.matte,
                           parts_from=Path(args.parts_from) if args.parts_from else None,
                           rig_name=args.rig, glb=Path(args.glb) if args.glb else None,
                           clip_name=args.clip,
                           reference=Path(args.reference) if args.reference else None,
                           client=client, timeout_s=args.timeout)
    except ValueError as exc:
        raise SystemExit(f"anim turnaround: {exc}") from None
    done = [d for d in state["plan"] if (state["cells"].get(d) or {}).get("status") == "done"]
    print(f"turnaround: {len(done)}/{len(state['plan'])} direction(s) done "
          f"({', '.join(done) or 'none'}), seed {state['seed']}")
    for d in state["plan"]:
        r = state["cells"].get(d) or {}
        if r.get("status") == "done":
            t = r.get("timings") or {}
            ref = r.get("reference") or {}
            print(f"  {d:<5} {r['seconds']:6.1f} s  (rig {t.get('rig')} + gpu {t.get('gpu')} "
                  f"+ post {t.get('post')}; service {r.get('gpu_seconds')} s)"
                  f"  reference {ref.get('view', '?')}{' (fallback)' if ref.get('fallback') else ''}")
        elif r:
            print(f"  {d:<5} {r.get('status')}: {r.get('error', '')}")
    print(f"  page   -> {state['page']}")
    print(f"  record -> {state_path(args.unit)}")
    print(f"  publish: mkdir -p {VISUALS_TARGET} && cp -r {Path(state['page']).parent}/. "
          f"{VISUALS_TARGET}/")
    print(f"  then:    {VISUALS_URL}")
    print(f"  the gate: {QUESTION}  anim curate {args.unit} --accept-gate turnaround  (or the Board)")
    return 0


run = run_stage
