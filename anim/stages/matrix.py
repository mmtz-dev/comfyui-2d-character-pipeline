"""Stage: `anim matrix` - every (action x direction) cell of a unit (a GPU stage)

The action matrix feature (#278, the fifth feature of the sprite studio epic).
Once the owner has accepted the base and the views of it, this is the stage
that spends the card: one whole clip per selected (action, direction) cell,
batched, resumable, one seed.

    anim matrix knight                            # every studio action at s0
    anim matrix knight --actions attack,walk      # a subset of the actions
    anim matrix knight --dirs s0,s270             # a subset of the directions
    anim matrix knight --all-dirs --actions idle  # one action, all eight facings
    anim matrix knight --dry-run                  # the plan and every verdict; no GPU
    anim matrix knight --force                    # redo done cells (and skip the gates)

    work/<unit>_<verb>[_<dir>]/   an ordinary rig -> clip -> pick -> post job per cell
    work/<unit>_matrix.json      the plan, one record per cell, the blocked reasons

THE ROUTE per cell is route V exactly as the bake-off ruled it
(`config.MATRIX_*`): `rig` samples the rig's clip for that action into the
clip cell at the cell's yaw -> `clip` sends Wan 2.2 Fun-Control the whole
control video with the accepted view as the reference, skeleton over
full-body depth at gain 0.35 -> `pick --phases` picks the action's budget out
of the finished clip against the studio's phase template -> `post --input clip
--parts --parts-from` mattes, downscales, snaps and LOCKS every part to the
BASE's part palettes (`work/<unit>_base/post/parts.json`), the same palettes
in every direction.
No interpolation, no new weights, no new route: the five phases are PICKED out
of a clip that already exists, never generated one at a time.

JOB NAMES USE THE PIPELINE VERB, not the studio name. attack at s45 is
`knight_strike_s45` (`curation.canonical_job`), and `rig` is called with
`action=cell.verb` so the manifest says `strike` too. `export.job_name_for`
compares the manifest's action to the directory name; a job named for the
studio action would be skipped with "not the matrix job" and the cell would
vanish from the Board and the sheet.

THE HARD PREREQUISITE GATE is the point of the stage. `reference_for` on its
own accepts a PENDING view and, when the base carries no view for a facing at
all, falls back to the front with a warning. A matrix cell is a whole clip,
and the 2026-09-22 identity gate showed what the front-view fallback produces
off-front: bare human heads on five of eight facings. So the matrix REFUSES. A cell whose
direction has no ACCEPTED view on the base is blocked with the reason, the
reason is what the CLI prints and the greyed cell in the tab says, and nothing
renders. There is deliberately no `--allow-fallback`: the escape hatch is
`anim concept --views`, then accepting the view on the Board.

Four things can block a cell, best reason first: the base gate, a rig that
ships no whole clip for the action (the UAL has Jump_Start / Jump_Loop /
Jump_Land and no whole jump, so the studio's jump on that rig waits for a clip
joiner), no accepted view for the direction, and no part palettes on the base
to lock to. `--force` skips the GATE - that is `cli.main`'s existing rule for
every stage - and NOT the other three, which are checked inside the stage
where no flag reaches them.

RESUMABLE, PER CELL, AND CHECKPOINTED AFTER EVERY ONE. A cell whose job says
`matrix.status` done (or kept) and has its report, its phased picks and its
frames is skipped; `--force` redoes it. The state file is written after every
single cell, before the next GPU call, for one reason: Cancel in the studio is
SIGTERM to the process group plus ComfyUI's `/interrupt`, which kills the stage
mid-clip, and the next run must resume from the last FINISHED cell rather than
from the beginning. One seed for the whole unit (`--seed`, else derived from
the unit, the rig and the base render's hash - a new base is a new seed),
recorded and reused by a resume.

PROGRESS. The run manager reads stdout and matches `^\\[(\\d+)/(\\d+)\\]` on the
unstripped line (`anim.api.runs._STEP`), so this stage prints `[n/m]` at column
0 AFTER a cell finishes, and for a skipped one. The line that ANNOUNCES a cell
deliberately carries no brackets, so an in-flight cell does not advance the
bar; `m` is the RUNNABLE count, not the planned count, so a run with six
blocked cells still reaches 100%. The bar therefore means "cells finished",
is monotone, and stays honest when a run is cancelled half way.

AND THE OUTCOME, which is not the same thing (2026-09-24). "Cells finished"
counts a FAILED cell too - it finished, badly - so the bar climbed to 100% on a
batch whose every cell raised, and `anim.api.runs` read the
clean exit as `done`: a run that produced nothing read as a full success. So
the stage now ends with `outcome: done=N failed=N blocked=N of=N` for the run
manager (`anim.outcome`), which rebases the FINAL percent onto the cells that
are actually done and gives the run the status `partial` or `failed`. The live
bar is unchanged; only its last value is corrected, and `blocked` counts as
not-done there - a run of forty blocked cells asked for forty cells and got
none. `--dry-run` reports no tally and exits 0: it was asked to run nothing.

WHAT IT DOES NOT DO. It builds no page: `export` writes the preview and the
Board plus the Matrix tab are the review surfaces. It adds no animation (v1 is
the five `config.STUDIO_ACTIONS`), no new GPU route and no new prompt lever,
and it changes nothing in `rig`, `clip`, `pick` or `post` - every cell is those
stages called as a library, and every job can be re-run by hand.

THE #277 HOLD. `config.MATRIX_MAX_CELLS` (8) caps one run until the motion
exaggeration feature lands, because sprite-studio.md orders #277 before #278 so
the matrix renders once instead of twice. The refusal happens before any GPU
call; `--max-cells 0` is the owner accepting the re-render.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from anim import comfy, config, curation, outcome, partkit, rigkit
from anim.api import jobs as jobs_api
from anim.job import Job
from anim.stages import bakeoff as bakeoff_stage
from anim.stages import clip as clip_stage
from anim.stages import pick as pick_stage
from anim.stages import post as post_stage
from anim.stages import rig as rig_stage
from anim.util import read_json, sha256_bytes, sha256_file, write_json_in_place

NAME = "matrix"
HELP = ("every selected (action x direction) cell: rig -> clip -> pick --phases "
        "-> post (a GPU stage)")

MATRIX_SCHEMA = "anim.matrix/1"
STATE_SUFFIX = "_matrix.json"
RIGS = bakeoff_stage.RIGS
ROUTE = "V"
# What one cell costs, warm, from the measured numbers in DEVLOG.md
# (2026-09-21/22): rig 4-5 s, a 13-frame Wan clip 17-20 s of GPU on a loaded
# model, `post --parts` ~45 s of CPU for twelve frames. The tab multiplies this
# by the cell count to say "~15 min of GPU, warm" before the owner presses
# Generate; it is an estimate on the record, never a timeout.
CELL_SECONDS = 85


# ---------------------------------------------------------------------------
# The plan (pure)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    """One (action, direction) of the matrix, named every way the pipeline
    names it: the board's key, the studio's action, the PIPELINE VERB that the
    job and the manifest use, and the direction cell with its yaw."""

    key: str             # "attack/s45" - curation.cell_key's spelling
    action: str          # the studio's name: attack
    verb: str            # the pipeline verb: strike  <- the job name uses THIS
    direction: str       # s45
    yaw: int
    elevation: str
    job: str             # knight_strike_s45 - curation.canonical_job
    clip: str | None     # the rig's clip for the action; None when it ships none


def directions_for(elevation: str = config.DEFAULT_ELEVATION) -> list[str]:
    """The eight cells of one elevation, in compass order."""
    if elevation not in config.DIRECTION_ELEVATIONS:
        raise ValueError(f"unknown elevation {elevation!r}; one of "
                         f"{list(config.DIRECTION_ELEVATIONS)}")
    return [name for name, (_yaw, el) in config.DIRECTIONS.items() if el == elevation]


def base_reference(unit: str, override: Path | None, work_dir: Path | None,
                   assets_dir: Path | None) -> tuple[Path, str]:
    """The identity image `clip` would use for this unit (the base render, or
    the override), and how it found it."""
    probe = Job.open(curation.canonical_job(unit, "idle", config.DEFAULT_DIRECTION), work_dir)
    return clip_stage.find_reference(probe, {"slug": unit}, override, assets_dir)


def base_views(unit: str, work_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """The views the base carries (`concept.views` of work/<unit>_base) that
    still have their file, with the curation state of each. The one reader of
    "what does the base carry": the Board's ring, the Concept tab's mirror
    check and this stage's gate all ask it, so they cannot disagree."""
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
    `config.MATRIX_REFERENCE_BY_DIR` that the base carries and the owner has
    not rejected, else the front (`front` = (path, kind) from
    `base_reference`) with a warning naming what was wanted. Pure. The stage
    hands it `accepted_views` only, so the walk can never land on a pending
    view or the fallback (see `prerequisite`)."""
    wanted = config.matrix_reference_views(direction)
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


def clip_for(rig_name: str, action: str) -> str | None:
    """The rig's clip for one studio action, or None when that rig ships no
    whole clip for it. A missing clip is NOT an error here: the UAL's jump is three clips and no whole one, and that is a
    per-cell blocked reason the owner can read, not a run that refuses to
    start because one of forty cells cannot render."""
    return config.STUDIO_ACTIONS[action]["clips"].get(rig_name)


def rig_for(rig_name: str = config.MATRIX_RIG,
            glb: Path | None = None) -> tuple[str, Path]:
    """(rig name, GLB path) - the bake-off's rig table, or the override."""
    if rig_name not in RIGS:
        raise ValueError(f"unknown rig {rig_name!r}; one of {sorted(RIGS)}")
    return rig_name, Path(glb) if glb is not None else Path(RIGS[rig_name]["glb"])


def plan(unit: str, actions: Sequence[str] | None = None,
         dirs: Sequence[str] | None = None,
         elevation: str = config.DEFAULT_ELEVATION,
         rig_name: str = config.MATRIX_RIG) -> list[Cell]:
    """The cells to render, in `curation.cell_order` order - DIRECTION-MAJOR,
    every action of s0 before any of s45, so a run that is cancelled half way
    leaves a complete direction rather than a ragged row across all eight.

    `actions=None` is all five studio actions. `dirs=None` is the DEFAULT
    DIRECTION ALONE: a matrix run at every facing is forty clips and the thing
    the #277 hold exists to stop somebody doing by accident. The Matrix tab's "all 8
    directions" toggle (and `--all-dirs`) is how the owner asks for the rest.

    Either spelling of an action is accepted (`attack` and `strike` both give
    the attack row); an unknown action or direction is refused in
    `curation.cell_key`'s words, which is what the API's 400 already says.
    """
    wanted_actions = [a.strip().lower() for a in actions if str(a).strip()] \
        if actions is not None else list(config.MATRIX_ACTIONS)
    wanted_dirs = [d.strip().lower() for d in dirs if str(d).strip()] \
        if dirs is not None else [config.DEFAULT_DIRECTION]
    if not wanted_actions:
        raise ValueError(f"no actions selected; one of {list(config.STUDIO_ACTIONS)}")
    if not wanted_dirs:
        raise ValueError(f"no directions selected; one of "
                         f"{directions_for(elevation)}")
    allowed = directions_for(elevation)   # also validates it
    cells: list[Cell] = []
    for raw_action in wanted_actions:
        for raw_dir in wanted_dirs:
            # cell_key does both validations and raises CurationError(400) with
            # the message the board and the API already use - no second wording.
            key = curation.cell_key(raw_action, raw_dir)
            action, direction = key.split("/", 1)
            if direction not in allowed:
                raise ValueError(f"unknown direction {direction!r} for elevation "
                                 f"{elevation!r}; one of {allowed}")
            yaw, el = config.direction(direction)
            cells.append(Cell(
                key=key, action=action, verb=config.STUDIO_ACTIONS[action]["verb"],
                direction=direction, yaw=yaw, elevation=el,
                job=curation.canonical_job(unit, action, direction),
                clip=clip_for(rig_name, action)))
    cells.sort(key=lambda c: curation.cell_order(c.key))
    return cells


def state_path(unit: str, work_dir: Path | None = None) -> Path:
    """`work/<unit>_matrix.json` - a flat sibling of the curation record, for the reason curation.py's docstring gives: a unit has
    no directory of its own, and `work/<unit>/` would read as a job with no
    manifest to `GET /jobs`."""
    base = Path(work_dir) if work_dir is not None else config.WORK_DIR
    return base / f"{unit}{STATE_SUFFIX}"


def derive_seed(unit: str, rig_name: str, reference_sha256: str) -> int:
    """ONE seed for every cell of the unit; a new base render is a new seed.

    "matrix" is in the material so no other stage that derives a seed from
    the same unit and base lands on the same number.
    """
    material = "|".join([unit, "matrix", rig_name, reference_sha256])
    return int(sha256_bytes(material.encode("utf-8"))[:13], 16) \
        % clip_stage.render_stage.SEED_MAX


def clip_times(duration: float, action: str,
               max_frames: int | None = None) -> list[float] | None:
    """The instants to sample the rig clip at, or None to let `rig` sample it
    at `MATRIX_CLIP_FPS` (which is what the whole pipeline has always done).

    A matrix cell's cost is linear in its frame count and the frame count is
    the clip's duration x 12, unbounded - the UAL's Idle_Loop is 2.5 s, which
    is 31 frames. Past `MATRIX_CLIP_MAX_FRAMES`
    the clip is sampled at that many evenly spaced instants across [0,
    duration] instead: fewer, wider samples of the WHOLE motion, never a
    truncated clip, because a truncated clip would lose the recovery and
    `pick --phases` picks against the whole arc.

    The cap is checked against the action's own budget here. A cap that leaves
    the picker fewer frames than the phase plan asks for would silently short
    every cell of that action (risk 2 of the brief), so it is a ValueError
    naming both numbers rather than a short clip nobody notices.

    NOTE `pick_stage.phase_instants` is the same "even instants across the
    duration" lever the bake-off's `phase_times` uses, but it can only be
    reached through `phase_plan`, which refuses a budget over
    `config.PHASE_CAP` (12) - and this cap is 25 by default. So the instants
    are built here with the same `np.linspace` both of those end in, and the
    frame count that decides WHETHER to cap comes from `rigkit.sample_times`,
    which is the sampler `rig` itself would have used.
    """
    max_frames = config.MATRIX_CLIP_MAX_FRAMES if max_frames is None else int(max_frames)
    row = config.phases_for_action(action)
    if row is not None and max_frames < int(row["budget"]) + 1:
        raise ValueError(
            f"a matrix cell for {action!r} may not be capped at {max_frames} frame(s): "
            f"`pick --phases` picks {row['budget']} of them and keeps the clip's end, "
            f"so the cap must be at least {int(row['budget']) + 1} "
            f"(config.MATRIX_CLIP_MAX_FRAMES / ANIM_MATRIX_MAX_FRAMES)")
    frames = len(rigkit.sample_times(float(duration), config.MATRIX_CLIP_FPS))
    if frames <= max_frames:
        return None
    return [round(float(t), 6) for t in np.linspace(0.0, float(duration), max_frames)]


# ---------------------------------------------------------------------------
# The prerequisites - the hard gate (the heart of the stage)
# ---------------------------------------------------------------------------
def base_parts_path(unit: str, work_dir: Path | None = None) -> Path:
    """`work/<unit>_base/post/parts.json` - the part palettes `post --parts`
    measured on the accepted base, which is what every cell locks to, in every
    direction. The base is a front view, so a part the front never shows (the
    back of a cloak) locks to the front's colours for it - a known trade: one
    palette per part for the whole unit, the same one the Board's "measure
    part palettes" button makes."""
    return Job.open(curation.base_parts_job(unit), work_dir).post_dir / "parts.json"


def prerequisite(cell: Cell, *, unit: str, rig_name: str,
                 views: dict[str, dict[str, Any]], gates: dict[str, str],
                 parts_path: Path | None,
                 gate_reason: str | None = None) -> dict[str, Any]:
    """Whether this cell may render, and the reason it may not.

    `views` is `base_views` (the views the base carries, each with its
    curation state); `gates` the unit's gate states; `parts_path` the base's
    `post/parts.json` (its existence is the one thing looked up on disk);
    `gate_reason` the sentence `curation.gate_blocks` already produced for
    this unit, so the gate wording has exactly one home.

    Returns `{"ok", "view", "parts_from", "reason", "wanted"}`. The reason is a
    whole sentence naming what is missing and the one command that fixes it -
    it is printed by the CLI and shown inside the greyed cell in the tab, so
    "blocked" is never a silent grey square.

    THE VIEW RULE IS STRICTER THAN `reference_for`'s. That walk takes the
    first view down `config.matrix_reference_views` that the base carries and
    the owner has not REJECTED - a pending view counts - and falls back to the
    front with a warning when there is none. Here the walk only accepts a view
    whose curation state is `accepted`, and reaching "front" for any direction
    whose row is not `("front",)` is a refusal, not a fallback: the matrix
    refuses to spend a whole clip on a facing nobody has confirmed. s270's row IS `("front",)` - the
    base itself is the front view - so it is satisfied by the base gate alone.
    """
    wanted = list(config.matrix_reference_views(cell.direction))
    blocked = {"ok": False, "view": None, "parts_from": None, "wanted": wanted}

    # 1. The gate. `curation.gate_blocks` answers it in one sentence and
    # honours the project's curation mode, which a re-worded copy here would
    # not. The views, accepted one by one on the Board, are the rest of the
    # human gate and are step 3.
    if gate_reason:
        return {**blocked, "reason": gate_reason}
    if gates.get("base") != "accepted":
        return {**blocked, "reason": (
            f"{unit}: the base gate is pending - accept the base first "
            f"(`anim concept --unit {unit} --pick NN`, `anim curate {unit} "
            f"--accept-gate base`, or the board)")}

    # 2. The rig ships no whole clip for the action. The honest jump surface.
    if not cell.clip:
        return {**blocked, "reason": (
            f"the {rig_name} rig ships no whole {cell.action} clip (the UAL has "
            f"Jump_Start / Jump_Loop / Jump_Land; the studio's {cell.action} on "
            f"that rig waits for a clip joiner) - config.STUDIO_ACTIONS")}

    # 3. The required concept view, accepted. Walk the preference list; "front"
    # ends it. A row that is only ("front",) - s270 - is the base itself.
    view: str | None = None
    for candidate in wanted:
        if candidate == config.FRONT_VIEW:
            break
        entry = views.get(candidate)
        if entry is not None and entry.get("state") == "accepted":
            view = candidate
            break
    others = [v for v in wanted if v != config.FRONT_VIEW]
    if view is None and others:
        why = ", ".join(f"{v} ({_view_state(views, v)})" for v in others)
        return {**blocked, "reason": (
            f"{cell.direction}: no accepted view for it - wanted {why}; the matrix "
            f"does not render a cell from the front-view fallback (`anim concept "
            f"--unit {unit} --views {others[0]}`, then accept it on the Board)")}
    if view is None:
        view = config.FRONT_VIEW          # s270: the base render is the front

    # 4. The base's part palettes, which every cell locks to. Checked here,
    # before the GPU, because `post --parts` would otherwise discover the
    # missing file AFTER Wan has rendered the clip.
    parts_from = parts_path
    if parts_from is None or not Path(parts_from).exists():
        return {**blocked, "view": view, "reason": (
            f"{unit}: no part palettes on the base ({parts_from}) - the cell has "
            f"nothing to lock its colours to; run `anim post "
            f"{curation.base_parts_job(unit)} --parts` (CPU, no GPU), or the "
            f"Board's \"measure part palettes\"")}

    return {"ok": True, "view": view, "parts_from": str(parts_from),
            "reason": None, "wanted": wanted}


def _view_state(views: dict[str, dict[str, Any]], view: str) -> str:
    """How a wanted view reads in a blocked cell's reason: the owner needs to
    know whether to RENDER it or to ACCEPT it, and those are different jobs."""
    entry = views.get(view)
    if entry is None:
        return "not on the base"
    return str(entry.get("state") or "pending")


def prerequisites(unit: str, cells: Sequence[Cell], work_dir: Path | None = None,
                  *, rig_name: str = config.MATRIX_RIG,
                  mode: str | None = None) -> dict[str, dict[str, Any]]:
    """`{cell key: the verdict}` for a whole plan, reading the curation record
    and the base's views ONCE for all of them.

    The gate sentence is per (action, direction), not per run: `gate_blocks`
    names the cell it is refusing ("... before a s315 cell render"), and in the
    `every` curation mode the answer genuinely differs cell by cell. Asking it
    once with the default direction gave all forty cells s0's sentence, which
    read as a bug in the gate rather than a gate doing its job. Memoised on the
    pair, so it is at most one call per distinct cell and usually far fewer.
    """
    record = curation.load(unit, work_dir)
    gates = dict(record.get("gates") or {})
    views = base_views(unit, work_dir)
    parts_path = base_parts_path(unit, work_dir)
    reasons: dict[tuple[str, str], str | None] = {}

    def gate_reason_for(cell: Cell) -> str | None:
        pair = (cell.action, cell.direction)
        if pair not in reasons:
            reasons[pair] = curation.gate_blocks(
                unit, NAME, action=cell.action, direction=cell.direction,
                work_dir=work_dir, mode=mode)
        return reasons[pair]

    return {c.key: prerequisite(c, unit=unit, rig_name=rig_name, views=views,
                                gates=gates, parts_path=parts_path,
                                gate_reason=gate_reason_for(c))
            for c in cells}


def accepted_views(views: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The base's views with everything but `accepted` dropped.

    This is what makes `reference_for` usable here without loosening
    the rule: handed only the accepted views, its walk reaches the same view
    this stage's own walk did, and its `fallback` flag can be asserted False.
    Reusing it unfiltered would let a PENDING view win the walk and the cell
    would render against an image the owner never confirmed.
    """
    return {v: e for v, e in views.items() if e.get("state") == "accepted"}


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def is_done(job: Job) -> bool:
    """A cell is done when its own job says so AND the files are there: the
    `matrix` block in job.json, the post report, a PHASED picks.json (an
    unphased one is a job from before this stage, or a `pick` that fell back)
    and at least one finished frame.

    `kept` counts as done. That status means the chain ran and the owner's
    hand-edited picks.json was left alone (see `run_cell_chain`); re-running it
    would spend the GPU again on a cell somebody has already curated, which is
    the opposite of what refusing to overwrite the picks was protecting.
    """
    if not job.manifest_path.exists():
        return False
    manifest = job.read_manifest()
    if (manifest.get("matrix") or {}).get("status") not in ("done", "kept"):
        return False
    if not job.report_path.exists():
        return False
    picks = read_json(job.picks_path) if job.picks_path.exists() else {}
    if not isinstance(picks, dict) or not picks.get("phases"):
        return False
    return len(bakeoff_stage.frame_files(job, manifest)) >= 1


def run_cell_chain(cell: Cell, *, unit: str, glb: Path, seed: int, matte: str,
                   parts_from: Path | None, times: Sequence[float] | None,
                   reference: dict[str, Any], force: bool,
                   client: comfy.ComfyClient, segmenter: partkit.Segmenter | None,
                   work_dir: Path | None, assets_dir: Path | None,
                   workflow_dir: Path | None, lock_path: Path | None,
                   timeout_s: float) -> dict[str, Any]:
    """rig -> clip -> pick --phases -> post --parts for one cell, timed.

    Every stage is called as a library function;
    the stage functions are silent, so every line the owner sees is printed by
    the orchestrator above.
    """
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    # `action=cell.verb`, not cell.action: `export` reads manifest["action"] and
    # compares `<unit>_<action>[_<dir>]` to the directory name, so a manifest
    # that said "attack" beside a job named knight_strike_s45 would be skipped
    # as "not the matrix job" and the cell would never reach a sheet.
    job = rig_stage.rig(glb, clip=cell.clip, action=cell.verb, slug=unit,
                        job_name=cell.job, cell=tuple(config.MATRIX_CELL),
                        depth="full", yaw=cell.yaw, elevation=cell.elevation,
                        times=times, fps=config.MATRIX_CLIP_FPS, work_dir=work_dir)
    timings["rig"] = round(time.monotonic() - t0, 1)
    rig_frames = len(job.skeletons())

    t0 = time.monotonic()
    # The accepted view goes to `clip` as the reference override. The front is
    # left to `clip.find_reference` so the clip record says "base render" for
    # it - and the direction-aware prompt and
    # the face-point hiding ride along inside clip/rigkit for free.
    override = (Path(reference["file"])
                if reference.get("override") or reference["view"] != config.FRONT_VIEW
                else None)
    clip_record = clip_stage.clip(job.name, seed=seed, control=config.MATRIX_CONTROL,
                                  depth_gain=config.MATRIX_DEPTH_GAIN,
                                  reference=override, client=client,
                                  workflow_dir=workflow_dir, lock_path=lock_path,
                                  work_dir=work_dir, assets_dir=assets_dir,
                                  timeout_s=timeout_s)
    timings["gpu"] = round(time.monotonic() - t0, 1)
    vram = bakeoff_stage._vram_used_gb(client)

    t0 = time.monotonic()
    # `phases=True` is the feature's first constraint: the five phases are
    # picked out of the finished clip. `config.phases_default` would leave
    # strike / idle / walk on the art bible's key-pose picker - its own
    # docstring says "the action matrix asks for every cell", and this is that.
    #
    # `force=force` is the RUN's flag, never True: `pick` refuses to overwrite a
    # picks.json somebody edited by hand, and a batch that re-picked a cell the
    # owner had already curated would throw that work away. Such a cell is
    # recorded `kept` with the message and the run goes on to `post`, which
    # then packs the frames the owner chose.
    kept: str | None = None
    try:
        picks_record = pick_stage.pick(job.name, action=cell.verb, phases=True,
                                       force=force, work_dir=work_dir)
    except RuntimeError as exc:
        kept = str(exc).splitlines()[0]
        picks_record = read_json(job.picks_path) if job.picks_path.exists() else {}
    timings["pick"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    post_error: str | None = None
    try:
        post_stage.post(job.name, matte=matte, source=post_stage.CLIP_SOURCE,
                        parts=True, parts_from=parts_from, segmenter=segmenter,
                        assets_dir=assets_dir, work_dir=work_dir)
    except RuntimeError as exc:      # verification failures: the report is on disk
        post_error = str(exc).splitlines()[0]
    timings["post"] = round(time.monotonic() - t0, 1)

    report = read_json(job.report_path) if job.report_path.exists() else {}
    frames = bakeoff_stage.frame_files(job)
    if not frames:
        raise RuntimeError(post_error or f"post wrote no frame for {cell.key}")
    phases = picks_record.get("phases") or {}
    keypoints = read_json(job.keypoints_path)
    return {
        "job": job.name, "timings": timings, "kept": kept,
        "gpu_seconds": round(float(clip_record.get("seconds") or 0.0), 1),
        "vram_used_gb": vram, "seed": seed,
        "clip_frames": int(clip_record["frames"]), "rig_frames": rig_frames,
        "frames": len(frames), "picks": list(picks_record.get("picks") or []),
        "control": clip_record["control"], "depth_gain": clip_record["depth_gain"],
        "prompt": clip_record.get("prompt"), "negative": clip_record.get("negative"),
        "parts_phrase": clip_record.get("parts_phrase"),
        # `closure` is INFORMATION, not a failure: idle / walk / run are loops
        # and Wan does not know it, so a cycle whose last pick does not lead
        # back into its first is worth showing and not worth refusing.
        "phases": {"tags": list(phases.get("tags") or []),
                   "budget": (phases.get("plan") or {}).get("budget"),
                   "anticipation": phases.get("anticipation"),
                   "loop": bool((phases.get("plan") or {}).get("loop")),
                   "closure": phases.get("closure")},
        "reference": {k: v for k, v in reference.items() if k != "path"},
        "parts_from": str(parts_from) if parts_from else None,
        "locked_to": (report.get("parts") or {}).get("palettes_from"),
        "hidden_face": list((keypoints["frames"][0].get("hidden") or [])),
        "post_error": post_error,
    }


def run_cell(cell: Cell, **kw: Any) -> dict[str, Any]:
    """One cell end to end. The record goes into the state file AND into the
    job's own job.json (`matrix` block) so a resume can tell a finished cell
    from the job alone, without the state file. One failed cell is recorded
    with its error and the run goes on; forty cells must not cost each other.
    """
    started = time.monotonic()
    record: dict[str, Any] = {**asdict(cell), "status": "running", "started": _now(),
                              "reason": None, "error": None}
    try:
        record.update(run_cell_chain(cell, **kw))
        record["status"] = "kept" if record.get("kept") else "done"
    except Exception as exc:  # noqa: BLE001 - one bad cell must not cost the rest
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["seconds"] = round(time.monotonic() - started, 1)
    job = Job.open(cell.job, kw.get("work_dir"))
    if job.manifest_path.exists():
        job.update_manifest(matrix={"status": record["status"], "key": cell.key,
                                    "seconds": record["seconds"],
                                    "error": record.get("error")})
    return record


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------
def matrix(unit: str, *, actions: Sequence[str] | None = None,
           dirs: Sequence[str] | None = None,
           elevation: str = config.DEFAULT_ELEVATION, seed: int | None = None,
           force: bool = False, dry_run: bool = False, matte: str = "rembg",
           rig_name: str = config.MATRIX_RIG, glb: Path | None = None,
           reference: Path | None = None, max_cells: int | None = None,
           client: comfy.ComfyClient | None = None,
           segmenter: partkit.Segmenter | None = None,
           work_dir: Path | None = None, assets_dir: Path | None = None,
           workflow_dir: Path | None = None, lock_path: Path | None = None,
           timeout_s: float | None = None,
           log: Callable[[str], None] = print) -> dict[str, Any]:
    """Every selected cell, in order, checkpointed after each one."""
    planned = plan(unit, actions, dirs, elevation, rig_name)
    rig_name, glb = rig_for(rig_name, glb)
    timeout_s = config.MATRIX_TIMEOUT_S if timeout_s is None else float(timeout_s)
    max_cells = config.MATRIX_MAX_CELLS if max_cells is None else int(max_cells)

    path = state_path(unit, work_dir)
    state = read_json(path) if path.exists() else {}
    records: dict[str, Any] = dict(state.get("cells") or {})

    # A unit with no base at all has no reference to find, and `find_reference`
    # says so by raising. That must not be how the owner learns it: the base
    # gate is one of the reasons the verdicts below are about to print, and a
    # dry run exists to read them. So the failure becomes a null reference row
    # and the gate does the talking - nothing is runnable without a base anyway.
    try:
        ref_path, ref_kind = base_reference(unit, reference, work_dir, assets_dir)
    except (FileNotFoundError, OSError, ValueError) as exc:
        ref_path, ref_kind = None, f"none ({exc})"
    reference_row = {"file": str(ref_path) if ref_path else None, "kind": ref_kind,
                     "sha256": sha256_file(ref_path)
                     if ref_path is not None and ref_path.exists() else None}
    if seed is None:
        seed = int(state["seed"]) if state.get("seed") and not force \
            else derive_seed(unit, rig_name, reference_row["sha256"] or "")
    previous = (state.get("reference") or {}).get("sha256")
    if previous and previous != reference_row["sha256"] and not force:
        raise ValueError(
            f"{unit}: the base render changed since this matrix was rendered "
            f"({previous[:12]} -> {str(reference_row['sha256'])[:12]}); run "
            f"`anim matrix {unit} --force` to re-render every cell from the new "
            f"base rather than mix two")

    # The hard gate, for the whole plan at once. A blocked cell is recorded
    # with its reason and is NOT counted in the [n/m] denominator - eight
    # blocked cells must not strand the progress bar below 100.
    verdicts = prerequisites(unit, planned, work_dir, rig_name=rig_name)
    views = base_views(unit, work_dir)
    for cell in planned:
        verdict = verdicts[cell.key]
        previous = records.get(cell.key) or {}
        if verdict["ok"]:
            # Name every selected cell in the record even before it runs, so a
            # `--dry-run` state file IS the plan with its verdicts and not an
            # empty `cells` block.
            records.setdefault(cell.key, {**asdict(cell), "status": "not run",
                                          "reason": None, "error": None})
            continue
        log(f"matrix: BLOCKED {cell.key} - {verdict['reason']}")
        if previous.get("status") in ("done", "kept"):
            # A cell that already rendered keeps its record; a view rejected
            # afterwards must not erase the frames it produced. It is still out
            # of `runnable`, so nothing re-renders it.
            continue
        records[cell.key] = {**asdict(cell), "status": "blocked",
                             "reason": verdict["reason"], "error": None,
                             "reference": {"view": None, "wanted": verdict["wanted"]}}
    runnable = [c for c in planned if verdicts[c.key]["ok"]]

    # The #277 hold, before a single GPU call.
    if max_cells and len(runnable) > max_cells:
        raise ValueError(
            f"{unit}: {len(runnable)} cells asked for, and the matrix is held to "
            f"{max_cells} until the motion exaggeration feature (#277) lands - "
            f"sprite-studio.md orders #277 before #278 so the matrix renders once. "
            f"Run a subset (--actions / --dirs), or raise the cap with --max-cells / "
            f"ANIM_MATRIX_MAX_CELLS if you accept re-rendering.")

    references: dict[str, dict[str, Any]] = {}
    clip_frames: dict[str, int] = {}
    if runnable and ref_path is None and not dry_run:
        raise ValueError(
            f"{unit}: no identity image to render against ({ref_kind}); pick a base "
            f"first (`anim concept --unit {unit} --pick NN`, or the Concept tab)")
    if not dry_run and runnable:
        client = client or comfy.ComfyClient()
        # One SAM load for every cell of the run (a 184 MB checkpoint), as the
        # bake-off does - the segmenter is the wall-clock cost of `post --parts`.
        segmenter = segmenter or partkit.Sam2Segmenter()
        loaded = rigkit.load_rig(glb)          # once: the clip durations
        total = len(runnable)
        for n, cell in enumerate(runnable, start=1):
            verdict = verdicts[cell.key]
            job = Job.open(cell.job, work_dir)
            if is_done(job) and not force:
                # THE PROGRESS CONTRACT: a skipped cell advances the bar too,
                # at column 0, so a resume of a finished matrix reaches 100%.
                log(f"[{n}/{total}] {cell.key} done -> skipped (--force redoes it)")
                records.setdefault(cell.key, {**asdict(cell), "status": "done",
                                              "reason": None, "error": None})
                continue
            duration = loaded.clip(cell.clip).duration
            times = clip_times(duration, cell.action)
            frames = len(times) if times is not None \
                else len(rigkit.sample_times(duration, config.MATRIX_CLIP_FPS))
            clip_frames[cell.key] = frames
            # Resolved from the ACCEPTED views only, so `reference_for`'s walk
            # cannot reach a pending view that the verdict above refused.
            ref = reference_for(cell.direction, (ref_path, ref_kind),
                                accepted_views(views))
            assert ref["fallback"] is False, \
                f"{cell.key}: a runnable cell must never render on the front-view fallback"
            references[cell.direction] = ref
            # No brackets on the "starting" line, on purpose: an in-flight cell
            # must not advance the bar.
            log(f"matrix: {cell.key} (yaw {cell.yaw}, {cell.clip}, {frames} frame(s); "
                f"reference {ref['view']}) ...")
            record = run_cell(cell, unit=unit, glb=glb, seed=seed, matte=matte,
                              parts_from=Path(verdict["parts_from"]), times=times,
                              reference=ref, force=force, client=client,
                              segmenter=segmenter, work_dir=work_dir,
                              assets_dir=assets_dir, workflow_dir=workflow_dir,
                              lock_path=lock_path, timeout_s=timeout_s)
            record.setdefault("reason", None)
            records[cell.key] = record
            if record["status"] in ("done", "kept"):
                tags = ", ".join(record["phases"]["tags"]) or "none"
                log(f"[{n}/{total}] {cell.key}: {record['status']} - "
                    f"{record['frames']} frame(s), phases {tags}, gpu "
                    f"{record['gpu_seconds']} s, total {record['seconds']} s"
                    + (f", post: {record['post_error']}" if record.get("post_error") else "")
                    + (f", picks kept: {record['kept']}" if record.get("kept") else ""))
            else:
                log(f"[{n}/{total}] {cell.key}: FAILED after {record['seconds']} s "
                    f"- {record['error']}")
            # Before the next GPU call, never after the loop: a Cancel is
            # SIGTERM mid-clip, and the next run resumes from here.
            write_json_in_place(path, _state(unit, elevation, rig_name, glb, seed,
                                             matte, reference_row, planned, records,
                                             verdicts, references, max_cells,
                                             clip_frames))

    state = _state(unit, elevation, rig_name, glb, seed, matte, reference_row,
                   planned, records, verdicts, references, max_cells, clip_frames)
    write_json_in_place(path, state)
    return state


def _state(unit: str, elevation: str, rig_name: str, glb: Path, seed: int | None,
           matte: str, reference: dict[str, Any], planned: Sequence[Cell],
           records: dict[str, Any], verdicts: dict[str, dict[str, Any]],
           references: dict[str, dict[str, Any]], max_cells: int,
           clip_frames: dict[str, int]) -> dict[str, Any]:
    """`anim.matrix/1`. `plan` and `cells` are split on purpose: `plan` is what
    THIS run selected and what `summary` counts, while `cells` keeps every
    record the file has ever held, so `anim matrix knight --dirs s0` does not
    erase what an earlier `--all-dirs` run rendered."""
    keys = [c.key for c in planned]
    counted = [records.get(k) or {} for k in keys]
    every = sorted(records, key=curation.cell_order)
    # One tally, shared with `bakeoff` (`anim.outcome`), so the
    # exit code, the run status in the studio and this block cannot disagree.
    tally = outcome.count(records, keys)
    return {
        "schema": MATRIX_SCHEMA, "unit": unit, "elevation": elevation,
        "rig": {"name": rig_name, "glb": str(glb)},
        "route": ROUTE, "control": config.MATRIX_CONTROL,
        "depth_gain": config.MATRIX_DEPTH_GAIN, "cell": list(config.MATRIX_CELL),
        "clip_fps": config.MATRIX_CLIP_FPS,
        "max_frames": config.MATRIX_CLIP_MAX_FRAMES,
        "seed": seed, "matte": matte, "reference": reference,
        "references": {d: references[d] for d in sorted(references)},
        "reference_map": {c.direction: list(config.matrix_reference_views(c.direction))
                          for c in planned},
        "actions": sorted({c.action for c in planned},
                          key=lambda a: list(config.STUDIO_ACTIONS).index(a)),
        "directions": sorted({c.direction for c in planned},
                             key=lambda d: list(config.DIRECTIONS).index(d)),
        "clip_frames": dict(clip_frames),
        "max_cells": max_cells, "updated": _now(),
        "plan": keys,
        "blocked": {k: verdicts[k]["reason"] for k in keys
                    if k in verdicts and not verdicts[k]["ok"]},
        "cells": {k: records[k] for k in every},
        "summary": {
            **tally.counts,
            "gpu_seconds": round(sum(float(r.get("gpu_seconds") or 0.0)
                                     for r in counted), 1),
            "seconds": round(sum(float(r.get("seconds") or 0.0) for r in counted), 1),
        },
    }


# ---------------------------------------------------------------------------
# What the API answers: GET /matrix/{unit}
# ---------------------------------------------------------------------------
def listing(unit: str, work_dir: Path | None = None) -> dict[str, Any] | None:
    """Every PLANNED cell of the full grid with its live prerequisite verdict,
    or None when `unit` is not a unit slug.

    This answers BEFORE anything has run: the tab
    greys the cells it cannot generate on first load, so a unit with no state
    file still gets a row per cell of the whole five-by-eight grid. The verdict
    (`runnable`, `reason`) is recomputed from the curation record every time,
    never read back from the state file - the owner accepts a view on the Board
    and the cell must un-grey on the next reload, without a re-run.
    """
    if not jobs_api.valid_name(unit):
        return None
    elevation = config.DEFAULT_ELEVATION
    directions = directions_for(elevation)
    planned = plan(unit, None, directions, elevation, config.MATRIX_RIG)
    verdicts = prerequisites(unit, planned, work_dir, rig_name=config.MATRIX_RIG)
    path = state_path(unit, work_dir)
    state = read_json(path) if path.exists() else {}
    stored: dict[str, Any] = dict(state.get("cells") or {})
    curated = (curation.read(unit, work_dir) or {}).get("cells") or {}

    rows: dict[str, Any] = {}
    for cell in planned:
        record = stored.get(cell.key) or {}
        verdict = verdicts[cell.key]
        job = Job.open(cell.job, work_dir)
        manifest = job.read_manifest() if job.manifest_path.exists() else {}
        frames = bakeoff_stage.frame_files(job, manifest) if manifest else []
        phases = record.get("phases") or {}
        tags = list(phases.get("tags") or [])
        status = record.get("status") or "not run"
        if not verdict["ok"] and status in ("not run", "blocked"):
            status = "blocked"
        row = config.phases_for_action(cell.action)
        rows[cell.key] = {
            "key": cell.key, "action": cell.action, "verb": cell.verb,
            "direction": cell.direction, "yaw": cell.yaw, "job": cell.job,
            "clip": cell.clip, "status": status,
            "runnable": bool(verdict["ok"]), "reason": verdict["reason"],
            "view": verdict["view"], "wanted": verdict["wanted"],
            "frames": len(frames),
            "strip": [f"/jobs/{job.name}/frames/post/{i}.png" for i in range(len(frames))],
            "phases": [tags[i] if i < len(tags) else None for i in range(len(frames))],
            "fps": int(manifest.get("fps") or config.MATRIX_CLIP_FPS),
            "loop": bool((row or {}).get("loop")),
            "closure": phases.get("closure"),
            "seconds": record.get("seconds"), "gpu_seconds": record.get("gpu_seconds"),
            "seed": record.get("seed") or state.get("seed"),
            "error": record.get("error"),
            "curation": (curated.get(cell.key) or {}).get("status"),
        }
    selectable = sum(1 for r in rows.values() if r["runnable"])
    return {
        "unit": unit, "schema": MATRIX_SCHEMA, "elevation": elevation,
        "seed": state.get("seed"), "rig": state.get("rig") or
        {"name": config.MATRIX_RIG, "glb": str(RIGS[config.MATRIX_RIG]["glb"])},
        "actions": list(config.MATRIX_ACTIONS), "directions": directions,
        "cells": rows,
        "max_cells": config.MATRIX_MAX_CELLS, "selectable": selectable,
        # Per CELL, warm - the tab multiplies by how many are selected.
        "estimate_seconds": CELL_SECONDS,
        "summary": state.get("summary"), "updated": state.get("updated"),
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("unit", help="unit slug, e.g. knight")
    parser.add_argument("--actions", default=None,
                        help=f"comma-separated studio actions, e.g. attack,walk "
                             f"(default: all of {list(config.STUDIO_ACTIONS)}); "
                             f"the pipeline verb is accepted too")
    parser.add_argument("--dirs", default=None,
                        help=f"comma-separated direction cells, e.g. s0,s90 "
                             f"(default: {config.DEFAULT_DIRECTION} alone - the "
                             f"other facings are opt-in, they are the expensive ones)")
    parser.add_argument("--all-dirs", dest="all_dirs", action="store_true",
                        help="every one of the elevation's eight facings "
                             "(what the tab's 'all 8 directions' toggle sends)")
    parser.add_argument("--elevation", choices=config.DIRECTION_ELEVATIONS,
                        default=config.DEFAULT_ELEVATION,
                        help="'side' (default, v1) or 'top' (the top-down pitch, #285)")
    parser.add_argument("--seed", type=int, default=None,
                        help="one seed for every cell (default: the state file's, "
                             "else derived from the unit, the rig and the base render)")
    parser.add_argument("--force", action="store_true",
                        help="redo cells that are done; also skips the curation "
                             "gate (not the per-cell view / parts / clip checks)")
    parser.add_argument("--matte", default="rembg", choices=sorted(post_stage.MATTES))
    parser.add_argument("--rig", default=config.MATRIX_RIG, choices=sorted(RIGS),
                        help=f"the rig (default {config.MATRIX_RIG}, the bake-off ruling)")
    parser.add_argument("--glb", default=None, metavar="FILE",
                        help="the rig file (default: the rig's vendored GLB)")
    parser.add_argument("--reference", default=None,
                        help="identity image for the front (default: "
                             "work/<unit>_base/render/<final>)")
    parser.add_argument("--max-cells", dest="max_cells", type=int, default=None,
                        help=f"refuse a run wider than this before any GPU call "
                             f"(default {config.MATRIX_MAX_CELLS}, the #277 hold; "
                             f"0 removes the cap)")
    parser.add_argument("--timeout", type=float, default=None,
                        help=f"seconds to wait for one cell's GPU job (default "
                             f"{config.MATRIX_TIMEOUT_S:.0f}); past it the cell is "
                             f"recorded as failed and the run goes on")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="print the plan and every prerequisite verdict; no GPU")


def run_stage(args: argparse.Namespace) -> int:
    client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
    actions = [a for a in args.actions.split(",")] if args.actions else None
    if args.all_dirs:
        dirs = directions_for(args.elevation)
    else:
        dirs = [d for d in args.dirs.split(",")] if args.dirs else None
    try:
        state = matrix(args.unit, actions=actions, dirs=dirs, elevation=args.elevation,
                       seed=args.seed, force=args.force, dry_run=args.dry_run,
                       matte=args.matte, rig_name=args.rig,
                       glb=Path(args.glb) if args.glb else None,
                       reference=Path(args.reference) if args.reference else None,
                       max_cells=args.max_cells, client=client, timeout_s=args.timeout)
    except (ValueError, curation.CurationError) as exc:
        raise SystemExit(f"anim matrix: {exc}") from None
    summary = state["summary"]
    print(f"matrix: {summary['done']}/{summary['of']} cell(s) done, "
          f"{summary['failed']} failed, {summary['blocked']} blocked; "
          f"seed {state['seed']}, {summary['gpu_seconds']} s of GPU")
    for key in state["plan"]:
        record = state["cells"].get(key) or {}
        status = record.get("status") or "not run"
        if status in ("done", "kept"):
            t = record.get("timings") or {}
            tags = ", ".join((record.get("phases") or {}).get("tags") or [])
            print(f"  {key:<14} {status:<7} {record['seconds']:6.1f} s  "
                  f"(rig {t.get('rig')} + gpu {t.get('gpu')} + pick {t.get('pick')} "
                  f"+ post {t.get('post')})  {record.get('frames')} frame(s): {tags}")
        elif status == "blocked":
            print(f"  {key:<14} blocked  {record.get('reason')}")
        else:
            print(f"  {key:<14} {status:<7} {record.get('error') or ''}")
    print(f"  record -> {state_path(args.unit)}")
    print(f"  then:    anim curate {args.unit}     (accept the cells on the Board)")
    print(f"           anim export {args.unit}     (the sheets + atlas.json)")
    if args.dry_run:
        # A dry run rendered no cell on purpose, and succeeded at that. Its
        # records are all `not run`, which is not a tally of anything.
        return 0
    # The exit code is the tally, not the process result: a run whose every
    # cell failed or was blocked exits non-zero even though the loop caught
    # each one and carried on (`anim.outcome`).
    return outcome.Outcome(done=summary["done"], failed=summary["failed"],
                           blocked=summary["blocked"], of=summary["of"]).report()


run = run_stage
