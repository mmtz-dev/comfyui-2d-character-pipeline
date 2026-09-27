"""The base gate and the views step - where the human says yes or no.

    work/<unit>_curation.json      one record per unit: the gate and the views

THE RECORD (`anim.curation/1`):

    {
      "schema": "anim.curation/1", "unit": "knight", "origin": "concept --pick",
      "gates": {"base": "accepted"},
      "views": {"back": "accepted", "left": "pending"}
    }

Why one file per unit beside the jobs rather than `work/<unit>/curation.json`:
a unit has no directory of its own - its jobs are siblings named
`<unit>_<action>` - and a `work/<unit>/` would read as a job with no manifest
to `GET /jobs`.

THE GATE. There is one, `base`, and it is always on: `concept --pick` opens
it. A view is drawn FROM the base, so the views wait for it; every frame of
an animation is drawn from its direction's accepted view, so the animate
stage waits for it too (`views_blocked`); and the older GPU stages in
`BASE_GATED` wait for it before spending the card on a character nobody has
approved. CPU stages (`rig`, `pick`, `strip`, `post`) never block: they cost
seconds.

(The pipeline tabs feature, 2026-09-27, retired the curation board, its
per-cell accept / reject records and the `every` / `end` curation modes: the
Animate tab's keyframe gate is where a picture is judged now. Older records
may still carry `cells` or a `turnaround` gate; they are left in the file as
written and ignored.)

LEGACY: a unit with no curation file is not gated. The knight's published
jobs predate the gate, and a rule that refused `anim render knight_strike`
until someone accepted a base nobody was asked about would be a regression.
Only `concept --pick` writes a real gate record (base accepted).
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

from anim import config
from anim.api import jobs as jobs_api
from anim.job import Job
from anim.util import read_json, write_json_in_place

CURATION_SCHEMA = "anim.curation/1"
GATES: tuple[str, ...] = ("base",)
GATE_STATES: tuple[str, ...] = ("pending", "accepted")
# The multi-view reference feature: the views of the base (`concept
# --views`) are a step under the base gate, one state per view.
VIEW_STATES: tuple[str, ...] = ("pending", "accepted", "rejected")

# The stages that spend the GPU on a unit's cells, and so wait for its gate.
# `base` and `concept` MAKE the base; they are never gated.
BASE_GATED = frozenset({"clip", "render", "bakeoff", "run"})


class CurationError(ValueError):
    """A request the record cannot honour. `status` is the HTTP code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _work(work_dir: Path | None) -> Path:
    return Path(work_dir) if work_dir is not None else config.WORK_DIR


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def curation_path(unit: str, work_dir: Path | None = None) -> Path:
    if not jobs_api.valid_name(unit):
        raise CurationError(404, f"{unit!r} is not a unit name")
    return _work(work_dir) / f"{unit}_curation.json"


def new_record(unit: str, *, gates: dict[str, str] | None = None,
               origin: str = "curate") -> dict[str, Any]:
    return {
        "schema": CURATION_SCHEMA, "unit": unit, "origin": origin,
        "created": _now(),
        "gates": {gate: (gates or {}).get(gate, "pending") for gate in GATES},
        "views": {},
    }


def read(unit: str, work_dir: Path | None = None) -> dict[str, Any] | None:
    """The record on disk, or None for a unit nothing has curated yet."""
    path = curation_path(unit, work_dir)
    if not path.exists():
        return None
    record = read_json(path)
    if not isinstance(record, dict) or record.get("schema") != CURATION_SCHEMA:
        raise CurationError(409, f"{path} is not an {CURATION_SCHEMA} record")
    record.setdefault("gates", {})
    record.setdefault("views", {})
    for gate in GATES:
        record["gates"].setdefault(gate, "pending")
    return record


def write(unit: str, record: dict[str, Any], work_dir: Path | None = None) -> Path:
    record["updated"] = _now()
    return write_json_in_place(curation_path(unit, work_dir), record)


def load(unit: str, work_dir: Path | None = None) -> dict[str, Any]:
    """The record, created as a LEGACY one (every gate accepted, nothing
    written yet) when the unit has none - see the module docstring."""
    record = read(unit, work_dir)
    if record is None:
        record = new_record(unit, gates={g: "accepted" for g in GATES},
                            origin="legacy")
    return record


# ---------------------------------------------------------------------------
# Names and what the base carries
# ---------------------------------------------------------------------------
def canonical_job(unit: str, action: str, direction: str) -> str:
    """The name a finished animation's job has: `<unit>_<verb>[_<direction>]`
    (what `rig` gives it, what `anim animate --output` writes, what `export`
    finds)."""
    name = config.studio_action(action)
    verb = config.STUDIO_ACTIONS[name]["verb"] if name else action
    stem = f"{unit}_{verb}"
    return stem if direction == config.DEFAULT_DIRECTION else f"{stem}_{direction}"


def base_views(unit: str, work_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """The views the base carries (`concept.views` of work/<unit>_base) that
    still have their file, with the curation state of each. The one reader of
    "what does the base carry": the Views tab, the mirror check and the
    Animate tab's gate all ask it, so they cannot disagree."""
    base_job = Job.open(f"{unit}_base", work_dir)
    if not base_job.manifest_path.exists():
        return {}
    stored = (base_job.read_manifest().get("concept") or {}).get("views") or {}
    record = read(unit, work_dir) or {}
    states = record.get("views") or {}
    out: dict[str, dict[str, Any]] = {}
    for view, entry in stored.items():
        path = base_job.render_dir / str((entry or {}).get("file") or "")
        if not (entry or {}).get("file") or not path.exists():
            continue
        out[view] = {**entry, "path": path, "state": states.get(view, "pending")}
    return out


def recorded_units(work_dir: Path | None = None) -> list[str]:
    """Every unit with a curation record (`work/<unit>_curation.json`)."""
    work = _work(work_dir)
    if not work.is_dir():
        return []
    return sorted(path.name[: -len("_curation.json")] for path in work.glob("*_curation.json"))


def base_parts_job(unit: str) -> str:
    """The job whose `post --parts` measures this unit's part palettes."""
    return f"{unit.strip().lower()}_base"


def base_part_palettes(unit: str, work_dir: Path | None = None) -> bool:
    """Whether the unit's base has measured part palettes on disk.

    Two different things have to be true before a cell can lock its colours,
    and only the second is a file: `config.UNIT_PARTS` DEFINES the regions
    (which joints locate them), and `<unit>_base/post/parts.json` MEASURES
    them - SAM run once over the accepted base sprite, recording each region's
    actual colours. Every direction then snaps to that measurement, which is
    what holds a cloak the same colour at s0 and s180.

    `post --parts` would ask for it only AFTER a render - so a unit without it
    would burn renders to discover a missing CPU artifact (observed
    2026-09-24, twice). Reported here so the Views tab can say so, and offer
    the one job that fixes it ("measure part palettes"), before the card is
    spent.
    """
    return (_work(work_dir) / base_parts_job(unit) / "post" / "parts.json").is_file()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def accept_base_from_concept(unit: str, work_dir: Path | None = None) -> dict[str, Any]:
    """`concept --pick`: the base is chosen, so the base gate opens - and the
    views of the old base are gone (the multi-view reference feature)."""
    record = read(unit, work_dir)
    if record is None:
        record = new_record(unit, origin="concept --pick")
    record["gates"]["base"] = "accepted"
    record["views"] = {}
    record.setdefault("gate_log", []).append(
        {"gate": "base", "state": "accepted", "at": _now(), "by": "concept --pick"})
    write(unit, record, work_dir)
    return record


def set_views(unit: str, states: dict[str, str], work_dir: Path | None = None
              ) -> dict[str, str]:
    """The views step under the base gate: `{view: state}` merged over what
    is there. `concept --pick-view` writes accepted (pending when
    provisional); `anim curate --accept-view back` / `--reject-view` / the
    Board write the rest. Returns every view's state."""
    for view, state in states.items():
        if view not in config.CONCEPT_VIEWS:
            raise CurationError(404, f"no view {view!r}; one of {list(config.CONCEPT_VIEWS)}")
        if state not in VIEW_STATES:
            raise CurationError(400, f"a view is one of {list(VIEW_STATES)}, not {state!r}")
    record = load(unit, work_dir)
    record.setdefault("views", {})
    for view, state in states.items():
        record["views"][view] = state
    record.setdefault("gate_log", []).extend(
        {"view": view, "state": state, "at": _now()} for view, state in states.items())
    write(unit, record, work_dir)
    return dict(record["views"])


def wanted_by(view: str) -> list[str]:
    """The side directions that ask for this view FIRST
    (`config.REFERENCE_BY_DIR`) - s0 wants `right`, s90 the back.
    It is what makes a missing slot on the ring worth rendering: the owner
    can see which facing goes without a reference if it stays missing. The
    table names the side cells only, so `top` (whose row is the t0 .. t315
    cells, #285, accepted but not v1) answers with nothing."""
    return [direction for direction, order in config.REFERENCE_BY_DIR.items()
            if order and order[0] == view]


def views_of(unit: str, record: dict[str, Any], work_dir: Path | None = None
             ) -> dict[str, dict[str, Any]]:
    """The views ring the Views tab shows: ALL EIGHT of `config.CONCEPT_VIEWS`,
    each with its curation state, file URL, whether the pick was provisional,
    and `missing` for the ones the base does not carry (or carries a record
    of, with the file gone).

    Why all eight rather than only the ones on disk: the ring is the owner's
    view of a fixed set of slots, and the one that matters most is usually
    the one that is NOT there - hooded_archer had attempts at back, left and
    top, and the gap was `right`, the view s0 hands Wan. Listing only what
    exists printed one flat sentence and never named the gap.

    `missing` is "the BASE does not carry this view" and `attempts` is how
    many candidates `concept --views` has drawn for it, because those are two
    different gaps with two different answers: an attempt nobody picked needs
    `--pick-view`, a view nobody drew needs the GPU. The file half of each row
    is `base_views`, which already merges the base's `concept.views`
    with their curation states and drops a view whose file is gone, so the
    ring and the Animate tab's reference cannot disagree about what the base
    carries.
    """
    work = _work(work_dir)
    base_job = Job.open(f"{unit}_base", work)
    have = base_views(unit, work)
    render_files = [p.name for p in jobs_api.FRAME_STAGES["render"](base_job)]
    attempts = view_attempts(unit, work)
    out: dict[str, dict[str, Any]] = {}
    for view, spec in config.CONCEPT_VIEWS.items():
        entry = have.get(view) or {}
        file = entry.get("file")
        url = (f"/jobs/{base_job.name}/frames/render/{render_files.index(file)}.png"
               if file and file in render_files else None)
        out[view] = {"view": view, "yaw": spec["yaw"], "elevation": spec["elevation"],
                     "state": (record.get("views") or {}).get(view) or "pending",
                     "missing": file is None, "attempts": attempts.get(view, 0),
                     "wanted_by": wanted_by(view),
                     "file": file, "url": url,
                     "candidate": entry.get("candidate"), "seed": entry.get("seed"),
                     "provisional": bool(entry.get("provisional")),
                     "picked": entry.get("picked"),
                     # The mirrored view feature. `mirror_of` is the partner
                     # this slot COULD be flipped from (null for back and top,
                     # which sit on the symmetry plane); `mirrored` /
                     # `mirrored_from` say the picture on the base already is
                     # one. Two different facts, so two different keys - the
                     # ring marks the second and never the first.
                     "mirror_of": config.mirror_of(view),
                     "mirrored": bool(entry.get("mirrored")),
                     "mirrored_from": (entry.get("mirror") or {}).get("view")}
    return out


def view_attempts(unit: str, work_dir: Path | None = None) -> dict[str, int]:
    """`{view: how many candidates `concept --views` has drawn}` from
    work/<unit>_concept/views/views.json, hidden attempts not counted (the
    Concept tab's own rule). A unit that has never run `--views` answers {}.

    Imported inside the call: `anim.stages.concept` imports this module to
    open the base gate on a pick.
    """
    from anim.stages import concept as concept_stage

    path = concept_stage.views_path(unit, _work(work_dir))
    if not path.exists():
        return {}
    try:
        record = concept_stage.migrate_views(read_json(path))
    except (OSError, ValueError, TypeError):
        return {}
    return {view: sum(1 for c in row.get("candidates") or [] if not c.get("hidden"))
            for view, row in (record.get("views") or {}).items()
            if view in config.CONCEPT_VIEWS}


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------
def gate_blocks(unit: str, stage: str, work_dir: Path | None = None) -> str | None:
    """Why `stage` may not run for `unit` now, or None when it may: a stage in
    `BASE_GATED` waits for the base gate (see the module docstring)."""
    if stage not in BASE_GATED:
        return None
    record = read(unit, _work(work_dir))
    if record is None:
        return None         # legacy: nothing asked for curation
    if record["gates"].get("base") != "accepted":
        return (f"{unit}: the base gate is pending - accept the base first "
                f"(`anim concept --unit {unit} --pick NN`, or the Concept tab)")
    return None


def views_blocked(unit: str, work_dir: Path | None = None) -> str | None:
    """The concept generation feature: why the views of `unit` may not be
    drawn now, or None. A view is drawn FROM the base, so it waits for a
    base to exist and for its gate to be accepted - GPU time on a character
    nobody has approved is the thing the gate exists to save. A unit with a
    base and no curation record at all (legacy) is not blocked."""
    work = _work(work_dir)
    base_job = Job.open(f"{unit}_base", work)
    final = None
    if base_job.manifest_path.exists():
        final = (jobs_api._manifest(base_job).get("render") or {}).get("final")
    if not final or not (base_job.render_dir / final).exists():
        return (f"{unit}: no base to draw the views from - pick one first "
                f"(`anim concept --unit {unit} --pick NN`, or the Concept tab)")
    record = read(unit, work)
    if record is None:
        return None
    if record["gates"].get("base") != "accepted":
        return (f"{unit}: the base gate is pending - accept the base before its views "
                f"are drawn (`anim concept --unit {unit} --pick NN`, or the Concept tab)")
    return None


def _concept_touches_views(args: argparse.Namespace) -> bool:
    if getattr(args, "views", None) is not None:
        return True
    for flag in ("img2img", "edit"):
        ref = getattr(args, flag, None)
        if isinstance(ref, str) and ref.partition(":")[0].strip().lower() in config.CONCEPT_VIEWS:
            return True
    return False


def gate_for_namespace(stage: str, args: argparse.Namespace,
                       work_dir: Path | None = None) -> str | None:
    """The gate answer for a parsed CLI line - what `anim <stage>` asks in
    `cli.main` and what `POST /jobs` asks with the same parser."""
    if stage == "concept":
        unit = getattr(args, "slug", None) or getattr(args, "unit", None)
        if unit and _concept_touches_views(args):
            return views_blocked(unit, work_dir)
        return None
    if stage == "animate":
        # The animate tab feature: every frame is drawn from the accepted base
        # (through the direction's view), so every line that reaches the GPU
        # waits behind the same gate as a view. An approval, a layout change,
        # a mirror or the scrub skeletons spend none of it.
        from anim.stages import animate as animate_stage

        unit = getattr(args, "slug", None) or getattr(args, "unit", None)
        if unit and animate_stage.spends_gpu(args):
            return views_blocked(unit, work_dir)
        return None
    if stage not in BASE_GATED:
        return None
    # Flags that make a GPU stage spend nothing: `--page-only` rebuilds a page
    # from what is on disk, `--skeleton-only` stops after the rig, and
    # `--dry-run` prints the plan. A gate exists to stop the card being spent on a
    # character nobody has approved, so a line that spends none of it must not
    # be a 423 - refusing a dry run would hide the very reasons the owner
    # opened it to read.
    if getattr(args, "page_only", False) or getattr(args, "skeleton_only", False) \
            or getattr(args, "dry_run", False):
        return None         # no GPU in it
    unit: str | None = None
    job_name = getattr(args, "job", None)
    if isinstance(job_name, str) and job_name and stage in ("clip", "render"):
        job = jobs_api.open_job(job_name, _work(work_dir))
        if job is None:
            return None     # the stage itself will say the job is missing
        unit = jobs_api._manifest(job).get("slug") or None
    else:
        unit = getattr(args, "slug", None) or getattr(args, "unit", None)
    if not unit:
        return None
    return gate_blocks(unit, stage, work_dir)


def gate_for_request(stage: str, argv: list[str], work_dir: Path | None = None
                     ) -> str | None:
    """`POST /jobs`: parse the argv with the stage's own parser, then ask.
    Argv that does not parse is not this function's problem (the run manager
    turns it into the 400); `--force` on the line skips the gate as the CLI
    would."""
    import contextlib
    import io

    from anim import cli

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            args = cli.build_parser().parse_args([stage, *argv])
    except SystemExit:
        return None
    if getattr(args, "force", False):
        return None
    return gate_for_namespace(stage, args, work_dir)
