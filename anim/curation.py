"""The curation board feature - where the human says yes or no.

    work/<unit>_curation.json      one record per unit: the gates and every cell

A unit's board is a grid of cells, one per (studio action x direction), and
each cell is one job under work/ (`<unit>_<verb>[_<direction>]`, the name `rig`
gives it). The owner accepts or rejects a cell, marks single frames, or asks
for the cell again with a new seed; the marks live here, and a rejected frame
is also dropped from the job's picks.json through the same `ReviewApp` the
review page uses, so `strip` / `post` read the decision the next time they run.

THE RECORD (`anim.curation/1`):

    {
      "schema": "anim.curation/1", "unit": "knight", "origin": "concept --pick",
      "gates": {"base": "accepted"},
      "views": {"back": "accepted", "left": "pending"},
      "cells": {
        "attack/s0": {"job": "knight_strike", "status": "accepted",
                      "frames": {"3": "reject"}, "seed": 12345,
                      "regenerated_from": 777, "decided_at": "..."}
      }
    }

Why one file per unit beside the jobs rather than `work/<unit>/curation.json`:
a unit has no directory of its own - its jobs are siblings named
`<unit>_<action>` - and a `work/<unit>/` would read as a job with no manifest
to `GET /jobs`. A flat file sits where `project.json` and `gate_parts.json`
already sit, and one `ls work/*_curation.json` lists every unit under review.

THE GATES are the project setting `curation` in work/project.json
(sprite-studio.md S2 "Curation": a project setting; default a gate after the
base with batch review at the end; also every step, or end only). There is
one gate, `base`. The rest of the human approval is per picture: each view of
the base is accepted on the Board, and the matrix refuses a direction whose
view is not (`anim.stages.matrix.prerequisite`). `gate_blocks` is the one
question the API's `POST /jobs`, the board's regenerate and the CLI's GPU
stages ask before spending the card:

    gates   `clip`, `render`, `bakeoff`, `matrix` and `run` wait for the
            base gate
    every   the same gates, and a cell's GPU stage also waits until every
            EARLIER cell that has a job is accepted - direction-major
            (every action of s0, then s45, ...), so the side view game-02
            ships is finished before a second direction is started
    end     nothing blocks; the board is a batch review at the end

CPU stages (`rig`, `pick`, `strip`, `post`) never block: they are what makes
the frames there are to judge, and they cost seconds.

LEGACY: a unit with no curation file is not gated. The knight's published
jobs predate the board and the gates, and a rule that refused `anim render
knight_strike` until someone accepted a base nobody was asked about would be
a regression, not a feature. Only `concept --pick` writes a real gate record
(base accepted); a board edit on a legacy unit creates its record with the
gate accepted and `"origin": "legacy"` so nothing that worked yesterday stops
today.

OLDER RECORDS may carry a `turnaround` key under `gates`, from a stage that no
longer exists. It is left in the file as written and ignored: no gate check
reads it and the board does not report it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import secrets
from pathlib import Path
from typing import Any, Callable, Iterable

from anim import config
from anim.api import jobs as jobs_api
from anim.api import settings
from anim.job import Job
from anim.util import read_json, write_json_in_place

CURATION_SCHEMA = "anim.curation/1"
GATES: tuple[str, ...] = ("base",)
GATE_STATES: tuple[str, ...] = ("pending", "accepted")
# The multi-view reference feature: the views of the base (`concept
# --views`) are a step under the base gate, one state per view.
VIEW_STATES: tuple[str, ...] = ("pending", "accepted", "rejected")
CELL_STATES: tuple[str, ...] = ("pending", "accepted", "rejected")
FRAME_MARKS: tuple[str, ...] = ("accept", "reject")
MODES = settings.CURATION_MODES

# The stages that spend the GPU on a unit's cells, and so wait for its gate.
# `base` and `concept` MAKE the base; they are never gated.
BASE_GATED = frozenset({"clip", "render", "bakeoff", "run", "matrix"})
# Which set the strip is drawn from, best first (the editor's rule, frames.ts).
STRIP_SETS = ("post", "clip", "frames")
SEED_MAX = 2 ** 53


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
        "cells": {},
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
    record.setdefault("cells", {})
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


def cell_key(action: str, direction: str) -> str:
    """`("strike" | "attack", "s0")` -> `"attack/s0"`: the studio's action name
    and the direction cell, which is how the board and the CLI name a cell."""
    name = config.studio_action(action)
    if name is None:
        raise CurationError(400, f"{action!r} is not a studio action; one of "
                                 f"{list(config.STUDIO_ACTIONS)} (or the verbs "
                                 f"{[r['verb'] for r in config.STUDIO_ACTIONS.values()]})")
    key = str(direction).strip().lower()
    if key not in config.DIRECTIONS:
        raise CurationError(400, f"unknown direction {direction!r}; one of "
                                 f"{list(config.DIRECTIONS)}")
    return f"{name}/{key}"


def parse_key(key: str) -> tuple[str, str]:
    """`"attack/s0"` (or `"strike"`, the default cell) -> `("attack", "s0")`."""
    action, sep, direction = str(key).partition("/")
    name, cell = cell_key(action, direction if sep else config.DEFAULT_DIRECTION).split("/", 1)
    return name, cell


def _blank_cell(job: str | None = None) -> dict[str, Any]:
    return {"job": job, "status": "pending", "frames": {}, "seed": None,
            "regenerated_from": None, "decided_at": None}


def cell_order(key: str) -> tuple[int, int]:
    """Direction-major: every action of s0 before any of s45 (`every` mode)."""
    action, direction = key.split("/", 1)
    return (list(config.DIRECTIONS).index(direction),
            list(config.STUDIO_ACTIONS).index(action))


# ---------------------------------------------------------------------------
# The unit's cells: which job is which cell
# ---------------------------------------------------------------------------
def canonical_job(unit: str, action: str, direction: str) -> str:
    """The name `rig` gives the cell's job: `<unit>_<verb>[_<direction>]`."""
    name = config.studio_action(action)
    verb = config.STUDIO_ACTIONS[name]["verb"] if name else action
    stem = f"{unit}_{verb}"
    return stem if direction == config.DEFAULT_DIRECTION else f"{stem}_{direction}"


def cell_of_row(row: dict[str, Any]) -> str | None:
    """The cell a `GET /jobs` row belongs to, or None when it is not one
    (a base or concept job, a bake-off cell, a chunk, a job of another kind)."""
    if row["name"].startswith("bakeoff_"):
        return None
    name = config.studio_action(row.get("action") or "")
    direction = str(row.get("direction") or config.DEFAULT_DIRECTION).lower()
    if name is None or direction not in config.DIRECTIONS:
        return None
    return f"{name}/{direction}"


def cells_for_unit(unit: str, rows: list[dict[str, Any]],
                   record: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """`{cell key: the job row}` for the unit's jobs, in board order.

    When two jobs land on one cell (`knight_strike` and `knight_strike_phases`
    are both the knight's attack at s0) the record's own choice wins, then the
    canonical name, then the newest - the same rule every time, so a cell
    does not change job between two reloads.
    """
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("slug") != unit:
            continue
        key = cell_of_row(row)
        if key is not None:
            by_cell.setdefault(key, []).append(row)
    chosen: dict[str, dict[str, Any]] = {}
    for key, candidates in by_cell.items():
        pinned = ((record or {}).get("cells", {}).get(key) or {}).get("job")
        action, direction = key.split("/", 1)
        canonical = canonical_job(unit, action, direction)
        for want in (pinned, canonical):
            hit = next((r for r in candidates if r["name"] == want), None)
            if hit is not None:
                chosen[key] = hit
                break
        else:
            chosen[key] = max(candidates, key=lambda r: r["updated"])
    return dict(sorted(chosen.items(), key=lambda item: cell_order(item[0])))


def unit_rows(unit: str, work_dir: Path | None = None) -> list[dict[str, Any]]:
    return [row for row in jobs_api.list_jobs(_work(work_dir)) if row.get("slug") == unit]


def list_units(work_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every unit that has a cell job or a curation record."""
    work = _work(work_dir)
    units: dict[str, dict[str, Any]] = {}
    for row in jobs_api.list_jobs(work):
        if row.get("slug") and cell_of_row(row) is not None:
            units.setdefault(row["slug"], {"unit": row["slug"], "cells": 0, "record": False})
            units[row["slug"]]["cells"] += 1
    if work.is_dir():
        for path in sorted(work.glob("*_curation.json")):
            unit = path.name[: -len("_curation.json")]
            units.setdefault(unit, {"unit": unit, "cells": 0, "record": False})
            units[unit]["record"] = True
    return [units[k] for k in sorted(units)]


# ---------------------------------------------------------------------------
# The strip of one cell
# ---------------------------------------------------------------------------
def _seed_of(manifest: dict[str, Any], stage: str) -> int | None:
    block = manifest.get("clip_render" if stage == "clip" else "render") or {}
    seed = block.get("seed")
    return int(seed) if isinstance(seed, int) else None


def regenerate_stage(row: dict[str, Any]) -> str:
    """Which GPU stage re-makes the cell: `clip` on the clip route (a rig job
    with a clip, or a rig job that has not been stripped), else `render`."""
    stages = row.get("stages") or []
    if "clip" in stages or ("rig" in stages and "strip" not in stages):
        return "clip"
    return "render"


def strip_of(job: Job, marks: dict[str, str]) -> dict[str, Any]:
    """The `strip` the board shows for one cell: each frame with its source
    frame number (what picks.json and the `frames` marks use), URL and mark.

    With picks, the strip is the picks PLUS the frames the marks reject - a
    rejected frame has left picks.json but stays on the board, dimmed, so the
    owner sees what they threw out and can take it back. From `post` when it
    has one frame per shown pick (post frame k is pick k; the rejects are
    what `post` still holds from before the reject), else the `clip` /
    `frames` PNG at each index. Without picks, every frame of the best set.
    `render` and `strip` hold N-cell strips, not frames, so they are not shown.
    """
    picks: list[int] = []
    if job.picks_path.exists():
        try:
            picks = [int(i) for i in read_json(job.picks_path).get("picks") or []]
        except (OSError, ValueError, TypeError):
            picks = []
    rejected = {int(n) for n, m in marks.items() if m == "reject" and str(n).isdigit()}
    shown = sorted(set(picks) | rejected)
    sets = {name: jobs_api.FRAME_STAGES[name](job) for name in STRIP_SETS}
    chosen: str | None = next((name for name in STRIP_SETS if sets[name]), None)
    strip: list[dict[str, Any]] = []
    if chosen is None:
        return {"set": None, "picks": picks, "strip": strip}
    if shown and chosen == "post" and len(sets["post"]) == len(shown):
        pairs = list(zip(shown, range(len(shown))))
    elif shown and chosen == "post":
        # post/ does not match the picks (a re-pick since it ran): show the
        # source at the picks rather than a post set that means something else.
        chosen = next((n for n in ("clip", "frames") if sets[n]), None)
        if chosen is None:
            return {"set": "post", "picks": picks, "strip": strip, "stale": True}
        pairs = [(i, i) for i in shown if 0 <= i < len(sets[chosen])]
    elif shown:
        pairs = [(i, i) for i in shown if 0 <= i < len(sets[chosen])]
    else:
        pairs = [(i, i) for i in range(len(sets[chosen]))]
    for source, index in pairs:
        strip.append({"index": source, "mark": marks.get(str(source)),
                      "url": f"/jobs/{job.name}/frames/{chosen}/{index}.png"})
    return {"set": chosen, "picks": picks, "strip": strip}


# ---------------------------------------------------------------------------
# The board - GET /curation/{unit}
# ---------------------------------------------------------------------------
def mode_of(work_dir: Path | None = None) -> str:
    return str(settings.stored(_work(work_dir)).get("curation") or "gates")


def board_directions(keys: Iterable[str]) -> list[str]:
    """The grid's columns: the default elevation's eight, and any other
    direction the unit's cells actually use (a `top` cell, or a hand-made job
    that landed on one). In `config.DIRECTIONS` order.

    Why the eight are always there: the columns ARE the question the board
    asks - which facings of this unit are done - and a unit with nothing
    rendered has the same eight to answer for as one with fifteen cells.
    Deriving the columns from the cells alone drew an empty unit as a table
    with one header and no body, which reads as a broken widget rather than
    as "nothing here yet" (hooded_archer, 2026-09-23).
    """
    order = list(config.DIRECTIONS)
    used = {str(key).split("/", 1)[1] for key in keys}
    eight = {config.direction_name(yaw, config.DEFAULT_ELEVATION)
             for yaw in config.DIRECTION_YAWS}
    return sorted(used | eight, key=order.index)


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

    `matrix` locks every cell to it; `post` would ask for it only AFTER Wan
    has rendered the clip - so a unit without it would burn renders to
    discover a missing CPU artifact (observed 2026-09-24, twice). Reported
    here so the Board can say so, and offer the one job that fixes it, before
    the card is spent; the matrix also refuses the cell (`prerequisite`).
    """
    return (_work(work_dir) / base_parts_job(unit) / "post" / "parts.json").is_file()


def gate_blockers(unit: str) -> dict[str, str | None]:
    """Why each gate cannot be ACCEPTED right now, or None when it can. Pure.
    The base gate has no blocker: `concept --pick` is the only thing that
    opens it and it opens it over a real file. Kept as a map so the API's
    shape (`gate_blocked`) does not change with the number of gates."""
    return {gate: None for gate in GATES}


def waiting_views(unit: str, views: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """The views step as `waiting` entries. Pure over `views_of`'s rows.

    The slots nobody has drawn at all come first - they cost GPU and nothing
    downstream can start without them - then the ones that need a decision:
    an attempt waiting for a pick, and a picked view waiting for a verdict.
    Each `why` says the thing to DO, since these are the entries an owner
    reads when the board has nothing else to show.
    """
    def entry(view: str, why: str) -> dict[str, str]:
        return {"kind": "view", "key": view, "why": why}

    def because(row: dict[str, Any]) -> str:
        wanted = ", ".join(row.get("wanted_by") or [])
        return f"; {wanted} wants it" if wanted else ""

    undrawn = [entry(view, f"no {view} view of the base yet - render it on the Concept tab "
                           f"(`anim concept --unit {unit} --views {view}`){because(row)}")
               for view, row in views.items()
               if row.get("missing") and not row.get("attempts")]
    decide = []
    for view, row in views.items():
        attempts = int(row.get("attempts") or 0)
        if row.get("missing") and attempts:
            decide.append(entry(view, f"{attempts} attempt{'' if attempts == 1 else 's'} at the "
                                      f"{view} view and none picked - pick one on the Concept "
                                      f"tab (`anim concept --unit {unit} --pick-view "
                                      f"{view}=NN`){because(row)}"))
        elif not row.get("missing") and row.get("state") == "pending":
            decide.append(entry(view, f"judge the {view} view of the base - accept it "
                                      f"or reject it"))
    return undrawn + decide


def waiting_on(record: dict[str, Any], cells: dict[str, dict[str, Any]],
               mode: str, *, unit: str | None = None,
               views: dict[str, dict[str, Any]] | None = None,
               exists: bool = False,
               base_parts: bool = True) -> list[dict[str, str]]:
    """What the owner has to decide before the pipeline goes on. Pure.

    In pipeline order: the gate, the views of the base, the base's part
    palettes, then the cells. A unit with no cells is the first three and
    none of the last - which is how a freshly picked unit came to report
    "nothing waiting on you" while nothing at all had been done to it
    (hooded_archer).

    The views are named while any of them is undrawn, unpicked or unjudged:
    an accepted view is what a direction of the matrix waits for. And a
    legacy unit (`exists` False: no record, nothing gated - see the
    module docstring) is not on the studio's path at all, so only its cells
    wait on anyone; the knight's published jobs must not grow a to-do list
    for a base nobody ever asked about.
    """
    out: list[dict[str, str]] = []
    unit = unit or str(record.get("unit") or "<unit>")
    if mode != "end":
        if record["gates"].get("base") != "accepted":
            out.append({"kind": "gate", "key": "base",
                        "why": "accept the base (the concept pick) before any cell renders"})
        if exists:
            out.extend(waiting_views(unit, views or {}))
            # The part palettes every matrix cell locks to. A CPU job, not a
            # decision, but it is on the owner's path and the matrix refuses
            # every cell until it exists.
            if not base_parts:
                out.append({"kind": "parts", "key": base_parts_job(unit),
                            "why": f"no part palettes yet - `anim post {base_parts_job(unit)} "
                                   f"--parts` measures them off the accepted base (CPU, no GPU); "
                                   f"every direction locks its colours to them"})
    for key, cell in cells.items():
        status = cell.get("status", "pending")
        if status == "rejected":
            out.append({"kind": "cell", "key": key,
                        "why": f"rejected - regenerate {cell.get('job') or key}"})
        elif status == "pending" and cell.get("set"):
            out.append({"kind": "cell", "key": key,
                        "why": f"review {cell.get('job') or key}"})
        else:
            continue
        if mode == "every":
            break         # one at a time: the rest are queued behind this one
    return out


def board(unit: str, work_dir: Path | None = None, *, mode: str | None = None
          ) -> dict[str, Any]:
    """Everything the board page needs, from the files as they are now."""
    work = _work(work_dir)
    mode = mode or mode_of(work)
    record = load(unit, work)
    rows = cells_for_unit(unit, jobs_api.list_jobs(work), record)
    cells: dict[str, dict[str, Any]] = {}
    for key, row in rows.items():
        stored = dict(record["cells"].get(key) or _blank_cell(row["name"]))
        stored["job"] = row["name"]
        job = Job.open(row["name"], work)
        manifest = jobs_api._manifest(job)
        stage = regenerate_stage(row)
        action, direction = key.split("/", 1)
        cells[key] = {
            **stored, "key": key, "action": action, "direction": direction,
            "verb": row.get("action"), "stages": row.get("stages") or [],
            "stage": stage, "current_seed": _seed_of(manifest, stage),
            "rerender_requested": bool(row.get("rerender_requested")),
            **strip_of(job, stored.get("frames") or {}),
        }
    # Cells the record knows but whose job is gone: shown empty, still counted.
    for key, stored in record["cells"].items():
        if key in cells:
            continue
        action, direction = key.split("/", 1)
        cells[key] = {**stored, "key": key, "action": action, "direction": direction,
                      "verb": None, "stages": [], "stage": None, "current_seed": None,
                      "rerender_requested": False, "set": None, "picks": [], "strip": []}
    ordered = dict(sorted(cells.items(), key=lambda item: cell_order(item[0])))
    exists = curation_path(unit, work).exists()
    views = views_of(unit, record, work)
    return {
        "unit": unit, "mode": mode, "origin": record.get("origin"),
        "exists": exists,
        "path": str(curation_path(unit, work)),
        # Only the gates that exist: an older record's `turnaround` key stays
        # in the file and is not reported (the module docstring).
        "gates": {gate: record["gates"].get(gate, "pending") for gate in GATES},
        # Why a second map beside `gates` rather than a richer `gates`: the
        # record's `gates` IS `{gate: state}` on disk and in the reducer, and
        # widening it here would make the API's shape and the file's diverge.
        "gate_blocked": gate_blockers(unit),
        "views": views,
        # The part palettes the cells lock to - a CPU artifact, not a decision,
        # but the matrix refuses every cell without it.
        "base_parts": base_part_palettes(unit, work),
        "base_parts_job": base_parts_job(unit),
        "actions": list(config.STUDIO_ACTIONS),
        "directions": board_directions(ordered),
        "cells": ordered,
        "waiting": waiting_on(record, ordered, mode, unit=unit, views=views,
                              exists=exists,
                              base_parts=base_part_palettes(unit, work)),
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def set_gate(unit: str, gate: str, state: str, work_dir: Path | None = None,
             *, origin: str | None = None) -> dict[str, Any]:
    if gate not in GATES:
        raise CurationError(404, f"no gate {gate!r}; one of {list(GATES)}")
    if state not in GATE_STATES:
        raise CurationError(400, f"a gate is one of {list(GATE_STATES)}, not {state!r}")
    record = load(unit, work_dir)
    if origin:
        record["origin"] = origin
    record["gates"][gate] = state
    record.setdefault("gate_log", []).append({"gate": gate, "state": state, "at": _now()})
    write(unit, record, work_dir)
    return record


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
    """The matrix directions that ask for this view FIRST
    (`config.MATRIX_REFERENCE_BY_DIR`) - s0 wants `right`, s90 the back.
    It is what makes a missing slot on the ring worth rendering: the owner
    can see which facing goes without a reference if it stays missing. The
    table names the side cells only, so `top` (whose row is the t0 .. t315
    cells, #285, accepted but not v1) answers with nothing."""
    return [direction for direction, order in config.MATRIX_REFERENCE_BY_DIR.items()
            if order and order[0] == view]


def views_of(unit: str, record: dict[str, Any], work_dir: Path | None = None
             ) -> dict[str, dict[str, Any]]:
    """The views ring the board shows: ALL EIGHT of `config.CONCEPT_VIEWS`,
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
    is `matrix.base_views`, which already merges the base's `concept.views`
    with their curation states and drops a view whose file is gone, so the
    board and the matrix's own reference picking cannot disagree about what
    the base carries. Imported inside the call because `anim.stages.matrix`
    imports this module.
    """
    from anim.stages import matrix as matrix_stage

    work = _work(work_dir)
    base_job = Job.open(f"{unit}_base", work)
    have = matrix_stage.base_views(unit, work)
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


def _cell_and_job(unit: str, key: str, work_dir: Path | None
                  ) -> tuple[dict[str, Any], dict[str, Any], Job | None]:
    work = _work(work_dir)
    record = load(unit, work)
    rows = cells_for_unit(unit, jobs_api.list_jobs(work), record)
    row = rows.get(key)
    cell = record["cells"].get(key)
    if cell is None:
        if row is None:
            raise CurationError(404, f"{unit} has no job for cell {key}")
        cell = _blank_cell(row["name"])
        record["cells"][key] = cell
    if row is not None:
        cell["job"] = row["name"]
    job = Job.open(cell["job"], work) if cell.get("job") else None
    return record, cell, job


def set_cell_status(unit: str, action: str, direction: str, state: str,
                    work_dir: Path | None = None) -> dict[str, Any]:
    """Accept / reject / reopen one cell. Accepting locks its frames."""
    if state not in CELL_STATES:
        raise CurationError(400, f"a cell is one of {list(CELL_STATES)}, not {state!r}")
    key = cell_key(action, direction)
    record, cell, _ = _cell_and_job(unit, key, work_dir)
    cell["status"] = state
    cell["decided_at"] = _now() if state != "pending" else None
    write(unit, record, work_dir)
    return record["cells"][key]


def set_frames(unit: str, action: str, direction: str, marks: dict[Any, Any],
               work_dir: Path | None = None) -> dict[str, Any]:
    """Mark frames of one cell; `{n: "accept" | "reject" | null}` merges over
    the marks there are (null clears one).

    A rejected frame leaves picks.json - through `ReviewApp.set_picks`, the
    review page's own write, so the file is byte for byte what a click there
    produces - and a frame whose reject is cleared comes back. An accepted
    cell is locked: reopen it (status pending) to mark frames again.
    """
    from anim.stages.review import ReviewApp

    key = cell_key(action, direction)
    record, cell, job = _cell_and_job(unit, key, work_dir)
    if cell.get("status") == "accepted":
        raise CurationError(409, f"{key} is accepted and locked; set it pending "
                                 f"to mark frames again")
    current: dict[str, str] = dict(cell.get("frames") or {})
    for raw, mark in marks.items():
        try:
            index = int(raw)
        except (TypeError, ValueError):
            raise CurationError(400, f"frame keys are frame numbers, not {raw!r}") from None
        if index < 0:
            raise CurationError(400, f"frame {index} is negative")
        if mark is None or mark == "":
            current.pop(str(index), None)
        elif mark in FRAME_MARKS:
            current[str(index)] = mark
        else:
            raise CurationError(400, f"a frame mark is one of {list(FRAME_MARKS)} "
                                     f"or null, not {mark!r}")
    was_rejected = {int(n) for n, m in (cell.get("frames") or {}).items() if m == "reject"}
    now_rejected = {int(n) for n, m in current.items() if m == "reject"}
    picks: list[int] | None = None
    if job is not None and (was_rejected or now_rejected):
        try:
            review = ReviewApp(job)
        except FileNotFoundError as exc:
            raise CurationError(409, str(exc)) from None
        have = set(review.state()["picks"])
        wanted = (have | (was_rejected - now_rejected)) - now_rejected
        picks = review.set_picks(sorted(wanted))["picks"]
    cell["frames"] = dict(sorted(current.items(), key=lambda item: int(item[0])))
    write(unit, record, work_dir)
    out = dict(record["cells"][key])
    if picks is not None:
        out["picks"] = picks
    return out


def new_seed(avoid: int | None = None) -> int:
    while True:
        seed = secrets.randbelow(SEED_MAX)
        if seed != avoid:
            return seed


def regenerate(unit: str, action: str, direction: str, start: Callable[..., Any],
               work_dir: Path | None = None, *, seed: int | None = None) -> dict[str, Any]:
    """Ask for the cell again with a new seed.

    `start(stage, args=...)` is the run manager's `start` (a GPU run, so the
    GPU rule applies and a busy card is its 409); the record is written only
    once the run is accepted, so a refused request leaves no half-recorded
    seed. Returns the cell plus the run.
    """
    key = cell_key(action, direction)
    work = _work(work_dir)
    record, cell, job = _cell_and_job(unit, key, work)
    if job is None:
        raise CurationError(404, f"{key} has no job to regenerate")
    row = next((r for r in jobs_api.list_jobs(work) if r["name"] == job.name), None)
    if row is None:
        raise CurationError(404, f"no job {job.name!r} in {work}")
    if "animate" in jobs_api._manifest(job):
        # The animate tab feature wrote this cell frame by frame; there is no
        # clip or strip behind it to re-render with a new seed.
        raise CurationError(409, f"{key} was drawn on the Animate tab - redraw its frames "
                                 f"there (`anim animate --unit {unit} --action {action} "
                                 f"--dir {direction}`)")
    stage = regenerate_stage(row)
    previous = cell.get("seed")
    if previous is None:
        previous = _seed_of(jobs_api._manifest(job), stage)
    seed = new_seed(previous) if seed is None else int(seed)
    run = start(stage, args={"job": job.name, "seed": seed})
    cell.update({"seed": seed, "regenerated_from": previous, "status": "pending",
                 "frames": {}, "decided_at": None, "regenerated_at": _now(),
                 "run": getattr(run, "id", None), "stage": stage})
    write(unit, record, work)
    out = dict(record["cells"][key])
    out["run"] = run.record() if hasattr(run, "record") else run
    return out


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------
def gate_blocks(unit: str, stage: str, *, action: str | None = None,
                direction: str = config.DEFAULT_DIRECTION,
                work_dir: Path | None = None, mode: str | None = None) -> str | None:
    """Why `stage` may not run for `unit` now, or None when it may.

    The module docstring has the three modes. `direction` is the cell the
    stage renders; `action` is what `every` mode orders on (None: only the
    gate applies).
    """
    if stage not in BASE_GATED:
        return None
    work = _work(work_dir)
    mode = mode or mode_of(work)
    if mode == "end":
        return None
    record = read(unit, work)
    if record is None:
        return None         # legacy: nothing asked for curation
    hint = f"anim curate {unit} --accept-gate"
    if stage in BASE_GATED and record["gates"].get("base") != "accepted":
        return (f"{unit}: the base gate is pending - accept the base first "
                f"(`anim concept --unit {unit} --pick NN`, `{hint} base`, or the board)")
    direction = str(direction or config.DEFAULT_DIRECTION).lower()
    if mode == "every" and action is not None:
        try:
            key = cell_key(action, direction)
        except CurationError:
            return None
        cells = cells_for_unit(unit, jobs_api.list_jobs(work), record)
        for earlier, row in cells.items():
            if cell_order(earlier) >= cell_order(key):
                break
            status = (record["cells"].get(earlier) or {}).get("status", "pending")
            if status != "accepted":
                return (f"{unit}: every-step curation - {earlier} ({row['name']}) is "
                        f"{status}; accept it before {key} renders "
                        f"(`anim curate {unit} --accept-cell {earlier}`, or the board)")
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
                f"are drawn (`anim curate {unit} --accept-gate base`, or the board)")
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
                       work_dir: Path | None = None, *, mode: str | None = None
                       ) -> str | None:
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
    # `--dry-run` (the action matrix feature) prints the plan and every
    # prerequisite verdict. A gate exists to stop the card being spent on a
    # character nobody has approved, so a line that spends none of it must not
    # be a 423 - refusing a dry run would hide the very reasons the owner
    # opened it to read.
    if getattr(args, "page_only", False) or getattr(args, "skeleton_only", False) \
            or getattr(args, "dry_run", False):
        return None         # no GPU in it
    work = _work(work_dir)
    unit: str | None = None
    action: str | None = None
    direction = config.DEFAULT_DIRECTION
    job_name = getattr(args, "job", None)
    if isinstance(job_name, str) and job_name and stage in ("clip", "render"):
        job = jobs_api.open_job(job_name, work)
        if job is None:
            return None     # the stage itself will say the job is missing
        manifest = jobs_api._manifest(job)
        unit = manifest.get("slug") or None
        action = manifest.get("action") or None
        direction = jobs_api.direction_of(manifest)
    else:
        unit = getattr(args, "slug", None) or getattr(args, "unit", None)
        action = getattr(args, "action", None)
        yaw = getattr(args, "yaw", None)
        elevation = getattr(args, "elevation", None)
        if yaw is not None and elevation is not None:
            try:
                direction = config.direction_name(yaw, elevation)
            except ValueError:
                direction = config.DEFAULT_DIRECTION
    if not unit:
        return None
    return gate_blocks(unit, stage, action=action, direction=direction,
                       work_dir=work, mode=mode)


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
