"""Stage: `anim animate` - the owner directs every frame of one animation, by hand

The animate tab feature. The video routes and the Keyframes tab were rejected
(DEVLOG 2026-09-26): the owner draws every frame of an animation himself,
direction by direction, with approval as the gate, and the approved frames
ship to the game. One animation is one (unit, action, direction) cell.

    anim animate --unit hooded_archer --action attack --dir left --frame wind-up \\
        --draw-from view --skeleton                 # one keyframe, drawn from the left view
    anim animate --unit hooded_archer --action attack --dir left --frame 4   # an in-between
    anim animate ... --frame 3 --approve            # or --reject, --select NN
    anim animate ... --batch all                    # keyframes first; once approved, the in-betweens
    anim animate ... --batch keys | rejected | inbetweens
    anim animate ... --approve-reviewed             # every frame to review -> approved
    anim animate ... --add-frame 5                  # a new frame after frame 5 (0: at the start)
    anim animate ... --frame 6 --remove-frame       # its attempts are kept in the record
    anim animate ... --frame 6 --key action         # mark as a keyframe; --unkey un-marks
    anim animate ... --mirror                       # flip the partner direction's frames in
    anim animate ... --output [--replace-job]       # the finished frames -> work/<unit>_<verb>[_<dir>]/
    anim animate ... --inpaint f03:01 --mask canvas.png   # --replace REF --image, --hide, --show
    anim animate ... --list | --skeletons

    work/<unit>_animate_<action>_<direction>/
      job.json                  schema anim.animate/1: the frame row (id, keyframe?, phase,
                                where in the clip), per frame every attempt, the current one
                                and the verdict; the mirror and the output blocks
      attempts/<id>_NN.png      one per attempt; never deleted (hide, then delete-file)
      skeletons/<W>x<H>_<fps>fps/   the rig clip posed at the direction's yaw, frame by frame
      game/<sha>.png            a frame at game size, cached by the source's hash

THE ROW. It starts as the phase template: `ACTION_PHASES[action]["budget"]`
frames, each sampling the rig clip where `pick.phase_positions` puts it, and
the phase keys (`PhasePlan.keys()` - for attack 0/2/6/9/11) marked as
keyframes with their phase. The owner adds and removes frames and marks or
unmarks keyframes; a frame keeps a stable id (`f03`) through all of it, so
its attempt files never collide, and a removed frame's attempts move to the
record's `removed` list rather than vanishing.

STATUS per frame: `generating` (an attempt is planned and not rendered),
`empty` (nothing to look at), `review` (the current attempt has no verdict),
`approved` / `rejected` (the verdict on the current attempt). Every attempt is
kept; making an older one current puts the frame back to review unless that
attempt is the judged one. Editing the approved attempt (pixel edit, inpaint)
clears the approval: the owner approved other pixels.

DRAW FROM. `view` - the direction's own accepted view (the base itself at
s270); a direction whose view is not accepted is blocked with the reason,
never substituted. `keys` - the two nearest keyframes on either side of the
frame (a loop wraps round), which must both be APPROVED or the frame is
refused. `both` - the view and the two keys. The skeleton, when ticked, is the
last image; Qwen-Image-Edit-2509 takes three, so `both` + skeleton is refused
rather than silently dropping one. Keyframes default to the view, in-betweens
to the keys. Every attempt records what it was drawn from, with hashes.

GENERATE ALL is two steps with approval between them: while any keyframe is
unapproved it draws the keyframes that are empty or rejected, then stops; once
every keyframe is approved the same press fills the in-betweens. Frames
waiting for review are not drawn again - pressing twice tops up, it does not
pile attempts onto work the owner has not looked at.

MIRROR is per animation and only ever a press: the partner direction's frames
(right<->left, back_right<->back_left, front_left<->front_right) flipped
horizontally, no GPU, each landing as a new attempt TO REVIEW with a
handedness warning.

OUTPUT. When every frame is approved the frames go through the pipeline's own
post chain (matte, downscale to the unit's height on the 64 px canvas, palette
snap) into the cell's ordinary job - `work/<unit>_<verb>[_<dir>]/` with
job.json, post/<unit>_<verb>_<n>.png, post/report.json and a picks.json whose
timing block carries the phases - which is what `anim export` already
reads. A job there that this stage
did not write (the video route's) is refused unless `--replace-job` moves it
to work/_replaced/ first.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image, ImageOps

from anim import comfy, config, curation, imaging, models_lock, outcome, palettekit, rigkit
from anim.job import JOB_SCHEMA, Job
from anim.stages import concept as concept_stage
from anim.stages import pick as pick_stage
from anim.stages import pose as pose_stage
from anim.stages import render as render_stage
from anim.util import read_json, sha256_bytes, sha256_file, write_bytes_in_place, write_json_in_place

NAME = "animate"
HELP = "the owner draws every frame of one animation, approves each, and ships the approved"

SCHEMA = "anim.animate/1"
SOURCE = "animate"
MIRROR_SOURCE = "mirror"
WORKFLOW = concept_stage.VIEW_WORKFLOW
FAMILY = "qwen-edit"
ATTEMPTS_DIR = "attempts"
SKELETONS_DIR = "skeletons"
GAME_DIR = "game"
REPLACED_DIR = "_replaced"
STATUSES = ("empty", "generating", "review", "approved", "rejected")
BATCHES = ("all", "keys", "rejected", "inbetweens")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Names: the action, the direction, the job
# ---------------------------------------------------------------------------
def action_name(action: str) -> str:
    """The studio's name for an action ("strike" and "attack" both give
    "attack"); anything else is refused with the list."""
    name = config.studio_action(str(action or ""))
    if name is None:
        raise ValueError(f"no studio action {action!r}; one of {list(config.STUDIO_ACTIONS)}")
    return name


def side_directions() -> list[str]:
    """The eight side cells, in compass order."""
    return [name for name, (_y, el) in config.DIRECTIONS.items()
            if el == config.ANIMATE_ELEVATION]


def direction_name(direction: str | None) -> str:
    """A side direction cell. A view name is accepted for it ("left" is s180,
    "front" s270); a top cell is refused - top-down is a later milestone."""
    key = str(direction or "").strip().lower()
    if not key:
        raise ValueError(f"a direction is needed; one of {side_directions()} (or a view name)")
    by_view = {config.view_for_direction(d): d for d in side_directions()}
    key = by_view.get(key, key)
    _yaw, elevation = config.direction(key)       # refuses a name outside the table
    if elevation != config.ANIMATE_ELEVATION:
        raise ValueError(f"{key} is a top-down cell; the Animate tab is side-on for now "
                         f"(top-down is a later milestone). One of {side_directions()}")
    return key


def job_name(unit: str, action: str, direction: str) -> str:
    return f"{unit}_animate_{action_name(action)}_{direction_name(direction)}"


def phase_plan(action: str, unit: str | None = None) -> pick_stage.PhasePlan:
    return pick_stage.phase_plan(action_name(action), slug=unit)


# ---------------------------------------------------------------------------
# The frame row (pure)
# ---------------------------------------------------------------------------
def frame_row(frame_id: str, *, key: bool, phase: str | None, at: float) -> dict[str, Any]:
    return {"id": frame_id, "key": bool(key), "phase": phase if key else None,
            "at": round(float(at), 6), "prompt": None, "skeleton_frame": None,
            "candidates": [], "current": None, "verdict": None}


def default_layout(action: str, unit: str | None = None) -> list[dict[str, Any]]:
    """The phase template as a frame row: `budget` frames at the template's
    positions in the clip, the phase keys marked as keyframes."""
    plan = phase_plan(action, unit)
    positions = pick_stage.phase_positions(plan)
    keys = plan.keys()
    phases = [name for name in config.PHASE_NAMES if plan.frames.get(name)]
    return [frame_row(f"f{i + 1:02d}", key=i in keys,
                      phase=phases[keys.index(i)] if i in keys else None, at=positions[i])
            for i in range(plan.budget)]


def frame_label(frame: dict[str, Any], position: int) -> str:
    """"Wind-up" for a keyframe, "Frame 4" for an in-between (1-based)."""
    if frame.get("key") and frame.get("phase"):
        return config.phase_label(frame["phase"])
    return f"Frame {position + 1}"


def is_rendered(row: dict[str, Any]) -> bool:
    return row.get("seconds") is not None or row.get("source") == MIRROR_SOURCE


def frame_status(frame: dict[str, Any]) -> str:
    rows = frame.get("candidates") or []
    if any(r.get("source") == SOURCE and r.get("seconds") is None and not r.get("file_deleted")
           for r in rows):
        return "generating"
    current = frame.get("current")
    if current is None:
        return "empty"
    verdict = frame.get("verdict") or {}
    if verdict.get("index") == current and verdict.get("state") in ("approved", "rejected"):
        return str(verdict["state"])
    return "review"


def neighbour_keys(frames: Sequence[dict[str, Any]], position: int, *, loop: bool
                   ) -> tuple[int | None, int | None]:
    """The positions of the nearest keyframe before and after `position`
    (never the frame itself). A loop wraps round the ends; a one-shot has
    None where there is no keyframe on that side."""
    keys = [i for i, f in enumerate(frames) if f.get("key") and i != position]
    before = [i for i in keys if i < position]
    after = [i for i in keys if i > position]
    prev = before[-1] if before else (after[-1] if loop and after else None)
    nxt = after[0] if after else (before[0] if loop and before else None)
    return prev, nxt


def default_draw_from(frame: dict[str, Any]) -> str:
    return "view" if frame.get("key") else "keys"


def default_prompt(action: str, frames: Sequence[dict[str, Any]], position: int, *,
                   loop: bool, unit: str | None = None) -> str:
    frame = frames[position]
    if frame.get("key") and frame.get("phase"):
        return config.animate_phase_prompt(action, frame["phase"], unit)
    prev, nxt = neighbour_keys(frames, position, loop=loop)
    label = lambda i: (frames[i].get("phase") if i is not None else None)  # noqa: E731
    return config.animate_inbetween_prompt(action, label(prev), label(nxt))


def counts(frames: Sequence[dict[str, Any]]) -> dict[str, int]:
    out = {status: 0 for status in STATUSES}
    for frame in frames:
        out[frame_status(frame)] += 1
    out["total"] = len(frames)
    return out


# What a cell of the Animate tab's grid reads as, in pipeline order.
CELL_STATES = ("none", "keys", "keys_ok", "done")


def cell_summary(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One animation, as a cell of the Animate tab's grid: `state` is `none`
    (no frame has a picture), `keys` (some keyframe is not approved),
    `keys_ok` (every keyframe is approved - the in-betweens may be drawn) or
    `done` (every frame is approved). Pure."""
    keys = [f for f in frames if f.get("key")]
    keys_approved = sum(1 for f in keys if frame_status(f) == "approved")
    approved = sum(1 for f in frames if frame_status(f) == "approved")
    pictures = sum(1 for f in frames if f.get("current") is not None)
    if not pictures:
        state = "none"
    elif frames and approved == len(frames):
        state = "done"
    elif keys_approved == len(keys):
        state = "keys_ok"
    else:
        state = "keys"
    return {"state": state, "frames": len(frames), "approved": approved,
            "keys": len(keys), "keys_approved": keys_approved, "pictures": pictures}


def next_step(frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """What Generate all does now - the two-step gate. `step` is `keys`
    (some keyframe is unapproved), `inbetweens` (every keyframe is approved),
    or `done`; `targets` are the positions it would draw (empty or rejected
    frames of that kind); `waiting` those of that kind still to review."""
    keys = [i for i, f in enumerate(frames) if f.get("key")]
    unapproved = [i for i in keys if frame_status(frames[i]) != "approved"]
    kind = "keys" if unapproved else "inbetweens"
    pool = keys if unapproved else [i for i, f in enumerate(frames) if not f.get("key")]
    targets = [i for i in pool if frame_status(frames[i]) in ("empty", "rejected")]
    waiting = [i for i in pool if frame_status(frames[i]) in ("review", "generating")]
    if kind == "inbetweens" and not targets and not waiting and \
            all(frame_status(f) == "approved" for f in frames):
        kind = "done"
    return {"step": kind, "targets": targets, "waiting": waiting}


def batch_targets(frames: Sequence[dict[str, Any]], which: str) -> list[int]:
    """The frame positions one batch press draws."""
    if which == "all":
        return list(next_step(frames)["targets"])
    if which == "keys":
        return [i for i, f in enumerate(frames) if f.get("key")
                and frame_status(f) not in ("approved", "generating")]
    if which == "rejected":
        return [i for i, f in enumerate(frames) if frame_status(f) == "rejected"]
    if which == "inbetweens":
        return [i for i, f in enumerate(frames) if not f.get("key")
                and frame_status(f) in ("empty", "rejected")]
    raise ValueError(f"a batch is one of {list(BATCHES)}, not {which!r}")


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def empty_record(unit: str, action: str, direction: str) -> dict[str, Any]:
    name = action_name(action)
    direction = direction_name(direction)
    yaw, elevation = config.direction(direction)
    frames = default_layout(name, unit)
    return {
        "schema": SCHEMA, "unit": unit, "studio_action": name,
        "verb": config.STUDIO_ACTIONS[name]["verb"], "cell": direction,
        "view": config.view_for_direction(direction), "yaw": yaw, "elevation": elevation,
        "loop": bool(phase_plan(name, unit).loop),
        "created": _now(), "updated": _now(), "rig": None, "workflow": f"{WORKFLOW}.json",
        "comfyui": None, "models": [], "last_seed": None,
        "frames": frames, "next_id": len(frames) + 1, "removed": [],
        "mirror": None, "output": None, "output_error": None,
    }


def open_job(unit: str, action: str, direction: str, work_dir: Path | None = None, *,
             create: bool = True) -> tuple[Job, dict[str, Any]]:
    """The animate job and its record; a new one is written when `create`."""
    job = Job.open(job_name(unit, action, direction), work_dir)
    if job.manifest_path.exists():
        return job, job.read_manifest()
    record = empty_record(unit, action, direction)
    if create:
        (job.root / ATTEMPTS_DIR).mkdir(parents=True, exist_ok=True)
        job.write_manifest(record)
    return job, record


def save(job: Job, record: dict[str, Any]) -> None:
    record["updated"] = _now()
    job.write_manifest(record)


def find_frame(record: dict[str, Any], frame: str | int) -> int:
    """A frame's position from its id (`f03`), its 1-based number (`3`), or a
    keyframe's phase (`wind-up`)."""
    frames = record["frames"]
    text = str(frame).strip().lower()
    for i, row in enumerate(frames):
        if row["id"] == text:
            return i
    if text.isdigit():
        number = int(text)
        if 1 <= number <= len(frames):
            return number - 1
        raise ValueError(f"no frame {number}; this animation has {len(frames)}")
    hits = [i for i, row in enumerate(frames) if row.get("key") and row.get("phase") == text]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} keyframes are {text}; name one by number "
                         f"({[h + 1 for h in hits]})")
    raise ValueError(f"no frame {frame!r}: a frame is its number (1..{len(frames)}), its id "
                     f"(f01..) or a keyframe's phase")


class FrameAttempts(concept_stage.AttemptStore):
    """One frame's attempts: rows under the frame in the animate job's
    job.json, files under attempts/. The Concept tab's store shape, so hide /
    delete-file / the editor's Save / inpaint run on it unchanged."""

    RENDERED_SOURCES = (*concept_stage.GPU_SOURCES, SOURCE)
    label = "animate"

    def __init__(self, unit: str, action: str, direction: str, frame: str | int,
                 work_dir: Path | None = None, *, record: dict[str, Any] | None = None,
                 job: Job | None = None) -> None:
        self.slug = unit
        self.action = action_name(action)
        self.direction = direction_name(direction)
        self.work_dir = work_dir
        self.cli = f"anim animate --unit {unit} --action {self.action} --dir {self.direction}"
        if record is None or job is None:
            job, loaded = open_job(unit, self.action, self.direction, work_dir)
            record = loaded if record is None else record
        self.job, self.record = job, record
        self.position = find_frame(record, frame)
        self.target = record["frames"][self.position]["id"]
        self.dir = self.job.root / ATTEMPTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def frame(self) -> dict[str, Any]:
        return self.record["frames"][self.position]

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.frame["candidates"]

    def _set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.frame["candidates"] = rows
        if self.frame.get("current") is not None and \
                not any(r["index"] == self.frame["current"] for r in rows):
            self.frame["current"] = None

    def file_for(self, index: int) -> str:
        return f"{self.target}_{index:02d}.png"

    def accepted_index(self) -> int | None:
        verdict = self.frame.get("verdict") or {}
        return int(verdict["index"]) if verdict.get("state") == "approved" else None

    def save(self) -> None:
        save(self.job, self.record)


def parse_ref(ref: str) -> tuple[str, int]:
    """`f03:02` -> ("f03", 2)."""
    frame, sep, number = str(ref or "").strip().lower().rpartition(":")
    if not sep or not number.strip().isdigit() or not frame:
        raise ValueError(f"an attempt is named <frame>:NN (f03:02, or 3:02), not {ref!r}")
    return frame, int(number)


def store_for(unit: str, action: str, direction: str, ref: str,
              work_dir: Path | None = None) -> tuple[FrameAttempts, int]:
    frame, index = parse_ref(ref)
    return FrameAttempts(unit, action, direction, frame, work_dir), index


# ---------------------------------------------------------------------------
# The rig clip and the skeleton frames
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _load_rig(path: str, mtime: float) -> rigkit.Rig:
    """One parse per GLB per process: the API scrubs the same clip a frame at
    a time. `mtime` is in the key so a replaced file is read again."""
    return rigkit.load_rig(Path(path))


def load_rig(glb: Path) -> rigkit.Rig:
    glb = Path(glb)
    return _load_rig(str(glb), glb.stat().st_mtime if glb.exists() else 0.0)


def clip_for(rig_name: str, action: str) -> str | None:
    """The rig's clip for one studio action, or None when that rig ships no
    whole clip for it (the UAL's jump is three clips and no whole one)."""
    return config.STUDIO_ACTIONS[action]["clips"].get(rig_name)


def rig_for(rig_name: str = config.ANIMATE_RIG,
            glb: Path | None = None) -> tuple[str, Path]:
    """(rig name, GLB path) - the bake-off's rig table, or the override.
    Imported inside the call: the bake-off stage pulls in the whole Wan chain."""
    from anim.stages import bakeoff as bakeoff_stage

    if rig_name not in bakeoff_stage.RIGS:
        raise ValueError(f"unknown rig {rig_name!r}; one of {sorted(bakeoff_stage.RIGS)}")
    return rig_name, Path(glb) if glb is not None else Path(bakeoff_stage.RIGS[rig_name]["glb"])


def rig_clip(action: str, *, glb: Path | None = None) -> dict[str, Any]:
    """The rig clip for this action: `{rig, glb, clip, duration}`.
    Refused when the rig ships no whole clip for it (the UAL's jump)."""
    name = action_name(action)
    rig_name, path = rig_for(config.ANIMATE_RIG, glb)
    clip = clip_for(rig_name, name)
    if clip is None:
        raise ValueError(f"the {rig_name} rig ships no whole {name} clip, so there is no "
                         f"skeleton to pose; draw this action's frames without one")
    return {"rig": rig_name, "glb": path, "clip": clip,
            "duration": round(float(load_rig(path).clip(clip).duration), 4)}


def scrub_times(duration: float, fps: int | None = None) -> list[float]:
    """The instants the slider steps through: the whole clip, both ends."""
    return [round(float(t), 6) for t in
            rigkit.sample_times(float(duration), int(fps or config.ANIMATE_SCRUB_FPS))]


def nearest_frame(times: Sequence[float], t: float) -> int:
    return min(range(len(times)), key=lambda i: (abs(times[i] - t), i))


def default_skeleton_frame(frame: dict[str, Any], duration: float,
                           fps: int | None = None) -> int:
    """The scrub frame the slider starts on: the frame's template instant."""
    return nearest_frame(scrub_times(duration, fps), float(frame.get("at") or 0.0) * duration)


def skeleton_dir(job: Job, cell: tuple[int, int], fps: int) -> Path:
    return job.root / SKELETONS_DIR / f"{cell[0]}x{cell[1]}_{fps}fps"


def skeletons(unit: str, action: str, direction: str, *,
              cell: tuple[int, int] | None = None, fps: int | None = None,
              glb: Path | None = None, work_dir: Path | None = None) -> dict[str, Any]:
    """The clip posed at the direction's yaw, one skeleton PNG per scrub frame,
    rendered on the CPU once and cached: `{dir, files, times, fps, cell, clip,
    rig}`. The cache is rebuilt when what it was drawn from changes."""
    direction = direction_name(direction)
    fps = int(fps or config.ANIMATE_SCRUB_FPS)
    cell = tuple(cell or _reference_size(unit, direction, work_dir))
    rig = rig_clip(action, glb=glb)
    props = tuple(config.props_for_unit(unit))
    yaw, elevation = config.direction(direction)
    job = Job.open(job_name(unit, action, direction), work_dir)
    out = skeleton_dir(job, cell, fps)
    times = scrub_times(rig["duration"], fps)
    glb_path = Path(rig["glb"])
    meta = {"glb": glb_path.name, "glb_bytes": glb_path.stat().st_size, "clip": rig["clip"],
            "props": list(props), "yaw": yaw, "elevation": elevation, "cell": list(cell),
            "fps": fps, "frames": len(times)}
    files = [out / f"skel_{i + 1:04d}.png" for i in range(len(times))]
    meta_path = out / "meta.json"
    if not (meta_path.exists() and read_json(meta_path) == meta and all(p.exists() for p in files)):
        loaded = load_rig(glb_path)
        frame = rigkit.reference_frame(loaded, rigkit.detect_map(loaded.names), props=props,
                                       yaw=yaw, elevation=elevation)
        payload = rigkit.clip_keypoints(loaded, rig["clip"], fps=fps, cell=cell, frame=frame,
                                        times=times)
        out.mkdir(parents=True, exist_ok=True)
        pose_stage.render_skeletons(payload, out, cell)
        write_json_in_place(meta_path, meta)
    return {"dir": out, "files": files, "times": times, "fps": fps, "cell": list(cell),
            "clip": rig["clip"], "rig": rig["rig"], "duration": rig["duration"]}


def skeleton_file(unit: str, action: str, direction: str, frame: int, **kw: Any) -> Path:
    """One scrub frame's skeleton PNG (the whole clip is rendered on first ask)."""
    rendered = skeletons(unit, action, direction, **kw)
    if not 0 <= int(frame) < len(rendered["files"]):
        raise ValueError(f"frame {frame} is outside the clip: 0..{len(rendered['files']) - 1} "
                         f"({rendered['clip']} at {rendered['fps']} fps)")
    return rendered["files"][int(frame)]


# ---------------------------------------------------------------------------
# The view: each direction is drawn from its OWN accepted view
# ---------------------------------------------------------------------------
def reference(unit: str, direction: str, work_dir: Path | None = None) -> dict[str, Any]:
    """The direction's own accepted view: `{view, path, sha256}` - the base
    itself at s270. Refused with the reason when it is not accepted: a
    direction is never drawn from another direction's view."""
    direction = direction_name(direction)
    view = config.view_for_direction(direction)
    blocked = curation.views_blocked(unit, work_dir)
    if view == config.FRONT_VIEW:
        if blocked:
            raise ValueError(blocked)
        _job, base_path, _manifest = concept_stage.base_render(unit, work_dir)
        return {"view": view, "path": base_path, "sha256": sha256_file(base_path)}
    if blocked:
        raise ValueError(blocked)
    entry = curation.base_views(unit, work_dir).get(view)
    if entry is None:
        raise ValueError(f"{unit} has no {view} view yet, so {direction} has nothing to be "
                         f"drawn from - draw it on the Views tab (`anim concept --unit "
                         f"{unit} --views {view}`) and accept one")
    if entry.get("state") != "accepted":
        raise ValueError(f"{unit}'s {view} view is {entry.get('state')}, not accepted - accept "
                         f"it on the Views tab before drawing {direction}")
    path = Path(entry["path"])
    return {"view": view, "path": path, "sha256": sha256_file(path)}


def _reference_size(unit: str, direction: str, work_dir: Path | None) -> tuple[int, int]:
    with Image.open(reference(unit, direction, work_dir)["path"]) as img:
        return img.size


# ---------------------------------------------------------------------------
# Draw from: view / keys / both
# ---------------------------------------------------------------------------
def _approved_image(record: dict[str, Any], job: Job, position: int, role: str
                    ) -> dict[str, Any]:
    frame = record["frames"][position]
    label = frame_label(frame, position)
    if frame_status(frame) != "approved":
        raise ValueError(f"{label} (frame {position + 1}) is not approved yet - approve it "
                         f"first, or draw this frame from the view")
    index = int(frame["verdict"]["index"])
    row = next((r for r in frame["candidates"] if int(r["index"]) == index), None)
    path = job.root / ATTEMPTS_DIR / str((row or {}).get("file") or "")
    if row is None or row.get("file_deleted") or not path.is_file():
        raise FileNotFoundError(f"{label}'s approved attempt is missing on disk")
    return {"role": role, "ref": f"{frame['id']}:{index:02d}", "label": label,
            "path": str(path), "sha256": sha256_file(path)}


def resolve_draw_from(record: dict[str, Any], job: Job, position: int, draw_from: str,
                      view_image: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The images an attempt is drawn from, in image order, before the
    skeleton: `[{role, ref, path, sha256}]`. The keys are the nearest
    keyframes either side, and both must be approved."""
    draw_from = str(draw_from or "").strip().lower()
    if draw_from not in config.ANIMATE_DRAW_FROM:
        raise ValueError(f"--draw-from is one of {list(config.ANIMATE_DRAW_FROM)}, "
                         f"not {draw_from!r}")
    images: list[dict[str, Any]] = []
    if draw_from in ("view", "both"):
        if view_image is None:
            raise ValueError("no view to draw from")
        images.append({"role": "view", "ref": view_image["view"],
                       "path": str(view_image["path"]), "sha256": view_image["sha256"]})
    if draw_from in ("keys", "both"):
        frames = record["frames"]
        label = frame_label(frames[position], position)
        prev, nxt = neighbour_keys(frames, position, loop=bool(record.get("loop")))
        if prev is None or nxt is None:
            side = "before" if prev is None else "after"
            raise ValueError(f"{label} has no keyframe {side} it, so there are no two keys "
                             f"to draw it from - mark one, or draw this frame from the view")
        images.append(_approved_image(record, job, prev, "before"))
        if nxt != prev:
            images.append(_approved_image(record, job, nxt, "after"))
    return images


# ---------------------------------------------------------------------------
# Plan and render
# ---------------------------------------------------------------------------
def plan(unit: str, action: str, direction: str, frame: str | int, *,
         prompt: str | None = None, draw_from: str | None = None,
         skeleton: bool | None = None, skeleton_frame: int | None = None,
         count: int | None = None, seed: int | None = None, model: str | None = None,
         loras: Sequence[str | dict[str, Any]] | None = None,
         registry_path: Path | None = None, lock_path: Path | None = None,
         glb: Path | None = None, work_dir: Path | None = None
         ) -> tuple[FrameAttempts, list[dict[str, Any]]]:
    """`count` attempts on one frame, recorded but not rendered: what each is
    drawn from, the skeleton frame, the prompt, the seed, the model.
    `skeleton` None means "yes when the rig ships a clip for the action"."""
    name = action_name(action)
    direction = direction_name(direction)
    store = FrameAttempts(unit, name, direction, frame, work_dir)
    store.drop_unrendered()                       # what a killed run left planned
    record, position = store.record, store.position
    count = 1 if count is None else int(count)
    if count < 1:
        raise ValueError("--count must be at least 1")
    how = str(draw_from or default_draw_from(store.frame)).strip().lower()
    view_image = reference(unit, direction, work_dir) if how in ("view", "both") else None
    images = resolve_draw_from(record, store.job, position, how, view_image)
    pose = ((prompt or "").strip() or store.frame.get("prompt")
            or default_prompt(name, record["frames"], position, loop=bool(record.get("loop")),
                              unit=unit))
    skeleton_block: dict[str, Any] | None = None
    want_skeleton = skeleton_frame is not None or skeleton is True
    if skeleton is None and skeleton_frame is None:
        try:
            rig_clip(name, glb=glb)
            want_skeleton = True
        except ValueError:
            want_skeleton = False
    if want_skeleton:
        with Image.open(images[0]["path"]) as img:
            cell = img.size
        rendered = skeletons(unit, name, direction, cell=cell, glb=glb, work_dir=work_dir)
        default = default_skeleton_frame(store.frame, rendered["duration"], rendered["fps"])
        chosen = (skeleton_frame if skeleton_frame is not None
                  else store.frame.get("skeleton_frame"))
        chosen = default if chosen is None else int(chosen)
        path = skeleton_file(unit, name, direction, chosen, cell=cell, glb=glb,
                             work_dir=work_dir)
        skeleton_block = {"frame": chosen, "default_frame": default,
                          "frames": len(rendered["files"]), "fps": rendered["fps"],
                          "time": rendered["times"][chosen], "clip": rendered["clip"],
                          "rig": rendered["rig"], "file": str(path)}
        images = [*images, {"role": "skeleton", "ref": f"{rendered['clip']}@{chosen}",
                            "path": str(path), "sha256": sha256_file(path)}]
    if len(images) > config.ANIMATE_MAX_IMAGES:
        raise ValueError(f"{len(images)} images ({', '.join(i['role'] for i in images)}); "
                         f"Qwen-Image-Edit takes {config.ANIMATE_MAX_IMAGES} - untick the "
                         f"skeleton, or draw from the keys or the view alone")
    parts, avoid = concept_stage.base_phrases(unit, work_dir)
    text = config.animate_prompt(pose, [i["role"] for i in images], parts, avoid)
    seed = concept_stage.random_seed() if seed is None else int(seed)
    generation = concept_stage.resolve_generation(
        FAMILY, model=model, loras=loras, defaults=store.defaults(),
        registry_path=registry_path, lock_path=lock_path)
    rows = [store.append(concept_stage.attempt_row(
        SOURCE, prompt=text, seed=seed + i, generation=generation, workflow=WORKFLOW,
        pose_prompt=" ".join(pose.split()), draw_from=how,
        images=[{k: v for k, v in img.items() if k != "label"} for img in images],
        skeleton=skeleton_block))
        for i in range(count)]
    store.frame["prompt"] = " ".join(pose.split())
    if skeleton_frame is not None:
        store.frame["skeleton_frame"] = int(skeleton_frame)
    record["last_seed"] = seed
    if skeleton_block:
        record["rig"] = {"rig": skeleton_block["rig"], "clip": skeleton_block["clip"],
                         "fps": skeleton_block["fps"], "frames": skeleton_block["frames"]}
    store.save()
    return store, rows


def build_workflow(prompt: str, *, seed: int, images: Sequence[str], job: str,
                   model: str | None = None, loras: Sequence[dict[str, Any]] | None = None,
                   workflow_dir: Path | None = None) -> dict[str, Any]:
    """view_pass.json with up to three images: image1 is the first (and the
    latent, so the frame is its size), image2 / image3 are new LoadImage
    nodes wired into both text encoders."""
    if not 1 <= len(images) <= config.ANIMATE_MAX_IMAGES:
        raise ValueError(f"1..{config.ANIMATE_MAX_IMAGES} images, not {len(images)}")
    workflow = concept_stage.build_view_workflow(
        prompt, seed=seed, base_name=images[0], slug=job, workflow_dir=workflow_dir,
        model=model, loras=loras, prefix=f"{SOURCE}/{WORKFLOW}", job=job)
    encoders = [node_id for node_id, node in workflow.items()
                if node.get("class_type") == "TextEncodeQwenImageEditPlus"]
    for number, name in enumerate(images[1:], start=2):
        node_id = f"an_image{number}"
        workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": name},
                             "_meta": {"title": f"image{number} (animate)"}}
        for encoder in encoders:
            workflow[encoder]["inputs"][f"image{number}"] = [node_id, 0]
    return workflow


def render_pending(store: FrameAttempts, *, client: comfy.ComfyClient | None = None,
                   workflow_dir: Path | None = None, lock_path: Path | None = None,
                   timeout_s: float = 900.0, log: Callable[[str], None] = lambda _: None
                   ) -> list[dict[str, Any]]:
    """Render the frame's attempts still waiting for the GPU, one /prompt
    each. The last one rendered becomes the frame's current attempt."""
    pending = [r for r in store.rows if r.get("seconds") is None and r.get("source") == SOURCE]
    done: list[dict[str, Any]] = []
    if not pending:
        return done
    client, stats = concept_stage._client(client)
    try:
        concept_stage.check_installed(
            client, FAMILY, model=next((r["model"] for r in pending if r.get("model")), None),
            loras=[lora for r in pending for lora in (r.get("loras") or [])])
        uploaded: dict[str, str] = {}
        for row in pending:
            names = []
            for image in row["images"]:
                key = f"{image['role']}_{image['sha256'][:12]}"
                if key not in uploaded:
                    uploaded[key] = client.upload_image(
                        Path(image["path"]), name=f"anim_{store.job.name}_{key}.png")
                names.append(uploaded[key])
            workflow = build_workflow(row["prompt"], seed=row["seed"], images=names,
                                      job=store.job.name, model=row.get("model"),
                                      loras=row.get("loras"), workflow_dir=workflow_dir)
            log(f"animate: {frame_label(store.frame, store.position)} ({store.target}:"
                f"{row['index']:02d}) from {row['draw_from']}"
                f"{' + skeleton ' + str(row['skeleton']['frame']) if row.get('skeleton') else ''}"
                f" (seed {row['seed']}) ...")
            prompt_id, _entry, png, elapsed = concept_stage._run_one(client, workflow,
                                                                     timeout_s=timeout_s)
            path = write_bytes_in_place(store.path(row), png)
            log(f"  -> {path.name}  {elapsed:.1f} s")
            row |= {"prompt_id": prompt_id, "seconds": round(elapsed, 1),
                    "sha256": sha256_bytes(png),
                    "workflow_sha256": comfy.workflow_sha256(workflow),
                    "models": [m.as_sidecar_entry()
                               for m in models_lock.for_workflow(workflow, lock_path)],
                    "embedded_prompt": render_stage.prompt_chunk(png) is not None}
            store.frame["current"] = int(row["index"])
            done.append(row)
            store.save()
    except BaseException:
        store.drop_unrendered()
        store.save()
        raise
    seen = list(store.record.get("models") or [])
    for row in done:
        for m in row["models"]:
            if m not in seen:
                seen.append(m)
    store.record |= {"models": seen, "comfyui": render_stage.build_provenance(
        client.base_url, stats, store.record.get("last_seed") or 0, [])["comfyui"]}
    store.save()
    return done


def generate(unit: str, action: str, direction: str, frame: str | int, *,
             client: comfy.ComfyClient | None = None, workflow_dir: Path | None = None,
             lock_path: Path | None = None, timeout_s: float = 900.0,
             log: Callable[[str], None] = lambda _: None, work_dir: Path | None = None,
             **options: Any) -> list[dict[str, Any]]:
    """Plan and render attempts on one frame; the rows rendered."""
    store, rows = plan(unit, action, direction, frame, lock_path=lock_path, work_dir=work_dir,
                       **options)
    render_pending(store, client=client, workflow_dir=workflow_dir, lock_path=lock_path,
                   timeout_s=timeout_s, log=log)
    return [store.get(int(r["index"])) for r in rows]


def generate_batch(unit: str, action: str, direction: str, which: str, *,
                   client: comfy.ComfyClient | None = None, workflow_dir: Path | None = None,
                   lock_path: Path | None = None, timeout_s: float = 900.0,
                   log: Callable[[str], None] = lambda _: None, work_dir: Path | None = None,
                   progress: bool = True, **options: Any) -> dict[str, Any]:
    """One press over several frames, each with its own defaults (prompt,
    draw from, skeleton at its template instant). A frame that may not be
    drawn - a neighbour keyframe not approved - is BLOCKED and the rest go
    on; one that raises is FAILED and the rest go on. `[n/m]` after each,
    unless `progress` is off (a sweep counts cells, not frames)."""
    job, record = open_job(unit, action, direction, work_dir)
    targets = batch_targets(record["frames"], which)
    ids = [record["frames"][i]["id"] for i in targets]
    tally = {"done": 0, "failed": 0, "blocked": 0, "of": len(ids)}
    results: list[dict[str, Any]] = []
    if not ids:
        step = next_step(record["frames"])
        log(f"animate: nothing to draw for --batch {which}"
            + (f" - {len(step['waiting'])} frame(s) wait for your review" if step["waiting"]
               else ""))
    for n, frame_id in enumerate(ids, start=1):
        try:
            rows = generate(unit, action, direction, frame_id, client=client,
                            workflow_dir=workflow_dir, lock_path=lock_path, timeout_s=timeout_s,
                            log=log, work_dir=work_dir, **options)
            tally["done"] += 1
            results.append({"frame": frame_id, "status": "done",
                            "attempts": [r["index"] for r in rows]})
        except (ValueError, FileNotFoundError) as exc:
            tally["blocked"] += 1
            results.append({"frame": frame_id, "status": "blocked", "reason": str(exc)})
            log(f"animate: {frame_id} blocked - {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad frame must not cost the rest
            tally["failed"] += 1
            results.append({"frame": frame_id, "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}"})
            log(f"animate: {frame_id} failed - {type(exc).__name__}: {exc}")
        if progress:
            print(f"[{n}/{len(ids)}] {frame_id} {results[-1]['status']}", flush=True)
        else:
            log(f"animate: {frame_id} {results[-1]['status']}")
    return {"which": which, "frames": results, "cells": tally}


# ---------------------------------------------------------------------------
# The sweep: one batch over several animations (the keyframe sweep feature)
# ---------------------------------------------------------------------------
SWEEP_BATCHES = ("keys", "inbetweens")


def parse_cell(spec: str) -> tuple[str, str]:
    """`walk:s90` (or `walk:left`) -> ("walk", "s90")."""
    action, sep, direction = str(spec).partition(":")
    if not sep:
        raise ValueError(f"a sweep cell is ACTION:DIRECTION (walk:s90), not {spec!r}")
    return action_name(action), direction_name(direction)


def sweep_blocked(unit: str, action: str, direction: str, which: str,
                  work_dir: Path | None = None) -> str | None:
    """Why this cell may not join a sweep, or None. Every cell needs its
    direction's accepted view; an in-between sweep also needs every keyframe
    of the cell approved - the keyframe gate, which a sweep never skips."""
    try:
        reference(unit, direction, work_dir)
    except (ValueError, FileNotFoundError) as exc:
        return str(exc)
    if which == "inbetweens":
        _job, record = open_job(unit, action, direction, work_dir, create=False)
        summary = cell_summary(record["frames"])
        if summary["keys_approved"] < summary["keys"]:
            return (f"{summary['keys'] - summary['keys_approved']} keyframe(s) of {action} "
                    f"{direction} not approved yet - approve the keys before the in-betweens")
    return None


def sweep(unit: str, cells: Sequence[tuple[str, str]], which: str, *,
          client: comfy.ComfyClient | None = None, workflow_dir: Path | None = None,
          lock_path: Path | None = None, timeout_s: float = 900.0,
          log: Callable[[str], None] = lambda _: None, work_dir: Path | None = None,
          max_cells: int | None = None, **options: Any) -> dict[str, Any]:
    """The keyframes (or the in-betweens) of several animations as ONE run:
    `generate_batch(which)` on each cell in turn. A blocked cell is reported
    and skipped, never drawn from a fallback; a cell whose frames raised is
    FAILED and the rest go on. `[n/m]` after each CELL. Refused before any
    GPU call when `which` is not a sweep batch or there are too many cells."""
    if which not in SWEEP_BATCHES:
        raise ValueError(f"a sweep draws {' or '.join(SWEEP_BATCHES)}, not {which!r}")
    cap = config.ANIMATE_SWEEP_MAX_CELLS if max_cells is None else max_cells
    unique = list(dict.fromkeys((action_name(a), direction_name(d)) for a, d in cells))
    if not unique:
        raise ValueError("a sweep needs at least one cell (ACTION:DIRECTION)")
    if cap and len(unique) > cap:
        raise ValueError(f"{len(unique)} cells; one sweep takes at most {cap} "
                         f"(ANIM_SWEEP_MAX_CELLS) - tick fewer")
    tally = {"done": 0, "failed": 0, "blocked": 0, "of": len(unique)}
    results: list[dict[str, Any]] = []
    for n, (action, direction) in enumerate(unique, start=1):
        cell = f"{action}:{direction}"
        reason = sweep_blocked(unit, action, direction, which, work_dir)
        if reason is not None:
            tally["blocked"] += 1
            results.append({"cell": cell, "status": "blocked", "reason": reason})
            log(f"animate: {cell} blocked - {reason}")
        else:
            log(f"animate: {cell} - {which}")
            batch = generate_batch(unit, action, direction, which, client=client,
                                   workflow_dir=workflow_dir, lock_path=lock_path,
                                   timeout_s=timeout_s, log=log, work_dir=work_dir,
                                   progress=False, **options)
            frames = batch["cells"]
            status = "failed" if frames["failed"] else "done"
            tally[status] += 1
            results.append({"cell": cell, "status": status, "frames": frames})
        print(f"[{n}/{len(unique)}] {cell} {results[-1]['status']}", flush=True)
    return {"which": which, "cells": results, "tally": tally}


# ---------------------------------------------------------------------------
# The row: add, remove, mark keyframes
# ---------------------------------------------------------------------------
def add_frame(unit: str, action: str, direction: str, after: int, *,
              work_dir: Path | None = None) -> dict[str, Any]:
    """A new in-between after frame `after` (1-based; 0 puts it first),
    sampling the clip halfway between its neighbours."""
    job, record = open_job(unit, action, direction, work_dir)
    frames = record["frames"]
    if len(frames) >= config.ANIMATE_MAX_FRAMES:
        raise ValueError(f"an animation is at most {config.ANIMATE_MAX_FRAMES} frames")
    after = int(after)
    if not 0 <= after <= len(frames):
        raise ValueError(f"--add-frame takes 0..{len(frames)} (the frame it goes after)")
    left = frames[after - 1]["at"] if after >= 1 else 0.0
    right = frames[after]["at"] if after < len(frames) else 1.0
    frame_id = f"f{int(record['next_id']):02d}"
    record["next_id"] = int(record["next_id"]) + 1
    row = frame_row(frame_id, key=False, phase=None, at=(float(left) + float(right)) / 2)
    frames.insert(after, row)
    save(job, record)
    return row


def remove_frame(unit: str, action: str, direction: str, frame: str | int, *,
                 work_dir: Path | None = None) -> dict[str, Any]:
    """Take a frame out of the row. Its attempts are the owner's work, so the
    row moves to the record's `removed` list, files and all."""
    job, record = open_job(unit, action, direction, work_dir)
    position = find_frame(record, frame)
    if len(record["frames"]) <= 1:
        raise ValueError("an animation needs at least one frame")
    row = record["frames"].pop(position)
    row["removed"] = _now()
    row["position"] = position + 1
    record.setdefault("removed", []).append(row)
    save(job, record)
    return row


def set_key(unit: str, action: str, direction: str, frame: str | int, *,
            key: bool, phase: str | None = None, work_dir: Path | None = None
            ) -> dict[str, Any]:
    """Mark a frame as a keyframe with a phase label, or un-mark it."""
    job, record = open_job(unit, action, direction, work_dir)
    row = record["frames"][find_frame(record, frame)]
    if key:
        phase = str(phase or "").strip().lower()
        if phase not in config.PHASE_NAMES:
            raise ValueError(f"a keyframe's phase is one of {list(config.PHASE_NAMES)}, "
                             f"not {phase!r}")
        row["key"], row["phase"] = True, phase
    else:
        row["key"], row["phase"] = False, None
    save(job, record)
    return row


# ---------------------------------------------------------------------------
# Verdicts: select, approve, reject
# ---------------------------------------------------------------------------
def _judgeable(store: FrameAttempts, index: int) -> dict[str, Any]:
    row = store.get(int(index))
    if row.get("file_deleted"):
        raise ValueError(f"{store.ref(index)}'s file was deleted")
    if not is_rendered(row):
        raise ValueError(f"{store.ref(index)} has not been rendered yet")
    if not store.path(row).exists():
        raise FileNotFoundError(f"{store.path(row)} is missing")
    return row


def select(unit: str, action: str, direction: str, frame: str | int, index: int, *,
           work_dir: Path | None = None) -> dict[str, Any]:
    """Make an attempt the frame's current one (all attempts are kept)."""
    store = FrameAttempts(unit, action, direction, frame, work_dir)
    row = _judgeable(store, index)
    row.pop("hidden", None)
    store.frame["current"] = int(index)
    store.save()
    return store.frame


def judge(unit: str, action: str, direction: str, frame: str | int, state: str, *,
          index: int | None = None, work_dir: Path | None = None,
          write_output: bool = True, palette: Path | None = None) -> dict[str, Any]:
    """Approve or reject the frame's current attempt (or `index`, which then
    becomes current). Always a person's act. When the approval completes the
    row, the finished frames are written (`output`); a refusal there is kept
    on the record for the tab to show, never raised into the approval."""
    if state not in ("approved", "rejected"):
        raise ValueError(f"a verdict is approved or rejected, not {state!r}")
    store = FrameAttempts(unit, action, direction, frame, work_dir)
    index = store.frame.get("current") if index is None else int(index)
    if index is None:
        raise ValueError(f"{frame_label(store.frame, store.position)} has no attempt to "
                         f"{'approve' if state == 'approved' else 'reject'}")
    row = _judgeable(store, index)
    row.pop("hidden", None)
    store.frame["current"] = int(index)
    store.frame["verdict"] = {"state": state, "index": int(index),
                              "sha256": sha256_file(store.path(row)), "at": _now()}
    store.save()
    if write_output and state == "approved":
        _maybe_output(unit, action, direction, work_dir=work_dir, palette=palette)
    return store.frame


def approve_reviewed(unit: str, action: str, direction: str, *,
                     work_dir: Path | None = None, palette: Path | None = None) -> list[str]:
    """Every frame to review -> approved; the ids approved."""
    job, record = open_job(unit, action, direction, work_dir)
    ids = [f["id"] for f in record["frames"] if frame_status(f) == "review"]
    for frame_id in ids:
        judge(unit, action, direction, frame_id, "approved", work_dir=work_dir,
              write_output=False)
    if ids:
        _maybe_output(unit, action, direction, work_dir=work_dir, palette=palette)
    return ids


def _clear_verdict_if(store: FrameAttempts, index: int) -> None:
    """An edit of the judged attempt clears the verdict: the owner judged
    other pixels."""
    verdict = store.frame.get("verdict") or {}
    if verdict.get("index") == int(index):
        store.frame["verdict"] = None
    store.frame["current"] = int(index)


def replace(unit: str, action: str, direction: str, ref: str, data: bytes, *,
            work_dir: Path | None = None) -> dict[str, Any]:
    """The editor's Save over one attempt; back to review if it was judged."""
    store, index = store_for(unit, action, direction, ref, work_dir)
    row = concept_stage.replace_in(store, index, data)
    _clear_verdict_if(store, index)
    store.save()
    return row


def stash_canvas(unit: str, action: str, direction: str, ref: str, data: bytes, *,
                 work_dir: Path | None = None) -> Path:
    store, index = store_for(unit, action, direction, ref, work_dir)
    return concept_stage.stash_canvas_in(store, index, data)


def inpaint(unit: str, action: str, direction: str, ref: str, canvas: bytes | Path, *,
            work_dir: Path | None = None, **options: Any) -> dict[str, Any]:
    """The retouch feature's masked redraw on a frame's attempt (Z-Image,
    gpu/workflows/inpaint_pass.json); back to review if it was judged."""
    store, index = store_for(unit, action, direction, ref, work_dir)
    description = (concept_stage._open_job(unit, work_dir).read_manifest()
                   .get("description") or "character")
    pose = store.get(index).get("pose_prompt") or store.frame.get("prompt") or ""
    text = config.concept_prompt(f"{description.strip().rstrip('.')}, {pose}".strip(", "))
    row = concept_stage.inpaint_in(store, index, canvas, default_prompt=text, **options)
    _clear_verdict_if(store, index)
    store.save()
    return row


def hide(unit: str, action: str, direction: str, ref: str, *, hidden: bool = True,
         work_dir: Path | None = None) -> dict[str, Any]:
    store, index = store_for(unit, action, direction, ref, work_dir)
    row = store.set_hidden(index, hidden)
    if hidden and store.frame.get("current") == index:
        store.frame["current"] = None
        verdict = store.frame.get("verdict") or {}
        if verdict.get("index") == index:
            store.frame["verdict"] = None
    store.save()
    return row


def delete_file(unit: str, action: str, direction: str, ref: str, *,
                work_dir: Path | None = None) -> dict[str, Any]:
    store, index = store_for(unit, action, direction, ref, work_dir)
    row = store.delete_file(index)
    store.save()
    return row


# ---------------------------------------------------------------------------
# Mirror: the partner direction's frames, flipped, back to review
# ---------------------------------------------------------------------------
def handedness_warning(unit: str, work_dir: Path | None = None) -> str:
    gear = concept_stage.handed_gear(unit, work_dir)
    held = (f"{unit} carries {', '.join(gear)}, so it comes back on the other side"
            if gear else "anything held in one hand comes back in the other")
    return (f"the flip swaps handedness: {held}. Every mirrored frame is to review, not "
            f"approved - judge each one, and draw a frame instead where the hand matters.")


def mirror(unit: str, action: str, direction: str, *, work_dir: Path | None = None,
           log: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """Flip every frame of the partner direction's animation into this one.
    No GPU. The row follows the partner's (frames, keyframes, phases); each
    frame whose partner has a picture - the approved attempt, else the
    current one - gets the flip as a new attempt, current and TO REVIEW."""
    name = action_name(action)
    direction = direction_name(direction)
    partner = config.mirror_direction(direction)
    if partner is None:
        raise ValueError(f"{direction} has no mirror partner - it looks straight down the "
                         f"symmetry plane. Mirrorable: s0<->s180 (right/left), s45<->s135 "
                         f"(back_right/back_left), s225<->s315 (front_left/front_right)")
    pjob = Job.open(job_name(unit, name, partner), work_dir)
    if not pjob.manifest_path.exists():
        raise ValueError(f"{unit}'s {name} at {partner} has no frames yet - nothing to flip")
    precord = pjob.read_manifest()
    sources: list[tuple[int, dict[str, Any], Path] | None] = []
    for frame in precord["frames"]:
        verdict = frame.get("verdict") or {}
        index = verdict.get("index") if verdict.get("state") == "approved" else frame.get("current")
        row = next((r for r in frame.get("candidates") or [] if r["index"] == index), None)
        path = pjob.root / ATTEMPTS_DIR / str((row or {}).get("file") or "")
        ok = row is not None and not row.get("file_deleted") and path.is_file()
        sources.append((int(index), frame, path) if ok else None)
    if not any(sources):
        raise ValueError(f"{unit}'s {name} at {partner} has no frame with a picture yet - "
                         f"nothing to flip")
    job, record = open_job(unit, name, direction, work_dir)
    if len(record["frames"]) != len(precord["frames"]):
        for frame in record["frames"]:
            if frame.get("candidates"):
                record.setdefault("removed", []).append({**frame, "removed": _now()})
        fresh = []
        for pframe in precord["frames"]:
            fresh.append(frame_row(f"f{int(record['next_id']):02d}", key=bool(pframe.get("key")),
                                   phase=pframe.get("phase"), at=pframe.get("at") or 0.0))
            record["next_id"] = int(record["next_id"]) + 1
        record["frames"] = fresh
    else:
        for frame, pframe in zip(record["frames"], precord["frames"]):
            frame["key"], frame["phase"] = bool(pframe.get("key")), pframe.get("phase")
            frame["at"] = pframe.get("at", frame.get("at"))
    flipped = 0
    for position, source in enumerate(sources):
        if source is None:
            continue
        index, pframe, path = source
        store = FrameAttempts(unit, name, direction, record["frames"][position]["id"],
                              work_dir, record=record, job=job)
        from io import BytesIO

        out = BytesIO()
        with Image.open(path) as img:
            ImageOps.mirror(img).save(out, format="PNG")
        png = out.getvalue()
        row = store.append({"source": MIRROR_SOURCE, "prompt": None, "seed": None,
                            "model": None, "loras": [], "parent": None, "seconds": None,
                            "sha256": sha256_bytes(png), "draw_from": None,
                            "pose_prompt": pframe.get("prompt"),
                            "mirror": {"direction": partner, "frame": pframe["id"],
                                       "ref": f"{pframe['id']}:{index:02d}",
                                       "sha256": sha256_file(path),
                                       "how": "flip horizontally"}})
        write_bytes_in_place(store.path(row), png)
        store.frame["current"] = int(row["index"])
        store.frame["verdict"] = None
        flipped += 1
        log(f"animate: {frame_label(store.frame, position)} <- {partner} "
            f"{pframe['id']}:{index:02d} flipped")
    warning = handedness_warning(unit, work_dir)
    record["mirror"] = {"from": partner, "at": _now(), "frames": flipped, "warning": warning}
    save(job, record)
    log(f"animate: warning - {warning}")
    return {"from": partner, "frames": flipped, "warning": warning, "job": job.name}


# ---------------------------------------------------------------------------
# Game size: the pipeline's own post chain, CPU only
# ---------------------------------------------------------------------------
def game_frame(source: Path, unit: str, dest: Path, *, palette: Path | None = None) -> Path:
    """One image through the pipeline's own post chain: matte the flat white
    off (`post.matte_chroma`), downscale to the unit's height on the 64 px
    canvas, snap to the unit's palette. No part lock."""
    from anim.stages import post as post_stage

    palette_path = Path(palette) if palette else config.PALETTE
    colours, _record = palettekit.unit_colours(unit, imaging.read_palette(palette_path),
                                               palette_path)
    palette_png = imaging.palette_png(colours, dest.parent.parent / "palette.png")
    with Image.open(source) as img:
        cut = post_stage.matte_chroma(img.convert("RGB"))
    frame, _steps = post_stage.process_cell(cut, target_height=config.height_for_unit(unit),
                                            palette_png=palette_png,
                                            chroma=imaging.chroma_key_for(colours))
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.save(dest)
    return dest


def game_file(unit: str, action: str, direction: str, frame: str | int, *,
              index: int | None = None, work_dir: Path | None = None,
              palette: Path | None = None) -> Path:
    """The frame's current attempt (or `index`) at game size, cached by the
    source's hash under the job's game/."""
    store = FrameAttempts(unit, action, direction, frame, work_dir)
    index = store.frame.get("current") if index is None else int(index)
    if index is None:
        raise ValueError(f"{frame_label(store.frame, store.position)} has no picture yet")
    row = _judgeable(store, index)
    path = store.path(row)
    digest = sha256_file(path)
    dest = store.job.root / GAME_DIR / f"{digest[:16]}.png"
    if not dest.exists():
        game_frame(path, unit, dest, palette=palette)
    return dest


# ---------------------------------------------------------------------------
# Output: the approved frames as the cell's finished frames
# ---------------------------------------------------------------------------
def output_timing(record: dict[str, Any]) -> dict[str, Any]:
    """The timing block the export carries: one tick per frame at 12 fps,
    each frame tagged with its phase (an in-between carries the phase of the
    keyframe before it), the anchor on the first action frame."""
    frames = record["frames"]
    tags: list[str] = []
    current = config.PHASE_NAMES[0]
    for frame in frames:
        if frame.get("key") and frame.get("phase"):
            current = frame["phase"]
        tags.append(current)
    fps = int(config.ANIMATE_FPS)
    weights = [1] * len(frames)
    anchor = tags.index("action") if "action" in tags else 0
    return {"weights": weights, "anchor": anchor, "source": f"animate:{record['studio_action']}",
            "fps": fps, "loop": bool(record.get("loop")), "phases": tags,
            "durations_ms": [int(round(1000 / fps)) for _ in weights],
            "keys": [i for i, f in enumerate(frames) if f.get("key")]}


def output_state(record: dict[str, Any], work_dir: Path | None = None) -> dict[str, Any]:
    """Whether the finished frames are written and still match the approvals."""
    block = record.get("output") or {}
    target = curation.canonical_job(record["unit"], record["studio_action"], record["cell"])
    approved = [(f.get("verdict") or {}).get("sha256") for f in record["frames"]]
    job = Job.open(target, work_dir)
    written = bool(block) and job.manifest_path.exists()
    return {"job": target, "at": block.get("at"), "written": written,
            "stale": written and list(block.get("frames") or []) != approved,
            "ready": all(frame_status(f) == "approved" for f in record["frames"]),
            "error": record.get("output_error")}


def _maybe_output(unit: str, action: str, direction: str, *, work_dir: Path | None,
                  palette: Path | None) -> None:
    job, record = open_job(unit, action, direction, work_dir)
    if not all(frame_status(f) == "approved" for f in record["frames"]):
        return
    try:
        output(unit, action, direction, work_dir=work_dir, palette=palette)
    except (ValueError, FileNotFoundError, OSError, palettekit.PaletteError) as exc:
        job, record = open_job(unit, action, direction, work_dir)
        record["output_error"] = str(exc)
        save(job, record)


def output(unit: str, action: str, direction: str, *, replace_job: bool = False,
           work_dir: Path | None = None, palette: Path | None = None,
           log: Callable[[str], None] = lambda _: None) -> dict[str, Any]:
    """Write every approved frame, at game size, as the cell's finished
    frames in the job layout `export` reads."""
    name = action_name(action)
    direction = direction_name(direction)
    ajob, record = open_job(unit, name, direction, work_dir)
    waiting = [frame_label(f, i) for i, f in enumerate(record["frames"])
               if frame_status(f) != "approved"]
    if waiting:
        raise ValueError(f"not every frame is approved yet - {', '.join(waiting)}")
    sources: list[tuple[dict[str, Any], Path]] = []
    for position, frame in enumerate(record["frames"]):
        store = FrameAttempts(unit, name, direction, frame["id"], work_dir, record=record,
                              job=ajob)
        row = _judgeable(store, int(frame["verdict"]["index"]))
        path = store.path(row)
        if sha256_file(path) != frame["verdict"].get("sha256"):
            raise ValueError(f"{frame_label(frame, position)} changed since it was approved; "
                             f"approve it again")
        sources.append((frame, path))
    verb = config.STUDIO_ACTIONS[name]["verb"]
    target = curation.canonical_job(unit, name, direction)
    job = Job.open(target, work_dir)
    if job.manifest_path.exists() and "animate" not in job.read_manifest():
        if not replace_job:
            raise ValueError(f"work/{target} holds this cell's frames from another route (the "
                             f"video matrix); replace it with `--replace-job` (the old job "
                             f"moves to work/{REPLACED_DIR}/), or leave it")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        aside = job.root.parent / REPLACED_DIR / f"{target}_{stamp}"
        aside.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(job.root), str(aside))
        log(f"animate: work/{target} moved aside to {aside}")
    job.post_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{unit}_{verb}"
    for old in job.post_dir.glob(f"{stem}_*.png"):
        if old.stem[len(stem) + 1:].isdigit():
            old.unlink()
    palette_path = Path(palette) if palette else config.PALETTE
    report_frames = []
    for n, (frame, path) in enumerate(sources):
        dest = job.post_dir / f"{stem}_{n}.png"
        game_frame(path, unit, dest, palette=palette_path)
        report_frames.append({"file": dest.name, "frame": frame["id"],
                              "attempt": f"{frame['id']}:{int(frame['verdict']['index']):02d}",
                              "source_sha256": frame["verdict"]["sha256"],
                              "sha256": sha256_file(dest)})
    timing = output_timing(record)
    yaw, elevation = config.direction(direction)
    write_json_in_place(job.report_path, {
        "schema": "anim.post-report/1", "route": "animate", "job": target,
        "palette": {"snap": True, "file": str(palette_path)},
        "height": config.height_for_unit(unit), "frames": report_frames})
    picks = list(range(len(sources)))
    plan_ = phase_plan(name, unit)
    write_json_in_place(job.picks_path, {
        "schema": pick_stage.PICKS_SCHEMA, "n": len(picks), "action": verb, "dedup": None,
        "picks": picks, "auto": list(picks), "timing": dict(timing),
        "auto_timing": dict(timing), "signals": {},
        "phases": {"tags": timing["phases"], "plan": plan_.as_record(),
                   "keys": timing["keys"], "source": "animate"}})
    now = _now()
    manifest = {
        "schema": JOB_SCHEMA, "fps": timing["fps"], "source_fps": None, "frames": len(picks),
        "frame_size": list(config.CANVAS), "crop": None, "action": verb, "slug": unit,
        "created": now, "rerender_requested": False,
        "motion_source": {"kind": "animate", "direction": direction, "yaw": yaw,
                          "elevation": elevation, "job": ajob.name},
        "animate": {"job": ajob.name, "at": now, "studio_action": name,
                    "frames": [{"id": f["id"], "key": bool(f.get("key")), "phase": f.get("phase"),
                                "attempt": f"{f['id']}:{int(f['verdict']['index']):02d}",
                                "sha256": f["verdict"]["sha256"]} for f, _p in sources]},
    }
    job.write_manifest(manifest)
    record["output"] = {"job": target, "at": now,
                        "frames": [f["verdict"]["sha256"] for f, _p in sources]}
    record["output_error"] = None
    save(ajob, record)
    log(f"animate: {len(picks)} frame(s) -> work/{target}/post/")
    return {"job": target, "frames": len(picks), "timing": timing}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unit", "--slug", dest="slug", required=True,
                        help="unit slug, e.g. hooded_archer")
    parser.add_argument("--action", default=None,
                        help=f"a studio action: {', '.join(config.STUDIO_ACTIONS)} "
                             f"(or its pipeline verb); required unless --sweep")
    parser.add_argument("--dir", dest="direction", default=None, metavar="CELL",
                        help="the side direction: s0..s315, or its view (right, back_right, "
                             "back, back_left, left, front_left, front, front_right); "
                             "required unless --sweep")
    parser.add_argument("--sweep", nargs="+", default=None, metavar="ACTION:DIR",
                        help="the keyframe sweep: --batch keys (or inbetweens) on each of "
                             "these animations as one run, e.g. --sweep walk:s90 run:s90 "
                             f"(at most {config.ANIMATE_SWEEP_MAX_CELLS} cells)")
    parser.add_argument("--frame", default=None,
                        help="the frame: its number (1..), its id (f03) or a keyframe's phase "
                             "(wind-up). Alone, it draws a new attempt of that frame")
    parser.add_argument("--prompt", default=None,
                        help="the pose, in words (default: the frame's last, else the phase's "
                             "sentence, or 'moving from <key> toward <key>' for an in-between)")
    parser.add_argument("--draw-from", dest="draw_from", default=None,
                        choices=config.ANIMATE_DRAW_FROM,
                        help="view (the direction's accepted view), keys (the approved "
                             "keyframes either side) or both; default view for a keyframe, "
                             "keys for an in-between")
    parser.add_argument("--skeleton", dest="skeleton", action="store_true", default=None,
                        help="send the rig clip's skeleton as the last image (the default "
                             "when the rig has a clip for the action)")
    parser.add_argument("--no-skeleton", dest="skeleton", action="store_false",
                        help="no skeleton")
    parser.add_argument("--skeleton-frame", dest="skeleton_frame", type=int, default=None,
                        metavar="N",
                        help=f"the skeleton at scrub frame N of the clip "
                             f"({config.ANIMATE_SCRUB_FPS} fps; implies --skeleton)")
    parser.add_argument("--count", type=int, default=None, help="attempts (default 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="first seed; attempt i uses seed + i (default: random)")
    parser.add_argument("--model", default=None, metavar="FILE",
                        help="the Qwen-Image-Edit checkpoint, from gpu/models.registry.tsv")
    parser.add_argument("--lora", action="append", default=None, metavar="FILE[:STRENGTH]",
                        help="a LoRA to stack, repeatable, from the registry")
    parser.add_argument("--batch", default=None, choices=BATCHES,
                        help="all (keyframes; once every keyframe is approved, the "
                             "in-betweens), keys, rejected or inbetweens")
    parser.add_argument("--approve", action="store_true",
                        help="approve the frame's current attempt (or --select NN's)")
    parser.add_argument("--reject", action="store_true",
                        help="reject the frame's current attempt (or --select NN's)")
    parser.add_argument("--select", type=int, default=None, metavar="NN",
                        help="make attempt NN the frame's current one")
    parser.add_argument("--approve-reviewed", dest="approve_reviewed", action="store_true",
                        help="approve every frame waiting for review")
    parser.add_argument("--add-frame", dest="add_frame", type=int, default=None, metavar="N",
                        help="insert an in-between after frame N (0: first)")
    parser.add_argument("--remove-frame", dest="remove_frame", action="store_true",
                        help="take --frame out of the row (its attempts are kept)")
    parser.add_argument("--key", default=None, metavar="PHASE",
                        help=f"mark --frame as a keyframe: {', '.join(config.PHASE_NAMES)}")
    parser.add_argument("--unkey", action="store_true", help="un-mark --frame as a keyframe")
    parser.add_argument("--mirror", action="store_true",
                        help="flip the partner direction's frames into this one (no GPU; "
                             "they come back to review)")
    parser.add_argument("--output", action="store_true",
                        help="write the approved frames as the cell's finished frames")
    parser.add_argument("--replace-job", dest="replace_job", action="store_true",
                        help="with --output: move another route's job for the cell aside first")
    parser.add_argument("--inpaint", default=None, metavar="REF",
                        help="redraw the erased region of --mask into attempt REF (f03:01)")
    parser.add_argument("--mask", default=None, metavar="PNG",
                        help="with --inpaint: the attempt as left in the editor, erased = redraw")
    parser.add_argument("--strength", type=float, default=None,
                        help=f"with --inpaint: how hard (default {config.CONCEPT_INPAINT_STRENGTH})")
    parser.add_argument("--replace", default=None, metavar="REF",
                        help="write --image over attempt REF (the editor's Save)")
    parser.add_argument("--image", default=None, metavar="PNG",
                        help="with --replace: the image, the attempt's own size")
    parser.add_argument("--hide", default=None, metavar="REF", help="hide attempt REF")
    parser.add_argument("--show", default=None, metavar="REF", help="show a hidden attempt")
    parser.add_argument("--delete-file", dest="delete_file", default=None, metavar="REF",
                        help="delete a HIDDEN attempt's PNG; the row stays")
    parser.add_argument("--list", dest="list_", action="store_true",
                        help="print the frame row, statuses and attempts, and stop")
    parser.add_argument("--skeletons", action="store_true",
                        help="render the scrub skeletons of the clip (CPU) and stop")
    parser.add_argument("--force", action="store_true",
                        help="draw frames of a base whose gate is still pending")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="seconds to wait for one /prompt job")


# The flags that spend no GPU: the curation gate lets these through.
CPU_FLAGS = ("approve", "reject", "select", "approve_reviewed", "add_frame", "remove_frame",
             "key", "unkey", "mirror", "output", "replace", "hide", "show", "delete_file",
             "list_", "skeletons")


def spends_gpu(args: argparse.Namespace) -> bool:
    """Whether this line reaches ComfyUI - what the base gate is asked about."""
    return not any(getattr(args, flag, None) not in (None, False) for flag in CPU_FLAGS)


def _print_row(record: dict[str, Any]) -> None:
    for position, frame in enumerate(record["frames"]):
        status = frame_status(frame)
        current = frame.get("current")
        print(f" {position + 1:>2} {frame['id']:<4} {frame_label(frame, position):<12} "
              f"{'key' if frame.get('key') else '   '} {status:<10} "
              f"{'' if current is None else f'current {current:02d}'}  "
              f"{len(frame.get('candidates') or [])} attempt(s)")


def _run_sweep(args: argparse.Namespace) -> int:
    if args.batch not in SWEEP_BATCHES:
        raise SystemExit(f"anim animate: --sweep needs --batch {' or '.join(SWEEP_BATCHES)}")
    try:
        cells = [parse_cell(spec) for spec in args.sweep]
        options = {"draw_from": args.draw_from, "skeleton": args.skeleton,
                   "count": args.count, "seed": args.seed, "model": args.model,
                   "loras": args.lora}
        result = sweep(args.slug, cells, args.batch,
                       client=comfy.ComfyClient(getattr(args, "comfyui_url", None)),
                       timeout_s=args.timeout, log=print, **options)
    except (ValueError, FileNotFoundError, OSError, models_lock.RegistryError) as exc:
        raise SystemExit(f"anim animate: {exc}") from None
    return outcome.Outcome(**result["tally"]).report()


def run_stage(args: argparse.Namespace) -> int:
    if getattr(args, "sweep", None):
        return _run_sweep(args)
    if args.action is None or args.direction is None:
        raise SystemExit("anim animate: --action and --dir are required (or --sweep)")
    try:
        unit = args.slug
        action = action_name(args.action)
        direction = direction_name(args.direction)
    except ValueError as exc:
        raise SystemExit(f"anim animate: {exc}") from None
    say = lambda exc: SystemExit(f"anim animate: {exc}")  # noqa: E731
    errors = (ValueError, FileNotFoundError, OSError, models_lock.RegistryError,
              palettekit.PaletteError)
    frame = args.frame

    def need_frame(flag: str) -> str:
        if frame is None:
            raise SystemExit(f"anim animate: {flag} needs --frame")
        return frame

    try:
        for flag, verb_ in (("hide", "hidden"), ("show", "shown"), ("delete_file", "file deleted")):
            ref = getattr(args, flag)
            if ref is None:
                continue
            row = (delete_file(unit, action, direction, ref) if flag == "delete_file"
                   else hide(unit, action, direction, ref, hidden=flag == "hide"))
            print(f"animate: {ref} {verb_} ({row['file']})")
            return 0
        if args.list_:
            job, record = open_job(unit, action, direction, create=False)
            print(f"animate: {job.name}  ({'on disk' if job.manifest_path.exists() else 'not started'})"
                  f"  {counts(record['frames'])['approved']} / {len(record['frames'])} approved")
            _print_row(record)
            return 0
        if args.skeletons:
            rendered = skeletons(unit, action, direction)
            print(f"animate: {len(rendered['files'])} skeletons of {rendered['clip']} at "
                  f"{rendered['fps']} fps -> {rendered['dir']}")
            return 0
        if args.add_frame is not None:
            row = add_frame(unit, action, direction, args.add_frame)
            print(f"animate: frame {row['id']} added after frame {args.add_frame}")
            return 0
        if args.remove_frame:
            row = remove_frame(unit, action, direction, need_frame("--remove-frame"))
            print(f"animate: frame {row['id']} removed; its "
                  f"{len(row.get('candidates') or [])} attempt(s) are kept in the record")
            return 0
        if args.key is not None or args.unkey:
            row = set_key(unit, action, direction, need_frame("--key"), key=not args.unkey,
                          phase=args.key)
            print(f"animate: {row['id']} is {'a ' + row['phase'] + ' keyframe' if row['key'] else 'an in-between'}")
            return 0
        if args.approve_reviewed:
            ids = approve_reviewed(unit, action, direction)
            print(f"animate: {len(ids)} frame(s) approved: {', '.join(ids) or '-'}")
            return 0
        if args.approve or args.reject or args.select is not None:
            target = need_frame("--approve / --reject / --select")
            if args.approve or args.reject:
                row = judge(unit, action, direction, target,
                            "approved" if args.approve else "rejected", index=args.select)
                print(f"animate: {row['id']} {row['verdict']['state']} "
                      f"({row['id']}:{row['verdict']['index']:02d})")
            else:
                row = select(unit, action, direction, target, args.select)
                print(f"animate: {row['id']} shows {row['id']}:{args.select:02d}")
            return 0
        if args.mirror:
            result = mirror(unit, action, direction, log=print)
            print(f"animate: {result['frames']} frame(s) mirrored from {result['from']}, "
                  f"to review")
            return 0
        if args.output:
            result = output(unit, action, direction, replace_job=args.replace_job, log=print)
            print(f"animate: {result['frames']} finished frame(s) in work/{result['job']}")
            return 0
        if args.replace is not None:
            if not args.image:
                raise SystemExit("anim animate: --replace needs --image PNG")
            row = replace(unit, action, direction, args.replace, Path(args.image).read_bytes())
            print(f"animate: {args.image} -> {args.replace} ({row['file']}), edited by hand")
            return 0
        client = comfy.ComfyClient(getattr(args, "comfyui_url", None))
        if args.inpaint is not None:
            if not args.mask:
                raise SystemExit("anim animate: --inpaint needs --mask PNG")
            row = inpaint(unit, action, direction, args.inpaint, Path(args.mask),
                          prompt=args.prompt, strength=args.strength, seed=args.seed,
                          client=client, timeout_s=args.timeout, log=print)
            print(f"animate: {args.inpaint} inpainted in place ({row['file']})")
            return 0
        options = {"prompt": args.prompt, "draw_from": args.draw_from, "skeleton": args.skeleton,
                   "skeleton_frame": args.skeleton_frame, "count": args.count,
                   "seed": args.seed, "model": args.model, "loras": args.lora}
        if args.batch is not None:
            options.pop("prompt")                 # each frame keeps its own
            options.pop("skeleton_frame")
            result = generate_batch(unit, action, direction, args.batch, client=client,
                                    timeout_s=args.timeout, log=print, **options)
            return outcome.Outcome(**result["cells"]).report()
        if frame is None:
            raise SystemExit("anim animate: --frame is required to draw a frame (or --batch, "
                             "--approve, --list, --mirror, --output, ...)")
        started = time.monotonic()
        rows = generate(unit, action, direction, frame, client=client, timeout_s=args.timeout,
                        log=print, **options)
    except errors as exc:
        raise say(exc) from None
    wall = time.monotonic() - started
    job = Job.open(job_name(unit, action, direction))
    print(f"animate: {len(rows)} attempt(s) of {unit} {action} {frame} at {direction} in "
          f"{wall:.1f} s wall -> {job.root / ATTEMPTS_DIR}")
    for row in rows:
        print(f"  {row['file']}  from {row['draw_from']}"
              f"{' + skeleton ' + str(row['skeleton']['frame']) if row.get('skeleton') else ''}"
              f"  seed {row['seed']}  {row['seconds']} s")
    print(f"  then: anim animate --unit {unit} --action {action} --dir {direction} "
          f"--frame {frame} --approve   (or the Animate tab)")
    return 0


run = run_stage
