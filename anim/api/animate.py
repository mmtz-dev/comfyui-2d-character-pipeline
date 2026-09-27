"""The animate tab feature's read side: one unit's animation of one action at
one direction, for the studio's Animate tab.

    GET /animate/{unit}/{action}/{direction}   -> state(...)

Pure over the files `anim animate` writes (work/<unit>_animate_<action>_<dir>/
job.json and attempts/) plus the base job and its views; the writes are the
stage's functions, called inline for the CPU ones (approve, reject, select,
the row's layout, mirror, output, the editor's Save) and through the run
manager for the GPU ones (`POST .../generate`, `.../inpaint`). Attempt images
are served by the frame-set routes the API already has (`/jobs/<job>/frames/
animate/<n>.png` - a file's position in its listing, the Concept tab's trick);
the skeletons and the game-size frames have routes of their own because they
are rendered on first ask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anim import config, curation
from anim.api import concept as concept_api
from anim.api import jobs as jobs_api
from anim.job import Job
from anim.stages import animate as animate_stage
from anim.stages import concept as concept_stage
from anim.stages import matrix as matrix_stage


def _url(job: Job, set_name: str, path: Path | str | None) -> str | None:
    if not path:
        return None
    files = jobs_api.FRAME_STAGES[set_name](job)
    index = concept_api._url_index(files, Path(path).name)
    return None if index is None else f"/jobs/{job.name}/frames/{set_name}/{index}.png"


def compass(unit: str, action: str, work_dir: Path) -> list[dict[str, Any]]:
    """The eight side directions: each one's own view (a thumbnail when it is
    accepted), why it is blocked when it is, its mirror partner, and how far
    this action has got there."""
    base_job = Job.open(f"{unit}_base", work_dir)
    views = matrix_stage.base_views(unit, work_dir)
    gate = curation.views_blocked(unit, work_dir)
    rows = []
    for direction in animate_stage.side_directions():
        view = config.view_for_direction(direction)
        url: str | None = None
        blocked: str | None = gate
        if view == config.FRONT_VIEW:
            if not blocked:
                try:
                    _j, path, _m = concept_stage.base_render(unit, work_dir)
                    url = _url(base_job, "render", path)
                except (FileNotFoundError, ValueError) as exc:
                    blocked = str(exc)
        elif not blocked:
            entry = views.get(view)
            if entry is None:
                blocked = f"no {view} view yet - draw it on the Concept tab"
            elif entry.get("state") != "accepted":
                blocked = f"the {view} view is {entry.get('state')} - accept it first"
            else:
                url = _url(base_job, "render", entry["path"])
        job = Job.open(animate_stage.job_name(unit, action, direction), work_dir)
        approved = total = 0
        if job.manifest_path.exists():
            frames = job.read_manifest().get("frames") or []
            total = len(frames)
            approved = sum(1 for f in frames if animate_stage.frame_status(f) == "approved")
        rows.append({"direction": direction, "view": view,
                     "yaw": config.direction(direction)[0], "url": url, "blocked": blocked,
                     "mirror_of": config.mirror_direction(direction),
                     "approved": approved, "frames": total, "exists": job.manifest_path.exists()})
    return rows


def actions(unit: str, direction: str, work_dir: Path) -> list[dict[str, Any]]:
    """Every studio action with its frame count at this direction (the row
    on disk, else the template's budget) and the clip the skeleton is from."""
    rows = []
    for name, row in config.STUDIO_ACTIONS.items():
        job = Job.open(animate_stage.job_name(unit, name, direction), work_dir)
        frames = (len(job.read_manifest().get("frames") or []) if job.manifest_path.exists()
                  else int(animate_stage.phase_plan(name, unit).budget))
        rows.append({"action": name, "verb": row["verb"], "frames": frames,
                     "loop": bool(config.ACTION_PHASES[name]["loop"]),
                     "clip": matrix_stage.clip_for(config.MATRIX_RIG, name)})
    return rows


def _neighbour(frames: list[dict[str, Any]], position: int | None) -> dict[str, Any] | None:
    if position is None:
        return None
    frame = frames[position]
    return {"id": frame["id"], "position": position + 1,
            "label": animate_stage.frame_label(frame, position),
            "approved": animate_stage.frame_status(frame) == "approved"}


def state(unit: str, action: str, direction: str, work_dir: Path) -> dict[str, Any] | None:
    """Everything the Animate tab shows for one (unit, action, direction), or
    None when the unit is not a slug. Raises ValueError for an action or a
    direction that is not one. A cell with nothing yet answers `exists: false`
    with the template's row, so the tab can offer Generate."""
    if not jobs_api.valid_name(unit):
        return None
    name = animate_stage.action_name(action)
    direction = animate_stage.direction_name(direction)
    job, record = animate_stage.open_job(unit, name, direction, work_dir, create=False)
    exists = job.manifest_path.exists()
    attempts = jobs_api.FRAME_STAGES["animate"](job)
    base_job = Job.open(f"{unit}_base", work_dir)

    reference: dict[str, Any] | None = None
    reference_error: str | None = None
    try:
        ref = animate_stage.reference(unit, direction, work_dir)
        reference = {"view": ref["view"], "sha256": ref["sha256"],
                     "url": _url(base_job, "render", ref["path"])}
    except (FileNotFoundError, ValueError) as exc:
        reference_error = str(exc)

    rig: dict[str, Any] | None = None
    rig_error: str | None = None
    try:
        clip = animate_stage.rig_clip(name)
        times = animate_stage.scrub_times(clip["duration"])
        rig = {"rig": clip["rig"], "clip": clip["clip"], "duration": clip["duration"],
               "fps": config.ANIMATE_SCRUB_FPS, "frames": len(times), "times": times}
    except (ValueError, FileNotFoundError) as exc:
        rig_error = str(exc)

    base = f"/animate/{unit}/{name}/{direction}"
    frames = record["frames"]
    loop = bool(record.get("loop"))
    rows: list[dict[str, Any]] = []
    for position, frame in enumerate(frames):
        candidates = frame.get("candidates") or []
        prev, nxt = animate_stage.neighbour_keys(frames, position, loop=loop)
        default_frame = (animate_stage.default_skeleton_frame(frame, rig["duration"])
                         if rig is not None else None)
        status = animate_stage.frame_status(frame)
        current = frame.get("current")
        current_row = next((c for c in candidates if c.get("index") == current), None)
        sha = (current_row or {}).get("sha256") or ""
        rows.append({
            "id": frame["id"], "position": position + 1,
            "label": animate_stage.frame_label(frame, position),
            "key": bool(frame.get("key")), "phase": frame.get("phase"), "at": frame.get("at"),
            "status": status, "current": current, "verdict": frame.get("verdict"),
            "prompt": frame.get("prompt") or animate_stage.default_prompt(
                name, frames, position, loop=loop, unit=unit),
            "default_prompt": animate_stage.default_prompt(name, frames, position, loop=loop,
                                                           unit=unit),
            "draw_from": animate_stage.default_draw_from(frame),
            "before": _neighbour(frames, prev), "after": _neighbour(frames, nxt),
            "default_skeleton_frame": default_frame,
            "skeleton_frame": frame.get("skeleton_frame"),
            "candidates": [concept_api._attempt_view(c, job_name=job.name, set_name="animate",
                                                     files=attempts) for c in candidates],
            "hidden_count": concept_api._hidden_count(candidates),
            "game_url": (f"{base}/game/{frame['id']}.png?v={sha[:12]}"
                         if current_row is not None and animate_stage.is_rendered(current_row)
                         and not current_row.get("file_deleted") else None),
        })

    concept_job = Job.open(concept_stage.job_name(unit), work_dir)
    defaults = ((jobs_api._manifest(concept_job).get("defaults") or {})
                if concept_job.manifest_path.exists() else {})
    partner = config.mirror_direction(direction)
    partner_job = Job.open(animate_stage.job_name(unit, name, partner), work_dir) if partner else None
    partner_frames = (partner_job.read_manifest().get("frames") or []
                      if partner_job is not None and partner_job.manifest_path.exists() else [])
    partner_pictures = sum(1 for f in partner_frames if f.get("current") is not None)
    return {
        "unit": unit, "action": name, "verb": config.STUDIO_ACTIONS[name]["verb"],
        "direction": direction, "view": config.view_for_direction(direction),
        "job": job.name, "exists": exists, "loop": loop,
        "fps": config.ANIMATE_FPS, "height": config.height_for_unit(unit),
        "max_frames": config.ANIMATE_MAX_FRAMES, "phases": list(config.PHASE_NAMES),
        "draw_from_options": list(config.ANIMATE_DRAW_FROM),
        "max_images": config.ANIMATE_MAX_IMAGES,
        "compass": compass(unit, name, work_dir),
        "actions": actions(unit, direction, work_dir),
        "locked": curation.views_blocked(unit, work_dir),
        "reference": reference, "reference_error": reference_error,
        "rig": rig, "rig_error": rig_error,
        "skeleton_url": f"{base}/skeleton/{{frame}}.png",
        "frames": rows,
        "counts": animate_stage.counts(frames),
        "next_step": animate_stage.next_step(frames),
        "mirror": {"partner": partner,
                   "partner_view": config.view_for_direction(partner) if partner else None,
                   "available": bool(partner_pictures),
                   "partner_pictures": partner_pictures,
                   "last": record.get("mirror"),
                   "warning": animate_stage.handedness_warning(unit, work_dir) if partner else None},
        "output": animate_stage.output_state(record, work_dir),
        "defaults": defaults,
        "last_seed": record.get("last_seed"),
    }
