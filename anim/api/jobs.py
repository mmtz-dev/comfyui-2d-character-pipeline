"""Reading `work/<job>/` for the API - listing, stage status, frame URLs.

Nothing here writes. A job's status is derived from the files the stages leave
behind (job.py's layout), never from a status field, so a job hand-edited or
re-run from the CLI reads the same over the API as it does on disk.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any, Callable

from anim import config
from anim.job import Job
from anim.util import read_json

# A job name is a directory name the stages themselves would make: slug,
# action, direction cell, joined by underscores. Anything else is refused
# before it is ever joined onto a path, so no URL reaches outside work/.
JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Which file proves which stage ran. Order is the chain's order, so the list a
# job reports reads like the CLI's stage order. `rig` counts as `extract` +
# `pose` (it writes both of their outputs) and is reported by name as well.
STAGE_MARKERS: list[tuple[str, Callable[[Job, dict[str, Any]], bool]]] = [
    ("extract", lambda job, m: bool(job.frames())),
    ("pose", lambda job, m: job.keypoints_path.exists()),
    ("rig", lambda job, m: (m.get("motion_source") or {}).get("kind") == "rig"),
    ("clip", lambda job, m: job.clip_path.exists()),
    ("pick", lambda job, m: job.picks_path.exists()),
    ("strip", lambda job, m: job.layout_path.exists()),
    ("render", lambda job, m: job.provenance_path.exists()),
    ("post", lambda job, m: job.report_path.exists()),
    ("publish", lambda job, m: "publish" in m),
    ("concept", lambda job, m: job.candidates_dir.is_dir()),
    ("bakeoff", lambda job, m: "bakeoff" in m or (job.root / "bakeoff.json").exists()),
]

# The PNG sets the API serves, by the name a URL uses. Each is the job's own
# listing, so `/jobs/x/frames/frames/3.png` is "the fourth file of
# job.frames()" - an index into a list, never a path built from the URL.
FRAME_STAGES: dict[str, Callable[[Job], list[Path]]] = {
    "frames": lambda job: job.frames(),
    "pose": lambda job: job.skeletons(),
    "depth": lambda job: job.depths(),
    "clip": lambda job: job.clip_frames(),
    "strip": lambda job: _pngs(job.strip_dir),
    "render": lambda job: _pngs(job.render_dir),
    "post": lambda job: job.post_frames(),
    "candidates": lambda job: _pngs(job.candidates_dir),
}


def _pngs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.png") if p.is_file())


def valid_name(name: str) -> bool:
    return bool(JOB_NAME.match(name)) and not name.startswith((".", "_"))


def open_job(name: str, work_dir: Path) -> Job | None:
    """The job, or None when the name is not one or nothing is there."""
    if not valid_name(name):
        return None
    job = Job.open(name, work_dir)
    if not job.root.is_dir() or not job.manifest_path.exists():
        return None
    return job


def _manifest(job: Job) -> dict[str, Any]:
    try:
        return read_json(job.manifest_path)
    except (OSError, ValueError):
        return {}


def _updated(job: Job) -> str:
    """When the job last changed: the newest of job.json and its top-level
    entries, as an ISO-8601 UTC timestamp."""
    newest = 0.0
    try:
        for entry in [job.root, job.manifest_path, *job.root.iterdir()]:
            try:
                newest = max(newest, entry.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return dt.datetime.fromtimestamp(newest, dt.timezone.utc).isoformat(
        timespec="seconds")


def stages_done(job: Job, manifest: dict[str, Any] | None = None) -> list[str]:
    manifest = _manifest(job) if manifest is None else manifest
    return [name for name, check in STAGE_MARKERS if check(job, manifest)]


def direction_of(manifest: dict[str, Any]) -> str:
    """The direction cell: what `rig` recorded, else the contract's default
    (a filmed clip is always the side view, party facing right)."""
    source = manifest.get("motion_source") or {}
    return str(source.get("direction") or config.DEFAULT_DIRECTION)


def summary(job: Job) -> dict[str, Any]:
    """One row of `GET /jobs`."""
    manifest = _manifest(job)
    return {
        "name": job.name,
        "slug": manifest.get("slug", ""),
        "action": manifest.get("action", ""),
        "direction": direction_of(manifest),
        "frames": manifest.get("frames"),
        "stages": stages_done(job, manifest),
        "rerender_requested": bool(manifest.get("rerender_requested")),
        "updated": _updated(job),
        "url": f"/jobs/{job.name}",
    }


def list_jobs(work_dir: Path) -> list[dict[str, Any]]:
    """Every directory under work/ that has a job.json, newest first.

    Directories without one (`_runs`, `_exp`, `sheets`, a stray file) are not
    jobs and are skipped rather than reported half-empty.
    """
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return []
    rows = []
    for entry in sorted(work_dir.iterdir()):
        if not entry.is_dir():
            continue
        job = open_job(entry.name, work_dir)
        if job is not None:
            rows.append(summary(job))
    rows.sort(key=lambda row: row["updated"], reverse=True)
    return rows


def detail(job: Job) -> dict[str, Any]:
    """`GET /jobs/<job>`: the manifest as written, plus what the API derives."""
    manifest = _manifest(job)
    row = summary(job)
    row["manifest"] = manifest
    row["picks"] = (read_json(job.picks_path).get("picks")
                    if job.picks_path.exists() else None)
    row["frame_sets"] = {name: len(paths) for name, lister in FRAME_STAGES.items()
                         if (paths := lister(job))}
    return row


def frames(job: Job) -> dict[str, Any]:
    """`GET /jobs/<job>/frames`: every PNG set, each entry with its URL."""
    sets: dict[str, list[dict[str, Any]]] = {}
    for stage, lister in FRAME_STAGES.items():
        paths = lister(job)
        if not paths:
            continue
        sets[stage] = [
            {"index": i, "file": path.name,
             "url": f"/jobs/{job.name}/frames/{stage}/{i}.png"}
            for i, path in enumerate(paths)]
    return {"job": job.name, "stages": sets}


def frame_path(job: Job, stage: str, index: int) -> Path | None:
    """The file behind `/jobs/<job>/frames/<stage>/<n>.png`, or None."""
    lister = FRAME_STAGES.get(stage)
    if lister is None:
        return None
    paths = lister(job)
    if not 0 <= index < len(paths):
        return None
    return paths[index]
