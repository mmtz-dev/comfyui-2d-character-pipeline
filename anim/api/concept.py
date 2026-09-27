"""The concept generation feature's read side: one unit's concept state for
the studio's Concept tab, and the model registry listing.

    GET /concept/{unit}   -> state(unit, work_dir)
    GET /models           -> models(probe)

Pure over the files `anim concept` writes (work/<slug>_concept/job.json,
views/views.json, work/<slug>_base/job.json, work/<slug>_curation.json) -
nothing here starts a run or touches the GPU; the writes are the stage's
functions, called inline for the CPU ones (upload, pick, defaults) and
through the run manager for the GPU ones (`POST /concept/{unit}/generate`).
Every attempt's image is served by the frame-set routes the API already
has (`/jobs/<slug>_concept/frames/candidates/<n>.png`, `.../views/<n>.png`):
the URL is the file's position in that set's listing, the same trick
`curation.views_of` uses, so the page and the API agree on what a URL is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anim import comfy, config, curation, models_lock
from anim.api import jobs as jobs_api
from anim.job import Job
from anim.stages import concept as concept_stage
from anim.util import read_json

# The loaders whose combo lists say which files ComfyUI can see, per family and
# role; `GET /models?probe=1` reads them out of `/object_info`. The same map
# is what `concept.check_installed` refuses on before posting.
PROBE_INPUTS = comfy.LOADER_INPUTS


def _url_index(files: list[Path], name: str | None) -> int | None:
    names = [p.name for p in files]
    return names.index(name) if name and name in names else None


def _attempt_view(row: dict[str, Any], *, job_name: str, set_name: str,
                  files: list[Path]) -> dict[str, Any]:
    """One attempt as the tab sees it. Hidden rows are reported, not dropped:
    the tab filters them (and its "show hidden" toggle costs no round trip),
    and a row whose file was deleted keeps its record with no URL."""
    index = None if row.get("file_deleted") else _url_index(files, row.get("file"))
    return {**row, "url": (f"/jobs/{job_name}/frames/{set_name}/{index}.png"
                           if index is not None else None),
            "hidden": bool(row.get("hidden")),
            "file_deleted": row.get("file_deleted"),
            "rendered": row.get("seconds") is not None
                        or row.get("source") in ("upload", "existing")}


def _hidden_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("hidden"))


def _edited_since_accepted(rows: list[dict[str, Any]], pick: dict[str, Any] | None) -> bool:
    """The retouch feature: the accepted attempt no longer matches the copy
    that was taken when it was accepted, so the picture under
    `<unit>_base/render/` is the older one. Accepting it again is what pushes
    the edit through; nothing is re-copied behind the owner's back. The
    comparison is the hash recorded AT PICK TIME against the attempt's
    current one - no file is read here, and an edit in the same second as the
    accept still counts."""
    index = (pick or {}).get("index")
    if index is None:
        return False
    row = next((r for r in rows if r.get("index") == index), None)
    if row is None:
        return False
    copied = (pick or {}).get("sha256")
    if copied and row.get("sha256"):
        return str(copied) != str(row["sha256"])
    # An older record with no hash of the copy: fall back to the timestamps
    # (second resolution, so an edit in the same second as the accept is
    # missed - which is why the hash is written now).
    edited = row.get("edited") or {}
    picked = (pick or {}).get("picked")
    return bool(edited.get("at") and picked and str(edited["at"]) > str(picked))


def _recorded_seconds(views_record: dict[str, Any]) -> list[float]:
    """Every view render this record clocked, in the order it wrote them."""
    return [c["seconds"] for row in (views_record.get("views") or {}).values()
            for c in (row.get("candidates") or [])
            if isinstance(c.get("seconds"), (int, float))]


def estimate_seconds(unit: str, work_dir: Path,
                     views_record: dict[str, Any] | None = None) -> float:
    """How long one view render takes, for the estimate the tab quotes before
    a sweep. This unit's own warm median first; failing that every unit's, so
    a brand-new unit still gets a real number off the runs already on disk;
    failing that the measured constant."""
    own = _recorded_seconds(views_record) if views_record is not None else []
    if not own:
        own = _recorded_seconds(_views_record(unit, work_dir))
    warm = [s for s in own if 0 < s < config.CONCEPT_WARM_CUTOFF_S]
    if not warm:
        warm = [s for path in sorted(work_dir.glob("*_concept/views/views.json"))
                for s in _recorded_seconds(read_json(path) or {})
                if 0 < s < config.CONCEPT_WARM_CUTOFF_S]
    return config.concept_view_seconds(warm)


def _views_record(unit: str, work_dir: Path) -> dict[str, Any]:
    path = concept_stage.views_path(unit, work_dir)
    return (concept_stage.migrate_views(read_json(path)) if path.exists()
            else concept_stage.empty_views(unit))


def state(unit: str, work_dir: Path) -> dict[str, Any] | None:
    """Everything the Concept tab shows for one unit, or None when the unit
    is not a slug. A unit with nothing yet answers `exists: false` with empty
    rows, so the tab can offer the first prompt."""
    if not jobs_api.valid_name(unit):
        return None
    job = Job.open(concept_stage.job_name(unit), work_dir)
    manifest: dict[str, Any] = {}
    if job.manifest_path.exists():
        manifest = concept_stage.migrate_manifest(jobs_api._manifest(job))
    candidates = jobs_api.FRAME_STAGES["candidates"](job)
    view_files = jobs_api.FRAME_STAGES["views"](job)

    base_job = Job.open(f"{unit}_base", work_dir)
    base_manifest = jobs_api._manifest(base_job) if base_job.manifest_path.exists() else {}
    final = (base_manifest.get("render") or {}).get("final")
    render_files = jobs_api.FRAME_STAGES["render"](base_job)
    front_index = _url_index(render_files, final)
    concept_block = base_manifest.get("concept") or {}

    record = curation.read(unit, work_dir)
    gates = dict(record["gates"]) if record else {}
    view_states = dict(record.get("views") or {}) if record else {}

    views_record = _views_record(unit, work_dir)
    stale_views = bool(views_record.get("base_sha256")) and final is not None and \
        (base_job.render_dir / final).exists() and \
        views_record.get("base_sha256") != concept_stage.sha256_file(base_job.render_dir / final)
    parts, _avoid = concept_stage.base_phrases(unit, work_dir)

    views: dict[str, Any] = {}
    for view, spec in config.CONCEPT_VIEWS.items():
        row = (views_record.get("views") or {}).get(view) or concept_stage.view_row(view, None)
        stored = (concept_block.get("views") or {}).get(view) or {}
        views[view] = {
            "view": view, "yaw": spec["yaw"], "elevation": spec.get("elevation", "side"),
            "group": "main" if view in config.CONCEPT_MAIN_VIEWS else "advanced",
            "prompt": row.get("prompt") or config.concept_view_prompt(view, parts),
            "default_prompt": config.concept_view_prompt(view, parts),
            "candidates": [] if stale_views else [
                _attempt_view(c, job_name=job.name, set_name="views", files=view_files)
                for c in row.get("candidates") or []],
            "pick": None if stale_views else (row.get("pick") or {}).get("index"),
            "hidden_count": 0 if stale_views else _hidden_count(row.get("candidates") or []),
            "pick_stale": (not stale_views) and _edited_since_accepted(
                row.get("candidates") or [], row.get("pick")),
            "state": view_states.get(view) or ("accepted" if stored else "pending"),
            "accepted_file": stored.get("file"),
            # The mirrored view feature: the partner this view can be derived
            # from (null for back and top), and whether the picture the base
            # carries already is that flip. The tab offers the button off the
            # first and draws the marker off the second.
            "mirror_of": config.mirror_of(view),
            "mirrored": bool(stored.get("mirrored")),
            "mirrored_from": (stored.get("mirror") or {}).get("view"),
        }

    return {
        "unit": unit, "job": job.name,
        "exists": bool(manifest) or base_job.manifest_path.exists(),
        "cell": manifest.get("cell") or list(config.CONCEPT_CELL),
        "locked": curation.views_blocked(unit, work_dir),
        "gates": gates,
        "defaults": manifest.get("defaults") or {},
        "last_seed": manifest.get("last_seed"),
        "parts_phrase": parts,
        "base": {
            "description": manifest.get("description"),
            "prompt": manifest.get("prompt"),
            "sketch": manifest.get("sketch"),
            "candidates": [_attempt_view(c, job_name=job.name, set_name="candidates",
                                         files=candidates)
                           for c in manifest.get("candidates") or []],
            "pick": (manifest.get("pick") or {}).get("index"),
            "hidden_count": _hidden_count(manifest.get("candidates") or []),
            "pick_stale": _edited_since_accepted(manifest.get("candidates") or [],
                                                 manifest.get("pick")),
            "accepted": final is not None and (base_job.render_dir / final).exists(),
            "front_url": (f"/jobs/{base_job.name}/frames/render/{front_index}.png"
                          if front_index is not None else None),
        },
        "views": views,
        "views_stale": stale_views,
        # What one view render costs, so the tab can quote a sweep before the
        # press instead of guessing (the advanced views sweep feature). Warm:
        # a cold run's first render is CONCEPT_VIEW_COLD_SECONDS on top.
        "estimate_seconds": estimate_seconds(unit, work_dir, views_record),
        "page": str(concept_stage.views_page_dir(unit, work_dir) / "index.html"),
    }


def models(*, probe: bool = False, comfyui_url: str | None = None,
           registry_path: Path | None = None, lock_path: Path | None = None
           ) -> dict[str, Any]:
    """The registry grouped by family and role, each row with `locked`
    (in models.lock) and - when probed - `installed` (ComfyUI lists the
    file). The concept ops' families and their workflow defaults come with
    it so the picker can show "the workflow's own" as a choice."""
    entries = models_lock.load_registry(registry_path, lock_path)
    seen: dict[tuple[str, str], list[str] | None] = {}
    client = comfy.ComfyClient(comfyui_url, timeout=3.0) if probe else None
    families: dict[str, dict[str, list[dict[str, Any]]]] = {
        f: {r: [] for r in models_lock.ROLES} for f in models_lock.FAMILIES}
    for entry in entries:
        row = entry.as_row()
        key = (entry.family, entry.role)
        if client is not None and key in PROBE_INPUTS:
            if key not in seen:
                seen[key] = client.combo_values(*PROBE_INPUTS[key])
            row["installed"] = entry.file in (seen[key] or [])
        else:
            row["installed"] = None
        families[entry.family][entry.role].append(row)
    return {
        "registry": str(Path(registry_path) if registry_path else config.MODELS_REGISTRY),
        "probed": bool(probe),
        "families": families,
        "ops": dict(config.CONCEPT_FAMILIES),
        "workflow_defaults": dict(config.CONCEPT_MODELS),
        "default_lora": {"zimage": [{"file": config.CONCEPT_LORA["file"],
                                     "strength": config.CONCEPT_LORA["strength"]}],
                         "qwen-edit": []},
    }
