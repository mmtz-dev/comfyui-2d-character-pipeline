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

# The loaders whose combo lists say which files ComfyUI can see, per family
# and role - `GET /models?probe=1` reads them out of `/object_info`.
PROBE_INPUTS: dict[tuple[str, str], tuple[str, str]] = {
    ("zimage", "checkpoint"): ("UNETLoader", "unet_name"),
    ("zimage", "lora"): ("LoraLoaderModelOnly", "lora_name"),
    ("qwen-edit", "checkpoint"): ("NunchakuQwenImageDiTLoader", "model_name"),
    ("qwen-edit", "lora"): ("LoraLoaderModelOnly", "lora_name"),
    ("wan", "checkpoint"): ("UNETLoader", "unet_name"),
}


def _url_index(files: list[Path], name: str | None) -> int | None:
    names = [p.name for p in files]
    return names.index(name) if name and name in names else None


def _attempt_view(row: dict[str, Any], *, job_name: str, set_name: str,
                  files: list[Path]) -> dict[str, Any]:
    index = _url_index(files, row.get("file"))
    return {**row, "url": (f"/jobs/{job_name}/frames/{set_name}/{index}.png"
                           if index is not None else None),
            "rendered": row.get("seconds") is not None
                        or row.get("source") in ("upload", "existing")}


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

    views_json = concept_stage.views_path(unit, work_dir)
    views_record = (concept_stage.migrate_views(read_json(views_json))
                    if views_json.exists() else concept_stage.empty_views(unit))
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
            "state": view_states.get(view) or ("accepted" if stored else "pending"),
            "accepted_file": stored.get("file"),
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
            "accepted": final is not None and (base_job.render_dir / final).exists(),
            "front_url": (f"/jobs/{base_job.name}/frames/render/{front_index}.png"
                          if front_index is not None else None),
        },
        "views": views,
        "views_stale": stale_views,
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
