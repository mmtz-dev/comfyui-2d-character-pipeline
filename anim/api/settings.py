"""`work/project.json` - the project settings the studio's features read.

Just the store: `GET /settings` returns the file merged over the defaults and
`PUT /settings` validates and writes it in place. Nothing here applies a
setting - the palette feature reads `palette`, the keyframe phase feature
reads `frame_budgets`, the curation board reads `curation`, the export reads
`frame_size`. Until those land, the CLI's config.py values stay in force and
this file is a note the UI can already edit.

    palette        path to the project's .gpl / one-row .png (null = config.PALETTE)
    frame_budgets  {action: frames} overriding config.ACTION_FRAMES, capped at 12
    curation       gates | every | end   (sprite-studio.md S2 "Curation")
    frame_size     [w, h] of one exported frame (game-02's contract is 64x64)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from anim import config
from anim.util import read_json, write_json_in_place

SETTINGS_SCHEMA = "anim.project/1"
SETTINGS_FILENAME = "project.json"
CURATION_MODES = ("gates", "every", "end")
# The art bible's cap (sprite-studio.md S2 "Frame budget": cap 12).
FRAME_BUDGET_MAX = 12

DEFAULTS: dict[str, Any] = {
    "schema": SETTINGS_SCHEMA,
    "palette": None,
    "frame_budgets": {},
    "curation": "gates",
    "frame_size": list(config.CANVAS),
}


class SettingsError(ValueError):
    """A PUT that does not describe a project."""


def settings_path(work_dir: Path) -> Path:
    return Path(work_dir) / SETTINGS_FILENAME


def load(work_dir: Path) -> dict[str, Any]:
    """The stored settings over the defaults; the defaults alone when there
    is no file yet, so a fresh work/ answers the same shape."""
    merged = copy.deepcopy(DEFAULTS)
    path = settings_path(work_dir)
    if path.exists():
        stored = read_json(path)
        if isinstance(stored, dict):
            merged.update({k: v for k, v in stored.items() if k in DEFAULTS})
    return merged


def validate(body: dict[str, Any]) -> dict[str, Any]:
    """The body merged over the defaults, every field checked, or SettingsError."""
    if not isinstance(body, dict):
        raise SettingsError("settings must be a JSON object")
    unknown = sorted(set(body) - set(DEFAULTS))
    if unknown:
        raise SettingsError(f"unknown setting(s) {unknown}; "
                            f"known: {sorted(k for k in DEFAULTS if k != 'schema')}")
    merged = copy.deepcopy(DEFAULTS)
    merged.update(body)
    merged["schema"] = SETTINGS_SCHEMA

    palette = merged["palette"]
    if palette is not None and (not isinstance(palette, str) or not palette.strip()):
        raise SettingsError("palette must be a path string or null")
    merged["palette"] = palette.strip() if isinstance(palette, str) else None

    budgets = merged["frame_budgets"]
    if not isinstance(budgets, dict):
        raise SettingsError("frame_budgets must be an object of action -> frames")
    clean: dict[str, int] = {}
    for action, count in budgets.items():
        if not isinstance(action, str) or not action.strip():
            raise SettingsError("frame_budgets keys are action names")
        if isinstance(count, bool) or not isinstance(count, int) \
                or not 1 <= count <= FRAME_BUDGET_MAX:
            raise SettingsError(f"frame_budgets[{action!r}] must be an integer "
                                f"1..{FRAME_BUDGET_MAX}, not {count!r}")
        clean[action.strip().lower()] = count
    merged["frame_budgets"] = clean

    if merged["curation"] not in CURATION_MODES:
        raise SettingsError(f"curation must be one of {list(CURATION_MODES)}")

    size = merged["frame_size"]
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in size)):
        raise SettingsError("frame_size must be [width, height], both >= 1")
    merged["frame_size"] = [int(size[0]), int(size[1])]
    return merged


def save(work_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    """Validate, write in place, return what was written."""
    merged = validate(body)
    write_json_in_place(settings_path(work_dir), merged)
    return merged
