"""Stage: `anim export` - a unit's accepted frames -> sheets + atlas + SpriteFrames

    anim export knight
    anim export knight --sheet packed --size 48 --adapter game02

    work/_export/knight/
      sheets/knight_s0.png     one row per action, frames left to right (per-direction)
      sheets/knight.png        or ONE row-major sheet with --sheet packed
      atlas.json               where every frame is, and how long it shows
      knight.tres              a Godot 4 SpriteFrames over the sheet(s)
      preview.html             the sheets at x4 with the rects drawn, every animation playing

THE GENERIC EXPORT CONTRACT (the spritesheet export feature, sprite-studio.md
S2 "Curation, palette, export"). Any game imports the three files; nothing in
them is game-02's. `publish` is game-02's ADAPTER - `--adapter game02` runs it
per job on top of these files, so game-02 keeps its numbered PNGs, its sidecar
and its audit unchanged - and `export.adapter` in work/project.json picks the
default per project.

WHICH JOBS. A unit's animations are the jobs named `<unit>_<action>` (the
default direction, `s0`) and `<unit>_<action>_<dir>` for any other cell of
`config.DIRECTIONS` - exactly the names `anim rig` makes. The manifest's slug,
action and direction must agree with the name: `knight_strike_phases` has
slug knight and action strike but is not the matrix's strike job, so it is
skipped with a warning rather than exported under a name it does not have.
A job without `post/` frames, or posted with `--no-snap` (a raw-colour
preview; the palette is snapped before export, never at it), is skipped the
same way. The matrix is the studio's five actions x the eight side cells,
plus whatever else the unit has; every cell with no job is a listed warning,
not an error - v1 has one direction for the knight and that is fine.

FRAMES are each job's `post/<stem>_<n>.png` in the order of n - the picked
frames, the ones `publish` ships and the editor saves - pasted into a cell of
`frame_size` (work/project.json, default config.CANVAS; `--size` overrides)
with transparent padding, bottom-centred when the frame is smaller. The cell
is the atlas's unit and the pivot is its bottom centre.

TIMING comes from picks.json through the very same `publish.resolve_timing`
(hand edit > what `pick` recorded > the template > uniform), in TICKS at the
animation's fps: a phased pick's weights, the strip template's holds, or 1 per
frame. The atlas carries `ticks` and `duration_ms` (= ticks * 1000 / fps) per
frame; the .tres carries `speed = fps` and each frame's `duration = ticks`,
which is Godot's own per-frame unit (a frame shows for duration / speed s).
So a 12 fps clip with weights [2, 3, 4, 3] plays 167 / 250 / 333 / 250 ms in
both. The loop flag is the phase plan's (`timing.loop`), else the studio's
row for the action (idle / walk / run loop), else off.

THE .tres is Godot 4's text format: one `ext_resource` per sheet with a path
RELATIVE to the .tres (Godot resolves it against the resource's own folder,
so the export directory drops into any project as one unit), one
`AtlasTexture` sub-resource per frame, one animation per `<action>_<dir>`.
`parse_tres` reads it back, and the stage checks its own output against the
atlas before it returns - a .tres the game cannot load is the one thing this
stage must never leave behind.

Files are written IN PLACE (anim.util.write_bytes_in_place) and stale sheets
from an earlier export are removed, for the same inode reason `publish` does
both. `git rev-parse HEAD` of this checkout goes into `tool.commit` so a sheet
can be traced to the code that packed it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from anim import __version__, config, palettekit
from anim.api import settings as project_settings
from anim.api.jobs import post_frames
from anim.job import Job
from anim.stages import publish as publish_stage
from anim.util import read_json, write_bytes_in_place, write_text_in_place

NAME = "export"
HELP = "a unit's frames -> sheets + atlas.json + Godot SpriteFrames (.tres)"

ATLAS_SCHEMA = "anim.atlas/1"
EXPORT_DIRNAME = "_export"
SHEETS_DIRNAME = "sheets"
ADAPTERS = ("generic", "game02")
SHEET_MODES = ("per-direction", "packed")
PIVOT = "bottom-center"
TOOL = "tools/anim-pipeline (anim %s) - the spritesheet export feature" % __version__
UNIT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
REPO = Path(__file__).resolve().parents[2]
PREVIEW_SCALE = 4


# ---------------------------------------------------------------------------
# Finding the unit's jobs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Animation:
    """One (action, direction) cell of the unit, ready to pack. Pure data."""

    action: str
    direction: str
    job: str
    frames: tuple[Path, ...]
    ticks: tuple[float, ...]
    fps: int
    loop: bool
    phases: tuple[str | None, ...]
    anchor: int | None
    timing_source: str
    picks: tuple[int, ...]
    provenance: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.action}_{self.direction}"

    @property
    def durations_ms(self) -> list[int]:
        return [int(round(t * 1000 / self.fps)) for t in self.ticks]


def job_name_for(unit: str, action: str, direction: str) -> str:
    """The matrix job name for a cell: `<unit>_<action>` in the default
    direction, `<unit>_<action>_<dir>` elsewhere (anim.stages.rig)."""
    stem = f"{unit}_{action}"
    return stem if direction == config.DEFAULT_DIRECTION else f"{stem}_{direction}"


def default_loop(action: str) -> bool:
    """The studio's loop flag for an action when the pick recorded none."""
    studio = config.studio_action(action)
    if studio and studio in config.ACTION_PHASES:
        return bool(config.ACTION_PHASES[studio]["loop"])
    return False


def plan_timing(action: str, count: int, picks_record: dict[str, Any] | None
                ) -> dict[str, Any]:
    """Ticks, fps, loop, phases and anchor for `count` frames - pure.

    `publish.resolve_timing` decides the shape exactly as the sidecar does,
    so the atlas, the .tres and game-02's sidecar never disagree on a hold.
    """
    block, source = publish_stage.resolve_timing(action, count, picks_record)
    block = block or {}
    fps = int(block.get("fps") or config.PHASE_FPS)
    weights = block.get("weights")
    ticks = [float(w) for w in weights] if weights else [1.0] * count
    phases = block.get("phases")
    if not (isinstance(phases, list) and len(phases) == count):
        phases = [None] * count
    loop = bool(block["loop"]) if "loop" in block else default_loop(action)
    anchor = block.get("anchor")
    return {"ticks": ticks, "fps": fps, "loop": loop, "phases": list(phases),
            "anchor": int(anchor) if anchor is not None else None, "source": source}


def _provenance(manifest: dict[str, Any], report: dict[str, Any] | None,
                job: Job) -> dict[str, Any]:
    route = (report or {}).get("route") or "strip"
    path = job.clip_provenance_path if route == "clip" else job.provenance_path
    provenance = read_json(path) if path.exists() else {}
    motion = manifest.get("motion_source") or None
    return {
        "route": route,
        "seed": provenance.get("seed"),
        "clip": Path(manifest.get("clip", "")).name or None,
        "clip_sha256": manifest.get("clip_sha256"),
        "motion_source": ({k: motion.get(k) for k in ("kind", "pack", "clip", "license", "url")}
                          if motion else None),
    }


def find_animations(unit: str, work_dir: Path | None = None
                    ) -> tuple[list[Animation], list[dict[str, Any]]]:
    """Every exportable (action, direction) of the unit, and the warnings.

    Scans `work/<unit>_*`; a directory is a cell when it has a job.json whose
    slug, action and direction spell that exact name. Anything else under the
    prefix is either not an animation (a base or concept job, the export dir
    itself) or is listed as skipped with the reason.
    """
    work_dir = Path(work_dir) if work_dir is not None else config.WORK_DIR
    if not UNIT_RE.match(unit):
        raise ValueError(f"{unit!r} is not a unit slug")
    animations: list[Animation] = []
    warnings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in sorted(work_dir.glob(f"{unit}_*")):
        if not entry.is_dir() or not (entry / "job.json").exists():
            continue
        job = Job.open(entry.name, work_dir)
        try:
            manifest = job.read_manifest()
        except (OSError, ValueError) as exc:
            warnings.append({"kind": "skipped", "job": job.name,
                             "reason": f"job.json unreadable: {exc}"})
            continue
        action = str(manifest.get("action") or "").strip().lower()
        if not action:
            continue  # a base / concept job: a still, not an animation
        if str(manifest.get("slug") or "") != unit:
            continue  # another unit whose slug happens to start with this one
        direction = str((manifest.get("motion_source") or {}).get("direction")
                        or config.DEFAULT_DIRECTION)
        expected = job_name_for(unit, action, direction)
        if job.name != expected and job.name != f"{unit}_{action}_{direction}":
            warnings.append({"kind": "skipped", "job": job.name,
                             "reason": f"not the matrix job for {action} {direction} "
                                       f"(that is {expected})"})
            continue
        frames = post_frames(job)
        if not frames:
            warnings.append({"kind": "skipped", "job": job.name,
                             "reason": "no finished frames in post/ - run `anim post`"})
            continue
        report = read_json(job.report_path) if job.report_path.exists() else None
        if ((report or {}).get("palette") or {}).get("snap") is False:
            warnings.append({"kind": "skipped", "job": job.name,
                             "reason": "posted with --no-snap (a raw-colour preview)"})
            continue
        if (action, direction) in seen:
            warnings.append({"kind": "skipped", "job": job.name,
                             "reason": f"{action} {direction} already exported"})
            continue
        seen.add((action, direction))
        picks_record = read_json(job.picks_path) if job.picks_path.exists() else None
        timing = plan_timing(action, len(frames), picks_record)
        animations.append(Animation(
            action=action, direction=direction, job=job.name,
            frames=tuple(frames), ticks=tuple(timing["ticks"]), fps=timing["fps"],
            loop=timing["loop"], phases=tuple(timing["phases"]),
            anchor=timing["anchor"], timing_source=timing["source"],
            picks=tuple(int(p) for p in (picks_record or {}).get("picks") or []),
            provenance=_provenance(manifest, report, job)))
    warnings.extend(missing_cells(unit, animations))
    animations.sort(key=lambda a: (_direction_rank(a.direction), a.action))
    return animations, warnings


def _direction_rank(direction: str) -> int:
    names = list(config.DIRECTIONS)
    return names.index(direction) if direction in names else len(names)


def missing_cells(unit: str, animations: list[Animation]) -> list[dict[str, Any]]:
    """One warning per (action, direction) the matrix expects and has no job
    for: the studio's five actions plus any the unit has, across the eight
    side cells plus any the unit has."""
    have = {(a.action, a.direction) for a in animations}
    actions = sorted({row["verb"] for row in config.STUDIO_ACTIONS.values()}
                     | {a.action for a in animations})
    directions = [d for d in config.DIRECTIONS
                  if config.DIRECTIONS[d][1] == config.DEFAULT_ELEVATION
                  or any(a.direction == d for a in animations)]
    return [{"kind": "missing", "action": action, "direction": direction,
             "job": job_name_for(unit, action, direction)}
            for action in actions for direction in directions
            if (action, direction) not in have]


# ---------------------------------------------------------------------------
# Rect math - the two sheet layouts, pure
# ---------------------------------------------------------------------------
def layout(unit: str, animations: list[Animation], cell: tuple[int, int],
           mode: str = "per-direction") -> dict[str, Any]:
    """Where every frame lands: `{"sheets": {file: {"size": [w, h], ...}},
    "rects": {anim_key: {"sheet": file, "frames": [[x, y, w, h], ...]}}}`.

    per-direction: one sheet per direction, one row per action (alphabetical),
    frames left to right, the sheet as wide as its longest row. packed: one
    sheet, every frame in animation order, row-major, about as many columns
    as rows.
    """
    if mode not in SHEET_MODES:
        raise ValueError(f"sheet mode must be one of {list(SHEET_MODES)}, not {mode!r}")
    cw, ch = int(cell[0]), int(cell[1])
    sheets: dict[str, dict[str, Any]] = {}
    rects: dict[str, dict[str, Any]] = {}
    if mode == "per-direction":
        by_direction: dict[str, list[Animation]] = {}
        for anim in animations:
            by_direction.setdefault(anim.direction, []).append(anim)
        for direction, rows in by_direction.items():
            rows = sorted(rows, key=lambda a: a.action)
            file = f"{SHEETS_DIRNAME}/{unit}_{direction}.png"
            width = max(len(a.frames) for a in rows) * cw
            sheets[file] = {"size": [width, len(rows) * ch], "direction": direction,
                            "rows": [a.action for a in rows]}
            for r, anim in enumerate(rows):
                rects[anim.key] = {"sheet": file,
                                   "frames": [[i * cw, r * ch, cw, ch]
                                              for i in range(len(anim.frames))]}
        return {"mode": mode, "cell": [cw, ch], "sheets": sheets, "rects": rects}
    total = sum(len(a.frames) for a in animations)
    columns = max(1, int(math.ceil(math.sqrt(total)))) if total else 1
    row_count = max(1, int(math.ceil(total / columns))) if total else 1
    file = f"{SHEETS_DIRNAME}/{unit}.png"
    sheets[file] = {"size": [columns * cw, row_count * ch], "columns": columns}
    k = 0
    for anim in animations:
        frames = []
        for _ in anim.frames:
            frames.append([(k % columns) * cw, (k // columns) * ch, cw, ch])
            k += 1
        rects[anim.key] = {"sheet": file, "frames": frames}
    return {"mode": mode, "cell": [cw, ch], "sheets": sheets, "rects": rects}


def paste_cell(sheet: Image.Image, frame: Image.Image, rect: list[int]) -> None:
    """The frame into its cell: at the origin when it fills it, else bottom-
    centred with transparent padding. A frame larger than the cell is refused
    - the cell IS the contract, cropping would hide that silently."""
    x, y, w, h = rect
    if frame.width > w or frame.height > h:
        raise ValueError(f"a {frame.width}x{frame.height} frame does not fit a "
                         f"{w}x{h} cell; pass --size at least that, or re-post")
    dx = (w - frame.width) // 2
    dy = h - frame.height
    sheet.alpha_composite(frame.convert("RGBA"), (x + dx, y + dy))


def render_sheets(animations: list[Animation], plan: dict[str, Any]
                  ) -> dict[str, Image.Image]:
    """The sheet images, transparent where no frame is."""
    images = {file: Image.new("RGBA", tuple(meta["size"]), (0, 0, 0, 0))
              for file, meta in plan["sheets"].items()}
    for anim in animations:
        rect_row = plan["rects"][anim.key]
        sheet = images[rect_row["sheet"]]
        for path, rect in zip(anim.frames, rect_row["frames"]):
            with Image.open(path) as frame:
                frame.load()
                paste_cell(sheet, frame, rect)
    return images


# ---------------------------------------------------------------------------
# The atlas
# ---------------------------------------------------------------------------
def git_commit() -> str | None:
    """This checkout's HEAD, for `tool.commit`; None outside a checkout."""
    try:
        proc = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def atlas_for(unit: str, animations: list[Animation], plan: dict[str, Any], *,
              palette: str | None, warnings: list[dict[str, Any]],
              commit: str | None = None, created: str | None = None) -> dict[str, Any]:
    """The atlas.json document. Pure - takes the plan, returns the dict."""
    cw, ch = plan["cell"]
    rows: dict[str, dict[str, Any]] = {}
    for anim in animations:
        rect_row = plan["rects"][anim.key]
        frames = []
        for i, (x, y, w, h) in enumerate(rect_row["frames"]):
            frames.append({"x": x, "y": y, "w": w, "h": h,
                           "phase": anim.phases[i], "ticks": anim.ticks[i],
                           "duration_ms": anim.durations_ms[i]})
        rows[anim.key] = {
            "action": anim.action,
            "direction": anim.direction,
            "sheet": rect_row["sheet"],
            "loop": anim.loop,
            "fps": anim.fps,
            "pivot": [cw // 2, ch],
            "anchor": anim.anchor,
            "frames": frames,
            "source": {"job": anim.job, "picks": list(anim.picks),
                       "timing_source": anim.timing_source, **anim.provenance},
        }
    return {
        "schema": ATLAS_SCHEMA,
        "unit": unit,
        "frame_size": [cw, ch],
        "fps": config.PHASE_FPS,
        "pivot": PIVOT,
        "palette": palette,
        "sheet_mode": plan["mode"],
        "sheets": {file: dict(meta) for file, meta in plan["sheets"].items()},
        "animations": rows,
        "tool": {"name": TOOL, "version": __version__, "commit": commit,
                 "created": created or dt.datetime.now(dt.timezone.utc).isoformat(
                     timespec="seconds")},
        "warnings": list(warnings),
    }


# ---------------------------------------------------------------------------
# The Godot SpriteFrames resource - written, and read back
# ---------------------------------------------------------------------------
def _ident(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def _num(value: float) -> str:
    """Godot writes floats with a decimal point; ticks are usually whole."""
    value = float(value)
    return str(int(value)) + ".0" if value.is_integer() else repr(round(value, 6))


def tres_for(unit: str, atlas: dict[str, Any]) -> str:
    """The SpriteFrames .tres text for an atlas: one ext_resource per sheet,
    one AtlasTexture per frame, one animation per `<action>_<dir>`."""
    sheets = list(atlas["sheets"])
    ext_ids = {file: f"{i + 1}_{_ident(Path(file).stem)}" for i, file in enumerate(sheets)}
    subs: list[str] = []
    anims: list[str] = []
    for key, row in atlas["animations"].items():
        frame_lines = []
        for i, frame in enumerate(row["frames"]):
            sub_id = f"AtlasTexture_{_ident(key)}_{i}"
            subs.append(f'[sub_resource type="AtlasTexture" id="{sub_id}"]\n'
                        f'atlas = ExtResource("{ext_ids[row["sheet"]]}")\n'
                        f'region = Rect2({frame["x"]}, {frame["y"]}, {frame["w"]}, {frame["h"]})\n')
            frame_lines.append(f'{{\n"duration": {_num(frame["ticks"])},\n'
                               f'"texture": SubResource("{sub_id}")\n}}')
        anims.append('{\n"frames": [' + ", ".join(frame_lines) + '],\n'
                     f'"loop": {"true" if row["loop"] else "false"},\n'
                     f'"name": &"{key}",\n'
                     f'"speed": {_num(row["fps"])}\n}}')
    steps = len(sheets) + len(subs) + 1
    out = [f'[gd_resource type="SpriteFrames" load_steps={steps} format=3]\n']
    for file in sheets:
        out.append(f'[ext_resource type="Texture2D" path="{file}" id="{ext_ids[file]}"]\n')
    out.extend(subs)
    out.append("[resource]\nanimations = [" + ", ".join(anims) + "]\n")
    return "\n".join(out)


_SECTION = re.compile(r'^\[(?P<kind>\w+)(?P<attrs>(?:\s+\w+=(?:"[^"]*"|\S+))*)\s*\]$')
_ATTR = re.compile(r'(\w+)=("([^"]*)"|(\S+))')
_KEY = re.compile(r"^(\w+) = ", re.M)


def _godot_value(text: str) -> Any:
    """A Godot text-resource literal to Python: the few constructs a
    SpriteFrames uses, rewritten into JSON and parsed."""
    text = text.strip()
    text = re.sub(r'&"([^"]*)"', r'"\1"', text)
    text = re.sub(r'SubResource\("([^"]*)"\)', r'{"$sub": "\1"}', text)
    text = re.sub(r'ExtResource\("([^"]*)"\)', r'{"$ext": "\1"}', text)
    text = re.sub(r"Rect2\(([^)]*)\)", r"[\1]", text)
    return json.loads(text)


def parse_tres(text: str) -> dict[str, Any]:
    """Read a SpriteFrames .tres back: `{"type", "ext": {id: {...}},
    "sub": {id: {...}}, "animations": [...]}`. Small on purpose - it reads
    what `tres_for` writes (and what Godot writes for the same resource),
    not the whole text-resource grammar."""
    result: dict[str, Any] = {"type": None, "ext": {}, "sub": {}, "animations": []}
    current: dict[str, Any] | None = None
    body: list[str] = []

    def flush() -> None:
        if current is None:
            return
        block = "\n".join(body)
        keys = list(_KEY.finditer(block))
        fields: dict[str, Any] = {}
        for i, match in enumerate(keys):
            end = keys[i + 1].start() if i + 1 < len(keys) else len(block)
            fields[match.group(1)] = _godot_value(block[match.end():end])
        kind = current["kind"]
        if kind == "gd_resource":
            result["type"] = current["attrs"].get("type")
        elif kind == "ext_resource":
            result["ext"][current["attrs"]["id"]] = {**current["attrs"], **fields}
        elif kind == "sub_resource":
            result["sub"][current["attrs"]["id"]] = {**current["attrs"], **fields}
        elif kind == "resource":
            result["animations"] = fields.get("animations", [])

    for line in text.splitlines():
        match = _SECTION.match(line)
        if match:
            flush()
            attrs = {m.group(1): (m.group(3) if m.group(3) is not None else m.group(4))
                     for m in _ATTR.finditer(match.group("attrs"))}
            current = {"kind": match.group("kind"), "attrs": attrs}
            body = []
        else:
            body.append(line)
    flush()
    return result


def verify_tres(text: str, atlas: dict[str, Any]) -> list[str]:
    """Every way the .tres could disagree with the atlas, as messages; empty
    when it is the same document in Godot's words."""
    problems: list[str] = []
    try:
        parsed = parse_tres(text)
    except (ValueError, KeyError) as exc:
        return [f"unparseable: {exc}"]
    if parsed["type"] != "SpriteFrames":
        problems.append(f"type is {parsed['type']!r}")
    paths = {row["path"] for row in parsed["ext"].values()}
    if paths != set(atlas["sheets"]):
        problems.append(f"sheets {sorted(paths)} != {sorted(atlas['sheets'])}")
    by_name = {row["name"]: row for row in parsed["animations"]}
    if set(by_name) != set(atlas["animations"]):
        problems.append(f"animations {sorted(by_name)} != {sorted(atlas['animations'])}")
    for key, row in atlas["animations"].items():
        got = by_name.get(key)
        if got is None:
            continue
        if bool(got.get("loop")) != bool(row["loop"]):
            problems.append(f"{key}: loop {got.get('loop')} != {row['loop']}")
        if float(got.get("speed", 0)) != float(row["fps"]):
            problems.append(f"{key}: speed {got.get('speed')} != {row['fps']}")
        frames = got.get("frames") or []
        if len(frames) != len(row["frames"]):
            problems.append(f"{key}: {len(frames)} frame(s) != {len(row['frames'])}")
            continue
        for i, (a, b) in enumerate(zip(frames, row["frames"])):
            if abs(float(a["duration"]) - float(b["ticks"])) > 1e-6:
                problems.append(f"{key}[{i}]: duration {a['duration']} != {b['ticks']}")
            sub = parsed["sub"].get((a.get("texture") or {}).get("$sub"))
            if sub is None:
                problems.append(f"{key}[{i}]: texture is not a sub-resource")
                continue
            ext = parsed["ext"].get((sub.get("atlas") or {}).get("$ext"), {})
            if ext.get("path") != row["sheet"]:
                problems.append(f"{key}[{i}]: sheet {ext.get('path')} != {row['sheet']}")
            if [int(v) for v in sub.get("region", [])] != [b["x"], b["y"], b["w"], b["h"]]:
                problems.append(f"{key}[{i}]: region {sub.get('region')} != "
                                f"{[b['x'], b['y'], b['w'], b['h']]}")
    return problems


# ---------------------------------------------------------------------------
# The preview page
# ---------------------------------------------------------------------------
def preview_html(atlas: dict[str, Any]) -> str:
    """One static page: each sheet at x{PREVIEW_SCALE} nearest-neighbour with
    the atlas rects drawn over it, and every animation playing at its own
    durations. The atlas is inlined so the page works from a plain file
    server with no fetch."""
    payload = json.dumps(atlas).replace("</", "<\\/")
    return PREVIEW_TEMPLATE.replace("__ATLAS__", payload).replace(
        "__SCALE__", str(PREVIEW_SCALE)).replace("__UNIT__", atlas["unit"])


PREVIEW_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__UNIT__ export</title>
<style>
  :root { --bg: #1b1d22; --fg: #e8e6e3; --dim: #9a9791; --line: #3a3d45; --accent: #f2c14e; }
  body { margin: 0; padding: 24px 16px; background: var(--bg); color: var(--fg);
         font: 14px/1.45 system-ui, sans-serif; }
  h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 28px 0 8px; }
  .meta { color: var(--dim); margin-bottom: 8px; }
  .sheet { position: relative; display: inline-block; margin: 8px 16px 8px 0; vertical-align: top;
           background: repeating-conic-gradient(#2a2d33 0 25%, #23262b 0 50%) 0 0 / 16px 16px; }
  .sheet img { display: block; image-rendering: pixelated; image-rendering: crisp-edges; }
  .sheet svg { position: absolute; inset: 0; }
  .sheet rect { fill: none; stroke: var(--accent); stroke-width: 1; opacity: .8; }
  .sheet text { fill: var(--accent); font: 10px monospace; }
  .anims { display: flex; flex-wrap: wrap; gap: 16px; }
  .anim { border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
  .anim canvas { display: block; image-rendering: pixelated;
                 background: repeating-conic-gradient(#2a2d33 0 25%, #23262b 0 50%) 0 0 / 16px 16px; }
  .anim .name { font-weight: 600; margin-bottom: 4px; }
  .anim .info { color: var(--dim); font-size: 12px; }
  ul.warn { color: var(--dim); font-size: 12px; columns: 2; }
  code { color: var(--accent); }
</style>
</head>
<body>
<h1>__UNIT__ — spritesheet export</h1>
<div class="meta" id="meta"></div>
<h2>Sheets (×__SCALE__, atlas rects drawn over)</h2>
<div id="sheets"></div>
<h2>Animations (each at its own per-frame durations)</h2>
<div class="anims" id="anims"></div>
<h2>Warnings</h2>
<ul class="warn" id="warnings"></ul>
<script>
const atlas = __ATLAS__;
const S = __SCALE__;
document.getElementById('meta').textContent =
  `frame ${atlas.frame_size[0]}×${atlas.frame_size[1]} · ${atlas.fps} fps · palette ${atlas.palette ?? '(none)'} ` +
  `· ${atlas.sheet_mode} · ${Object.keys(atlas.animations).length} animation(s) · ` +
  `tool ${atlas.tool.version}${atlas.tool.commit ? ' @ ' + atlas.tool.commit : ''} · ${atlas.tool.created}`;
const sheetsEl = document.getElementById('sheets');
const images = {};
for (const [file, meta] of Object.entries(atlas.sheets)) {
  const box = document.createElement('div'); box.className = 'sheet';
  const img = new Image(); img.src = file; img.width = meta.size[0] * S; img.height = meta.size[1] * S;
  images[file] = img;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', meta.size[0] * S); svg.setAttribute('height', meta.size[1] * S);
  for (const [key, row] of Object.entries(atlas.animations)) {
    if (row.sheet !== file) continue;
    row.frames.forEach((f, i) => {
      const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      r.setAttribute('x', f.x * S + .5); r.setAttribute('y', f.y * S + .5);
      r.setAttribute('width', f.w * S - 1); r.setAttribute('height', f.h * S - 1);
      svg.appendChild(r);
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('x', f.x * S + 3); t.setAttribute('y', f.y * S + 11);
      t.textContent = i === 0 ? `${key} ${i}` : `${i}`;
      svg.appendChild(t);
    });
  }
  const cap = document.createElement('div'); cap.className = 'meta';
  cap.textContent = `${file} — ${meta.size[0]}×${meta.size[1]}`;
  box.appendChild(img); box.appendChild(svg);
  sheetsEl.appendChild(box); sheetsEl.appendChild(cap);
}
const animsEl = document.getElementById('anims');
for (const [key, row] of Object.entries(atlas.animations)) {
  const card = document.createElement('div'); card.className = 'anim';
  const name = document.createElement('div'); name.className = 'name'; name.textContent = key;
  const canvas = document.createElement('canvas');
  canvas.width = atlas.frame_size[0] * S; canvas.height = atlas.frame_size[1] * S;
  const info = document.createElement('div'); info.className = 'info';
  const ms = row.frames.map(f => f.duration_ms);
  info.innerHTML = `${row.frames.length} frames · ${row.loop ? 'loop' : 'one-shot'} · ${row.fps} fps<br>` +
    `${ms.join(' / ')} ms · <code>${row.source.job}</code> · ${row.source.timing_source}`;
  card.appendChild(name); card.appendChild(canvas); card.appendChild(info);
  animsEl.appendChild(card);
  const ctx = canvas.getContext('2d'); ctx.imageSmoothingEnabled = false;
  const img = images[row.sheet];
  let i = 0;
  const step = () => {
    const f = row.frames[i];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img.complete) ctx.drawImage(img, f.x, f.y, f.w, f.h, 0, 0, f.w * S, f.h * S);
    const last = i === row.frames.length - 1;
    i = (i + 1) % row.frames.length;
    setTimeout(step, f.duration_ms + (last && !row.loop ? 500 : 0));
  };
  img.addEventListener('load', step, { once: true });
  if (img.complete) step();
}
const warnEl = document.getElementById('warnings');
for (const w of atlas.warnings) {
  const li = document.createElement('li');
  li.textContent = w.kind === 'missing' ? `missing ${w.action} ${w.direction} (${w.job})`
                                        : `skipped ${w.job}: ${w.reason}`;
  warnEl.appendChild(li);
}
if (!atlas.warnings.length) warnEl.innerHTML = '<li>none</li>';
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def export_dir(unit: str, work_dir: Path | None = None) -> Path:
    """`work/_export/<unit>/` - an underscore directory, so the API's job
    listing never mistakes it for a job (anim.api.jobs.valid_name)."""
    base = Path(work_dir) if work_dir is not None else config.WORK_DIR
    return base / EXPORT_DIRNAME / unit


def parse_size(value: str | int | None, default: tuple[int, int]) -> tuple[int, int]:
    """`--size 64` or `--size 64x96` -> (w, h); None -> the default."""
    if value is None:
        return (int(default[0]), int(default[1]))
    text = str(value).strip().lower()
    parts = text.split("x") if "x" in text else [text, text]
    try:
        w, h = (int(p) for p in parts)
    except ValueError:
        raise ValueError(f"--size takes N or WxH, not {value!r}") from None
    if w < 1 or h < 1:
        raise ValueError(f"--size must be positive, not {value!r}")
    return (w, h)


def listing(unit: str, work_dir: Path | None = None) -> dict[str, Any] | None:
    """`GET /export/<unit>`: what the last export wrote, or None when there
    has been none. The atlas is the record; the files are listed beside it."""
    out_dir = export_dir(unit, work_dir)
    atlas_path = out_dir / "atlas.json"
    if not atlas_path.exists():
        return None
    atlas = read_json(atlas_path)
    files = [{"file": str(p.relative_to(out_dir)), "bytes": p.stat().st_size}
             for p in sorted(out_dir.rglob("*")) if p.is_file()]
    return {
        "unit": unit,
        "dir": str(out_dir),
        "created": (atlas.get("tool") or {}).get("created"),
        "commit": (atlas.get("tool") or {}).get("commit"),
        "frame_size": atlas.get("frame_size"),
        "sheet_mode": atlas.get("sheet_mode"),
        "palette": atlas.get("palette"),
        "sheets": atlas.get("sheets"),
        "animations": {key: {"frames": len(row["frames"]), "loop": row["loop"],
                             "fps": row["fps"], "sheet": row["sheet"],
                             "job": row["source"]["job"]}
                       for key, row in (atlas.get("animations") or {}).items()},
        "warnings": atlas.get("warnings") or [],
        "adapter": atlas.get("adapter"),
        "files": files,
    }


def export(unit: str, *, out_dir: Path | None = None, adapter: str | None = None,
           sheet: str = "per-direction", size: str | int | None = None,
           work_dir: Path | None = None, assets_dir: Path | None = None,
           run_audit: bool = True, audit_script: Path | None = None,
           dry_run: bool = False) -> dict[str, Any]:
    work_dir = Path(work_dir) if work_dir is not None else config.WORK_DIR
    unit = unit.strip().lower()
    stored = project_settings.stored(work_dir)
    adapter = adapter or str((stored.get("export") or {}).get("adapter") or "generic")
    if adapter not in ADAPTERS:
        raise ValueError(f"adapter must be one of {list(ADAPTERS)}, not {adapter!r}")
    cell = parse_size(size, tuple(stored.get("frame_size") or config.CANVAS))
    out_dir = Path(out_dir) if out_dir is not None else export_dir(unit, work_dir)

    animations, warnings = find_animations(unit, work_dir)
    if not animations:
        raise FileNotFoundError(
            f"no exportable animation for {unit!r} under {work_dir}: a job named "
            f"{unit}_<action>[_<dir>] with post/ frames (`anim post`) is needed")
    plan = layout(unit, animations, cell, sheet)
    palette = palettekit.load(project_settings.palette_path(work_dir))
    atlas = atlas_for(unit, animations, plan,
                      palette=palette["name"] if palette else None,
                      warnings=warnings, commit=git_commit())
    if adapter != "generic":
        atlas["adapter"] = adapter
    tres = tres_for(unit, atlas)
    problems = verify_tres(tres, atlas)
    if problems:
        raise RuntimeError(f"the .tres does not read back as the atlas:\n  "
                           + "\n  ".join(problems))

    result: dict[str, Any] = {
        "unit": unit, "dir": str(out_dir), "adapter": adapter, "sheet_mode": sheet,
        "frame_size": list(cell),
        "animations": [a.key for a in animations],
        "sheets": list(plan["sheets"]),
        "files": [*plan["sheets"], "atlas.json", f"{unit}.tres", "preview.html"],
        "warnings": warnings, "removed": [], "published": [], "audit": None,
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    images = render_sheets(animations, plan)
    sheets_dir = out_dir / SHEETS_DIRNAME
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for file, image in images.items():
        buffer = _png_bytes(image)
        write_bytes_in_place(out_dir / file, buffer)
    for stale in sorted(sheets_dir.glob("*.png")):
        if f"{SHEETS_DIRNAME}/{stale.name}" not in images:
            stale.unlink()
            result["removed"].append(f"{SHEETS_DIRNAME}/{stale.name}")
    write_text_in_place(out_dir / f"{unit}.tres", tres)
    write_text_in_place(out_dir / "preview.html", preview_html(atlas))

    if adapter == "game02":
        for anim in animations:
            row = publish_stage.publish(anim.job, assets_dir=assets_dir, run_audit=False,
                                        audit_script=audit_script, work_dir=work_dir)
            result["published"].append(row["stem"])
        if run_audit:
            code, output = publish_stage.audit(audit_script)
            result["audit"] = {"exit": code, "output": output.strip()}
            if code != 0:
                raise RuntimeError(
                    f"asset_audit.py exited {code} after publishing "
                    f"{', '.join(result['published'])}:\n{output.strip()}")
        atlas["adapter_result"] = {"published": list(result["published"]),
                                   "audit_exit": (result["audit"] or {}).get("exit")}
    # The atlas last: it names everything else, so its presence means the
    # export is whole (`listing` keys off it).
    write_text_in_place(out_dir / "atlas.json", json.dumps(atlas, indent=2) + "\n")
    return result


def _png_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("unit", help="unit slug, e.g. knight")
    parser.add_argument("--out", default=None,
                        help="output directory (default work/_export/<unit>)")
    parser.add_argument("--adapter", default=None, choices=ADAPTERS,
                        help="generic: the three files; game02: also run `publish` "
                             "per job (default: export.adapter in work/project.json)")
    parser.add_argument("--sheet", default="per-direction", choices=SHEET_MODES,
                        help="per-direction: one sheet per direction, one row per "
                             "action; packed: one row-major sheet (default per-direction)")
    parser.add_argument("--size", default=None, metavar="N|WxH",
                        help="the cell (default: frame_size in work/project.json, "
                             f"else {config.CANVAS[0]}x{config.CANVAS[1]})")
    parser.add_argument("--assets", default=None,
                        help=f"game02 adapter: assets root (default {config.ASSETS_DIR})")
    parser.add_argument("--no-audit", dest="audit", action="store_false",
                        help="game02 adapter: skip the asset audit")
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would be written and stop")
    parser.set_defaults(audit=True)


def run_stage(args: argparse.Namespace) -> int:
    result = export(args.unit, out_dir=Path(args.out) if args.out else None,
                    adapter=args.adapter, sheet=args.sheet, size=args.size,
                    assets_dir=Path(args.assets) if args.assets else None,
                    run_audit=args.audit, dry_run=args.dry_run)
    verb = "would write" if result["dry_run"] else "wrote"
    print(f"export: {verb} {len(result['animations'])} animation(s) of {result['unit']} "
          f"to {result['dir']} ({result['sheet_mode']}, cell "
          f"{result['frame_size'][0]}x{result['frame_size'][1]}, adapter {result['adapter']})")
    for name in result["files"]:
        print(f"  {name}")
    for name in result["removed"]:
        print(f"  removed stale {name}")
    for stem in result["published"]:
        print(f"  published {stem} (game02)")
    if result["audit"]:
        print(f"  asset_audit.py: {result['audit']['output'].splitlines()[-1]}")
    skipped = [w for w in result["warnings"] if w["kind"] == "skipped"]
    missing = [w for w in result["warnings"] if w["kind"] == "missing"]
    for w in skipped:
        print(f"  skipped {w['job']}: {w['reason']}", file=sys.stderr)
    by_action: dict[str, list[str]] = {}
    for w in missing:
        by_action.setdefault(w["action"], []).append(w["direction"])
    for action, directions in by_action.items():
        print(f"  missing {action}: {' '.join(directions)} ({len(directions)} cell(s))",
              file=sys.stderr)
    return 0


run = run_stage
