"""`python -m anim.tools.direction_sheet` - the camera direction sheet (#288)

The manual test for the camera direction feature (`--yaw` / `--elevation` on
`anim rig` and `anim base`) says the owner runs ONE command and looks at ONE
page. This module is that command:

    cd tools/anim-pipeline
    docker compose run --rm anim-pipeline \\
        python -m anim.tools.direction_sheet knight Idle

It renders frame 0 (the rest pose) of the named clip in every cell of
`config.DIRECTIONS` - 8 side facings + 8 top-down facings, skeleton and depth
each - plus the legacy `--mirror` render next to `s180`, and writes one static
HTML contact sheet naming every cell, so the five pass criteria in #288 can be
checked at a glance instead of by running `anim rig` sixteen times by hand:

    1. s0 faces right
    2. s180 == mirror
    3. the side row turns anticlockwise seen from above, s90 = back, s270 =
       facing the viewer
    4. the top row makes the same turn, head over hips
    5. the same floor row and size in every cell
    6. the top depth channel is one solid body, with the sword/shield near or
       far by yaw

Every cell goes through the exact functions `anim rig` and `anim base` call -
`rigkit.reference_frame`, `rigkit.clip_keypoints`, `rigkit.clip_depth`,
`posekit.draw` - never a second projection (see `anim.rigkit`, DIRECTION, and
`anim.stages.rig` / `anim.stages.base`). Nothing is written under `work/<job>/`
- there is no job here, just the sixteen cells' pixels and one page - so
nothing this tool does can collide with a real job name.

Output. The `anim-pipeline` container has no mount to
`~/.cache/claude/visuals` (unlike `/assets` or `/game02`, that host path is
not one of `compose.yml`'s volumes), so the sheet is written under
`work/sheets/camera-direction/` in the repo - which IS mounted (`.:/work`) -
and the module prints the one `cp -r` line that puts it where the owner's
browser looks:

    http://192.168.1.158:8090/sprite-studio/camera-direction/
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anim import config, posekit, rigkit
from anim.stages import rig as rig_stage

REPO = Path(__file__).resolve().parents[2]

# No unit -> rig lookup exists anywhere in the pipeline today (every stage
# takes the GLB path on the command line); this is the one this tool needs,
# kept local rather than added to anim/config.py so this feature stays a
# self-contained new module. The KayKit Adventurers Knight, not the Quaternius
# UAL rig: both bodies are skinned (README, "the rig props feature"), but the
# manual test's own example clip - `Idle` - is a KayKit clip name; the
# Quaternius rig calls its equivalent `Idle_Loop` / `Sword_Idle` and has none
# named exactly `Idle` (`anim rig <ual glb> --list`).
DEFAULT_RIG_FOR_UNIT: dict[str, Path] = {
    "knight": REPO / "rigs" / "kaykit-adventurers" / "Knight.glb",
}

OUT_DIR_NAME = "camera-direction"
VISUALS_TARGET = "~/.cache/claude/visuals/sprite-studio/camera-direction"
VISUALS_URL = "http://192.168.1.158:8090/sprite-studio/camera-direction/"

# The README's yaw table (the camera direction feature), split by elevation.
SIDE_FACING: dict[int, str] = {
    0: "screen right — today's default profile", 45: "away-right",
    90: "away (the back)", 135: "away-left", 180: "screen left — --mirror",
    225: "toward-left", 270: "the viewer", 315: "toward-right",
}
TOP_FACING: dict[int, str] = {
    0: "east", 45: "north-east", 90: "north (up-screen)", 135: "north-west",
    180: "west", 225: "south-west", 270: "south (down-screen)", 315: "south-east",
}

PASS_CRITERIA = (
    "s0 faces right",
    "s180 == mirror",
    "the side row turns anticlockwise seen from above, s90 = back, "
    "s270 = facing the viewer",
    "the top row makes the same turn, head over hips",
    "the same floor row and size in every cell",
    "the top depth channel is one solid body, with the sword/shield near or "
    "far by yaw",
)


def rig_for_unit(unit: str, rig: Path | str | None) -> Path:
    """`--rig`, or `DEFAULT_RIG_FOR_UNIT[unit]` - a clear error either way."""
    if rig is not None:
        return Path(rig)
    key = unit.strip().lower()
    if key not in DEFAULT_RIG_FOR_UNIT:
        raise SystemExit(
            f"anim.tools.direction_sheet: no default rig for unit {unit!r}; "
            f"pass --rig, or add it to DEFAULT_RIG_FOR_UNIT (known: "
            f"{sorted(DEFAULT_RIG_FOR_UNIT)})")
    return DEFAULT_RIG_FOR_UNIT[key]


def _placeholder_depth(cell: tuple[int, int]) -> Image.Image:
    """A flat frame for a rig with no skinned mesh (a bare animation library,
    or the KayKit Knight's static-helmet-only body) so every cell keeps the
    same two-channel shape instead of a missing file."""
    return Image.new("L", (int(cell[0]), int(cell[1])), rigkit.DEPTH_FAR_GREY // 2)


def render_cell(loaded: rigkit.Rig, clip: str, *, frame_index: int,
                cell: tuple[int, int], props: tuple[str, ...],
                yaw: int, elevation: str, mirror: bool = False,
                jmap: rigkit.JointMap | None = None
                ) -> tuple[Image.Image, Image.Image | None, dict[str, Any]]:
    """One cell's skeleton PNG, its depth PNG (`None` for an unskinned rig),
    and the `motion` block `anim.rigkit.clip_keypoints` records for it - via
    the same calls `anim.stages.rig.rig` / `anim.stages.base.rig_payload`
    make, at one sampled time instead of the whole clip."""
    frame = rigkit.reference_frame(loaded, jmap, mirror=mirror, props=props,
                                   yaw=yaw, elevation=elevation)
    clip_obj = loaded.clip(clip)
    times = rigkit.sample_times(clip_obj.duration, config.EXTRACT_FPS)
    index = min(max(frame_index, 0), len(times) - 1)
    payload = rigkit.clip_keypoints(loaded, clip, fps=config.EXTRACT_FPS,
                                    cell=cell, mirror=mirror, frame=frame,
                                    times=[float(times[index])])
    entry = payload["frames"][0]
    canvas = posekit.draw(entry["keypoints"], entry["scores"],
                          tuple(payload["source_size"]), cell,
                          kpt_thr=float(payload.get("kpt_thr", 0.4)),
                          props=entry.get("props"))
    skeleton = Image.fromarray(canvas[:, :, ::-1])  # BGR -> RGB, as anim base does

    depth_img: Image.Image | None = None
    if loaded.skinned:
        depth_clip = rigkit.clip_depth(loaded, clip, fps=config.EXTRACT_FPS,
                                       cell=cell, frame=frame,
                                       times=[float(times[index])], mode="full")
        grey = depth_clip.frames()[0]
        depth_img = Image.fromarray(np.asarray(grey, dtype=np.uint8), mode="L")

    return skeleton, depth_img, payload["motion"]


def _commit_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_sheet(unit: str, clip: str, *, rig: Path | str | None = None,
                frame_index: int = 0, cell: tuple[int, int] | None = None,
                out_dir: Path | None = None,
                jmap: rigkit.JointMap | None = None) -> Path:
    """Write the contact sheet and return the path to its `index.html`.

    `jmap` is for tests only: it lets a fixture rig with bones no
    `rigkit.JointMap` in `anim/rigkit.py` would ever detect stand in for a
    real one, without touching `rigkit.detect_map`.
    """
    glb = rig_for_unit(unit, rig)
    cell = tuple(cell or config.STRIP_CELL)
    props = config.props_for_unit(unit)
    loaded = rigkit.load_rig(glb)

    out_dir = Path(out_dir) if out_dir is not None \
        else config.WORK_DIR / "sheets" / OUT_DIR_NAME
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for stale in img_dir.glob("*.png"):
        stale.unlink()

    cells: list[dict[str, Any]] = []
    for name, (yaw, elevation) in config.DIRECTIONS.items():
        skeleton, depth, motion = render_cell(
            loaded, clip, frame_index=frame_index, cell=cell, props=props,
            yaw=yaw, elevation=elevation, jmap=jmap)
        skel_path = img_dir / f"{name}_skeleton.png"
        skeleton.save(skel_path)
        depth_path = img_dir / f"{name}_depth.png"
        (depth if depth is not None else _placeholder_depth(cell)).save(depth_path)
        cells.append({
            "name": name, "yaw": yaw, "elevation": elevation,
            "facing": (TOP_FACING if elevation == "top" else SIDE_FACING)[yaw],
            "skeleton": skel_path.relative_to(out_dir).as_posix(),
            "depth": depth_path.relative_to(out_dir).as_posix(),
            "has_depth": depth is not None,
            "motion": motion,
        })

    mirror_skel, _mirror_depth, mirror_motion = render_cell(
        loaded, clip, frame_index=frame_index, cell=cell, props=props,
        yaw=config.DEFAULT_YAW, elevation=config.DEFAULT_ELEVATION,
        mirror=True, jmap=jmap)
    mirror_path = img_dir / "mirror.png"
    mirror_skel.save(mirror_path)

    index_path = out_dir / "index.html"
    index_path.write_text(
        _render_html(unit=unit, clip=clip, glb=glb, frame_index=frame_index,
                    commit=_commit_hash(), cells=cells,
                    mirror_rel=mirror_path.relative_to(out_dir).as_posix(),
                    mirror_motion=mirror_motion),
        encoding="utf-8")
    return index_path


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
_STYLE = """
:root { color-scheme: dark; }
body { background: #14161b; color: #d8dee9; font: 14px/1.4 -apple-system,
       BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 24px; }
h1 { font-size: 18px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 32px 0 8px; color: #9fb0c9; }
.caption { color: #8b96a5; margin-bottom: 20px; }
.row { display: grid; grid-template-columns: repeat(8, 1fr); gap: 10px; }
.cell { background: #1c1f26; border: 1px solid #2a2f3a; border-radius: 6px;
        padding: 6px; }
.cell.default { border-color: #d6a94a; }
.cell img { width: 100%; image-rendering: pixelated; background: #000;
            display: block; }
.cell .name { font-weight: 600; color: #e8ecf1; }
.cell .facing { color: #8b96a5; font-size: 12px; margin-bottom: 4px; }
.chan { display: flex; gap: 4px; margin-top: 4px; }
.chan div { flex: 1; }
.chan span { display: block; font-size: 10px; color: #6b7686; text-align: center; }
.no-depth { color: #c47a4a; font-size: 10px; }
.mirror { display: flex; gap: 20px; align-items: flex-start; margin-top: 8px; }
.mirror figure { margin: 0; text-align: center; }
.mirror img { width: 240px; image-rendering: pixelated; background: #000; }
figcaption { color: #8b96a5; font-size: 12px; margin-top: 4px; }
ol.criteria { padding-left: 20px; }
ol.criteria li { margin-bottom: 4px; }
footer { margin-top: 32px; color: #6b7686; font-size: 12px; }
"""


def _cell_html(c: dict[str, Any]) -> str:
    cls = "cell default" if c["name"] == config.DEFAULT_DIRECTION else "cell"
    depth_note = "" if c["has_depth"] else '<div class="no-depth">no depth: unskinned rig</div>'
    return f"""
    <div class="{cls}">
      <div class="name">{c['name']}</div>
      <div class="facing">yaw {c['yaw']} — {c['facing']}</div>
      <div class="chan">
        <div><img src="{c['skeleton']}" alt="{c['name']} skeleton"><span>skeleton</span></div>
        <div><img src="{c['depth']}" alt="{c['name']} depth"><span>depth</span></div>
      </div>
      {depth_note}
    </div>"""


def _render_html(*, unit: str, clip: str, glb: Path, frame_index: int,
                 commit: str, cells: list[dict[str, Any]], mirror_rel: str,
                 mirror_motion: dict[str, Any]) -> str:
    side = [c for c in cells if c["elevation"] == "side"]
    top = [c for c in cells if c["elevation"] == "top"]
    s180 = next(c for c in cells if c["name"] == "s180")
    criteria = "".join(f"<li>{item}</li>" for item in PASS_CRITERIA)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>camera direction sheet — {unit} / {clip}</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>The camera direction sheet</h1>
<p class="caption">{unit} / {clip} (frame {frame_index}), rig {glb.name},
commit {commit} — the manual test for
<a href="http://192.168.1.158:8700/work_packages/288" style="color:#7aa2d6">#288</a>.</p>

<h2>side (elevation = side)</h2>
<div class="row">{''.join(_cell_html(c) for c in side)}</div>

<h2>top (elevation = top, straight down)</h2>
<div class="row">{''.join(_cell_html(c) for c in top)}</div>

<h2>s180 vs --mirror</h2>
<div class="mirror">
  <figure><img src="{s180['skeleton']}" alt="s180 skeleton"><figcaption>s180 (--yaw 180)</figcaption></figure>
  <figure><img src="{mirror_rel}" alt="mirror skeleton"><figcaption>--mirror (yaw 0, mirrored)</figcaption></figure>
</div>

<h2>Pass criteria (#288)</h2>
<ol class="criteria">{criteria}</ol>

<footer>Generated by <code>python -m anim.tools.direction_sheet {unit} {clip}</code>.
Every cell rendered through <code>anim.rigkit.reference_frame</code> /
<code>clip_keypoints</code> / <code>clip_depth</code> and <code>anim.posekit.draw</code>
— the same calls <code>anim rig</code> and <code>anim base</code> make.</footer>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m anim.tools.direction_sheet",
        description="the camera direction sheet (#288): every cell of "
                     "config.DIRECTIONS for one unit's rest pose, skeleton + "
                     "depth, plus the --mirror comparison, as one page")
    parser.add_argument("unit", help="unit slug, e.g. knight")
    parser.add_argument("clip", help="clip name in the rig, e.g. Idle")
    parser.add_argument("--rig", default=None, metavar="GLB",
                        help="rig file (default: DEFAULT_RIG_FOR_UNIT[unit])")
    parser.add_argument("--frame", type=int, default=0,
                        help="sampled frame of the clip (default 0, the rest pose)")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the sheet (default "
                             "work/sheets/camera-direction)")
    args = parser.parse_args(argv)

    index_path = build_sheet(args.unit, args.clip, rig=args.rig,
                             frame_index=args.frame,
                             out_dir=Path(args.out_dir) if args.out_dir else None)
    out_dir = index_path.parent
    n_png = sum(1 for _ in (out_dir / "img").glob("*.png"))
    print(f"direction sheet: {n_png} PNG(s) -> {out_dir}")
    print(f"  index -> {index_path}")
    print(f"  publish: mkdir -p {VISUALS_TARGET} && "
          f"cp -r {out_dir}/. {VISUALS_TARGET}/")
    print(f"  then:    {VISUALS_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
