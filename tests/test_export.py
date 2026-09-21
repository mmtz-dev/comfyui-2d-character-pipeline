"""`anim export` - the spritesheet export feature.

The rect math, the atlas, the .tres and its read-back are tested on jobs
fabricated straight onto disk (a job IS its directory), so a five-cell matrix
costs milliseconds and no pipeline stage. The game-02 adapter is tested on the
real `post` output and the real `asset_audit.py`, as test_publish.py does,
because that is the audit that will judge what it writes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anim import config
from anim.api.app import create_app
from anim.job import Job, new_manifest
from anim.stages.export import (Animation, atlas_for, export, find_animations, layout,
                                parse_tres, plan_timing, tres_for, verify_tres)
from test_publish import posted_job  # noqa: F401 - the real `post` output, for the adapter

UNIT = "knight"


def _colour(index: int) -> tuple[int, int, int, int]:
    return (40 * (index + 1) % 256, 90, 200 - 30 * index, 255)


def make_job(work_dir: Path, name: str, *, action: str, direction: str = "s0",
             frames: int | None = 3, timing: dict | None = None,
             slug: str = UNIT, size: tuple[int, int] = (64, 64)) -> Job:
    """A finished job on disk: job.json, picks.json, post/<stem>_<n>.png.
    Each frame is a distinct flat colour over a 20x30 body, so a sheet's
    cells can be traced back to the frame they came from."""
    job = Job.open(name, work_dir)
    manifest = new_manifest(clip=f"{name}.mp4", clip_sha256="a" * 64, fps=12,
                            frames=20, action=action, slug=slug,
                            frame_size=[304, 320], crop=None, source_fps=None,
                            created="2026-09-21T00:00:00+00:00")
    if direction != config.DEFAULT_DIRECTION:
        manifest["motion_source"] = {"kind": "rig", "pack": "test rig",
                                     "clip": "Clip", "license": "CC0-1.0",
                                     "direction": direction}
    job.write_manifest(manifest)
    if frames is None:
        return job
    record = {"schema": "anim.picks/1", "n": frames, "action": action,
              "picks": list(range(frames))}
    if timing:
        record["timing"] = timing
        record["auto_timing"] = dict(timing)
    job.picks_path.write_text(json.dumps(record))
    job.post_dir.mkdir(parents=True)
    stem = f"{slug}_{action}"
    for i in range(frames):
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        for x in range(size[0] // 2 - 10, size[0] // 2 + 10):
            for y in range(size[1] - 30, size[1]):
                img.putpixel((x, y), _colour(i))
        img.save(job.post_dir / f"{stem}_{i}.png")
    Image.new("RGB", (4, 1)).save(job.post_dir / "palette.png")
    job.report_path.write_text(json.dumps({"schema": "anim.post/1", "route": "strip",
                                           "palette": {"snap": True}}))
    return job


PHASED = {"weights": [1, 1, 2, 1], "anchor": 2, "source": "phases:attack", "fps": 12,
          "loop": False, "phases": ["start", "wind-up", "action", "end"],
          "durations_ms": [83, 83, 167, 83]}


@pytest.fixture
def matrix(work_dir: Path) -> Path:
    """The knight with three cells and three near-misses, as work/ looks."""
    make_job(work_dir, "knight_strike", action="strike", frames=4, timing=PHASED)
    make_job(work_dir, "knight_idle", action="idle", frames=2)
    make_job(work_dir, "knight_walk_s90", action="walk", direction="s90", frames=3)
    # slug knight, action strike, but not the matrix's strike job
    make_job(work_dir, "knight_strike_phases", action="strike", frames=4)
    # picked, never posted
    make_job(work_dir, "knight_run", action="run", frames=None)
    # a base job: a still, not an animation
    make_job(work_dir, "knight_base", action="", frames=None)
    # another unit whose name starts with this one
    make_job(work_dir, "knightq_death", action="death", frames=2, slug="knightq")
    (work_dir / "_runs").mkdir()
    return work_dir


# ---------------------------------------------------------------------------
# Timing: phases -> ticks -> durations
# ---------------------------------------------------------------------------
def test_a_phased_pick_gives_ticks_phases_and_milliseconds():
    plan = plan_timing("strike", 4, {"timing": PHASED, "auto_timing": dict(PHASED)})
    assert plan["ticks"] == [1.0, 1.0, 2.0, 1.0]
    assert plan["phases"] == ["start", "wind-up", "action", "end"]
    assert plan["loop"] is False and plan["fps"] == 12 and plan["anchor"] == 2
    assert plan["source"] == "phases:attack"


def test_the_strip_template_and_uniform_both_become_ticks():
    # A picks.json from before timing: the strike template's holds.
    plan = plan_timing("strike", 4, {"picks": [0, 6, 10, 18]})
    assert plan["ticks"] == [2.0, 3.0, 4.0, 3.0] and plan["source"] == "template:strike"
    # No row at all: one tick per frame, and idle loops because the studio says so.
    plan = plan_timing("idle", 2, None)
    assert plan["ticks"] == [1.0, 1.0] and plan["loop"] is True
    assert plan["phases"] == [None, None] and plan["source"] == "uniform"
    assert plan_timing("hit", 2, None)["loop"] is False


def _anim(action: str, direction: str, n: int, **kw) -> Animation:
    fields = dict(action=action, direction=direction, job=f"{UNIT}_{action}",
                  frames=tuple(Path(f"{i}.png") for i in range(n)),
                  ticks=tuple(1.0 for _ in range(n)), fps=12, loop=False,
                  phases=tuple(None for _ in range(n)), anchor=None,
                  timing_source="uniform", picks=tuple(range(n)), provenance={})
    fields.update(kw)
    return Animation(**fields)


def test_durations_ms_come_from_ticks_at_the_animations_fps():
    anim = _anim("strike", "s0", 4, ticks=(2.0, 3.0, 4.0, 3.0))
    assert anim.durations_ms == [167, 250, 333, 250]
    assert _anim("idle", "s0", 2, fps=24).durations_ms == [42, 42]


# ---------------------------------------------------------------------------
# Rect math
# ---------------------------------------------------------------------------
def test_per_direction_sheets_have_one_row_per_action():
    anims = [_anim("strike", "s0", 4), _anim("idle", "s0", 2), _anim("walk", "s90", 3)]
    plan = layout(UNIT, anims, (64, 64), "per-direction")
    assert set(plan["sheets"]) == {"sheets/knight_s0.png", "sheets/knight_s90.png"}
    assert plan["sheets"]["sheets/knight_s0.png"]["size"] == [4 * 64, 2 * 64]
    assert plan["sheets"]["sheets/knight_s0.png"]["rows"] == ["idle", "strike"], \
        "rows are alphabetical, and the atlas says which is which"
    assert plan["rects"]["idle_s0"]["frames"] == [[0, 0, 64, 64], [64, 0, 64, 64]]
    assert plan["rects"]["strike_s0"]["frames"] == [[i * 64, 64, 64, 64] for i in range(4)]
    assert plan["rects"]["walk_s90"] == {"sheet": "sheets/knight_s90.png",
                                         "frames": [[i * 64, 0, 64, 64] for i in range(3)]}
    assert plan["sheets"]["sheets/knight_s90.png"]["size"] == [3 * 64, 64]


def test_a_packed_sheet_is_row_major_and_roughly_square():
    anims = [_anim("strike", "s0", 4), _anim("idle", "s0", 2), _anim("walk", "s90", 3)]
    plan = layout(UNIT, anims, (32, 48), "packed")
    assert list(plan["sheets"]) == ["sheets/knight.png"]
    assert plan["sheets"]["sheets/knight.png"] == {"size": [3 * 32, 3 * 48], "columns": 3}
    rects = [r for key in ("strike_s0", "idle_s0", "walk_s90")
             for r in plan["rects"][key]["frames"]]
    assert rects == [[(k % 3) * 32, (k // 3) * 48, 32, 48] for k in range(9)]


def test_an_unknown_sheet_mode_is_refused():
    with pytest.raises(ValueError, match="sheet mode"):
        layout(UNIT, [_anim("idle", "s0", 2)], (64, 64), "grid")


# ---------------------------------------------------------------------------
# The atlas and the .tres
# ---------------------------------------------------------------------------
def test_the_atlas_carries_the_schema_the_rects_and_the_provenance():
    anims = [_anim("strike", "s0", 4, ticks=(1.0, 1.0, 2.0, 1.0),
                   phases=("start", "wind-up", "action", "end"), anchor=2,
                   timing_source="phases:attack", picks=(0, 4, 9, 18),
                   provenance={"route": "clip", "seed": 7, "clip": "x.glb",
                               "clip_sha256": "a" * 64, "motion_source": None})]
    plan = layout(UNIT, anims, (64, 64))
    atlas = atlas_for(UNIT, anims, plan, palette="Lospec 2000",
                      warnings=[{"kind": "missing", "action": "jump",
                                 "direction": "s0", "job": "knight_jump"}],
                      commit="abc1234", created="2026-09-21T00:00:00+00:00")
    assert atlas["schema"] == "anim.atlas/1"
    assert atlas["unit"] == "knight" and atlas["frame_size"] == [64, 64]
    assert atlas["fps"] == 12 and atlas["pivot"] == "bottom-center"
    assert atlas["palette"] == "Lospec 2000" and atlas["sheet_mode"] == "per-direction"
    row = atlas["animations"]["strike_s0"]
    assert row["sheet"] == "sheets/knight_s0.png" and row["loop"] is False
    assert row["pivot"] == [32, 64] and row["anchor"] == 2 and row["fps"] == 12
    assert row["frames"][2] == {"x": 128, "y": 0, "w": 64, "h": 64, "phase": "action",
                                "ticks": 2.0, "duration_ms": 167}
    assert row["source"] == {"job": "knight_strike", "picks": [0, 4, 9, 18],
                             "timing_source": "phases:attack", "route": "clip",
                             "seed": 7, "clip": "x.glb", "clip_sha256": "a" * 64,
                             "motion_source": None}
    assert atlas["tool"]["commit"] == "abc1234" and atlas["tool"]["version"]
    assert atlas["warnings"][0]["job"] == "knight_jump"


@pytest.fixture
def two_sheet_atlas():
    anims = [_anim("strike", "s0", 4, ticks=(1.0, 1.0, 2.0, 1.0)),
             _anim("idle", "s0", 2, loop=True),
             _anim("walk", "s90", 3, loop=True, fps=24, ticks=(1.5, 1.0, 1.5))]
    plan = layout(UNIT, anims, (64, 64))
    return atlas_for(UNIT, anims, plan, palette=None, warnings=[], commit=None,
                     created="2026-09-21T00:00:00+00:00")


def test_the_tres_reads_back_as_the_atlas(two_sheet_atlas):
    text = tres_for(UNIT, two_sheet_atlas)
    assert text.startswith('[gd_resource type="SpriteFrames" load_steps=12 format=3]'), \
        "2 sheets + 9 frames + the resource"
    parsed = parse_tres(text)
    assert parsed["type"] == "SpriteFrames"
    assert {row["path"] for row in parsed["ext"].values()} == \
        {"sheets/knight_s0.png", "sheets/knight_s90.png"}
    assert all(row["type"] == "Texture2D" for row in parsed["ext"].values())
    by_name = {row["name"]: row for row in parsed["animations"]}
    assert set(by_name) == {"strike_s0", "idle_s0", "walk_s90"}
    strike = by_name["strike_s0"]
    assert len(strike["frames"]) == 4 and strike["loop"] is False and strike["speed"] == 12.0
    assert [f["duration"] for f in strike["frames"]] == [1.0, 1.0, 2.0, 1.0]
    walk = by_name["walk_s90"]
    assert walk["loop"] is True and walk["speed"] == 24.0
    assert [f["duration"] for f in walk["frames"]] == [1.5, 1.0, 1.5]
    # every frame is an AtlasTexture over the right sheet at the right rect
    third = parsed["sub"][strike["frames"][2]["texture"]["$sub"]]
    assert third["type"] == "AtlasTexture" and third["region"] == [128, 64, 64, 64]
    assert parsed["ext"][third["atlas"]["$ext"]]["path"] == "sheets/knight_s0.png"
    assert verify_tres(text, two_sheet_atlas) == []


def test_verify_tres_catches_a_resource_that_drifted(two_sheet_atlas):
    text = tres_for(UNIT, two_sheet_atlas)
    assert verify_tres(text.replace('"loop": true', '"loop": false'), two_sheet_atlas)
    assert verify_tres(text.replace("Rect2(128, 64, 64, 64)", "Rect2(128, 0, 64, 64)"),
                       two_sheet_atlas)
    assert verify_tres(text.replace('"duration": 2.0', '"duration": 1.0'), two_sheet_atlas)
    assert verify_tres("[gd_resource type=\"Nope\" format=3]\n", two_sheet_atlas)


def test_parse_tres_reads_what_godot_itself_writes():
    """The editor's own spelling of the same resource: uid on the header, a
    typed array, the frames one per line."""
    text = '''[gd_resource type="SpriteFrames" load_steps=3 format=3 uid="uid://b1"]

[ext_resource type="Texture2D" uid="uid://c2" path="res://units/knight_s0.png" id="1_a"]

[sub_resource type="AtlasTexture" id="AtlasTexture_x"]
atlas = ExtResource("1_a")
region = Rect2(0, 0, 64, 64)

[resource]
animations = [{
"frames": [{
"duration": 1.0,
"texture": SubResource("AtlasTexture_x")
}],
"loop": true,
"name": &"idle_s0",
"speed": 12.0
}]
'''
    parsed = parse_tres(text)
    assert parsed["ext"]["1_a"]["path"] == "res://units/knight_s0.png"
    assert parsed["ext"]["1_a"]["uid"] == "uid://c2"
    assert parsed["animations"][0]["name"] == "idle_s0"
    assert parsed["animations"][0]["frames"][0]["texture"] == {"$sub": "AtlasTexture_x"}


# ---------------------------------------------------------------------------
# Finding the jobs: missing cells are warnings, near-misses are skipped
# ---------------------------------------------------------------------------
def test_the_matrix_is_found_by_name_and_the_gaps_are_warnings(matrix):
    anims, warnings = find_animations(UNIT, matrix)
    assert [a.key for a in anims] == ["idle_s0", "strike_s0", "walk_s90"], \
        "direction order, then action"
    assert [a.job for a in anims] == ["knight_idle", "knight_strike", "knight_walk_s90"]
    skipped = {w["job"]: w["reason"] for w in warnings if w["kind"] == "skipped"}
    assert "not the matrix job for strike s0 (that is knight_strike)" in \
        skipped["knight_strike_phases"]
    assert "anim post" in skipped["knight_run"]
    assert "knight_base" not in skipped and "knightq_death" not in skipped
    missing = {(w["action"], w["direction"]) for w in warnings if w["kind"] == "missing"}
    side = [d for d in config.DIRECTIONS if d.startswith("s")]
    assert ("strike", "s45") in missing and ("jump", "s0") in missing
    assert ("walk", "s0") in missing and ("walk", "s90") not in missing
    assert ("idle", "s90") in missing, "a direction the unit has is expected of every action"
    assert ("idle", "t0") not in missing, "top-down is a later milestone"
    expected = {(a, d) for a in ("strike", "idle", "walk", "run", "jump") for d in side + ["s90"]}
    assert missing == expected - {("idle", "s0"), ("strike", "s0"), ("walk", "s90")}
    walk = next(a for a in anims if a.action == "walk")
    assert walk.direction == "s90" and walk.loop is True


def test_a_no_snap_preview_is_skipped_with_the_reason(matrix):
    job = Job.open("knight_idle", matrix)
    job.report_path.write_text(json.dumps({"palette": {"snap": False}}))
    anims, warnings = find_animations(UNIT, matrix)
    assert "idle_s0" not in [a.key for a in anims]
    assert any("--no-snap" in w.get("reason", "") for w in warnings)


def test_a_unit_with_no_finished_job_is_an_error_that_says_what_is_needed(work_dir):
    make_job(work_dir, "hound_run", action="run", frames=None)
    with pytest.raises(FileNotFoundError, match="hound_<action>"):
        export("hound", work_dir=work_dir)
    with pytest.raises(ValueError, match="unit slug"):
        find_animations("../x", work_dir)


# ---------------------------------------------------------------------------
# The stage end to end
# ---------------------------------------------------------------------------
def test_export_writes_sheets_atlas_tres_and_preview(matrix):
    result = export(UNIT, work_dir=matrix)
    out = matrix / "_export" / "knight"
    assert result["dir"] == str(out) and result["adapter"] == "generic"
    assert result["frame_size"] == [64, 64]
    assert sorted(p.name for p in (out / "sheets").glob("*.png")) == \
        ["knight_s0.png", "knight_s90.png"]
    assert (out / "atlas.json").exists() and (out / "knight.tres").exists()
    assert (out / "preview.html").exists()
    atlas = json.loads((out / "atlas.json").read_text())
    assert atlas["schema"] == "anim.atlas/1" and atlas["unit"] == "knight"
    assert list(atlas["animations"]) == ["idle_s0", "strike_s0", "walk_s90"]
    assert atlas["animations"]["strike_s0"]["frames"][2]["duration_ms"] == 167
    assert atlas["animations"]["strike_s0"]["frames"][2]["phase"] == "action"
    assert atlas["animations"]["idle_s0"]["loop"] is True
    assert atlas["animations"]["walk_s90"]["source"]["motion_source"]["pack"] == "test rig"
    assert len(atlas["warnings"]) == len(result["warnings"]) > 0
    assert "tool" in atlas and "created" in atlas["tool"]
    assert verify_tres((out / "knight.tres").read_text(), atlas) == []
    # the frames landed in their cells: the sheet's cell IS the post frame
    with Image.open(out / "sheets" / "knight_s0.png") as sheet:
        assert sheet.size == (4 * 64, 2 * 64)
        strike2 = atlas["animations"]["strike_s0"]["frames"][2]
        cell = sheet.crop((strike2["x"], strike2["y"],
                           strike2["x"] + 64, strike2["y"] + 64))
        with Image.open(matrix / "knight_strike" / "post" / "knight_strike_2.png") as src:
            assert cell.tobytes() == src.convert("RGBA").tobytes()
        assert sheet.getpixel((3 * 64 + 10, 10)) == (0, 0, 0, 0), \
            "idle has two frames; the rest of its row is transparent"
    preview = (out / "preview.html").read_text()
    assert '"schema": "anim.atlas/1"' in preview and "sheets/knight_s0.png" in preview


def test_a_packed_export_replaces_the_per_direction_sheets(matrix):
    export(UNIT, work_dir=matrix)
    result = export(UNIT, work_dir=matrix, sheet="packed")
    out = matrix / "_export" / "knight"
    assert [p.name for p in (out / "sheets").glob("*.png")] == ["knight.png"]
    assert sorted(result["removed"]) == ["sheets/knight_s0.png", "sheets/knight_s90.png"]
    atlas = json.loads((out / "atlas.json").read_text())
    assert atlas["sheet_mode"] == "packed" and atlas["sheets"]["sheets/knight.png"]["columns"] == 3
    assert atlas["animations"]["walk_s90"]["frames"][0]["x"] == 0
    assert atlas["animations"]["walk_s90"]["frames"][0]["y"] == 2 * 64


def test_a_larger_cell_pads_bottom_centre_and_a_smaller_one_is_refused(matrix, tmp_path):
    result = export(UNIT, work_dir=matrix, size="96", out_dir=tmp_path / "big")
    assert result["frame_size"] == [96, 96]
    with Image.open(tmp_path / "big" / "sheets" / "knight_s0.png") as sheet:
        assert sheet.size == (4 * 96, 2 * 96)
        # idle frame 0 body: x in 22..42 of the 64 frame -> +16; y 34..64 -> +32
        assert sheet.getpixel((16 + 22, 32 + 34)) == _colour(0)
        assert sheet.getpixel((16 + 21, 32 + 34)) == (0, 0, 0, 0)
        assert sheet.getpixel((16 + 22, 95)) == _colour(0)
    with pytest.raises(ValueError, match="does not fit"):
        export(UNIT, work_dir=matrix, size="32x64", out_dir=tmp_path / "small")
    with pytest.raises(ValueError, match="N or WxH"):
        export(UNIT, work_dir=matrix, size="big", out_dir=tmp_path / "bad")


def test_the_cell_comes_from_the_project_settings(matrix, tmp_path):
    (matrix / "project.json").write_text(json.dumps({"frame_size": [80, 72]}))
    result = export(UNIT, work_dir=matrix, out_dir=tmp_path / "proj")
    assert result["frame_size"] == [80, 72]
    atlas = json.loads((tmp_path / "proj" / "atlas.json").read_text())
    assert atlas["animations"]["idle_s0"]["pivot"] == [40, 72]


def test_a_dry_run_writes_nothing(matrix):
    result = export(UNIT, work_dir=matrix, dry_run=True)
    assert result["dry_run"] is True and result["animations"] == ["idle_s0", "strike_s0", "walk_s90"]
    assert not (matrix / "_export").exists()


def test_the_export_dir_is_not_a_job(matrix):
    from anim.api import jobs as jobs_api

    export(UNIT, work_dir=matrix)
    assert "_export" not in [row["name"] for row in jobs_api.list_jobs(matrix)]
    assert jobs_api.open_job("_export", matrix) is None


def test_the_cli_prints_the_files_and_the_gaps(matrix, monkeypatch, capsys):
    from anim import cli

    monkeypatch.setattr(config, "WORK_DIR", matrix)
    assert cli.main(["export", UNIT]) == 0
    captured = capsys.readouterr()
    assert "export: wrote 3 animation(s) of knight" in captured.out
    assert "sheets/knight_s0.png" in captured.out and "knight.tres" in captured.out
    assert "skipped knight_strike_phases" in captured.err
    assert "missing jump: s0 s45 s90 s135 s180 s225 s270 s315 (8 cell(s))" in captured.err


# ---------------------------------------------------------------------------
# The game-02 adapter: today's publish, unchanged, per job
# ---------------------------------------------------------------------------
def test_the_game02_adapter_publishes_every_job_and_passes_the_audit(
        posted_job, assets_tree, audit_script):
    from anim.stages.publish import publish

    work = posted_job.root.parent
    result = export(UNIT, work_dir=work, adapter="game02",
                    assets_dir=assets_tree / "assets", audit_script=audit_script)
    assert result["published"] == ["knight_strike"]
    assert result["audit"]["exit"] == 0, result["audit"]["output"]
    units = assets_tree / "assets" / "2d" / "units"
    assert sorted(p.name for p in units.glob("*.png")) == \
        [f"knight_strike_{i}.png" for i in range(4)]
    out = work / "_export" / "knight"
    assert (out / "knight.tres").exists() and (out / "sheets" / "knight_s0.png").exists()
    atlas = json.loads((out / "atlas.json").read_text())
    assert atlas["adapter"] == "game02"
    assert atlas["adapter_result"] == {"published": ["knight_strike"], "audit_exit": 0}
    assert atlas["animations"]["strike_s0"]["frames"][0]["ticks"] == 2.0, \
        "the strike template, the same shape the sidecar carries"
    # The sidecar the adapter wrote is byte-for-byte what `publish` alone writes.
    alone = assets_tree / "alone"
    (alone / "2d" / "units").mkdir(parents=True)
    publish(posted_job.name, assets_dir=alone, run_audit=False, work_dir=work)
    assert (units / "knight_strike.json").read_bytes() == \
        (alone / "2d" / "units" / "knight_strike.json").read_bytes()
    for i in range(4):
        assert (units / f"knight_strike_{i}.png").read_bytes() == \
            (alone / "2d" / "units" / f"knight_strike_{i}.png").read_bytes()
    proc = subprocess.run([sys.executable, str(audit_script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_adapter_default_comes_from_the_project_settings(posted_job, assets_tree):
    work = posted_job.root.parent
    (work / "project.json").write_text(json.dumps({"export": {"adapter": "game02"}}))
    result = export(UNIT, work_dir=work, assets_dir=assets_tree / "assets", run_audit=False)
    assert result["adapter"] == "game02" and result["published"] == ["knight_strike"]
    assert (assets_tree / "assets" / "2d" / "units" / "knight_strike.json").exists()
    generic = export(UNIT, work_dir=work, adapter="generic")
    assert generic["adapter"] == "generic" and generic["published"] == []
    with pytest.raises(ValueError, match="adapter"):
        export(UNIT, work_dir=work, adapter="unity")


# ---------------------------------------------------------------------------
# The API: a run like any stage, and the listing
# ---------------------------------------------------------------------------
@pytest.fixture
def api(matrix, tmp_path):
    app = create_app(matrix, web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", cpu_slots=1)
    with TestClient(app) as client:
        yield client


def test_export_runs_through_the_run_manager_and_the_listing_reads_it(api, matrix):
    from test_api import wait_for

    assert api.get("/export/knight").status_code == 404
    assert "POST /jobs" in api.get("/export/knight").json()["detail"]
    assert api.get("/export/../x").status_code == 404
    response = api.post("/jobs", json={"stage": "export",
                                       "args": {"unit": "knight", "sheet": "packed"}})
    assert response.status_code == 202, response.text
    run = response.json()
    assert run["gpu"] is False and run["argv"] == ["--sheet", "packed", "knight"]
    final = wait_for(api, run["id"])
    assert final["status"] == "done", "\n".join(final["log"])
    assert any(line.startswith("export: wrote 3 animation(s)") for line in final["log"])

    row = api.get("/export/knight").json()
    assert row["unit"] == "knight" and row["dir"] == str(matrix / "_export" / "knight")
    assert row["sheet_mode"] == "packed" and row["frame_size"] == [64, 64]
    assert [f["file"] for f in row["files"]] == \
        ["atlas.json", "knight.tres", "preview.html", "sheets/knight.png"]
    assert all(f["bytes"] > 0 for f in row["files"])
    assert row["animations"]["strike_s0"] == {"frames": 4, "loop": False, "fps": 12,
                                              "sheet": "sheets/knight.png",
                                              "job": "knight_strike"}
    assert any(w["kind"] == "missing" for w in row["warnings"])
    assert row["created"] and row["adapter"] is None


def test_export_adapter_is_a_project_setting(api, matrix):
    fresh = api.get("/settings").json()
    assert fresh["export"] == {"adapter": "generic"}
    stored = api.put("/settings", json={"export": {"adapter": "game02"}}).json()
    assert stored["export"] == {"adapter": "game02"}
    assert json.loads((matrix / "project.json").read_text())["export"] == {"adapter": "game02"}
    assert api.put("/settings", json={"export": {"adapter": "unity"}}).status_code == 400
    assert api.put("/settings", json={"export": {"sheet": "packed"}}).status_code == 400
    assert api.put("/settings", json={"export": "game02"}).status_code == 400
