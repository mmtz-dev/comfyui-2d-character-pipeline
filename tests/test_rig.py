"""`anim rig` end to end on the vendored CC0 Knight, then pick and strip unchanged.

The Knight is a legitimate fixture: CC0 by its own LICENSE.txt, vendored with
its hash in rigs/kaykit-adventurers/rig.json. These tests are the proof that a
rig job is indistinguishable from a filmed one to every stage after `pose`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anim import config
from anim.job import Job
from anim.stages import base as base_stage
from anim.stages import pick as pick_stage
from anim.stages import rig as rig_stage
from anim.stages import strip as strip_stage
from anim.stages.publish import sidecar_for
from anim.stages.review import ReviewApp

REPO = Path(__file__).resolve().parent.parent
KNIGHT = REPO / "rigs" / "kaykit-adventurers" / "Knight.glb"

ANCHOR = [
    # clip, action, the art bible's budget
    ("Idle", "idle", 2),
    ("1H_Melee_Attack_Slice_Horizontal", "strike", 4),
    ("Hit_A", "hit", 2),
    ("Death_A", "death", 4),
]


@pytest.fixture(scope="module")
def knight_glb() -> Path:
    if not KNIGHT.exists():
        pytest.skip("rigs/kaykit-adventurers/Knight.glb is not vendored")
    return KNIGHT


@pytest.fixture
def hit_job(knight_glb, work_dir) -> Job:
    return rig_stage.rig(knight_glb, clip="Hit_A", action="hit", slug="knight",
                         work_dir=work_dir)


# ---------------------------------------------------------------------------
# What the stage writes
# ---------------------------------------------------------------------------
def test_rig_writes_exactly_what_extract_and_pose_would(hit_job):
    manifest = hit_job.read_manifest()
    assert manifest["schema"] == "anim.job/1"
    assert manifest["slug"] == "knight" and manifest["action"] == "hit"
    assert manifest["frames"] == 9 == len(hit_job.frames()) == len(hit_job.skeletons())
    assert manifest["frame_size"] == list(config.STRIP_CELL)
    assert manifest["pose"]["model"] == "rig"
    assert manifest["pose"]["rig"] == "openpose18"
    assert manifest["pose"]["detected"] == 9
    payload = json.loads(hit_job.keypoints_path.read_text())
    assert payload["schema"] == "anim.keypoints/1"
    assert [f["file"] for f in payload["frames"]] == [p.name for p in hit_job.frames()]


def test_the_manifest_names_the_rig_as_the_motion_source(hit_job):
    src = hit_job.read_manifest()["motion_source"]
    assert src["kind"] == "rig"
    assert src["pack"] == "KayKit Adventurers"
    assert src["clip"] == "Hit_A"
    assert src["license"] == "CC0-1.0"
    assert src["url"].startswith("https://github.com/KayKit-Game-Assets/")
    assert len(src["file_sha256"]) == 64
    assert src["file_sha256"] == hit_job.read_manifest()["clip_sha256"]
    assert src["rig_map"] == "kaykit"
    assert src["forward"] == pytest.approx([0, 0, 1], abs=1e-3)
    assert src["mirror"] is False


def test_the_skeletons_are_drawn_by_the_pose_stages_renderer(hit_job):
    """Same size, same black ground, same rtmlib colours as `anim pose`."""
    with Image.open(hit_job.pose_dir / "skel_0001.png") as img:
        assert img.size == tuple(config.STRIP_CELL)
        colours = {tuple(px) for px in np.asarray(img.convert("RGB")).reshape(-1, 3)}
        assert (0, 0, 0) in colours, "black ground"
        assert len(colours) > 8, "a coloured rig, not a monochrome stick figure"


def test_the_previews_are_not_the_conditioning_images(hit_job):
    with Image.open(hit_job.frames_dir / "0001.png") as img:
        assert img.size == tuple(config.STRIP_CELL)
        assert img.getpixel((2, 2)) == rig_stage.PREVIEW_BG, "dark ground, not black"


def test_a_rig_without_provenance_is_refused(knight_glb, tmp_path, work_dir):
    bare = tmp_path / "loose" / "Knight.glb"
    bare.parent.mkdir()
    shutil.copy(knight_glb, bare)
    with pytest.raises(FileNotFoundError, match="rig.json"):
        rig_stage.rig(bare, clip="Idle", action="idle", slug="knight", work_dir=work_dir)


def test_a_rig_whose_hash_moved_is_refused(knight_glb, tmp_path, work_dir):
    moved = tmp_path / "moved" / "Knight.glb"
    moved.parent.mkdir()
    shutil.copy(knight_glb, moved)
    record = json.loads((knight_glb.parent / "rig.json").read_text())
    record["files"]["Knight.glb"] = "0" * 64
    (moved.parent / "rig.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hashes to"):
        rig_stage.rig(moved, clip="Idle", action="idle", slug="knight", work_dir=work_dir)


def test_an_unknown_clip_is_refused_with_a_hint(knight_glb, work_dir):
    with pytest.raises(KeyError, match="Did you mean"):
        rig_stage.rig(knight_glb, clip="hit", action="hit", slug="knight", work_dir=work_dir)


def test_list_prints_every_clip_with_a_duration(knight_glb):
    clips = rig_stage.list_clips(knight_glb)
    names = {c["name"]: c for c in clips}
    assert len(clips) == 76
    assert names["Hit_A"]["duration"] == pytest.approx(0.6667, abs=1e-3)
    assert names["1H_Melee_Attack_Slice_Horizontal"]["duration"] == pytest.approx(1.0667, abs=1e-3)


def test_mirror_is_recorded_and_flips_the_skeleton(knight_glb, work_dir):
    job = rig_stage.rig(knight_glb, clip="Idle", action="idle", slug="knight",
                        mirror=True, job_name="knight_idle_left", work_dir=work_dir)
    manifest = job.read_manifest()
    assert manifest["motion_source"]["mirror"] is True
    assert manifest["motion_source"]["forward"] == pytest.approx([0, 0, 1], abs=1e-3), \
        "the recorded facing is the rig's true facing; mirror is a separate flag"
    payload = json.loads(job.keypoints_path.read_text())
    names, pts = payload["names"], payload["frames"][0]["keypoints"]
    assert pts[names.index("nose")][0] < pts[names.index("neck")][0], "faces left"


# ---------------------------------------------------------------------------
# Props - the skeleton props feature
# ---------------------------------------------------------------------------
def _has_white(path: Path) -> bool:
    with Image.open(path) as img:
        return bool((np.asarray(img.convert("RGB")) == 255).all(axis=2).any())


def test_the_knight_carries_its_kit_by_default(hit_job):
    """`config.UNIT_PROPS` says a knight holds a sword and a shield; the stage
    draws them onto every skeleton and records what it drew."""
    assert hit_job.read_manifest()["motion_source"]["props"] == ["sword", "shield"]
    payload = json.loads(hit_job.keypoints_path.read_text())
    assert payload["motion"]["props"] == ["sword", "shield"]
    assert all([p["kind"] for p in e["props"]] == ["sword", "shield"]
               for e in payload["frames"])
    assert all(_has_white(p) for p in hit_job.skeletons()), "white outlines on every cell"
    with Image.open(hit_job.frames_dir / "0001.png") as img:
        assert rig_stage.PREVIEW_PROP in {tuple(px) for px in
                                          np.asarray(img.convert("RGB")).reshape(-1, 3)}, \
            "the review preview shows them too"


def test_props_none_draws_a_bare_skeleton(knight_glb, work_dir):
    job = rig_stage.rig(knight_glb, clip="Hit_A", action="hit", slug="knight",
                        props=(), work_dir=work_dir)
    assert job.read_manifest()["motion_source"]["props"] == []
    payload = json.loads(job.keypoints_path.read_text())
    assert all("props" not in e for e in payload["frames"])
    assert not any(_has_white(p) for p in job.skeletons())


def test_a_unit_with_no_kit_gets_no_props(knight_glb, work_dir):
    job = rig_stage.rig(knight_glb, clip="Hit_A", action="hit", slug="hound",
                        work_dir=work_dir)
    assert job.read_manifest()["motion_source"]["props"] == []


def test_the_props_flag_is_parsed_and_checked():
    assert rig_stage.parse_props(None) is None
    assert rig_stage.parse_props("none") == ()
    assert rig_stage.parse_props("sword, shield") == ("sword", "shield")
    with pytest.raises(SystemExit, match="halberd"):
        rig_stage.parse_props("halberd")


def test_base_draws_the_kit_on_the_standing_skeleton(knight_glb, work_dir):
    job = base_stage.prepare("knight", rig=knight_glb, clip="Idle", frame=0,
                             work_dir=work_dir)
    assert job.read_manifest()["motion_source"]["props"] == ["sword", "shield"]
    assert _has_white(job.strip_dir / "skeleton_strip.png")
    bare = base_stage.prepare("knight", rig=knight_glb, clip="Idle", frame=0,
                              props=(), work_dir=work_dir)
    assert not _has_white(bare.strip_dir / "skeleton_strip.png")


# ---------------------------------------------------------------------------
# The stages after it, unchanged
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("clip,action,budget", ANCHOR, ids=[a[1] for a in ANCHOR])
def test_pick_and_strip_run_unchanged_on_a_rig_job(knight_glb, work_dir,
                                                   reference_sprite, clip, action,
                                                   budget):
    job = rig_stage.rig(knight_glb, clip=clip, action=action, slug="knight",
                        work_dir=work_dir)
    record = pick_stage.pick(job.name, work_dir=work_dir)
    assert record["n"] == budget == config.frames_for_action(action)
    assert len(record["picks"]) == budget
    assert record["picks"] == sorted(set(record["picks"]))

    layout = strip_stage.strip(job.name, reference=reference_sprite, work_dir=work_dir)
    width, height = layout["strip"]["size"]
    assert (width, height) == (config.STRIP_CELL[0] * budget, config.STRIP_CELL[1])
    assert width != height, "never square"
    with Image.open(job.strip_dir / "skeleton_strip.png") as img:
        assert img.size == (width, height)


def test_a_two_pose_reaction_gets_the_flinch_not_rest_twice(knight_glb, work_dir):
    """Hit_A starts and ends at rest; the two frames must not both be rest."""
    job = rig_stage.rig(knight_glb, clip="Hit_A", action="hit", slug="knight",
                        work_dir=work_dir)
    record = pick_stage.pick(job.name, work_dir=work_dir)
    first, last = record["picks"]
    deviation = record["signals"]["deviation"]
    assert last == len(deviation) - 1, "the recovery frame stays"
    assert first == deviation.index(max(deviation)), "the extreme of the flinch"


def test_the_review_page_serves_a_rig_job(hit_job):
    app = ReviewApp(hit_job)
    state = app.state()
    assert len(state["frames"]) == 9
    assert all(f["skeleton"] for f in state["frames"])
    assert app.image("frames", "0001.png")[:8] == b"\x89PNG\r\n\x1a\n"
    assert app.image("skel", "skel_0001.png")[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# `anim base --rig`
# ---------------------------------------------------------------------------
def test_base_takes_its_standing_pose_from_a_rig_frame(knight_glb, work_dir):
    job = base_stage.prepare("knight", rig=knight_glb, clip="Idle", frame=0,
                             work_dir=work_dir)
    manifest = job.read_manifest()
    assert manifest["action"] == ""
    assert manifest["clip"] == "Knight.glb::Idle#0"
    assert manifest["motion_source"]["kind"] == "rig"
    assert manifest["motion_source"]["frame"] == 0
    assert manifest["pose"]["model"] == "rig"
    payload = json.loads(job.keypoints_path.read_text())
    assert len(payload["frames"]) == 1
    assert payload["source_size"] == list(config.BASE_CELL)
    with Image.open(job.strip_dir / "skeleton_strip.png") as img:
        assert img.size == tuple(config.BASE_CELL)
    layout = json.loads(job.layout_path.read_text())
    assert layout["strip"]["cells"] == 1


def test_base_refuses_a_frame_past_the_end_of_the_clip(knight_glb, work_dir):
    with pytest.raises(ValueError, match="outside 0"):
        base_stage.prepare("knight", rig=knight_glb, clip="Idle", frame=99,
                           work_dir=work_dir)


def test_the_canned_base_still_works_without_a_rig(work_dir):
    job = base_stage.prepare("knight", work_dir=work_dir)
    assert job.read_manifest()["pose"]["model"] == "canned"
    assert "motion_source" not in job.read_manifest()


# ---------------------------------------------------------------------------
# `anim publish` carries the rig into the sidecar
# ---------------------------------------------------------------------------
def test_the_sidecar_records_the_rig_as_the_motion_source(hit_job):
    manifest = hit_job.read_manifest()
    meta = sidecar_for(slug="knight", action="hit", frames=2, manifest=manifest,
                       provenance={"seed": 7, "models": [],
                                   "passes": [{"workflow": "identity_pass.json",
                                               "workflow_sha256": "b" * 64}]},
                       picks=[3, 8], report=None)
    motion = meta["generation"]["motion_source"]
    assert motion == manifest["motion_source"]
    assert motion["license"] == "CC0-1.0"
    assert "rig clip" in meta["source"]["author"]
    assert meta["generation"]["clip"] == "Knight.glb::Hit_A"
    assert meta["generation"]["seed"] == 7, "the pixel provenance is untouched"


def test_a_filmed_job_has_no_motion_source():
    meta = sidecar_for(slug="knight", action="strike", frames=4,
                       manifest={"clip_sha256": "a" * 64, "clip": "x.mp4", "fps": 12},
                       provenance={"seed": 1, "models": [], "passes": []},
                       picks=[0, 1, 2, 3], report=None)
    assert meta["generation"]["motion_source"] is None
    assert "filmed" in meta["source"]["author"]


def test_game_02s_audit_accepts_the_extra_generation_key(hit_job, assets_tree,
                                                         audit_script):
    """The real audit, on a sidecar carrying motion_source. Not weakened: the
    same audit still fails without a seed."""
    units = assets_tree / "assets" / "2d" / "units"
    for i in range(2):
        Image.new("RGBA", config.CANVAS, (0, 0, 0, 0)).save(units / f"knight_hit_{i}.png")
    meta = sidecar_for(slug="knight", action="hit", frames=2,
                       manifest=hit_job.read_manifest(),
                       provenance={"seed": 7, "models": [],
                                   "passes": [{"workflow": "identity_pass.json",
                                               "workflow_sha256": "b" * 64}]},
                       picks=[3, 8], report=None)
    (units / "knight_hit.json").write_text(json.dumps(meta, indent=2))
    proc = subprocess.run([sys.executable, str(audit_script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    meta["generation"]["seed"] = None
    (units / "knight_hit.json").write_text(json.dumps(meta, indent=2))
    proc = subprocess.run([sys.executable, str(audit_script)], capture_output=True, text=True)
    assert proc.returncode != 0, "the audit must still demand a seed"


# ---------------------------------------------------------------------------
# Depth - the rig props feature
# ---------------------------------------------------------------------------
def test_rig_writes_one_depth_frame_per_sampled_frame(hit_job):
    depths = hit_job.depths()
    assert [p.name for p in depths] == [p.name for p in hit_job.frames()], \
        "depth/%04d.png indexes exactly as frames/%04d.png"
    manifest = hit_job.read_manifest()
    src = manifest["motion_source"]
    assert src["depth"] is True and src["depth_mode"] == "full"
    far, near = src["depth_range"]
    assert far < near
    assert src["depth_far_grey"] == 48
    for path in depths:
        with Image.open(path) as img:
            assert img.mode == "L", "8-bit grayscale"
            assert img.size == tuple(manifest["pose"]["cell"])
            arr = np.asarray(img)
        assert arr.max() > 0, "a body was drawn"
        assert (arr == 0).mean() > 0.5, "on a black background"


def test_no_depth_skips_the_directory_and_says_so(knight_glb, work_dir):
    job = rig_stage.rig(knight_glb, clip="Hit_A", action="hit", slug="knight",
                        depth=False, work_dir=work_dir)
    assert job.depths() == []
    src = job.read_manifest()["motion_source"]
    assert src["depth"] is False and src["depth_mode"] == "none" and "depth_range" not in src
    # And a re-run with depth on fills it; a re-run with it off clears it.
    rig_stage.rig(knight_glb, clip="Hit_A", action="hit", slug="knight", work_dir=work_dir)
    assert len(job.depths()) == len(job.frames())
    rig_stage.rig(knight_glb, clip="Hit_A", action="hit", slug="knight",
                  depth=False, work_dir=work_dir)
    assert job.depths() == []


def test_the_props_show_up_in_depth_not_just_as_outlines(work_dir):
    """On the Quaternius recovery frame (blade out front) the tip's pixels are
    a surface with the kit and nothing without it (the mannequin has no props)."""
    ual = REPO / "rigs" / "quaternius" / "UAL1_Standard.glb"
    if not ual.exists():
        pytest.skip("rigs/quaternius/UAL1_Standard.glb is not vendored")
    armed = rig_stage.rig(ual, clip="Sword_Attack", action="strike", slug="knight",
                          work_dir=work_dir, job_name="armed")
    bare = rig_stage.rig(ual, clip="Sword_Attack", action="strike", slug="knight",
                         props=(), work_dir=work_dir, job_name="bare")
    assert armed.read_manifest()["motion_source"]["props"] == ["sword", "shield"]
    blade = np.asarray(json.loads(armed.keypoints_path.read_text())
                       ["frames"][10]["props"][0]["polygons"][0], dtype=float)
    axis = blade[2] - (blade[0] + blade[4]) / 2
    x, y = (blade[2] - 40 * axis / np.linalg.norm(axis)).astype(int)   # the tip leaves the cell
    window = (slice(y - 2, y + 3), slice(x - 2, x + 3))
    with Image.open(armed.depths()[10]) as img:
        assert np.asarray(img)[window].max() > 0
    with Image.open(bare.depths()[10]) as img:
        assert np.asarray(img)[window].max() == 0


def test_base_from_a_rig_writes_its_one_depth_frame(knight_glb, work_dir):
    job = base_stage.prepare("knight", rig=knight_glb, clip="Idle", frame=0,
                             work_dir=work_dir)
    assert [p.name for p in job.depths()] == ["0001.png"]
    with Image.open(job.depths()[0]) as img:
        assert img.size == tuple(config.BASE_CELL) and img.mode == "L"
    src = job.read_manifest()["motion_source"]
    assert src["depth"] is True and src["depth_range"][0] < src["depth_range"][1]


def test_the_canned_base_writes_no_depth(work_dir):
    job = base_stage.prepare("knight", work_dir=work_dir)
    assert job.depths() == []


def test_depth_props_writes_the_kit_only_and_records_the_mode(work_dir):
    ual = REPO / "rigs" / "quaternius" / "UAL1_Standard.glb"
    if not ual.exists():
        pytest.skip("rigs/quaternius/UAL1_Standard.glb is not vendored")
    job = rig_stage.rig(ual, clip="Hit_Chest", action="hit", slug="knight",
                        depth="props", work_dir=work_dir)
    src = job.read_manifest()["motion_source"]
    assert src["depth"] is True and src["depth_mode"] == "props"
    assert len(job.depths()) == len(job.frames())
    with Image.open(job.depths()[0]) as img:
        arr = np.asarray(img)
    assert 0 < (arr > 0).sum() < 6000, "two props (a kite is ~3k px), no ~12k-px body"
    full = rig_stage.rig(ual, clip="Hit_Chest", action="hit", slug="knight",
                         work_dir=work_dir, job_name="full")
    assert full.read_manifest()["motion_source"]["depth_mode"] == "full"
    assert full.read_manifest()["motion_source"]["depth_range"] == src["depth_range"]
    # A unit with no kit has nothing to draw in props mode: no depth at all.
    bare = rig_stage.rig(ual, clip="Hit_Chest", action="hit", slug="knight",
                         props=(), depth="props", work_dir=work_dir, job_name="bare")
    assert bare.depths() == []
    assert bare.read_manifest()["motion_source"]["depth_mode"] == "none"


def test_the_depth_mode_flag_and_its_alias():
    import argparse
    parser = argparse.ArgumentParser()
    rig_stage.add_arguments(parser)
    assert parser.parse_args(["x.glb"]).depth == "full"
    assert parser.parse_args(["x.glb", "--depth", "props"]).depth == "props"
    assert parser.parse_args(["x.glb", "--no-depth"]).depth == "none"
    with pytest.raises(SystemExit):
        parser.parse_args(["x.glb", "--depth", "halberd"])


# ---------------------------------------------------------------------------
# Direction - the camera direction feature
# ---------------------------------------------------------------------------
def test_the_default_cell_keeps_the_job_name_and_records_s0(hit_job):
    assert hit_job.name == "knight_hit"
    src = hit_job.read_manifest()["motion_source"]
    assert (src["direction"], src["yaw"], src["elevation"]) == ("s0", 0, "side")


def test_a_direction_suffixes_the_job_name_and_writes_every_output(knight_glb, work_dir):
    """End to end on the vendored Knight for one non-default cell: the same
    layout `extract` + `pose` leave, plus depth, with the cell in job.json."""
    job = rig_stage.rig(knight_glb, clip="Hit_A", action="hit", slug="knight",
                        yaw=270, elevation="top", work_dir=work_dir)
    assert job.name == "knight_hit_t270"
    manifest = job.read_manifest()
    src = manifest["motion_source"]
    assert (src["direction"], src["yaw"], src["elevation"]) == ("t270", 270, "top")
    assert src["mirror"] is False
    assert src["forward"] == pytest.approx([0, 0, 1], abs=1e-3), "the rig's own facing, not the view"
    assert manifest["frames"] == 9 == len(job.frames()) == len(job.skeletons()) == len(job.depths())
    payload = json.loads(job.keypoints_path.read_text())
    assert payload["motion"]["direction"] == "t270"
    plain = rig_stage.rig(knight_glb, clip="Hit_A", action="hit", slug="knight",
                          work_dir=work_dir)
    assert src["scale_px_per_unit"] == plain.read_manifest()["motion_source"]["scale_px_per_unit"]
    with Image.open(job.depths()[0]) as img:
        assert img.mode == "L" and img.size == tuple(config.STRIP_CELL)
        assert np.asarray(img).max() == 255, "the head is the nearest surface, and drawn"
    with Image.open(job.frames_dir / "0001.png") as img:
        # No floor line through a top-down figure: the preview's floor row is the ground colour.
        row = int(payload["motion"]["baseline_row"])
        assert img.getpixel((2, row)) == rig_stage.PREVIEW_BG
    # The stages after it cannot tell: pick runs unchanged.
    record = pick_stage.pick(job.name, work_dir=work_dir)
    assert len(record["picks"]) == 2


def test_an_explicit_job_name_is_not_suffixed(knight_glb, work_dir):
    job = rig_stage.rig(knight_glb, clip="Idle", action="idle", slug="knight",
                        yaw=90, job_name="facing_n", work_dir=work_dir)
    assert job.name == "facing_n"
    assert job.read_manifest()["motion_source"]["direction"] == "s90"


def test_default_job_names_are_untouched_in_the_default_cell():
    assert rig_stage.default_job_name("knight_hit") == "knight_hit"
    assert rig_stage.default_job_name("knight_hit", 0, "side") == "knight_hit"
    assert rig_stage.default_job_name("knight_hit", 45) == "knight_hit_s45"
    assert rig_stage.default_job_name("knight_base", 0, "top") == "knight_base_t0"


def test_the_cli_refuses_a_yaw_off_the_table_and_mirror_with_a_direction(capsys):
    import argparse
    parser = argparse.ArgumentParser(prog="anim rig")
    rig_stage.add_arguments(parser)
    args = parser.parse_args(["x.glb", "--yaw", "315", "--elevation", "top"])
    assert (args.yaw, args.elevation) == (315, "top")
    assert (parser.parse_args(["x.glb"]).yaw, parser.parse_args(["x.glb"]).elevation) == (0, "side")
    with pytest.raises(SystemExit):
        parser.parse_args(["x.glb", "--yaw", "30"])
    assert "315" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parser.parse_args(["x.glb", "--elevation", "front"])
    # mirror + a direction is refused before the rig is even opened (x.glb does not exist).
    with pytest.raises(SystemExit, match="--yaw 180"):
        rig_stage.run_stage(parser.parse_args(
            ["x.glb", "--clip", "Idle", "--unit", "knight", "--action", "idle",
             "--mirror", "--yaw", "180"]))
    with pytest.raises(SystemExit, match="mirror"):
        rig_stage.run_stage(parser.parse_args(
            ["x.glb", "--clip", "Idle", "--unit", "knight", "--action", "idle",
             "--mirror", "--elevation", "top"]))
    assert rig_stage.check_direction("anim rig", yaw=0, elevation="side", mirror=True) == "s0"


def test_mirror_alone_still_works_and_is_refused_with_a_cell_in_the_library(knight_glb, work_dir):
    job = rig_stage.rig(knight_glb, clip="Idle", action="idle", slug="knight",
                        mirror=True, work_dir=work_dir)
    assert job.read_manifest()["motion_source"]["mirror"] is True
    with pytest.raises(ValueError, match="mirror"):
        rig_stage.rig(knight_glb, clip="Idle", action="idle", slug="knight",
                      mirror=True, yaw=45, work_dir=work_dir)


def test_base_takes_a_direction_and_suffixes_its_job(knight_glb, work_dir):
    job = base_stage.prepare("knight", rig=knight_glb, clip="Idle", frame=0,
                             yaw=270, work_dir=work_dir)
    assert job.name == "knight_base_s270"
    src = job.read_manifest()["motion_source"]
    assert (src["direction"], src["yaw"], src["elevation"]) == ("s270", 270, "side")
    assert [p.name for p in job.depths()] == ["0001.png"]
    payload = json.loads(job.keypoints_path.read_text())
    names, pts = payload["names"], payload["frames"][0]["keypoints"]
    # Facing the viewer: the two shoulders straddle the neck, right shoulder on the viewer's left.
    assert pts[names.index("right_shoulder")][0] < pts[names.index("neck")][0] < pts[names.index("left_shoulder")][0]
    plain = base_stage.prepare("knight", rig=knight_glb, clip="Idle", frame=0, work_dir=work_dir)
    assert plain.name == "knight_base", "the s0 base keeps the name the later stages look for"


def test_the_canned_base_has_no_direction_to_turn(work_dir):
    with pytest.raises(ValueError, match="needs --rig"):
        base_stage.prepare("knight", yaw=90, work_dir=work_dir)
    import argparse
    parser = argparse.ArgumentParser(prog="anim base")
    base_stage.add_arguments(parser)
    with pytest.raises(SystemExit, match="need --rig"):
        base_stage.run_stage(parser.parse_args(["--unit", "knight", "--yaw", "90", "--skeleton-only"]))
    with pytest.raises(SystemExit, match="mirror"):
        base_stage.run_stage(parser.parse_args(
            ["--unit", "knight", "--rig", "x.glb", "--mirror", "--yaw", "180", "--skeleton-only"]))
