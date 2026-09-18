"""`anim post --parts` and what `publish` does with it - the part palette lock
feature end to end, on drawn strips with the colour-keyed FakeSegmenter.

Three promises are pinned here:
  1. OFF is off - with the lock off `post` writes the same bytes as before.
  2. A base job EXTRACTS part palettes; an action job LOCKS to them, and
     every locked pixel is a member of its part's palette.
  3. The sidecar carries the record and still passes game-02's audit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anim import config, imaging, partkit
from anim.partkit import Part
from anim.stages import base as base_stage
from anim.stages import pick as pick_stage
from anim.stages import post as post_stage
from anim.stages import publish as publish_stage
from anim.stages import strip as strip_stage
from fixtures import synthetic
from fixtures.fakeparts import STRIP_COLOURS, FakeSegmenter

TABLE = (Part("sword", prop="sword"),
         Part("head", joints=("nose", "right_eye", "left_eye")),
         Part("torso", joints=("neck",), mids=(("neck", "right_hip", 0.5),)),
         Part("legs", joints=("right_knee", "left_knee")))


@pytest.fixture
def parts_table(monkeypatch):
    monkeypatch.setitem(config.UNIT_PARTS, "knight", TABLE)
    return TABLE


@pytest.fixture
def base_job(work_dir, parts_table):
    """A knight base job as `anim base` leaves it before `post`: canned
    standing skeleton (no props -> the sword prompt is unusable), one 512x768
    cell, a drawn 'render'."""
    job = base_stage.prepare("knight", work_dir=work_dir)
    job.render_dir.mkdir(parents=True, exist_ok=True)
    synthetic.write_strip(job.render_dir / "base_sprite.png", cells=1,
                          cell=tuple(config.BASE_CELL))
    job.update_manifest(render={"seed": 1, "final": "base_sprite.png"})
    job.provenance_path.write_text(json.dumps({
        "schema": "anim.provenance/1", "seed": 1,
        "comfyui": {"version": "0.35.0-fake"},
        "models": [{"repo": "Comfy-Org/z_image_turbo",
                    "file": "z_image_turbo_int8_convrot.safetensors",
                    "sha256": "c" * 64, "license": "apache-2.0",
                    "revision": "deadbeef"}],
        "passes": [{"name": "base_sprite", "workflow": "base_sprite.json",
                    "workflow_sha256": "b" * 64, "file": "base_sprite.png"}],
    }))
    return job


@pytest.fixture
def action_job(posed_job, reference_sprite, tmp_path):
    """A knight_strike job with a drawn strip whose torso is a DIFFERENT
    colour from the base's, so the lock has something to correct."""
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    strip_stage.strip(posed_job.name, reference=reference_sprite,
                      work_dir=posed_job.root.parent)
    strip = synthetic.write_strip(tmp_path / "strip.png")
    with Image.open(strip) as img:
        arr = np.asarray(img.convert("RGB")).copy()
    torso = (arr == np.array(STRIP_COLOURS["torso"], np.uint8)).all(axis=2)
    arr[torso] = (150, 60, 60)
    posed_job.render_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(posed_job.render_dir / "identity_pass.png")
    posed_job.update_manifest(render={"seed": 4242, "final": "identity_pass.png"})
    posed_job.provenance_path.write_text(json.dumps({
        "schema": "anim.provenance/1", "seed": 4242,
        "comfyui": {"version": "0.35.0-fake"}, "models": [],
        "passes": [{"name": "identity_pass", "workflow": "identity_pass.json",
                    "workflow_sha256": "b" * 64, "file": "identity_pass.png"}],
    }))
    return posed_job


def action_segmenter() -> FakeSegmenter:
    return FakeSegmenter(dict(STRIP_COLOURS, torso=(150, 60, 60)))


# ---------------------------------------------------------------------------
# 1. Off is off
# ---------------------------------------------------------------------------
def test_the_lock_is_off_by_default_and_changes_nothing(action_job, palette):
    wd = action_job.root.parent
    post_stage.post(action_job.name, palette=palette, matte="chroma", work_dir=wd)
    before = [p.read_bytes() for p in sorted(action_job.post_dir.glob("knight_strike_*.png"))]
    report = json.loads(action_job.report_path.read_text())
    assert "parts" not in report
    assert "parts" not in report["frames"][0]
    assert not (action_job.post_dir / "parts").exists()
    post_stage.post(action_job.name, palette=palette, matte="chroma",
                    parts=False, work_dir=wd)
    after = [p.read_bytes() for p in sorted(action_job.post_dir.glob("knight_strike_*.png"))]
    assert before == after
    assert config.part_lock_default("knight") is False


def test_process_cell_without_masks_is_the_old_chain(strip_png, palette, tmp_path):
    colours = imaging.read_palette(palette)
    palette_file = imaging.palette_png(colours, tmp_path / "palette.png")
    with Image.open(strip_png) as img:
        cell = post_stage.matte_chroma(img).crop((0, 0, 304, 320))
    kw = dict(target_height=53, palette_png=palette_file,
              chroma=imaging.chroma_key_for(colours))
    plain, steps = post_stage.process_cell(cell, **kw)
    explicit, steps2 = post_stage.process_cell(cell, part_masks=None, **kw)
    assert plain.tobytes() == explicit.tobytes()
    assert steps == steps2
    assert not any("part" in s for s in steps)


# ---------------------------------------------------------------------------
# 2. Extract on the base, lock on the action
# ---------------------------------------------------------------------------
def test_a_base_job_extracts_its_part_palettes(base_job, palette):
    wd = base_job.root.parent
    seg = FakeSegmenter()
    report = post_stage.post(base_job.name, palette=palette, matte="chroma",
                             parts=True, segmenter=seg, work_dir=wd)
    assert report["parts"]["mode"] == "extract"
    assert report["parts"]["parts"] == ["sword", "head", "torso", "legs"]
    parts = json.loads((base_job.post_dir / "parts.json").read_text())
    assert parts["schema"] == partkit.PARTS_SCHEMA
    assert parts["model"]["repo"] == "facebook/sam2.1-hiera-small"
    colours = set(imaging.read_palette(palette))
    for name in ("head", "torso", "legs"):
        assert parts["palettes"][name], f"{name} must have a palette"
        assert {tuple(c) for c in parts["palettes"][name]} <= colours
        assert parts["pixels"][name] >= config.PART_MIN_PALETTE_PIXELS
    # the canned skeleton has no prop polygons: reported, not invented
    frame = report["frames"][0]["parts"]
    assert frame["sword"]["found"] is False
    assert frame["sword"]["reason"] == "no prompt this frame"
    assert parts["palettes"]["sword"] == []
    assert (base_job.post_dir / "parts" / "knight_0_hd.png").exists()
    assert (base_job.post_dir / "parts" / "knight_0_labels.png").exists()
    # one call to the segmenter per cell, one prompt per part
    assert len(seg.calls) == 1 and [p.part for p in seg.calls[0]] == list(parts["parts"])


def test_a_part_too_small_on_the_base_yields_no_palette(base_job, palette, monkeypatch):
    monkeypatch.setattr(config, "PART_MIN_PALETTE_PIXELS", 10_000)
    post_stage.post(base_job.name, palette=palette, matte="chroma",
                    parts=True, segmenter=FakeSegmenter(), work_dir=base_job.root.parent)
    parts = json.loads((base_job.post_dir / "parts.json").read_text())
    assert parts["palettes"]["torso"] == []
    assert "under 10000 px" in parts["rejected"]["torso"]["reason"]
    assert parts["rejected"]["torso"]["colours"]


def test_an_action_locks_every_part_pixel_to_the_base_palette(base_job, action_job,
                                                             palette):
    wd = base_job.root.parent
    post_stage.post(base_job.name, palette=palette, matte="chroma",
                    parts=True, segmenter=FakeSegmenter(), work_dir=wd)
    base_palettes = json.loads((base_job.post_dir / "parts.json").read_text())["palettes"]

    report = post_stage.post(action_job.name, palette=palette, matte="chroma",
                             parts=True, segmenter=action_segmenter(), work_dir=wd)
    assert report["parts"]["mode"] == "lock"
    assert report["parts"]["palettes_from"].endswith("knight_base/post/parts.json")
    assert report["parts"]["palettes"] == base_palettes
    colours = imaging.read_palette(palette)
    torso_palette = {tuple(c) for c in base_palettes["torso"]}
    for record in report["frames"]:
        assert record["problems"] == []
        torso = record["parts"]["torso"]
        assert torso["locked"] and torso["pixels"] > 0
        assert torso["drift"] > 0, "the red torso was outside the blue palette"
        assert {tuple(c) for c in torso["colours"]} <= torso_palette
        assert any("part palette lock" in s for s in record["steps"])
        with Image.open(action_job.post_dir / record["file"]) as img:
            assert imaging.verify_frame(img.convert("RGBA"), colours) == []
    # the label map on the canvas agrees with the report's pixel counts
    labels = np.asarray(Image.open(action_job.post_dir / "parts" / "knight_strike_0_labels.png"))
    assert (labels[:, :, 3] > 0).sum() == sum(
        p["pixels"] for p in report["frames"][0]["parts"].values())


def test_a_part_with_no_base_palette_is_left_alone_and_says_so(base_job, action_job,
                                                              palette):
    wd = base_job.root.parent
    post_stage.post(base_job.name, palette=palette, matte="chroma",
                    parts=True, segmenter=FakeSegmenter(), work_dir=wd)
    # the action strip HAS a sword the fake can find; the base had none
    entry = json.loads(action_job.keypoints_path.read_text())
    for frame in entry["frames"]:
        frame["props"] = [{"kind": "sword", "polygons": [[[200, 100], [260, 60],
                                                          [264, 66], [204, 106]]]}]
    action_job.keypoints_path.write_text(json.dumps(entry))
    report = post_stage.post(action_job.name, palette=palette, matte="chroma",
                             parts=True, segmenter=action_segmenter(), work_dir=wd)
    sword = report["frames"][0]["parts"]["sword"]
    assert sword["found"] is True and sword["pixels"] > 0
    assert sword["locked"] is False
    assert sword["reason"] == "no palette for this part"


def test_parts_from_overrides_the_lookup(base_job, action_job, palette, tmp_path):
    wd = base_job.root.parent
    colours = imaging.read_palette(palette)
    hand = tmp_path / "hand.json"
    hand.write_text(json.dumps({"schema": partkit.PARTS_SCHEMA,
                                "palettes": {"torso": [list(colours[0])]}}))
    report = post_stage.post(action_job.name, palette=palette, matte="chroma",
                             parts=True, parts_from=hand,
                             segmenter=action_segmenter(), work_dir=wd)
    assert report["parts"]["palettes_from"] == str(hand)
    for record in report["frames"]:
        assert record["parts"]["torso"]["colours"] == [list(colours[0])]


def test_the_lookup_prefers_the_published_sprites_sidecar(tmp_path, work_dir):
    units = tmp_path / "assets" / "2d" / "units"
    units.mkdir(parents=True)
    (units / "knight.json").write_text(json.dumps(
        {"generation": {"parts": {"palettes": {"torso": [[1, 2, 3]]}}}}))
    base_parts = work_dir / "knight_base" / "post"
    base_parts.mkdir(parents=True)
    (base_parts / "parts.json").write_text(json.dumps(
        {"schema": partkit.PARTS_SCHEMA, "palettes": {"torso": [[9, 9, 9]]}}))
    palettes, source = post_stage.load_part_palettes(
        "knight", assets_dir=tmp_path / "assets", work_dir=work_dir)
    assert palettes == {"torso": [[1, 2, 3]]}
    assert source.endswith("knight.json")
    (units / "knight.json").unlink()
    palettes, source = post_stage.load_part_palettes(
        "knight", assets_dir=tmp_path / "assets", work_dir=work_dir)
    assert palettes == {"torso": [[9, 9, 9]]}


def test_locking_without_any_palettes_says_what_to_run(action_job, palette, tmp_path):
    with pytest.raises(FileNotFoundError, match="anim post knight_base --parts"):
        post_stage.post(action_job.name, palette=palette, matte="chroma",
                        parts=True, segmenter=action_segmenter(),
                        assets_dir=tmp_path / "nowhere",
                        work_dir=action_job.root.parent)


def test_a_unit_without_a_part_table_cannot_be_locked(action_job, palette, monkeypatch):
    monkeypatch.delitem(config.UNIT_PARTS, "knight", raising=False)
    with pytest.raises(ValueError, match="config.UNIT_PARTS"):
        post_stage.post(action_job.name, palette=palette, matte="chroma",
                        parts=True, segmenter=action_segmenter(),
                        work_dir=action_job.root.parent)


def test_the_cli_exposes_parts_and_parts_from():
    import argparse

    parser = argparse.ArgumentParser()
    post_stage.add_arguments(parser)
    assert parser.parse_args(["knight_strike"]).parts is None
    assert parser.parse_args(["knight_strike", "--parts"]).parts is True
    assert parser.parse_args(["knight_strike", "--no-parts"]).parts is False
    args = parser.parse_args(["knight_strike", "--parts", "--parts-from", "x.json"])
    assert args.parts_from == "x.json"


# ---------------------------------------------------------------------------
# 3. The sidecar
# ---------------------------------------------------------------------------
def test_publish_carries_the_parts_record_and_the_sam_weights(base_job, action_job,
                                                             palette, assets_tree,
                                                             audit_script):
    wd = base_job.root.parent
    assets = assets_tree / "assets"
    post_stage.post(base_job.name, palette=palette, matte="chroma",
                    parts=True, segmenter=FakeSegmenter(), work_dir=wd)
    base_result = publish_stage.publish(base_job.name, assets_dir=assets,
                                        audit_script=audit_script, work_dir=wd)
    base_meta = json.loads((assets / "2d" / "units" / "knight.json").read_text())
    assert base_meta["generation"]["parts"]["mode"] == "extract"
    assert base_meta["generation"]["parts"]["palettes"]["torso"]
    assert any(m["repo"] == "facebook/sam2.1-hiera-small"
               for m in base_meta["generation"]["models"])
    assert base_result["audit"]["exit"] == 0

    # the action now reads its palettes from that published sidecar
    report = post_stage.post(action_job.name, palette=palette, matte="chroma",
                             parts=True, segmenter=action_segmenter(),
                             assets_dir=assets, work_dir=wd)
    assert report["parts"]["palettes_from"].endswith("knight.json")
    result = publish_stage.publish(action_job.name, assets_dir=assets,
                                   audit_script=audit_script, work_dir=wd)
    meta = json.loads((assets / "2d" / "units" / "knight_strike.json").read_text())
    block = meta["generation"]["parts"]
    assert block["mode"] == "lock"
    assert block["palettes_from"].endswith("knight.json")
    assert len(block["frames"]) == 4
    assert block["frames"][0]["parts"]["torso"]["locked"] is True
    assert block["frames"][0]["parts"]["torso"]["drift"] > 0
    assert result["audit"]["exit"] == 0


def test_a_sidecar_without_the_feature_says_null(action_job, palette, assets_tree,
                                                 audit_script):
    wd = action_job.root.parent
    post_stage.post(action_job.name, palette=palette, matte="chroma", work_dir=wd)
    publish_stage.publish(action_job.name, assets_dir=assets_tree / "assets",
                          audit_script=audit_script, work_dir=wd)
    meta = json.loads((assets_tree / "assets" / "2d" / "units" / "knight_strike.json").read_text())
    assert meta["generation"]["parts"] is None


# ---------------------------------------------------------------------------
# Opt-in: the 2026-09-15 published frames are the byte-exact regression
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("ANIM_POST_REGRESSION"),
                    reason="set ANIM_POST_REGRESSION=1 to re-run post on the real "
                           "knight_hit/knight_death jobs and compare with /assets "
                           "(needs the rembg model in .cache)")
@pytest.mark.parametrize("job,matte", [("knight_hit", "rembg"), ("knight_death", "chroma")])
def test_post_with_the_lock_off_reproduces_the_published_bytes(job, matte, tmp_path):
    import shutil

    src = Path("/work/work") / job
    if not (src / "render" / "pose_pass.png").exists():
        pytest.skip(f"no work/{job} in this checkout")
    published = Path("/assets/2d/units")
    wd = tmp_path / "work"
    shutil.copytree(src, wd / job)
    post_stage.post(job, matte=matte, source=wd / job / "render" / "pose_pass.png",
                    work_dir=wd)
    for i in range(4):
        ours = (wd / job / "post" / f"{job}_{i}.png").read_bytes()
        theirs = (published / f"{job}_{i}.png").read_bytes()
        assert ours == theirs, f"{job}_{i}.png differs from the published frame"
