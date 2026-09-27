"""`anim clip` - the clip render feature - against the fake ComfyUI.

The control video assembly is pure and tested directly; the drive is tested
end to end against `FakeComfy` returning a canned frame batch; then the clip
route through `post --input clip` and the sidecar's `generation.route`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from fixtures.fakecomfy import FakeComfy, canned_png

from anim import comfy, config
from anim.job import Job
from anim.stages import clip as clip_stage
from anim.stages import pick as pick_stage
from anim.stages import post as post_stage
from anim.stages import publish as publish_stage


# ---------------------------------------------------------------------------
# Control video assembly
# ---------------------------------------------------------------------------
def test_the_clip_is_padded_to_4n_plus_1_by_holding_the_last_frame():
    assert clip_stage.padded_length(1) == 1
    assert clip_stage.padded_length(13) == 13
    assert clip_stage.padded_length(19) == 21
    assert clip_stage.padded_length(21) == 21
    assert clip_stage.padded_length(22) == 25
    assert clip_stage.frame_map(19, 21) == list(range(19)) + [18, 18]
    with pytest.raises(ValueError):
        clip_stage.padded_length(0)


def test_auto_control_is_pose_without_depth_and_blend_with_it():
    assert clip_stage.resolve_control("auto", have_depth=False) == "pose"
    assert clip_stage.resolve_control("auto", have_depth=True) == "blend"
    assert clip_stage.resolve_control("pose", have_depth=True) == "pose"
    with pytest.raises(FileNotFoundError, match="depth/"):
        clip_stage.resolve_control("depth", have_depth=False)
    with pytest.raises(ValueError, match="unknown control mode"):
        clip_stage.resolve_control("canny", have_depth=True)


def test_blend_draws_the_skeleton_over_the_depth():
    skel = Image.new("RGB", (32, 32), (0, 0, 0))
    ImageDraw.Draw(skel).line([(0, 16), (31, 16)], fill=(255, 0, 0), width=2)
    depth = Image.new("L", (32, 32), 0)
    ImageDraw.Draw(depth).rectangle([8, 8, 24, 24], fill=200)
    out = np.asarray(clip_stage.compose_control(skel, depth, "blend"))
    assert tuple(out[16, 16]) == (255, 0, 0), "the skeleton wins where drawn"
    assert tuple(out[10, 10]) == (200, 200, 200), "depth shows through elsewhere"
    assert tuple(out[2, 2]) == (0, 0, 0)
    assert clip_stage.compose_control(skel, None, "pose").mode == "RGB"
    assert np.asarray(clip_stage.compose_control(None, depth, "depth"))[10, 10, 0] == 200


def test_depth_gain_dims_the_depth_but_never_the_skeleton():
    skel = Image.new("RGB", (32, 32), (0, 0, 0))
    ImageDraw.Draw(skel).line([(0, 16), (31, 16)], fill=(255, 255, 255), width=2)
    depth = Image.new("L", (32, 32), 0)
    ImageDraw.Draw(depth).rectangle([8, 8, 24, 24], fill=200)
    out = np.asarray(clip_stage.compose_control(skel, depth, "blend", depth_gain=0.5))
    assert tuple(out[10, 10]) == (100, 100, 100)
    assert tuple(out[16, 16]) == (255, 255, 255)
    only = np.asarray(clip_stage.compose_control(None, depth, "depth", depth_gain=0.25))
    assert only[10, 10, 0] == 50


def test_the_cell_must_be_a_multiple_of_the_latent_step():
    clip_stage.check_cell(512, 768)
    with pytest.raises(ValueError, match="step 16"):
        clip_stage.check_cell(510, 768)


def test_the_seed_is_derived_from_the_clip_unit_action_and_control():
    manifest = {"clip_sha256": "a" * 64, "action": "strike", "slug": "knight"}
    seed = clip_stage.derive_seed(manifest, "pose")
    assert seed == clip_stage.derive_seed(manifest, "pose")
    assert seed != clip_stage.derive_seed(manifest, "blend")
    assert seed != clip_stage.derive_seed({**manifest, "action": "hit"}, "pose")


# ---------------------------------------------------------------------------
# The drive
# ---------------------------------------------------------------------------
@pytest.fixture
def batch_pngs(posed_job):
    """One canned frame per padded position, sized like the skeleton cell."""
    frames = len(posed_job.skeletons())
    length = clip_stage.padded_length(frames)
    with Image.open(posed_job.skeletons()[0]) as first:
        cell = first.size
    pngs = []
    for i in range(length):
        img = Image.new("RGB", cell, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        cx = cell[0] // 2
        draw.rectangle([cx - 40, 120, cx + 40, 300], fill=(60, 78, 128))
        draw.ellipse([cx - 36, 40, cx + 36, 120], fill=(196, 152, 118))
        draw.text((4, 4), str(i), fill=(0, 0, 0))
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        pngs.append(buffer.getvalue())
    return pngs


def test_the_shipped_clip_workflow_declares_its_patch_points(workflow_dir):
    workflow = comfy.load_workflow("clip_render", workflow_dir)
    points = comfy.patch_points(workflow)
    for needle in ("reference image", "control video", "clip size", "seed",
                   "output prefix", "positive prompt", "negative prompt"):
        assert comfy.find_node(workflow, needle)
    node = workflow[comfy.find_node(workflow, "clip size")]
    assert node["class_type"] == "Wan22FunControlToVideo"
    assert set(node["inputs"]) >= {"ref_image", "control_video", "length"}
    assert workflow[comfy.find_node(workflow, "control video")]["class_type"] == "LoadVideo"


def test_clip_writes_one_generated_frame_per_rig_frame(posed_job, batch_pngs,
                                                       workflow_dir, tmp_path):
    frames = len(posed_job.skeletons())
    with FakeComfy(batch=batch_pngs) as fake:
        client = comfy.ComfyClient(fake.url)
        record = clip_stage.clip(posed_job.name, client=client, control="pose",
                                 workflow_dir=workflow_dir, work_dir=posed_job.root.parent,
                                 assets_dir=tmp_path / "no-assets", reference=Path(
                                     posed_job.skeletons()[0]))
    job = posed_job
    assert record["frames"] == frames
    assert record["length"] == clip_stage.padded_length(frames)
    assert record["control"] == "pose"
    assert len(job.clip_frames()) == frames
    assert [p.name for p in job.clip_frames()] == [f"{i + 1:04d}.png" for i in range(frames)]
    # The control video went up under a job-unique name, the reference too.
    assert f"anim_{job.name}_control.mp4" in fake.uploads
    assert f"anim_{job.name}_reference.png" in fake.uploads
    assert (job.clip_dir / "control.mp4").exists()
    assert len(list((job.clip_dir / "control").glob("*.png"))) == record["length"]
    # The workflow that was sent carries the size, length and seed.
    sent = fake.prompts[0]
    size_node = sent[comfy.find_node(sent, "clip size")]["inputs"]
    assert (size_node["width"], size_node["height"]) == tuple(record["cell"])
    assert size_node["length"] == record["length"]
    assert sent[comfy.find_node(sent, "seed")]["inputs"]["seed"] == record["seed"]
    assert sent[comfy.find_node(sent, "control video")]["inputs"]["file"] == \
        f"anim_{job.name}_control.mp4"
    # clip.json and provenance.json are there and agree.
    assert job.clip_path.exists() and job.clip_provenance_path.exists()
    provenance = json.loads(job.clip_provenance_path.read_text())
    assert provenance["seed"] == record["seed"]
    assert provenance["passes"][0]["name"] == "clip_render"
    assert provenance["passes"][0]["workflow"] == "clip_render.json"
    assert job.read_manifest()["clip_render"]["frames"] == frames


def test_clip_uses_the_units_base_render_as_the_reference_when_there_is_one(
        posed_job, batch_pngs, workflow_dir, tmp_path):
    base = Job.open("knight_base", posed_job.root.parent)
    base.render_dir.mkdir(parents=True)
    base.write_manifest({"schema": "anim.job/1", "slug": "knight", "action": "",
                         "render": {"final": "base_sprite.png"}})
    (base.render_dir / "base_sprite.png").write_bytes(canned_png((512, 768)))
    with FakeComfy(batch=batch_pngs) as fake:
        record = clip_stage.clip(posed_job.name, client=comfy.ComfyClient(fake.url),
                                 control="pose", workflow_dir=workflow_dir,
                                 work_dir=posed_job.root.parent)
    assert record["reference"]["kind"] == "base render"
    assert record["reference"]["file"].endswith("knight_base/render/base_sprite.png")


def test_clip_blends_depth_in_when_the_rig_wrote_it(posed_job, batch_pngs,
                                                    workflow_dir, tmp_path):
    with Image.open(posed_job.skeletons()[0]) as first:
        cell = first.size
    posed_job.depth_dir.mkdir()
    for skel in posed_job.skeletons():
        depth = Image.new("L", cell, 0)
        ImageDraw.Draw(depth).rectangle([cell[0] // 3, cell[1] // 4,
                                         2 * cell[0] // 3, cell[1] - 10], fill=180)
        depth.save(posed_job.depth_dir / skel.name.replace("skel_", ""))
    with FakeComfy(batch=batch_pngs) as fake:
        record = clip_stage.clip(posed_job.name, client=comfy.ComfyClient(fake.url),
                                 workflow_dir=workflow_dir, work_dir=posed_job.root.parent,
                                 reference=Path(posed_job.skeletons()[0]))
    assert record["control"] == "blend"
    with Image.open(posed_job.clip_dir / "control" / "0001.png") as control:
        arr = np.asarray(control)
    assert (arr.max(axis=2) == 180).any(), "the depth body is in the control frame"


def test_a_short_batch_from_the_service_is_an_error(posed_job, batch_pngs,
                                                    workflow_dir):
    with FakeComfy(batch=batch_pngs[:3]) as fake:
        with pytest.raises(comfy.ComfyError, match="3 frame"):
            clip_stage.clip(posed_job.name, client=comfy.ComfyClient(fake.url),
                            control="pose", workflow_dir=workflow_dir,
                            work_dir=posed_job.root.parent,
                            reference=Path(posed_job.skeletons()[0]))


def test_clip_without_the_gpu_says_how_to_start_it(posed_job, workflow_dir):
    client = comfy.ComfyClient("http://127.0.0.1:9", timeout=0.5)
    with pytest.raises(comfy.ComfyUnreachable, match="podman-compose"):
        clip_stage.clip(posed_job.name, client=client, workflow_dir=workflow_dir,
                        work_dir=posed_job.root.parent)


# ---------------------------------------------------------------------------
# Downstream: post --input clip, publish route
# ---------------------------------------------------------------------------
@pytest.fixture
def clipped_job(posed_job, batch_pngs, workflow_dir):
    with FakeComfy(batch=batch_pngs) as fake:
        clip_stage.clip(posed_job.name, client=comfy.ComfyClient(fake.url),
                        control="pose", workflow_dir=workflow_dir,
                        work_dir=posed_job.root.parent,
                        reference=Path(posed_job.skeletons()[0]))
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    return posed_job


def test_post_takes_the_picked_frames_from_the_clip(clipped_job, palette):
    report = post_stage.post(clipped_job.name, palette=palette, matte="chroma",
                             source="clip", work_dir=clipped_job.root.parent)
    picks = json.loads(clipped_job.picks_path.read_text())["picks"]
    assert report["route"] == "clip"
    assert [r["frame"] for r in report["frames"]] == picks
    assert len(list(clipped_job.post_dir.glob("knight_strike_*.png"))) == len(picks)
    for record in report["frames"]:
        assert not record["problems"]
        with Image.open(clipped_job.post_dir / record["file"]) as frame:
            assert frame.size == config.CANVAS
    assert clipped_job.read_manifest()["post"]["route"] == "clip"


def test_post_on_the_clip_route_needs_no_strip(clipped_job, palette):
    assert not clipped_job.layout_path.exists()
    post_stage.post(clipped_job.name, palette=palette, matte="chroma",
                    source="clip", work_dir=clipped_job.root.parent)


def test_post_without_a_clip_says_to_run_clip(posed_job, palette):
    with pytest.raises(FileNotFoundError, match="anim clip"):
        post_stage.post(posed_job.name, palette=palette, matte="chroma",
                        source="clip", work_dir=posed_job.root.parent)


def test_the_sidecar_records_the_clip_route_and_its_provenance(clipped_job, palette,
                                                               assets_tree):
    post_stage.post(clipped_job.name, palette=palette, matte="chroma",
                    source="clip", work_dir=clipped_job.root.parent)
    result = publish_stage.publish(clipped_job.name, assets_dir=assets_tree / "assets",
                                   run_audit=False, work_dir=clipped_job.root.parent)
    sidecar = json.loads((assets_tree / "assets" / "2d" / "units" /
                          result["sidecar"]).read_text())
    clip_record = json.loads(clipped_job.clip_path.read_text())
    assert sidecar["generation"]["route"] == "clip"
    assert sidecar["generation"]["seed"] == clip_record["seed"]
    assert sidecar["generation"]["workflow"] == "clip_render.json"
    assert sidecar["generation"]["workflow_sha256"] == clip_record["workflow_sha256"]


def test_the_strip_route_is_still_the_default_route():
    assert publish_stage.sidecar_for(slug="knight", action="hit", frames=2,
                                     manifest={}, provenance=None, picks=[0, 1],
                                     report=None)["generation"]["route"] == "strip"
