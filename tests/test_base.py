"""`anim base`: the canned standing skeleton and the single-frame chain."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from fixtures.fakecomfy import FakeComfy, canned_png

from anim import comfy, config
from anim.stages import base as base_stage


def test_the_canned_pose_is_synthesised_not_photographed():
    """No photograph means no licence question for a file that ships in git."""
    payload = base_stage.stand_payload()
    assert payload["model"] == "canned"
    assert payload["rig"] == "openpose18"
    frame = payload["frames"][0]
    assert len(frame["keypoints"]) == len(frame["scores"]) == 18
    assert "SYNTHESISED" in payload["_note"]


def test_the_canned_figure_stands_the_way_the_art_bible_asks():
    payload = base_stage.stand_payload()
    kp = payload["frames"][0]["keypoints"]
    names = payload.get("names") or []
    nose, neck = kp[0], kp[1]
    left_ankle, right_ankle = kp[13], kp[10]
    assert nose[0] > neck[0], "profile facing right (art bible S1)"
    assert min(left_ankle[1], right_ankle[1]) > kp[8][1], "feet below the hips"
    height = max(left_ankle[1], right_ankle[1]) - nose[1]
    assert 400 < height < 700, "roughly four heads in a 768-tall cell"


def test_prepare_lays_out_a_job_the_later_stages_can_read(work_dir):
    job = base_stage.prepare("knight", work_dir=work_dir)
    manifest = job.read_manifest()
    assert manifest["slug"] == "knight"
    assert manifest["action"] == "", "a base sprite has no action"
    layout = json.loads(job.layout_path.read_text())
    assert layout["strip"]["cells"] == 1
    assert layout["strip"]["size"] == list(config.BASE_CELL)
    assert layout["cells"][0]["rect"] == [0, 0, *config.BASE_CELL]
    assert json.loads(job.picks_path.read_text())["picks"] == [0]


def test_the_standing_skeleton_renders_at_the_portrait_cell(work_dir):
    job = base_stage.prepare("knight", work_dir=work_dir)
    with Image.open(job.strip_dir / "skeleton_strip.png") as img:
        assert img.size == tuple(config.BASE_CELL)
        assert np.asarray(img).any(), "an all-black skeleton is no skeleton"


def test_the_base_cell_is_non_square_and_on_the_latent_grid():
    width, height = config.BASE_CELL
    assert width != height
    assert width % config.LATENT_STEP == 0 and height % config.LATENT_STEP == 0


def test_render_base_runs_one_pass_not_two(work_dir, workflow_dir):
    job = base_stage.prepare("knight", work_dir=work_dir)
    with FakeComfy(png=canned_png(config.BASE_CELL)) as fake:
        provenance = base_stage.render_base(
            job, prompt="a plate-armoured knight", seed=7,
            client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir)
    assert len(fake.prompts) == 1, "there is no reference sprite to re-skin onto"
    assert [p["name"] for p in provenance["passes"]] == ["base_sprite"]
    graph = fake.prompts[0]
    latent = next(n for n in graph.values()
                  if n["class_type"] == "EmptySD3LatentImage")
    assert [latent["inputs"]["width"], latent["inputs"]["height"]] == \
        list(config.BASE_CELL)
    text = next(n for n in graph.values() if n["class_type"] == "CLIPTextEncode")
    assert text["inputs"]["text"] == "a plate-armoured knight"
    assert (job.render_dir / "base_sprite.png").exists()


def test_base_end_to_end_publishes_one_frame(work_dir, workflow_dir, palette,
                                             assets_tree, audit_script,
                                             monkeypatch, tmp_path):
    """The same post/publish chain as an action, with contract.frames [1, 1]."""
    from fixtures import synthetic

    # A one-cell "generated" sprite, drawn rather than sampled.
    strip = synthetic.write_strip(tmp_path / "base.png", cells=1,
                                  cell=config.BASE_CELL)
    monkeypatch.setattr(base_stage, "render_base",
                        lambda job, **kwargs: _fake_render(job, strip))

    result = base_stage.base("knight", palette=palette, matte="chroma",
                             assets_dir=assets_tree / "assets",
                             audit_script=audit_script, work_dir=work_dir)
    units = assets_tree / "assets" / "2d" / "units"
    assert result["frames"] == ["knight.png"]
    assert (units / "knight.png").exists()
    meta = json.loads((units / "knight.json").read_text())
    assert meta["id"] == "unit/knight"
    assert meta["contract"]["frames"] == [1, 1]
    with Image.open(units / "knight.png") as img:
        assert img.size == tuple(config.CANVAS)
    proc = subprocess.run([sys.executable, str(audit_script)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _fake_render(job, strip):
    """Stand in for the GPU: drop a drawn sprite where `render` would have."""
    from anim.util import write_json_in_place

    job.render_dir.mkdir(parents=True, exist_ok=True)
    (job.render_dir / "base_sprite.png").write_bytes(strip.read_bytes())
    provenance = {
        "schema": "anim.provenance/1", "seed": 7,
        "comfyui": {"version": "0.35.0-fake", "devices": []},
        "models": [{"repo": "Comfy-Org/z_image_turbo", "file": "z.safetensors",
                    "sha256": "c" * 64, "license": "apache-2.0"}],
        "passes": [{"name": "base_sprite", "workflow": "base_sprite.json",
                    "workflow_sha256": "d" * 64, "file": "base_sprite.png"}],
        "final": "base_sprite.png",
    }
    write_json_in_place(job.provenance_path, provenance)
    job.update_manifest(render={"seed": 7, "passes": ["base_sprite"],
                                "final": "base_sprite.png"})
    return provenance


def test_base_against_an_unreachable_service_says_so(work_dir, workflow_dir):
    job = base_stage.prepare("knight", work_dir=work_dir)
    with pytest.raises(comfy.ComfyUnreachable):
        base_stage.render_base(job,
                               client=comfy.ComfyClient("http://127.0.0.1:1",
                                                        timeout=1),
                               workflow_dir=workflow_dir)


# ---------------------------------------------------------------------------
# The front base pose feature
# ---------------------------------------------------------------------------
def test_the_front_pose_is_synthesised_not_photographed():
    """Same licence question as its side-view sibling, same answer."""
    payload = base_stage.stand_payload(base_stage.STAND_POSE_FRONT)
    assert payload["model"] == "canned"
    assert payload["rig"] == "openpose18"
    frame = payload["frames"][0]
    assert len(frame["keypoints"]) == len(frame["scores"]) == 18
    assert "SYNTHESISED" in payload["_note"]


def test_the_front_figure_faces_the_viewer():
    """The whole point of the file: limbs spread across x, not stacked in a
    column. The side pose collapses every joint into ~66px, which is what put
    hooded_archer's arms, torso and legs on one patch of cloak."""
    front = base_stage.stand_payload(base_stage.STAND_POSE_FRONT)["frames"][0]["keypoints"]
    side = base_stage.stand_payload()["frames"][0]["keypoints"]
    spread = max(p[0] for p in front) - min(p[0] for p in front)
    assert spread > 200, "a front view spreads the limbs across the cell"
    assert spread > 3 * (max(p[0] for p in side) - min(p[0] for p in side))
    # OpenPose-18: a person facing us has their right on our left.
    right_shoulder, left_shoulder = front[2], front[5]
    right_ankle, left_ankle = front[10], front[13]
    assert right_shoulder[0] < left_shoulder[0]
    assert right_ankle[0] < left_ankle[0]
    assert front[0][1] < front[1][1] < front[8][1] < right_ankle[1]


def test_repose_swaps_the_pose_and_leaves_the_render_alone(work_dir):
    """Why `repose` exists at all: re-picking would clear the accepted views,
    so a unit picked before the front pose landed must be able to fix its part
    prompts without paying for its views twice."""
    job = base_stage.prepare("hooded_archer", cell=(768, 768), work_dir=work_dir)
    job.render_dir.mkdir(parents=True, exist_ok=True)
    (job.render_dir / "concept_pick.png").write_bytes(b"not really a png")
    views = {"back": {"file": "view_back.png"}}
    job.update_manifest(render={"seed": 11, "passes": ["concept_pass"],
                                "final": "concept_pick.png"},
                        concept={"job": "hooded_archer_concept", "views": views})
    before = json.loads(job.keypoints_path.read_text())

    result = base_stage.repose("hooded_archer", work_dir=work_dir)

    after = json.loads(job.keypoints_path.read_text())
    assert before["frames"][0]["keypoints"] != after["frames"][0]["keypoints"]
    assert after["source_size"] == [768, 768]
    manifest = job.read_manifest()
    assert manifest["render"]["final"] == "concept_pick.png", "render untouched"
    assert manifest["concept"]["views"] == views, "accepted views untouched"
    assert manifest["pose"]["cell"] == [768, 768]
    assert result["job"] == job.name


def test_repose_refuses_a_rig_posed_base(work_dir):
    """A rig base's pose is the rig's; swapping in the canned skeleton would
    silently decouple it from the motion source the manifest names."""
    job = base_stage.prepare("knight", work_dir=work_dir)
    job.update_manifest(pose={"rig": "openpose18", "model": "rig",
                              "mode": "kaykit", "cell": [512, 768],
                              "detected": 1})
    with pytest.raises(ValueError, match="rig"):
        base_stage.repose("knight", work_dir=work_dir)


def test_repose_needs_a_base_job_to_exist(work_dir):
    with pytest.raises(FileNotFoundError, match="no base job"):
        base_stage.repose("nobody", work_dir=work_dir)
