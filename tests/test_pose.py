"""`anim pose`: the schema, the renderer, and the cache.

Inference itself is only exercised by the opt-in smoke test at the bottom: the
DWPose ONNX weights are a ~200 MB download, and a suite that needs the network
to go green is a suite that goes red for the wrong reasons. Everything that is
OURS - the schema, the normaliser, the renderer's geometry, the caching rule -
is tested here without a model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anim import config, posekit
from anim.stages import pose as pose_stage

FIXTURE_PHOTO = Path(__file__).resolve().parent / "fixtures" / "pose" / "cc0_performer.png"


def test_the_first_eighteen_joints_are_the_same_in_both_rigs():
    """Why the normaliser can look joints up by NAME and not care about the rig."""
    assert posekit.rig_names("openpose134")[:18] == posekit.rig_names("openpose18")


def test_rig_for_matches_rtmlibs_own_dispatch():
    assert posekit.rig_for(18) == "openpose18"
    assert posekit.rig_for(134) == "openpose134"


def test_line_width_scales_with_the_canvas():
    thin = posekit.line_width_for((128, 128))
    cell = posekit.line_width_for(config.STRIP_CELL)
    portrait = posekit.line_width_for(config.BASE_CELL)
    assert thin < cell < portrait
    assert cell >= 4, "a hairline rig at 304px is one the ControlNet cannot see"


def test_normalise_makes_the_same_pose_at_two_distances_identical(keypoints_payload):
    names = keypoints_payload["names"]
    frame = keypoints_payload["frames"][0]
    near = posekit.normalise(frame["keypoints"], frame["scores"], names)
    # Same pose, filmed twice as close and shifted across the frame.
    far = [[x * 2 + 40, y * 2 - 15] for x, y in frame["keypoints"]]
    assert posekit.pose_distance(near, posekit.normalise(far, frame["scores"],
                                                         names)) < 1e-6


def test_normalise_drops_a_joint_the_detector_was_unsure_of(keypoints_payload):
    names = keypoints_payload["names"]
    frame = keypoints_payload["frames"][0]
    scores = list(frame["scores"])
    scores[4] = 0.05
    out = posekit.normalise(frame["keypoints"], scores, names, kpt_thr=0.4)
    assert np.isnan(out[4]).all()
    assert not np.isnan(out[3]).any()


def test_pose_distance_is_infinite_when_nothing_can_be_compared():
    a = np.full((4, 2), np.nan)
    assert posekit.pose_distance(a, a) == float("inf")
    assert posekit.pose_distance(None, np.zeros((4, 2))) == float("inf")


def test_the_renderer_draws_at_the_cell_size_and_draws_something(keypoints_payload):
    frame = keypoints_payload["frames"][6]
    canvas = posekit.draw(frame["keypoints"], frame["scores"],
                          tuple(keypoints_payload["source_size"]),
                          config.STRIP_CELL)
    assert canvas.shape == (config.STRIP_CELL[1], config.STRIP_CELL[0], 3)
    assert canvas.any(), "an all-black skeleton is no skeleton"


def test_the_renderer_letterboxes_rather_than_stretching(keypoints_payload):
    """A squashed rig is a rig with the wrong proportions.

    The same pose rendered into a wide cell and a tall one must keep the same
    aspect, which shows up as the drawn content having the same bounding-box
    shape (up to the letterbox offset).
    """
    frame = keypoints_payload["frames"][0]
    source = tuple(keypoints_payload["source_size"])

    def drawn_aspect(size):
        canvas = posekit.draw(frame["keypoints"], frame["scores"], source, size)
        rows = np.flatnonzero(canvas.any(axis=(1, 2)))
        cols = np.flatnonzero(canvas.any(axis=(0, 2)))
        return (cols[-1] - cols[0]) / (rows[-1] - rows[0])

    assert drawn_aspect((304, 320)) == pytest.approx(drawn_aspect((608, 640)),
                                                     rel=0.1)


def test_render_skeletons_names_files_after_their_frames(keypoints_payload, tmp_path):
    written = pose_stage.render_skeletons(keypoints_payload, tmp_path)
    assert [p.name for p in written][:3] == ["skel_0001.png", "skel_0002.png",
                                             "skel_0003.png"]
    with Image.open(written[0]) as img:
        assert img.size == tuple(config.STRIP_CELL)


def test_a_frame_with_no_person_gets_no_skeleton(keypoints_payload, tmp_path):
    keypoints_payload["frames"][2]["keypoints"] = None
    keypoints_payload["frames"][2]["scores"] = None
    written = pose_stage.render_skeletons(keypoints_payload, tmp_path)
    names = {p.name for p in written}
    assert "skel_0003.png" not in names
    assert len(written) == len(keypoints_payload["frames"]) - 1


def test_stale_skeletons_are_cleared_before_a_re_render(keypoints_payload, tmp_path):
    (tmp_path / "skel_0099.png").write_bytes(b"not a png")
    written = pose_stage.render_skeletons(keypoints_payload, tmp_path)
    assert not (tmp_path / "skel_0099.png").exists()
    assert written


def test_best_person_prefers_the_most_confident_skeleton():
    keypoints = np.zeros((3, 18, 2))
    scores = np.array([[0.1] * 18, [0.9] * 18, [0.5] * 18])
    assert pose_stage.best_person(keypoints, scores) == 1
    assert pose_stage.best_person(np.zeros((0, 18, 2)), np.zeros((0, 18))) is None


def test_pose_does_not_re_infer_when_keypoints_are_cached(posed_job, monkeypatch):
    """The whole point of caching: pick/review/strip must never pay for inference."""
    def explode(*args, **kwargs):
        raise AssertionError("estimate() was called despite a cached "
                             "keypoints.json")

    monkeypatch.setattr(pose_stage, "estimate", explode)
    job = pose_stage.pose(posed_job.name, work_dir=posed_job.root.parent)
    assert job.skeletons()


def test_a_cache_that_does_not_match_the_frames_on_disk_is_refused(posed_job,
                                                                   monkeypatch):
    payload = json.loads(posed_job.keypoints_path.read_text())
    payload["frames"] = payload["frames"][:4]
    posed_job.keypoints_path.write_text(json.dumps(payload))
    monkeypatch.setattr(pose_stage, "estimate",
                        lambda *a, **k: pytest.fail("should not re-infer"))
    with pytest.raises(RuntimeError, match="the clip changed"):
        pose_stage.pose(posed_job.name, work_dir=posed_job.root.parent)


def test_pose_without_frames_says_to_run_extract(work_dir):
    from anim.job import Job, new_manifest

    job = Job.open("empty", work_dir)
    job.write_manifest(new_manifest(
        clip="x.mp4", clip_sha256="", fps=12, frames=0, action="strike",
        slug="knight", frame_size=[1, 1], crop=None, source_fps=None,
        created="2026-09-13T00:00:00+00:00"))
    with pytest.raises(FileNotFoundError, match="anim extract"):
        pose_stage.pose("empty", work_dir=work_dir)


# ---------------------------------------------------------------------------
# Opt-in: the real model, on the real photograph
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("ANIM_POSE_SMOKE"),
                    reason="set ANIM_POSE_SMOKE=1 to download the DWPose ONNX "
                           "weights (~200 MB) and run inference on the CC0 "
                           "fixture photo")
def test_smoke_dwpose_finds_the_person_in_the_cc0_photo(tmp_path):
    """End-to-end proof the inference path works on a real human frame.

    Opt-in because it needs the network on first run. The fixture is a CC0
    photograph - see tests/fixtures/FIXTURES_NOTICE.md.
    """
    assert FIXTURE_PHOTO.exists(), FIXTURE_PHOTO
    frames = []
    for index in range(2):
        path = tmp_path / f"{index + 1:04d}.png"
        with Image.open(FIXTURE_PHOTO) as img:
            img.convert("RGB").save(path)
        frames.append(path)
    payload = pose_stage.estimate(frames)
    assert payload["rig"] in ("openpose18", "openpose134")
    assert len(payload["frames"]) == 2
    assert payload["frames"][0]["keypoints"], "no person detected"
    names = payload["names"]
    normalised = posekit.normalise(payload["frames"][0]["keypoints"],
                                   payload["frames"][0]["scores"], names)
    assert normalised is not None, "hips and shoulders were not both found"
    written = pose_stage.render_skeletons(payload, tmp_path / "pose")
    assert len(written) == 2
