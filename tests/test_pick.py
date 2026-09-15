"""`anim pick`: the key-pose heuristic and the never-clobber-a-human rule."""

from __future__ import annotations

import json

import pytest

from anim import config, posekit
from anim.stages import pick as pick_stage


def series(payload):
    poses, _ = posekit.series_from(payload)
    return poses


def test_the_budget_comes_from_the_art_bible():
    assert config.frames_for_action("strike") == 4
    assert config.frames_for_action("hit") == 2
    assert config.frames_for_action("idle") == 2
    assert config.frames_for_action("walk") == 6
    assert config.frames_for_action("SHOOT") == 4, "case does not matter"
    assert config.frames_for_action("nonsense") == config.DEFAULT_FRAMES


def test_the_endpoints_are_always_picked(keypoints_payload):
    poses = series(keypoints_payload)
    picks = pick_stage.auto_picks(poses, 4)
    assert picks[0] == 0
    assert picks[-1] == len(poses) - 1


def test_the_extreme_of_the_motion_is_picked(keypoints_payload):
    """Frame 6 is the top of the swing - the pose the whole clip is about."""
    picks = pick_stage.auto_picks(series(keypoints_payload), 4)
    assert 6 in picks


def test_exactly_n_poses_come_back(keypoints_payload):
    poses = series(keypoints_payload)
    for n in (2, 3, 4, 5, 6):
        assert len(pick_stage.auto_picks(poses, n)) == n


def test_picks_are_in_frame_order(keypoints_payload):
    picks = pick_stage.auto_picks(series(keypoints_payload), 5)
    assert picks == sorted(picks)


def test_picks_are_deterministic(keypoints_payload):
    poses = series(keypoints_payload)
    assert pick_stage.auto_picks(poses, 4) == pick_stage.auto_picks(poses, 4)


def test_near_identical_poses_are_not_all_picked(keypoints_payload):
    """A clip padded with a long still hold must not spend its budget on it."""
    frames = keypoints_payload["frames"]
    still = json.loads(json.dumps(frames[0]))
    for extra in range(10):
        copy = json.loads(json.dumps(still))
        copy["index"] = len(frames)
        frames.append(copy)
    poses = series(keypoints_payload)
    picks = pick_stage.auto_picks(poses, 4, dedup=0.12)
    duplicates = [p for p in picks if p >= 13]
    assert len(duplicates) <= 1, f"the still tail swallowed the budget: {picks}"


def test_a_clip_with_no_motion_still_fills_its_budget(keypoints_payload):
    """An idle take is all duplicates; the strip still has N cells to fill."""
    frames = keypoints_payload["frames"]
    for frame in frames:
        frame["keypoints"] = json.loads(json.dumps(frames[0]["keypoints"]))
    poses = series(keypoints_payload)
    assert len(pick_stage.auto_picks(poses, 4)) == 4


def test_asking_for_more_poses_than_frames_returns_every_frame(keypoints_payload):
    poses = series(keypoints_payload)
    assert pick_stage.auto_picks(poses, 99) == list(range(len(poses)))


def test_frames_with_no_person_are_never_picked(keypoints_payload):
    keypoints_payload["frames"][6]["keypoints"] = None
    keypoints_payload["frames"][6]["scores"] = None
    picks = pick_stage.auto_picks(series(keypoints_payload), 5)
    assert 6 not in picks


def test_the_signals_peak_where_the_motion_does(keypoints_payload):
    curves = pick_stage.signals(series(keypoints_payload))
    deviation = curves["deviation"]
    assert deviation.index(max(deviation)) == 6
    assert deviation[0] == 0.0, "frame 0 is the rest pose, by definition"


def test_pick_writes_picks_json_with_auto_and_picks_in_step(posed_job):
    record = pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    assert record["picks"] == record["auto"]
    assert record["n"] == 4, "strike's budget"
    assert record["action"] == "strike"
    on_disk = json.loads(posed_job.picks_path.read_text())
    assert on_disk["picks"] == record["picks"]


def test_an_explicit_n_overrides_the_budget(posed_job):
    record = pick_stage.pick(posed_job.name, n=2, work_dir=posed_job.root.parent)
    assert len(record["picks"]) == 2


def test_pick_refuses_to_overwrite_a_hand_edited_file(posed_job):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    record = json.loads(posed_job.picks_path.read_text())
    record["picks"] = [0, 1, 2]          # as if a human clicked
    posed_job.picks_path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="edited by hand"):
        pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)


def test_force_overwrites_a_hand_edited_file(posed_job):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    record = json.loads(posed_job.picks_path.read_text())
    record["picks"] = [0, 1, 2]
    posed_job.picks_path.write_text(json.dumps(record))
    fresh = pick_stage.pick(posed_job.name, force=True,
                            work_dir=posed_job.root.parent)
    assert fresh["picks"] == fresh["auto"] != [0, 1, 2]


def test_was_edited_only_fires_on_a_real_difference():
    assert not pick_stage.was_edited(None)
    assert not pick_stage.was_edited({"picks": [1, 2], "auto": [1, 2]})
    assert pick_stage.was_edited({"picks": [1, 3], "auto": [1, 2]})


def test_an_action_the_art_bible_gives_no_motion_is_refused(posed_job):
    with pytest.raises(ValueError, match="no motion"):
        pick_stage.pick(posed_job.name, action="wait",
                        work_dir=posed_job.root.parent)


def test_pick_without_keypoints_says_to_run_pose(posed_job):
    posed_job.keypoints_path.unlink()
    with pytest.raises(FileNotFoundError, match="anim pose"):
        pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)


# ---------------------------------------------------------------------------
# The timing shape (the strip timing feature)
# ---------------------------------------------------------------------------
def test_the_template_shape_comes_from_the_art_bible_table():
    assert config.timing_for_action("strike", 4) == ([2, 3, 4, 3], 2)
    assert config.timing_for_action("HIT", 4) == ([1, 3, 2, 2], 0), "case does not matter"
    assert config.timing_for_action("shove", 3) == ([2, 4, 3], 1)
    assert config.timing_for_action("strike", 3) is None, "no row, no shape"
    for (action, n), (weights, anchor) in config.ACTION_TIMING.items():
        assert len(weights) == n and all(w > 0 for w in weights), (action, n)
        assert 0 <= anchor < n, (action, n)


def test_clip_timing_is_the_gap_to_the_next_pick_with_a_tail_of_one():
    """The knight's Sword_Attack: 19 frames, picks [0, 6, 10, 18]."""
    assert pick_stage.clip_timing([0, 6, 10, 18], 19) == [6, 4, 8, 1]
    assert pick_stage.clip_timing([3, 4, 5], 12) == [1, 1, 1]
    with pytest.raises(ValueError, match="outside"):
        pick_stage.clip_timing([0, 6, 30], 19)


def test_pick_records_the_template_shape_by_default(posed_job):
    record = pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    assert record["timing"] == {"weights": [2, 3, 4, 3], "anchor": 2,
                                "source": "template:strike"}
    assert record["auto_timing"] == record["timing"]
    assert record["timing"] is not record["auto_timing"], "two copies, or an edit is invisible"
    on_disk = json.loads(posed_job.picks_path.read_text())
    assert on_disk["timing"] == record["timing"]


def test_a_count_with_no_template_row_records_no_shape(posed_job):
    record = pick_stage.pick(posed_job.name, n=3, work_dir=posed_job.root.parent)
    assert record["timing"] is None and record["auto_timing"] is None


def test_clip_mode_takes_the_gaps_and_the_template_anchor(posed_job):
    record = pick_stage.pick(posed_job.name, timing="clip",
                             work_dir=posed_job.root.parent)
    shape = record["timing"]
    assert shape["source"] == "clip"
    assert len(shape["weights"]) == 4
    assert shape["weights"] == pick_stage.clip_timing(record["picks"], 13)
    assert shape["anchor"] == 2, "the template says which pose is the contact"


def test_clip_mode_refuses_without_a_template_row_to_place_the_anchor(posed_job):
    with pytest.raises(ValueError, match="place the anchor"):
        pick_stage.pick(posed_job.name, n=3, timing="clip",
                        work_dir=posed_job.root.parent)


def test_a_hand_edited_timing_alone_is_protected_from_the_picker(posed_job):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    record = json.loads(posed_job.picks_path.read_text())
    record["timing"]["weights"] = [1, 2, 6, 3]  # the owner lengthened the contact hold
    posed_job.picks_path.write_text(json.dumps(record))
    before = posed_job.picks_path.read_bytes()
    with pytest.raises(RuntimeError, match="timing"):
        pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    assert posed_job.picks_path.read_bytes() == before, "the refusal must not touch the file"
    fresh = pick_stage.pick(posed_job.name, force=True, work_dir=posed_job.root.parent)
    assert fresh["timing"]["weights"] == [2, 3, 4, 3]


def test_was_edited_sees_a_timing_difference():
    same = {"picks": [1, 2], "auto": [1, 2],
            "timing": {"weights": [3, 2], "anchor": 0}, "auto_timing": {"weights": [3, 2], "anchor": 0}}
    assert not pick_stage.was_edited(same)
    edited = dict(same, timing={"weights": [4, 2], "anchor": 0})
    assert pick_stage.was_edited(edited)
    assert not pick_stage.was_edited({"picks": [1, 2], "auto": [1, 2]}), \
        "a record from before the feature has no timing keys and is not an edit"
