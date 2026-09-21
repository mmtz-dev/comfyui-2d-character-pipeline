"""`anim pick`: the key-pose heuristic and the never-clobber-a-human rule."""

from __future__ import annotations

import json
from pathlib import Path

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


# ---------------------------------------------------------------------------
# The phase template (the keyframe phase feature)
# ---------------------------------------------------------------------------
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "picks_head_447ad25.json"
STUDIO = ("idle", "walk", "run", "jump", "attack")


def test_the_template_defaults_are_the_studios():
    """sprite-studio.md S2: cap 12; idle 6, walk 8, run 8, jump 10, attack 12."""
    budgets = {a: config.phases_for_action(a)["budget"] for a in STUDIO}
    assert budgets == {"idle": 6, "walk": 8, "run": 8, "jump": 10, "attack": 12}
    assert max(budgets.values()) <= config.PHASE_CAP == 12
    assert config.PHASE_FPS == 12
    assert config.PHASE_NAMES == ("start", "wind-up", "action", "wind-down", "end")
    for action in STUDIO:
        row = config.ACTION_PHASES[action]
        assert len(row["split"]) == len(row["span"]) == 5, action
    assert {a: config.ACTION_PHASES[a]["loop"] for a in STUDIO} == \
        {"idle": True, "walk": True, "run": True, "jump": False, "attack": False}


def test_the_studio_names_map_to_the_pipelines_verbs_and_the_rigs_clips():
    assert config.studio_action("attack") == config.studio_action("STRIKE") == "attack"
    assert config.studio_action("idle") == "idle" and config.studio_action("hit") is None
    assert config.STUDIO_ACTIONS["attack"]["verb"] == "strike"
    assert config.STUDIO_ACTIONS["attack"]["clips"] == \
        {"kaykit": "1H_Melee_Attack_Chop", "quaternius": "Sword_Attack"}
    for action in STUDIO:
        assert set(config.STUDIO_ACTIONS[action]["clips"]) == {"kaykit", "quaternius"}
    assert config.phases_for_action("strike")["action"] == "attack"
    assert config.phases_for_action("strike")["verb"] == "strike"
    assert config.phases_for_action("hit") is None


def test_the_default_is_on_only_for_a_studio_action_the_bible_does_not_budget():
    assert config.phases_default("attack") and config.phases_default("run") \
        and config.phases_default("jump")
    for verb in ("strike", "idle", "walk", "hit", "death", "nonsense"):
        assert not config.phases_default(verb), verb


def test_the_split_for_each_action_fills_its_budget_with_every_phase_present():
    expected = {
        "idle": {"start": 1, "wind-up": 1, "action": 2, "wind-down": 1, "end": 1},
        "walk": {"start": 1, "wind-up": 2, "action": 2, "wind-down": 2, "end": 1},
        "run": {"start": 1, "wind-up": 2, "action": 2, "wind-down": 2, "end": 1},
        "jump": {"start": 1, "wind-up": 3, "action": 3, "wind-down": 2, "end": 1},
        "attack": {"start": 1, "wind-up": 3, "action": 4, "wind-down": 2, "end": 2},
    }
    for action, frames in expected.items():
        plan = pick_stage.phase_plan(action)
        assert plan.frames == frames, action
        assert sum(plan.frames.values()) == plan.budget
        assert len(plan.tags()) == plan.budget
        assert len(plan.keys()) == 5 and plan.keys()[0] == 0 and plan.keys()[-1] == plan.budget - 1


def test_ease_means_the_action_phase_is_sampled_denser_than_rest():
    for action in STUDIO:
        plan = pick_stage.phase_plan(action)
        density = {n: plan.frames[n] / plan.span[n] for n in config.PHASE_NAMES}
        assert density["action"] > density["start"], action
        assert density["action"] > density["end"], action
        assert plan.frames["action"] == max(plan.frames.values()), action
        positions = pick_stage.phase_positions(plan)
        assert positions == sorted(positions) and positions[0] == 0.0
        # The picks follow: on a 60-frame clip the action window holds more
        # picks per frame than the rest windows do.
        picks = pick_stage.phase_frames(60, plan)
        tags = pick_stage.phase_tags(plan, picks, 60)
        assert len(picks) == plan.budget and picks == sorted(set(picks))
        assert tags == plan.tags()


def test_a_one_shot_spends_one_frame_on_anticipation_and_a_loop_does_not():
    for action in ("jump", "attack"):
        plan = pick_stage.phase_plan(action)
        assert plan.anticipation
        without = pick_stage.allocate(plan.budget, config.ACTION_PHASES[action]["split"],
                                      anticipation=False)
        assert plan.frames["wind-up"] >= without[1] + 1 - 1  # never fewer than the plain split
        slot = plan.anticipation_slot()
        assert plan.tags()[slot] == "wind-up" and plan.tags()[slot + 1] == "action"
        assert pick_stage.phase_positions(plan)[-1] == 1.0, "a one-shot ends on the last frame"
    for action in ("idle", "walk", "run"):
        plan = pick_stage.phase_plan(action)
        assert not plan.anticipation and plan.anticipation_slot() is None
        assert pick_stage.phase_positions(plan)[-1] < 1.0, \
            "a loop's last pick stops short of the frame that is its first again"


def test_the_even_spread_is_the_old_even_spread_to_the_frame():
    plan = pick_stage.phase_plan("attack")
    for frames in (12, 13, 14, 19, 37, 60):
        assert pick_stage.phase_frames(frames, plan, "even") == \
            pick_stage._even_spread(list(range(frames)), 12)
    assert pick_stage.phase_frames(9, plan, "even") == list(range(9))
    assert pick_stage.phase_instants(1.5, plan, "even")[-1] == 1.5
    assert len(pick_stage.phase_instants(1.5, plan)) == 12


def test_a_short_clip_gives_every_frame_and_the_hold_rule_keeps_the_length():
    """9 frames against attack's 12: all nine, tagged by window, and the
    weights still sum to the budget so the clip lasts 12 / 12 s."""
    plan = pick_stage.phase_plan("attack")
    picks = pick_stage.phase_frames(9, plan)
    assert picks == list(range(9))
    tags = pick_stage.phase_tags(plan, picks, 9)
    assert len(tags) == 9 and set(tags) <= set(config.PHASE_NAMES)
    shape = pick_stage.phase_timing(plan, tags)
    assert abs(sum(shape["weights"]) - plan.budget) < 1e-6
    assert sum(shape["durations_ms"]) == pytest.approx(1000 * plan.budget / 12, abs=len(tags))
    assert shape["anchor"] == tags.index("action")
    assert shape["source"] == "phases:attack" and shape["fps"] == 12
    # A full pick: one tick per frame, 83 ms each.
    full = pick_stage.phase_timing(plan, plan.tags())
    assert full["weights"] == [1] * 12 and full["durations_ms"] == [83] * 12
    assert full["phases"] == plan.tags() and full["loop"] is False


def test_an_empty_phase_lends_its_time_to_its_neighbour():
    plan = pick_stage.phase_plan("attack")
    tags = ["start", "action", "action", "end"]          # no wind-up, no wind-down
    shape = pick_stage.phase_timing(plan, tags)
    assert abs(sum(shape["weights"]) - 12) < 1e-6
    assert shape["weights"][0] == plan.frames["start"] + plan.frames["wind-up"]
    assert shape["anchor"] == 1


def _walk_payload(keypoints_payload, closes: bool):
    frames = keypoints_payload["frames"]
    if not closes:
        # The last frame stays at the top of the swing with the legs thrown
        # forward: a take that never came home.
        kp = json.loads(json.dumps(frames[6]["keypoints"]))
        for joint in (9, 10, 12, 13):          # knees and ankles
            kp[joint] = [kp[joint][0] + 150, kp[joint][1] - 60]
        frames[-1]["keypoints"] = kp
    return keypoints_payload


def test_loop_closure_passes_and_fails_on_synthetic_keypoints(keypoints_payload):
    poses = series(_walk_payload(json.loads(json.dumps(keypoints_payload)), closes=True))
    ok = pick_stage.loop_closure(poses, 0, len(poses) - 1)
    assert ok["ok"] is True and ok["metric"] == "pose" and ok["distance"] <= ok["tolerance"]
    poses = series(_walk_payload(keypoints_payload, closes=False))
    bad = pick_stage.loop_closure(poses, 0, len(poses) - 1)
    assert bad["ok"] is False and bad["distance"] > bad["tolerance"]
    assert bad["first"] == 0 and bad["last"] == len(poses) - 1


def test_loop_closure_falls_back_to_pixels_when_a_pose_is_missing(posed_job, keypoints_payload):
    poses = series(keypoints_payload)
    poses[-1] = None
    verdict = pick_stage.loop_closure(poses, 0, len(poses) - 1, frames_dir=posed_job.frames_dir)
    assert verdict["metric"] == "pixel" and verdict["ok"] in (True, False)
    assert verdict["tolerance"] == config.LOOP_PIXEL_TOLERANCE
    unknown = pick_stage.loop_closure(poses, 0, len(poses) - 1)
    assert unknown["metric"] == "none" and unknown["ok"] is None


def test_pick_with_phases_records_the_plan_the_tags_and_the_phase_timing(posed_job):
    record = pick_stage.pick(posed_job.name, phases=True, work_dir=posed_job.root.parent)
    block = record["phases"]
    assert block["plan"]["action"] == "attack" and block["plan"]["verb"] == "strike"
    assert record["n"] == 12 == block["plan"]["budget"]
    assert len(record["picks"]) == 12 and record["picks"] == record["auto"]
    assert record["picks"][0] == 0 and record["picks"][-1] == 12
    assert block["tags"] == pick_stage.phase_plan("attack").tags()
    assert block["spread"] == "phase" and block["closure"] is None
    assert block["anticipation"] == 3 and block["tags"][3] == "wind-up"
    assert record["timing"]["source"] == "phases:attack"
    assert record["timing"]["weights"] == [1] * 12
    assert record["timing"]["phases"] == block["tags"]
    assert record["timing"]["anchor"] == 4
    assert record["auto_timing"] == record["timing"]
    on_disk = json.loads(posed_job.picks_path.read_text())
    assert on_disk["phases"] == block


def test_a_loop_action_records_its_closure_and_the_review_page_shows_it(posed_job):
    from anim.stages import review as review_stage

    record = pick_stage.pick(posed_job.name, action="idle", phases=True,
                             work_dir=posed_job.root.parent)
    closure = record["phases"]["closure"]
    assert closure["metric"] == "pose" and closure["ok"] is True
    assert record["timing"]["loop"] is True and record["timing"]["anchor"] == 2
    page = review_stage.ReviewApp(posed_job).page().decode("utf-8")
    assert "loop closes" in page and "phases: idle 6" in page
    assert ">wind-up<" in page and ">action<" in page


def test_phases_off_is_byte_identical_for_an_existing_job(posed_job):
    """The golden file is picks.json as HEAD 447ad25 wrote it for this very
    fixture, before the feature. `strike` is an art-bible verb, so the
    default is off too - both spellings must give the same bytes."""
    pick_stage.pick(posed_job.name, phases=False, work_dir=posed_job.root.parent)
    assert posed_job.picks_path.read_bytes() == GOLDEN.read_bytes()
    pick_stage.pick(posed_job.name, force=True, work_dir=posed_job.root.parent)
    assert posed_job.picks_path.read_bytes() == GOLDEN.read_bytes()


def test_the_phase_default_follows_the_action(posed_job):
    on = pick_stage.pick(posed_job.name, action="attack", work_dir=posed_job.root.parent)
    assert "phases" in on and on["n"] == 12
    off = pick_stage.pick(posed_job.name, force=True, work_dir=posed_job.root.parent)
    assert "phases" not in off and off["n"] == 4
    with pytest.raises(ValueError, match="not a studio action"):
        pick_stage.pick(posed_job.name, action="hit", phases=True, force=True,
                        work_dir=posed_job.root.parent)
    with pytest.raises(ValueError, match="caps a clip"):
        pick_stage.pick(posed_job.name, action="attack", n=20, force=True,
                        work_dir=posed_job.root.parent)


def test_the_projects_timing_json_wins_over_the_defaults(tmp_path):
    path = tmp_path / "timing.json"
    path.write_text(json.dumps({
        "attack": {"budget": 10},
        "run": {"split": [1, 1, 6, 1, 1], "loop": True},
        "units": {"hound": {"strike": {"budget": 8, "loop": True}}},
    }))
    row = config.phases_for_action("strike", timing_file=path)
    assert row["budget"] == 10 and row["source"] == "timing.json"
    assert row["split"] == config.ACTION_PHASES["attack"]["split"], "the rest is kept"
    assert config.phases_for_action("run", timing_file=path)["split"] == (1, 1, 6, 1, 1)
    hound = config.phases_for_action("attack", slug="hound", timing_file=path)
    assert hound["budget"] == 8 and hound["loop"] is True and hound["source"] == "timing.json:hound"
    assert config.phases_for_action("attack", slug="knight", timing_file=path)["budget"] == 10
    assert config.phases_for_action("idle", timing_file=path)["source"] == "template"
    assert config.phases_for_action("idle", timing_file=tmp_path / "none.json")["budget"] == 6
    plan = pick_stage.phase_plan("attack", slug="hound", timing_file=path)
    assert plan.budget == 8 and plan.loop and not plan.anticipation
    path.write_text(json.dumps({"attack": {"budget": 40}}))
    with pytest.raises(ValueError, match="caps a clip"):
        config.phases_for_action("attack", timing_file=path)
    path.write_text(json.dumps({"attack": {"nope": 1}}))
    with pytest.raises(ValueError, match="unknown key"):
        config.phases_for_action("attack", timing_file=path)
