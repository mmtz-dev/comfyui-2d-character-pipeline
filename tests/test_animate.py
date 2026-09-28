"""The animate tab feature: the frame row from the phase template, adding /
removing / marking frames, statuses and the two-step Generate all, draw-from
resolution and the in-between refusal, mirror, the output the export reads,
the CLI's gate, and the API.

The base comes from the real concept stage against FakeComfy (a small canned
image, so the skeletons render fast). Most tests work at s270, whose view is
the accepted base itself; the mirror tests build their partner cell by hand."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageOps

from fixtures.fakecomfy import FakeComfy
from test_api import FAKE_STAGE, wait_for

from anim import cli, comfy, config, curation
from anim.api import runs as runs_api
from anim.api.app import create_app
from anim.job import Job
from anim.stages import animate as an
from anim.stages import concept as concept_stage
from anim.stages import export as export_stage
from anim.stages import pick as pick_stage

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "gpu" / "workflows"
REGISTRY = REPO / "gpu" / "models.registry.tsv"
UAL = REPO / "rigs" / "quaternius" / "UAL1_Standard.glb"
CELL = (128, 128)
UNIT = "knight"
FRONT = "s270"

needs_ual = pytest.mark.skipif(not UAL.exists(), reason="rigs/quaternius/UAL1_Standard.glb is not vendored")


def figure_png(size=CELL, colour=(40, 90, 50), shift: int = 0) -> bytes:
    """A dark, deliberately lopsided figure on white - something the matte can
    find and a flip visibly changes."""
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.rectangle((w // 3 + shift, h // 6, 2 * w // 3 + shift, h - 4), fill=colour)
    draw.rectangle((2 * w // 3 + shift, h // 3, w - 8, h // 3 + 6), fill=(150, 20, 20))
    draw.ellipse((w // 3, 2, 2 * w // 3, h // 4), fill=(20, 20, 20))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _base(work_dir: Path, *, pick: bool = True) -> None:
    with FakeComfy(png=figure_png(CELL, colour=(60, 60, 140))) as fake:
        concept_stage.concept(UNIT, description="a knight", count=1, seed=10,
                              client=comfy.ComfyClient(fake.url), workflow_dir=WORKFLOWS,
                              work_dir=work_dir)
    if pick:
        concept_stage.pick(UNIT, "00", parts="closed helm, kite shield", work_dir=work_dir)


def _generate(work_dir: Path, frame, *, direction: str = FRONT, png: bytes | None = None,
              **options):
    options.setdefault("skeleton", False)
    with FakeComfy(png=png or figure_png()) as fake:
        rows = an.generate(UNIT, "attack", direction, frame, client=comfy.ComfyClient(fake.url),
                           workflow_dir=WORKFLOWS, work_dir=work_dir, **options)
    return rows, fake


def _approve(work_dir: Path, frame, direction: str = FRONT, **kw):
    return an.judge(UNIT, "attack", direction, frame, "approved", work_dir=work_dir, **kw)


def _fake_frames(statuses: list[str], keys: list[int]) -> list[dict]:
    """A row of frames in the given statuses, without any files: the pure
    functions only read the record."""
    frames = []
    for i, status in enumerate(statuses):
        frame = an.frame_row(f"f{i + 1:02d}", key=i in keys, phase="action" if i in keys else None,
                             at=i / max(1, len(statuses) - 1))
        if status != "empty":
            frame["candidates"] = [{"index": 0, "file": "x.png", "source": "animate",
                                    "seconds": None if status == "generating" else 1.0}]
            if status != "generating":
                frame["current"] = 0
            if status in ("approved", "rejected"):
                frame["verdict"] = {"state": status, "index": 0, "sha256": "x"}
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# The row: defaults, add / remove / mark
# ---------------------------------------------------------------------------
def test_the_default_row_is_the_phase_template():
    frames = an.default_layout("attack")
    assert len(frames) == config.ACTION_PHASES["attack"]["budget"] == 12
    keys = [i for i, f in enumerate(frames) if f["key"]]
    assert keys == [0, 2, 6, 9, 11], "attack's keys: Start / Wind-up / Action / Wind-down / End"
    assert [frames[i]["phase"] for i in keys] == list(config.PHASE_NAMES)
    assert [an.frame_label(frames[i], i) for i in keys] == [
        "Start", "Wind-up", "Action", "Wind-down", "End"]
    assert an.frame_label(frames[3], 3) == "Frame 4"
    plan = pick_stage.phase_plan("attack")
    assert [f["at"] for f in frames] == pytest.approx(pick_stage.phase_positions(plan), abs=1e-5)
    assert [f["id"] for f in frames] == [f"f{i:02d}" for i in range(1, 13)]
    for action in config.STUDIO_ACTIONS:
        row = an.default_layout(action)
        plan = pick_stage.phase_plan(action)
        assert len(row) == plan.budget
        assert [i for i, f in enumerate(row) if f["key"]] == plan.keys(), action
    assert an.default_layout("strike")[2]["phase"] == "wind-up", "the verb works too"


def test_add_remove_and_mark_keep_ids_and_attempts(work_dir):
    job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    row = an.add_frame(UNIT, "attack", FRONT, 3, work_dir=work_dir)
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert len(record["frames"]) == 13 and record["frames"][3]["id"] == row["id"] == "f13"
    assert not row["key"] and record["frames"][2]["at"] < row["at"] < record["frames"][4]["at"]
    first = an.add_frame(UNIT, "attack", FRONT, 0, work_dir=work_dir)
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert record["frames"][0]["id"] == first["id"] == "f14" and first["at"] <= record["frames"][1]["at"]
    # Frame numbers move, ids do not.
    assert an.find_frame(record, "f13") == 4 and an.find_frame(record, "5") == 4
    assert an.find_frame(record, "wind-up") == 3
    marked = an.set_key(UNIT, "attack", FRONT, "f13", key=True, phase="action", work_dir=work_dir)
    assert marked["key"] and marked["phase"] == "action"
    with pytest.raises(ValueError, match="2 keyframes are action"):
        an.find_frame(an.open_job(UNIT, "attack", FRONT, work_dir)[1], "action")
    with pytest.raises(ValueError, match="phase is one of"):
        an.set_key(UNIT, "attack", FRONT, "f13", key=True, phase="dance", work_dir=work_dir)
    unmarked = an.set_key(UNIT, "attack", FRONT, "f13", key=False, work_dir=work_dir)
    assert not unmarked["key"] and unmarked["phase"] is None
    # A removed frame keeps its attempts in the record.
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    record["frames"][4]["candidates"] = [{"index": 0, "file": "f13_00.png"}]
    an.save(job, record)
    gone = an.remove_frame(UNIT, "attack", FRONT, "f13", work_dir=work_dir)
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert len(record["frames"]) == 13 and "f13" not in [f["id"] for f in record["frames"]]
    assert record["removed"][0]["id"] == "f13" and record["removed"][0]["candidates"] == \
        gone["candidates"] == [{"index": 0, "file": "f13_00.png"}]
    again = an.add_frame(UNIT, "attack", FRONT, 1, work_dir=work_dir)
    assert again["id"] == "f15", "a removed frame's id (and so its files) is never reused"
    with pytest.raises(ValueError, match="0..14"):
        an.add_frame(UNIT, "attack", FRONT, 99, work_dir=work_dir)


def test_the_row_is_capped(work_dir, monkeypatch):
    monkeypatch.setattr(config, "ANIMATE_MAX_FRAMES", 13)
    an.add_frame(UNIT, "attack", FRONT, 0, work_dir=work_dir)
    with pytest.raises(ValueError, match="at most 13"):
        an.add_frame(UNIT, "attack", FRONT, 0, work_dir=work_dir)
    job, record = an.open_job(UNIT, "idle", FRONT, work_dir)
    record["frames"] = record["frames"][:1]
    an.save(job, record)
    with pytest.raises(ValueError, match="at least one frame"):
        an.remove_frame(UNIT, "idle", FRONT, 1, work_dir=work_dir)


# ---------------------------------------------------------------------------
# Statuses, neighbours, the two-step Generate all
# ---------------------------------------------------------------------------
def test_frame_status():
    frames = _fake_frames(["empty", "generating", "review", "approved", "rejected"], keys=[])
    assert [an.frame_status(f) for f in frames] == ["empty", "generating", "review",
                                                    "approved", "rejected"]
    frames[3]["current"] = 5
    assert an.frame_status(frames[3]) == "review", "another attempt shown: not the judged one"
    assert an.counts(frames)["review"] == 2 and an.counts(frames)["total"] == 5


def test_neighbour_keys_and_the_loop_wrap():
    frames = _fake_frames(["empty"] * 8, keys=[0, 3, 6])
    assert an.neighbour_keys(frames, 1, loop=False) == (0, 3)
    assert an.neighbour_keys(frames, 3, loop=False) == (0, 6), "never the frame itself"
    assert an.neighbour_keys(frames, 7, loop=False) == (6, None)
    assert an.neighbour_keys(frames, 7, loop=True) == (6, 0), "a cycle wraps round"
    assert an.neighbour_keys(frames, 0, loop=True) == (6, 3)


def test_generate_all_is_two_steps_with_approval_between():
    # Step 1: keyframes that are empty or rejected; a keyframe to review is left alone.
    frames = _fake_frames(["empty", "empty", "review", "empty", "rejected", "empty"], keys=[0, 2, 4])
    step = an.next_step(frames)
    assert step["step"] == "keys" and step["targets"] == [0, 4] and step["waiting"] == [2]
    assert an.batch_targets(frames, "all") == [0, 4], "in-betweens wait for the keys"
    assert an.batch_targets(frames, "keys") == [0, 2, 4], "keyframes only: every unapproved key"
    assert an.batch_targets(frames, "rejected") == [4]
    # Nothing left to draw but a key still to review: Generate all stops.
    frames = _fake_frames(["approved", "empty", "review", "empty", "approved", "empty"], keys=[0, 2, 4])
    step = an.next_step(frames)
    assert step == {"step": "keys", "targets": [], "waiting": [2]}
    # Step 2: every keyframe approved - the same press fills the in-betweens.
    frames = _fake_frames(["approved", "empty", "approved", "rejected", "approved", "review"], keys=[0, 2, 4])
    step = an.next_step(frames)
    assert step == {"step": "inbetweens", "targets": [1, 3], "waiting": [5]}
    assert an.batch_targets(frames, "inbetweens") == [1, 3]
    done = _fake_frames(["approved"] * 4, keys=[0, 3])
    assert an.next_step(done)["step"] == "done" and an.batch_targets(done, "all") == []
    with pytest.raises(ValueError, match="batch is one of"):
        an.batch_targets(done, "some")


# ---------------------------------------------------------------------------
# Directions, prompts, draw from
# ---------------------------------------------------------------------------
def test_direction_names_views_and_pairs():
    assert an.direction_name("left") == "s180" and an.direction_name("front") == "s270"
    assert an.direction_name("S45") == "s45"
    with pytest.raises(ValueError, match="top-down"):
        an.direction_name("t90")
    assert [config.view_for_direction(d) for d in an.side_directions()] == [
        "right", "back_right", "back", "back_left", "left", "front_left", "front", "front_right"]
    pairs = {d: config.mirror_direction(d) for d in an.side_directions()}
    assert pairs == {"s0": "s180", "s45": "s135", "s90": None, "s135": "s45",
                     "s180": "s0", "s225": "s315", "s270": None, "s315": "s225"}


def test_prompts_name_each_image():
    assert "moving from the wind-up toward the action" in config.animate_inbetween_prompt(
        "attack", "wind-up", "action")
    text = config.animate_prompt("mid-swing", ["before", "after", "skeleton"], "a hood")
    assert text.startswith("Redraw the same pixel art character from image1, mid-swing.")
    assert "image1 is the same character a moment earlier and image2 a moment later" in text
    assert "skeleton in image3" in text and "no skeleton lines" in text
    view = config.animate_prompt("x", ["view", "before", "after"], None)
    assert "image2 is the same character a moment earlier and image3" in view
    with pytest.raises(ValueError, match="what the pose is"):
        config.animate_prompt(" ", ["view"], None)
    frames = an.default_layout("attack")
    assert an.default_prompt("attack", frames, 2, loop=False) == \
        config.animate_phase_prompt("attack", "wind-up")
    assert an.default_prompt("attack", frames, 4, loop=False) == \
        config.animate_inbetween_prompt("attack", "wind-up", "action")


def test_a_direction_whose_view_is_not_accepted_is_blocked(work_dir):
    with pytest.raises(ValueError, match="no base"):
        an.reference(UNIT, FRONT, work_dir)
    _base(work_dir)
    assert an.reference(UNIT, FRONT, work_dir)["view"] == "front"
    with pytest.raises(ValueError, match="no left view"):
        an.reference(UNIT, "s180", work_dir)
    with pytest.raises(ValueError, match="no left view"):
        an.plan(UNIT, "attack", "left", "wind-up", work_dir=work_dir)


def test_draw_from_and_the_inbetween_refusal(work_dir):
    _base(work_dir)
    # A keyframe from the view.
    _s, rows = an.plan(UNIT, "attack", FRONT, "wind-up", skeleton=False, work_dir=work_dir)
    assert rows[0]["draw_from"] == "view" and [i["role"] for i in rows[0]["images"]] == ["view"]
    # An in-between defaults to the keys, and both must be approved.
    with pytest.raises(ValueError, match=r"Wind-up \(frame 3\) is not approved yet"):
        an.plan(UNIT, "attack", FRONT, 4, skeleton=False, work_dir=work_dir)
    with pytest.raises(ValueError, match="--draw-from is one of"):
        an.plan(UNIT, "attack", FRONT, 4, draw_from="sideways", work_dir=work_dir)
    _generate(work_dir, "wind-up")
    _approve(work_dir, "wind-up")
    with pytest.raises(ValueError, match=r"Action \(frame 7\) is not approved yet"):
        an.plan(UNIT, "attack", FRONT, 4, skeleton=False, work_dir=work_dir)
    # ...but the view alone is always allowed.
    _s, rows = an.plan(UNIT, "attack", FRONT, 4, draw_from="view", skeleton=False,
                       work_dir=work_dir)
    assert [i["role"] for i in rows[0]["images"]] == ["view"]
    _generate(work_dir, "action")
    _approve(work_dir, "action")
    _s, rows = an.plan(UNIT, "attack", FRONT, 4, skeleton=False, work_dir=work_dir)
    assert [(i["role"], i["ref"]) for i in rows[0]["images"]] == [("before", "f03:00"),
                                                                  ("after", "f07:00")]
    assert "moving from the wind-up toward the action" in rows[0]["pose_prompt"]
    _s, rows = an.plan(UNIT, "attack", FRONT, 4, draw_from="both", skeleton=False,
                       work_dir=work_dir)
    assert [i["role"] for i in rows[0]["images"]] == ["view", "before", "after"]
    with pytest.raises(ValueError, match="Qwen-Image-Edit takes 3"):
        an.plan(UNIT, "attack", FRONT, 4, draw_from="both", skeleton=True, work_dir=work_dir)
    # A keyframe can be drawn from its neighbouring keys too - the other keys.
    with pytest.raises(ValueError, match=r"Start \(frame 1\) is not approved"):
        an.plan(UNIT, "attack", FRONT, "wind-up", draw_from="keys", skeleton=False,
                work_dir=work_dir)


def test_the_workflow_wires_every_image_into_both_encoders():
    wf = an.build_workflow("p", seed=3, images=["a.png", "b.png", "c.png"], job="j",
                           workflow_dir=WORKFLOWS)
    encoders = [n for n in wf.values() if n["class_type"] == "TextEncodeQwenImageEditPlus"]
    assert len(encoders) == 2
    for node in encoders:
        assert node["inputs"]["image2"] == ["an_image2", 0]
        assert node["inputs"]["image3"] == ["an_image3", 0]
    with pytest.raises(ValueError):
        an.build_workflow("p", seed=3, images=["a", "b", "c", "d"], job="j", workflow_dir=WORKFLOWS)


@needs_ual
def test_the_skeleton_defaults_to_the_frames_template_instant(work_dir):
    _base(work_dir)
    _s, rows = an.plan(UNIT, "attack", FRONT, "wind-up", work_dir=work_dir)
    block = rows[0]["skeleton"]
    clip = an.rig_clip("attack")
    frames = an.default_layout("attack")
    assert block["frame"] == block["default_frame"] == an.default_skeleton_frame(
        frames[2], clip["duration"])
    assert [i["role"] for i in rows[0]["images"]] == ["view", "skeleton"]
    _s, rows = an.plan(UNIT, "attack", FRONT, "wind-up", skeleton_frame=5, work_dir=work_dir)
    assert rows[0]["skeleton"]["frame"] == 5
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert record["frames"][2]["skeleton_frame"] == 5, "the slider's choice is remembered"
    _s, rows = an.plan(UNIT, "attack", FRONT, "wind-up", work_dir=work_dir)
    assert rows[0]["skeleton"]["frame"] == 5


# ---------------------------------------------------------------------------
# Generate, judge, edit
# ---------------------------------------------------------------------------
def test_generate_judge_select_round_trip(work_dir):
    _base(work_dir)
    rows, fake = _generate(work_dir, "wind-up", count=2, seed=7, prompt="coiled")
    assert len(fake.prompts) == 2 and [r["seed"] for r in rows] == [7, 8]
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    frame = record["frames"][2]
    assert [r["file"] for r in frame["candidates"]] == ["f03_00.png", "f03_01.png"]
    assert frame["current"] == 1 and an.frame_status(frame) == "review"
    assert frame["prompt"] == "coiled"
    rejected = an.judge(UNIT, "attack", FRONT, "wind-up", "rejected", work_dir=work_dir)
    assert an.frame_status(rejected) == "rejected"
    shown = an.select(UNIT, "attack", FRONT, 3, 0, work_dir=work_dir)
    assert shown["current"] == 0 and an.frame_status(shown) == "review", "the other attempt is unjudged"
    approved = _approve(work_dir, "wind-up")
    assert approved["verdict"]["index"] == 0 and an.frame_status(approved) == "approved"
    with pytest.raises(ValueError, match="accepted one"):
        an.hide(UNIT, "attack", FRONT, "f03:00", work_dir=work_dir)
    # The editor's Save over the approved attempt: back to review.
    row = an.replace(UNIT, "attack", FRONT, "f03:00", figure_png(colour=(200, 30, 30)),
                     work_dir=work_dir)
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert row["edited"]["how"] == "hand" and an.frame_status(record["frames"][2]) == "review"
    with pytest.raises(ValueError, match="no attempt to approve"):
        _approve(work_dir, "action")


def test_a_failed_render_leaves_nothing_planned(work_dir):
    _base(work_dir)
    with FakeComfy(fail_prompt=True) as fake:
        with pytest.raises(comfy.ComfyError):
            an.generate(UNIT, "attack", FRONT, "wind-up", skeleton=False,
                        client=comfy.ComfyClient(fake.url), workflow_dir=WORKFLOWS,
                        work_dir=work_dir)
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert record["frames"][2]["candidates"] == [] and an.frame_status(record["frames"][2]) == "empty"


def test_batch_all_draws_the_keys_then_stops(work_dir, capsys):
    _base(work_dir)
    with FakeComfy(png=figure_png()) as fake:
        result = an.generate_batch(UNIT, "attack", FRONT, "all", skeleton=False,
                                   client=comfy.ComfyClient(fake.url), workflow_dir=WORKFLOWS,
                                   work_dir=work_dir)
    assert result["cells"] == {"done": 5, "failed": 0, "blocked": 0, "of": 5}
    assert len(fake.prompts) == 3, "Start and End are view keys: copies, no GPU"
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert [an.frame_status(f) for f in record["frames"]].count("review") == 5
    assert all(an.frame_status(f) == "empty" for f in record["frames"] if not f["key"])
    assert "[5/5]" in capsys.readouterr().out
    # Pressed again before any review: nothing is drawn.
    with FakeComfy(png=figure_png()) as fake:
        again = an.generate_batch(UNIT, "attack", FRONT, "all", skeleton=False,
                                  client=comfy.ComfyClient(fake.url), workflow_dir=WORKFLOWS,
                                  work_dir=work_dir)
    assert again["cells"]["of"] == 0 and fake.prompts == []
    # Approve the keys, and the same press fills the in-betweens from them.
    assert len(an.approve_reviewed(UNIT, "attack", FRONT, work_dir=work_dir)) == 5
    with FakeComfy(png=figure_png()) as fake:
        fill = an.generate_batch(UNIT, "attack", FRONT, "all", skeleton=False,
                                 client=comfy.ComfyClient(fake.url), workflow_dir=WORKFLOWS,
                                 work_dir=work_dir)
    assert fill["cells"] == {"done": 7, "failed": 0, "blocked": 0, "of": 7}
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    inbetween = record["frames"][1]["candidates"][0]
    assert inbetween["draw_from"] == "keys" and [i["role"] for i in inbetween["images"]] == [
        "before", "after"]


# ---------------------------------------------------------------------------
# The view keys feature: start (and a one-shot's end) is a copy of the view
# ---------------------------------------------------------------------------
def test_which_keyframes_are_view_keys(tmp_path):
    assert config.view_key_phases("attack") == ("start", "end")
    assert config.view_key_phases("jump") == ("start", "end")
    for loop in ("idle", "walk", "run"):
        assert config.view_key_phases(loop) == ("start",), f"{loop}'s end flows into its start"
    assert config.view_key_phases("hit") == ()
    path = tmp_path / "timing.json"
    path.write_text(json.dumps({"attack": {"view_keys": False},
                                "units": {"hound": {"walk": {"view_keys": False}},
                                          "knight": {"attack": {"view_keys": True}}}}))
    assert config.view_key_phases("attack", timing_file=path) == ()
    assert config.view_key_phases("attack", slug="knight", timing_file=path) == ("start", "end")
    assert config.view_key_phases("walk", slug="hound", timing_file=path) == ()
    assert config.view_key_phases("walk", timing_file=path) == ("start",)


def test_default_draw_from_copies_a_view_key_once():
    start = an.frame_row("f01", key=True, phase="start", at=0.0)
    action = an.frame_row("f02", key=True, phase="action", at=0.5)
    between = an.frame_row("f03", key=False, phase=None, at=0.7)
    keys = ("start", "end")
    assert an.default_draw_from(start, keys) == "copy"
    assert an.default_draw_from(start) == "view", "view keys off"
    assert an.default_draw_from(action, keys) == "view"
    assert an.default_draw_from(between, keys) == "keys"
    start["candidates"] = [{"index": 0, "source": an.COPY_SOURCE}]
    assert an.default_draw_from(start, keys) == "view", "it has its copy; the next press draws"
    start["candidates"][0]["file_deleted"] = "x"
    assert an.default_draw_from(start, keys) == "copy"


def test_a_view_key_is_the_view_to_review(work_dir):
    _base(work_dir)
    view = an.reference(UNIT, FRONT, work_dir)
    rows, fake = _generate(work_dir, "start")
    assert fake.prompts == [], "a copy is no GPU"
    (row,) = rows
    assert row["source"] == an.COPY_SOURCE and row["draw_from"] == "copy"
    assert row["copy"] == {"view": view["view"], "sha256": view["sha256"]}
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    frame = record["frames"][0]
    assert an.frame_status(frame) == "review", "copied, never approved for you"
    path = Job.open(an.job_name(UNIT, "attack", FRONT), work_dir).root / an.ATTEMPTS_DIR / row["file"]
    assert path.read_bytes() == Path(view["path"]).read_bytes()
    assert not an.copy_stale(row, view["sha256"])
    assert an.copy_stale(row, "0" * 64), "the view was re-accepted since"
    assert not an.copy_stale({"source": "animate"}, "0" * 64)
    # Rejected, the next press draws it from the view rather than copying again.
    an.judge(UNIT, "attack", FRONT, "start", "rejected", work_dir=work_dir)
    rows, fake = _generate(work_dir, "start")
    assert rows[0]["draw_from"] == "view" and len(fake.prompts) == 1
    # Any keyframe can be a copy by hand; an in-between cannot.
    rows, fake = _generate(work_dir, "action", draw_from="copy")
    assert rows[0]["source"] == an.COPY_SOURCE and fake.prompts == []
    with pytest.raises(ValueError, match="only a keyframe"):
        _generate(work_dir, 2, draw_from="copy")


def test_a_batch_blocks_a_frame_whose_keys_are_not_approved(work_dir):
    _base(work_dir)
    _generate(work_dir, 2, draw_from="view")
    an.judge(UNIT, "attack", FRONT, 2, "rejected", work_dir=work_dir)
    with FakeComfy(png=figure_png()) as fake:
        result = an.generate_batch(UNIT, "attack", FRONT, "rejected", skeleton=False,
                                   client=comfy.ComfyClient(fake.url), workflow_dir=WORKFLOWS,
                                   work_dir=work_dir)
    assert result["cells"] == {"done": 0, "failed": 0, "blocked": 1, "of": 1}
    assert "not approved" in result["frames"][0]["reason"] and fake.prompts == []


def test_inpaint_on_a_frame_goes_back_to_review(work_dir):
    _base(work_dir)
    _generate(work_dir, "wind-up")
    _approve(work_dir, "wind-up")
    img = Image.open(io.BytesIO(figure_png())).convert("RGBA")
    ImageDraw.Draw(img).rectangle((10, 10, 40, 40), fill=(0, 0, 0, 0))
    canvas = io.BytesIO()
    img.save(canvas, format="PNG")
    mask = an.stash_canvas(UNIT, "attack", FRONT, "f03:00", canvas.getvalue(), work_dir=work_dir)
    assert mask.exists() and mask.parent.name == "masks"
    with FakeComfy(png=figure_png(colour=(250, 0, 0))) as fake:
        row = an.inpaint(UNIT, "attack", FRONT, "f03:00", mask, strength=0.8, seed=3,
                         registry_path=REGISTRY, client=comfy.ComfyClient(fake.url),
                         workflow_dir=WORKFLOWS, work_dir=work_dir)
    assert len(fake.prompts) == 1 and row["edited"]["how"] == "inpaint"
    assert "a knight" in row["edited"]["prompt"], "blank prompt: the unit doing the pose"
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert an.frame_status(record["frames"][2]) == "review"


# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------
def _partner_cell(work_dir: Path, direction: str = "s0", *, approve: bool = True) -> list[bytes]:
    """The partner's frames, written by hand: one attempt per frame."""
    job, record = an.open_job(UNIT, "attack", direction, work_dir)
    pictures = []
    for position, frame in enumerate(record["frames"]):
        store = an.FrameAttempts(UNIT, "attack", direction, frame["id"], work_dir,
                                 record=record, job=job)
        png = figure_png(shift=position % 3)
        row = store.append({"source": "animate", "seconds": 1.0, "prompt": "p"})
        (store.dir / row["file"]).write_bytes(png)
        frame["current"] = row["index"]
        if approve:
            frame["verdict"] = {"state": "approved", "index": row["index"], "sha256": "x"}
        pictures.append(png)
    an.save(job, record)
    return pictures


def test_mirror_flips_the_partner_into_review(work_dir):
    _base(work_dir)
    pictures = _partner_cell(work_dir, "s0")
    result = an.mirror(UNIT, "attack", "s180", work_dir=work_dir)
    assert result["from"] == "s0" and result["frames"] == 12
    assert "swaps handedness" in result["warning"]
    job, record = an.open_job(UNIT, "attack", "s180", work_dir)
    assert all(an.frame_status(f) == "review" for f in record["frames"]), "never approved"
    frame = record["frames"][2]
    row = frame["candidates"][-1]
    assert row["source"] == "mirror" and row["mirror"]["direction"] == "s0"
    with Image.open(job.root / "attempts" / row["file"]) as got, \
            Image.open(io.BytesIO(pictures[2])) as src:
        assert got.convert("RGB").tobytes() == ImageOps.mirror(src.convert("RGB")).tobytes()
    assert record["frames"][2]["key"] and record["frames"][2]["phase"] == "wind-up"
    assert record["mirror"]["from"] == "s0"


def test_mirror_is_pairs_only_and_needs_pictures(work_dir):
    _base(work_dir)
    with pytest.raises(ValueError, match="no mirror partner"):
        an.mirror(UNIT, "attack", "s90", work_dir=work_dir)
    with pytest.raises(ValueError, match="no mirror partner"):
        an.mirror(UNIT, "attack", FRONT, work_dir=work_dir)
    with pytest.raises(ValueError, match="nothing to flip"):
        an.mirror(UNIT, "attack", "s135", work_dir=work_dir)
    an.open_job(UNIT, "attack", "s45", work_dir)
    with pytest.raises(ValueError, match="nothing to flip"):
        an.mirror(UNIT, "attack", "s135", work_dir=work_dir)


def test_mirror_follows_the_partners_row_and_keeps_old_attempts(work_dir):
    _base(work_dir)
    _partner_cell(work_dir, "s225", approve=False)
    job, record = an.open_job(UNIT, "attack", "s315", work_dir)
    record["frames"][0]["candidates"] = [{"index": 0, "file": "f01_00.png"}]
    record["frames"] = record["frames"][:3]
    an.save(job, record)
    an.mirror(UNIT, "attack", "s315", work_dir=work_dir)
    _job, record = an.open_job(UNIT, "attack", "s315", work_dir)
    assert len(record["frames"]) == 12 and record["removed"][0]["id"] == "f01"
    assert record["frames"][0]["id"] not in ("f01", "f02", "f03")


# ---------------------------------------------------------------------------
# Output: the job layout the export and the Board read
# ---------------------------------------------------------------------------
def _all_approved(work_dir: Path, action: str = "attack") -> None:
    with FakeComfy(png=figure_png()) as fake:
        client = comfy.ComfyClient(fake.url)
        an.generate_batch(UNIT, action, FRONT, "keys", skeleton=False, draw_from="view",
                          client=client, workflow_dir=WORKFLOWS, work_dir=work_dir)
        an.approve_reviewed(UNIT, action, FRONT, work_dir=work_dir)
        an.generate_batch(UNIT, action, FRONT, "all", skeleton=False, client=client,
                          workflow_dir=WORKFLOWS, work_dir=work_dir)


def test_output_writes_what_export_reads(work_dir, palette, monkeypatch):
    monkeypatch.setattr(config, "PALETTE", palette)
    _base(work_dir)
    _all_approved(work_dir)
    with pytest.raises(ValueError, match="not every frame is approved"):
        an.output(UNIT, "attack", FRONT, work_dir=work_dir, palette=palette)
    ids = an.approve_reviewed(UNIT, "attack", FRONT, work_dir=work_dir, palette=palette)
    assert len(ids) == 7
    # The approval that completed the row wrote the finished frames.
    target = curation.canonical_job(UNIT, "attack", FRONT)
    assert target == "knight_strike_s270"
    job = Job.open(target, work_dir)
    manifest = job.read_manifest()
    assert manifest["slug"] == UNIT and manifest["action"] == "strike"
    assert manifest["motion_source"]["direction"] == FRONT and "animate" in manifest
    animations, _warnings = export_stage.find_animations(UNIT, work_dir)
    cell = next(a for a in animations if a.job == target)
    assert len(cell.frames) == 12 and cell.fps == 12 and cell.loop is False
    assert list(cell.phases) == ["start", "start", "wind-up", "wind-up", "wind-up", "wind-up",
                                 "action", "action", "action", "wind-down", "wind-down", "end"]
    assert cell.anchor == 6 and cell.timing_source == "animate:attack"
    for path in cell.frames:
        with Image.open(path) as img:
            assert img.size == tuple(config.CANVAS) and img.mode == "RGBA"
    picks = json.loads(job.picks_path.read_text())
    assert picks["timing"]["keys"] == [0, 2, 6, 9, 11]
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    state = an.output_state(record, work_dir)
    assert state["written"] and not state["stale"] and state["error"] is None
    # An edit after the output makes it stale.
    an.select(UNIT, "attack", FRONT, 1, 0, work_dir=work_dir)
    an.replace(UNIT, "attack", FRONT, "f01:00", figure_png(colour=(1, 2, 3)), work_dir=work_dir)
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert an.output_state(record, work_dir)["ready"] is False


def test_output_will_not_overwrite_another_routes_job(work_dir, palette):
    _base(work_dir)
    _all_approved(work_dir)
    target = Job.open(curation.canonical_job(UNIT, "attack", FRONT), work_dir)
    target.write_manifest({"schema": "anim.job/1", "slug": UNIT, "action": "strike",
                           "motion_source": {"kind": "rig", "direction": FRONT}})
    ids = an.approve_reviewed(UNIT, "attack", FRONT, work_dir=work_dir, palette=palette)
    assert ids
    _job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    assert "another route" in record["output_error"], "kept for the tab, not raised"
    assert "animate" not in target.read_manifest()
    result = an.output(UNIT, "attack", FRONT, replace_job=True, work_dir=work_dir,
                       palette=palette)
    assert result["frames"] == 12 and "animate" in target.read_manifest()
    aside = list((work_dir / "_replaced").iterdir())
    assert len(aside) == 1 and (aside[0] / "job.json").exists()


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------
def _ns(*argv):
    return cli.build_parser().parse_args(["animate", "--unit", UNIT, "--action", "attack",
                                          "--dir", FRONT, *argv])


def test_only_a_gpu_line_waits_for_the_base_gate(work_dir):
    assert an.spends_gpu(_ns("--frame", "3"))
    assert an.spends_gpu(_ns("--batch", "all"))
    assert an.spends_gpu(_ns("--inpaint", "f03:00", "--mask", "x.png"))
    for argv in (["--frame", "3", "--approve"], ["--hide", "f03:00"], ["--list"], ["--mirror"],
                 ["--skeletons"], ["--output"], ["--add-frame", "2"], ["--approve-reviewed"],
                 ["--frame", "3", "--key", "action"]):
        assert not an.spends_gpu(_ns(*argv)), argv
    reason = curation.gate_for_namespace("animate", _ns("--frame", "3"), work_dir)
    assert reason and "no base" in reason
    assert curation.gate_for_namespace("animate", _ns("--list"), work_dir) is None
    _base(work_dir)
    assert curation.gate_for_namespace("animate", _ns("--frame", "3"), work_dir) is None


def test_the_cli_lists_approves_and_edits_the_row(work_dir, monkeypatch, capsys):
    monkeypatch.setattr(config, "WORK_DIR", work_dir)
    _base(work_dir)
    _generate(work_dir, "wind-up")
    base = ["animate", "--unit", UNIT, "--action", "attack", "--dir", "front"]
    assert cli.main([*base, "--frame", "wind-up", "--approve"]) == 0
    assert "f03 approved" in capsys.readouterr().out
    assert cli.main([*base, "--add-frame", "3"]) == 0
    assert cli.main([*base, "--frame", "4", "--key", "action"]) == 0
    assert cli.main([*base, "--list"]) == 0
    out = capsys.readouterr().out
    assert "1 / 13 approved" in out and "approved" in out
    with pytest.raises(SystemExit, match="needs --frame"):
        cli.main([*base, "--approve"])


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------
@pytest.fixture
def api(work_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.1")
    manager = runs_api.RunManager(work_dir, cpu_slots=1,
                                  python=[sys.executable, "-c", FAKE_STAGE])
    app = create_app(work_dir, web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", runs=manager)
    with TestClient(app) as client:
        yield client


def test_get_animate_before_and_after(api, work_dir):
    assert api.get("/animate/knight/dance/s270").status_code == 404
    assert api.get("/animate/knight/attack/t90").status_code == 404
    empty = api.get("/animate/knight/attack/s270").json()
    assert empty["exists"] is False and "no base" in empty["locked"]
    assert len(empty["frames"]) == 12 and [f["position"] for f in empty["frames"] if f["key"]] == [
        1, 3, 7, 10, 12]
    assert empty["frames"][2]["label"] == "Wind-up" and empty["frames"][2]["draw_from"] == "view"
    assert empty["frames"][3]["draw_from"] == "keys"
    assert [f["position"] for f in empty["frames"] if f["view_key"]] == [1, 12]
    assert empty["frames"][0]["draw_from"] == empty["frames"][11]["draw_from"] == "copy"
    assert empty["frames"][3]["before"]["label"] == "Wind-up" and empty["frames"][3]["after"]["label"] == "Action"
    assert empty["counts"]["empty"] == 12 and empty["next_step"]["step"] == "keys"
    assert [c["direction"] for c in empty["compass"]] == an.side_directions()
    assert all(c["blocked"] for c in empty["compass"])
    assert {a["action"]: a["frames"] for a in empty["actions"]} == {
        name: row["budget"] for name, row in config.ACTION_PHASES.items()}
    _base(work_dir)
    _generate(work_dir, "wind-up", count=2)
    row = api.get("/animate/knight/attack/front").json()
    assert row["direction"] == "s270" and row["exists"] is True and row["locked"] is None
    compass = {c["direction"]: c for c in row["compass"]}
    assert compass["s270"]["url"] and compass["s270"]["blocked"] is None
    assert "no left view" in compass["s180"]["blocked"]
    wind = row["frames"][2]
    assert wind["status"] == "review" and wind["current"] == 1
    assert [c["url"] for c in wind["candidates"]] == [
        "/jobs/knight_animate_attack_s270/frames/animate/0.png",
        "/jobs/knight_animate_attack_s270/frames/animate/1.png"]
    assert api.get(wind["candidates"][1]["url"]).status_code == 200
    assert wind["game_url"].startswith("/animate/knight/attack/s270/game/f03.png")
    assert row["mirror"]["partner"] is None
    assert api.get("/animate/knight/attack/s180").json()["mirror"]["partner"] == "s0"
    _generate(work_dir, "start")
    start = api.get("/animate/knight/attack/s270").json()["frames"][0]
    assert start["status"] == "review" and start["stale"] is False
    assert start["draw_from"] == "view", "it has its copy"
    assert start["candidates"][0]["rendered"] and start["game_url"]
    assert api.get(start["candidates"][0]["url"]).status_code == 200


def test_game_route_renders_the_current_attempt(api, work_dir, palette, monkeypatch):
    monkeypatch.setattr(config, "PALETTE", palette)
    _base(work_dir)
    _generate(work_dir, "wind-up")
    got = api.get("/animate/knight/attack/s270/game/f03.png")
    assert got.status_code == 200
    assert Image.open(io.BytesIO(got.content)).size == tuple(config.CANVAS)
    assert api.get("/animate/knight/attack/s270/game/f04.png").status_code == 400


def test_generate_maps_onto_the_stages_flags(api, work_dir):
    response = api.post("/animate/knight/attack/s270/generate", json={"frame": "f03"})
    assert response.status_code == 423 and "no base" in response.json()["error"]
    _base(work_dir)
    argv = wait_for(api, api.post("/animate/knight/attack/s270/generate", json={
        "frame": "f03", "prompt": " coiled ", "draw_from": "view", "skeleton_frame": 9,
        "count": 2, "seed": 4, "model": "q.safetensors"}).json()["id"])["argv"]
    assert argv == ["--unit", "knight", "--action", "attack", "--dir", "s270", "--frame", "f03",
                    "--prompt", "coiled", "--draw-from", "view", "--skeleton-frame", "9",
                    "--count", "2", "--seed", "4", "--model", "q.safetensors"]
    argv = wait_for(api, api.post("/animate/knight/strike/s270/generate", json={
        "batch": "all", "skeleton": False}).json()["id"])["argv"]
    assert argv == ["--unit", "knight", "--action", "attack", "--dir", "s270", "--no-skeleton",
                    "--batch", "all"]
    refused = api.post("/animate/knight/attack/s270/generate", json={"frame": "4"})
    assert refused.status_code == 400 and "not approved yet" in refused.json()["detail"]
    assert api.post("/animate/knight/attack/s270/generate", json={}).status_code == 400
    assert api.post("/animate/knight/attack/s270/generate",
                    json={"batch": "some"}).status_code == 400
    assert api.post("/animate/knight/attack/s270/generate",
                    json={"frame": "f03", "draw_from": "x"}).status_code == 400
    copy = api.post("/animate/knight/attack/s270/generate", json={"frame": "2", "draw_from": "copy"})
    assert copy.status_code == 400 and "only a keyframe" in copy.json()["detail"]
    assert api.post("/animate/knight/attack/s180/generate",
                    json={"frame": "f03"}).status_code == 400, "no accepted left view"


def test_row_verdict_and_attempt_endpoints(api, work_dir):
    _base(work_dir)
    _generate(work_dir, "wind-up", count=2)
    base = "/animate/knight/attack/s270"
    row = api.post(f"{base}/frames", json={"after": 3}).json()
    assert row["added"] == "f13" and len(row["frames"]) == 13 and row["frames"][3]["id"] == "f13"
    row = api.put(f"{base}/frames/f13", json={"key": True, "phase": "action"}).json()
    assert row["frames"][3]["key"] and row["frames"][3]["label"] == "Action"
    assert api.put(f"{base}/frames/f13", json={"key": True, "phase": "x"}).status_code == 400
    row = api.delete(f"{base}/frames/f13").json()
    assert len(row["frames"]) == 12
    assert api.post(f"{base}/frames/f03/approve", json={"index": 7}).status_code == 400
    row = api.post(f"{base}/frames/f03/approve").json()
    assert row["frames"][2]["status"] == "approved" and row["counts"]["approved"] == 1
    row = api.put(f"{base}/frames/f03", json={"current": 0}).json()
    assert row["frames"][2]["status"] == "review"
    row = api.post(f"{base}/frames/f03/reject", json={"index": 0}).json()
    assert row["frames"][2]["status"] == "rejected"
    assert api.post(f"{base}/frames/f03/maybe").status_code == 404
    row = api.put(f"{base}/attempts/f03/1", json={"hidden": True}).json()
    assert row["frames"][2]["hidden_count"] == 1
    row = api.delete(f"{base}/attempts/f03/1").json()
    assert row["frames"][2]["candidates"][1]["file_deleted"]
    saved = api.put(f"{base}/attempts/f03/0/image", content=figure_png(colour=(9, 9, 200)),
                    headers={"Content-Type": "image/png"})
    assert saved.status_code == 200 and saved.json()["edited"]["edited"]["how"] == "hand"
    assert saved.json()["frames"][2]["status"] == "review", "the edit clears the verdict"
    assert api.put(f"{base}/attempts/f03/0/image", content=b"nope").status_code == 415
    row = api.post(f"{base}/approve-reviewed").json()
    assert row["approved"] == ["f03"]
    run = api.post(f"{base}/attempts/f03/0/inpaint", content=figure_png(),
                   headers={"Content-Type": "image/png"})
    assert run.status_code == 400 and "nothing is erased" in run.json()["detail"]


def test_mirror_and_output_endpoints(api, work_dir, palette, monkeypatch):
    monkeypatch.setattr(config, "PALETTE", palette)
    _base(work_dir)
    assert api.post("/animate/knight/attack/s90/mirror").status_code == 400
    _partner_cell(work_dir, "s0")
    row = api.post("/animate/knight/attack/s180/mirror").json()
    assert row["mirrored"] == 12 and "swaps handedness" in row["warning"]
    assert row["counts"]["review"] == 12
    assert api.post("/animate/knight/attack/s180/output").status_code == 400
    _all_approved(work_dir)
    job, record = an.open_job(UNIT, "attack", FRONT, work_dir)
    for frame in record["frames"]:
        if an.frame_status(frame) == "review":
            an.judge(UNIT, "attack", FRONT, frame["id"], "approved", work_dir=work_dir,
                     write_output=False)
    row = api.post("/animate/knight/attack/s270/output", json={}).json()
    assert row["written"]["frames"] == 12 and row["output"]["written"] is True


@needs_ual
def test_the_skeleton_route_renders_and_caches(api, work_dir):
    _base(work_dir)
    response = api.get("/animate/knight/attack/s270/skeleton/3.png")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == CELL
    assert api.get("/animate/knight/attack/s270/skeleton/99999.png").status_code == 400


def test_camera_words_come_out_of_the_description():
    """The archer's description was written for the front view; a frame at
    another direction must not be told the character faces forward."""
    assert config.strip_camera_words(
        "a hooded green archer. a fantasy ranger. facing forward") == \
        "a hooded green archer, a fantasy ranger"
    assert config.strip_camera_words("a knight, front view, seen from the side") == "a knight"
    text = config.animate_prompt("x", ["view"], "a green hood. facing the viewer")
    assert "facing" not in text and "the same a green hood" in text


def test_avoid_is_said_in_the_positive_prompt():
    """cfg 1 ignores a negative prompt, so the avoid phrase rides in the text."""
    text = config.animate_prompt("x", ["view"], "a hood", "no armour, no visible face.")
    assert text.endswith("Do not add: no armour, no visible face.")
    assert "Do not add" not in config.animate_prompt("x", ["view"], "a hood", "  ")


def test_a_unit_can_own_its_phase_prompts():
    """The archer has no weapon in its art; its attack is a spell, not a sword."""
    frames = an.default_layout("attack")
    archer = an.default_prompt("attack", frames, 2, loop=False, unit="hooded_archer")
    assert "spell" in archer and "sword" not in archer
    assert "sword" in an.default_prompt("attack", frames, 2, loop=False, unit="knight")
    assert config.animate_phase_prompt("walk", "action", "hooded_archer") == \
        config.animate_phase_prompt("walk", "action")


# ---------------------------------------------------------------------------
# The grid and the sweep (the pipeline tabs feature)
# ---------------------------------------------------------------------------
def test_cell_summary_reads_the_keyframe_gate():
    assert an.cell_summary(_fake_frames(["empty"] * 4, [0, 3]))["state"] == "none"
    keys = an.cell_summary(_fake_frames(["approved", "empty", "empty", "review"], [0, 3]))
    assert keys["state"] == "keys" and keys["keys_approved"] == 1 and keys["keys"] == 2
    ok = an.cell_summary(_fake_frames(["approved", "review", "empty", "approved"], [0, 3]))
    assert ok["state"] == "keys_ok" and ok["approved"] == 2 and ok["pictures"] == 3
    assert an.cell_summary(_fake_frames(["approved"] * 4, [0, 3]))["state"] == "done"


def _sweep(work_dir: Path, cells, which: str, **kw):
    with FakeComfy(png=figure_png()) as fake:
        result = an.sweep(UNIT, cells, which, skeleton=False, client=comfy.ComfyClient(fake.url),
                          workflow_dir=WORKFLOWS, work_dir=work_dir, **kw)
    return result, fake


def test_a_sweep_draws_keys_across_cells_and_keeps_the_gate(work_dir, capsys):
    _base(work_dir)
    cells = [("attack", FRONT), ("walk", "front"), ("attack", "s180")]
    result, fake = _sweep(work_dir, cells, "keys")
    assert result["tally"] == {"done": 2, "failed": 0, "blocked": 1, "of": 3}
    status = {c["cell"]: c for c in result["cells"]}
    assert status["attack:s180"]["status"] == "blocked"
    assert "no left view" in status["attack:s180"]["reason"]
    walk_keys = sum(1 for f in an.default_layout("walk", UNIT) if f["key"])
    # Keys only, both front cells - less the view keys, which are copies:
    # attack's start and end, walk's start (a loop's end is drawn).
    assert len(fake.prompts) == (5 - 2) + (walk_keys - 1), "keys only, both front cells"
    out = capsys.readouterr().out
    assert "[3/3] attack:s180 blocked" in out and "[1/5]" not in out, "cells, not frames"
    # The in-betweens wait for the keys, cell by cell.
    early, fake = _sweep(work_dir, [("attack", FRONT)], "inbetweens")
    assert early["tally"]["blocked"] == 1 and fake.prompts == []
    an.approve_reviewed(UNIT, "attack", FRONT, work_dir=work_dir)
    fill, fake = _sweep(work_dir, [("attack", FRONT), ("walk", FRONT)], "inbetweens")
    assert [c["status"] for c in fill["cells"]] == ["done", "blocked"]
    assert len(fake.prompts) == 7


def test_a_sweep_is_refused_before_the_gpu(work_dir):
    _base(work_dir)
    with pytest.raises(ValueError, match="at most 2"):
        _sweep(work_dir, [("attack", FRONT), ("walk", FRONT), ("run", FRONT)], "keys",
               max_cells=2)
    with pytest.raises(ValueError, match="keys or inbetweens"):
        _sweep(work_dir, [("attack", FRONT)], "rejected")
    with pytest.raises(ValueError, match="ACTION:DIRECTION"):
        an.parse_cell("attack")
    assert an.parse_cell("strike:front") == ("attack", "s270")


def test_the_grid_and_the_units(api, work_dir):
    assert api.get("/animate/Not A Slug").status_code == 404
    empty = api.get("/animate/knight").json()
    assert empty["actions"] == list(config.STUDIO_ACTIONS)
    assert [d["direction"] for d in empty["directions"]] == an.side_directions()
    assert "no base" in empty["locked"] and empty["max_cells"] == config.ANIMATE_SWEEP_MAX_CELLS
    assert empty["cells"]["attack"]["s270"]["state"] == "none"
    assert empty["cells"]["attack"]["s270"]["exists"] is False
    assert empty["cells"]["attack"]["s270"]["keys"] == 5, "the template's row"
    assert api.get("/units").json() == {"units": []}
    _base(work_dir)
    _generate(work_dir, "wind-up")
    _approve(work_dir, "wind-up")
    grid = api.get("/animate/knight").json()
    front = grid["cells"]["attack"]["s270"]
    assert front["state"] == "keys" and front["keys_approved"] == 1 and front["blocked"] is None
    assert "no left view" in grid["cells"]["attack"]["s180"]["blocked"]
    assert grid["cells"]["walk"]["s270"]["state"] == "none"
    assert "knight" in api.get("/units").json()["units"]


def test_the_sweep_route_maps_onto_one_run(api, work_dir):
    body = {"cells": [{"action": "attack", "direction": "s270"},
                      {"action": "strike", "direction": "left"}], "batch": "keys"}
    locked = api.post("/animate/knight/sweep", json=body)
    assert locked.status_code == 423 and "no base" in locked.json()["error"]
    _base(work_dir)
    argv = wait_for(api, api.post("/animate/knight/sweep", json={**body, "model": "q.safetensors"}
                                  ).json()["id"])["argv"]
    assert argv == ["--unit", "knight", "--sweep", "attack:s270", "attack:s180",
                    "--model", "q.safetensors", "--batch", "keys"]
    assert api.post("/animate/knight/sweep", json={**body, "batch": "all"}).status_code == 400
    assert api.post("/animate/knight/sweep", json={"cells": []}).status_code == 422
    bad = api.post("/animate/knight/sweep", json={"cells": [{"action": "dance", "direction": "s0"}]})
    assert bad.status_code == 400
