"""`anim matrix` - the action matrix feature, against FakeComfy.

What is pinned here:
  1. The plan is the cross product of the selected actions and directions in
     the board's DIRECTION-MAJOR order, named `<unit>_<VERB>[_<dir>]` - the
     pipeline verb, not the studio name - and `dirs=None` is s0 ALONE.
  2. The hard prerequisite gate: the base gate, a rig that ships no whole
     clip, a direction with no ACCEPTED view (the matrix refuses the
     front-view fallback) and a base with no part palettes to lock to. A
     blocked cell never reaches the GPU.
  3. Two actions x two directions run end to end through the fake GPU and the
     fake segmenter - rig -> clip -> pick --phases -> post --parts-from the
     BASE's part palettes - with one seed for all four.
  4. The prompt names the action phrase and the view, including the two
     phrases this feature added (`run`, `jump`).
  5. Resume: a second run spends no GPU, `--force` redoes a cell, a
     hand-edited picks.json is KEPT and not overwritten, a changed base is
     refused without --force.
  6. The progress contract: every finished or skipped cell prints `[n/m]` at
     column 0 with m = the RUNNABLE count, and the run manager's own
     `_percent` reads those lines as a monotone bar that ends at 100.
  7. One failed cell is recorded and the others are untouched.
  8. The #277 cap refuses a wide selection before any GPU call; `--dry-run`
     spends nothing.
  9. `GET /matrix/{unit}` answers before anything has run, and `POST /jobs`
     is the only way to launch one (423 while the base gate is pending).
 10. The registry: the stage is wired into the CLI, the GPU set and argv_for.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anim import cli, comfy, config, curation, outcome
from anim.api import runs as runs_api
from anim.api.app import create_app
from anim.job import Job
from anim.stages import base as base_stage
from anim.stages import clip as clip_stage
from anim.stages import matrix as matrix_stage
from fixtures import synthetic
from fixtures.fakecomfy import FakeComfy
from fixtures.fakeparts import FakeSegmenter
from test_bakeoff import COLOURS, _parts_file

REPO = Path(__file__).resolve().parent.parent
KNIGHT = REPO / "rigs" / "kaykit-adventurers" / "Knight.glb"
SIDE = ["s0", "s45", "s90", "s135", "s180", "s225", "s270", "s315"]
ACTIONS = ["attack", "idle", "walk", "run", "jump"]
PARTS = "closed great helm, kite shield, longsword, dark cloak"
# The two cells the module's run renders, in cell_order: direction-major.
RUN_KEYS = ["attack/s0", "walk/s0", "attack/s90", "walk/s90"]


# ---------------------------------------------------------------------------
# 1. The plan
# ---------------------------------------------------------------------------
def test_the_plan_is_the_cross_product_in_board_order():
    cells = matrix_stage.plan("knight")
    assert [c.key for c in cells] == [f"{a}/s0" for a in ACTIONS], \
        "dirs=None is the default direction ALONE, not all eight"
    assert [c.job for c in cells] == ["knight_strike", "knight_idle", "knight_walk",
                                      "knight_run", "knight_jump"], "the pipeline verb"
    assert [c.verb for c in cells] == ["strike", "idle", "walk", "run", "jump"]
    assert [c.action for c in cells] == ACTIONS
    assert {c.yaw for c in cells} == {0} and {c.elevation for c in cells} == {"side"}


def test_two_directions_are_ten_cells_direction_major():
    cells = matrix_stage.plan("knight", dirs=["s90", "s0"])
    assert len(cells) == 10
    assert [c.direction for c in cells] == ["s0"] * 5 + ["s90"] * 5, \
        "direction-major: a cancelled run leaves a complete direction"
    assert [c.key for c in cells[:5]] == [f"{a}/s0" for a in ACTIONS]
    by_key = {c.key: c.job for c in cells}
    assert by_key["attack/s90"] == "knight_strike_s90"
    assert by_key["walk/s0"] == "knight_walk"
    assert by_key["jump/s90"] == "knight_jump_s90"


def test_either_spelling_of_an_action_and_all_eight_directions():
    assert [c.key for c in matrix_stage.plan("knight", ["strike"])] == ["attack/s0"]
    assert [c.key for c in matrix_stage.plan("knight", ["attack"])] == ["attack/s0"]
    everything = matrix_stage.plan("knight", None, SIDE)
    assert len(everything) == 40 and everything[0].key == "attack/s0"
    assert [c.direction for c in everything[:5]] == ["s0"] * 5


def test_an_unknown_action_or_direction_is_refused_in_the_boards_words():
    with pytest.raises(curation.CurationError, match="not a studio action"):
        matrix_stage.plan("knight", ["sneak"])
    with pytest.raises(curation.CurationError, match="unknown direction"):
        matrix_stage.plan("knight", None, ["s17"])
    with pytest.raises(ValueError, match="unknown direction"):
        matrix_stage.plan("knight", None, ["t90"])       # the other elevation
    with pytest.raises(ValueError, match="elevation"):
        matrix_stage.plan("knight", None, None, "front")


def test_the_rig_table_and_the_missing_jump_clip():
    name, glb = matrix_stage.rig_for()
    assert name == "quaternius" and glb.name == "UAL1_Standard.glb"
    assert matrix_stage.rig_for("kaykit")[1].name == "Knight.glb"
    with pytest.raises(ValueError, match="unknown rig"):
        matrix_stage.rig_for("mixamo")
    # The UAL ships no whole jump: a per-cell None, never a run-level crash.
    assert matrix_stage.clip_for("quaternius", "jump") is None
    assert matrix_stage.clip_for("kaykit", "jump") == "Jump_Full_Short"
    assert {c.clip for c in matrix_stage.plan("knight", ["jump"])} == {None}


def test_the_seed_is_one_per_unit_and_follows_the_base():
    a = matrix_stage.derive_seed("knight", "kaykit", "a" * 64)
    assert a == matrix_stage.derive_seed("knight", "kaykit", "a" * 64)
    assert a != matrix_stage.derive_seed("knight", "kaykit", "b" * 64)
    assert a != matrix_stage.derive_seed("orc", "kaykit", "a" * 64)


def test_clip_times_caps_a_long_clip_and_leaves_a_short_one_to_rig():
    # 1 s at 12 fps is 13 frames, under the 25-frame cap: `rig` samples it.
    assert matrix_stage.clip_times(1.0, "attack") is None
    # 4 s is 49 frames: capped to MATRIX_CLIP_MAX_FRAMES instants over the
    # WHOLE clip, first and last included - never a truncated clip.
    times = matrix_stage.clip_times(4.0, "attack")
    assert len(times) == config.MATRIX_CLIP_MAX_FRAMES
    assert times[0] == 0.0 and times[-1] == 4.0
    assert times == sorted(times) and len(set(times)) == len(times)
    assert matrix_stage.clip_times(4.0, "attack", 14)[-1] == 4.0
    # A cap under the action's own budget would short every cell of that action.
    with pytest.raises(ValueError, match="pick --phases"):
        matrix_stage.clip_times(4.0, "attack", 12)       # attack budget is 12
    assert len(matrix_stage.clip_times(4.0, "idle", 7)) == 7   # idle budget is 6


def test_the_config_cap_can_always_feed_the_biggest_budget():
    biggest = max(row["budget"] for row in config.ACTION_PHASES.values())
    assert config.MATRIX_CLIP_MAX_FRAMES >= biggest + 1
    assert config.MATRIX_CELL == tuple(config.CLIP_CELL)
    assert config.MATRIX_CONTROL == "blend"
    assert config.MATRIX_DEPTH_GAIN == 0.35
    assert config.MATRIX_RIG == "quaternius"
    assert not hasattr(config, "TURNAROUND_CONTROL"), "the removed stage's constants are gone"
    assert config.MATRIX_MAX_CELLS == 8


# ---------------------------------------------------------------------------
# A unit on disk: the base, its views, the gate and the base's part palettes
# ---------------------------------------------------------------------------
def _unit(tmp_path: Path, palette: Path, *, views: dict[str, str],
          gates: dict[str, str] | None = None,
          base_parts: bool = True) -> Path:
    """A work/ with the knight's accepted base, the named views in the named
    states, the gate, and (unless `base_parts` is False) the base's measured
    parts.json - everything the matrix's prerequisites look at."""
    work_dir = tmp_path / "work"
    work_dir.mkdir(exist_ok=True)
    base_job = base_stage.prepare("knight", cell=(512, 768), work_dir=work_dir)
    base_job.render_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 768), (250, 250, 250)).save(base_job.render_dir / "concept_pick.png")
    stored = {}
    for view in views:
        Image.new("RGB", (512, 768), (240, 240, 250)).save(
            base_job.render_dir / f"view_{view}.png")
        stored[view] = {"view": view, "yaw": config.CONCEPT_VIEWS[view]["yaw"],
                        "file": f"view_{view}.png", "candidate": 1, "seed": 5,
                        "provisional": False}
    base_job.update_manifest(
        render={"seed": 1, "passes": ["concept_pass"], "final": "concept_pick.png"},
        concept={"job": "knight_concept", "candidate": 1, "parts_phrase": PARTS,
                 "avoid_phrase": "bare head, uncovered face", "views": stored})
    for gate, state in (gates or {"base": "accepted"}).items():
        curation.set_gate("knight", gate, state, work_dir)
    curation.set_views("knight", dict(views), work_dir)
    if base_parts:
        path = matrix_stage.base_parts_path("knight", work_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        _parts_file(path, palette)
    return work_dir


@pytest.fixture(scope="module")
def palette(tmp_path_factory) -> Path:
    from conftest import PALETTE_CANDIDATE

    if PALETTE_CANDIDATE.exists():
        return PALETTE_CANDIDATE
    path = tmp_path_factory.mktemp("palette") / "fallback.gpl"
    ramp = [(8 * i, 6 * i, 10 * i) for i in range(16)]
    path.write_text("GIMP Palette\nName: Fallback\nColumns: 8\n" + "".join(
        f"{r:3d} {g:3d} {b:3d}\t#{r:02x}{g:02x}{b:02x}\n" for r, g, b in ramp))
    return path


# ---------------------------------------------------------------------------
# 2. The prerequisites - the hard gate
# ---------------------------------------------------------------------------
def _verdicts(work_dir: Path, *, actions=None, dirs=SIDE, rig="kaykit"):
    cells = matrix_stage.plan("knight", actions, dirs, rig_name=rig)
    return cells, matrix_stage.prerequisites("knight", cells, work_dir, rig_name=rig)


def test_a_pending_base_gate_blocks_every_cell(tmp_path, palette):
    work_dir = _unit(tmp_path, palette, views={"back": "accepted"},
                     gates={"base": "pending"})
    _cells, verdicts = _verdicts(work_dir)
    assert all(not v["ok"] for v in verdicts.values())
    assert all("base gate is pending" in v["reason"] for v in verdicts.values())
    assert "--accept-gate" in verdicts["attack/s0"]["reason"]


def test_an_old_records_turnaround_gate_is_ignored(tmp_path, palette):
    """A curation record written before the turnaround stage was removed may
    still say `turnaround: pending`. Nothing reads it: an accepted base and
    an accepted view are enough for a cell to run."""
    work_dir = _unit(tmp_path, palette, views={"back": "accepted"})
    curation.set_gate("knight", "base", "accepted", work_dir)
    path = curation.curation_path("knight", work_dir)
    record = json.loads(path.read_text())
    record["gates"]["turnaround"] = "pending"
    path.write_text(json.dumps(record))
    _cells, verdicts = _verdicts(work_dir, actions=["idle"], dirs=["s90"])
    assert verdicts["idle/s90"]["ok"] is True, verdicts["idle/s90"]["reason"]


def test_only_an_accepted_view_makes_a_direction_runnable(tmp_path, palette):
    """With the back accepted and nothing else: s45 / s90 / s135 take the back,
    s270 is the base itself, and every other facing is BLOCKED naming what it
    wanted - the matrix does not render on the front-view fallback."""
    work_dir = _unit(tmp_path, palette, views={"back": "accepted", "right": "pending",
                                               "left": "rejected"})
    _cells, verdicts = _verdicts(work_dir, actions=["idle"])
    ok = {d for d in SIDE if verdicts[f"idle/{d}"]["ok"]}
    assert ok == {"s45", "s90", "s135", "s270"}
    assert verdicts["idle/s90"]["view"] == "back"
    assert verdicts["idle/s45"]["view"] == "back", "the oblique falls to the back"
    assert verdicts["idle/s270"]["view"] == "front", "s270's row is ('front',)"
    # A PENDING view is not enough - `reference_for` alone would accept it.
    assert "right (pending)" in verdicts["idle/s0"]["reason"]
    assert "front-view fallback" in verdicts["idle/s0"]["reason"]
    assert "anim concept" in verdicts["idle/s0"]["reason"]
    # A REJECTED one says so, and a view the base never had says that instead.
    assert "left (rejected)" in verdicts["idle/s180"]["reason"]
    assert "front_left (not on the base)" in verdicts["idle/s225"]["reason"]
    # The bare walk, handed every view, would take s0 and s180 anyway - which
    # is why the stage hands it `accepted_views` only.
    views = matrix_stage.base_views("knight", work_dir)
    front = (Path("/nowhere/front.png"), "base render")
    assert matrix_stage.reference_for("s0", front, views)["view"] == "right"
    assert matrix_stage.reference_for("s180", front, views)["fallback"] is True
    assert matrix_stage.reference_for(
        "s0", front, matrix_stage.accepted_views(views))["fallback"] is True


def test_accepting_a_view_un_greys_its_cells_with_no_re_run(tmp_path, palette):
    work_dir = _unit(tmp_path, palette, views={"right": "pending"})
    assert not matrix_stage.prerequisites(
        "knight", matrix_stage.plan("knight", ["idle"], ["s0"], rig_name="kaykit"),
        work_dir, rig_name="kaykit")["idle/s0"]["ok"]
    curation.set_views("knight", {"right": "accepted"}, work_dir)
    verdict = matrix_stage.prerequisites(
        "knight", matrix_stage.plan("knight", ["idle"], ["s0"], rig_name="kaykit"),
        work_dir, rig_name="kaykit")["idle/s0"]
    assert verdict["ok"] and verdict["view"] == "right"


def test_a_mirrored_view_satisfies_the_gate_for_its_own_facing(tmp_path, palette):
    """The mirrored view feature: `right` derived from an accepted `left` is an
    accepted view like any other, so s0 - the facing that wants `right` first -
    goes from blocked to runnable with no GPU spent on either the view or the
    cell. The flip is recorded on the base, and the gate reads the base."""
    from anim.stages import concept as concept_stage

    work_dir = _unit(tmp_path, palette, views={"left": "accepted"})
    blocked = _verdicts(work_dir, actions=["idle"], dirs=["s0"])[1]["idle/s0"]
    assert not blocked["ok"] and "right (not on the base)" in blocked["reason"]
    result = concept_stage.mirror_view("knight", "right", work_dir=work_dir,
                                       log=lambda _: None)
    assert result["views"]["right"]["mirrored"] is True
    assert "handedness" in result["warning"], "the knight carries a sword and a shield"
    verdict = _verdicts(work_dir, actions=["idle"], dirs=["s0"])[1]["idle/s0"]
    assert verdict["ok"] and verdict["view"] == "right"
    # And the reference walk, handed the same base, picks it as s0's reference.
    views = matrix_stage.accepted_views(matrix_stage.base_views("knight", work_dir))
    front = (Path("/nowhere/front.png"), "base render")
    reference = matrix_stage.reference_for("s0", front, views)
    assert reference["view"] == "right" and reference["fallback"] is False


def test_a_base_with_no_part_palettes_blocks_every_direction(tmp_path, palette):
    work_dir = _unit(tmp_path, palette, views={"back": "accepted"}, base_parts=False)
    _cells, verdicts = _verdicts(work_dir, actions=["idle"], dirs=["s90", "s270"])
    for key in ("idle/s90", "idle/s270"):
        reason = verdicts[key]["reason"]
        assert "no part palettes on the base" in reason
        assert "knight_base/post/parts.json" in reason
        assert "anim post knight_base --parts" in reason
    assert verdicts["idle/s90"]["view"] == "back", "the view was found; the parts were not"


def test_jump_is_blocked_on_the_ual_and_runnable_on_kaykit(tmp_path, palette):
    work_dir = _unit(tmp_path, palette, views={"right": "accepted"})
    _cells, ual = _verdicts(work_dir, actions=["jump"], dirs=["s0"], rig="quaternius")
    reason = ual["jump/s0"]["reason"]
    assert "quaternius rig ships no whole jump clip" in reason
    assert "Jump_Start / Jump_Loop / Jump_Land" in reason and "clip joiner" in reason
    _cells, kaykit = _verdicts(work_dir, actions=["jump"], dirs=["s0"], rig="kaykit")
    assert kaykit["jump/s0"]["ok"] is True


def test_a_blocked_cell_never_reaches_the_gpu(tmp_path, palette):
    """The whole point of the gate: not one prompt is posted for a selection
    that is entirely blocked, so a FakeComfy that fails every prompt is silent."""
    work_dir = _unit(tmp_path, palette, views={"back": "accepted"},
                     gates={"base": "pending"})
    logs: list[str] = []
    with FakeComfy(fail_prompt=True) as fake:
        state = matrix_stage.matrix("knight", actions=["idle"], dirs=["s0"],
                                    rig_name="kaykit", glb=KNIGHT,
                                    client=comfy.ComfyClient(fake.url),
                                    segmenter=FakeSegmenter(COLOURS),
                                    work_dir=work_dir, log=logs.append)
    assert not fake.prompts, "a blocked cell must not spend the card"
    assert state["cells"]["idle/s0"]["status"] == "blocked"
    assert "base gate" in state["cells"]["idle/s0"]["reason"]
    assert any(line.startswith("matrix: BLOCKED idle/s0 - ") for line in logs)
    assert state["summary"] == {**state["summary"], "blocked": 1, "done": 0, "of": 1}


# ---------------------------------------------------------------------------
# 3 + 4 + 6. The run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ran(tmp_path_factory, palette):
    """Two actions x two directions once for the module, on the KayKit Knight
    (the test rig: it ships a whole jump and its GLB is vendored) against a
    FakeComfy that answers a whole clip per job."""
    if not KNIGHT.exists():
        pytest.skip("rigs/kaykit-adventurers/Knight.glb is not vendored")
    tmp_path = tmp_path_factory.mktemp("matrix")
    work_dir = _unit(tmp_path, palette,
                     views={"right": "accepted", "back": "accepted"})
    # attack and walk are both 1.067 s on this rig, so both cells send Wan the
    # same number of frames and one canned batch answers every job.
    from anim import rigkit

    loaded = rigkit.load_rig(KNIGHT)
    frames = len(rigkit.sample_times(
        loaded.clip(config.STUDIO_ACTIONS["attack"]["clips"]["kaykit"]).duration,
        config.MATRIX_CLIP_FPS))
    frame = synthetic.write_strip(tmp_path / "frame.png", cells=1,
                                  cell=tuple(config.MATRIX_CELL)).read_bytes()
    batch = [frame] * clip_stage.padded_length(frames)
    logs: list[str] = []
    checkpoints: dict[str, list[str]] = {}

    def log(line: str) -> None:
        # Test 6: read the state file from inside the log, so the assertion is
        # about what is on disk WHILE the run is in flight, not after it.
        logs.append(line)
        if line.startswith("matrix: "):
            path = matrix_stage.state_path("knight", work_dir)
            cells = json.loads(path.read_text())["cells"] if path.exists() else {}
            checkpoints[line.split()[1]] = {k: v["status"] for k, v in cells.items()}

    common = dict(matte="chroma", rig_name="kaykit", glb=KNIGHT, work_dir=work_dir,
                  assets_dir=tmp_path / "no-assets",
                  workflow_dir=REPO / "gpu" / "workflows", log=log)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PALETTE", palette)
        with FakeComfy(batch=batch) as fake:
            state = matrix_stage.matrix(
                "knight", actions=["attack", "walk"], dirs=["s0", "s90"],
                client=comfy.ComfyClient(fake.url), segmenter=FakeSegmenter(COLOURS),
                **common)
        yield {"state": state, "logs": logs, "prompts": fake.prompts,
               "work_dir": work_dir, "common": common, "tmp_path": tmp_path,
               "palette": palette, "batch": batch, "checkpoints": checkpoints,
               "rig_frames": frames}


def test_every_cell_runs_the_whole_chain_with_one_seed(ran):
    state = ran["state"]
    assert state["schema"] == "anim.matrix/1" and state["plan"] == RUN_KEYS
    assert state["route"] == "V" and state["control"] == "blend"
    assert len(ran["prompts"]) == 4, "one Wan job per cell"
    for key in RUN_KEYS:
        record = state["cells"][key]
        assert record["status"] == "done", record.get("error")
        assert record["seed"] == state["seed"], "ONE seed for the whole unit"
        assert set(record["timings"]) == {"rig", "gpu", "pick", "post"}
        assert record["seconds"] > 0 and record["clip_frames"] == ran["rig_frames"]
        job = Job.open(record["job"], ran["work_dir"])
        manifest = job.read_manifest()
        assert manifest["action"] == record["verb"], "the manifest says the VERB"
        assert manifest["motion_source"]["direction"] == record["direction"]
        assert manifest["matrix"]["status"] == "done"
        assert manifest["matrix"]["key"] == key
        assert matrix_stage.is_done(job)
    assert [state["cells"][k]["job"] for k in RUN_KEYS] == [
        "knight_strike", "knight_walk", "knight_strike_s90", "knight_walk_s90"]
    assert state["summary"]["done"] == 4 and state["summary"]["of"] == 4
    assert state["summary"]["failed"] == 0 and state["summary"]["blocked"] == 0


def test_every_cell_is_phase_picked_and_posted_to_the_canvas(ran):
    for key in RUN_KEYS:
        record = ran["state"]["cells"][key]
        job = Job.open(record["job"], ran["work_dir"])
        picks = json.loads(job.picks_path.read_text())
        block = picks["phases"]
        budget = config.ACTION_PHASES[record["action"]]["budget"]
        assert block["plan"]["budget"] == budget == record["phases"]["budget"]
        assert block["plan"]["verb"] == record["verb"]
        assert set(block["tags"]) <= set(config.PHASE_NAMES)
        assert len(picks["picks"]) == budget == len(set(picks["picks"])), \
            "a capped sample must still give the picker distinct frames"
        assert record["frames"] == budget
        stem = f"knight_{record['verb']}"
        with Image.open(job.post_dir / f"{stem}_0.png") as first:
            assert first.size == tuple(config.CANVAS)
    # A loop action records its closure as INFORMATION; a one-shot has none.
    assert ran["state"]["cells"]["walk/s0"]["phases"]["loop"] is True
    assert ran["state"]["cells"]["walk/s0"]["phases"]["closure"] is not None
    assert ran["state"]["cells"]["attack/s0"]["phases"]["loop"] is False
    assert ran["state"]["cells"]["attack/s0"]["phases"]["closure"] is None
    assert ran["state"]["cells"]["attack/s0"]["phases"]["anticipation"] == 3


def test_every_cell_locks_to_the_bases_part_palettes(ran):
    expected = matrix_stage.base_parts_path("knight", ran["work_dir"])
    for key in RUN_KEYS:
        record = ran["state"]["cells"][key]
        assert record["parts_from"] == str(expected)
        report = json.loads(Job.open(record["job"], ran["work_dir"]).report_path.read_text())
        assert report["parts"]["palettes_from"] == str(expected)
        assert record["locked_to"] == str(expected)
    assert ran["state"]["cells"]["attack/s90"]["locked_to"].endswith(
        "knight_base/post/parts.json"), "the same palettes in every direction"


def test_each_cell_hands_wan_its_accepted_view_and_never_a_fallback(ran):
    state = ran["state"]
    used = {d: (row["view"], row["fallback"]) for d, row in state["references"].items()}
    assert used == {"s0": ("right", False), "s90": ("back", False)}
    for key in RUN_KEYS:
        record = state["cells"][key]
        assert record["reference"]["fallback"] is False
        job = Job.open(record["job"], ran["work_dir"])
        clip_record = json.loads(job.clip_path.read_text())
        assert clip_record["reference"]["file"] == record["reference"]["file"]
        assert (job.clip_dir / "reference.png").exists()
    assert state["references"]["s90"]["file"].endswith("knight_base/render/view_back.png")


def test_the_prompt_names_the_action_phrase_and_the_view(ran):
    texts = [prompt[comfy.find_node(prompt, "positive prompt")]["inputs"]["text"]
             for prompt in ran["prompts"]]
    by_key = dict(zip(RUN_KEYS, texts))
    assert by_key["attack/s0"] == config.clip_prompt("strike", "s0", PARTS)
    assert "swinging a sword" in by_key["attack/s0"]
    assert "side-view, facing right" in by_key["walk/s0"]
    assert "back view" in by_key["attack/s90"] and "facing right" not in by_key["attack/s90"]
    assert "no face visible" in by_key["walk/s90"]
    for key in RUN_KEYS:
        assert PARTS in by_key[key] and "exactly one figure" in by_key[key]
        assert ran["state"]["cells"][key]["prompt"] == by_key[key]


def test_the_two_phrases_this_feature_added():
    """`run` and `jump` had no CLIP_ACTION_PHRASES row, so the prompt said
    "performing a run". Every other action's prompt must be byte-identical."""
    assert config.CLIP_ACTION_PHRASES["run"] == "running at full stride"
    assert config.CLIP_ACTION_PHRASES["jump"] == "leaping, feet off the ground"
    assert "running at full stride" in config.clip_prompt("run", "s0")
    assert "leaping, feet off the ground" in config.clip_prompt("jump", "s90", PARTS)
    assert "performing a" not in config.clip_prompt("run", "s0")
    for verb in ("strike", "idle", "walk"):
        assert config.CLIP_ACTION_PHRASES[verb] in config.clip_prompt(verb, "s0")
    assert config.clip_prompt("idle", "s0") == (
        "a single side-view, facing right pixel art character sprite, full body in "
        "frame, standing at rest, breathing, the same character and outfit as the "
        "reference image in every frame, plain flat white background, no ground "
        "shadow, no floor"), "the sentence every existing job was rendered with"
    # Every studio action now has a phrase of its own, keyed by the VERB.
    assert all(row["verb"] in config.CLIP_ACTION_PHRASES
               for row in config.STUDIO_ACTIONS.values())


def test_the_progress_lines_are_what_the_run_manager_reads(ran):
    """THE PROGRESS CONTRACT. `runs.RunManager._percent` matches `^\\[n/m\\]` on
    the UNSTRIPPED line, so the bar means "cells finished", is monotone and
    ends at 100."""
    steps = [line for line in ran["logs"] if line.startswith("[")]
    assert len(steps) == 4
    parsed = [runs_api._STEP.match(line) for line in steps]
    assert all(m is not None for m in parsed)
    assert [int(m.group(1)) for m in parsed] == [1, 2, 3, 4]
    assert {int(m.group(2)) for m in parsed} == {4}, "m is the RUNNABLE count"
    # The announce line carries no brackets: an in-flight cell must not move it.
    announce = [line for line in ran["logs"] if line.startswith("matrix: ")]
    assert len(announce) == 4 and not any(line.startswith("[") for line in announce)
    assert all(runs_api.RunManager._percent(line) is None for line in announce)
    percents = [runs_api.RunManager._percent(line) for line in ran["logs"]]
    seen = [p for p in percents if p is not None]
    assert seen == sorted(seen) and seen[-1] == 100
    assert seen == [25, 50, 75, 100]


def test_the_state_file_is_checkpointed_after_every_cell(ran):
    """Read from inside the log callback: when cell n is announced, the file on
    disk holds exactly the n-1 cells before it. A Cancel mid-clip resumes here."""
    checkpoints = ran["checkpoints"]
    assert checkpoints["attack/s0"] == {}, "nothing is written before the first cell"
    for n, key in enumerate(RUN_KEYS):
        if n == 0:
            continue
        assert checkpoints[key] == {k: ("done" if i < n else "not run")
                                    for i, k in enumerate(RUN_KEYS)}, key
    on_disk = json.loads(matrix_stage.state_path("knight", ran["work_dir"]).read_text())
    assert sorted(on_disk["cells"]) == sorted(RUN_KEYS)
    assert on_disk["seed"] == ran["state"]["seed"]


# ---------------------------------------------------------------------------
# 5. Resume
# ---------------------------------------------------------------------------
def test_a_second_run_skips_every_cell_and_spends_nothing(ran):
    logs: list[str] = []
    common = dict(ran["common"], log=logs.append)
    with FakeComfy(fail_prompt=True) as fake:       # any GPU call would fail loudly
        state = matrix_stage.matrix("knight", actions=["attack", "walk"],
                                    dirs=["s0", "s90"],
                                    client=comfy.ComfyClient(fake.url),
                                    segmenter=FakeSegmenter(COLOURS), **common)
    assert not fake.prompts
    skipped = [line for line in logs if "skipped" in line]
    assert len(skipped) == 4 and all(line.startswith("[") for line in skipped)
    assert all(state["cells"][k]["status"] == "done" for k in RUN_KEYS)
    assert [runs_api.RunManager._percent(line) for line in skipped] == [25, 50, 75, 100]


def test_force_re_runs_a_done_cell(ran):
    logs: list[str] = []
    common = dict(ran["common"], log=logs.append)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PALETTE", ran["palette"])
        with FakeComfy(batch=ran["batch"]) as fake:
            state = matrix_stage.matrix("knight", actions=["attack"], dirs=["s0"],
                                        force=True, client=comfy.ComfyClient(fake.url),
                                        segmenter=FakeSegmenter(COLOURS), **common)
    assert len(fake.prompts) == 1 and "skipped" not in "".join(logs)
    assert state["cells"]["attack/s0"]["status"] == "done"
    assert state["plan"] == ["attack/s0"], "this run selected one cell"
    assert sorted(state["cells"]) == sorted(RUN_KEYS), \
        "a cell the plan no longer names keeps its record"


def test_a_hand_edited_picks_json_is_kept_not_overwritten(ran):
    """`pick` refuses to overwrite picks somebody edited, and the matrix must
    not force past that: the cell is recorded `kept` and the run goes on."""
    copy = ran["tmp_path"] / "work_kept"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(ran["work_dir"], copy)
    job = Job.open("knight_strike", copy)
    record = json.loads(job.picks_path.read_text())
    record["picks"] = list(record["auto"])[:-1] + [record["auto"][-1] - 1]
    job.picks_path.write_text(json.dumps(record))
    before = job.picks_path.read_bytes()
    job.update_manifest(matrix={"status": "not run"})     # so it is not skipped
    logs: list[str] = []
    common = dict(ran["common"], log=logs.append, work_dir=copy)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PALETTE", ran["palette"])
        with FakeComfy(batch=ran["batch"]) as fake:
            state = matrix_stage.matrix("knight", actions=["attack"], dirs=["s0"],
                                        client=comfy.ComfyClient(fake.url),
                                        segmenter=FakeSegmenter(COLOURS), **common)
    cell = state["cells"]["attack/s0"]
    assert cell["status"] == "kept" and "edited by hand" in cell["kept"]
    assert job.picks_path.read_bytes() == before, "the owner's picks survive"
    assert cell["picks"] == record["picks"]
    assert any("picks kept" in line for line in logs)
    # And a kept cell counts as done, so a resume does not spend the card again.
    assert matrix_stage.is_done(job) and state["summary"]["done"] == 1


def test_a_changed_base_is_refused_without_force(ran):
    path = matrix_stage.state_path("knight", ran["work_dir"])
    record = json.loads(path.read_text())
    keep = record["reference"]["sha256"]
    record["reference"]["sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    try:
        with FakeComfy(fail_prompt=True) as fake:
            with pytest.raises(ValueError, match="--force"):
                matrix_stage.matrix("knight", actions=["attack"], dirs=["s0"],
                                    client=comfy.ComfyClient(fake.url),
                                    segmenter=FakeSegmenter(COLOURS), **ran["common"])
        assert not fake.prompts
    finally:
        record["reference"]["sha256"] = keep
        path.write_text(json.dumps(record))


# ---------------------------------------------------------------------------
# 7. One failed cell
# ---------------------------------------------------------------------------
def test_a_failed_cell_is_recorded_and_the_others_are_untouched(ran):
    copy = ran["tmp_path"] / "work_failed"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(ran["work_dir"], copy)
    logs: list[str] = []
    common = dict(ran["common"], log=logs.append, work_dir=copy)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PALETTE", ran["palette"])
        with FakeComfy(fail_prompt=True) as fake:
            state = matrix_stage.matrix("knight", actions=["walk"], dirs=["s90"],
                                        force=True, client=comfy.ComfyClient(fake.url),
                                        segmenter=FakeSegmenter(COLOURS), **common)
    record = state["cells"]["walk/s90"]
    assert record["status"] == "failed" and "ComfyError" in record["error"]
    assert state["summary"]["failed"] == 1 and state["summary"]["done"] == 0
    # The run went on past the failed cell, as it must - and the tally is what
    # keeps that from reading as a clean exit in the studio (anim.outcome).
    tally = outcome.Outcome(done=state["summary"]["done"],
                            failed=state["summary"]["failed"],
                            blocked=state["summary"]["blocked"],
                            of=state["summary"]["of"])
    assert tally.exit_code == outcome.EXIT_NOTHING
    assert tally.line == "outcome: done=0 failed=1 blocked=0 of=1"
    assert state["cells"]["attack/s0"]["status"] == "done", "the others are untouched"
    assert any("FAILED" in line and line.startswith("[1/1]") for line in logs)
    assert Job.open("knight_walk_s90", copy).read_manifest()["matrix"]["status"] == "failed"


# ---------------------------------------------------------------------------
# 8. The #277 cap and the dry run
# ---------------------------------------------------------------------------
def test_the_cap_refuses_a_wide_selection_before_any_gpu_call(tmp_path, palette):
    work_dir = _unit(tmp_path, palette,
                     views={"right": "accepted", "back": "accepted"})
    common = dict(rig_name="kaykit", glb=KNIGHT, work_dir=work_dir, log=lambda _l: None)
    with FakeComfy(fail_prompt=True) as fake:
        client = comfy.ComfyClient(fake.url)
        with pytest.raises(ValueError, match="#277") as excinfo:
            matrix_stage.matrix("knight", dirs=["s0", "s45", "s90"], client=client,
                                segmenter=FakeSegmenter(COLOURS), **common)
        assert "15 cells asked for" in str(excinfo.value)
        assert "--max-cells" in str(excinfo.value) and "--actions / --dirs" in str(excinfo.value)
        assert not fake.prompts, "the hold is checked before the card is touched"
        # 0 removes the cap; --dry-run then spends nothing all the same.
        state = matrix_stage.matrix("knight", dirs=["s0", "s45", "s90"], max_cells=0,
                                    dry_run=True, client=client,
                                    segmenter=FakeSegmenter(COLOURS), **common)
        assert not fake.prompts
    assert state["max_cells"] == 0 and len(state["plan"]) == 15
    assert state["summary"]["done"] == 0
    # A narrower selection is under the default cap and would have started.
    assert config.MATRIX_MAX_CELLS == 8


def test_a_dry_run_prints_every_verdict_and_is_never_a_423(tmp_path, palette):
    work_dir = _unit(tmp_path, palette, views={"right": "accepted"})
    logs: list[str] = []
    with FakeComfy(fail_prompt=True) as fake:
        state = matrix_stage.matrix("knight", dirs=["s0", "s90"], dry_run=True,
                                    max_cells=0, rig_name="kaykit", glb=KNIGHT,
                                    client=comfy.ComfyClient(fake.url),
                                    work_dir=work_dir, log=logs.append)
    assert not fake.prompts
    assert state["summary"]["blocked"] == 5, "every s90 cell has no accepted view"
    assert len([line for line in logs if line.startswith("matrix: BLOCKED")]) == 5
    assert state["cells"]["attack/s0"]["status"] == "not run"
    # A dry run must never be refused by the curation gate: it spends nothing.
    args = cli.build_parser().parse_args(["matrix", "knight", "--dry-run"])
    curation.set_gate("knight", "base", "pending", work_dir)
    assert curation.gate_for_namespace("matrix", args, work_dir) is None
    plain = cli.build_parser().parse_args(["matrix", "knight"])
    assert "base gate" in curation.gate_for_namespace("matrix", plain, work_dir)


# ---------------------------------------------------------------------------
# 9. The API
# ---------------------------------------------------------------------------
def test_get_matrix_lists_every_planned_cell_with_its_verdict(ran, tmp_path):
    app = create_app(ran["work_dir"], web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", cpu_slots=1)
    with TestClient(app) as api:
        row = api.get("/matrix/knight").json()
        assert row["unit"] == "knight" and len(row["cells"]) == 40
        assert row["actions"] == ACTIONS and row["directions"] == SIDE
        assert row["max_cells"] == config.MATRIX_MAX_CELLS
        assert row["estimate_seconds"] == matrix_stage.CELL_SECONDS
        assert row["selectable"] == sum(1 for c in row["cells"].values()
                                        if c["runnable"]) > 0
        cell = row["cells"]["attack/s0"]
        assert cell["status"] == "done" and cell["runnable"] is True
        assert cell["job"] == "knight_strike" and cell["verb"] == "strike"
        assert cell["view"] == "right" and cell["reason"] is None
        assert cell["frames"] == config.ACTION_PHASES["attack"]["budget"]
        assert len(cell["strip"]) == cell["frames"] == len(cell["phases"])
        assert cell["strip"][0] == "/jobs/knight_strike/frames/post/0.png"
        for url in cell["strip"][:3]:
            response = api.get(url)
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"
        assert cell["loop"] is False and row["cells"]["walk/s0"]["loop"] is True
        assert cell["fps"] == config.MATRIX_CLIP_FPS
        # A cell that cannot run says why, in the same words the CLI printed.
        blocked = row["cells"]["idle/s180"]
        assert blocked["runnable"] is False and blocked["status"] == "blocked"
        assert "no accepted view" in blocked["reason"]
        assert blocked["strip"] == [] and blocked["frames"] == 0
        # s45 has no view of its own, and takes the accepted back: runnable.
        assert row["cells"]["idle/s45"]["runnable"] is True
        assert row["cells"]["idle/s45"]["view"] == "back"
        assert "turnaround" not in row, "the matrix listing names no removed stage"
        # The jump on the DEFAULT rig ships no clip: the honest reason, not grey.
        assert row["cells"]["jump/s0"]["clip"] is None
        assert "no whole jump clip" in row["cells"]["jump/s0"]["reason"]
        assert api.get("/matrix/orc").json()["cells"]["attack/s0"]["runnable"] is False
        assert api.get("/matrix/_runs").status_code == 404
        assert api.get("/matrix/not a slug").status_code == 404


def test_a_matrix_launches_through_post_jobs_and_waits_for_the_gate(ran, tmp_path,
                                                                    monkeypatch):
    """No POST /matrix/{unit}: `POST /jobs` already carries the 423, the 409
    and argparse's 400. The stage is never actually run here - the fake
    launcher stands in, so nothing touches the card."""
    import sys

    from test_api import FAKE_STAGE, wait_for

    monkeypatch.setenv("FAKE_DELAY", "0.1")
    work_dir = ran["tmp_path"] / "work_api"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(ran["work_dir"], work_dir)
    manager = runs_api.RunManager(work_dir, cpu_slots=1,
                                  python=[sys.executable, "-c", FAKE_STAGE])
    app = create_app(work_dir, web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", runs=manager)
    with TestClient(app) as api:
        curation.set_gate("knight", "base", "pending", work_dir)
        refused = api.post("/jobs", json={"stage": "matrix", "args": {"unit": "knight"}})
        assert refused.status_code == 423 and refused.json()["gate"] is True
        assert "base gate" in refused.json()["error"]
        assert api.get("/runs").json() == [], "nothing started"
        curation.set_gate("knight", "base", "accepted", work_dir)
        started = api.post("/jobs", json={"stage": "matrix",
                                          "args": {"unit": "knight",
                                                   "actions": "attack,walk",
                                                   "dirs": "s0,s90"}})
        assert started.status_code == 202
        run = started.json()
        assert run["gpu"] is True, "a matrix holds the one GPU slot"
        assert run["argv"] == ["--actions", "attack,walk", "--dirs", "s0,s90", "knight"]
        wait_for(api, run["id"])
        bad = api.post("/jobs", json={"stage": "matrix", "args": {"unit": "knight",
                                                                 "nonsense": 1}})
        assert bad.status_code == 400


# ---------------------------------------------------------------------------
# 10. The registry
# ---------------------------------------------------------------------------
def test_the_stage_is_wired_into_the_cli_and_the_gpu_rule():
    assert "matrix" in runs_api.GPU_STAGES and "matrix" in runs_api.RUNNABLE
    from anim.stages import STAGE_ORDER

    assert STAGE_ORDER.index("matrix") == STAGE_ORDER.index("bakeoff") + 1
    assert "turnaround" not in STAGE_ORDER and "turnaround" not in runs_api.GPU_STAGES
    assert matrix_stage.NAME == "matrix" and matrix_stage.run is matrix_stage.run_stage
    args = cli.build_parser().parse_args(["matrix", "knight", "--dirs", "s0"])
    assert args.unit == "knight" and args.dirs == "s0" and args.all_dirs is False
    assert args.max_cells is None and args.rig == config.MATRIX_RIG
    assert runs_api.argv_for("matrix", {"unit": "knight", "actions": "attack",
                                        "dirs": "s0,s90"}) == \
        ["--actions", "attack", "--dirs", "s0,s90", "knight"]
    assert runs_api.argv_for("matrix", {"unit": "knight", "all_dirs": True}) == \
        ["--all-dirs", "knight"]
    assert "matrix" in cli.EPILOG and "matrix" in curation.BASE_GATED
    assert curation.GATES == ("base",)
