"""`anim turnaround` - the turnaround feature, against FakeComfy.

What is pinned here:
  1. The plan is the eight cells of one elevation in compass order, named
     `<unit>_idle_<dir>`, a subset by `--dirs`, the top cells by `--elevation`.
  2. All eight run end to end through the fake GPU and the fake segmenter -
     rig (frame 0 held five times) -> clip (one 5-frame job per direction,
     the prompt naming the view) -> the middle frame picked -> post --parts
     -> that direction's parts.json - and the state file, the identity table
     and the page name every direction.
  3. A second run skips the directions that are done; a changed base is
     refused without --force; a page-only rebuild needs no service.
  4. The back cells hide every face point (the rule is in rigkit; here it
     shows up in the job's keypoints and the record).
  5. `GET /turnaround/{unit}` serves the same data and the stills resolve.
  6. The gate: the stage waits for the base gate and never for its own; after
     the run, `anim curate --accept-gate turnaround` flips it and a cell
     outside s0 may render.
  7. The multi-view reference feature: the prompt names the parts phrase and
     the view per direction, each direction hands Wan the base's nearest
     view (s0 the right profile, s90 the back ...), a missing view falls
     back to the front with a warning, the record, the page and the API say
     which reference every still used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anim import cli, comfy, config, curation
from anim.api.app import create_app
from anim.job import Job
from anim.stages import base as base_stage
from anim.stages import turnaround as turnaround_stage
from fixtures import synthetic
from fixtures.fakecomfy import FakeComfy
from fixtures.fakeparts import FakeSegmenter
from test_bakeoff import COLOURS, _parts_file

REPO = Path(__file__).resolve().parent.parent
KNIGHT = REPO / "rigs" / "kaykit-adventurers" / "Knight.glb"
SIDE = ["s0", "s45", "s90", "s135", "s180", "s225", "s270", "s315"]


# ---------------------------------------------------------------------------
# 1. The plan
# ---------------------------------------------------------------------------
def test_the_plan_is_eight_cells_in_compass_order():
    cells = turnaround_stage.plan("knight")
    assert [c.name for c in cells] == SIDE
    assert [c.job for c in cells] == [f"knight_idle_{d}" for d in SIDE]
    assert [c.yaw for c in cells] == list(config.DIRECTION_YAWS)
    assert {c.elevation for c in cells} == {"side"}
    assert [c.name for c in turnaround_stage.plan("knight", ["s270", "s90"])] == ["s90", "s270"]
    top = turnaround_stage.plan("knight", elevation="top")
    assert [c.name for c in top] == [f"t{y}" for y in config.DIRECTION_YAWS]
    assert top[0].job == "knight_idle_t0"
    with pytest.raises(ValueError, match="unknown direction"):
        turnaround_stage.plan("knight", ["t90"])          # the other elevation's cell
    with pytest.raises(ValueError, match="elevation"):
        turnaround_stage.plan("knight", elevation="front")


def test_the_rig_and_clip_are_the_ruling_and_the_idle_row():
    name, glb, clip = turnaround_stage.rig_for()
    assert name == "quaternius" and glb.name == "UAL1_Standard.glb" and clip == "Idle_Loop"
    assert turnaround_stage.rig_for("kaykit")[2] == "Idle"
    with pytest.raises(ValueError, match="unknown rig"):
        turnaround_stage.rig_for("mixamo")


def test_the_seed_follows_the_base_render():
    a = turnaround_stage.derive_seed("knight", "Idle_Loop", "a" * 64)
    b = turnaround_stage.derive_seed("knight", "Idle_Loop", "b" * 64)
    assert a != b and a == turnaround_stage.derive_seed("knight", "Idle_Loop", "a" * 64)


def test_the_clip_prompt_names_the_view():
    assert config.clip_prompt("idle") == config.clip_prompt("idle", "s0")
    assert "side-view, facing right" in config.clip_prompt("idle", "s0")
    assert "back view" in config.clip_prompt("idle", "s90") and "facing right" not in config.clip_prompt("idle", "s90")
    assert "front view, facing the viewer" in config.clip_prompt("idle", "s270")
    assert "top-down" in config.clip_prompt("walk", "t90") and "north" in config.clip_prompt("walk", "t90")


PARTS = "closed great helm, kite shield, longsword, dark cloak"


def test_the_parts_prompt_names_the_parts_and_the_view_per_direction():
    """The multi-view reference feature: with a parts phrase every direction's
    prompt carries the parts, the view phrase, what that view shows of the
    head, and 'exactly one figure'; without one it is the old sentence."""
    plain = config.clip_prompt("idle", "s90")
    assert config.clip_prompt("idle", "s90", None) == plain == config.clip_prompt("idle", "s90", "  ")
    for direction in SIDE:
        text = config.clip_prompt("idle", direction, PARTS)
        assert f"with the same {PARTS} as the reference image" in text, direction
        assert config.clip_view_phrase(direction) in text
        assert config.clip_identity_phrase(direction) in text
        assert "exactly one figure" in text and text.endswith("no ground shadow, no floor")
        assert "standing at rest, breathing" in text
    assert "no face visible" in config.clip_prompt("idle", "s90", PARTS)
    assert "no face visible" in config.clip_prompt("idle", "s45", PARTS)
    assert "no face visible" not in config.clip_prompt("idle", "s270", PARTS)
    assert "seen in profile" in config.clip_prompt("idle", "s0", PARTS)
    assert "from above" in config.clip_prompt("walk", "t90", PARTS)
    # A trailing full stop or comma on the owner's phrase does not double up.
    assert config.clip_prompt("idle", "s0", PARTS + ".") == config.clip_prompt("idle", "s0", PARTS)
    # The negative: the workflow's own, then "a second figure", then the unit's avoid phrase.
    negative = config.clip_negative("色调艳丽，过曝", "bare head, uncovered face")
    assert negative.startswith("色调艳丽，过曝, ") and config.CLIP_NEGATIVE_EXTRA in negative
    assert negative.endswith(", bare head, uncovered face")
    assert config.clip_negative("x", None) == f"x, {config.CLIP_NEGATIVE_EXTRA}"


def test_the_reference_map_follows_the_compass():
    """yaw 0 is the right profile, 180 the left, 90 the back, 270 the front;
    an oblique wants its exact view, then back / front; front is always last."""
    m = config.TURNAROUND_REFERENCE_BY_DIR
    assert m["s0"][0] == "right" and m["s180"][0] == "left"
    assert m["s90"][0] == "back" and m["s270"] == ("front",)
    assert m["s45"] == ("back_right", "back", "front") and m["s135"] == ("back_left", "back", "front")
    assert m["s225"] == ("front_left", "front") and m["s315"] == ("front_right", "front")
    for d in SIDE:
        assert config.turnaround_reference_views(d)[-1] == config.FRONT_VIEW
        assert config.turnaround_reference_views(f"t{config.direction(d)[0]}") == m[d]
    # Every view a direction may want is a view `concept --views` can render.
    for views in m.values():
        assert all(v == "front" or v in config.CONCEPT_VIEWS for v in views)


def test_reference_for_takes_the_nearest_view_and_falls_back_with_a_warning(tmp_path):
    front = (tmp_path / "front.png", "base render")
    back = tmp_path / "back.png"
    right = tmp_path / "right.png"
    for path in (front[0], back, right):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(path)
    views = {"back": {"path": back, "state": "accepted", "provisional": False},
             "right": {"path": right, "state": "pending", "provisional": True},
             "left": {"path": tmp_path / "left.png", "state": "rejected", "provisional": False}}
    s90 = turnaround_stage.reference_for("s90", front, views)
    assert s90["view"] == "back" and s90["fallback"] is False and s90["warning"] is None
    assert s90["file"] == str(back) and len(s90["sha256"]) == 64 and s90["kind"] == "view back"
    s0 = turnaround_stage.reference_for("s0", front, views)
    assert s0["view"] == "right" and s0["provisional"] is True and s0["state"] == "pending"
    # No back_right on the base: s45 takes the back, no warning (the back was on its list).
    s45 = turnaround_stage.reference_for("s45", front, views)
    assert s45["view"] == "back" and s45["fallback"] is False
    # The left is rejected: s180 falls back to the front and says why.
    s180 = turnaround_stage.reference_for("s180", front, views)
    assert s180["view"] == "front" and s180["fallback"] is True and s180["file"] == str(front[0])
    assert "left (rejected)" in s180["warning"] and "--pick-view" in s180["warning"]
    # No front_left: s225 falls back to the front with a warning naming it.
    s225 = turnaround_stage.reference_for("s225", front, views)
    assert s225["fallback"] is True and "front_left (not on the base)" in s225["warning"]
    # The front never warns.
    s270 = turnaround_stage.reference_for("s270", front, views)
    assert s270["view"] == "front" and s270["fallback"] is False and s270["warning"] is None
    assert turnaround_stage.reference_for("s90", front, {})["fallback"] is True


# ---------------------------------------------------------------------------
# 2 + 3 + 4. The run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """All eight side cells once for the module, on the KayKit Knight (the
    test rig) against a FakeComfy that answers five frames per job."""
    if not KNIGHT.exists():
        pytest.skip("rigs/kaykit-adventurers/Knight.glb is not vendored")
    from conftest import PALETTE_CANDIDATE

    tmp_path = tmp_path_factory.mktemp("turnaround")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    if PALETTE_CANDIDATE.exists():
        palette = PALETTE_CANDIDATE
    else:
        palette = tmp_path / "fallback.gpl"
        ramp = [(8 * i, 6 * i, 10 * i) for i in range(16)]
        palette.write_text("GIMP Palette\nName: Fallback\nColumns: 8\n" + "".join(
            f"{r:3d} {g:3d} {b:3d}\t#{r:02x}{g:02x}{b:02x}\n" for r, g, b in ramp))
    # The base: a front render, a parts phrase (the multi-view reference
    # feature) and two views on it - the back (accepted) and the right
    # profile (a provisional pick, still pending); no left, no obliques.
    base_job = base_stage.prepare("knight", cell=(512, 768), work_dir=work_dir)
    base_job.render_dir.mkdir(parents=True, exist_ok=True)
    reference = base_job.render_dir / "concept_pick.png"
    Image.new("RGB", (512, 768), (250, 250, 250)).save(reference)
    Image.new("RGB", (512, 768), (240, 240, 250)).save(base_job.render_dir / "view_back.png")
    Image.new("RGB", (512, 768), (250, 240, 240)).save(base_job.render_dir / "view_right.png")
    base_job.update_manifest(
        render={"seed": 1, "passes": ["concept_pass"], "final": "concept_pick.png"},
        concept={"job": "knight_concept", "candidate": 1, "parts_phrase": PARTS,
                 "avoid_phrase": "bare head, uncovered face",
                 "views": {"back": {"view": "back", "yaw": 90, "file": "view_back.png",
                                    "candidate": 1, "seed": 5, "provisional": False},
                           "right": {"view": "right", "yaw": 0, "file": "view_right.png",
                                     "candidate": 0, "seed": 6, "provisional": True}}})
    curation.set_views("knight", {"back": "accepted"}, work_dir)
    parts_file = _parts_file(tmp_path / "parts.json", palette)
    frame = synthetic.write_strip(tmp_path / "frame.png", cells=1,
                                  cell=tuple(config.TURNAROUND_CELL)).read_bytes()
    logs: list[str] = []
    common = dict(matte="chroma", parts_from=parts_file, rig_name="kaykit", glb=KNIGHT,
                  work_dir=work_dir, assets_dir=tmp_path / "no-assets",
                  workflow_dir=REPO / "gpu" / "workflows", log=logs.append)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PALETTE", palette)
        with FakeComfy(batch=[frame] * config.TURNAROUND_CLIP_FRAMES) as fake:
            state = turnaround_stage.turnaround(
                "knight", client=comfy.ComfyClient(fake.url),
                segmenter=FakeSegmenter(COLOURS), **common)
        yield {"state": state, "logs": logs, "prompts": fake.prompts, "work_dir": work_dir,
               "common": common, "tmp_path": tmp_path, "palette": palette}


def test_every_direction_renders_one_still_from_the_middle_frame(ran):
    state = ran["state"]
    assert state["plan"] == SIDE and state["schema"] == "anim.turnaround/1"
    assert len(ran["prompts"]) == 8, "one Wan job per direction"
    for direction in SIDE:
        record = state["cells"][direction]
        assert record["status"] == "done", record.get("error")
        assert record["seed"] == state["seed"], "one seed for the eight"
        assert record["clip_frames"] == config.TURNAROUND_CLIP_FRAMES
        assert record["still_frame"] == config.TURNAROUND_STILL_INDEX == 2
        assert record["control"] == "blend" and record["depth_gain"] == 0.35
        assert set(record["timings"]) == {"rig", "gpu", "post"}
        assert record["seconds"] > 0
        job = Job.open(record["job"], ran["work_dir"])
        assert job.name == f"knight_idle_{direction}"
        manifest = job.read_manifest()
        assert manifest["motion_source"]["direction"] == direction
        assert manifest["action"] == "idle" and manifest["frames"] == 5
        assert manifest["turnaround"]["status"] == "done"
        assert len(job.clip_frames()) == 5
        assert json.loads(job.picks_path.read_text())["picks"] == [2]
        assert (job.post_dir / "knight_idle_0.png").exists()
        with Image.open(job.post_dir / "knight_idle_0.png") as still:
            assert still.size == tuple(config.CANVAS)
        sil = record["metrics"]["silhouette"]
        assert sil["height"] > 0 and sil["width"] > 0 and sil["pixels"] > 0


def test_the_prompt_names_each_view_and_the_parts(ran):
    texts = [prompt[comfy.find_node(prompt, "positive prompt")]["inputs"]["text"]
             for prompt in ran["prompts"]]
    assert "side-view, facing right" in texts[0]
    assert "back view" in texts[2] and "facing right" not in texts[2]
    assert "front view, facing the viewer" in texts[6]
    assert "standing at rest" in texts[0]
    # The multi-view reference feature: the base's parts phrase and the view
    # sentence are in every direction's prompt, the avoid phrase in the negative.
    for direction, text in zip(SIDE, texts):
        assert text == config.clip_prompt("idle", direction, PARTS), direction
        assert PARTS in text and config.clip_identity_phrase(direction) in text
    negatives = [prompt[comfy.find_node(prompt, "negative prompt")]["inputs"]["text"]
                 for prompt in ran["prompts"]]
    assert all(n.endswith("a second figure, two people, an extra person, bare head, uncovered face")
               for n in negatives)
    assert all(n.startswith("色调艳丽") for n in negatives), "Wan's own negative stays in front"
    record = ran["state"]["cells"]["s90"]
    assert record["parts_phrase"] == PARTS and record["negative"] == negatives[2]
    assert record["prompt"] == texts[2]


def test_each_direction_hands_wan_its_nearest_view(ran):
    """s90 the back (accepted), s0 the right (provisional), s45 / s135 the
    back (no obliques on the base), s180 / s225 / s315 the front as a
    FALLBACK with a warning, s270 the front by right."""
    state = ran["state"]
    refs = state["references"]
    assert list(refs) == SIDE
    used = {d: (refs[d]["view"], refs[d]["fallback"]) for d in SIDE}
    assert used == {"s0": ("right", False), "s45": ("back", False), "s90": ("back", False),
                    "s135": ("back", False), "s180": ("front", True), "s225": ("front", True),
                    "s270": ("front", False), "s315": ("front", True)}
    assert refs["s0"]["provisional"] is True and refs["s0"]["state"] == "pending"
    assert refs["s90"]["state"] == "accepted" and refs["s90"]["kind"] == "view back"
    assert refs["s180"]["warning"] and "left (not on the base)" in refs["s180"]["warning"]
    assert refs["s270"]["warning"] is None and refs["s270"]["kind"] == "base render"
    assert state["reference_map"]["s45"] == ["back_right", "back", "front"]
    # The record per cell says the same, and the file `clip` sent IS that view.
    for d in SIDE:
        cell = state["cells"][d]
        assert cell["reference"]["view"] == refs[d]["view"]
        job = Job.open(f"knight_idle_{d}", ran["work_dir"])
        clip_record = json.loads(job.clip_path.read_text())
        assert clip_record["reference"]["file"] == refs[d]["file"]
        assert clip_record["reference"]["kind"] == ("base render" if refs[d]["view"] == "front"
                                                    else "override")
        assert (job.clip_dir / "reference.png").exists()
    assert refs["s90"]["file"].endswith("knight_base/render/view_back.png")
    assert refs["s270"]["file"].endswith("knight_base/render/concept_pick.png")
    # The warnings were logged before the first direction ran.
    warnings = [line for line in ran["logs"] if "WARNING" in line]
    assert len(warnings) == 3 and all("using the front" in w for w in warnings)
    assert any("reference back" in line for line in ran["logs"] if line.startswith("turnaround: s90"))
    assert any("reference front - fallback" in line for line in ran["logs"] if line.startswith("turnaround: s180"))


def test_each_direction_writes_its_own_parts_json(ran):
    for direction in SIDE:
        job = Job.open(f"knight_idle_{direction}", ran["work_dir"])
        parts = json.loads((job.post_dir / "parts.json").read_text())
        assert parts["schema"] == "anim.parts/1" and parts["direction"] == direction
        assert parts["job"] == job.name and parts["slug"] == "knight"
        assert parts["parts"] == ["sword", "shield", "helmet", "arms", "torso", "legs"]
        assert parts["locked_to"] and "parts.json" in parts["locked_to"]
        # The fake segmenter has no colour for the shield or the arms.
        assert parts["found"]["shield"] is False and parts["palettes"]["shield"] == []
        assert parts["found"]["torso"] is True and parts["locked"]["torso"] is True
        assert parts["pixels"]["torso"] > 0 and parts["palettes"]["torso"]
        for colour in parts["palettes"]["torso"]:
            assert len(colour) == 3
    # And `post --parts-from` accepts it as a palette source.
    from anim.stages import post as post_stage

    palettes, where = post_stage.load_part_palettes(
        "knight", Job.open("knight_idle_s90", ran["work_dir"]).post_dir / "parts.json")
    assert palettes["torso"] and where.endswith("knight_idle_s90/post/parts.json")


def test_the_identity_table_is_numbers_across_directions(ran):
    identity = ran["state"]["identity"]
    assert identity["directions"] == SIDE and identity["done"] == SIDE
    torso = identity["parts"]["torso"]
    assert set(torso["pixels"]) == set(SIDE) and all(torso["pixels"][d] > 0 for d in SIDE)
    assert torso["found"] == SIDE and torso["missing"] == []
    assert torso["spread"] == torso["max"] - torso["min"] >= 0
    assert set(torso["colours"]) == set(SIDE) and set(torso["drift"]) == set(SIDE)
    shield = identity["parts"]["shield"]
    assert shield["found"] == [] and shield["missing"] == SIDE and shield["spread"] is None
    assert set(identity["silhouette"]) == set(SIDE)
    assert identity["height"]["spread"] == identity["height"]["max"] - identity["height"]["min"]
    assert identity["width"]["min"] > 0
    # No verdict anywhere in the record.
    assert "verdict" not in json.dumps(ran["state"]).lower()


def test_the_page_names_every_direction_and_asks_the_question(ran):
    page = Path(ran["state"]["page"])
    html = page.read_text(encoding="utf-8")
    assert turnaround_stage.QUESTION in html
    for direction in SIDE:
        assert f'id="{direction}"' in html
        for kind in ("still", "raw", "control", "parts"):
            assert (page.parent / "img" / f"{direction}_{kind}.png").exists(), (direction, kind)
            assert f"{direction}_{kind}.png" in html
    assert (page.parent / "img" / "turntable.png").exists()
    with Image.open(page.parent / "img" / "turntable.png") as sheet:
        assert sheet.size == (config.CANVAS[0] * 8, config.CANVAS[1])
    assert (page.parent / "img" / "reference.png").exists()
    assert f"seed <b>{ran['state']['seed']}</b>" in html
    # The multi-view reference feature: the reference each still used is on
    # the page - a thumbnail per still, the map, the fallbacks, the parts phrase.
    for direction in SIDE:
        assert (page.parent / "img" / f"{direction}_ref.png").exists()
        assert f"{direction}_ref.png" in html
    assert "fallback to the front" in html and "back_right → back → front" in html
    assert PARTS in html and "(provisional pick)" in html
    assert "not found" in html and "spread" in html
    assert "--accept-gate turnaround" in html
    assert page == turnaround_stage.page_dir("knight", ran["work_dir"]) / "index.html"
    record = json.loads(turnaround_stage.state_path("knight", ran["work_dir"]).read_text())
    assert record["cells"]["s0"]["timings"]["gpu"] >= 0


def test_the_back_cells_hide_the_face_and_the_record_says_so(ran):
    state = ran["state"]
    every = ["nose", "right_eye", "left_eye", "right_ear", "left_ear"]
    assert state["cells"]["s0"]["hidden_face"] == []
    assert state["cells"]["s90"]["hidden_face"] == every
    assert state["cells"]["s45"]["hidden_face"] == every
    assert state["cells"]["s135"]["hidden_face"] == every
    assert state["cells"]["s180"]["hidden_face"] == []
    assert state["cells"]["s270"]["hidden_face"] == []
    payload = json.loads(Job.open("knight_idle_s90", ran["work_dir"]).keypoints_path.read_text())
    nose = payload["names"].index("nose")
    assert all(f["scores"][nose] == 0.0 for f in payload["frames"])
    assert "face hidden: nose" in Path(state["page"]).read_text(encoding="utf-8")


def test_a_second_run_skips_the_directions_that_are_done(ran):
    logs: list[str] = []
    common = dict(ran["common"], log=logs.append)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PALETTE", ran["palette"])
        with FakeComfy(fail_prompt=True) as fake:      # any GPU call would fail loudly
            state = turnaround_stage.turnaround("knight", client=comfy.ComfyClient(fake.url),
                                                segmenter=FakeSegmenter(COLOURS), **common)
    assert not fake.prompts
    assert len([line for line in logs if "skipped" in line]) == 8
    assert all(state["cells"][d]["status"] == "done" for d in SIDE)
    assert state["cells"]["s90"]["metrics"]["silhouette"]["height"] > 0, "the records survive a resume"
    # A page-only rebuild needs no service at all.
    state = turnaround_stage.turnaround("knight", page_only=True, **common)
    assert Path(state["page"]).exists() and state["seed"] == ran["state"]["seed"]


def test_a_changed_base_is_refused_without_force(ran):
    path = turnaround_stage.state_path("knight", ran["work_dir"])
    record = json.loads(path.read_text())
    record["reference"]["sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    try:
        with FakeComfy(fail_prompt=True) as fake:
            with pytest.raises(ValueError, match="--force"):
                turnaround_stage.turnaround("knight", client=comfy.ComfyClient(fake.url),
                                            segmenter=FakeSegmenter(COLOURS), **ran["common"])
    finally:
        record["reference"]["sha256"] = ran["state"]["reference"]["sha256"]
        path.write_text(json.dumps(record))


def test_a_failed_direction_is_recorded_and_the_page_still_builds(ran):
    import shutil

    copy = ran["tmp_path"] / "work_copy"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(ran["work_dir"], copy)
    logs: list[str] = []
    common = dict(ran["common"], log=logs.append, work_dir=copy)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PALETTE", ran["palette"])
        with FakeComfy(fail_prompt=True) as fake:
            state = turnaround_stage.turnaround("knight", dirs=["s180"], force=True,
                                                client=comfy.ComfyClient(fake.url),
                                                segmenter=FakeSegmenter(COLOURS), **common)
    record = state["cells"]["s180"]
    assert record["status"] == "failed" and "ComfyError" in record["error"]
    assert state["cells"]["s0"]["status"] == "done", "the others are untouched"
    assert state["identity"]["done"] == [d for d in SIDE if d != "s180"]
    html = Path(state["page"]).read_text(encoding="utf-8")
    assert "failed" in html and 'id="s180"' in html


# ---------------------------------------------------------------------------
# 5. The API
# ---------------------------------------------------------------------------
def test_get_turnaround_serves_the_directions_and_their_stills(ran, tmp_path):
    app = create_app(ran["work_dir"], web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", cpu_slots=1)
    with TestClient(app) as api:
        row = api.get("/turnaround/knight").json()
        assert row["unit"] == "knight" and row["done"] == 8 and row["of"] == 8
        assert list(row["directions"]) == SIDE
        s90 = row["directions"]["s90"]
        assert s90["status"] == "done" and s90["job"] == "knight_idle_s90"
        assert s90["still"] == "/jobs/knight_idle_s90/frames/post/0.png"
        assert s90["raw"] == "/jobs/knight_idle_s90/frames/clip/2.png"
        assert s90["control"] == "/jobs/knight_idle_s90/frames/control/2.png"
        assert s90["hidden_face"] == ["nose", "right_eye", "left_eye", "right_ear", "left_ear"]
        assert s90["parts"]["torso"]["found"] is True
        for key in ("still", "raw", "control", "skeleton", "depth"):
            response = api.get(s90[key])
            assert response.status_code == 200 and response.headers["content-type"] == "image/png", key
        # The multi-view reference feature: the reference per still, with the
        # URL of the image `clip` sent.
        assert s90["reference"]["view"] == "back" and s90["reference"]["fallback"] is False
        assert s90["reference"]["url"] == "/jobs/knight_idle_s90/frames/reference/0.png"
        assert api.get(s90["reference"]["url"]).status_code == 200
        s180 = row["directions"]["s180"]
        assert s180["reference"]["view"] == "front" and s180["reference"]["fallback"] is True
        assert "left (not on the base)" in s180["reference"]["warning"]
        assert row["references"]["s0"]["view"] == "right" and row["reference_map"]["s0"] == ["right", "front"]
        assert PARTS in s90["prompt"]
        assert row["identity"]["directions"] == SIDE and row["question"] == turnaround_stage.QUESTION
        assert api.get("/turnaround/orc").status_code == 404
        assert api.get("/turnaround/_runs").status_code == 404
        sets = api.get("/jobs/knight_idle_s90/frames").json()["stages"]
        assert len(sets["control"]) == 5 and len(sets["clip"]) == 5


# ---------------------------------------------------------------------------
# 6. The gate
# ---------------------------------------------------------------------------
def test_the_stage_waits_for_the_base_gate_and_never_for_its_own(ran):
    work_dir = ran["work_dir"]
    curation.set_gate("knight", "base", "pending", work_dir)
    curation.set_gate("knight", "turnaround", "pending", work_dir)
    args = cli.build_parser().parse_args(["turnaround", "knight"])
    assert "base gate" in curation.gate_for_namespace("turnaround", args, work_dir)
    curation.set_gate("knight", "base", "accepted", work_dir)
    assert curation.gate_for_namespace("turnaround", args, work_dir) is None, \
        "the turnaround gate is what this stage produces"
    args = cli.build_parser().parse_args(["turnaround", "knight", "--page-only"])
    assert curation.gate_for_namespace("turnaround", args, work_dir) is None
    assert "turnaround" in curation.gate_blocks("knight", "clip", action="idle", direction="s45",
                                                 work_dir=work_dir, mode="gates")


def test_accepting_the_gate_after_the_run_opens_the_other_directions(ran, monkeypatch, capsys):
    work_dir = ran["work_dir"]
    monkeypatch.setattr(config, "WORK_DIR", work_dir)
    curation.set_gate("knight", "turnaround", "pending", work_dir)
    board = curation.board("knight", work_dir)
    assert [c for c in board["cells"] if c.startswith("idle/")] == [f"idle/{d}" for d in SIDE], \
        "the eight stills are the board's idle row"
    assert board["cells"]["idle/s90"]["job"] == "knight_idle_s90"
    assert board["cells"]["idle/s90"]["set"] == "post" and len(board["cells"]["idle/s90"]["strip"]) == 1
    assert board["waiting"][0] == {"kind": "gate", "key": "turnaround", "why": board["waiting"][0]["why"]}
    assert cli.main(["curate", "knight", "--accept-gate", "turnaround"]) == 0
    assert "gate turnaround -> accepted" in capsys.readouterr().out
    record = json.loads((work_dir / "knight_curation.json").read_text())
    assert record["gates"]["turnaround"] == "accepted"
    assert curation.gate_blocks("knight", "clip", action="idle", direction="s45",
                                work_dir=work_dir, mode="gates") is None
