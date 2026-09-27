"""The curation board feature: the record, the gates, the board over the API,
and the CLI that types the same file.

The unit is the knight with three cell jobs: the posed `knight_strike`
(attack/s0, the fixture every review test uses), a hand-made `knight_idle`
(idle/s0) and a rig-style `knight_walk_s45` (walk/s45). No GPU: a regenerate
goes through the run manager with the fake launcher from test_api.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anim import cli, config, curation
from anim.api import runs as runs_api
from anim.api import settings as settings_api
from anim.api.app import create_app
from anim.job import Job, new_manifest
from anim.stages import pick as pick_stage
from anim.stages.review import ReviewApp
from test_api import FAKE_STAGE, wait_for


def _job(work_dir: Path, name: str, action: str, *, frames: int = 4,
         direction: str | None = None, slug: str = "knight") -> Job:
    job = Job.open(name, work_dir)
    job.frames_dir.mkdir(parents=True)
    for i in range(frames):
        Image.new("RGBA", (64, 64), (i * 40, 80, 120, 255)).save(
            job.frames_dir / f"{i + 1:04d}.png")
    manifest = new_manifest(clip=f"{name}.mp4", clip_sha256="b" * 64, fps=12,
                            frames=frames, action=action, slug=slug,
                            frame_size=[64, 64], crop=None, source_fps=12.0,
                            created="2026-09-21T00:00:00+00:00")
    if direction:
        manifest["motion_source"] = {"kind": "rig", "direction": direction}
    job.write_manifest(manifest)
    return job


@pytest.fixture
def unit(posed_job, work_dir):
    """The knight's three cells; picks on the posed one."""
    pick_stage.pick(posed_job.name, work_dir=work_dir)
    _job(work_dir, "knight_idle", "idle")
    _job(work_dir, "knight_walk_s45", "walk", direction="s45")
    (work_dir / "sheets").mkdir()
    return "knight"


@pytest.fixture
def api(unit, work_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.2")
    manager = runs_api.RunManager(work_dir, cpu_slots=1,
                                  python=[sys.executable, "-c", FAKE_STAGE])
    app = create_app(work_dir, web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", runs=manager)
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def test_the_record_round_trips_beside_the_jobs(unit, work_dir):
    assert curation.read(unit, work_dir) is None, "nothing curated yet"
    curation.set_gate(unit, "base", "pending", work_dir)
    curation.set_cell_status(unit, "strike", "s0", "accepted", work_dir)
    cell = curation.set_frames(unit, "idle", "s0", {2: "reject", "0": "accept"}, work_dir)
    assert cell["frames"] == {"0": "accept", "2": "reject"}, "keys are frame numbers, as strings"
    path = work_dir / "knight_curation.json"
    assert path.exists() and curation.curation_path(unit, work_dir) == path
    record = json.loads(path.read_text())
    assert record["schema"] == "anim.curation/1" and record["unit"] == "knight"
    assert record["gates"] == {"base": "pending"}, \
        "the first edit on a legacy unit opens the gate; it was then closed"
    assert record["origin"] == "legacy"
    assert record["cells"]["attack/s0"]["job"] == "knight_strike", \
        "the verb and the studio name are the same cell"
    assert record["cells"]["attack/s0"]["status"] == "accepted"
    assert record["cells"]["attack/s0"]["decided_at"]
    assert record["cells"]["idle/s0"] == {
        "job": "knight_idle", "status": "pending", "frames": {"0": "accept", "2": "reject"},
        "seed": None, "regenerated_from": None, "decided_at": None}
    assert curation.read(unit, work_dir)["cells"] == record["cells"]


def test_a_legacy_unit_reads_as_gates_accepted_and_is_not_written(unit, work_dir):
    record = curation.load(unit, work_dir)
    assert record["origin"] == "legacy"
    assert record["gates"] == {"base": "accepted"}
    assert not (work_dir / "knight_curation.json").exists()
    board = curation.board(unit, work_dir)
    assert board["exists"] is False and board["gates"]["base"] == "accepted"


def test_the_board_maps_the_units_jobs_onto_cells_in_direction_major_order(unit, work_dir):
    board = curation.board(unit, work_dir)
    assert list(board["cells"]) == ["attack/s0", "idle/s0", "walk/s45"]
    assert board["directions"] == ["s0", "s45", "s90", "s135", "s180", "s225", "s270", "s315"], \
        "the columns are the eight side facings, cells or no cells"
    assert board["actions"] == ["attack", "idle", "walk", "run", "jump"]
    attack = board["cells"]["attack/s0"]
    assert attack["job"] == "knight_strike" and attack["verb"] == "strike"
    assert attack["set"] == "frames" and attack["stage"] == "render"
    assert [f["index"] for f in attack["strip"]] == attack["picks"], \
        "with picks, the strip is the picks"
    assert attack["strip"][0]["url"] == f"/jobs/knight_strike/frames/frames/{attack['picks'][0]}.png"
    idle = board["cells"]["idle/s0"]
    assert idle["picks"] == [] and [f["index"] for f in idle["strip"]] == [0, 1, 2, 3], \
        "no picks yet: every frame"
    assert board["cells"]["walk/s45"]["stage"] == "clip", "a rig job with no strip is the clip route"
    assert [w["key"] for w in board["waiting"]] == ["attack/s0", "idle/s0", "walk/s45"]


def test_the_board_ignores_other_units_and_the_bakeoff_cells(unit, work_dir):
    _job(work_dir, "orc_strike", "strike").update_manifest(slug="orc")
    _job(work_dir, "bakeoff_knight_attack_V-kaykit", "strike")
    _job(work_dir, "knight_base", "")
    assert list(curation.board(unit, work_dir)["cells"]) == ["attack/s0", "idle/s0", "walk/s45"]
    assert [u["unit"] for u in curation.list_units(work_dir)] == ["knight", "orc"]


def test_two_jobs_on_one_cell_prefer_the_canonical_name(unit, work_dir):
    shutil.copytree(work_dir / "knight_strike", work_dir / "knight_strike_phases")
    assert curation.board(unit, work_dir)["cells"]["attack/s0"]["job"] == "knight_strike"


def test_the_post_set_is_the_strip_when_it_matches_the_picks(unit, work_dir):
    job = Job.open("knight_strike", work_dir)
    picks = json.loads(job.picks_path.read_text())["picks"]
    job.post_dir.mkdir()
    for k in range(len(picks)):
        Image.new("RGBA", (64, 64)).save(job.post_dir / f"knight_strike_{k}.png")
    cell = curation.board(unit, work_dir)["cells"]["attack/s0"]
    assert cell["set"] == "post"
    assert [f["index"] for f in cell["strip"]] == sorted(picks)
    assert cell["strip"][1]["url"] == "/jobs/knight_strike/frames/post/1.png", \
        "post frame k is pick k, shown under the source frame's number"
    # A reject drops the pick but post/ still has its frame: the strip keeps
    # showing post, the rejected one dimmed in its old slot.
    curation.set_frames(unit, "attack", "s0", {picks[1]: "reject"}, work_dir)
    cell = curation.board(unit, work_dir)["cells"]["attack/s0"]
    assert cell["set"] == "post" and cell["picks"] == [p for p in picks if p != picks[1]]
    assert [(f["index"], f["mark"]) for f in cell["strip"]][1] == (picks[1], "reject")
    assert cell["strip"][1]["url"] == "/jobs/knight_strike/frames/post/1.png"


# ---------------------------------------------------------------------------
# A unit with nothing on it yet - the board's empty state (2026-09-23)
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh(work_dir):
    """`concept --pick`, one `--views back,left,top` and nothing since: a
    record, a base that carries no view yet, one attempt at three of the eight
    views, no cells. This is hooded_archer as it stood on
    2026-09-23, the unit that showed the board reporting an empty unit as
    all-clear."""
    from anim.stages import base as base_stage
    from anim.stages import concept as concept_stage

    base_stage.prepare("archer", work_dir=work_dir)
    record = concept_stage.empty_views("archer")
    views_dir = concept_stage.views_dir("archer", work_dir)
    views_dir.mkdir(parents=True, exist_ok=True)
    for view in ("back", "left", "top"):
        Image.new("RGB", (8, 8), (255, 255, 255)).save(views_dir / f"{view}_00.png")
        row = concept_stage.view_row(view, "a prompt")
        row["candidates"] = [{"index": 0, "file": f"{view}_00.png", "source": "views"}]
        record["views"][view] = row
    concept_stage.views_path("archer", work_dir).write_text(json.dumps(record))
    curation.accept_base_from_concept("archer", work_dir)
    return "archer"


def _pick_view(work_dir: Path, unit: str, view: str) -> None:
    """What `concept --pick-view` leaves behind: the file on the base job and
    the row in its `concept.views` block."""
    base_job = Job.open(f"{unit}_base", work_dir)
    base_job.render_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(base_job.render_dir / f"view_{view}.png")
    manifest = base_job.read_manifest()
    views = dict((manifest.get("concept") or {}).get("views") or {})
    views[view] = {"view": view, "yaw": config.CONCEPT_VIEWS[view]["yaw"],
                   "file": f"view_{view}.png", "candidate": 0, "seed": 7,
                   "picked": "2026-09-22T17:47:00+00:00"}
    base_job.update_manifest(concept={**(manifest.get("concept") or {}), "views": views})


def test_a_unit_with_no_cells_still_has_the_eight_columns(fresh, unit, work_dir):
    """Fix 1: the grid's columns are the eight side facings whether or not a
    cell landed in them, plus any other direction a cell does use."""
    board = curation.board(fresh, work_dir)
    assert board["cells"] == {}, "nothing rendered for this unit"
    assert board["directions"] == [f"s{yaw}" for yaw in config.DIRECTION_YAWS]
    # A top cell is a column too: the eight are a floor, not a whitelist.
    _job(work_dir, "knight_idle_t90", "idle", direction="t90")
    assert curation.board(unit, work_dir)["directions"][-1] == "t90"
    assert curation.board_directions([]) == curation.board_directions(["attack/s45"])


def test_waiting_names_the_missing_views(fresh, work_dir):
    """Fix 2: a unit with no cells is not "nothing waiting on you" - it is
    the whole studio path, from the views up."""
    waiting = curation.board(fresh, work_dir)["waiting"]
    assert waiting, "everything is waiting on the owner, not nothing"
    views = [w for w in waiting if w["kind"] == "view"]
    assert [w["key"] for w in views] == ["right", "back_right", "back_left", "front_left",
                                         "front_right", "back", "left", "top"], \
        "the five nobody has drawn first, then the three drawn and not picked"
    assert "anim concept --unit archer --views right" in views[0]["why"]
    assert "s0 wants it" in views[0]["why"], "the direction that asks for it first"
    assert "1 attempt at the back view and none picked" in views[5]["why"]
    assert "--pick-view back=NN" in views[5]["why"]
    _pick_view(work_dir, fresh, "back")
    judge = [w for w in curation.board(fresh, work_dir)["waiting"] if w["key"] == "back"]
    assert judge[0]["why"] == "judge the back view of the base - accept it or reject it", \
        "picked, so the thing waiting is a verdict and not a pick"
    assert not any(w["kind"] == "turnaround" for w in waiting), "no such step any more"
    assert [w["kind"] for w in waiting].count("gate") == 0, "the base gate is accepted"
    assert waiting[-1]["kind"] == "parts", "the base's part palettes, after the views"
    assert curation.board(fresh, work_dir, mode="end")["waiting"] == [], \
        "end-only review: nothing is asked of the owner until the end"


def test_an_old_record_with_a_turnaround_gate_loads_and_is_ignored(fresh, work_dir):
    """Records written while the turnaround stage existed carry
    `gates.turnaround`. They load without error, the key stays in the file as
    written, and the board, the waiting list and every gate check ignore it."""
    path = curation.curation_path(fresh, work_dir)
    record = json.loads(path.read_text())
    record["gates"]["turnaround"] = "pending"
    path.write_text(json.dumps(record))
    board = curation.board(fresh, work_dir)
    assert board["gates"] == {"base": "accepted"}
    assert board["gate_blocked"] == {"base": None}
    assert "stills" not in board
    assert not any(w["key"] == "turnaround" for w in board["waiting"])
    assert curation.gate_blocks(fresh, "matrix", direction="s90", work_dir=work_dir,
                                mode="gates") is None
    assert curation.read(fresh, work_dir)["gates"]["turnaround"] == "pending", \
        "left in the file as written"
    with pytest.raises(curation.CurationError) as raised:
        curation.set_gate(fresh, "turnaround", "accepted", work_dir)
    assert raised.value.status == 404


def test_the_views_ring_has_all_eight_slots(fresh, work_dir):
    """Fix 4: every view of `config.CONCEPT_VIEWS` is a slot, with the three
    states the ring draws - rendered and accepted, rendered and pending, and
    missing entirely."""
    views = curation.board(fresh, work_dir)["views"]
    assert list(views) == list(config.CONCEPT_VIEWS)
    assert all(row["missing"] for row in views.values()), "nothing is on the base yet"
    assert {v: row["attempts"] for v, row in views.items()} == {
        "back": 1, "left": 1, "top": 1, "right": 0, "back_right": 0, "back_left": 0,
        "front_left": 0, "front_right": 0}, \
        "an attempt nobody picked is a different gap from a view nobody drew"
    assert views["right"] == {**views["right"], "missing": True, "attempts": 0, "file": None,
                              "url": None, "state": "pending", "wanted_by": ["s0"]}, \
        "the one s0 hands Wan is the one that is not there at all"
    assert views["top"]["elevation"] == "top" and views["top"]["wanted_by"] == []
    _pick_view(work_dir, fresh, "back")
    curation.set_views(fresh, {"back": "accepted"}, work_dir)
    views = curation.board(fresh, work_dir)["views"]
    assert views["back"] == {**views["back"], "missing": False, "state": "accepted",
                             "file": "view_back.png", "wanted_by": ["s90"]}
    assert views["back"]["url"].startswith("/jobs/archer_base/frames/render/")
    assert views["left"]["state"] == "pending" and views["left"]["missing"] is True
    assert curation.wanted_by("back") == ["s90"]


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", curation.MODES)
def test_no_record_means_nothing_is_gated(unit, work_dir, mode):
    for stage in ("clip", "render", "bakeoff", "run"):
        assert curation.gate_blocks(unit, stage, action="strike", direction="s45",
                                    work_dir=work_dir, mode=mode) is None


def test_gates_mode_waits_for_the_base_only(unit, work_dir):
    curation.set_gate(unit, "base", "pending", work_dir)
    reason = curation.gate_blocks(unit, "clip", action="strike", work_dir=work_dir, mode="gates")
    assert reason and "base gate is pending" in reason and "concept --unit knight --pick" in reason
    assert curation.gate_blocks(unit, "pick", action="strike", work_dir=work_dir,
                                mode="gates") is None, "a CPU stage never waits"
    assert curation.gate_blocks(unit, "clip", work_dir=work_dir, mode="end") is None
    curation.set_gate(unit, "base", "accepted", work_dir)
    assert curation.gate_blocks(unit, "clip", action="strike", work_dir=work_dir,
                                mode="gates") is None
    # A direction outside s0 no longer waits for a second gate: its view,
    # accepted on the Board, is what the matrix checks per cell.
    assert curation.gate_blocks(unit, "clip", action="walk", direction="s45",
                                work_dir=work_dir, mode="gates") is None
    assert curation.gate_blocks(unit, "matrix", direction="s90", work_dir=work_dir,
                                mode="gates") is None


def test_every_mode_orders_the_cells_direction_major(unit, work_dir):
    curation.set_gate(unit, "base", "accepted", work_dir)     # a record, the gate open
    assert curation.gate_blocks(unit, "render", action="strike", work_dir=work_dir,
                                mode="every") is None, "the first cell waits for nobody"
    reason = curation.gate_blocks(unit, "render", action="idle", work_dir=work_dir, mode="every")
    assert reason and "attack/s0 (knight_strike) is pending" in reason
    assert curation.gate_blocks(unit, "render", action="idle", work_dir=work_dir,
                                mode="gates") is None, "only `every` orders cells"
    curation.set_cell_status(unit, "attack", "s0", "accepted", work_dir)
    assert curation.gate_blocks(unit, "render", action="idle", work_dir=work_dir,
                                mode="every") is None
    reason = curation.gate_blocks(unit, "clip", action="walk", direction="s45",
                                  work_dir=work_dir, mode="every")
    assert reason and "idle/s0 (knight_idle)" in reason, "every s0 cell before any s45"
    curation.set_cell_status(unit, "idle", "s0", "rejected", work_dir)
    assert "is rejected" in curation.gate_blocks(unit, "clip", action="walk", direction="s45",
                                                 work_dir=work_dir, mode="every")
    board = curation.board(unit, work_dir, mode="every")
    assert [w["key"] for w in board["waiting"] if w["kind"] == "cell"] == ["idle/s0"], \
        "one CELL at a time; the views step above it is its own queue"


def test_the_matrix_waits_for_the_base_gate_and_a_dry_run_never_does(unit, work_dir):
    """The action matrix feature: `matrix` is base-gated like every GPU stage;
    a dry run spends nothing, so it is never a 423 - refusing it would hide
    the very prerequisite reasons the owner opened it to read."""
    curation.set_gate(unit, "base", "pending", work_dir)
    reason = curation.gate_blocks(unit, "matrix", action="strike",
                                  direction=config.DEFAULT_DIRECTION,
                                  work_dir=work_dir, mode="gates")
    assert reason and "base gate is pending" in reason
    args = cli.build_parser().parse_args(["matrix", "knight"])
    assert "base gate" in curation.gate_for_namespace("matrix", args, work_dir)
    dry = cli.build_parser().parse_args(["matrix", "knight", "--dry-run"])
    assert curation.gate_for_namespace("matrix", dry, work_dir) is None
    assert curation.gate_for_request("matrix", ["knight", "--dry-run"], work_dir) is None
    assert "base gate" in curation.gate_for_request("matrix", ["knight"], work_dir)
    curation.set_gate(unit, "base", "accepted", work_dir)
    assert curation.gate_for_namespace("matrix", args, work_dir) is None


def test_the_gate_is_read_off_the_stages_own_arguments(unit, work_dir):
    curation.set_gate(unit, "base", "pending", work_dir)
    args = cli.build_parser().parse_args(["render", "knight_strike"])
    assert "base gate" in curation.gate_for_namespace("render", args, work_dir)
    args = cli.build_parser().parse_args(["bakeoff", "--unit", "knight", "--action", "attack"])
    assert "base gate" in curation.gate_for_namespace("bakeoff", args, work_dir)
    args = cli.build_parser().parse_args(["bakeoff", "--unit", "knight", "--action", "attack",
                                          "--page-only"])
    assert curation.gate_for_namespace("bakeoff", args, work_dir) is None, "no GPU in it"
    args = cli.build_parser().parse_args(["render", "nope"])
    assert curation.gate_for_namespace("render", args, work_dir) is None, \
        "a missing job is the stage's own error"
    assert curation.gate_for_request("render", ["--force", "knight_strike"], work_dir) is None
    assert "base gate" in curation.gate_for_request("render", ["knight_strike"], work_dir)


def test_the_cli_refuses_a_gated_stage_and_says_why(unit, work_dir, monkeypatch, capsys):
    monkeypatch.setattr(config, "WORK_DIR", work_dir)
    curation.set_gate(unit, "base", "pending", work_dir)
    assert cli.main(["render", "knight_strike"]) == 2
    out = capsys.readouterr().out
    assert "anim render: refused" in out and "base gate is pending" in out and "--force" in out
    assert not Job.open("knight_strike", work_dir).render_dir.exists()


def test_concept_pick_accepts_the_base_gate(work_dir, workflow_dir):
    from test_concept import _concept_job
    from anim.stages import concept as concept_stage

    _concept_job(work_dir, workflow_dir)
    curation.set_gate("knight", "base", "pending", work_dir)
    result = concept_stage.pick("knight", "01", work_dir=work_dir)
    assert result["gates"] == {"base": "accepted"}
    record = json.loads((work_dir / "knight_curation.json").read_text())
    assert record["gates"] == {"base": "accepted"}
    assert record["gate_log"][-1]["by"] == "concept --pick"


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------
def test_post_jobs_is_423_while_the_gate_is_pending(api, unit, work_dir):
    curation.set_gate(unit, "base", "pending", work_dir)
    response = api.post("/jobs", json={"stage": "render", "args": {"job": "knight_strike"}})
    assert response.status_code == 423
    assert response.json()["gate"] is True and "base gate" in response.json()["error"]
    assert api.get("/runs").json() == [], "nothing started"
    cpu = api.post("/jobs", json={"stage": "pick", "args": {"job": "knight_strike"}})
    assert cpu.status_code == 202, "CPU stages are not gated"
    wait_for(api, cpu.json()["id"])
    forced = api.post("/jobs", json={"stage": "render",
                                     "args": {"job": "knight_strike", "force": True}})
    assert forced.status_code == 202
    wait_for(api, forced.json()["id"])
    api.put("/settings", json={"curation": "end"})
    assert api.post("/jobs", json={"stage": "render", "args": {"job": "knight_strike"}}
                    ).status_code == 202, "end mode: nothing blocks"
    assert api.post("/jobs", json={"stage": "nope", "args": {}}).status_code == 400, \
        "an unknown stage is still the manager's 400"


def test_get_curation_lists_units_and_the_board(api, unit):
    listing = api.get("/curation").json()
    assert listing["mode"] == "gates"
    assert listing["units"] == [{"unit": "knight", "cells": 3, "record": False}]
    board = api.get("/curation/knight").json()
    assert set(board["cells"]) == {"attack/s0", "idle/s0", "walk/s45"}
    assert board["waiting"][0]["kind"] == "cell"
    assert api.get("/curation/orc").json()["cells"] == {}
    assert api.get("/curation/../x").status_code in (404, 422)


def test_put_cell_frames_rewrites_picks_exactly_as_review_does(api, unit, work_dir, tmp_path):
    job = Job.open("knight_strike", work_dir)
    picks = json.loads(job.picks_path.read_text())["picks"]
    dropped = picks[1]
    twin_root = tmp_path / "twin"
    shutil.copytree(job.root, twin_root / job.name)
    twin = Job.open(job.name, twin_root)
    expected_state = ReviewApp(twin).set_picks([p for p in picks if p != dropped])
    expected_bytes = twin.picks_path.read_bytes()

    response = api.put("/curation/knight/cells/attack/s0",
                       json={"frames": {str(dropped): "reject"}})
    assert response.status_code == 200, response.text
    cell = response.json()
    assert cell["frames"] == {str(dropped): "reject"}
    assert cell["picks"] == expected_state["picks"]
    assert job.picks_path.read_bytes() == expected_bytes
    assert [f["index"] for f in cell["strip"]] == picks, \
        "the rejected frame stays on the board, marked, so it can be taken back"
    assert [f["mark"] for f in cell["strip"] if f["index"] == dropped] == ["reject"]
    assert api.get("/jobs/knight_strike/picks").json() == ReviewApp(job).state()

    # Clearing the reject brings the frame back; the other picks are untouched.
    back = api.put("/curation/knight/cells/attack/s0", json={"frames": {str(dropped): None}}).json()
    assert back["frames"] == {} and back["picks"] == picks
    expected_state = ReviewApp(twin).set_picks(picks)
    assert job.picks_path.read_bytes() == twin.picks_path.read_bytes()


def test_accepting_a_cell_locks_its_frames_until_reopened(api, unit):
    accepted = api.put("/curation/knight/cells/strike/s0", json={"status": "accepted"}).json()
    assert accepted["status"] == "accepted" and accepted["key"] == "attack/s0"
    locked = api.put("/curation/knight/cells/attack/s0", json={"frames": {"0": "reject"}})
    assert locked.status_code == 409 and "locked" in locked.json()["detail"]
    api.put("/curation/knight/cells/attack/s0", json={"status": "pending"})
    assert api.put("/curation/knight/cells/attack/s0",
                   json={"frames": {"0": "reject"}}).status_code == 200
    assert api.put("/curation/knight/cells/attack/s0", json={}).status_code == 400
    assert api.put("/curation/knight/cells/attack/s0", json={"status": "maybe"}).status_code == 400
    assert api.put("/curation/knight/cells/dance/s0", json={"status": "accepted"}).status_code == 400
    assert api.put("/curation/knight/cells/jump/s0", json={"status": "accepted"}).status_code == 404, \
        "no job for that cell"
    record = json.loads((api.app.state.work_dir / "knight_curation.json").read_text())
    assert record["origin"] == "legacy" and record["gates"]["base"] == "accepted", \
        "a board edit on a legacy unit keeps it ungated"


def test_regenerate_records_the_seed_and_runs_through_the_manager(api, unit, work_dir, monkeypatch):
    Job.open("knight_strike", work_dir).update_manifest(render={"seed": 5})
    monkeypatch.setenv("FAKE_DELAY", "1.0")
    response = api.post("/curation/knight/cells/attack/s0/regenerate")
    assert response.status_code == 202, response.text
    cell = response.json()
    run = cell["run"]
    assert run["stage"] == "render" and run["gpu"] is True
    assert run["argv"] == ["--seed", str(cell["seed"]), "knight_strike"]
    assert cell["regenerated_from"] == 5 and cell["seed"] != 5
    assert cell["status"] == "pending" and cell["frames"] == {}
    assert response.headers["Location"] == f"/runs/{run['id']}"
    record = json.loads((work_dir / "knight_curation.json").read_text())["cells"]["attack/s0"]
    assert record["seed"] == cell["seed"] and record["run"] == run["id"]

    busy = api.post("/curation/knight/cells/idle/s0/regenerate")
    assert busy.status_code == 409 and busy.json()["run"]["id"] == run["id"], \
        "a GPU run: the one-at-a-time rule"
    assert "idle/s0" not in json.loads((work_dir / "knight_curation.json").read_text())["cells"], \
        "a refused regenerate records nothing"
    wait_for(api, run["id"])

    again = api.post("/curation/knight/cells/attack/s0/regenerate", json={"seed": 99}).json()
    assert again["seed"] == 99 and again["regenerated_from"] == cell["seed"]
    wait_for(api, again["run"]["id"])
    assert api.post("/curation/knight/cells/jump/s0/regenerate").status_code == 404

    curation.set_gate("knight", "base", "pending", work_dir)
    assert api.post("/curation/knight/cells/attack/s0/regenerate").status_code == 423


def test_put_gate_and_the_settings_mode(api, unit, work_dir):
    board = api.put("/curation/knight/gates/base", json={"state": "pending"}).json()
    assert board["gates"] == {"base": "pending"}
    assert board["waiting"][0] == {"kind": "gate", "key": "base",
                                   "why": board["waiting"][0]["why"]}
    assert api.put("/curation/knight/gates/turnaround",
                   json={"state": "accepted"}).status_code == 404, "no such gate any more"
    api.put("/curation/knight/gates/base", json={"state": "accepted"})
    assert api.put("/curation/knight/gates/side", json={"state": "accepted"}).status_code == 404
    assert api.put("/curation/knight/gates/base", json={"state": "open"}).status_code == 400
    assert api.put("/settings", json={"curation": "every"}).json()["curation"] == "every"
    assert api.get("/curation/knight").json()["mode"] == "every"


def test_put_view_is_the_views_step_under_the_base_gate(api, unit, work_dir):
    """The multi-view reference feature: `PUT /curation/{unit}/views/{view}`
    accepts / rejects / reopens one view of the base and the board carries
    the row (state, file URL when the base has the view, provisional)."""
    from anim.stages import base as base_stage

    base_job = base_stage.prepare("knight", work_dir=work_dir)
    base_job.render_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (255, 255, 255)).save(base_job.render_dir / "view_back.png")
    base_job.update_manifest(concept={"views": {"back": {"view": "back", "yaw": 90,
                                                         "file": "view_back.png", "candidate": 1,
                                                         "seed": 7, "provisional": True}}})
    board = api.get("/curation/knight").json()
    assert list(board["views"]) == list(config.CONCEPT_VIEWS), "all eight slots, table order"
    assert board["views"]["back"] == {**board["views"]["back"], "state": "pending",
                                      "missing": False, "provisional": True,
                                      "candidate": 1, "seed": 7,
                                      "url": "/jobs/knight_base/frames/render/0.png"}
    assert api.get(board["views"]["back"]["url"]).status_code == 200
    board = api.put("/curation/knight/views/back", json={"state": "accepted"}).json()
    assert board["views"]["back"]["state"] == "accepted"
    assert json.loads((work_dir / "knight_curation.json").read_text())["views"] == {"back": "accepted"}
    board = api.put("/curation/knight/views/left", json={"state": "rejected"}).json()
    assert board["views"]["left"] == {**board["views"]["left"], "state": "rejected", "file": None,
                                      "url": None, "provisional": False, "missing": True}, \
        "a view the base does not carry still takes a state - it just has nothing to show"
    assert api.put("/curation/knight/views/top", json={"state": "accepted"}).status_code == 200
    assert api.put("/curation/knight/views/under", json={"state": "accepted"}).status_code == 404
    assert api.put("/curation/knight/views/back", json={"state": "maybe"}).status_code == 400
    assert board["gates"]["base"] == "accepted", "the views step moves no gate"


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------
def test_anim_curate_types_the_same_file(unit, work_dir, monkeypatch, capsys):
    monkeypatch.setattr(config, "WORK_DIR", work_dir)
    job = Job.open("knight_strike", work_dir)
    picks = json.loads(job.picks_path.read_text())["picks"]
    assert cli.main(["curate", "knight", "--reject-frame", f"attack/s0/{picks[0]}",
                     "--accept-cell", "idle/s0", "--accept-gate", "base"]) == 0
    out = capsys.readouterr().out
    assert f"attack/s0/{picks[0]} rejected; picks now {picks[1:]}" in out
    assert "idle/s0 -> accepted (knight_idle)" in out
    assert "gate base -> accepted" in out
    assert "waiting on you:" in out and "walk/s45" in out
    record = json.loads((work_dir / "knight_curation.json").read_text())
    assert record["cells"]["attack/s0"]["frames"] == {str(picks[0]): "reject"}
    assert record["cells"]["idle/s0"]["status"] == "accepted"
    assert json.loads(job.picks_path.read_text())["picks"] == picks[1:]

    assert cli.main(["curate", "knight", "--accept-view", "back", "--reject-view", "left"]) == 0
    out = capsys.readouterr().out
    assert "view back -> accepted" in out and "view left -> rejected" in out
    assert "views: back=accepted (no file), left=rejected (no file)" in out
    assert json.loads((work_dir / "knight_curation.json").read_text())["views"] \
        == {"back": "accepted", "left": "rejected"}

    assert cli.main(["curate", "knight", "--regenerate", "attack/s0", "--seed", "42"]) == 0
    out = capsys.readouterr().out
    assert "anim render knight_strike --seed 42" in out, "printed, never started"
    assert not (work_dir / "_runs").exists()

    assert cli.main(["curate", "--mode", "end"]) == 0
    assert settings_api.stored(work_dir)["curation"] == "end"
    assert cli.main(["curate"]) == 0
    assert "knight" in capsys.readouterr().out

    with pytest.raises(SystemExit) as raised:
        cli.main(["curate", "knight", "--accept-frame", "attack/s0"])
    assert "<action>/<direction>/<n>" in str(raised.value)
    with pytest.raises(SystemExit) as raised:
        cli.main(["curate", "knight", "--accept-cell", "jump/s0"])
    assert "no job for cell jump/s0" in str(raised.value)


def test_the_board_reports_whether_the_base_has_measured_part_palettes(unit, work_dir):
    """The part palettes every direction locks its colours to.

    `config.UNIT_PARTS` DEFINES a unit's regions; `<unit>_base/post/parts.json`
    MEASURES them. Every matrix cell locks to the second, and `post` would
    only ask for it after Wan had rendered the clip (2026-09-24, twice). The
    board reports it so the Board can say so, and offer the one CPU job that
    fixes it, before the card is spent.
    """
    # The step is named only for a unit on the studio's path: `waiting_on`
    # leaves a legacy unit (no record, nothing gated) alone on purpose, so the
    # record has to exist before the parts step can be waited on.
    curation.set_gate("knight", "base", "accepted", work_dir=work_dir)

    board = curation.board("knight", work_dir)
    assert board["base_parts"] is False        # the fixture has no _base job
    assert board["base_parts_job"] == "knight_base"
    parts = [w for w in board["waiting"] if w["kind"] == "parts"]
    assert len(parts) == 1
    assert parts[0]["key"] == "knight_base"
    assert "anim post knight_base --parts" in parts[0]["why"]
    kinds = [w["kind"] for w in board["waiting"]]
    assert "turnaround" not in kinds

    # Measured: the step disappears and nothing else about the board moves.
    measured = work_dir / "knight_base" / "post"
    measured.mkdir(parents=True)
    (measured / "parts.json").write_text('{"schema": "anim.parts/1", "parts": []}')
    after = curation.board("knight", work_dir)
    assert after["base_parts"] is True
    assert [w for w in after["waiting"] if w["kind"] == "parts"] == []
    assert curation.base_part_palettes("knight", work_dir) is True
    assert curation.base_part_palettes("hooded_archer", work_dir) is False
