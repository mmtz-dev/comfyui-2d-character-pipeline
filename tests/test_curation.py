"""The base gate and the views step: the record, the gate the GPU stages ask
before spending the card, the views ring, and the API's 423.

The unit is the knight with three cell jobs: the posed `knight_strike`
(attack/s0, the fixture every review test uses), a hand-made `knight_idle`
(idle/s0) and a rig-style `knight_walk_s45` (walk/s45). No GPU: a run goes
through the run manager with the fake launcher from test_api.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anim import cli, config, curation
from anim.api import runs as runs_api
from anim.api.app import create_app
from anim.job import Job, new_manifest
from anim.stages import pick as pick_stage
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


def _gate(unit: str, state: str, work_dir: Path) -> None:
    """The base gate as `concept --pick` (accepted) or an owner reopening it
    (pending) leaves it."""
    record = curation.load(unit, work_dir)
    record["gates"]["base"] = state
    curation.write(unit, record, work_dir)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def test_the_record_round_trips_beside_the_jobs(unit, work_dir):
    assert curation.read(unit, work_dir) is None, "nothing curated yet"
    _gate(unit, "pending", work_dir)
    path = work_dir / "knight_curation.json"
    assert path.exists() and curation.curation_path(unit, work_dir) == path
    record = json.loads(path.read_text())
    assert record["schema"] == "anim.curation/1" and record["unit"] == "knight"
    assert record["gates"] == {"base": "pending"} and record["origin"] == "legacy"
    assert "cells" not in record, "the board's per-cell records are retired"
    assert curation.recorded_units(work_dir) == ["knight"]


def test_a_legacy_unit_reads_as_gates_accepted_and_is_not_written(unit, work_dir):
    record = curation.load(unit, work_dir)
    assert record["origin"] == "legacy"
    assert record["gates"] == {"base": "accepted"}
    assert not (work_dir / "knight_curation.json").exists()
    assert curation.recorded_units(work_dir) == []


def test_an_old_record_with_cells_and_a_turnaround_gate_loads(unit, work_dir):
    """Records written while the board and the turnaround stage existed carry
    `cells` and `gates.turnaround`. They load, stay in the file as written,
    and no gate check reads them."""
    _gate(unit, "accepted", work_dir)
    path = curation.curation_path(unit, work_dir)
    record = json.loads(path.read_text())
    record["gates"]["turnaround"] = "pending"
    record["cells"] = {"attack/s0": {"status": "rejected"}}
    path.write_text(json.dumps(record))
    assert curation.gate_blocks(unit, "render", work_dir) is None
    assert curation.read(unit, work_dir)["gates"]["turnaround"] == "pending"


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


def test_the_views_ring_has_all_eight_slots(fresh, work_dir):
    """Every view of `config.CONCEPT_VIEWS` is a slot, with the three states
    the ring draws - rendered and accepted, rendered and pending, and missing
    entirely."""
    views = curation.views_of(fresh, curation.read(fresh, work_dir), work_dir)
    assert list(views) == list(config.CONCEPT_VIEWS)
    assert all(row["missing"] for row in views.values()), "nothing is on the base yet"
    assert {v: row["attempts"] for v, row in views.items()} == {
        "back": 1, "left": 1, "top": 1, "right": 0, "back_right": 0, "back_left": 0,
        "front_left": 0, "front_right": 0}, \
        "an attempt nobody picked is a different gap from a view nobody drew"
    assert views["right"] == {**views["right"], "missing": True, "attempts": 0, "file": None,
                              "url": None, "state": "pending", "wanted_by": ["s0"]}
    assert views["top"]["elevation"] == "top" and views["top"]["wanted_by"] == []
    _pick_view(work_dir, fresh, "back")
    curation.set_views(fresh, {"back": "accepted"}, work_dir)
    views = curation.views_of(fresh, curation.read(fresh, work_dir), work_dir)
    assert views["back"] == {**views["back"], "missing": False, "state": "accepted",
                             "file": "view_back.png", "wanted_by": ["s90"]}
    assert views["back"]["url"].startswith("/jobs/archer_base/frames/render/")
    assert views["left"]["state"] == "pending" and views["left"]["missing"] is True
    assert set(curation.base_views(fresh, work_dir)) == {"back"}


def test_part_palettes_are_reported_per_unit(unit, work_dir):
    """`config.UNIT_PARTS` DEFINES a unit's regions; `<unit>_base/post/parts.json`
    MEASURES them - what the Views tab's "measure part palettes" makes."""
    assert curation.base_parts_job("knight") == "knight_base"
    assert curation.base_part_palettes("knight", work_dir) is False
    measured = work_dir / "knight_base" / "post"
    measured.mkdir(parents=True)
    (measured / "parts.json").write_text('{"schema": "anim.parts/1", "parts": []}')
    assert curation.base_part_palettes("knight", work_dir) is True


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def test_no_record_means_nothing_is_gated(unit, work_dir):
    for stage in ("clip", "render", "bakeoff", "run"):
        assert curation.gate_blocks(unit, stage, work_dir) is None


def test_a_gpu_stage_waits_for_the_base_and_a_cpu_stage_never_does(unit, work_dir):
    _gate(unit, "pending", work_dir)
    reason = curation.gate_blocks(unit, "clip", work_dir)
    assert reason and "base gate is pending" in reason and "concept --unit knight --pick" in reason
    assert curation.gate_blocks(unit, "pick", work_dir) is None, "a CPU stage never waits"
    _gate(unit, "accepted", work_dir)
    assert curation.gate_blocks(unit, "clip", work_dir) is None
    assert "matrix" not in curation.BASE_GATED


def test_the_gate_is_read_off_the_stages_own_arguments(unit, work_dir):
    _gate(unit, "pending", work_dir)
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
    _gate(unit, "pending", work_dir)
    assert cli.main(["render", "knight_strike"]) == 2
    out = capsys.readouterr().out
    assert "anim render: refused" in out and "base gate is pending" in out and "--force" in out
    assert not Job.open("knight_strike", work_dir).render_dir.exists()


def test_concept_pick_accepts_the_base_gate(work_dir, workflow_dir):
    from test_concept import _concept_job
    from anim.stages import concept as concept_stage

    _concept_job(work_dir, workflow_dir)
    _gate("knight", "pending", work_dir)
    result = concept_stage.pick("knight", "01", work_dir=work_dir)
    assert result["gates"] == {"base": "accepted"}
    record = json.loads((work_dir / "knight_curation.json").read_text())
    assert record["gates"] == {"base": "accepted"}
    assert record["gate_log"][-1]["by"] == "concept --pick"


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------
def test_post_jobs_is_423_while_the_gate_is_pending(api, unit, work_dir):
    _gate(unit, "pending", work_dir)
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
    assert api.post("/jobs", json={"stage": "nope", "args": {}}).status_code == 400, \
        "an unknown stage is still the manager's 400"


def test_the_board_routes_and_the_curation_mode_are_gone(api, unit):
    assert api.get("/curation").status_code == 404
    assert api.get("/curation/knight").status_code == 404
    assert api.get("/matrix/knight").status_code == 404
    assert "curation" not in api.get("/settings").json()
    assert api.put("/settings", json={"curation": "end"}).status_code == 400
    assert api.post("/jobs", json={"stage": "matrix", "args": {}}).status_code == 400
