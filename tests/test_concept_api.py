"""The concept generation feature's API: `GET /concept/{unit}`, the generate
op -> `anim concept` argv, the base-gate 423 for a view op, upload, picks,
defaults, and `POST /runs/{id}/cancel`. Runs go through the two-line fake
launcher (tests/test_api.py) because the test is about the argv and the
manager, not the GPU; the state is built by the real stage against
FakeComfy."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fixtures.fakecomfy import FakeComfy, canned_png
from test_api import FAKE_STAGE, wait_for

from anim import comfy, config, curation
from anim.api import runs as runs_api
from anim.api.app import create_app
from anim.job import Job
from anim.stages import concept as concept_stage

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "gpu" / "workflows"


@pytest.fixture
def api(work_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.2")
    manager = runs_api.RunManager(work_dir, cpu_slots=1,
                                  python=[sys.executable, "-c", FAKE_STAGE])
    app = create_app(work_dir, web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", runs=manager)
    with TestClient(app) as client:
        yield client


def _base(work_dir, *, pick: bool = True) -> None:
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.concept("knight", description="a knight", count=2, seed=10,
                              client=comfy.ComfyClient(fake.url), workflow_dir=WORKFLOWS,
                              work_dir=work_dir)
    if pick:
        concept_stage.pick("knight", "01", parts="closed helm, kite shield", work_dir=work_dir)


def _views(work_dir) -> None:
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.render_views("knight", views="back,left", count=1, seed=1,
                                   client=comfy.ComfyClient(fake.url), workflow_dir=WORKFLOWS,
                                   work_dir=work_dir, log=lambda _: None)


def test_get_concept_is_empty_then_full(api, work_dir):
    assert api.get("/concept/../x").status_code == 404
    empty = api.get("/concept/knight").json()
    assert empty["exists"] is False and empty["base"]["candidates"] == []
    assert set(empty["views"]) == set(config.CONCEPT_VIEWS)
    assert empty["views"]["top"]["group"] == "main" and empty["views"]["right"]["group"] == "advanced"
    assert "no base" in empty["locked"]
    _base(work_dir)
    _views(work_dir)
    concept_stage.pick_view("knight", {"back": 0}, work_dir=work_dir)
    row = api.get("/concept/knight").json()
    assert row["exists"] is True and row["locked"] is None and row["gates"]["base"] == "accepted"
    base = row["base"]
    assert base["description"] == "a knight" and base["pick"] == 1 and base["accepted"] is True
    assert [c["url"] for c in base["candidates"]] == [
        "/jobs/knight_concept/frames/candidates/0.png", "/jobs/knight_concept/frames/candidates/1.png"]
    assert base["candidates"][0]["rendered"] is True and base["candidates"][0]["source"] == "txt2img"
    assert base["front_url"].startswith("/jobs/knight_base/frames/render/")
    assert api.get(base["front_url"]).status_code == 200
    assert api.get(base["candidates"][1]["url"]).status_code == 200
    back = row["views"]["back"]
    assert back["pick"] == 0 and back["state"] == "accepted" and back["accepted_file"] == "view_back.png"
    assert back["candidates"][0]["url"] == "/jobs/knight_concept/frames/views/0.png"
    assert back["prompt"] == config.concept_view_prompt("back", "closed helm, kite shield")
    left = row["views"]["left"]
    assert left["pick"] is None and left["state"] == "pending"
    assert left["candidates"][0]["url"] == "/jobs/knight_concept/frames/views/1.png"
    assert row["views"]["top"]["candidates"] == [] and row["views"]["top"]["default_prompt"]
    assert row["parts_phrase"] == "closed helm, kite shield" and row["last_seed"] == 10
    assert row["defaults"] == {} and row["page"].endswith("/_views/knight/index.html")


def _argv(api, response) -> list[str]:
    """The run's argv, once it has finished (the GPU slot is one)."""
    assert response.status_code == 202, response.text
    return wait_for(api, response.json()["id"])["argv"]


def _accepted(api, response) -> None:
    assert response.status_code == 202, response.text
    wait_for(api, response.json()["id"])


def test_generate_maps_each_op_onto_the_stages_flags(api, work_dir):
    _base(work_dir)
    lora = config.CONCEPT_LORA["file"]
    argv = _argv(api, api.post("/concept/knight/generate", json={
        "op": "txt2img", "prompt": " a knight ", "count": 3, "seed": 5,
        "model": "z.safetensors", "loras": [{"file": lora, "strength": 0.3}, "b.safetensors"],
        "fresh": True}))
    assert argv == ["--unit", "knight", "--prompt", "a knight", "--count", "3", "--seed", "5",
                    "--fresh", "--model", "z.safetensors", "--lora", f"{lora}:0.3",
                    "--lora", "b.safetensors"]
    argv = _argv(api, api.post("/concept/knight/generate", json={
        "op": "views", "views": ["left"], "view_prompt": "from the left", "count": 1}))
    assert argv == ["--unit", "knight", "--count", "1", "--views", "left",
                    "--view-prompt", "from the left"]
    argv = _argv(api, api.post("/concept/knight/generate", json={"op": "views"}))
    assert argv == ["--unit", "knight", "--views", ""], "no list = the core views"
    argv = _argv(api, api.post("/concept/knight/generate", json={
        "op": "img2img", "ref": "base:01", "strength": 0.5, "seed": 9, "lora_strength": 0}))
    assert argv == ["--unit", "knight", "--seed", "9", "--lora-strength", "0.0",
                    "--img2img", "base:01", "--strength", "0.5"]
    argv = _argv(api, api.post("/concept/knight/generate", json={
        "op": "edit", "ref": "base:01", "instruction": "make the helmet red"}))
    assert argv == ["--unit", "knight", "--edit", "base:01", "--instruction", "make the helmet red"]
    # The refusals before a run.
    assert api.post("/concept/knight/generate", json={"op": "txt2img"}).status_code == 400
    assert api.post("/concept/knight/generate", json={"op": "edit", "ref": "base:01"}).status_code == 400
    assert api.post("/concept/knight/generate", json={"op": "img2img"}).status_code == 400
    assert api.post("/concept/knight/generate", json={"op": "paint", "prompt": "x"}).status_code == 400
    assert api.post("/concept/knight/generate", json={"op": "txt2img", "prompt": "x", "count": 5}).status_code == 422
    assert api.post("/concept/../x/generate", json={"op": "txt2img", "prompt": "x"}).status_code == 404


def test_a_view_op_is_423_until_the_base_is_picked(api, work_dir):
    r = api.post("/concept/knight/generate", json={"op": "views"})
    assert r.status_code == 423 and r.json()["gate"] is True and "no base" in r.json()["error"]
    _base(work_dir, pick=False)
    assert api.get("/concept/knight").json()["locked"]
    r = api.post("/concept/knight/pick", json={"index": 0, "parts": "helm"})
    assert r.status_code == 200 and r.json()["base"]["pick"] == 0 and r.json()["locked"] is None
    assert r.json()["gates"]["base"] == "accepted" and r.json()["parts_phrase"] == "helm"
    _accepted(api, api.post("/concept/knight/generate", json={"op": "views"}))
    # The base's own ops never wait; a view's edit does once the gate reopens.
    api.put("/curation/knight/gates/base", json={"state": "pending"})
    _accepted(api, api.post("/concept/knight/generate", json={"op": "txt2img", "prompt": "x"}))
    _accepted(api, api.post("/concept/knight/generate", json={"op": "img2img", "ref": "base:00"}))
    assert api.post("/concept/knight/generate",
                    json={"op": "edit", "ref": "left:00", "instruction": "x"}).status_code == 423
    _accepted(api, api.post("/concept/knight/generate",
                            json={"op": "edit", "ref": "left:00", "instruction": "x", "force": True}))
    assert api.post("/concept/knight/pick", json={"index": 7}).status_code == 400
    assert api.post("/concept/orc/pick", json={"index": 0}).status_code == 404


def test_upload_is_the_png_body(api, work_dir, reference_sprite):
    r = api.put("/concept/knight/upload", content=reference_sprite.read_bytes(),
                headers={"Content-Type": "image/png", "X-Filename": "../knight_mac.png"})
    assert r.status_code == 200, r.text
    row = r.json()["uploaded"]
    assert row["source"] == "upload" and row["name"] == "knight_mac.png" and row["index"] == 0
    assert r.json()["base"]["candidates"][0]["url"] == "/jobs/knight_concept/frames/candidates/0.png"
    assert api.get(r.json()["base"]["candidates"][0]["url"]).status_code == 200
    assert api.put("/concept/knight/upload", content=b"not a png").status_code == 415
    assert api.put("/concept/knight/upload", content=b"\x89PNG\r\n\x1a\ntrash").status_code == 415
    assert api.put("/concept/../x/upload", content=b"").status_code == 404


def test_view_pick_and_defaults(api, work_dir):
    _base(work_dir)
    _views(work_dir)
    r = api.post("/concept/knight/views/left/pick", json={"index": 0})
    assert r.status_code == 200 and r.json()["views"]["left"]["pick"] == 0
    assert r.json()["views"]["left"]["state"] == "accepted"
    assert (Job.open("knight_base", work_dir).render_dir / "view_left.png").exists()
    assert api.post("/concept/knight/views/left/pick", json={"index": 4}).status_code == 400
    assert api.post("/concept/knight/views/top/pick", json={"index": 0}).status_code == 400
    assert api.post("/concept/knight/views/under/pick", json={"index": 0}).status_code == 404
    lora = config.CONCEPT_LORA["file"]
    r = api.put("/concept/knight/defaults", json={"family": "zimage", "loras": [f"{lora}:0.5"]})
    assert r.status_code == 200, r.text
    assert r.json()["defaults"] == {"zimage": {"model": None, "loras": [{"file": lora, "strength": 0.5}]}}
    assert Job.open("knight_concept", work_dir).read_manifest()["defaults"] == r.json()["defaults"]
    assert api.put("/concept/knight/defaults", json={"loras": ["ghost.safetensors"]}).status_code == 400
    assert api.put("/concept/knight/defaults", json={"family": "sdxl"}).status_code == 400
    r = api.put("/concept/knight/defaults", json={"family": "qwen-edit",
                                                  "model": config.CONCEPT_MODELS["qwen-edit"]})
    assert r.json()["defaults"]["qwen-edit"] == {"model": config.CONCEPT_MODELS["qwen-edit"], "loras": []}


def test_the_view_pick_route_also_mirrors(api, work_dir):
    """The mirrored view feature rides on the route the Concept tab already
    posts to: `mirror: true` in place of an index. One accepted view comes out
    either way, so there is one endpoint."""
    _base(work_dir)
    _views(work_dir)
    # No index and no mirror is the one thing the body can't mean.
    assert api.post("/concept/knight/views/left/pick", json={}).status_code == 400
    # The partner is drawn but nobody has accepted it: 400, saying what to do.
    r = api.post("/concept/knight/views/right/pick", json={"mirror": True})
    assert r.status_code == 400 and "not on the base" in r.json()["detail"]
    assert api.post("/concept/knight/views/left/pick", json={"index": 0}).status_code == 200
    r = api.post("/concept/knight/views/right/pick", json={"mirror": True})
    assert r.status_code == 200, r.text
    row = r.json()["views"]["right"]
    assert row["state"] == "accepted" and row["mirrored"] is True
    assert row["mirrored_from"] == "left" and row["mirror_of"] == "left"
    assert row["accepted_file"] == "view_right.png" and row["pick"] is None
    assert (Job.open("knight_base", work_dir).render_dir / "view_right.png").exists()
    # The handedness line comes back with the state so the tab can show it.
    assert "handedness" in r.json()["warning"] and "shield" in r.json()["warning"]
    # A view on the symmetry plane has no partner: 400, not a silent no-op.
    r = api.post("/concept/knight/views/back/pick", json={"mirror": True})
    assert r.status_code == 400 and "no mirror partner" in r.json()["detail"]
    assert api.get("/concept/knight").json()["views"]["back"]["mirror_of"] is None
    assert api.get("/concept/knight").json()["views"]["top"]["mirror_of"] is None
    # The board's ring carries the same two facts.
    ring = api.get("/curation/knight").json()["views"]
    assert ring["right"]["mirrored"] is True and ring["right"]["mirrored_from"] == "left"
    assert ring["left"]["mirrored"] is False and ring["left"]["mirror_of"] == "right"


def test_cancel_stops_the_run_and_frees_the_gpu(api, work_dir, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "5")
    started = api.post("/jobs", json={"stage": "concept", "args": {"slug": "knight", "prompt": "x"}})
    assert started.status_code == 202
    run_id = started.json()["id"]
    deadline = time.monotonic() + 10
    while api.get(f"/runs/{run_id}").json()["status"] != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    busy = api.post("/jobs", json={"stage": "concept", "args": {"slug": "knight", "prompt": "y"}})
    assert busy.status_code == 409
    t0 = time.monotonic()
    r = api.post(f"/runs/{run_id}/cancel")
    assert r.status_code == 202, r.text
    row = wait_for(api, run_id, timeout=10)
    assert row["status"] == "cancelled" and row["cancelled"] is True
    assert time.monotonic() - t0 < 4, "SIGTERM, not the 5 s sleep"
    assert "cancelled from the API" in row["log"]
    assert api.get("/health").json()["gpu_run"] is None
    assert json.loads((work_dir / "_runs" / f"{run_id}.json").read_text())["status"] == "cancelled"
    events = api.get(f"/runs/{run_id}/events").text
    assert '"status": "cancelled"' in events
    assert api.post(f"/runs/{run_id}/cancel").status_code == 409, "nothing left to cancel"
    assert api.post("/runs/nope/cancel").status_code == 404
    monkeypatch.setenv("FAKE_DELAY", "0.1")
    again = api.post("/jobs", json={"stage": "concept", "args": {"slug": "knight", "prompt": "y"}})
    assert again.status_code == 202, "the GPU slot is free"
    wait_for(api, again.json()["id"])


def test_a_cancelled_gpu_run_interrupts_comfyui(work_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "5")
    with FakeComfy() as fake:
        manager = runs_api.RunManager(work_dir, cpu_slots=1, comfyui_url=fake.url,
                                      python=[sys.executable, "-c", FAKE_STAGE])
        run = manager.start("concept", args={"slug": "knight", "prompt": "x"})
        deadline = time.monotonic() + 10
        while run.status != "running" and time.monotonic() < deadline:
            time.sleep(0.05)
        manager.cancel(run.id)
        deadline = time.monotonic() + 10
        while run.active and time.monotonic() < deadline:
            time.sleep(0.05)
        assert run.status == "cancelled" and fake.interrupts == 1


def test_hide_and_delete_an_attempt_through_the_api(api, work_dir):
    _base(work_dir)          # two base attempts, 01 accepted
    _views(work_dir)
    row = api.get("/concept/knight").json()
    assert row["base"]["hidden_count"] == 0 and row["views"]["back"]["hidden_count"] == 0
    assert row["base"]["candidates"][0]["hidden"] is False
    assert row["base"]["candidates"][0]["file_deleted"] is None
    # Hide one, and the state says so; the file is still served.
    state = api.put("/concept/knight/attempts/base/0", json={"hidden": True}).json()
    assert state["base"]["candidates"][0]["hidden"] is True
    assert state["base"]["hidden_count"] == 1
    assert api.get(state["base"]["candidates"][0]["url"]).status_code == 200
    # Deleting needs it hidden; then the file goes and the row stays.
    assert api.delete("/concept/knight/attempts/base/1").status_code == 400, "accepted"
    deleted = api.delete("/concept/knight/attempts/base/0")
    assert deleted.status_code == 200, deleted.text
    kept = deleted.json()["base"]["candidates"][0]
    assert kept["file_deleted"] and kept["url"] is None and kept["hidden"] is True
    assert kept["seed"] == 10 and kept["prompt"], "its record survives its file"
    assert not (Job.open("knight_concept", work_dir).candidates_dir / "00.png").exists()
    # Restoring is the same route with hidden=false.
    shown = api.put("/concept/knight/attempts/base/0", json={"hidden": False}).json()
    assert shown["base"]["candidates"][0]["hidden"] is False
    # The accepted attempt is refused, with the reason.
    refused = api.put("/concept/knight/attempts/base/1", json={"hidden": True})
    assert refused.status_code == 400 and "accepted" in refused.json()["detail"]
    # A view attempt, and the 404s.
    view = api.put("/concept/knight/attempts/back/0", json={"hidden": True}).json()
    assert view["views"]["back"]["candidates"][0]["hidden"] is True
    assert view["views"]["back"]["hidden_count"] == 1
    assert api.put("/concept/knight/attempts/under/0", json={"hidden": True}).status_code == 404
    assert api.put("/concept/../x/attempts/base/0", json={"hidden": True}).status_code == 404
    assert api.put("/concept/knight/attempts/base/9", json={"hidden": True}).status_code == 400
    assert api.delete("/concept/knight/attempts/base/9").status_code == 400


def _canvas(size, erase=(20, 20, 80, 80)) -> bytes:
    from io import BytesIO

    from PIL import Image, ImageDraw

    img = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], fill=(40, 60, 90, 255))
    if erase:
        draw.rectangle(list(erase), fill=(0, 0, 0, 0))
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_the_editors_save_writes_over_the_attempt(api, work_dir):
    _base(work_dir)
    body = _canvas(config.CONCEPT_CELL)
    r = api.put("/concept/knight/attempts/base/0/image", content=body,
                headers={"Content-Type": "image/png"})
    assert r.status_code == 200, r.text
    row = r.json()["edited"]
    assert row["edited"]["how"] == "hand" and row["index"] == 0
    assert r.json()["base"]["candidates"][0]["edited"]["how"] == "hand"
    assert api.put("/concept/knight/attempts/base/0/image", content=b"nope").status_code == 415
    assert api.put("/concept/knight/attempts/base/9/image", content=body).status_code == 400
    assert api.put("/concept/knight/attempts/under/0/image", content=body).status_code == 404
    # The accepted one is editable - it is the copy under _base/render that waits.
    assert api.put("/concept/knight/attempts/base/1/image", content=body).status_code == 200
    state = api.get("/concept/knight").json()
    assert state["base"]["pick_stale"] is True, "edited after it was accepted"
    assert state["base"]["pick"] == 1
    # Accepting it again clears the flag.
    again = api.post("/concept/knight/pick", json={"index": 1}).json()
    assert again["base"]["pick_stale"] is False


def test_inpaint_stashes_the_canvas_and_runs_the_stage(api, work_dir):
    _base(work_dir)
    body = _canvas(config.CONCEPT_CELL)
    r = api.post("/concept/knight/attempts/base/0/inpaint?prompt=a+closed+helm&strength=0.9",
                 content=body, headers={"Content-Type": "image/png"})
    assert r.status_code == 202, r.text
    argv = wait_for(api, r.json()["id"])["argv"]
    assert argv[:4] == ["--unit", "knight", "--prompt", "a closed helm"]
    assert "--inpaint" in argv and "base:00" in argv and "--strength" in argv
    mask = Path(argv[argv.index("--mask") + 1])
    assert mask.exists() and mask.parent == Job.open("knight_concept", work_dir).root / "masks"
    assert mask.read_bytes() == body, "the canvas as the editor left it"
    # A canvas with nothing erased is refused before a run is started.
    flat = _canvas(config.CONCEPT_CELL, erase=None)
    refused = api.post("/concept/knight/attempts/base/0/inpaint", content=flat)
    assert refused.status_code == 400 and "erased" in refused.json()["detail"]
    assert api.post("/concept/knight/attempts/base/0/inpaint", content=b"nope").status_code == 415


def test_generate_in_place_carries_the_flag(api, work_dir):
    _base(work_dir)
    r = api.post("/concept/knight/generate",
                 json={"op": "img2img", "ref": "base:00", "strength": 0.5, "in_place": True,
                       "count": 3})
    assert r.status_code == 202
    argv = wait_for(api, r.json()["id"])["argv"]
    assert "--in-place" in argv and "--count" not in argv, "in place is one image"
    r = api.post("/concept/knight/generate",
                 json={"op": "edit", "ref": "base:00", "instruction": "red helm", "in_place": True})
    assert "--in-place" in wait_for(api, r.json()["id"])["argv"]


def _write_view_seconds(work_dir, unit: str, seconds: list[float]) -> None:
    """Stamp recorded render times onto a unit's views record, the way a real
    run does. FakeComfy answers instantly, so the fixtures carry ~0 s."""
    path = concept_stage.views_path(unit, work_dir)
    record = json.loads(path.read_text())
    rows = list(record["views"].values())
    for row in rows:
        for candidate in row.get("candidates") or []:
            if seconds:
                candidate["seconds"] = seconds.pop(0)
    path.write_text(json.dumps(record))


def test_the_state_carries_a_measured_estimate_for_the_sweep(api, work_dir):
    """The advanced views sweep feature: the Concept tab quotes what a sweep
    costs before the press, off this unit's own recorded renders - the warm
    median, with a cold first render dropped rather than trebling it."""
    _base(work_dir)
    _views(work_dir)
    _write_view_seconds(work_dir, "knight", [201.4, 12.0])
    state = api.get("/concept/knight").json()
    # Two renders recorded, one of them the cold one: the warm one is the median.
    assert state["estimate_seconds"] == 12.0

    # A unit with no records of its own borrows every unit's, so a brand-new
    # unit still gets a real number instead of the bare constant.
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.concept("squire", description="a squire", count=1, seed=3,
                              client=comfy.ComfyClient(fake.url), workflow_dir=WORKFLOWS,
                              work_dir=work_dir)
    assert api.get("/concept/squire").json()["estimate_seconds"] == 12.0


def test_the_estimate_falls_back_to_the_measured_constant(api, work_dir):
    """Nothing recorded anywhere - a fresh work dir - still quotes a number."""
    assert api.get("/concept/knight").json()["estimate_seconds"] == config.CONCEPT_VIEW_SECONDS
    _base(work_dir)
    assert api.get("/concept/knight").json()["estimate_seconds"] == config.CONCEPT_VIEW_SECONDS


def test_the_advanced_group_is_the_five_the_sweep_sends(api, work_dir):
    """What the tab filters the sweep out of: the group tag on each view, and
    the same five names `--views advanced` resolves to."""
    _base(work_dir)
    state = api.get("/concept/knight").json()
    advanced = [name for name, view in state["views"].items() if view["group"] == "advanced"]
    assert advanced == list(config.CONCEPT_ADVANCED_VIEWS)
    assert advanced == config.concept_view_names("advanced")
    assert [n for n, v in state["views"].items() if v["group"] == "main"] \
        == list(config.CONCEPT_MAIN_VIEWS)
