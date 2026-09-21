"""The pipeline API feature: FastAPI over a throwaway work/, through TestClient.

No socket, no browser, no GPU. A run is the real CLI in a subprocess where the
test is about the CLI (`pick` on the posed job), and a two-line fake launcher
where the test is about the manager (409s, SSE) - a real stage there would
only add seconds.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import textwrap
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anim import config, imaging, palettekit
from anim.api import jobs as jobs_api
from anim.api import runs as runs_api
from anim.api import settings as settings_api
from anim.api.app import create_app
from anim.job import Job
from anim.stages import pick as pick_stage
from anim.stages.review import ReviewApp
from test_palette import ENDESGA_32

# What `[*python, stage, *argv]` runs when the test is about the manager and
# not about a stage: prints two steps and exits, `FAKE_DELAY` seconds apart.
FAKE_STAGE = textwrap.dedent("""
    import os, sys, time
    print("[1/2] started " + " ".join(sys.argv[1:]), flush=True)
    time.sleep(float(os.environ.get("FAKE_DELAY", "0.2")))
    print("[2/2] finished", flush=True)
    sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
""")


@pytest.fixture
def api(posed_job, work_dir, tmp_path):
    """The app over the posed job's work/, no editor build, ComfyUI unreachable."""
    (work_dir / "sheets").mkdir()          # a non-job directory, like the real one
    (work_dir / "gate_parts.json").write_text("{}")
    app = create_app(work_dir, web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", cpu_slots=1)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def fake_api(posed_job, work_dir, tmp_path, monkeypatch):
    """Same app, but a run is the fake launcher above."""
    monkeypatch.setenv("FAKE_DELAY", "0.2")
    manager = runs_api.RunManager(work_dir, cpu_slots=1,
                                  python=[sys.executable, "-c", FAKE_STAGE])
    app = create_app(work_dir, web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", runs=manager)
    with TestClient(app) as client:
        yield client


def wait_for(client: TestClient, run_id: str, *, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = client.get(f"/runs/{run_id}").json()
        if row["status"] not in runs_api.ACTIVE:
            return row
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} still {row['status']} after {timeout}s")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def test_jobs_lists_the_fixture_job_and_nothing_else(api, posed_job):
    rows = api.get("/jobs").json()
    assert [r["name"] for r in rows] == ["knight_strike"], \
        "sheets/ and gate_parts.json are not jobs"
    row = rows[0]
    assert row["slug"] == "knight" and row["action"] == "strike"
    assert row["direction"] == "s0", "a filmed clip is the side view"
    assert row["stages"] == ["extract", "pose"]
    assert row["updated"].endswith("+00:00")
    assert row["url"] == "/jobs/knight_strike"


def test_get_job_returns_the_manifest_and_what_the_api_derives(api, posed_job):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    row = api.get("/jobs/knight_strike").json()
    assert row["manifest"] == json.loads(posed_job.manifest_path.read_text())
    assert row["stages"] == ["extract", "pose", "pick"]
    assert row["picks"] == json.loads(posed_job.picks_path.read_text())["picks"]
    assert row["frame_sets"] == {"frames": 13, "pose": 13}


def test_a_rig_job_reports_its_direction(api, posed_job):
    posed_job.update_manifest(motion_source={"kind": "rig", "direction": "t135"})
    row = api.get("/jobs/knight_strike").json()
    assert row["direction"] == "t135"
    assert "rig" in row["stages"]


@pytest.mark.parametrize("name", ["nope", "_runs", ".hidden", "sheets", "a%2Fb"])
def test_an_unknown_or_unsafe_job_name_is_404(api, name):
    assert api.get(f"/jobs/{name}").status_code == 404
    assert api.get(f"/jobs/{name}/picks").status_code == 404


def test_frames_are_listed_by_set_and_served_by_index(api, posed_job):
    listing = api.get("/jobs/knight_strike/frames").json()
    assert set(listing["stages"]) == {"frames", "pose"}
    first = listing["stages"]["frames"][0]
    assert first == {"index": 0, "file": "0001.png",
                     "url": "/jobs/knight_strike/frames/frames/0.png"}
    png = api.get(first["url"])
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content.startswith(b"\x89PNG")
    skel = api.get(listing["stages"]["pose"][12]["url"])
    assert skel.content.startswith(b"\x89PNG")
    assert api.get("/jobs/knight_strike/frames/frames/13.png").status_code == 404
    assert api.get("/jobs/knight_strike/frames/secrets/0.png").status_code == 404
    assert api.get("/jobs/knight_strike/frames/frames/x.png").status_code == 404


def test_the_post_set_is_the_numbered_frames_in_numeric_order(api, posed_job):
    """post/ also holds palette.png and, after --parts, _hd / _labels
    diagnostics; and a name sort puts _10 before _2. The set is what
    publish ships, in the order pick k -> frame k."""
    posed_job.post_dir.mkdir(parents=True)
    for name in ["knight_strike_0.png", "knight_strike_1.png", "knight_strike_2.png",
                 "knight_strike_10.png", "knight_strike_11.png", "knight_strike_3.png",
                 "knight_strike_0_hd.png", "knight_strike_0_labels.png", "palette.png"]:
        Image.new("RGBA", (64, 64)).save(posed_job.post_dir / name)
    listing = api.get("/jobs/knight_strike/frames").json()["stages"]["post"]
    assert [e["file"] for e in listing] == [
        "knight_strike_0.png", "knight_strike_1.png", "knight_strike_2.png",
        "knight_strike_3.png", "knight_strike_10.png", "knight_strike_11.png"]
    assert [e["index"] for e in listing] == list(range(6))
    assert api.get("/jobs/knight_strike").json()["frame_sets"]["post"] == 6
    assert jobs_api.frame_path(posed_job, "post", 4).name == "knight_strike_10.png"


# ---------------------------------------------------------------------------
# PUT a frame - the web pixel editor's save
# ---------------------------------------------------------------------------
def png_bytes(size, colour=(200, 40, 40, 255), *, soft: bool = False) -> bytes:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    px = img.load()
    for x in range(size[0] // 2):
        for y in range(size[1]):
            px[x, y] = colour
    if soft:
        px[0, 0] = (*colour[:3], 90)       # below the threshold: cleared
        px[1, 0] = (*colour[:3], 200)      # above: made opaque
        px[size[0] - 1, size[1] - 1] = (9, 9, 9, 0)   # colour under alpha 0: zeroed
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def post_job(posed_job):
    """The posed job with two 64x64 post frames, as `post` leaves them."""
    posed_job.post_dir.mkdir(parents=True)
    for i in range(2):
        (posed_job.post_dir / f"knight_strike_{i}.png").write_bytes(png_bytes((64, 64)))
    Image.new("RGB", (4, 1)).save(posed_job.post_dir / "palette.png")
    return posed_job


def test_put_frame_overwrites_the_png_in_place_with_binary_alpha(api, post_job):
    target = post_job.post_dir / "knight_strike_1.png"
    inode = target.stat().st_ino
    response = api.put("/jobs/knight_strike/frames/post/1.png",
                       content=png_bytes((64, 64), (10, 200, 10, 255), soft=True),
                       headers={"Content-Type": "image/png"})
    assert response.status_code == 200, response.text
    row = response.json()
    assert row["file"] == "knight_strike_1.png" and row["size"] == [64, 64]
    assert row["alpha_rounded"] == 2, "the two soft pixels, not the alpha-0 one"
    assert row["url"] == "/jobs/knight_strike/frames/post/1.png"
    with Image.open(target) as saved:
        saved = saved.convert("RGBA")
        assert saved.size == (64, 64)
        histogram = saved.getchannel("A").histogram()
        assert all(count == 0 for value, count in enumerate(histogram) if value not in (0, 255))
        assert saved.getpixel((0, 0)) == (0, 0, 0, 0), "soft below 128 -> transparent, zeroed"
        assert saved.getpixel((1, 0)) == (10, 200, 10, 255), "soft above 128 -> opaque"
        assert saved.getpixel((63, 63)) == (0, 0, 0, 0), "colour under alpha 0 is dropped"
        assert saved.getpixel((5, 5)) == (10, 200, 10, 255), "the new colour landed"
    assert target.stat().st_ino == inode, "written in place (the Docker Desktop inode rule)"
    assert api.get("/jobs/knight_strike/frames/post/1.png").content == target.read_bytes()
    with Image.open(post_job.post_dir / "knight_strike_0.png") as other:
        assert other.getpixel((5, 5)) == (200, 40, 40, 255), "the neighbour is untouched"


def test_put_frame_refuses_a_different_size_with_409(api, post_job):
    before = (post_job.post_dir / "knight_strike_0.png").read_bytes()
    response = api.put("/jobs/knight_strike/frames/post/0.png",
                       content=png_bytes((32, 32)), headers={"Content-Type": "image/png"})
    assert response.status_code == 409
    assert "64x64" in response.json()["detail"] and "32x32" in response.json()["detail"]
    assert (post_job.post_dir / "knight_strike_0.png").read_bytes() == before, "nothing written"


def test_put_frame_refuses_what_is_not_a_png_or_not_a_frame(api, post_job):
    assert api.put("/jobs/knight_strike/frames/post/0.png", content=b"not a png",
                   headers={"Content-Type": "image/png"}).status_code == 415
    assert api.put("/jobs/knight_strike/frames/post/0.png",
                   content=b"\x89PNG\r\n\x1a\n" + b"garbage",
                   headers={"Content-Type": "image/png"}).status_code == 415
    ok = png_bytes((64, 64))
    assert api.put("/jobs/knight_strike/frames/post/2.png", content=ok).status_code == 404, \
        "palette.png is not frame 2 and the route never creates a file"
    assert api.put("/jobs/knight_strike/frames/secrets/0.png", content=ok).status_code == 404
    assert api.put("/jobs/nope/frames/post/0.png", content=ok).status_code == 404
    assert not (post_job.post_dir / "knight_strike_2.png").exists()
    assert sorted(p.name for p in post_job.post_dir.iterdir()) == \
        ["knight_strike_0.png", "knight_strike_1.png", "palette.png"]


def test_put_frame_works_on_the_source_set_too(api, posed_job):
    """`frames` (the extracted 360x420 source) is a PNG set like any other."""
    target = posed_job.frames()[3]
    with Image.open(target) as img:
        size = img.size
    response = api.put("/jobs/knight_strike/frames/frames/3.png",
                       content=png_bytes(size, (1, 2, 3, 255)))
    assert response.status_code == 200, response.text
    with Image.open(target) as saved:
        assert saved.convert("RGBA").getpixel((0, 0)) == (1, 2, 3, 255)


# ---------------------------------------------------------------------------
# Picks - the same ReviewApp, byte for byte
# ---------------------------------------------------------------------------
def test_picks_read_what_review_reads(api, posed_job):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    assert api.get("/jobs/knight_strike/picks").json() == ReviewApp(posed_job).state()


def test_put_picks_writes_picks_json_exactly_as_review_does(api, posed_job, tmp_path):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    # The same starting file, driven by ReviewApp directly, is the oracle.
    twin_root = tmp_path / "twin"
    shutil.copytree(posed_job.root, twin_root / posed_job.name)
    twin = Job.open(posed_job.name, twin_root)
    expected_state = ReviewApp(twin).set_picks([2, 0, 2, 11])
    expected_bytes = twin.picks_path.read_bytes()

    response = api.put("/jobs/knight_strike/picks", json={"picks": [2, 0, 2, 11]})
    assert response.status_code == 200
    assert response.json() == expected_state
    assert response.json()["picks"] == [0, 2, 11], "sorted and de-duplicated"
    assert posed_job.picks_path.read_bytes() == expected_bytes


def test_put_picks_drops_out_of_range_indices_like_review(api, posed_job):
    assert api.put("/jobs/knight_strike/picks",
                   json={"picks": [0, 99, -3]}).json()["picks"] == [0]


def test_regenerate_only_flags_the_job(api, posed_job):
    state = api.post("/jobs/knight_strike/regenerate").json()
    assert state["rerender_requested"] is True
    assert json.loads(posed_job.manifest_path.read_text())["rerender_requested"]
    assert not posed_job.render_dir.exists(), "it must NEVER reach the GPU"


def test_picks_on_a_job_without_frames_is_a_409_that_says_why(api, posed_job):
    for frame in posed_job.frames():
        frame.unlink()
    response = api.get("/jobs/knight_strike/picks")
    assert response.status_code == 409
    assert "anim extract" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Runs - the CLI in a subprocess
# ---------------------------------------------------------------------------
def test_argv_is_built_from_the_stages_own_parser():
    assert runs_api.argv_for("pick", {"job": "x", "n": 3, "force": True}) == \
        ["-n", "3", "--force", "x"]
    assert runs_api.argv_for("pick", {"job": "x", "force": False}) == ["x"]
    argv = runs_api.argv_for("run", {"clip": "c.mp4", "slug": "knight",
                                     "action": "strike", "audit": False})
    assert argv == ["--unit", "knight", "--action", "strike", "--no-audit", "c.mp4"]
    with pytest.raises(runs_api.RunRefused) as refused:
        runs_api.argv_for("pick", {"job": "x", "seed": 1})
    assert refused.value.status == 400 and "seed" in str(refused.value)
    with pytest.raises(runs_api.RunRefused):
        runs_api.argv_for("nope", {})


def test_bad_arguments_are_a_400_in_argparses_words(fake_api):
    response = fake_api.post("/jobs", json={"stage": "pick", "argv": ["x", "--nope"]})
    assert response.status_code == 400
    assert "--nope" in response.json()["error"]
    response = fake_api.post("/jobs", json={"stage": "review", "args": {"job": "x"}})
    assert response.status_code == 400, "a page is not a stage to run"
    response = fake_api.post("/jobs", json={"stage": "pick", "args": {"job": "x"},
                                            "argv": ["x"]})
    assert response.status_code == 400


def test_post_a_cpu_stage_runs_the_real_cli_and_finishes(api, posed_job, work_dir):
    response = api.post("/jobs", json={"stage": "pick",
                                       "args": {"job": "knight_strike", "n": 3}})
    assert response.status_code == 202, response.text
    run = response.json()
    assert response.headers["location"] == f"/runs/{run['id']}"
    assert run["stage"] == "pick" and run["gpu"] is False
    assert run["argv"] == ["-n", "3", "knight_strike"]

    final = wait_for(api, run["id"])
    assert final["status"] == "done", "\n".join(final["log"])
    assert final["returncode"] == 0 and final["percent"] == 100
    assert any(line.startswith("pick:") for line in final["log"]), final["log"]
    assert json.loads(posed_job.picks_path.read_text())["n"] == 3, \
        "the subprocess wrote to the API's work/, not the container's"
    assert [r["id"] for r in api.get("/runs").json()] == [run["id"]]
    on_disk = json.loads((work_dir / "_runs" / f"{run['id']}.json").read_text())
    assert on_disk["status"] == "done"
    assert (work_dir / "_runs" / f"{run['id']}.log").read_text().startswith("pick:")
    assert api.get(f"/runs/{run['id']}?tail=1").json()["log"] == final["log"][-1:]
    assert [r["name"] for r in api.get("/jobs").json()] == ["knight_strike"], \
        "_runs/ is not a job"


def test_a_failed_stage_is_reported_as_failed_with_its_log(fake_api, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "3")
    run = fake_api.post("/jobs", json={"stage": "pick", "args": {"job": "x"}}).json()
    final = wait_for(fake_api, run["id"])
    assert final["status"] == "failed" and final["returncode"] == 3
    assert final["percent"] == 100, "the last step line was still [2/2]"


def test_sse_yields_log_progress_and_done(fake_api, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.6")
    run = fake_api.post("/jobs", json={"stage": "pick", "args": {"job": "x"}}).json()
    events = []
    with fake_api.stream("GET", f"/runs/{run['id']}/events") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        name = None
        for line in stream.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    names = [name for name, _ in events]
    # Both lines arrive before the stream opens on a slow box, so the exact
    # interleaving is not asserted - the content and the order of kinds are.
    assert names.count("log") == 2 and names[-1] == "done", names
    assert names.index("progress") > names.index("log")
    assert events[0][1]["line"].startswith("[1/2] started pick x")
    percents = [p["percent"] for n, p in events if n == "progress"]
    assert percents == sorted(percents) and percents[-1] == 100, percents
    assert events[-1][1] == {"status": "done", "returncode": 0, "percent": 100}


def test_sse_after_the_run_finished_replays_the_whole_log(fake_api):
    run = fake_api.post("/jobs", json={"stage": "pick", "args": {"job": "x"}}).json()
    wait_for(fake_api, run["id"])
    with fake_api.stream("GET", f"/runs/{run['id']}/events") as stream:
        body = "".join(stream.iter_text())
    assert body.count("event: log") == 2
    assert body.rstrip().endswith('"percent": 100}')
    assert fake_api.get("/runs/nope/events").status_code == 404


def test_a_second_gpu_run_is_refused_while_one_is_active(fake_api, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "1.5")
    first = fake_api.post("/jobs", json={"stage": "render",
                                         "args": {"job": "knight_strike"}})
    assert first.status_code == 202 and first.json()["gpu"] is True

    second = fake_api.post("/jobs", json={"stage": "base", "args": {"slug": "knight"}})
    assert second.status_code == 409
    assert second.json()["run"]["id"] == first.json()["id"]
    assert "one at a time" in second.json()["error"]
    assert fake_api.get("/health").json()["gpu_run"] == first.json()["id"]

    monkeypatch.setenv("FAKE_DELAY", "0.1")
    cpu = fake_api.post("/jobs", json={"stage": "pick", "args": {"job": "x"}})
    assert cpu.status_code == 202, "a CPU stage is not blocked by the GPU one"

    wait_for(fake_api, first.json()["id"])
    wait_for(fake_api, cpu.json()["id"])
    again = fake_api.post("/jobs", json={"stage": "base", "args": {"slug": "knight"}})
    assert again.status_code == 202, "free again once the first one is over"
    wait_for(fake_api, again.json()["id"])


def test_cpu_stages_stop_at_the_slot_limit(fake_api, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "1.0")
    first = fake_api.post("/jobs", json={"stage": "pick", "args": {"job": "x"}})
    assert first.status_code == 202
    second = fake_api.post("/jobs", json={"stage": "strip", "args": {"job": "x"}})
    assert second.status_code == 409
    assert "limit 1" in second.json()["error"]
    wait_for(fake_api, first.json()["id"])


def test_runs_from_a_previous_process_are_loaded_and_marked_lost(work_dir, tmp_path):
    runs_dir = work_dir / "_runs"
    runs_dir.mkdir()
    (runs_dir / "old-render-1.json").write_text(json.dumps({
        "id": "old-render-1", "stage": "render", "argv": ["knight_strike"],
        "gpu": True, "status": "running", "started": "2026-09-21T00:00:00+00:00"}))
    (runs_dir / "old-render-1.log").write_text("render: seed 1\n")
    manager = runs_api.RunManager(work_dir, python=[sys.executable, "-c", FAKE_STAGE])
    app = create_app(work_dir, web_dist=tmp_path / "no-dist",
                     comfyui_url="http://127.0.0.1:1", runs=manager)
    with TestClient(app) as client:
        row = client.get("/runs/old-render-1").json()
        assert row["status"] == "lost" and row["log"] == ["render: seed 1"]
        assert json.loads((runs_dir / "old-render-1.json").read_text())["status"] == "lost"
        fresh = client.post("/jobs", json={"stage": "render",
                                           "args": {"job": "knight_strike"}})
        assert fresh.status_code != 409, "a lost run does not hold the GPU"
        wait_for(client, fresh.json()["id"])


def test_percent_comes_only_from_step_and_progress_lines():
    percent = runs_api.RunManager._percent
    assert percent("[3/7] pick: [0, 4, 8]") == 43
    assert percent("progress: 80%") == 80
    assert percent("  render: strength 80% of default") is None
    assert percent("[0/0] nothing") is None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def test_settings_default_then_round_trip(api, work_dir, palette):
    fresh = api.get("/settings").json()
    assert {k: v for k, v in fresh.items() if k != "palette"} == \
        {k: v for k, v in settings_api.DEFAULTS.items() if k != "palette"}
    assert fresh["palette"]["path"] == str(config.PALETTE), \
        "no stored path: the CLI's palette, shown expanded"
    assert set(fresh["palette"]) == {"path", "name", "colours", "source"}
    body = {"palette": str(palette),
            "frame_budgets": {"Walk": 8, "attack": 12}, "curation": "every",
            "frame_size": [32, 32]}
    response = api.put("/settings", json=body)
    assert response.status_code == 200, response.text
    stored = response.json()
    assert stored["frame_budgets"] == {"walk": 8, "attack": 12}, "actions lower-cased"
    assert stored["curation"] == "every" and stored["frame_size"] == [32, 32]
    assert stored["palette"]["path"] == str(palette)
    assert stored["palette"]["colours"] == [palettekit.hex_of(c)
                                            for c in imaging.read_palette(palette)]
    assert stored["palette"]["name"] and stored["palette"]["source"]["kind"] == "file"
    assert api.get("/settings").json() == stored
    on_disk = json.loads((work_dir / "project.json").read_text())
    assert on_disk["palette"] == str(palette), "stored as the path"
    assert {k: v for k, v in on_disk.items() if k != "palette"} == \
        {k: v for k, v in stored.items() if k != "palette"}
    # what GET answered, PUT back unchanged: the object form is accepted
    assert api.put("/settings", json=stored).json() == stored
    partial = api.put("/settings", json={"curation": "end"}).json()
    assert partial["curation"] == "end"
    assert partial["frame_budgets"] == {}, "a PUT is the whole document"
    assert partial["palette"]["path"] == str(config.PALETTE)


@pytest.mark.parametrize("body, word", [
    ({"curation": "sometimes"}, "curation"),
    ({"frame_budgets": {"walk": 13}}, "1..12"),
    ({"frame_budgets": {"walk": True}}, "integer"),
    ({"frame_size": [64]}, "frame_size"),
    ({"frame_size": [64, 0]}, "frame_size"),
    ({"palette": ""}, "palette"),
    ({"palette": {"name": "x"}}, "palette"),
    ({"dither": True}, "unknown"),
])
def test_bad_settings_are_a_400(api, body, word, work_dir):
    response = api.put("/settings", json=body)
    assert response.status_code == 400
    assert word in response.json()["detail"]
    assert not (work_dir / "project.json").exists(), "nothing written"


# ---------------------------------------------------------------------------
# The palette feature
# ---------------------------------------------------------------------------
@pytest.fixture
def lospec_dir(tmp_path, monkeypatch):
    """A file:// stand-in for lospec.com, honoured by the subprocess too."""
    root = tmp_path / "lospec"
    root.mkdir()
    (root / "endesga-32.json").write_text(json.dumps(ENDESGA_32))
    monkeypatch.setenv("ANIM_LOSPEC_URL", f"file://{root}/{{slug}}.json")
    monkeypatch.setattr(config, "LOSPEC_URL", f"file://{root}/{{slug}}.json")
    return root


def test_palette_routes_read_what_the_stage_wrote(api, work_dir, tmp_path):
    target = tmp_path / "proj" / "palette.gpl"
    api.put("/settings", json={"palette": str(target)})
    missing = api.get("/palette")
    assert missing.status_code == 404 and "POST /palette" in missing.json()["detail"]
    assert api.get("/palette/units").json() == {"units": [], "path": str(target)}
    assert api.get("/palette/units/knight").status_code == 404

    record = palettekit.install([(0, 0, 0), (255, 255, 255), (60, 78, 128)],
                                name="Tiny", source={"kind": "file", "file": "tiny.hex"},
                                target=target)
    got = api.get("/palette").json()
    assert got == {**record, "path": str(target), "units": []}
    assert api.get("/settings").json()["palette"] == {
        "path": str(target), "name": "Tiny",
        "colours": ["#000000", "#ffffff", "#3c4e80"],
        "source": {"kind": "file", "file": "tiny.hex"}}

    unit = palettekit.write_unit_palette("knight", [(0, 0, 0)], source={"kind": "colours"},
                                         target=target)
    assert api.get("/palette/units/knight").json() == {
        **unit, "path": str(target.parent / "palette_units" / "knight.json")}
    assert api.get("/palette").json()["units"] == ["knight"]
    assert api.get("/palette/units/Bad%20Name").status_code == 400, "not a slug"


def test_post_palette_becomes_an_anim_palette_run(fake_api, work_dir, tmp_path):
    target = tmp_path / "proj" / "palette.gpl"
    fake_api.put("/settings", json={"palette": str(target)})
    response = fake_api.post("/palette", json={"lospec": "endesga-32"})
    assert response.status_code == 202, response.text
    run = response.json()
    assert run["stage"] == "palette" and run["gpu"] is False
    assert run["argv"] == ["--lospec", "endesga-32", "--into", str(target)]
    assert response.headers["location"] == f"/runs/{run['id']}"
    wait_for(fake_api, run["id"])

    run = fake_api.post("/palette", json={"file": "/tmp/x.hex", "name": "X"}).json()
    assert run["argv"] == ["--file", "/tmp/x.hex", "--name", "X", "--into", str(target)]
    wait_for(fake_api, run["id"])
    run = fake_api.post("/palette", json={"unit": "knight",
                                          "colours": ["#000000", "#ffffff"]}).json()
    assert run["argv"] == ["--into", str(target), "--unit", "knight",
                           "--colours", "#000000,#ffffff"]
    wait_for(fake_api, run["id"])
    run = fake_api.post("/palette", json={"unit": "knight", "from_sprite": "k.png"}).json()
    assert run["argv"][-2:] == ["--from-sprite", "k.png"]
    wait_for(fake_api, run["id"])
    run = fake_api.post("/palette", json={"unit": "knight", "clear": True}).json()
    assert run["argv"][-3:] == ["--unit", "knight", "--clear"]
    wait_for(fake_api, run["id"])

    for body in ({}, {"lospec": "a", "file": "b"}, {"unit": "knight"}):
        assert fake_api.post("/palette", json=body).status_code == 400, body


def test_post_palette_runs_the_real_stage_with_the_network_mocked(api, work_dir,
                                                                  tmp_path, lospec_dir):
    target = tmp_path / "proj" / "palette.gpl"
    api.put("/settings", json={"palette": str(target)})
    run = api.post("/palette", json={"lospec": "endesga-32"}).json()
    final = wait_for(api, run["id"])
    assert final["status"] == "done", "\n".join(final["log"])
    assert any(line.startswith("palette: Endesga 32") and "32 colour(s)" in line
               for line in final["log"]), final["log"]
    got = api.get("/palette").json()
    assert got["name"] == "Endesga 32" and got["source"]["slug"] == "endesga-32"
    assert got["colours"][0] == "#be4a2f" and len(got["colours"]) == 32
    assert imaging.read_palette(target)[0] == (0xBE, 0x4A, 0x2F)

    run = api.post("/palette", json={"lospec": "no-such"}).json()
    final = wait_for(api, run["id"])
    assert final["status"] == "failed"
    assert any("could not reach" in line or "no Lospec palette" in line
               for line in final["log"]), final["log"]


# ---------------------------------------------------------------------------
# The pages
# ---------------------------------------------------------------------------
def test_the_fallback_page_lists_jobs_when_there_is_no_editor_build(api):
    page = api.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "anim pipeline" in page.text and "getJSON('/jobs')" in page.text
    assert api.get("/index.html").url.path == "/"


def test_the_review_page_is_dark_pixelated_and_driven_by_the_picks_endpoint(api):
    page = api.get("/jobs/knight_strike/review").text
    assert "image-rendering:pixelated" in page.replace(" ", "")
    assert "#15171c" in page, "dark ground"
    assert 'data-job="knight_strike"' in page
    assert "/picks" in page and "Request regenerate" in page
    assert api.get("/jobs/nope/review").status_code == 404


def test_web_dist_is_served_at_the_root_when_it_exists(posed_job, work_dir, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>editor</title>")
    (dist / "app.js").write_text("console.log('editor')")
    app = create_app(work_dir, web_dist=dist, comfyui_url="http://127.0.0.1:1")
    with TestClient(app) as client:
        assert "<title>editor</title>" in client.get("/").text
        assert client.get("/app.js").text == "console.log('editor')"
        assert client.get("/jobs").json()[0]["name"] == "knight_strike", \
            "the API routes win over the static mount"
        assert "data-job" in client.get("/jobs/knight_strike/review").text


def test_health_does_not_touch_comfyui_unless_asked(api):
    row = api.get("/health").json()
    assert row["ok"] is True and "comfyui" not in row
    assert row["gpu_run"] is None and row["cpu_runs"] == []
    assert api.get("/health?probe=1").json()["comfyui"] is False
