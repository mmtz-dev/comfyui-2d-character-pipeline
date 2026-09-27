"""`anim bakeoff` - the frame route bake-off feature, against FakeComfy.

Four promises are pinned here:
  1. The plan is four named cells, V before K, both rigs, and the two
     samplers (the even spread, the phase instants) hit both ends.
  2. Both routes run end to end through the fake GPU and the fake segmenter -
     rig, clip / three 4-cell strips, pick, post --parts - and the page names
     every cell and every frame.
  3. A second run skips the cells that are done (and --force would not).
  4. The metric table is built from the part report and the recorded masks,
     and the two proxies say what they say on hand-built masks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anim import comfy, config, outcome, partkit
from anim.job import Job
from anim.stages import bakeoff as bakeoff_stage
from anim.stages import clip as clip_stage
from anim.stages import pick as pick_stage
from fixtures import synthetic
from fixtures.fakecomfy import FakeComfy
from fixtures.fakeparts import FakeSegmenter

REPO = Path(__file__).resolve().parent.parent
KNIGHT = REPO / "rigs" / "kaykit-adventurers" / "Knight.glb"
ONE_RIG = {"kaykit": {"glb": KNIGHT, "label": "KayKit"}}
ONE_CLIP = {"kaykit": "1H_Melee_Attack_Chop"}
# The synthetic figure's colours, mapped onto the knight's part table.
COLOURS = {"sword": (150, 150, 158), "helmet": (196, 152, 118),
           "torso": (60, 78, 128), "legs": (40, 42, 52)}


def _parts_file(path: Path, palette: Path) -> Path:
    """Part palettes to lock to: each synthetic colour's nearest palette
    member, because a locked pixel must still be a member of the unit
    palette or `post`'s verification refuses the frame."""
    from anim import imaging

    members = imaging.read_palette(palette)

    def nearest(rgb):
        return list(min(members, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, rgb))))

    path.write_text(json.dumps({
        "schema": partkit.PARTS_SCHEMA, "slug": "knight", "job": "test",
        "parts": ["sword", "shield", "helmet", "arms", "torso", "legs"],
        "palettes": {"sword": [nearest(COLOURS["sword"])], "shield": [],
                     "helmet": [nearest(COLOURS["helmet"])], "arms": [],
                     "torso": [nearest(COLOURS["torso"])], "legs": [nearest(COLOURS["legs"])]},
    }))
    return path


def _strip_png(cells: int, cell: tuple[int, int], tmp_path: Path, name: str) -> bytes:
    path = synthetic.write_strip(tmp_path / name, cells=cells, cell=cell)
    return path.read_bytes()


# ---------------------------------------------------------------------------
# 1. The plan
# ---------------------------------------------------------------------------
def test_the_plan_is_four_named_cells_v_before_k():
    cells = bakeoff_stage.plan("knight", "attack")
    assert [c.name for c in cells] == ["V-kaykit", "V-quaternius", "K-kaykit", "K-quaternius"]
    assert {c.job for c in cells} == {f"bakeoff_knight_attack_{c.name}" for c in cells}
    assert cells[0].clip == "1H_Melee_Attack_Chop" and cells[1].clip == "Sword_Attack"
    with pytest.raises(ValueError):
        bakeoff_stage.plan("knight", "dance")


def test_the_two_samplers_hit_both_ends():
    spread = bakeoff_stage.even_spread(19, 12)
    assert len(spread) == 12 and spread[0] == 0 and spread[-1] == 18
    assert spread == sorted(set(spread))
    assert bakeoff_stage.even_spread(9, 12) == list(range(9))
    times = bakeoff_stage.phase_times(1.5, 12)
    assert len(times) == 12 and times[0] == 0.0 and times[-1] == 1.5
    assert all(b > a for a, b in zip(times, times[1:]))
    assert len(bakeoff_stage.PHASES) == bakeoff_stage.BUDGET
    assert set(bakeoff_stage.PHASE_KEYS) <= set(range(bakeoff_stage.BUDGET))


# ---------------------------------------------------------------------------
# 2 + 3. Both routes through the fake GPU, then a resume
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """V-kaykit then K-kaykit, once for the module, each against a FakeComfy
    that answers the shape that route asks for (a frame batch; a 4-cell
    strip). Module-scoped because the two runs cost ~20 s; the tests below
    read the result and the one that breaks a cell works on a copy."""
    if not KNIGHT.exists():
        pytest.skip("rigs/kaykit-adventurers/Knight.glb is not vendored")
    from conftest import ASSETS_MOUNT, PALETTE_CANDIDATE

    tmp_path = tmp_path_factory.mktemp("bakeoff")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    if PALETTE_CANDIDATE.exists():
        palette = PALETTE_CANDIDATE
    else:
        palette = tmp_path / "fallback.gpl"
        ramp = [(8 * i, 6 * i, 10 * i) for i in range(16)]
        palette.write_text("GIMP Palette\nName: Fallback\nColumns: 8\n" + "".join(
            f"{r:3d} {g:3d} {b:3d}\t#{r:02x}{g:02x}{b:02x}\n" for r, g, b in ramp))
    reference = ASSETS_MOUNT / "2d" / "units" / "knight.png"
    if not reference.exists():
        reference = tmp_path / "knight.png"
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        from PIL import ImageDraw
        ImageDraw.Draw(img).rectangle([24, 26, 40, 58], fill=(90, 104, 150, 255))
        img.save(reference)
    parts_file = _parts_file(tmp_path / "parts.json", palette)
    logs: list[str] = []
    common = dict(matte="chroma", parts_from=parts_file, rigs=ONE_RIG, clips=ONE_CLIP,
                  reference=reference, work_dir=work_dir,
                  assets_dir=tmp_path / "no-assets", workflow_dir=REPO / "gpu" / "workflows",
                  log=logs.append)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PALETTE", palette)
        # Route V: the rig's frame count decides the padded batch length.
        from anim import rigkit
        duration = rigkit.load_rig(KNIGHT).clip(ONE_CLIP["kaykit"]).duration
        rig_frames = len(rigkit.sample_times(duration, config.EXTRACT_FPS))
        frame = _strip_png(1, tuple(bakeoff_stage.V_CELL), tmp_path, "frame.png")
        batch = [frame] * clip_stage.padded_length(rig_frames)
        with FakeComfy(batch=batch) as fake_v:
            state = bakeoff_stage.bakeoff(
                "knight", "attack", cells=["V-kaykit"], client=comfy.ComfyClient(fake_v.url),
                segmenter=FakeSegmenter(COLOURS), **common)
        v_prompts = len(fake_v.prompts)

        # Route K: every /prompt answers one drawn 4-cell strip.
        strip = _strip_png(4, tuple(bakeoff_stage.K_CELL), tmp_path, "strip.png")
        with FakeComfy(png=strip) as fake_k:
            state = bakeoff_stage.bakeoff(
                "knight", "attack", cells=["K-kaykit"], client=comfy.ComfyClient(fake_k.url),
                segmenter=FakeSegmenter(COLOURS), **common)
        yield {"state": state, "logs": logs, "v_prompts": v_prompts,
               "k_prompts": len(fake_k.prompts), "rig_frames": rig_frames,
               "work_dir": work_dir, "common": common, "tmp_path": tmp_path}


def test_route_v_renders_the_clip_and_downsamples_to_the_budget(ran):
    record = ran["state"]["cells"]["V-kaykit"]
    assert record["status"] == "done", record.get("error")
    assert ran["v_prompts"] == 1                       # one Wan job
    assert record["control"] == "blend" and record["depth_gain"] == 0.35
    assert record["rig_frames"] == ran["rig_frames"]
    assert len(record["picks"]) == bakeoff_stage.BUDGET
    assert record["picks"][0] == 0 and record["picks"][-1] == ran["rig_frames"] - 1
    job = Job.open(record["job"], ran["work_dir"])
    assert len(bakeoff_stage.frame_files(job)) == bakeoff_stage.BUDGET
    picks = json.loads(job.picks_path.read_text())
    assert picks["picks"] == record["picks"]        # the spread is in force ...
    assert len(picks["auto"]) == bakeoff_stage.BUDGET   # ... the picker's choice kept beside it
    assert job.read_manifest()["bakeoff"]["status"] == "done"
    assert set(record["timings"]) == {"rig", "gpu", "post"}


def test_route_k_samples_twelve_instants_and_renders_three_strips(ran):
    record = ran["state"]["cells"]["K-kaykit"]
    assert record["status"] == "done", record.get("error")
    assert record["rig_frames"] == bakeoff_stage.BUDGET
    assert len(record["times"]) == bakeoff_stage.BUDGET
    assert [c["picks"] for c in record["chunks"]] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert ran["k_prompts"] == 3 * 2                    # pose + identity per chunk
    assert record["strength"] == 1.0 and "knight" in record["pose_prompt"]
    job = Job.open(record["job"], ran["work_dir"])
    with Image.open(job.render_dir / "pose_pass.png") as joined:
        assert joined.size == (bakeoff_stage.K_CELL[0] * 12, bakeoff_stage.K_CELL[1])
    assert (job.render_dir / "identity_pass.png").exists()
    assert job.read_manifest()["render"]["final"] == "pose_pass.png"
    assert len(bakeoff_stage.frame_files(job)) == bakeoff_stage.BUDGET
    report = json.loads(job.report_path.read_text())
    assert report["route"] == "clip" and len(report["frames"]) == 12   # matted per cell
    assert len(job.clip_frames()) == 12
    for chunk in record["chunks"]:
        assert Job.open(chunk["job"], ran["work_dir"]).provenance_path.exists()


def test_the_page_names_every_cell_and_frame(ran):
    state = ran["state"]
    page = Path(state["page"])
    html = page.read_text(encoding="utf-8")
    for name in ("V-kaykit", "K-kaykit"):
        assert f'id="{name}"' in html
        for i in range(bakeoff_stage.BUDGET):
            assert (page.parent / "img" / f"{name}_post_{i:02d}.png").exists()
            assert f"{name}_post_{i:02d}.png" in html
            assert (page.parent / "img" / f"{name}_raw_{i:02d}.png").exists()
        assert (page.parent / "img" / f"{name}_sheet.png").exists()
    assert (page.parent / "img" / "K-kaykit_xfade_sheet.png").exists()
    assert (page.parent / "img" / "K-kaykit_identity.png").exists()
    for phase in ("start", "wind-up", "action", "wind-down", "end"):
        assert phase in html
    assert "Summary" in html and "one sword, sword hand" in html
    report = Path(state["report"]).read_text(encoding="utf-8")
    assert "| V-kaykit |" in report and "| K-kaykit |" in report
    assert "The owner's questions" in report and "Wan-side LoRA" in report


def test_a_second_run_skips_the_cells_that_are_done(ran):
    logs: list[str] = []
    common = dict(ran["common"], log=logs.append)
    with FakeComfy(fail_prompt=True) as fake:   # any GPU call would fail loudly
        state = bakeoff_stage.bakeoff("knight", "attack", cells=["V-kaykit", "K-kaykit"],
                                      client=comfy.ComfyClient(fake.url),
                                      segmenter=FakeSegmenter(COLOURS), **common)
    assert not fake.prompts
    assert all("skipped" in line for line in logs if "bakeoff:" in line)
    assert state["cells"]["V-kaykit"]["status"] == "done"
    assert state["cells"]["K-kaykit"]["metrics"]["frames"] == bakeoff_stage.BUDGET
    # And a page-only run needs no service at all.
    state = bakeoff_stage.bakeoff("knight", "attack", page_only=True, **common)
    assert Path(state["page"]).exists()


def test_a_failed_cell_is_recorded_and_the_page_still_builds(ran):
    import shutil

    copy = ran["tmp_path"] / "work_copy"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(ran["work_dir"], copy)
    logs: list[str] = []
    common = dict(ran["common"], log=logs.append, work_dir=copy)
    with FakeComfy(fail_prompt=True) as fake:
        state = bakeoff_stage.bakeoff("knight", "attack", cells=["V-kaykit"], force=True,
                                      client=comfy.ComfyClient(fake.url),
                                      segmenter=FakeSegmenter(COLOURS), **common)
    record = state["cells"]["V-kaykit"]
    assert record["status"] == "failed" and "ComfyError" in record["error"]
    assert Path(state["page"]).exists()
    assert "failed" in Path(state["page"]).read_text(encoding="utf-8")
    # The page still builds, and the run still says it accomplished nothing:
    # `summary` counts the `--cells` subset, `plan` keeps every row the page
    # draws (the fixture's plan is one rig, so two of the four).
    assert state["selected"] == ["V-kaykit"]
    assert state["plan"] == ["V-kaykit", "K-kaykit"]
    assert state["summary"] == {"done": 0, "failed": 1, "blocked": 0, "of": 1}
    assert outcome.count(state["cells"], state["selected"]).exit_code == \
        outcome.EXIT_NOTHING


def test_unknown_cells_are_refused(work_dir):
    with pytest.raises(ValueError):
        bakeoff_stage.bakeoff("knight", "attack", cells=["V-nope"], page_only=True,
                              work_dir=work_dir)


# ---------------------------------------------------------------------------
# 4. The metric table
# ---------------------------------------------------------------------------
def test_the_metric_table_reads_the_part_report_and_the_recorded_masks(ran):
    m = ran["state"]["cells"]["K-kaykit"]["metrics"]
    assert m["budget"] == 12 and m["frames"] == 12 and m["frames_attempted"] == 12
    assert set(m["parts"]) == {"sword", "shield", "helmet", "arms", "torso", "legs"}
    # The fake segmenter has no colour for the shield or the arms: not found
    # in every frame, reported as such.
    assert m["parts"]["shield"]["missing"] == list(range(12))
    assert m["parts"]["arms"]["missing"] == list(range(12))
    assert m["parts"]["torso"]["missing"] == [] and m["parts"]["torso"]["pixels"] > 0
    assert m["parts"]["torso"]["drift"] is not None
    assert m["one_sword"]["of"] == 12 and m["one_figure"]["of"] == 12
    assert m["one_figure"]["count"] == 12
    for entry in m["one_sword"]["frames"]:
        assert set(entry) >= {"frame", "ok", "blobs", "right", "left", "reason"}


def _prompts() -> list[partkit.Prompt]:
    return [partkit.Prompt(part="sword", points=[(10.0, 10.0)]),
            partkit.Prompt(part="helmet", points=[(50.0, 20.0)])]


def test_the_sword_proxy_wants_one_blob_near_the_right_wrist():
    alpha = np.full((100, 100), 255, dtype=np.uint8)
    sword = np.zeros((100, 100), dtype=bool)
    sword[20:60, 70:74] = True            # one blade, top at (72, 20)
    results = [(sword, 1.0), (np.zeros((100, 100), dtype=bool), 1.0)]
    near = {"right_wrist": (72.0, 62.0), "left_wrist": (10.0, 90.0)}
    assert bakeoff_stage.sword_check(_prompts(), results, alpha, near, 100)["ok"]
    far = {"right_wrist": (5.0, 95.0), "left_wrist": (0.0, 99.0)}
    verdict = bakeoff_stage.sword_check(_prompts(), results, alpha, far, 100)
    assert not verdict["ok"] and "sword hand" in verdict["reason"]
    swapped = {"right_wrist": (10.0, 90.0), "left_wrist": (72.0, 62.0)}
    verdict = bakeoff_stage.sword_check(_prompts(), results, alpha, swapped, 100)
    assert not verdict["ok"] and "shield hand" in verdict["reason"]
    two = sword.copy()
    two[20:60, 10:14] = True              # a second blade behind the legs
    verdict = bakeoff_stage.sword_check(_prompts(), [(two, 1.0), results[1]], alpha,
                                        near, 100)
    assert not verdict["ok"] and verdict["blobs"] == 2
    verdict = bakeoff_stage.sword_check(_prompts(), [None, results[1]], alpha, near, 100)
    assert not verdict["ok"] and verdict["reason"] == "no sword mask"


def test_the_figure_proxy_wants_one_body_and_at_most_one_helmet():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[10:90, 30:70] = 255
    helmet = np.zeros((100, 100), dtype=bool)
    helmet[10:25, 40:60] = True
    assert bakeoff_stage.figure_check(_prompts(), [None, (helmet, 1.0)], alpha)["ok"]
    two_heads = helmet.copy()
    two_heads[60:75, 40:60] = True
    verdict = bakeoff_stage.figure_check(_prompts(), [None, (two_heads, 1.0)], alpha)
    assert not verdict["ok"] and verdict["helmets"] == 2
    two_bodies = alpha.copy()
    two_bodies[:, 45:55] = 0
    verdict = bakeoff_stage.figure_check(_prompts(), [None, (helmet, 1.0)], two_bodies)
    assert not verdict["ok"] and verdict["bodies"] == 2


def test_the_crossfade_holds_the_keys_and_blends_between():
    canvas = tuple(config.CANVAS)
    red = Image.new("RGBA", canvas, (255, 0, 0, 255))
    blue = Image.new("RGBA", canvas, (0, 0, 255, 255))
    frames = bakeoff_stage.crossfade({0: red, 4: blue}, budget=6)
    assert len(frames) == 6
    assert frames[0].getpixel((0, 0)) == (255, 0, 0, 255)
    assert frames[4].getpixel((0, 0)) == (0, 0, 255, 255)
    mid = frames[2].getpixel((0, 0))
    assert 100 < mid[0] < 160 and 100 < mid[2] < 160
    assert frames[5].getpixel((0, 0)) == (0, 0, 255, 255)   # past the last key: held


def test_the_phase_labels_come_from_the_attack_template(ran):
    """The keyframe phase feature: the column labels and the crossfade keys
    are the attack row of config.ACTION_PHASES, and each cell's picks.json
    carries the tag per column with the even spread the owner judged."""
    plan = pick_stage.phase_plan("attack")
    assert bakeoff_stage.PHASES == tuple(plan.tags())
    assert bakeoff_stage.PHASE_KEYS == tuple(plan.keys())
    assert bakeoff_stage.BUDGET == plan.budget == config.ACTION_PHASES["attack"]["budget"]
    for name in ("V-kaykit", "K-kaykit"):
        job = Job.open(ran["state"]["cells"][name]["job"], ran["work_dir"])
        record = json.loads(job.picks_path.read_text())
        assert record["phases"]["spread"] == "even"
        assert record["phases"]["tags"] == list(bakeoff_stage.PHASES)
        assert record["timing"]["source"] == "phases:attack"
        assert record["picks"] == record["auto"], "no hand-set spread any more"
