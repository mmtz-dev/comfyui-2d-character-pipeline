"""`anim concept` - the character concept feature - against the fake ComfyUI.

The job layout and the pick are pure file work and are tested directly; the
drive is tested end to end against `FakeComfy`, asserting on the graph it was
sent (prompt, seed, LoRA, sketch, and - without a sketch - no ControlNet at
all); then the pick's hand-over to the base job every later stage reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

from fixtures.fakecomfy import FakeComfy, canned_png

from anim import comfy, config
from anim.job import Job
from anim.stages import base as base_stage
from anim.stages import clip as clip_stage
from anim.stages import concept as concept_stage


@pytest.fixture
def sketch(tmp_path: Path) -> Path:
    """A pencil sketch: dark strokes on white paper, portrait, not the cell size."""
    img = Image.new("RGB", (300, 500), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([110, 40, 190, 120], outline=(30, 30, 30), width=4)
    draw.line([(150, 120), (150, 330)], fill=(30, 30, 30), width=5)
    draw.line([(150, 330), (100, 470)], fill=(30, 30, 30), width=5)
    draw.line([(150, 330), (200, 470)], fill=(30, 30, 30), width=5)
    path = tmp_path / "sketch.png"
    img.save(path)
    return path


def _graph_classes(graph: dict) -> list[str]:
    return sorted(node["class_type"] for node in graph.values())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_the_prompt_leads_with_the_lora_trigger_and_asks_for_a_front_view():
    prompt = config.concept_prompt("a plate-armoured knight")
    assert prompt.startswith(config.CONCEPT_LORA["trigger"])
    assert "a plate-armoured knight" in prompt
    assert "front-view" in prompt and "white background" in prompt


def test_the_concept_cell_is_square_and_on_the_latent_grid():
    width, height = config.CONCEPT_CELL
    assert width == height
    assert width % config.LATENT_STEP == 0


def test_the_lora_row_is_locked_and_commercially_usable():
    from anim import models_lock

    rows = {m.file: m for m in models_lock.load(
        Path(__file__).resolve().parent.parent / "gpu" / "models.lock")}
    row = rows[config.CONCEPT_LORA["file"]]
    assert row.repo == config.CONCEPT_LORA["repo"]
    assert row.subdir == "loras"
    assert row.license == "apache-2.0"
    assert len(row.sha256) == 64 and row.size > 100_000_000


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def test_prepare_writes_the_job_with_one_seed_per_candidate(work_dir):
    job = concept_stage.prepare("knight", description="a knight", count=3,
                                seed=100, work_dir=work_dir)
    assert job.name == "knight_concept"
    manifest = job.read_manifest()
    assert manifest["schema"] == concept_stage.CONCEPT_SCHEMA
    assert manifest["prompt"] == config.concept_prompt("a knight")
    assert manifest["cell"] == list(config.CONCEPT_CELL)
    assert manifest["lora"] == {"repo": config.CONCEPT_LORA["repo"],
                                "file": config.CONCEPT_LORA["file"],
                                "trigger": config.CONCEPT_LORA["trigger"],
                                "strength": config.CONCEPT_LORA["strength"]}
    assert manifest["sketch"] is None and manifest["pick"] is None
    assert [c["file"] for c in manifest["candidates"]] == ["00.png", "01.png", "02.png"]
    assert [c["seed"] for c in manifest["candidates"]] == [100, 101, 102]
    assert {c["source"] for c in manifest["candidates"]} == {"txt2img"}
    assert manifest["workflow"] == "concept_pass.json"
    assert job.candidates_dir.is_dir()
    # The concept generation feature: every attempt says what it was made
    # with - its own prompt, checkpoint and LoRA list - and where from.
    first = manifest["candidates"][0]
    assert first["prompt"] == manifest["prompt"] and first["description"] == "a knight"
    assert first["model"] == config.CONCEPT_MODELS["zimage"]
    assert first["loras"] == [{"file": config.CONCEPT_LORA["file"],
                               "strength": config.CONCEPT_LORA["strength"]}]
    assert first["parent"] is None and first["workflow"] == "concept_pass.json"
    assert manifest["defaults"] == {} and manifest["next_index"] == 3
    assert manifest["last_seed"] == 100


def test_a_second_prepare_appends_and_fresh_wipes(work_dir):
    """Pressing Generate again keeps the earlier attempts (their files too);
    --fresh is the old wipe."""
    job = concept_stage.prepare("knight", description="a knight", count=2, seed=1,
                                work_dir=work_dir)
    for c in job.read_manifest()["candidates"]:
        (job.candidates_dir / c["file"]).write_bytes(b"png")
    concept_stage.prepare("knight", description="a taller knight", count=2, seed=50,
                          work_dir=work_dir)
    manifest = job.read_manifest()
    assert [c["file"] for c in manifest["candidates"]] == ["00.png", "01.png", "02.png", "03.png"]
    assert [c["seed"] for c in manifest["candidates"]] == [1, 2, 50, 51]
    assert manifest["candidates"][0]["description"] == "a knight"
    assert manifest["candidates"][2]["description"] == "a taller knight"
    assert manifest["description"] == "a taller knight", "the manifest keeps the last ask"
    assert (job.candidates_dir / "01.png").exists(), "nothing deleted"
    assert manifest["next_index"] == 4
    concept_stage.prepare("knight", description="a knight", count=1, seed=9, fresh=True,
                          work_dir=work_dir)
    manifest = job.read_manifest()
    assert [c["file"] for c in manifest["candidates"]] == ["00.png"]
    assert not (job.candidates_dir / "01.png").exists()


def test_a_v1_manifest_is_migrated_in_place(work_dir):
    job = Job.open("knight_concept", work_dir)
    job.write_manifest({
        "schema": "anim.concept/1", "slug": "knight", "created": "2026-09-20T00:00:00+00:00",
        "description": "a knight", "prompt": "Pixel art style. a knight", "seed": 5,
        "count": 2, "cell": [768, 768],
        "lora": {"repo": "r", "file": config.CONCEPT_LORA["file"], "trigger": "t", "strength": 0.8},
        "sketch": None, "workflow": "concept_pass.json", "workflow_sha256": "x",
        "comfyui": None, "models": [],
        "candidates": [{"index": 0, "file": "00.png", "source": "existing", "from": "a.png",
                        "seed": None, "seconds": None},
                       {"index": 1, "file": "01.png", "source": "generated", "seed": 5,
                        "seconds": 8.0, "prompt_id": "p"}],
        "pick": None})
    store = concept_stage.Attempts("knight", "base", work_dir)
    manifest = store.record
    assert manifest["schema"] == concept_stage.CONCEPT_SCHEMA
    assert [c["source"] for c in manifest["candidates"]] == ["existing", "txt2img"]
    generated = manifest["candidates"][1]
    assert generated["prompt"] == "Pixel art style. a knight"
    assert generated["loras"] == [{"file": config.CONCEPT_LORA["file"], "strength": 0.8,
                                   "trigger": "t"}]
    assert generated["model"] == config.CONCEPT_MODELS["zimage"]
    assert manifest["candidates"][0]["prompt"] is None
    assert manifest["next_index"] == 2 and manifest["defaults"] == {}
    assert json.loads(job.manifest_path.read_text())["schema"] == concept_stage.CONCEPT_SCHEMA


def test_the_seed_is_random_by_default_and_derived_for_a_fresh_set(work_dir):
    """Generate again must not redraw the same set, so a run without --seed
    is random; a --fresh run derives it from the ask, so a unit's first set
    reproduces."""
    a = concept_stage.prepare("knight", description="a knight", fresh=True, work_dir=work_dir)
    seed_a = a.read_manifest()["seed"]
    b = concept_stage.prepare("knight", description="a knight", fresh=True, work_dir=work_dir)
    assert b.read_manifest()["seed"] == seed_a, "same ask, same first set"
    c = concept_stage.prepare("knight", description="a hound", fresh=True, work_dir=work_dir)
    assert c.read_manifest()["seed"] != seed_a
    assert 0 <= seed_a < 2 ** 53
    seeds = {concept_stage.prepare("knight", description="a hound", count=1,
                                   work_dir=work_dir).read_manifest()["seed"] for _ in range(4)}
    assert len(seeds) > 1 and all(0 <= s < 2 ** 53 for s in seeds)


def test_prepare_refuses_an_empty_description_or_count(work_dir):
    with pytest.raises(ValueError, match="--prompt"):
        concept_stage.prepare("knight", description="  ", work_dir=work_dir)
    with pytest.raises(ValueError, match="--count"):
        concept_stage.prepare("knight", description="a knight", count=0,
                              work_dir=work_dir)


def test_existing_art_is_candidate_00_and_the_generated_ones_follow(
        work_dir, reference_sprite):
    job = concept_stage.prepare("knight", description="a knight", count=2, seed=5,
                                existing=reference_sprite, work_dir=work_dir)
    manifest = job.read_manifest()
    files = [c["file"] for c in manifest["candidates"]]
    assert files == ["00.png", "01.png", "02.png"]
    first = manifest["candidates"][0]
    assert first["source"] == "existing" and first["seed"] is None
    assert first["from"] == str(reference_sprite)
    assert [c["seed"] for c in manifest["candidates"][1:]] == [5, 6]
    with Image.open(job.candidates_dir / "00.png") as img:
        assert img.size == tuple(config.CONCEPT_CELL), "same format as a render"
        arr = np.asarray(img.convert("RGB"))
    assert tuple(arr[0, 0]) == (255, 255, 255), "on white, like the base pass"
    assert (arr != 255).any(), "the sprite is actually there"


def test_existing_auto_is_the_published_sprite_upscaled_x8(work_dir, assets_tree,
                                                          reference_sprite):
    units = assets_tree / "assets" / "2d" / "units"
    (units / "knight.png").write_bytes(reference_sprite.read_bytes())
    job = concept_stage.prepare("knight", description="a knight", count=1,
                                existing="auto", assets_dir=assets_tree / "assets",
                                work_dir=work_dir)
    first = job.read_manifest()["candidates"][0]
    assert first["source"] == "existing"
    assert first["from"] == str(units / "knight.png")
    with Image.open(job.candidates_dir / "00.png") as img:
        arr = np.asarray(img.convert("RGB"))
    rows = np.where((arr != 255).any(axis=(1, 2)))[0]
    assert rows.max() == config.CONCEPT_CELL[1] - 1, "feet on the bottom edge"
    assert rows.max() - rows.min() + 1 >= 8 * 8, "x8 nearest, not 64 px"


def test_existing_auto_without_a_published_sprite_says_so(work_dir, assets_tree):
    with pytest.raises(FileNotFoundError, match="anim base"):
        concept_stage.prepare("knight", description="a knight", existing="auto",
                              assets_dir=assets_tree / "assets", work_dir=work_dir)


# ---------------------------------------------------------------------------
# The sketch as a scribble
# ---------------------------------------------------------------------------
def test_a_pencil_sketch_becomes_white_strokes_on_black_in_the_cell(sketch, tmp_path):
    out = concept_stage.sketch_control(sketch, config.CONCEPT_CELL,
                                       tmp_path / "control.png")
    with Image.open(out) as img:
        assert img.size == tuple(config.CONCEPT_CELL)
        arr = np.asarray(img.convert("L"))
    assert arr[0, 0] == 0 and arr[0, -1] == 0, "black field (letterbox included)"
    assert (arr > 200).sum() > 0, "the strokes are white"
    assert (arr > 128).mean() < 0.2, "and sparse, as a scribble is"
    bottom = np.where((arr > 128).any(axis=1))[0].max()
    # The paper is letterboxed to the cell's bottom edge, margins kept: the
    # legs end 30 px above the paper's edge, so ~46 px above the cell's.
    assert config.CONCEPT_CELL[1] - 60 < bottom < config.CONCEPT_CELL[1] - 30


def test_a_transparent_sketch_treats_alpha_as_paper(tmp_path):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(img).line([(8, 8), (56, 56)], fill=(0, 0, 0, 255), width=3)
    src = tmp_path / "alpha.png"
    img.save(src)
    out = concept_stage.sketch_control(src, (128, 128), tmp_path / "c.png")
    with Image.open(out) as ctl:
        arr = np.asarray(ctl.convert("L"))
    assert arr[2, 120] == 0, "transparent -> paper -> black"
    assert arr[64, 64] > 128, "the ink -> white"


# ---------------------------------------------------------------------------
# The graph that goes over the wire
# ---------------------------------------------------------------------------
def test_the_workflow_loads_and_the_inputs_land_on_the_right_nodes(work_dir,
                                                                  workflow_dir):
    job = concept_stage.prepare("knight", description="a knight", count=1, seed=42,
                                work_dir=work_dir)
    graph = concept_stage.build_workflow(job.read_manifest(), seed=42,
                                         sketch_name=None, job=job,
                                         workflow_dir=workflow_dir)
    text = next(n for n in graph.values() if n["class_type"] == "CLIPTextEncode")
    assert text["inputs"]["text"] == config.concept_prompt("a knight")
    latent = next(n for n in graph.values() if n["class_type"] == "EmptySD3LatentImage")
    assert [latent["inputs"]["width"], latent["inputs"]["height"]] == \
        list(config.CONCEPT_CELL)
    sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["seed"] == 42
    assert sampler["inputs"]["sampler_name"] == "euler", "reproducibility rule"
    lora = next(n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly")
    assert lora["inputs"]["lora_name"] == config.CONCEPT_LORA["file"]
    assert lora["inputs"]["strength_model"] == config.CONCEPT_LORA["strength"]
    save = next(n for n in graph.values() if n["class_type"] == "SaveImage")
    assert save["inputs"]["filename_prefix"].startswith("anim_pipeline/knight_concept/")


def test_without_a_sketch_no_controlnet_node_is_in_the_graph(work_dir, workflow_dir):
    job = concept_stage.prepare("knight", description="a knight", count=1,
                                work_dir=work_dir)
    graph = concept_stage.build_workflow(job.read_manifest(), seed=1,
                                         sketch_name=None, job=job,
                                         workflow_dir=workflow_dir)
    classes = _graph_classes(graph)
    assert "ZImageFunControlnet" not in classes
    assert "ModelPatchLoader" not in classes
    assert "LoadImage" not in classes
    # The LoRA'd model feeds the sampler's shift node directly.
    shift = next(n for n in graph.values() if n["class_type"] == "ModelSamplingAuraFlow")
    assert graph[shift["inputs"]["model"][0]]["class_type"] == "LoraLoaderModelOnly"


def test_with_a_sketch_the_scribble_and_its_strength_are_wired(work_dir,
                                                               workflow_dir, sketch):
    job = concept_stage.prepare("knight", description="a knight", count=1,
                                sketch=sketch, sketch_strength=0.55,
                                work_dir=work_dir)
    manifest = job.read_manifest()
    assert manifest["sketch"]["control"] == "scribble"
    assert manifest["sketch"]["strength"] == 0.55
    assert (job.root / manifest["sketch"]["file"]).exists()
    graph = concept_stage.build_workflow(manifest, seed=1,
                                         sketch_name="anim_knight_concept_sketch.png",
                                         job=job, workflow_dir=workflow_dir)
    load = next(n for n in graph.values() if n["class_type"] == "LoadImage")
    assert load["inputs"]["image"] == "anim_knight_concept_sketch.png"
    control = next(n for n in graph.values() if n["class_type"] == "ZImageFunControlnet")
    assert control["inputs"]["strength"] == 0.55
    assert graph[control["inputs"]["model"][0]]["class_type"] == "LoraLoaderModelOnly"
    assert graph[control["inputs"]["image"][0]]["class_type"] == "LoadImage"
    assert "ModelPatchLoader" in _graph_classes(graph)


def test_lora_strength_zero_drops_the_lora_node(work_dir, workflow_dir):
    job = concept_stage.prepare("knight", description="a knight", count=1,
                                lora_strength=0, work_dir=work_dir)
    graph = concept_stage.build_workflow(job.read_manifest(), seed=1,
                                         sketch_name=None, job=job,
                                         workflow_dir=workflow_dir)
    assert "LoraLoaderModelOnly" not in _graph_classes(graph)
    shift = next(n for n in graph.values() if n["class_type"] == "ModelSamplingAuraFlow")
    assert graph[shift["inputs"]["model"][0]]["class_type"] == "UNETLoader"


def test_prune_orphans_keeps_the_output_node_and_removes_dead_loaders():
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "ModelPatchLoader", "inputs": {}},
        "3": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
        "4": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
    }
    removed = comfy.prune_orphans(graph)
    assert sorted(removed) == ["2", "3"]
    assert sorted(graph) == ["1", "4", "5"]


# ---------------------------------------------------------------------------
# The drive
# ---------------------------------------------------------------------------
def test_render_concept_posts_one_prompt_per_candidate(work_dir, workflow_dir):
    png = canned_png(config.CONCEPT_CELL)
    with FakeComfy(png=png) as fake:
        manifest = concept_stage.concept(
            "knight", description="a knight", count=3, seed=10,
            client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir,
            work_dir=work_dir)
    assert len(fake.prompts) == 3
    seeds = [next(n for n in g.values() if n["class_type"] == "KSampler")["inputs"]["seed"]
             for g in fake.prompts]
    assert seeds == [10, 11, 12]
    assert fake.uploads == [], "no sketch, nothing to upload"
    job = Job.open("knight_concept", work_dir)
    files = sorted(p.name for p in job.candidates_dir.glob("*.png"))
    assert files == ["00.png", "01.png", "02.png"]
    assert (job.candidates_dir / "01.png").read_bytes() == png
    for candidate in manifest["candidates"]:
        assert candidate["prompt_id"].startswith("fake-")
        assert candidate["seconds"] is not None
        assert len(candidate["workflow_sha256"]) == 64
    assert manifest["comfyui"]["version"] == "0.35.0-fake"
    assert len(manifest["workflow_sha256"]) == 64
    # The set's hash is seed-independent; the members' are not.
    assert len({c["workflow_sha256"] for c in manifest["candidates"]}) == 3
    # The lock rows the graph names, for the sidecar: the LoRA is among them.
    assert config.CONCEPT_LORA["file"] in {m["file"] for m in manifest["models"]}
    assert not any("Controlnet" in m["file"] for m in manifest["models"])


def test_render_concept_uploads_the_sketch_once_under_the_job_name(work_dir,
                                                                    workflow_dir,
                                                                    sketch):
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        manifest = concept_stage.concept(
            "knight", description="a knight", count=2, sketch=sketch,
            client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir,
            work_dir=work_dir)
    assert fake.uploads == ["anim_knight_concept_sketch.png"]
    for graph in fake.prompts:
        load = next(n for n in graph.values() if n["class_type"] == "LoadImage")
        assert load["inputs"]["image"] == "anim_knight_concept_sketch.png"
    assert any("Controlnet" in m["file"] for m in manifest["models"])


def test_render_concept_skips_the_existing_candidate(work_dir, workflow_dir,
                                                     reference_sprite):
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        manifest = concept_stage.concept(
            "knight", description="a knight", count=2, existing=reference_sprite,
            client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir,
            work_dir=work_dir)
    assert len(fake.prompts) == 2
    assert manifest["candidates"][0]["source"] == "existing"
    assert manifest["candidates"][0]["prompt_id"] is None
    assert manifest["candidates"][0]["seconds"] is None


def test_concept_against_an_unreachable_service_says_so(work_dir, workflow_dir):
    with pytest.raises(comfy.ComfyUnreachable):
        concept_stage.concept("knight", description="a knight",
                              client=comfy.ComfyClient("http://127.0.0.1:1", timeout=1),
                              workflow_dir=workflow_dir, work_dir=work_dir)


# ---------------------------------------------------------------------------
# The pick
# ---------------------------------------------------------------------------
def _concept_job(work_dir, workflow_dir, **kwargs):
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.concept("knight", description="a knight", count=2, seed=10,
                              client=comfy.ComfyClient(fake.url),
                              workflow_dir=workflow_dir, work_dir=work_dir, **kwargs)
    return Job.open("knight_concept", work_dir)


def test_pick_lands_where_clip_reads_the_base_render(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    result = concept_stage.pick("knight", "01", work_dir=work_dir)
    base_job = Job.open("knight_base", work_dir)
    assert result["base_job"] == "knight_base" and result["replaced"] is None
    manifest = base_job.read_manifest()
    assert manifest["slug"] == "knight" and manifest["action"] == ""
    assert manifest["render"] == {"seed": 11, "passes": ["concept_pass"],
                                  "final": "concept_pick.png"}
    assert manifest["concept"]["candidate"] == 1
    assert (base_job.render_dir / "concept_pick.png").exists()
    # What `clip` does to find the identity image - unchanged code, new answer.
    reference, kind = clip_stage.find_reference(
        Job.open("knight_strike", work_dir), {"slug": "knight"})
    assert kind == "base render"
    assert reference == base_job.render_dir / "concept_pick.png"
    # The base job is laid out at the candidate's size, so `post` slices it whole.
    layout = json.loads(base_job.layout_path.read_text())
    assert layout["cells"][0]["rect"] == [0, 0, *config.CONCEPT_CELL]
    provenance = json.loads(base_job.provenance_path.read_text())
    assert provenance["final"] == "concept_pick.png"
    assert provenance["passes"][0]["workflow"] == "concept_pass.json"
    assert len(provenance["passes"][0]["workflow_sha256"]) == 64
    assert config.CONCEPT_LORA["file"] in {m["file"] for m in provenance["models"]}
    assert provenance["concept"]["prompt"] == config.concept_prompt("a knight")
    concept = Job.open("knight_concept", work_dir).read_manifest()
    assert concept["pick"]["index"] == 1 and concept["pick"]["base_job"] == "knight_base"


def test_pick_lays_down_the_front_skeleton_not_the_side_profile(work_dir,
                                                                workflow_dir):
    """The front base pose feature. A concept base is never rendered from its
    skeleton - the picture is copied in - but `post --parts` prompts SAM from
    it, and what txt2img was asked for is a front view. Prompting from the side
    profile stacks every joint in a ~66px column, which is what cost
    hooded_archer its arms, torso and legs."""
    _concept_job(work_dir, workflow_dir)
    concept_stage.pick("knight", "01", work_dir=work_dir)
    base_job = Job.open("knight_base", work_dir)
    laid = json.loads(base_job.keypoints_path.read_text())
    front = json.loads(base_stage.STAND_POSE_FRONT.read_text())
    assert laid["frames"][0]["keypoints"] == front["frames"][0]["keypoints"]
    side = json.loads(base_stage.STAND_POSE.read_text())
    assert laid["frames"][0]["keypoints"] != side["frames"][0]["keypoints"]


def test_pick_existing_carries_no_generation_record(work_dir, workflow_dir,
                                                    reference_sprite):
    _concept_job(work_dir, workflow_dir, existing=reference_sprite)
    result = concept_stage.pick("knight", "existing", work_dir=work_dir)
    assert result["candidate"]["index"] == 0
    base_job = Job.open("knight_base", work_dir)
    manifest = base_job.read_manifest()
    assert manifest["render"] == {"seed": None, "passes": [], "final": "concept_pick.png"}
    provenance = json.loads(base_job.provenance_path.read_text())
    assert provenance["passes"] == [] and provenance["models"] == []
    assert provenance["concept"]["source"] == "existing"
    assert provenance["concept"]["from"] == str(reference_sprite)
    # `--pick 00` is the same candidate.
    again = concept_stage.pick("knight", "00", work_dir=work_dir)
    assert again["candidate"]["index"] == 0


def test_pick_reports_the_base_render_it_replaces(work_dir, workflow_dir):
    from anim.stages import base as base_stage

    old = base_stage.prepare("knight", work_dir=work_dir)
    old.render_dir.mkdir(parents=True)
    (old.render_dir / "base_sprite.png").write_bytes(canned_png(config.BASE_CELL))
    old.update_manifest(render={"seed": 1, "passes": ["base_sprite"],
                                "final": "base_sprite.png"})
    _concept_job(work_dir, workflow_dir)
    result = concept_stage.pick("knight", "00", work_dir=work_dir)
    assert result["replaced"] == "base_sprite.png"
    assert (old.render_dir / "base_sprite.png").exists(), "the old file stays"
    assert old.read_manifest()["render"]["final"] == "concept_pick.png"


def test_pick_refuses_a_candidate_that_does_not_exist(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    with pytest.raises(ValueError, match="no candidate 07"):
        concept_stage.pick("knight", "07", work_dir=work_dir)
    with pytest.raises(ValueError, match="no existing-art candidate"):
        concept_stage.pick("knight", "existing", work_dir=work_dir)
    with pytest.raises(ValueError, match="candidate number"):
        concept_stage.pick("knight", "best", work_dir=work_dir)


# ---------------------------------------------------------------------------
# The multi-view reference feature: the parts phrase, the views, the view pick
# ---------------------------------------------------------------------------
PARTS = "closed great helm, kite shield, longsword, dark cloak"


def test_pick_stores_the_parts_phrase_and_the_avoid_phrase(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    # No --parts: the whole description, explicit in the block.
    concept_stage.pick("knight", "01", work_dir=work_dir)
    block = Job.open("knight_base", work_dir).read_manifest()["concept"]
    assert block["parts_phrase"] == "a knight" and block["avoid_phrase"] == ""
    assert block["views"] == {}
    assert concept_stage.base_phrases("knight", work_dir) == ("a knight", None)
    # --parts / --avoid with the pick; a trailing stop and doubled spaces are cleaned.
    result = concept_stage.pick("knight", "01", parts=PARTS + ".", avoid="bare  head, uncovered face,",
                                work_dir=work_dir)
    assert result["parts_phrase"] == PARTS and result["avoid_phrase"] == "bare head, uncovered face"
    block = Job.open("knight_base", work_dir).read_manifest()["concept"]
    assert block["parts_phrase"] == PARTS and block["avoid_phrase"] == "bare head, uncovered face"
    provenance = json.loads(Job.open("knight_base", work_dir).provenance_path.read_text())
    assert provenance["concept"]["parts_phrase"] == PARTS
    assert concept_stage.base_phrases("knight", work_dir) == (PARTS, "bare head, uncovered face")
    # --parts alone: the phrase changes, nothing else moves.
    from anim import curation

    before = json.loads((work_dir / "knight_curation.json").read_text())
    block = concept_stage.set_phrases("knight", parts="great helm, cloak", work_dir=work_dir)
    assert block["parts_phrase"] == "great helm, cloak" and block["avoid_phrase"] == "bare head, uncovered face"
    assert json.loads((work_dir / "knight_curation.json").read_text())["gates"] == before["gates"]
    assert curation.read("knight", work_dir)["views"] == {}
    assert concept_stage.base_phrases("orc", work_dir) == (None, None)
    with pytest.raises(FileNotFoundError, match="no base job"):
        concept_stage.set_phrases("orc", parts="tusks", work_dir=work_dir)


def test_the_clip_prompt_reads_the_phrases_off_the_base(work_dir, workflow_dir):
    """`anim clip` composes the parts prompt and the negative from the base's
    block (the matrix's clips inherit it); a base without one is the old
    prompt, and the record says which."""
    _concept_job(work_dir, workflow_dir)
    concept_stage.pick("knight", "01", parts=PARTS, avoid="bare head", work_dir=work_dir)
    job = Job.open("knight_strike", work_dir)
    assert clip_stage.find_phrases(job, {"slug": "knight"}) == (PARTS, "bare head")
    assert clip_stage.find_phrases(job, {"slug": "orc"}) == (None, None)
    assert clip_stage.find_phrases(job, {}) == (None, None)


def _views_ready(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    concept_stage.pick("knight", "01", parts=PARTS, work_dir=work_dir)


def test_views_render_the_base_from_each_side_through_the_edit_path(work_dir, workflow_dir):
    _views_ready(work_dir, workflow_dir)
    png = canned_png(config.CONCEPT_CELL)
    logs: list[str] = []
    with FakeComfy(png=png) as fake:
        record = concept_stage.render_views(
            "knight", views="all", count=2, seed=500, client=comfy.ComfyClient(fake.url),
            workflow_dir=workflow_dir, work_dir=work_dir, log=logs.append)
    assert list(record["views"]) == list(config.CONCEPT_VIEWS) and len(fake.prompts) == 16
    assert fake.uploads == ["anim_knight_concept_view_base.png"], "the base once, under the job's name"
    # The graph is the edit path: Qwen-Image-Edit with the base as image1 and
    # its latent; no ControlNet, no Z-Image, no LoRA.
    classes = _graph_classes(fake.prompts[0])
    assert "NunchakuQwenImageDiTLoader" in classes and "TextEncodeQwenImageEditPlus" in classes
    assert "VAEEncode" in classes and "ZImageFunControlnet" not in classes
    assert "LoraLoaderModelOnly" not in classes and "UNETLoader" not in classes
    graph = fake.prompts[0]
    load = graph[comfy.find_node(graph, "base image")]
    assert load["inputs"]["image"] == "anim_knight_concept_view_base.png"
    positive = graph[comfy.find_node(graph, "positive prompt")]["inputs"]
    assert positive["prompt"] == config.concept_view_prompt("back", PARTS)
    assert "from directly behind" in positive["prompt"] and PARTS in positive["prompt"]
    assert positive["image1"] == [comfy.find_node(graph, "base image"), 0]
    assert graph[comfy.find_node(graph, "seed")]["inputs"]["seed"] == 500
    seeds = [g[comfy.find_node(g, "seed")]["inputs"]["seed"] for g in fake.prompts]
    assert seeds == list(range(500, 516)), "one seed per attempt, in order"
    assert graph[comfy.find_node(graph, "output prefix")]["inputs"]["filename_prefix"] \
        == "anim_pipeline/knight_concept/views/view_pass"
    # The layout: views/<view>_NN.png and views.json.
    out = concept_stage.views_dir("knight", work_dir)
    assert sorted(p.name for p in out.glob("*.png")) == sorted(
        f"{v}_{i:02d}.png" for v in config.CONCEPT_VIEWS for i in range(2))
    assert (out / "back_01.png").read_bytes() == png
    saved = json.loads((out / "views.json").read_text())
    assert saved["schema"] == "anim.concept-views/2" and saved["slug"] == "knight"
    assert saved["seed"] == 500 and saved["count"] == 2 and saved["parts_phrase"] == PARTS
    assert saved["base"]["job"] == "knight_base" and len(saved["base_sha256"]) == 64
    assert saved["workflow"] == "view_pass.json" and len(saved["workflow_sha256"]) == 64
    back = saved["views"]["back"]
    assert back["yaw"] == 90 and back["pick"] is None and back["prompt"] == positive["prompt"]
    assert back["elevation"] == "side" and saved["views"]["top"]["elevation"] == "top"
    assert [c["index"] for c in back["candidates"]] == [0, 1]
    assert back["candidates"][1]["seed"] == 501 and back["candidates"][1]["file"] == "back_01.png"
    assert back["candidates"][0]["source"] == "views" and back["candidates"][0]["parent"] is None
    assert back["candidates"][0]["model"] == config.CONCEPT_MODELS["qwen-edit"]
    assert back["candidates"][0]["loras"] == [], "no Qwen LoRA by default (not on disk)"
    assert "from directly above" in saved["views"]["top"]["prompt"]
    assert len(back["candidates"][0]["sha256"]) == 64 and back["candidates"][0]["seconds"] is not None
    files = {m["file"] for m in saved["models"]}
    assert "svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors" in files
    assert "qwen_2.5_vl_7b_fp8_scaled.safetensors" in files and "qwen_image_vae.safetensors" in files
    assert saved["comfyui"]["version"] == "0.35.0-fake"
    # The page: the base beside each view's candidates.
    page = Path(record["page"])
    html = page.read_text(encoding="utf-8")
    assert page == concept_stage.views_page_dir("knight", work_dir) / "index.html"
    assert (page.parent / "img" / "base.png").exists() and (page.parent / "img" / "front_right_01.png").exists()
    for view in config.CONCEPT_VIEWS:
        assert f'id="{view}"' in html and f'id="{view}_00"' in html
    assert "--pick-view" in html and PARTS in html
    assert sum(1 for line in logs if "concept views:" in line) == 16
    # The default is the core four, and a bad name is refused before the GPU.
    assert config.concept_view_names(None) == ["back", "left", "right", "top"]
    assert config.concept_view_names("right,back") == ["back", "right"]
    with pytest.raises(ValueError, match="unknown view"):
        config.concept_view_names("under")


def test_a_group_keyword_names_the_whole_group():
    """The advanced views sweep feature: `--views advanced` is what the tab's
    sweep button sends, so a script can say exactly what the button says."""
    assert config.concept_view_names("advanced") == list(config.CONCEPT_ADVANCED_VIEWS)
    assert config.concept_view_names("advanced") == ["right", "back_right", "back_left",
                                                    "front_left", "front_right"]
    assert config.concept_view_names("main") == list(config.CONCEPT_MAIN_VIEWS)
    assert config.concept_view_names(["advanced"]) == list(config.CONCEPT_ADVANCED_VIEWS)
    # A group sits in a list beside plain names, and the answer stays in table
    # order with no duplicate when the two overlap.
    assert config.concept_view_names("main,front_right") == ["back", "left", "top", "front_right"]
    assert config.concept_view_names("advanced,right") == list(config.CONCEPT_ADVANCED_VIEWS)
    # `all` still wins, and an unknown name is still refused - now naming the
    # groups as well.
    assert config.concept_view_names("advanced,all") == list(config.CONCEPT_VIEWS)
    with pytest.raises(ValueError, match="a group"):
        config.concept_view_names("sideways")


def test_the_view_estimate_drops_a_cold_first_render():
    """The measured estimate behind the sweep's confirm. A cold run's first
    render is ~190 s while Qwen-Image-Edit loads; taking it into the median
    would treble the number the owner is shown."""
    # The real figures recorded for knight_concept: the 200 s one is the cold
    # first render, so the median is of the five warm ones.
    assert config.concept_view_seconds([200.6, 7.8, 10.3, 7.9, 10.7, 8.0]) == 8.0
    # An even count averages the middle pair.
    assert config.concept_view_seconds([200.6, 10.3, 7.9, 10.7, 8.0]) == 9.2
    # Nothing but cold renders, and nothing at all, both fall back.
    assert config.concept_view_seconds([200.6, 164.3]) == config.CONCEPT_VIEW_SECONDS
    assert config.concept_view_seconds([]) == config.CONCEPT_VIEW_SECONDS
    # An odd count takes the middle one; a zero or a negative is not a render.
    assert config.concept_view_seconds([8.0, 10.0, 12.0]) == 10.0
    assert config.concept_view_seconds([0, -1, 11.0]) == 11.0


def test_views_of_a_second_run_keep_the_others_and_append_to_the_asked(work_dir, workflow_dir):
    """The concept generation feature: a view asked for again gets MORE
    attempts; nothing is deleted, the picks stay, --view-prompt is the
    owner's sentence for that view."""
    _views_ready(work_dir, workflow_dir)
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.render_views("knight", views="back,left", count=1, seed=1,
                                   client=comfy.ComfyClient(fake.url),
                                   workflow_dir=workflow_dir, work_dir=work_dir, log=lambda _: None)
        concept_stage.pick_view("knight", {"back": 0}, work_dir=work_dir)
        concept_stage.render_views("knight", views="left", count=2, seed=9,
                                   view_prompt="the knight from the left, shield arm nearest",
                                   client=comfy.ComfyClient(fake.url),
                                   workflow_dir=workflow_dir, work_dir=work_dir, log=lambda _: None)
    saved = json.loads(concept_stage.views_path("knight", work_dir).read_text())
    assert list(saved["views"]) == ["back", "left"]
    assert saved["views"]["back"]["pick"] == {**saved["views"]["back"]["pick"], "index": 0}, "the back's pick survives"
    left = saved["views"]["left"]
    assert [c["seed"] for c in left["candidates"]] == [2, 9, 10]
    assert [c["index"] for c in left["candidates"]] == [0, 1, 2]
    assert left["candidates"][0]["prompt"] == config.concept_view_prompt("left", PARTS)
    assert left["candidates"][1]["prompt"] == "the knight from the left, shield arm nearest"
    assert left["prompt"] == "the knight from the left, shield arm nearest"
    graph = fake.prompts[-1]
    assert graph[comfy.find_node(graph, "positive prompt")]["inputs"]["prompt"] \
        == "the knight from the left, shield arm nearest"
    assert sorted(p.name for p in concept_stage.views_dir("knight", work_dir).glob("*.png")) \
        == ["back_00.png", "left_00.png", "left_01.png", "left_02.png"]
    with pytest.raises(ValueError, match="exactly one"):
        concept_stage.render_views("knight", views="back,left", view_prompt="x",
                                   work_dir=work_dir, log=lambda _: None)


def test_pick_view_writes_into_the_base_and_the_curation_record(work_dir, workflow_dir):
    from anim import curation

    _views_ready(work_dir, workflow_dir)
    pngs = {}
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.render_views("knight", views="all", count=2, seed=1,
                                   client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir,
                                   work_dir=work_dir, log=lambda _: None)
    out = concept_stage.views_dir("knight", work_dir)
    # Make each candidate distinct so the copy can be checked byte for byte.
    for i, path in enumerate(sorted(out.glob("*.png"))):
        img = Image.new("RGB", (16, 16), (i, i, i))
        img.save(path)
        pngs[path.name] = path.read_bytes()
    assert concept_stage.parse_view_picks(["back=01", "left=0", "right=02"]) == {"back": 1, "left": 0, "right": 2}
    with pytest.raises(ValueError, match="--pick-view takes"):
        concept_stage.parse_view_picks(["under=01"])
    with pytest.raises(ValueError, match="--pick-view takes"):
        concept_stage.parse_view_picks(["back"])
    result = concept_stage.pick_view("knight", {"back": 1, "right": 0}, work_dir=work_dir)
    base_job = Job.open("knight_base", work_dir)
    assert result["base_job"] == "knight_base" and set(result["views"]) == {"back", "right"}
    assert 'id="back_01"' in Path(result["page"]).read_text(encoding="utf-8")
    assert '<figure class="picked" id="back_01">' in Path(result["page"]).read_text(encoding="utf-8")
    assert (base_job.render_dir / "view_back.png").read_bytes() == pngs["back_01.png"]
    assert (base_job.render_dir / "view_right.png").read_bytes() == pngs["right_00.png"]
    block = base_job.read_manifest()["concept"]
    assert list(block["views"]) == ["back", "right"], "table order"
    back = block["views"]["back"]
    assert back["file"] == "view_back.png" and back["candidate"] == 1 and back["source"] == "back_01.png"
    assert back["seed"] == 2 and back["yaw"] == 90 and back["provisional"] is False
    assert len(back["sha256"]) == 64 and back["picked"]
    assert json.loads(base_job.provenance_path.read_text())["concept"]["views"]["right"]["candidate"] == 0
    saved = json.loads(concept_stage.views_path("knight", work_dir).read_text())
    assert saved["views"]["back"]["pick"]["index"] == 1 and saved["views"]["left"]["pick"] is None
    # The curation record: the views step under the base gate, accepted by the pick.
    assert result["curation"] == {"back": "accepted", "right": "accepted"}
    record = curation.read("knight", work_dir)
    assert record["views"] == {"back": "accepted", "right": "accepted"}
    assert record["gates"] == {"base": "accepted"}
    # A provisional pick (made for a rerun) lands pending and says so.
    result = concept_stage.pick_view("knight", {"left": 1}, provisional=True, work_dir=work_dir)
    assert result["views"]["left"]["provisional"] is True and result["curation"]["left"] == "pending"
    assert base_job.read_manifest()["concept"]["views"]["left"]["provisional"] is True
    # The board shows the row with the file's URL and the state.
    board = curation.board("knight", work_dir)
    assert set(board["views"]) == set(config.CONCEPT_VIEWS), "all eight slots on the ring"
    assert {v for v, row in board["views"].items() if not row["missing"]} == {"back", "right", "left"}
    assert board["views"]["back"]["state"] == "accepted" and board["views"]["back"]["provisional"] is False
    assert board["views"]["left"]["state"] == "pending" and board["views"]["left"]["provisional"] is True
    assert board["views"]["back"]["url"].startswith("/jobs/knight_base/frames/render/")
    # Refusals: a candidate that is not there, a view that was not rendered.
    with pytest.raises(ValueError, match="no candidate 07"):
        concept_stage.pick_view("knight", {"back": 7}, work_dir=work_dir)
    with pytest.raises(FileNotFoundError, match="no views yet"):
        concept_stage.pick_view("orc", {"back": 0}, work_dir=work_dir)
    # A new --pick clears the views: the files, the block, the curation step.
    Image.new("RGB", (32, 32), (9, 9, 9)).save(Job.open("knight_concept", work_dir).candidates_dir / "00.png")
    concept_stage.pick("knight", "00", work_dir=work_dir)
    assert not list(base_job.render_dir.glob("view_*.png"))
    assert base_job.read_manifest()["concept"]["views"] == {}
    assert curation.read("knight", work_dir)["views"] == {}
    with pytest.raises(ValueError, match="earlier base"):
        concept_stage.pick_view("knight", {"back": 1}, work_dir=work_dir)


# ---------------------------------------------------------------------------
# The mirrored view feature: a view derived by flipping its accepted partner
# ---------------------------------------------------------------------------
def _asymmetric(path: Path, tint: int = 40) -> None:
    """A picture that is obviously NOT its own mirror: a stripe down the left
    edge only, so a flip moves it to the right edge."""
    img = Image.new("RGB", (16, 16), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([0, 0, 3, 15], fill=(tint, 20, 30))
    img.save(path)


def _mirror_ready(work_dir, workflow_dir, partner: str, *, slug: str = "knight",
                  parts: str = PARTS, accept: bool = True, provisional: bool = False):
    """`slug` with `partner` rendered, made asymmetric and (unless told not to)
    accepted - everything `--mirror-view` of the other half of the pair needs."""
    if slug == "knight":
        _concept_job(work_dir, workflow_dir)
    else:
        with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
            concept_stage.concept(slug, description="a hooded monk", count=1, seed=3,
                                  client=comfy.ComfyClient(fake.url),
                                  workflow_dir=workflow_dir, work_dir=work_dir)
    concept_stage.pick(slug, "00", parts=parts, work_dir=work_dir)
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.render_views(slug, views=partner, count=1, seed=7,
                                   client=comfy.ComfyClient(fake.url),
                                   workflow_dir=workflow_dir, work_dir=work_dir,
                                   log=lambda _: None)
    source = concept_stage.views_dir(slug, work_dir) / f"{partner}_00.png"
    _asymmetric(source)
    if accept:
        concept_stage.pick_view(slug, {partner: 0}, provisional=provisional,
                                work_dir=work_dir)
    return source


def test_the_mirror_pairs_come_off_the_compass_and_leave_the_plane_alone():
    """yaw t flips to 180 - t, so three of the eight views are pairs and the
    three that look down the symmetry plane have no partner."""
    assert config.mirror_of("right") == "left" and config.mirror_of("left") == "right"
    assert config.mirror_of("back_right") == "back_left"
    assert config.mirror_of("back_left") == "back_right"
    assert config.mirror_of("front_right") == "front_left"
    assert config.mirror_of("front_left") == "front_right"
    for lonely in ("back", "top", config.FRONT_VIEW, "nonsense", ""):
        assert config.mirror_of(lonely) is None, lonely
    # Derived from the yaws, so the pairing is exactly "the other camera at
    # 180 - yaw, same pitch", and it is an involution.
    for view, partner in config.CONCEPT_MIRROR_PAIRS.items():
        spec, other = config.CONCEPT_VIEWS[view], config.CONCEPT_VIEWS[partner]
        assert other["yaw"] == (180 - spec["yaw"]) % 360
        assert other["elevation"] == spec["elevation"], "a flip never changes the pitch"
        assert config.CONCEPT_MIRROR_PAIRS[partner] == view
    assert set(config.mirrorable_views()) == set(config.CONCEPT_MIRROR_PAIRS)
    assert len(config.mirrorable_views()) == 6, "three pairs, both ways round"


@pytest.mark.parametrize("view,partner", sorted(config.CONCEPT_MIRROR_PAIRS.items()))
def test_a_mirrored_view_is_its_partner_flipped_in_either_direction(
        view, partner, work_dir, workflow_dir):
    """All three pairs, both ways: the derived view is the partner's own
    pixels reversed, byte for byte, and it lands exactly where `--pick-view`
    would have put it."""
    _mirror_ready(work_dir, workflow_dir, partner)
    base_job = Job.open("knight_base", work_dir)
    concept_stage.mirror_view("knight", view, work_dir=work_dir, log=lambda _: None)
    out = base_job.render_dir / f"view_{view}.png"
    with Image.open(base_job.render_dir / f"view_{partner}.png") as src:
        expected = ImageOps.mirror(src).convert("RGB").tobytes()
    with Image.open(out) as got:
        assert got.convert("RGB").tobytes() == expected
        assert got.convert("RGB").tobytes() != src.convert("RGB").tobytes(), \
            "the fixture is asymmetric, so the flip has to be a different picture"
    assert base_job.read_manifest()["concept"]["views"][view]["file"] == f"view_{view}.png"


def test_a_mirrored_view_records_where_its_pixels_came_from(work_dir, workflow_dir):
    """A flip is as traceable as a GPU pass: the entry says it is mirrored,
    which view it came from, that file and its sha256, and when."""
    source = _mirror_ready(work_dir, workflow_dir, "left")
    base_job = Job.open("knight_base", work_dir)
    result = concept_stage.mirror_view("knight", ["right"], work_dir=work_dir,
                                       log=lambda _: None)
    assert result["base_job"] == "knight_base" and set(result["views"]) == {"right"}
    entry = base_job.read_manifest()["concept"]["views"]["right"]
    assert entry == result["views"]["right"]
    assert entry["mirrored"] is True and entry["provisional"] is False
    assert entry["yaw"] == 0 and entry["file"] == "view_right.png" and entry["picked"]
    # No candidate and no seed of its own - nothing was sampled for it.
    assert entry["candidate"] is None and entry["seed"] is None
    mirror = entry["mirror"]
    assert mirror["view"] == "left" and mirror["file"] == "view_left.png"
    assert mirror["how"] == "flip horizontally" and mirror["at"] == entry["picked"]
    # The source hash is the file on the base, and the entry's own hash is the
    # flipped bytes: the pair reproduces the picture with no seed at all.
    from anim.util import sha256_file

    assert mirror["sha256"] == sha256_file(base_job.render_dir / "view_left.png")
    assert mirror["sha256"] == sha256_file(source), "the accepted copy of the partner"
    assert entry["sha256"] == sha256_file(base_job.render_dir / "view_right.png")
    assert entry["sha256"] != mirror["sha256"] and len(entry["sha256"]) == 64
    assert mirror["candidate"] == 0 and mirror["seed"] == 7
    # The block keeps the table's compass order, and the sidecar agrees.
    assert list(base_job.read_manifest()["concept"]["views"]) == ["left", "right"]
    provenance = json.loads(base_job.provenance_path.read_text())
    assert provenance["concept"]["views"]["right"]["mirror"]["view"] == "left"
    # views.json says the view is accepted with no candidate of its own, so the
    # record of the views and the base cannot disagree about what is accepted.
    saved = json.loads(concept_stage.views_path("knight", work_dir).read_text())
    assert saved["views"]["right"]["pick"] == {
        "index": None, "mirrored": True, "from": "left", "file": "view_left.png",
        "sha256": entry["sha256"], "provisional": False, "picked": entry["picked"]}
    assert list(saved["views"]) == ["left", "right"], "compass order"
    # The curation step, the same one `--pick-view` writes.
    from anim import curation

    assert result["curation"]["right"] == "accepted"
    assert curation.read("knight", work_dir)["views"]["right"] == "accepted"
    # The board's ring and the static page both say it is a flip.
    board = curation.board("knight", work_dir)
    assert board["views"]["right"]["mirrored"] is True
    assert board["views"]["right"]["mirrored_from"] == "left"
    assert board["views"]["right"]["mirror_of"] == "left"
    assert board["views"]["left"]["mirrored"] is False
    assert board["views"]["back"]["mirror_of"] is None
    assert "the left view, flipped" in Path(result["page"]).read_text(encoding="utf-8")


def test_a_view_on_the_symmetry_plane_is_never_mirrorable(work_dir, workflow_dir):
    """back, top and front look straight down the plane: flipping them gives
    the same facing, so the refusal names the six that are pairs."""
    _mirror_ready(work_dir, workflow_dir, "left")
    for lonely in ("back", "top"):
        with pytest.raises(ValueError, match="no mirror partner") as caught:
            concept_stage.mirror_view("knight", lonely, work_dir=work_dir)
        assert "Mirrorable: " in str(caught.value)
        for name in config.mirrorable_views():
            assert name in str(caught.value)
        assert f"--views {lonely}" in str(caught.value), "it says how to render it instead"
    with pytest.raises(ValueError, match="no view 'front'"):
        concept_stage.mirror_view("knight", config.FRONT_VIEW, work_dir=work_dir)
    with pytest.raises(ValueError, match="no view 'under'"):
        concept_stage.mirror_view("knight", "under", work_dir=work_dir)
    with pytest.raises(ValueError, match="at least one view"):
        concept_stage.mirror_view("knight", [], work_dir=work_dir)
    # Nothing was written for any of them.
    assert not (Job.open("knight_base", work_dir).render_dir / "view_back.png").exists()


def test_a_mirror_needs_its_partner_ACCEPTED_not_merely_drawn(work_dir, workflow_dir):
    """The matrix's strictness, not the bare reference walk's: `hooded_archer` is four
    views drawn and none picked, and a mirror must refuse there rather than
    launder an unjudged attempt into an accepted view."""
    _mirror_ready(work_dir, workflow_dir, "left", accept=False)
    base_job = Job.open("knight_base", work_dir)
    with pytest.raises(ValueError, match="not on the base") as caught:
        concept_stage.mirror_view("knight", "right", work_dir=work_dir)
    assert "--views left" in str(caught.value) and "--pick-view left=NN" in str(caught.value)
    assert not (base_job.render_dir / "view_right.png").exists()
    # Picked, but provisionally - the curation step is pending, so it is still
    # a judgement nobody has made.
    concept_stage.pick_view("knight", {"left": 0}, provisional=True, work_dir=work_dir)
    with pytest.raises(ValueError, match="is pending, not accepted") as caught:
        concept_stage.mirror_view("knight", "right", work_dir=work_dir)
    assert "--pick-view left=NN" in str(caught.value)
    # And a rejected one.
    from anim import curation

    curation.set_views("knight", {"left": "rejected"}, work_dir)
    with pytest.raises(ValueError, match="is rejected, not accepted"):
        concept_stage.mirror_view("knight", "right", work_dir=work_dir)
    curation.set_views("knight", {"left": "accepted"}, work_dir)
    assert concept_stage.mirror_view("knight", "right", work_dir=work_dir,
                                     log=lambda _: None)["views"]["right"]
    # A flip of a flip is a round trip: it would only re-derive what is there.
    with pytest.raises(ValueError, match="itself the flip of"):
        concept_stage.mirror_view("knight", "left", work_dir=work_dir)
    # No base at all is the same refusal `--views` gives.
    with pytest.raises(FileNotFoundError, match="no base job"):
        concept_stage.mirror_view("orc", "right", work_dir=work_dir)


def test_the_handedness_warning_fires_once_and_only_for_handed_gear(work_dir, workflow_dir):
    """The one trade-off of a flip is the sword hand, so it is said once per
    invocation - not per view, and not at all for a unit with nothing handed."""
    _mirror_ready(work_dir, workflow_dir, "left")
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.render_views("knight", views="back_left", count=1, seed=11,
                                   client=comfy.ComfyClient(fake.url),
                                   workflow_dir=workflow_dir, work_dir=work_dir,
                                   log=lambda _: None)
    _asymmetric(concept_stage.views_dir("knight", work_dir) / "back_left_00.png", tint=90)
    concept_stage.pick_view("knight", {"back_left": 0}, work_dir=work_dir)
    logs: list[str] = []
    result = concept_stage.mirror_view("knight", ["right", "back_right"],
                                       work_dir=work_dir, log=logs.append)
    assert set(result["views"]) == {"right", "back_right"}
    assert sum(1 for line in logs if "handedness" in line) == 1, "once, not per view"
    assert config.props_for_unit("knight") == ("sword", "shield")
    assert "sword" in result["warning"] and "shield" in result["warning"]
    assert "right" in result["warning"] and "back_right" in result["warning"]
    assert concept_stage.handed_gear("knight", work_dir) == ["sword", "shield"]
    # A unit with no props and a parts phrase that names nothing handed: the
    # flip is the whole story, so there is nothing to warn about.
    _mirror_ready(work_dir, workflow_dir, "left", slug="hooded_monk",
                  parts="deep hood, long robe, rope belt")
    assert config.props_for_unit("hooded_monk") == ()
    assert concept_stage.handed_gear("hooded_monk", work_dir) == []
    quiet: list[str] = []
    result = concept_stage.mirror_view("hooded_monk", "right", work_dir=work_dir,
                                       log=quiet.append)
    assert result["warning"] is None
    assert not [line for line in quiet if "handedness" in line]
    assert [line for line in quiet if "flipped horizontally" in line], "it still says what it did"
    # A unit the props table says nothing about, whose OWN parts phrase names a
    # bow, is warned about the bow: the phrase is the only description there is.
    _mirror_ready(work_dir, workflow_dir, "left", slug="hooded_monk2",
                  parts="deep hood, short bow")
    assert concept_stage.handed_gear("hooded_monk2", work_dir) == ["bow"]
    assert "bow" in concept_stage.mirror_view("hooded_monk2", "right", work_dir=work_dir,
                                              log=lambda _: None)["warning"]


def test_views_need_a_base_and_say_so(work_dir, workflow_dir):
    with FakeComfy() as fake:
        with pytest.raises(FileNotFoundError, match="no base job"):
            concept_stage.render_views("knight", client=comfy.ComfyClient(fake.url),
                                       workflow_dir=workflow_dir, work_dir=work_dir)
    _views_ready(work_dir, workflow_dir)
    with FakeComfy() as fake:
        with pytest.raises(ValueError, match="--count"):
            concept_stage.render_views("knight", count=0, client=comfy.ComfyClient(fake.url),
                                       workflow_dir=workflow_dir, work_dir=work_dir)


def test_the_cli_knows_the_stage():
    from anim.cli import build_parser

    args = build_parser().parse_args(["concept", "--unit", "knight", "--pick", "02"])
    assert args.stage == "concept" and args.pick == "02"
    args = build_parser().parse_args(["concept", "--unit", "knight", "--prompt", "x",
                                      "--existing"])
    assert args.existing == "auto"
    args = build_parser().parse_args(["concept", "--unit", "knight", "--prompt", "x",
                                      "--existing", "art.png", "--count", "6"])
    assert args.existing == "art.png" and args.count == 6
    args = build_parser().parse_args(["concept", "--unit", "knight", "--views"])
    assert args.views == "" and args.pick_view is None
    args = build_parser().parse_args(["concept", "--unit", "knight", "--views", "all", "--count", "2"])
    assert args.views == "all" and args.count == 2
    args = build_parser().parse_args(["concept", "--unit", "knight", "--pick-view", "back=01",
                                      "left=00", "--provisional"])
    assert args.pick_view == ["back=01", "left=00"] and args.provisional is True
    args = build_parser().parse_args(["concept", "--unit", "knight", "--pick", "01",
                                      "--parts", "helm, cloak", "--avoid", "bare head"])
    assert args.parts == "helm, cloak" and args.avoid == "bare head"
    # The mirrored view feature, and the one combination that has no meaning.
    args = build_parser().parse_args(["concept", "--unit", "knight", "--mirror-view",
                                      "right", "front_left"])
    assert args.mirror_view == ["right", "front_left"] and args.pick_view is None
    args = build_parser().parse_args(["concept", "--unit", "knight", "--mirror-view", "right",
                                      "--pick-view", "left=00"])
    with pytest.raises(SystemExit, match="not both"):
        concept_stage.run_stage(args)


# ---------------------------------------------------------------------------
# The concept generation feature: upload, img2img, edit, model + LoRAs
# ---------------------------------------------------------------------------
REGISTRY = Path(__file__).resolve().parent.parent / "gpu" / "models.registry.tsv"


def _lora_nodes(graph: dict) -> list[dict]:
    return [n for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly"]


def test_upload_registers_the_owners_art_without_a_prompt_or_the_gpu(work_dir, reference_sprite):
    row = concept_stage.register_upload("knight", reference_sprite, work_dir=work_dir)
    job = Job.open("knight_concept", work_dir)
    manifest = job.read_manifest()
    assert row["source"] == "upload" and row["index"] == 0 and row["file"] == "00.png"
    assert row["seed"] is None and row["prompt"] is None and len(row["sha256"]) == 64
    assert manifest["candidates"] == [row] and manifest["description"] is None
    kept = Path(row["from"])
    assert kept.parent == job.root / "uploads" and kept.read_bytes() == reference_sprite.read_bytes()
    with Image.open(job.candidates_dir / "00.png") as img:
        assert img.size == tuple(config.CONCEPT_CELL), "fitted to the cell like the existing art"
    # Bytes work too (the API's PUT body), and a non-image is refused.
    row2 = concept_stage.register_upload("knight", reference_sprite.read_bytes(), name="mac.png",
                                         work_dir=work_dir)
    assert row2["index"] == 1 and row2["name"] == "mac.png"
    with pytest.raises(ValueError, match="not an image"):
        concept_stage.register_upload("knight", b"nope", work_dir=work_dir)
    # It can be picked like any candidate: no generation record.
    result = concept_stage.pick("knight", "00", work_dir=work_dir)
    assert result["candidate"]["source"] == "upload"
    provenance = json.loads(Job.open("knight_base", work_dir).provenance_path.read_text())
    assert provenance["passes"] == [] and provenance["concept"]["source"] == "upload"


def test_img2img_redraws_the_parent_at_the_strength_on_the_zimage_stack(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        result = concept_stage.rework(
            "knight", "base:01", op="img2img", strength=0.5, seed=77, count=2,
            client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir, work_dir=work_dir)
    assert result["target"] == "base" and [r["index"] for r in result["attempts"]] == [2, 3]
    rows = result["attempts"]
    assert rows[0]["source"] == "img2img" and rows[0]["parent"] == {"target": "base", "index": 1}
    assert rows[0]["strength"] == 0.5 and rows[0]["seed"] == 77 and rows[1]["seed"] == 78
    assert rows[0]["workflow"] == "img2img_pass.json" and len(rows[0]["workflow_sha256"]) == 64
    manifest = Job.open("knight_concept", work_dir).read_manifest()
    assert rows[0]["prompt"] == manifest["candidates"][1]["prompt"], "the parent's prompt"
    assert [c["file"] for c in manifest["candidates"]] == ["00.png", "01.png", "02.png", "03.png"]
    assert (Job.open("knight_concept", work_dir).candidates_dir / "03.png").exists()
    # The wire: the parent uploaded once under the job's name, the latent from
    # it, denoise = strength, no ControlNet, the same checkpoint + LoRA.
    assert fake.uploads == ["anim_knight_concept_src_base_01.png"]
    graph = fake.prompts[0]
    classes = _graph_classes(graph)
    assert "VAEEncode" in classes and "EmptySD3LatentImage" not in classes
    assert "ZImageFunControlnet" not in classes and "UNETLoader" in classes
    assert graph[comfy.find_node(graph, "source image")]["inputs"]["image"] == "anim_knight_concept_src_base_01.png"
    sampler = graph[comfy.find_node(graph, "seed")]["inputs"]
    assert sampler["denoise"] == 0.5 and sampler["seed"] == 77
    assert sampler["latent_image"] == [comfy.find_node(graph, "source image"), 0] or \
        graph[sampler["latent_image"][0]]["class_type"] == "VAEEncode"
    assert graph[comfy.find_node(graph, "checkpoint")]["inputs"]["unet_name"] == config.CONCEPT_MODELS["zimage"]
    assert [n["inputs"]["lora_name"] for n in _lora_nodes(graph)] == [config.CONCEPT_LORA["file"]]
    assert config.CONCEPT_LORA["file"] in {m["file"] for m in rows[0]["models"]}
    with pytest.raises(ValueError, match="--strength"):
        concept_stage.plan_rework("knight", "base:01", op="img2img", strength=1.5, work_dir=work_dir)
    with pytest.raises(ValueError, match="base:NN"):
        concept_stage.plan_rework("knight", "under:01", op="img2img", work_dir=work_dir)
    with pytest.raises(ValueError, match="no attempt base:09"):
        concept_stage.plan_rework("knight", "base:09", op="img2img", work_dir=work_dir)


def test_edit_sends_the_instruction_through_the_edit_path(work_dir, workflow_dir):
    _views_ready(work_dir, workflow_dir)
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.render_views("knight", views="left", count=1, seed=1,
                                   client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir,
                                   work_dir=work_dir, log=lambda _: None)
        result = concept_stage.rework(
            "knight", "left:00", op="edit", instruction="make the helmet red", seed=5,
            client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir, work_dir=work_dir)
    assert result["target"] == "left"
    row = result["attempts"][0]
    assert row["source"] == "edit" and row["index"] == 1 and row["file"] == "left_01.png"
    assert row["parent"] == {"target": "left", "index": 0} and row["instruction"] == "make the helmet red"
    assert row["prompt"] == config.concept_edit_prompt("make the helmet red", PARTS)
    assert row["prompt"].startswith("make the helmet red. Keep the same pixel art character")
    saved = json.loads(concept_stage.views_path("knight", work_dir).read_text())
    assert [c["index"] for c in saved["views"]["left"]["candidates"]] == [0, 1]
    assert (concept_stage.views_dir("knight", work_dir) / "left_01.png").exists()
    # The wire: the parent (the view attempt, not the base) as image1.
    assert fake.uploads[-1] == "anim_knight_concept_src_left_00.png"
    graph = fake.prompts[-1]
    assert graph[comfy.find_node(graph, "base image")]["inputs"]["image"] == "anim_knight_concept_src_left_00.png"
    assert graph[comfy.find_node(graph, "positive prompt")]["inputs"]["prompt"] == row["prompt"]
    assert "LoraLoaderModelOnly" not in _graph_classes(graph), "the optional LoRA is bypassed"
    assert graph[comfy.find_node(graph, "output prefix")]["inputs"]["filename_prefix"] \
        == "anim_pipeline/knight_concept/edit/view_pass"
    assert Path(result["page"]).exists() and "edit of left:00" in Path(result["page"]).read_text()
    # The edited attempt can be picked as the view like any other.
    concept_stage.pick_view("knight", {"left": 1}, work_dir=work_dir)
    assert Job.open("knight_base", work_dir).read_manifest()["concept"]["views"]["left"]["candidate"] == 1
    with pytest.raises(ValueError, match="--instruction"):
        concept_stage.plan_rework("knight", "left:00", op="edit", instruction="  ", work_dir=work_dir)


def test_img2img_of_a_view_describes_the_side_for_zimage(work_dir, workflow_dir):
    _views_ready(work_dir, workflow_dir)
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.render_views("knight", views="back", count=1, seed=1,
                                   client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir,
                                   work_dir=work_dir, log=lambda _: None)
        result = concept_stage.rework("knight", "back:00", op="img2img",
                                      client=comfy.ComfyClient(fake.url),
                                      workflow_dir=workflow_dir, work_dir=work_dir)
    row = result["attempts"][0]
    assert row["strength"] == config.CONCEPT_IMG2IMG_STRENGTH
    assert row["prompt"] == config.concept_view_img2img_prompt("a knight", "back", PARTS)
    assert "from directly behind" in row["prompt"] and "image1" not in row["prompt"]
    assert "UNETLoader" in _graph_classes(fake.prompts[-1])


def test_model_and_loras_are_chosen_per_call_from_the_registry(work_dir, workflow_dir):
    lora_a = config.CONCEPT_LORA["file"]
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        manifest = concept_stage.concept(
            "knight", description="a knight", count=1, seed=3,
            model="z_image_turbo_int8_convrot.safetensors",
            loras=[f"{lora_a}:0.3"],
            registry_path=REGISTRY, client=comfy.ComfyClient(fake.url),
            workflow_dir=workflow_dir, work_dir=work_dir)
    row = manifest["candidates"][0]
    assert row["model"] == "z_image_turbo_int8_convrot.safetensors"
    assert row["loras"] == [{"file": lora_a, "strength": 0.3}]
    graph = fake.prompts[0]
    assert [(n["inputs"]["lora_name"], n["inputs"]["strength_model"]) for n in _lora_nodes(graph)] \
        == [(lora_a, 0.3)]
    assert manifest["lora"]["strength"] == 0.3, "the manifest's last-used LoRA follows"
    # A checkpoint or LoRA the registry does not offer for Z-Image is refused
    # before anything is posted.
    from anim import models_lock
    with FakeComfy() as fake:
        with pytest.raises(models_lock.RegistryError, match="not a zimage checkpoint"):
            concept_stage.concept("knight", description="a knight", count=1,
                                  model="svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors",
                                  registry_path=REGISTRY, client=comfy.ComfyClient(fake.url),
                                  workflow_dir=workflow_dir, work_dir=work_dir)
        with pytest.raises(models_lock.RegistryError, match="not a zimage LoRA"):
            concept_stage.concept("knight", description="a knight", count=1,
                                  loras=["nope.safetensors:0.5"], registry_path=REGISTRY,
                                  client=comfy.ComfyClient(fake.url),
                                  workflow_dir=workflow_dir, work_dir=work_dir)
        assert fake.prompts == []


def test_two_loras_chain_and_the_unit_default_is_remembered(work_dir, workflow_dir, tmp_path):
    """Stacked LoRAs are chained LoraLoaderModelOnly nodes; a unit default
    (the studio's "save as unit default") applies when a call names none."""
    registry = tmp_path / "models.registry.tsv"
    registry.write_text(REGISTRY.read_text() + "\n"
                        "second_style.safetensors\tzimage\tlora\tSecond style\t-\t0.5\n")
    lora_a = config.CONCEPT_LORA["file"]
    defaults = concept_stage.set_defaults("knight", "zimage", model=None,
                                          loras=[f"{lora_a}:0.6", "second_style.safetensors"],
                                          registry_path=registry, work_dir=work_dir)
    assert defaults == {"zimage": {"model": None, "loras": [
        {"file": lora_a, "strength": 0.6}, {"file": "second_style.safetensors", "strength": 0.5}]}}
    assert Job.open("knight_concept", work_dir).read_manifest()["defaults"] == defaults
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        manifest = concept_stage.concept(
            "knight", description="a knight", count=1, seed=3, registry_path=registry,
            client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir, work_dir=work_dir)
    row = manifest["candidates"][0]
    assert row["loras"] == defaults["zimage"]["loras"]
    graph = fake.prompts[0]
    nodes = _lora_nodes(graph)
    assert [(n["inputs"]["lora_name"], n["inputs"]["strength_model"]) for n in nodes] \
        == [(lora_a, 0.6), ("second_style.safetensors", 0.5)]
    first = comfy.find_node(graph, "Pixel-Art LoRA")
    second = next(k for k, n in graph.items() if n["class_type"] == "LoraLoaderModelOnly" and k != first)
    assert graph[second]["inputs"]["model"] == [first, 0], "chained model -> model"
    consumers = {k: n["class_type"] for k, n in graph.items()
                 if n["inputs"].get("model") in ([first, 0], [second, 0])}
    assert consumers == {second: "LoraLoaderModelOnly",
                         next(k for k, n in graph.items() if n["class_type"] == "ModelSamplingAuraFlow"):
                         "ModelSamplingAuraFlow"}, "what took the LoRA's model now takes the last one's"
    assert graph[next(k for k, n in graph.items() if n["class_type"] == "ModelSamplingAuraFlow")]["inputs"]["model"] == [second, 0]
    assert comfy.find_node(graph, "Pixel-Art LoRA") == first, "one PATCH title, still unique"
    # The explicit flag beats the default; --lora-strength 0 means none.
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.concept("knight", description="a knight", count=1, lora_strength=0,
                              registry_path=registry, client=comfy.ComfyClient(fake.url),
                              workflow_dir=workflow_dir, work_dir=work_dir)
    assert _lora_nodes(fake.prompts[0]) == []
    with pytest.raises(Exception, match="not a zimage LoRA"):
        concept_stage.set_defaults("knight", "zimage", model=None, loras=["ghost.safetensors"],
                                   registry_path=registry, work_dir=work_dir)
    with pytest.raises(ValueError, match="family"):
        concept_stage.set_defaults("knight", "sdxl", model=None, loras=[], work_dir=work_dir)


def test_the_views_wait_behind_the_base_gate(work_dir, workflow_dir):
    """The concept generation feature: no base -> blocked; base picked (which
    accepts the gate) -> free; gate reopened -> blocked; --force runs."""
    from anim import curation
    from anim.cli import build_parser

    ns = build_parser().parse_args(["concept", "--unit", "knight", "--views"])
    assert "no base" in curation.gate_for_namespace("concept", ns, work_dir)
    ns = build_parser().parse_args(["concept", "--unit", "knight", "--prompt", "x"])
    assert curation.gate_for_namespace("concept", ns, work_dir) is None, "the base itself is free"
    _views_ready(work_dir, workflow_dir)
    ns = build_parser().parse_args(["concept", "--unit", "knight", "--views"])
    assert curation.gate_for_namespace("concept", ns, work_dir) is None
    curation.set_gate("knight", "base", "pending", work_dir)
    assert "base gate is pending" in curation.gate_for_namespace("concept", ns, work_dir)
    ns = build_parser().parse_args(["concept", "--unit", "knight", "--edit", "left:00",
                                    "--instruction", "x"])
    assert "base gate is pending" in curation.gate_for_namespace("concept", ns, work_dir)
    ns = build_parser().parse_args(["concept", "--unit", "knight", "--img2img", "base:00"])
    assert curation.gate_for_namespace("concept", ns, work_dir) is None, "a base rework is free"
    assert curation.gate_for_request("concept", ["--unit", "knight", "--views"], work_dir)
    assert curation.gate_for_request("concept", ["--unit", "knight", "--views", "--force"],
                                     work_dir) is None


def test_a_model_the_service_does_not_have_is_refused_before_anything_is_posted(
        work_dir, workflow_dir):
    """The Qwen pixel LoRA is registered and deliberately NOT installed (the
    KNOWN DISCREPANCY in models.lock). ComfyUI would answer 400 on the whole
    graph after the inputs went up, naming a node id; the stage says which
    file, before the first /prompt."""
    _views_ready(work_dir, workflow_dir)
    qwen_lora = "Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors"
    installed = {"UNETLoader": {"unet_name": [config.CONCEPT_MODELS["zimage"]]},
                 "LoraLoaderModelOnly": {"lora_name": [config.CONCEPT_LORA["file"]]},
                 "NunchakuQwenImageDiTLoader": {"model_name": [config.CONCEPT_MODELS["qwen-edit"]]}}
    with FakeComfy(png=canned_png(config.CONCEPT_CELL), object_info=installed) as fake:
        with pytest.raises(comfy.ComfyError, match=f"does not have the LoRA '{qwen_lora}'"):
            concept_stage.render_views("knight", views="back", count=1, seed=1,
                                       loras=[f"{qwen_lora}:0.8"], registry_path=REGISTRY,
                                       client=comfy.ComfyClient(fake.url),
                                       workflow_dir=workflow_dir, work_dir=work_dir,
                                       log=lambda _: None)
        assert fake.prompts == [] and fake.uploads == [], "nothing was sent"
        # The row the run planned is gone: its settings are the ones that failed.
        saved = json.loads(concept_stage.views_path("knight", work_dir).read_text())
        assert saved["views"].get("back", {}).get("candidates") == []
        # The same check on the base's own stack: a checkpoint the service does
        # not have, refused with the line that installs it.
        with pytest.raises(comfy.ComfyError, match="download_models.sh"):
            concept_stage.check_installed(comfy.ComfyClient(fake.url), "zimage",
                                          model="not_here.safetensors", loras=[])
        # What IS installed goes through, and the views land.
        record = concept_stage.render_views("knight", views="back", count=1, seed=2,
                                            registry_path=REGISTRY,
                                            client=comfy.ComfyClient(fake.url),
                                            workflow_dir=workflow_dir, work_dir=work_dir,
                                            log=lambda _: None)
    assert [c["index"] for c in record["views"]["back"]["candidates"]] == [0], "numbering starts clean"
    assert len(fake.prompts) == 1


def test_a_failed_render_leaves_no_phantom_attempts(work_dir, workflow_dir):
    """A run that dies mid-flight keeps what rendered and forgets what it only
    planned - the record never claims an attempt with no file."""
    _concept_job(work_dir, workflow_dir)
    job = Job.open("knight_concept", work_dir)
    before = [c["index"] for c in job.read_manifest()["candidates"]]
    with FakeComfy(png=canned_png(config.CONCEPT_CELL), error_job=True) as fake:
        with pytest.raises(comfy.ComfyError):
            concept_stage.concept("knight", description="a knight", count=2, seed=77,
                                  client=comfy.ComfyClient(fake.url),
                                  workflow_dir=workflow_dir, work_dir=work_dir)
    manifest = job.read_manifest()
    assert [c["index"] for c in manifest["candidates"]] == before, "the planned rows are gone"
    assert manifest["next_index"] == len(before)
    assert sorted(p.name for p in job.candidates_dir.glob("*.png")) == \
        sorted(f"{i:02d}.png" for i in before), "and no stray files"


# ---------------------------------------------------------------------------
# Hiding and deleting attempts
# ---------------------------------------------------------------------------
def test_hide_takes_an_attempt_out_of_the_strips_and_show_brings_it_back(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    job = Job.open("knight_concept", work_dir)
    row = concept_stage.hide_attempt("knight", "base:00", work_dir=work_dir)
    assert row["hidden"] is True
    manifest = job.read_manifest()
    assert manifest["candidates"][0]["hidden"] is True
    assert [c["index"] for c in manifest["candidates"]] == [0, 1], "the row stays in the list"
    assert (job.candidates_dir / "00.png").exists(), "the file stays"
    store = concept_stage.Attempts("knight", "base", work_dir)
    assert [r["index"] for r in store.visible_rows()] == [1]
    concept_stage.hide_attempt("knight", "base:00", hidden=False, work_dir=work_dir)
    assert "hidden" not in job.read_manifest()["candidates"][0]
    with pytest.raises(ValueError, match="no attempt base:09"):
        concept_stage.hide_attempt("knight", "base:09", work_dir=work_dir)


def test_a_hidden_attempt_keeps_its_number_so_the_next_one_does_not_reuse_it(
        work_dir, workflow_dir):
    """The filename comes from the index (`attempt_file`) and the next index is
    max + 1, so a row that left the list would hand its number - and its
    filename - to the next attempt, over a PNG still on disk."""
    _concept_job(work_dir, workflow_dir)
    job = Job.open("knight_concept", work_dir)
    concept_stage.hide_attempt("knight", "base:01", work_dir=work_dir)
    concept_stage.delete_attempt_file("knight", "base:01", work_dir=work_dir)
    assert not (job.candidates_dir / "01.png").exists()
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.concept("knight", description="a knight", count=1, seed=9,
                              client=comfy.ComfyClient(fake.url),
                              workflow_dir=workflow_dir, work_dir=work_dir)
    manifest = job.read_manifest()
    assert [c["index"] for c in manifest["candidates"]] == [0, 1, 2]
    assert manifest["candidates"][2]["file"] == "02.png"
    assert not (job.candidates_dir / "01.png").exists(), "the deleted number was not reused"
    assert sorted(p.name for p in job.candidates_dir.glob("*.png")) == ["00.png", "02.png"]


def test_delete_file_needs_a_hidden_attempt_and_leaves_its_record(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    job = Job.open("knight_concept", work_dir)
    with pytest.raises(ValueError, match="is not hidden"):
        concept_stage.delete_attempt_file("knight", "base:00", work_dir=work_dir)
    assert (job.candidates_dir / "00.png").exists()
    concept_stage.hide_attempt("knight", "base:00", work_dir=work_dir)
    row = concept_stage.delete_attempt_file("knight", "base:00", work_dir=work_dir)
    assert not (job.candidates_dir / "00.png").exists()
    kept = job.read_manifest()["candidates"][0]
    assert kept["file_deleted"] == row["file_deleted"] and kept["hidden"] is True
    # The record of what it was survives its file.
    assert kept["prompt"] and kept["seed"] == 10 and kept["file"] == "00.png"
    assert kept["model"] and kept["loras"] and kept["workflow_sha256"]
    # And it cannot be picked any more - the reason says so.
    with pytest.raises(ValueError, match="was deleted"):
        concept_stage.pick("knight", "00", work_dir=work_dir)


def test_the_accepted_attempt_is_refused_and_a_pick_un_hides(work_dir, workflow_dir):
    _views_ready(work_dir, workflow_dir)     # picks base:01
    with pytest.raises(ValueError, match="accepted one"):
        concept_stage.hide_attempt("knight", "base:01", work_dir=work_dir)
    with pytest.raises(ValueError, match="accepted one"):
        concept_stage.delete_attempt_file("knight", "base:01", work_dir=work_dir)
    # Hiding another, then accepting it, brings it back: accepting is the
    # opposite of hiding.
    concept_stage.hide_attempt("knight", "base:00", work_dir=work_dir)
    concept_stage.pick("knight", "00", work_dir=work_dir)
    manifest = Job.open("knight_concept", work_dir).read_manifest()
    assert "hidden" not in manifest["candidates"][0]
    assert manifest["pick"]["index"] == 0
    # base:01 is free now, and the new pick is the protected one.
    concept_stage.hide_attempt("knight", "base:01", work_dir=work_dir)
    with pytest.raises(ValueError, match="accepted one"):
        concept_stage.hide_attempt("knight", "base:00", work_dir=work_dir)


def test_a_hidden_view_attempt_leaves_the_page_and_its_children_still_name_it(
        work_dir, workflow_dir):
    _views_ready(work_dir, workflow_dir)
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.render_views("knight", views="back", count=2, seed=1,
                                   client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir,
                                   work_dir=work_dir, log=lambda _: None)
        result = concept_stage.rework("knight", "back:00", op="edit", instruction="red helm",
                                      client=comfy.ComfyClient(fake.url),
                                      workflow_dir=workflow_dir, work_dir=work_dir)
    child = result["attempts"][0]
    concept_stage.hide_attempt("knight", "back:00", work_dir=work_dir)
    saved = json.loads(concept_stage.views_path("knight", work_dir).read_text())
    rows = saved["views"]["back"]["candidates"]
    assert [c["index"] for c in rows] == [0, 1, 2] and rows[0]["hidden"] is True
    assert rows[2]["parent"] == {"target": "back", "index": 0}, "the child still names it"
    page = concept_stage.views_page_dir("knight", work_dir) / "index.html"
    html = page.read_text(encoding="utf-8")
    assert 'id="back_00"' not in html and 'id="back_01"' in html, "the printout is what was kept"
    # An accepted view attempt is refused, like the base's.
    concept_stage.pick_view("knight", {"back": 1}, work_dir=work_dir)
    with pytest.raises(ValueError, match="accepted one"):
        concept_stage.hide_attempt("knight", "back:01", work_dir=work_dir)
    assert concept_stage.hide_attempt("knight", f"back:{child['index']:02d}",
                                      work_dir=work_dir)["hidden"] is True


# ---------------------------------------------------------------------------
# The retouch feature: change an attempt where it sits
# ---------------------------------------------------------------------------
def _canvas_png(size, *, erase: tuple[int, int, int, int] | None = None,
                paint: tuple[int, int, int, int] | None = None) -> bytes:
    """An attempt as the editor leaves it: opaque, with a rubbed-out box."""
    from io import BytesIO

    img = Image.new("RGBA", size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], fill=(40, 60, 90, 255))
    if paint:
        draw.rectangle(list(paint), fill=(200, 30, 30, 255))
    if erase:
        draw.rectangle(list(erase), fill=(0, 0, 0, 0))
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def test_replace_writes_over_the_attempt_and_says_so_on_the_record(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    job = Job.open("knight_concept", work_dir)
    before = job.read_manifest()["candidates"][0]
    path = job.candidates_dir / "00.png"
    canvas = _canvas_png(config.CONCEPT_CELL, erase=(20, 20, 60, 60))
    row = concept_stage.replace_attempt("knight", "base:00", canvas, work_dir=work_dir)
    assert row["sha256"] != before["sha256"] and row["edited"]["how"] == "hand"
    assert row["embedded_prompt"] is False
    # The generation record is kept as history, not deleted.
    assert row["seed"] == before["seed"] and row["prompt"] == before["prompt"]
    assert row["model"] == before["model"] and row["index"] == 0 and row["file"] == "00.png"
    with Image.open(path) as saved:
        assert saved.size == tuple(config.CONCEPT_CELL)
        arr = np.asarray(saved.convert("RGBA"))
    # The erased box came back white and opaque - transparency means "mask",
    # not "hole", and every concept image is opaque on white.
    assert arr[..., 3].min() == 255
    assert tuple(arr[30, 30][:3]) == (255, 255, 255)
    assert tuple(arr[400, 400][:3]) == (40, 60, 90), "what was painted is kept"
    assert json.loads(job.manifest_path.read_text())["candidates"][0]["edited"]["how"] == "hand"
    # A different size is a different asset.
    with pytest.raises(ValueError, match="keeps its size"):
        concept_stage.replace_attempt("knight", "base:00", _canvas_png((256, 256)),
                                      work_dir=work_dir)
    with pytest.raises(ValueError, match="not an image"):
        concept_stage.replace_attempt("knight", "base:00", b"nope", work_dir=work_dir)


def test_retouch_refuses_a_hidden_or_deleted_attempt(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    canvas = _canvas_png(config.CONCEPT_CELL, erase=(20, 20, 60, 60))
    concept_stage.hide_attempt("knight", "base:00", work_dir=work_dir)
    with pytest.raises(ValueError, match="is hidden"):
        concept_stage.replace_attempt("knight", "base:00", canvas, work_dir=work_dir)
    concept_stage.delete_attempt_file("knight", "base:00", work_dir=work_dir)
    with pytest.raises(ValueError, match="was deleted"):
        concept_stage.replace_attempt("knight", "base:00", canvas, work_dir=work_dir)
    with pytest.raises(ValueError, match="was deleted"):
        concept_stage.stash_canvas("knight", "base:00", canvas, work_dir=work_dir)


def test_inpaint_redraws_only_what_was_erased(work_dir, workflow_dir):
    """The mask is the canvas's own transparency, and everything outside it is
    pasted back - so a decode that shifts the frame cannot change a pixel the
    owner did not rub out."""
    _concept_job(work_dir, workflow_dir)
    job = Job.open("knight_concept", work_dir)
    canvas = _canvas_png(config.CONCEPT_CELL, erase=(20, 20, 80, 80), paint=(300, 300, 360, 360))
    # What the GPU "draws": a flat red frame, so anything it touches is obvious.
    from io import BytesIO
    red = Image.new("RGB", tuple(config.CONCEPT_CELL), (255, 0, 0))
    buffer = BytesIO()
    red.save(buffer, format="PNG")
    with FakeComfy(png=buffer.getvalue()) as fake:
        row = concept_stage.inpaint_attempt("knight", "base:01", canvas, prompt="a closed helm",
                                            strength=0.9, seed=7, registry_path=REGISTRY,
                                            client=comfy.ComfyClient(fake.url),
                                            workflow_dir=workflow_dir, work_dir=work_dir)
    assert len(fake.prompts) == 1
    graph = fake.prompts[0]
    classes = _graph_classes(graph)
    assert "SetLatentNoiseMask" in classes and "VAEEncode" in classes
    assert "EmptySD3LatentImage" not in classes and "ZImageFunControlnet" not in classes
    mask_node = next(n for n in graph.values() if n["class_type"] == "SetLatentNoiseMask")
    load = comfy.find_node(graph, "source image")
    assert mask_node["inputs"]["mask"] == [load, 1], "the mask is the canvas's own alpha"
    sampler = graph[comfy.find_node(graph, "seed")]["inputs"]
    assert sampler["denoise"] == 0.9 and sampler["seed"] == 7
    assert graph[comfy.find_node(graph, "positive prompt")]["inputs"]["text"] == "a closed helm"
    assert fake.uploads == ["anim_knight_concept_inpaint_base_01.png"]
    # The record: edited in place, re-hashed, the generation history kept.
    assert row["index"] == 1 and row["file"] == "01.png"
    assert row["edited"]["how"] == "inpaint" and row["edited"]["strength"] == 0.9
    assert row["edited"]["seed"] == 7 and row["edited"]["prompt"] == "a closed helm"
    assert row["seed"] == 11, "the seed it was generated from is still there"
    assert len(job.read_manifest()["candidates"]) == 2, "no new attempt"
    # The pixels: red only inside the erased box.
    with Image.open(job.candidates_dir / "01.png") as saved:
        arr = np.asarray(saved.convert("RGB"))
    assert tuple(arr[40, 40]) == (255, 0, 0), "the erased area was redrawn"
    assert tuple(arr[330, 330]) == (200, 30, 30), "what was painted survived"
    assert tuple(arr[600, 600]) == (40, 60, 90), "and everything else is untouched"
    # A canvas with nothing erased has no mask, and says so before the GPU.
    with FakeComfy() as fake:
        with pytest.raises(ValueError, match="nothing is erased"):
            concept_stage.inpaint_attempt("knight", "base:01", _canvas_png(config.CONCEPT_CELL),
                                          client=comfy.ComfyClient(fake.url),
                                          workflow_dir=workflow_dir, work_dir=work_dir)
        assert fake.prompts == []


def test_img2img_in_place_overwrites_the_attempt_instead_of_appending(work_dir, workflow_dir):
    _concept_job(work_dir, workflow_dir)
    job = Job.open("knight_concept", work_dir)
    before = job.read_manifest()["candidates"][1]
    # A PNG the original run did not produce, so the bytes are visibly new.
    from io import BytesIO
    replacement = Image.new("RGB", tuple(config.CONCEPT_CELL), (12, 200, 90))
    buffer = BytesIO()
    replacement.save(buffer, format="PNG")
    png = buffer.getvalue()
    with FakeComfy(png=png) as fake:
        row = concept_stage.rework_in_place("knight", "base:01", op="img2img", strength=0.5,
                                            seed=3, registry_path=REGISTRY,
                                            client=comfy.ComfyClient(fake.url),
                                            workflow_dir=workflow_dir, work_dir=work_dir)
    manifest = job.read_manifest()
    assert [c["index"] for c in manifest["candidates"]] == [0, 1], "no new attempt"
    assert (job.candidates_dir / "01.png").read_bytes() == png
    assert row["edited"]["how"] == "img2img" and row["edited"]["strength"] == 0.5
    assert row["edited"]["seed"] == 3 and row["sha256"] != before["sha256"]
    assert row["prompt"] == before["prompt"] and row["parent"] is None
    assert fake.uploads == ["anim_knight_concept_src_base_01.png"]
    graph = fake.prompts[0]
    assert graph[comfy.find_node(graph, "seed")]["inputs"]["denoise"] == 0.5
    # And an edit in place, on a view attempt, through the edit path.
    concept_stage.pick("knight", "00", parts=PARTS, work_dir=work_dir)
    with FakeComfy(png=png) as fake:
        concept_stage.render_views("knight", views="back", count=1, seed=1,
                                   client=comfy.ComfyClient(fake.url), workflow_dir=workflow_dir,
                                   work_dir=work_dir, log=lambda _: None)
        edited = concept_stage.rework_in_place("knight", "back:00", op="edit",
                                               instruction="make the helmet red", seed=4,
                                               registry_path=REGISTRY,
                                               client=comfy.ComfyClient(fake.url),
                                               workflow_dir=workflow_dir, work_dir=work_dir)
    saved = json.loads(concept_stage.views_path("knight", work_dir).read_text())
    assert [c["index"] for c in saved["views"]["back"]["candidates"]] == [0]
    assert edited["edited"]["how"] == "edit"
    assert edited["edited"]["instruction"] == "make the helmet red"
    assert "TextEncodeQwenImageEditPlus" in _graph_classes(fake.prompts[-1])
