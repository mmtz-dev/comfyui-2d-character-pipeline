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
from PIL import Image, ImageDraw

from fixtures.fakecomfy import FakeComfy, canned_png

from anim import comfy, config
from anim.job import Job
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
    assert {c["source"] for c in manifest["candidates"]} == {"generated"}
    assert manifest["workflow"] == "concept_pass.json"
    assert job.candidates_dir.is_dir()


def test_the_derived_seed_is_stable_and_changes_with_the_ask(work_dir):
    a = concept_stage.prepare("knight", description="a knight", work_dir=work_dir)
    seed_a = a.read_manifest()["seed"]
    b = concept_stage.prepare("knight", description="a knight", work_dir=work_dir)
    assert b.read_manifest()["seed"] == seed_a, "same ask, same set"
    c = concept_stage.prepare("knight", description="a hound", work_dir=work_dir)
    assert c.read_manifest()["seed"] != seed_a
    assert 0 <= seed_a < 2 ** 53


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
    assert "prompt_id" not in manifest["candidates"][0]


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
    block (the turnaround's clips inherit it); a base without one is the old
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
    assert list(record["views"]) == list(config.CONCEPT_VIEWS) and len(fake.prompts) == 14
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
    assert seeds[:4] == [500, 501, 600, 601], "view offset x100, candidate +1"
    assert graph[comfy.find_node(graph, "output prefix")]["inputs"]["filename_prefix"] \
        == "anim_pipeline/knight_concept/views/view_pass"
    # The layout: views/<view>_NN.png and views.json.
    out = concept_stage.views_dir("knight", work_dir)
    assert sorted(p.name for p in out.glob("*.png")) == sorted(
        f"{v}_{i:02d}.png" for v in config.CONCEPT_VIEWS for i in range(2))
    assert (out / "back_01.png").read_bytes() == png
    saved = json.loads((out / "views.json").read_text())
    assert saved["schema"] == "anim.concept-views/1" and saved["slug"] == "knight"
    assert saved["seed"] == 500 and saved["count"] == 2 and saved["parts_phrase"] == PARTS
    assert saved["base"]["job"] == "knight_base" and len(saved["base_sha256"]) == 64
    assert saved["workflow"] == "view_pass.json" and len(saved["workflow_sha256"]) == 64
    back = saved["views"]["back"]
    assert back["yaw"] == 90 and back["pick"] is None and back["prompt"] == positive["prompt"]
    assert [c["index"] for c in back["candidates"]] == [0, 1]
    assert back["candidates"][1]["seed"] == 501 and back["candidates"][1]["file"] == "back_01.png"
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
    assert sum(1 for line in logs if "concept views:" in line) == 14
    # The default is the core three, and a bad name is refused before the GPU.
    assert config.concept_view_names(None) == ["back", "left", "right"]
    assert config.concept_view_names("right,back") == ["back", "right"]
    with pytest.raises(ValueError, match="unknown view"):
        config.concept_view_names("top")


def test_views_of_a_second_run_keep_the_others_and_redo_the_asked(work_dir, workflow_dir):
    _views_ready(work_dir, workflow_dir)
    with FakeComfy(png=canned_png(config.CONCEPT_CELL)) as fake:
        concept_stage.render_views("knight", views="back,left", count=1, seed=1,
                                   client=comfy.ComfyClient(fake.url),
                                   workflow_dir=workflow_dir, work_dir=work_dir, log=lambda _: None)
        concept_stage.pick_view("knight", {"back": 0}, work_dir=work_dir)
        concept_stage.render_views("knight", views="left", count=2, seed=9,
                                   client=comfy.ComfyClient(fake.url),
                                   workflow_dir=workflow_dir, work_dir=work_dir, log=lambda _: None)
    saved = json.loads(concept_stage.views_path("knight", work_dir).read_text())
    assert list(saved["views"]) == ["back", "left"]
    assert saved["views"]["back"]["pick"] == {**saved["views"]["back"]["pick"], "index": 0}, "the back's pick survives"
    assert [c["seed"] for c in saved["views"]["left"]["candidates"]] == [9, 10]
    assert sorted(p.name for p in concept_stage.views_dir("knight", work_dir).glob("*.png")) \
        == ["back_00.png", "left_00.png", "left_01.png"]


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
        concept_stage.parse_view_picks(["top=1"])
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
    assert record["gates"] == {"base": "accepted", "turnaround": "pending"}
    # A provisional pick (made for a rerun) lands pending and says so.
    result = concept_stage.pick_view("knight", {"left": 1}, provisional=True, work_dir=work_dir)
    assert result["views"]["left"]["provisional"] is True and result["curation"]["left"] == "pending"
    assert base_job.read_manifest()["concept"]["views"]["left"]["provisional"] is True
    # The board shows the row with the file's URL and the state.
    board = curation.board("knight", work_dir)
    assert set(board["views"]) == {"back", "right", "left"}
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
