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
