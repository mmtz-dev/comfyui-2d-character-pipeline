"""`anim render`: workflow templating and the drive loop, against a fake ComfyUI.

The real service is on the other Docker engine, needs 35 GB of weights and is
off by default. What this stage has to get right is templating the graph,
uploading the images, and reading `/history` - all of which a fake answers for.
"""

from __future__ import annotations

import json

import pytest

from fixtures.fakecomfy import FakeComfy, canned_png

from anim import comfy
from anim.stages import pick as pick_stage
from anim.stages import render as render_stage
from anim.stages import strip as strip_stage


@pytest.fixture
def striped_job(posed_job, reference_sprite):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    strip_stage.strip(posed_job.name, reference=reference_sprite,
                      work_dir=posed_job.root.parent)
    return posed_job


# ---------------------------------------------------------------------------
# Templating
# ---------------------------------------------------------------------------
def test_every_shipped_workflow_declares_its_patch_points(workflow_dir):
    for name in ("pose_pass", "identity_pass", "base_sprite"):
        points = comfy.patch_points(comfy.load_workflow(name, workflow_dir))
        assert points, f"{name}.json has no PATCH titles"
        assert any("seed" in t.lower() for t in points)
        assert any("output prefix" in t.lower() for t in points)


def test_nodes_are_found_by_title_not_by_id(workflow_dir):
    workflow = comfy.load_workflow("pose_pass", workflow_dir)
    node_id = comfy.find_node(workflow, "skeleton strip")
    assert workflow[node_id]["class_type"] == "LoadImage"


def test_an_ambiguous_or_absent_patch_point_is_an_error(workflow_dir):
    workflow = comfy.load_workflow("identity_pass", workflow_dir)
    with pytest.raises(KeyError, match="no node titled"):
        comfy.find_node(workflow, "nonexistent thing")
    with pytest.raises(KeyError, match="matches 3 patch points"):
        comfy.find_node(workflow, "image")


def test_patching_an_input_the_node_does_not_have_is_refused(workflow_dir):
    workflow = comfy.load_workflow("pose_pass", workflow_dir)
    with pytest.raises(KeyError, match="has no input"):
        comfy.patch(workflow, "seed", nonsense=1)


def test_the_workflow_hash_is_stable_and_order_independent(workflow_dir):
    workflow = comfy.load_workflow("pose_pass", workflow_dir)
    first = comfy.workflow_sha256(workflow)
    shuffled = {k: workflow[k] for k in reversed(list(workflow))}
    assert comfy.workflow_sha256(shuffled) == first
    comfy.patch(workflow, "seed", seed=7)
    assert comfy.workflow_sha256(workflow) != first


def test_the_seed_is_derived_from_the_clip_action_and_picks():
    manifest = {"clip_sha256": "a" * 64, "action": "strike", "slug": "knight"}
    seed = render_stage.derive_seed(manifest, [0, 4, 6, 12])
    assert seed == render_stage.derive_seed(manifest, [0, 4, 6, 12])
    assert seed != render_stage.derive_seed(manifest, [0, 4, 6, 11])
    assert seed != render_stage.derive_seed({**manifest, "action": "hit"},
                                            [0, 4, 6, 12])
    assert 0 <= seed < render_stage.SEED_MAX


# ---------------------------------------------------------------------------
# The drive loop
# ---------------------------------------------------------------------------
def test_reachable_is_false_when_nothing_is_listening():
    assert comfy.ComfyClient("http://127.0.0.1:1", timeout=1).reachable() is False


def test_reachable_is_true_against_the_fake():
    with FakeComfy() as fake:
        assert comfy.ComfyClient(fake.url).reachable() is True


def test_render_runs_two_passes_and_uploads_three_images(striped_job, workflow_dir):
    with FakeComfy() as fake:
        provenance = render_stage.render(
            striped_job.name, seed=4242,
            client=comfy.ComfyClient(fake.url),
            workflow_dir=workflow_dir,
            work_dir=striped_job.root.parent)

    assert [p["name"] for p in provenance["passes"]] == ["pose_pass",
                                                         "identity_pass"]
    assert len(fake.prompts) == 2
    # skeleton for the pose pass; reference + pose output for identity (the
    # skeleton is NOT sent to the identity pass - see render() and NOTES.md)
    assert fake.uploads == ["skeleton_strip.png", "reference_512.png",
                            "pose_pass.png"]
    assert (striped_job.render_dir / "pose_pass.png").exists()
    assert (striped_job.render_dir / "identity_pass.png").exists()


def test_the_posted_graph_carries_the_patched_values(striped_job, workflow_dir):
    with FakeComfy() as fake:
        render_stage.render(striped_job.name, seed=4242, strength=0.65,
                            client=comfy.ComfyClient(fake.url),
                            workflow_dir=workflow_dir,
                            work_dir=striped_job.root.parent)
    pose_graph = fake.prompts[0]
    sampler = next(n for n in pose_graph.values()
                   if n["class_type"] == "KSampler")
    assert sampler["inputs"]["seed"] == 4242
    assert sampler["inputs"]["sampler_name"] == "euler", "never ancestral"
    assert sampler["inputs"].get("batch_size", 1) == 1
    latent = next(n for n in pose_graph.values()
                  if n["class_type"] == "EmptySD3LatentImage")
    assert [latent["inputs"]["width"], latent["inputs"]["height"]] == [1216, 320]
    control = next(n for n in pose_graph.values()
                   if n["class_type"] == "ZImageFunControlnet")
    assert control["inputs"]["strength"] == 0.65

    identity_graph = fake.prompts[1]
    loads = {n["_meta"]["title"]: n["inputs"]["image"]
             for n in identity_graph.values() if n["class_type"] == "LoadImage"}
    assert loads["PATCH image1 - reference sprite (64px)"] == "reference_512.png"
    assert loads["PATCH image2 - skeleton strip"] == "pose_pass.png", \
        "the pose pass is image2; the skeleton is not sent (2026-09-15)"
    assert "PATCH image3 - pose-pass strip" not in loads
    encoders = [n for n in identity_graph.values()
                if n["class_type"] == "TextEncodeQwenImageEditPlus"]
    assert encoders and all("image3" not in n["inputs"] for n in encoders)
    image2_id = next(i for i, n in identity_graph.items()
                     if n.get("_meta", {}).get("title", "").startswith("PATCH image2"))
    encode = next(n for n in identity_graph.values() if n["class_type"] == "VAEEncode")
    assert encode["inputs"]["pixels"] == [image2_id, 0], "latent = the pose pass"


def test_the_lora_node_is_bypassed_unless_a_strength_is_asked_for(striped_job,
                                                                  workflow_dir):
    """ComfyUI validates a LoRA filename against the files on disk BEFORE it
    runs the graph, so a missing optional LoRA at strength 0.0 is a 400, not a
    no-op (knight_strike, 2026-09-15). The node must not be in the graph."""
    with FakeComfy() as fake:
        render_stage.render(striped_job.name, client=comfy.ComfyClient(fake.url),
                            workflow_dir=workflow_dir,
                            work_dir=striped_job.root.parent)
    identity_graph = fake.prompts[1]
    classes = [n["class_type"] for n in identity_graph.values()]
    assert "LoraLoaderModelOnly" not in classes
    loader_id = next(i for i, n in identity_graph.items()
                     if n["class_type"] == "NunchakuQwenImageDiTLoader")
    shift = next(n for n in identity_graph.values()
                 if n["class_type"] == "ModelSamplingAuraFlow")
    assert shift["inputs"]["model"] == [loader_id, 0], "consumer rewired past it"

    with FakeComfy() as fake:
        render_stage.render(striped_job.name, lora_strength=0.6,
                            client=comfy.ComfyClient(fake.url),
                            workflow_dir=workflow_dir,
                            work_dir=striped_job.root.parent)
    lora = next(n for n in fake.prompts[1].values()
                if n["class_type"] == "LoraLoaderModelOnly")
    assert lora["inputs"]["strength_model"] == 0.6


def test_provenance_records_what_it_takes_to_remake_the_frame(striped_job,
                                                              workflow_dir):
    with FakeComfy() as fake:
        provenance = render_stage.render(striped_job.name, seed=99,
                                         client=comfy.ComfyClient(fake.url),
                                         workflow_dir=workflow_dir,
                                         work_dir=striped_job.root.parent)
    assert provenance["seed"] == 99
    assert provenance["comfyui"]["version"] == "0.35.0-fake"
    assert "NVIDIA RTX A5000 (fake)" in provenance["comfyui"]["devices"]
    for record in provenance["passes"]:
        assert len(record["workflow_sha256"]) == 64
    assert provenance["final"] == "identity_pass.png"
    assert json.loads(striped_job.provenance_path.read_text()) == provenance


def test_provenance_carries_the_locked_model_hashes(striped_job, workflow_dir,
                                                    tmp_path):
    """The sidecar's promise is "this frame can be re-made" - hashes or nothing."""
    lock = tmp_path / "models.lock"
    lock.write_text(
        "# comment\n"
        "Comfy-Org/z_image_turbo\tdeadbeef\tsplit/x.safetensors\t"
        "diffusion_models\tz_image_turbo_int8_convrot.safetensors\t123\t"
        f"{'c' * 64}\tapache-2.0\n", encoding="utf-8")
    with FakeComfy() as fake:
        provenance = render_stage.render(striped_job.name, seed=1,
                                         client=comfy.ComfyClient(fake.url),
                                         workflow_dir=workflow_dir,
                                         lock_path=lock,
                                         work_dir=striped_job.root.parent)
    models = provenance["models"]
    assert models and models[0]["repo"] == "Comfy-Org/z_image_turbo"
    assert models[0]["sha256"] == "c" * 64
    assert models[0]["license"] == "apache-2.0"


def test_the_embedded_prompt_chunk_is_noticed(striped_job, workflow_dir):
    png = canned_png((1216, 320), prompt_chunk={"1": {"class_type": "Fake"}})
    with FakeComfy(png=png) as fake:
        provenance = render_stage.render(striped_job.name, seed=1,
                                         client=comfy.ComfyClient(fake.url),
                                         workflow_dir=workflow_dir,
                                         work_dir=striped_job.root.parent)
    assert all(p["embedded_prompt"] for p in provenance["passes"])


def test_an_unreachable_service_says_how_to_start_it(striped_job, workflow_dir):
    with pytest.raises(comfy.ComfyUnreachable, match="docker-default"):
        render_stage.render(striped_job.name,
                            client=comfy.ComfyClient("http://127.0.0.1:1",
                                                     timeout=1),
                            workflow_dir=workflow_dir,
                            work_dir=striped_job.root.parent)


def test_a_rejected_workflow_surfaces_comfyuis_own_words(striped_job, workflow_dir):
    with FakeComfy(fail_prompt=True) as fake:
        with pytest.raises(comfy.ComfyError, match="fake rejection"):
            render_stage.render(striped_job.name,
                                client=comfy.ComfyClient(fake.url),
                                workflow_dir=workflow_dir,
                                work_dir=striped_job.root.parent)


def test_an_errored_job_is_raised_not_polled_forever(striped_job, workflow_dir):
    with FakeComfy(error_job=True) as fake:
        with pytest.raises(comfy.ComfyError, match="errored"):
            render_stage.render(striped_job.name,
                                client=comfy.ComfyClient(fake.url),
                                workflow_dir=workflow_dir,
                                work_dir=striped_job.root.parent)


def test_render_without_a_strip_says_to_run_strip(posed_job, workflow_dir):
    with pytest.raises(FileNotFoundError, match="anim strip"):
        render_stage.render(posed_job.name, workflow_dir=workflow_dir,
                            work_dir=posed_job.root.parent)
