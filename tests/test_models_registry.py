"""The concept generation feature's model registry: gpu/models.registry.tsv
joined to the lock, resolution that refuses what a family cannot load, the
LoRA chaining in the graph, and `GET /models` with and without a probe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fixtures.fakecomfy import FakeComfy

from anim import comfy, config, models_lock
from anim.api.app import create_app

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "gpu" / "models.registry.tsv"
LOCK = REPO / "gpu" / "models.lock"


def test_the_registry_covers_the_lock_and_names_every_concept_default():
    entries = models_lock.load_registry(REGISTRY, LOCK)
    by_file = {e.file: e for e in entries}
    locked = {m.file for m in models_lock.load(LOCK)}
    assert set(by_file) == locked, "every locked weight has a family and a role"
    assert all(e.locked is not None for e in entries)
    for family, file in config.CONCEPT_MODELS.items():
        assert by_file[file].family == family and by_file[file].role == "checkpoint"
    pixel = by_file[config.CONCEPT_LORA["file"]]
    assert pixel.family == "zimage" and pixel.role == "lora"
    assert pixel.trigger == config.CONCEPT_LORA["trigger"]
    assert pixel.default_strength == config.CONCEPT_LORA["strength"]
    assert {e.family for e in entries} == set(models_lock.FAMILIES)
    row = pixel.as_row()
    assert row["locked"] is True and row["license"] == "apache-2.0" and row["repo"].startswith("tarn59/")


def test_a_bad_registry_row_is_refused_and_a_missing_file_is_empty(tmp_path):
    assert models_lock.load_registry(tmp_path / "none.tsv") == []
    bad = tmp_path / "r.tsv"
    bad.write_text("a.safetensors\tsdxl\tlora\tA\t-\t-\n")
    with pytest.raises(models_lock.RegistryError, match="family 'sdxl'"):
        models_lock.load_registry(bad)
    bad.write_text("a.safetensors\tzimage\tbrain\tA\t-\t-\n")
    with pytest.raises(models_lock.RegistryError, match="role 'brain'"):
        models_lock.load_registry(bad)
    bad.write_text("a.safetensors\tzimage\tlora\n")
    with pytest.raises(models_lock.RegistryError, match="6 tab-separated"):
        models_lock.load_registry(bad)
    # An unlocked file is offered, marked unlocked.
    bad.write_text("# a comment\n\nfresh.safetensors\tzimage\tlora\tFresh\tfresh style\t0.4\n")
    (entry,) = models_lock.load_registry(bad, LOCK)
    assert entry.locked is None and entry.trigger == "fresh style" and entry.default_strength == 0.4


def test_resolution_refuses_what_the_family_cannot_load():
    zimage = config.CONCEPT_MODELS["zimage"]
    qwen = config.CONCEPT_MODELS["qwen-edit"]
    assert models_lock.resolve_model("zimage", None, default="d", path=REGISTRY) == "d"
    assert models_lock.resolve_model("zimage", zimage, default="d", path=REGISTRY) == zimage
    with pytest.raises(models_lock.RegistryError, match="not a zimage checkpoint"):
        models_lock.resolve_model("zimage", qwen, default="d", path=REGISTRY)
    with pytest.raises(models_lock.RegistryError, match="not a qwen-edit checkpoint"):
        models_lock.resolve_model("qwen-edit", zimage, default="d", path=REGISTRY)
    lora = config.CONCEPT_LORA["file"]
    assert models_lock.resolve_loras("zimage", [f"{lora}:0.3"], path=REGISTRY) == [
        {"file": lora, "strength": 0.3, "trigger": config.CONCEPT_LORA["trigger"]}]
    assert models_lock.resolve_loras("zimage", [lora], path=REGISTRY)[0]["strength"] == 0.8, \
        "no strength -> the registry's default"
    assert models_lock.resolve_loras("zimage", [{"file": lora, "strength": 0.1}], path=REGISTRY)[0]["strength"] == 0.1
    assert models_lock.resolve_loras("zimage", None, path=REGISTRY) == []
    with pytest.raises(models_lock.RegistryError, match="not a qwen-edit LoRA"):
        models_lock.resolve_loras("qwen-edit", [lora], path=REGISTRY)
    with pytest.raises(models_lock.RegistryError, match="given twice"):
        models_lock.resolve_loras("zimage", [lora, f"{lora}:0.2"], path=REGISTRY)
    with pytest.raises(models_lock.RegistryError, match="FILE\\[:STRENGTH\\]"):
        models_lock.parse_lora_spec("a.safetensors:high")
    assert models_lock.parse_lora_spec("C:weird:0.5") == ("C:weird", 0.5)


def _chain_fixture() -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z.safetensors"},
              "_meta": {"title": "PATCH checkpoint (unet_name)"}},
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": "a", "strength_model": 0.8},
              "_meta": {"title": "PATCH Pixel-Art LoRA (strength_model)"}},
        "3": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["2", 0], "shift": 3},
              "_meta": {"title": "shift"}},
        "4": {"class_type": "KSampler", "inputs": {"model": ["3", 0], "seed": 1}, "_meta": {"title": "PATCH seed"}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}, "_meta": {"title": "out"}},
    }


def test_chain_loras_zero_one_and_three():
    g = _chain_fixture()
    assert comfy.chain_loras(g, "Pixel-Art LoRA", []) == []
    assert "2" not in g and g["3"]["inputs"]["model"] == ["1", 0], "none: bypassed, not strength 0"
    g = _chain_fixture()
    assert comfy.chain_loras(g, "Pixel-Art LoRA", [("x", 0.5)]) == ["2"]
    assert g["2"]["inputs"] == {"model": ["1", 0], "lora_name": "x", "strength_model": 0.5}
    g = _chain_fixture()
    ids = comfy.chain_loras(g, "Pixel-Art LoRA", [{"file": "x", "strength": 0.5}, ("y", 0.4), ("z", 0.3)])
    assert ids == ["2", "6", "7"]
    assert g["6"]["inputs"] == {"model": ["2", 0], "lora_name": "y", "strength_model": 0.4}
    assert g["7"]["inputs"] == {"model": ["6", 0], "lora_name": "z", "strength_model": 0.3}
    assert g["3"]["inputs"]["model"] == ["7", 0], "the consumer takes the last of the chain"
    assert g["6"]["_meta"]["title"].startswith("chained"), "clones are not PATCH points"
    assert comfy.find_node(g, "Pixel-Art LoRA") == "2"
    # A LoRA at strength 0 is dropped from the chain, not sent.
    g = _chain_fixture()
    assert comfy.chain_loras(g, "Pixel-Art LoRA", [("x", 0.0), ("y", 0.4)]) == ["2"]
    assert g["2"]["inputs"]["lora_name"] == "y"
    assert len(comfy.prune_orphans(g)) == 0


def test_patch_checkpoint_finds_the_loader_by_its_title():
    g = _chain_fixture()
    comfy.patch_checkpoint(g, "other.safetensors")
    assert g["1"]["inputs"]["unet_name"] == "other.safetensors"
    g["1"] = {"class_type": "NunchakuQwenImageDiTLoader", "inputs": {"model_name": "q"},
              "_meta": {"title": "PATCH checkpoint (model_name)"}}
    comfy.patch_checkpoint(g, "q2")
    assert g["1"]["inputs"]["model_name"] == "q2"
    g["1"]["inputs"] = {"nothing": 1}
    with pytest.raises(KeyError, match="model-file input"):
        comfy.patch_checkpoint(g, "q3")
    for name in ("concept_pass", "view_pass", "img2img_pass"):
        graph = comfy.load_workflow(name, REPO / "gpu" / "workflows")
        assert comfy.find_node(graph, "checkpoint"), name


def test_get_models_lists_the_registry_and_probes_comfyui(work_dir, tmp_path):
    zimage = config.CONCEPT_MODELS["zimage"]
    lora = config.CONCEPT_LORA["file"]
    info = {"UNETLoader": {"unet_name": [zimage]},
            "LoraLoaderModelOnly": {"lora_name": [lora]},
            "NunchakuQwenImageDiTLoader": {"model_name": []}}
    with FakeComfy(object_info=info) as fake:
        app = create_app(work_dir, web_dist=tmp_path / "no-dist", comfyui_url=fake.url)
        with TestClient(app) as api:
            listing = api.get("/models").json()
            assert listing["probed"] is False and listing["ops"] == config.CONCEPT_FAMILIES
            assert listing["workflow_defaults"] == config.CONCEPT_MODELS
            zi = listing["families"]["zimage"]
            assert [r["file"] for r in zi["checkpoint"]] == [zimage]
            assert zi["lora"][0] == {**zi["lora"][0], "file": lora, "installed": None, "locked": True,
                                     "trigger": config.CONCEPT_LORA["trigger"], "default_strength": 0.8}
            assert listing["families"]["qwen-edit"]["checkpoint"][0]["file"] == config.CONCEPT_MODELS["qwen-edit"]
            assert listing["families"]["wan"]["checkpoint"][0]["file"].startswith("wan2.2")
            probed = api.get("/models?probe=1").json()
            assert probed["probed"] is True
            assert probed["families"]["zimage"]["checkpoint"][0]["installed"] is True
            assert probed["families"]["zimage"]["lora"][0]["installed"] is True
            assert probed["families"]["qwen-edit"]["checkpoint"][0]["installed"] is False
            assert probed["families"]["qwen-edit"]["lora"][0]["installed"] is False, "the Qwen LoRA is not on disk"
            assert probed["families"]["zimage"]["te"][0]["installed"] is None, "no probe for a text encoder"
    # ComfyUI off: the listing still answers, nothing installed is known.
    app = create_app(work_dir, web_dist=tmp_path / "no-dist", comfyui_url="http://127.0.0.1:1")
    with TestClient(app) as api:
        off = api.get("/models?probe=1").json()
        assert off["families"]["zimage"]["checkpoint"][0]["installed"] is False
