# Workflow notes — verification status

Three API-format ComfyUI workflows (the flat `{node_id: {class_type, inputs, _meta}}`
dict that `POST /prompt` accepts). They are **not** UI workflows; drag one into the
editor and you get an ad-hoc auto-arranged graph.

Every `class_type` and every input key below was read from **source**, not guessed.
Sources, pinned at the commits read on 2026-09-13:

| Repo | Ref read |
|---|---|
| `comfyanonymous/ComfyUI` | `master` @ `19e1058f4c445ef74047e77a23f9ca7684c1e4b6` |
| `Comfy-Org/workflow_templates` | `main` @ `aaac56dd5cc5497533d92cbe50edc35ea660e587` (= the `comfyui-workflow-templates==0.11.59` pin in ComfyUI's `requirements.txt`) |
| `nunchaku-tech/ComfyUI-nunchaku` → now redirects to `nunchux-ai/ComfyUI-nunchaku` | `main` @ `b600f879a777fdef1466c362e850f7e8c31bf89a` (v1.2.1) |
| `alexopus/ComfyUI-Image-Saver` | `master` @ `b4b349e553d0f43713fa6914708b70b296392f73` (v1.24.1) |

---

## CONFIRMED — read from the node definition

| `class_type` | Inputs used here | Where it is defined |
|---|---|---|
| `ModelPatchLoader` | `name` | `comfy_extras/nodes_model_patch.py` — the key is **`name`**, not `model_patch_name`. Reads `models/model_patches/`. Its loader explicitly branches on `control_all_x_embedder.2-1.weight` → "alipai z image fun controlnet", i.e. our exact file. |
| `ZImageFunControlnet` | `model`, `model_patch`, `vae`, `strength` (required); `image` (optional) | Same file; subclasses `QwenImageDiffsynthControlnet`, registered in `NODE_CLASS_MAPPINGS`. **There is no `start_percent`/`end_percent`/`control_context_scale` input — `strength` IS the control scale.** The parent's execute branches on `isinstance(model_patch.model, ZImage_Control)` and applies the Z-Image noise-refiner + double-block patches. |
| `ModelSamplingAuraFlow` | `model`, `shift` | `comfy_extras/nodes_model_advanced.py`. Node default is `1.73`; both official Z-Image and Qwen-Edit templates override to **`3`**, which is what we use. |
| `TextEncodeQwenImageEditPlus` | `clip`, `prompt`, `vae`, `image1`, `image2`, `image3` | `comfy_extras/nodes_qwen.py`. Exactly three image slots. Passing `vae` is what emits `reference_latents` on the conditioning — **that is why no `ReferenceLatent` node is needed**. |
| `NunchakuQwenImageDiTLoader` | `model_name`, `cpu_offload`, `num_blocks_on_gpu`, `use_pin_memory` | `nodes/models/qwenimage.py` in the pinned clone at `/mnt/private/AI Data/comfyui-basedir/custom_nodes/`. Takes a **single `.safetensors` file** from `models/diffusion_models/`, not a folder. `cpu_offload` is `auto`/`enable`/`disable`; `auto` enables offload only under 15 GB VRAM. |
| `LoraLoaderModelOnly` | `model`, `lora_name`, `strength_model` | `nodes.py`. Three keys only, no `strength_clip`. |
| `CFGNorm` | `model`, `strength`, `pre_cfg` | Used by the official 2509 template as `[1, false]`. |
| `EmptySD3LatentImage` | `width`, `height`, `batch_size` | `comfy_extras/nodes_sd3.py`. **`step: 16`** on both dimensions — every size in these workflows is divisible by 16. |
| `UNETLoader` | `unet_name`, `weight_dtype` | `nodes.py`. Reads `models/diffusion_models/`. |
| `CLIPLoader` | `clip_name`, `type`, `device` | `nodes.py`. Reads `models/text_encoders/`. |
| `CLIPTextEncode` | `text`, `clip` | `nodes.py`. Note `text` comes first. |
| `ImageScale` | `image`, `upscale_method`, `width`, `height`, `crop` | `nodes.py`. `upscale_method` includes `nearest-exact`. |
| `LoadImage`, `SaveImage`, `VAELoader`, `VAEEncode`, `VAEDecode`, `ConditioningZeroOut`, `KSampler` | as written | `nodes.py` / `comfy_extras`. |

### Two findings worth calling out

**`CLIPLoader.type` has no `z_image` option.** The full type list at this HEAD is
`stable_diffusion, stable_cascade, sd3, stable_audio, mochi, ltxv, pixart, cosmos,
lumina2, wan, hidream, chroma, ace, omnigen2, qwen_image, hunyuan_image, flux2,
ovis, longcat_image, cogvideox, lens, pixeldit, ideogram4, boogu, krea2, joyimage,
mage, minimax, yue2`. Z-Image's text encoder is selected by **auto-detecting the
weights**: `comfy/sd.py` routes `TEModel.QWEN3_4B` to `comfy.text_encoders.z_image`
for any `clip_type` that is not `FLUX`/`FLUX2`. The official template passes
**`lumina2`** (Z-Image is a Lumina-architecture model), so that is what
`pose_pass.json` and `base_sprite.json` pass. Do not "fix" this to `z_image`.

**The zero-strength LoRA slot is a real no-op.** `LoraLoader.load_lora` returns the
model untouched *before* it resolves the file path:

```python
def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
    if strength_model == 0 and strength_clip == 0:
        return (model, clip)
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
```

So `identity_pass.json`'s Pixel-Art LoRA node at `strength_model: 0.0` costs nothing
and does not even require the file to be present. Raise the strength to enable it.

---

## TODO-verify — cannot be settled without a running ComfyUI on the GPU

Each of these is a real open question, not a naming guess. Settle them with
`GET /object_info` and one render, then delete the entry and note it in `DEVLOG.md`.

1. **`UNETLoader` + `z_image_turbo_int8_convrot.safetensors`.** The official
   template loads `z_image_turbo_bf16.safetensors`. The int8 file is an official
   Comfy-Org quant living in the same `split_files/diffusion_models/` folder, so
   core weight-detection is expected to handle it — but this was **not** verified
   against a load. If it fails, the fallbacks in order are: (a) leave
   `weight_dtype: default` and check the startup log for a dtype warning;
   (b) use `NunchakuZImageDiTLoader` (it exists in the pinned ComfyUI-nunchaku at
   `nodes/models/zimage.py`) with an INT4 Z-Image build instead;
   (c) fall back to `z_image_turbo_bf16.safetensors` (12.3 GB, fits in 24 GB alone
   but leaves less headroom).

2. **`qwen_3_4b_fp8_mixed.safetensors` vs the template's `qwen_3_4b.safetensors`.**
   The auto-detect keys off `TEModel.QWEN3_4B`; the fp8-mixed variant should detect
   identically, but it was not loaded. If detection fails, the 8.0 GB
   `qwen_3_4b.safetensors` is the unquantised fallback (add it to
   `relock_models.py`, do not hand-edit `models.lock`).

3. **`ZImageFunControlnet` vs `QwenImageDiffsynthControlnet`.** Both are real and
   registered, and both route Z-Image correctly (the shared execute branches on
   `ZImage_Control`). We use `ZImageFunControlnet` because it is the Z-Image-specific
   node and the plan names it. **But the official *active* template uses
   `QwenImageDiffsynthControlnet`**; `ZImageFunControlnet` appears only in two
   *archived* inpainting templates. The only signature difference is that `image` is
   **required** on the parent and **optional** on the subclass. If
   `ZImageFunControlnet` misbehaves, swapping the `class_type` string (keeping the
   same input keys) is a one-line change.

4. **`sampler_name: euler` is a deliberate deviation.** The official Z-Image-Turbo
   template uses **`res_multistep`** at 8 steps / cfg 1.0 / `simple`. The plan
   specifies Euler (and explicitly not Euler-ancestral) for reproducibility, so
   these workflows use `euler`. Worth an A/B on the fixture skeleton once the
   service is up — `res_multistep` may be visibly better at 8 steps. Whichever wins,
   pin it and record the choice.

5. **Strip cell alignment.** Nothing in the graph *forces* the model to place
   exactly four evenly spaced figures across a 1216×320 canvas. The skeleton strip
   is the only cell-boundary signal. Expect prompt and control-strength tuning here;
   if cells drift, the fallback is one `/prompt` per cell at 304×320 (slower, still
   reproducible, and the CPU `strip` stage already knows the cell geometry).

6. **Alpha on the reference sprite.** `LoadImage` returns IMAGE (RGB) and MASK as
   separate outputs, so the 64×64 sprite's transparency is dropped before
   `ImageScale`. Intended — the VL encoder wants the pixels, and the post-chain
   re-cuts alpha with `rembg`. But it means the sprite's background colour becomes
   visible context; keep fixtures on plain white.

7. **The `nunchaku` wheel.** `NunchakuQwenImageDiTLoader` will be **absent** from
   `/object_info` until the `nunchaku` Python wheel is installed into the container's
   venv — the node pack imports it at module load and does not vendor it. See
   `gpu/README.md` step 5. Until then `identity_pass.json` cannot run at all.

---

## Deliberate design choices

**Why `pose_pass` and `identity_pass` are two separate `/prompt` jobs.** Z-Image
(int8, ~6 GB) plus Qwen-Image-Edit (INT4, ~12 GB) plus two text encoders does not
comfortably fit in 24 GB in one graph. Upstream solved the same problem inside one
graph with its vendored `VRAMUnloadClip`/`VRAMUnloadModel` eviction nodes; splitting
into two jobs is simpler, lets ComfyUI's own model manager evict between them, and
means a failed identity pass can be re-rolled without re-running the pose pass.
That is why `comfyui_vram_helpers` is **not** in `custom_nodes.txt`.

**Why `base_sprite.json` is one stage, not two.** The other two workflows re-skin
onto a reference sprite that already exists. `base_sprite` is for a unit that has
**no** sprite yet, so there is nothing for Qwen-Edit to take identity from — the
prompt and the art bible are the only style input. It is therefore the pose stack
alone, at a single non-square portrait cell (512×768). Once a base sprite exists and
has been nudged by hand, later actions for that unit go through the full two-pass
chain with it as `image1`.

**Why nothing is square.** Qwen-Image-Edit degrades on square aspect ratios
(upstream issue #243). The 4×1 strip is 1216×320 (cell 304×320) and the base cell is
512×768. Keep every canvas non-square.

**Why `SaveImage` and not `"Image Saver"`.** Core `SaveImage` embeds the full prompt
graph in the PNG's `prompt` text chunk, which is what the pipeline's provenance
sidecar reads back, and it has a two-key input surface that cannot drift. The
`"Image Saver"` node (class_type is the literal string `Image Saver`, spaces
included) is installed and available if model/LoRA hash metadata in A1111 format is
wanted later — **but set `download_civitai_data: false` if you use it, because it
defaults to `true` and phones out to civitai.com on every save.**

**No API-format templates exist upstream.** All ~200 files in
`Comfy-Org/workflow_templates` are UI-format, and both graphs we needed are wrapped
in subgraphs, so they cannot be mechanically flattened. These three files are
hand-translated from the templates' node lists and widget orders.

---

## Patch points

Every input a driver is expected to rewrite is marked `PATCH …` in that node's
`_meta.title`. Match on the title, never on node id — upstream's driver does the
same, and for the same reason: re-exporting the JSON can renumber nodes, and
silently swapping the reference sprite for the skeleton would produce a posed
skeleton instead of a posed character.

| Workflow | Patch points |
|---|---|
| `pose_pass.json` | skeleton strip filename, control `strength`, positive prompt, strip width/height, seed, output prefix |
| `identity_pass.json` | image1 reference sprite, image2 skeleton strip, image3 pose-pass output, LoRA strength, positive prompt, seed, output prefix |
| `base_sprite.json` | standing skeleton filename, control `strength`, unit-description prompt, cell size, seed, output prefix |

`LoadImage` filenames resolve against the container's `/basedir/input/`, so the CPU
pipeline must copy its strips there before enqueuing.
