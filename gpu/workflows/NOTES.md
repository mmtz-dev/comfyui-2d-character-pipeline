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

**The zero-strength LoRA slot is NOT a no-op — it is a 400.** `LoraLoader.load_lora`
does return the model untouched at strength 0 before resolving the path, but that code
never runs: ComfyUI validates every combo input against the loader's file list *before*
execution, and a `lora_name` not on disk fails with `value_not_in_list`. Found on the
first identity pass (knight_strike, 2026-09-15). So `anim render` removes the node from
the graph with `comfy.bypass()` unless `--lora-strength` is given, and the file only has
to exist when it is actually asked for.

**The identity pass sends two images, not three.** `image1` = the reference sprite,
`image2` = the pose-pass strip, no `image3`, and `VAEEncode` takes its pixels from
`image2`. Measured 2026-09-15 on knight_strike: with the skeleton as `image2` and the
pose pass as `image3`, Qwen-Image-Edit dropped the strip layout (two zoomed figures,
wrong character); with the pose pass as `image2` alone it kept all four cells in place.
The JSON still carries all three `LoadImage` nodes; `render.py` rewires at run time.
**And the pass is currently unused**: it flattens the poses back toward the reference and
mangles 2-cell strips, so every published strip came from `anim post --input
render/pose_pass.png`. See DEVLOG 2026-09-15.

**The skeleton must be drawn at rtmlib's default 2 px.** Not a workflow key, but the
finding that made the pose pass work at all: `posekit.LINE_WIDTH`. With 5–8 px limbs the
Fun ControlNet (full or lite, any strength) treated the skeleton as no pose. See the
anchor-gate page and DEVLOG 2026-09-15.

**The model patch is the FULL union, not the lite one.**
`Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors` (15 control layers).
The lite build stays installed for A/B and is documented in `relock_models.py`.

---

## Verified on the GPU — 2026-09-15 (was "TODO-verify")

Settled with a running service; the numbers refer to the original list.

1. **`UNETLoader` + `z_image_turbo_int8_convrot.safetensors` — loads.** Startup log:
   `Found quantization metadata version 1` → `Detected mixed precision quantization` →
   `Native ops: convrot_w4a4, int8_tensorwise, asym_w4a8_int8`, `model_type FLOW`, no dtype
   warning. `weight_dtype: default` is right; none of the fallbacks were needed.
2. **`qwen_3_4b_fp8_mixed.safetensors` — detects as the Z-Image text encoder.** Log:
   `Requested to load ZImageTEModel_` (5370 MB). The unquantised fallback is not needed.
3. **`ZImageFunControlnet` vs `QwenImageDiffsynthControlnet` — the same code.** Read from
   the running install: `ZImageFunControlnet` subclasses the other and overrides only
   `INPUT_TYPES` (making `image` optional and adding `inpaint_image`); `diffsynth_controlnet`
   is inherited unchanged. Keeping `ZImageFunControlnet`.
5. **Strip cell alignment — holds once the ControlNet is actually reading the skeleton**
   (2 px lines, full union model, strength 1.0): four evenly spaced figures on 1216×320,
   every time, across strike / hit / death. Two residues: a sword tip that crosses a cell
   boundary lands in the neighbour cell as a floating shard (`imaging.keep_main_blobs`
   drops it in `post`), and a **2-cell 608×320 strip is unreliable** — the model drew three
   figures; `knight_hit` was re-picked at 4. The per-cell fallback was not needed.
6. **Alpha on the reference sprite — as expected.** White fixtures, `rembg` re-cuts. One
   addition: `rembg isnet-anime` returns an *empty* mask for a lying-down body (the death
   strip's cells 3–4), so that job uses `--matte chroma`.
7. **The `nunchaku` wheel — installed by `gpu/userscripts/10-nunchaku.sh` on every start.**
   `nunchaku-1.2.1+cu12.8torch2.11-cp312-cp312-linux_x86_64.whl`, sha256
   `962eaabc7be74299678d83719e3589e97136c91752d61227ef1f55a3b50b32d4`, against torch
   pinned to 2.11.0 by `TORCH_LOCK` in `gpu/compose.yml` (no wheel exists for the torch
   2.13 the image would otherwise install). `NunchakuQwenImageDiTLoader` is in
   `/object_info`.

## Still TODO-verify

4. **`sampler_name: euler` vs `res_multistep`.** Not A/B'd at the gate; euler kept for
   reproducibility. Worth one run on knight_strike with the same seed.

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
