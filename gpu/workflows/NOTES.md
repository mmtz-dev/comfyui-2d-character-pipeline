# Workflow notes — verification status

Six API-format ComfyUI workflows (the flat `{node_id: {class_type, inputs, _meta}}`
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

## `clip_render.json` — the clip render feature (2026-09-17)

Read from ComfyUI 0.35.0 at `db70adbd` (the running install under
`/mnt/private/AI Data/comfyui-run/ComfyUI`) and the official
`video_wan2_2_5B_fun_control` template in `comfyui_workflow_templates_json` 0.1.78
(the pin at that HEAD). The node list, widget values and model files are the
template's; the only structural change is `SaveImage` on the decoded batch in place
of `CreateVideo → SaveVideo`, so the CPU side gets one PNG per frame without a
video decode.

### CONFIRMED — read from the node definition

| `class_type` | Inputs used | Where / what |
|---|---|---|
| `Wan22FunControlToVideo` | `positive`, `negative`, `vae`, `width`, `height`, `length`, `batch_size`, `ref_image`, `control_video` | `comfy_extras/nodes_wan.py`. `length` has **step 4** (4n+1). Branches on `vae.latent_channels == 48` for the Wan 2.2 VAE. `ref_image` becomes `reference_latents` on the conditioning; `control_video` is VAE-encoded into the first `latent_channels` of `concat_latent_image`. **One control input** — pose + depth means one composited video. Both images are resized with `common_upscale(…, "center")`, i.e. **centre-cropped** to the cell aspect: `anim clip` pads the reference to the cell first. `start_image` exists in `execute` but is not in the schema — not usable from the API. |
| `UNETLoader` | `unet_name: wan2.2_fun_control_5B_bf16.safetensors`, `weight_dtype: default` | loads; startup log `model_type FLOW`, no dtype warning. |
| `CLIPLoader` | `clip_name: umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `type: wan` | the template's value. |
| `VAELoader` | `vae_name: wan2.2_vae.safetensors` | the 48-channel Wan 2.2 VAE the 5B/TI2V line needs. |
| `ModelSamplingSD3` | `shift: 8` | template value. |
| `KSampler` | `steps 20`, `cfg 5.0`, `uni_pc`, `simple`, `denoise 1` | template values. cfg 5 means the negative prompt is live (unlike the Z-Image graphs at cfg 1). |
| `LoadVideo` | `file` | `comfy_extras/nodes_video.py`. Lists **only the input root** (`os.listdir(input_dir)` filtered by video MIME); `validate_inputs` checks existence. Uploaded through the same `/upload/image` endpoint as PNGs — `server.py image_upload` writes any bytes — under a job-unique name. |
| `GetVideoComponents` | `video` | outputs `images` (the frame batch), `audio`, `fps`. |
| `SaveImage` | `images`, `filename_prefix` | one file per frame in the batch, `_00001_` … numbering, batch order. |

### Verified on the GPU — 2026-09-17

- **The 5B model runs on the A5000 without quantisation or block swap.** Knight
  `Sword_Attack` (19 rig frames → 21 sent) at 512×768: **20.2 s** of execution; `Hit_Chest`
  (5 frames) **8.8 s**; `Death01` (30 → 33) at 768×768 **48.7 s**. The first job after a
  model load adds ~65 s. All far inside the 5-minute budget.
- **Lossless H.264 4:4:4 (`-qp 0 -pix_fmt yuv444p`) decodes through `LoadVideo`** and the
  frame count comes back exactly (21 → 21 images, 33 → 33).
- **`LoadVideo` reports its input preview under `outputs[…].images` with `type: "input"`.**
  Measured: 22 image entries for a 21-frame clip. `anim clip` keeps only `type == "output"`.
- **A portrait 512×768 cell clips a lying skeleton.** The rig scales once per rig, and a
  lying 1.83-unit body is ~600 px wide; the death skeleton's bbox started at x = 0 and
  the model drew half a knight. Death is rigged at 768×768 (`rig --width 768 --height
  768`). Wan has no square-aspect penalty — that rule is Qwen-Image-Edit's.
- **Timing from the service's own stamps:** `/history` `status.messages` carries
  `execution_start` / `execution_success` timestamps (ms); `clip.json.seconds` is their
  difference, not the poll loop's wall clock.

### TODO-verify

- **Pose + depth composition.** Only pose-only control has been rendered; `--control
  blend` (skeleton drawn over the depth frame) and `--control depth` are wired and
  unit-tested but wait on the rig props feature's depth frames. Whether the model reads
  a white-on-grey skeleton over a depth silhouette as one control type or as noise is
  the open question.
- **Frame rate.** The rig samples at 12 fps and the model was trained at 24 fps (121
  frames); the clip is fed 1:1 at 12 fps so picks index the rig frames directly. Doubling
  every control frame (24 fps, 2× generation time) may give cleaner mid-swing frames.
- **The styly-agents `Wan2-2-pixel-animate` LoRA** (Apache-2.0, on Wan2.2-I2V-A14B, not
  5B) — not applicable to this model; revisit only if A14B becomes the fallback.
- **Prompt.** `config.CLIP_PROMPT` + `CLIP_ACTION_PHRASES` is a first draft; the
  Chinese negative is the template's. Not A/B'd.
- **A14B fallback** (`Wan2.2-Fun-A14B-Control`, two 14B experts, needs GGUF/fp8 +
  block swap under 24 GB). Not locked, not downloaded — only if 5B fails the gate.

---

## `concept_pass.json` — the character concept feature (2026-09-20)

Z-Image-Turbo text-to-image with the pixel-art style LoRA, and an optional sketch through
the Fun ControlNet Union as a **scribble**. Same loaders, sampler settings and text encoder
as `base_sprite.json` (int8 convrot, `lumina2`, AuraFlow shift 3, 8 steps, cfg 1, euler);
what differs is the node between the UNET and the ControlNet, and that the ControlNet is
optional.

### CONFIRMED — read from the node definition / the running install

| `class_type` | Inputs used | Where / what |
|---|---|---|
| `LoraLoaderModelOnly` | `model`, `lora_name`, `strength_model` | `nodes.py`; the three keys `identity_pass.json` already uses. Read from `/object_info` on ComfyUI 0.35.0: `strength_model` is FLOAT −100..100 step 0.01. The file list is the union of `/basedir/models/loras/` and the read-only shared `lora/` folder — `pixel_art_style_z_image_turbo.safetensors` appeared in it the moment `download_models.sh --only pixel_art_style` wrote the file (ComfyUI's folder cache is mtime-based), **no restart needed**. |
| `ZImageFunControlnet` | `model`, `model_patch`, `vae`, `strength`, `image` | as in `base_sprite.json`, with `model` fed from the LoRA node so the style is patched in before the control layers. |
| `EmptySD3LatentImage` | `768 × 768` | square. The square-aspect rule is Qwen-Image-Edit's (issue #243); this pass never reaches that model, and nothing downstream reads the size — the pick is trimmed to its opaque box by `post`. |

### Verified on the GPU — 2026-09-20

- **The LoRA loads on the int8 convrot UNET and styles the output.** Four knights at 768², 8
  steps: **109.9 s** for the first (Z-Image + Qwen3-4B paging in), then **8.7 / 8.8 / 8.9 s**
  each. `/system_stats` after the run: 12.7 GB of 25.3 GB VRAM in use.
- **The Union model reads a sketch as a scribble with no mode input.** The model card lists
  Canny / Depth / Pose / MLSD / HED / Scribble / Gray for the 2602 build and the node has only
  `image` + `strength`, so the condition is chosen by what the image looks like. `anim concept`
  sends the sketch as **white strokes on black** (inverting a dark-on-light scan), strength
  0.6: the pose, sword hand and shield arm followed the lines across two seeds (10.9 s/image).
  Canny was the alternative — it would mean running an edge detector on a drawing that is
  already lines, and the scribble condition is the one trained for exactly that input.
- **Without a sketch the ControlNet is not in the graph.** `anim concept` bypasses the apply
  node and then `comfy.prune_orphans` drops the now-unreferenced `ModelPatchLoader` and
  `LoadImage`, so the provenance hash and `models_lock.for_workflow` describe the graph the
  image actually came from (no control model listed, no never-uploaded filename validated).

### Patch points

sketch filename (`LoadImage`), control `strength`, LoRA `lora_name` + `strength_model`,
positive prompt, cell size, seed, output prefix — see the table below.

---

## `view_pass.json` — the multi-view reference feature (2026-09-22)

Qwen-Image-Edit 2509 (the INT4 Nunchaku build `identity_pass.json` loads) asked for the
accepted base **from another side**: `anim concept --views` sends the base as `image1` and a
per-view sentence (`config.CONCEPT_VIEWS`, "seen from directly behind, turned all the way
round …") with the unit's parts phrase, and the turnaround hands each direction the view that
already shows that side. One image in, one out; no LoRA node (the Qwen pixel LoRA is not on
disk and `identity_pass.json` bypasses it at 0.0 anyway), no ControlNet.

### CONFIRMED — the same nodes as `identity_pass.json`, read from `/object_info` on 0.35.0

| `class_type` | Inputs used | Where / what |
|---|---|---|
| `NunchakuQwenImageDiTLoader`, `ModelSamplingAuraFlow` (shift 3), `CFGNorm`, `CLIPLoader` (`qwen_image`), `VAELoader` | as in `identity_pass.json` | unchanged. |
| `TextEncodeQwenImageEditPlus` | `clip`, `prompt`, `vae`, `image1` | `image2` / `image3` are optional in the schema and left out — one image, so the encoder emits one `reference_latents`. |
| `VAEEncode` | `pixels` = the base | the latent's SIZE is the output's size: a 768² base gives a 768² view, the same scale `clip.fit_reference` letterboxes the base to, so a view is a drop-in for the base. `denoise: 1.0`, so its content is not. |
| `KSampler` | `steps 4`, `cfg 1.0`, euler / simple | the lightning build's settings. |

### Verified on the GPU — 2026-09-22

See DEVLOG 2026-09-22 for the seconds and what the views look like. **The square rule is
knowingly bent here**: the base is 768² (the concept pass's cell), so the view is too; the
alternative is padding the base to a 3:4 canvas first, which would have the model draw the
figure at a scale of its own choosing and break the drop-in. If the views degrade in a way
the aspect explains, that pad is the one-line change in `concept.render_views`.

### Patch points

base image (`LoadImage`, image1), positive prompt (`TextEncodeQwenImageEditPlus.prompt`),
seed, output prefix; since the concept generation feature also `PATCH checkpoint (model_name)`
on the DiT loader and `PATCH optional LoRA (0.0 = bypassed)` - a `LoraLoaderModelOnly` between
the loader and `ModelSamplingAuraFlow`, the same node `identity_pass.json` has, bypassed by
`anim concept` unless `--lora` names a qwen-edit LoRA from `gpu/models.registry.tsv`. The
stage also reuses this graph for an instruction edit (`anim concept --edit REF --instruction`):
the parent attempt as image1 and `config.CONCEPT_EDIT_PROMPT` as the prompt.

---

## `img2img_pass.json` — the concept generation feature (2026-09-22)

Classic img2img on the Z-Image stack: `anim concept --img2img REF --strength S` redraws one
attempt (a base candidate or a view attempt) keeping its composition. `concept_pass.json`
minus the ControlNet, with the latent coming from the source image instead of an empty one.

| `class_type` | Inputs used | Where / what |
|---|---|---|
| `UNETLoader` (`PATCH checkpoint (unet_name)`), `CLIPLoader` (`lumina2`), `VAELoader`, `LoraLoaderModelOnly` (`PATCH Pixel-Art LoRA`), `ModelSamplingAuraFlow` shift 3, `CLIPTextEncode`, `ConditioningZeroOut`, `VAEDecode`, `SaveImage` | as in `concept_pass.json` | unchanged. |
| `LoadImage` (`PATCH source image`) → `VAEEncode` | `pixels` = the attempt being reworked | the latent's size is the output's size, so a 768² attempt gives a 768² attempt - a drop-in for the one it reworks. The source is on the 16-px grid already (every attempt is the concept cell). |
| `KSampler` (`PATCH seed - KSampler 8-step (denoise = strength)`) | `seed`, `denoise` = `--strength` | 8 steps, cfg 1, euler / simple as the turbo build wants; `denoise` is how much of the source is redrawn. `config.CONCEPT_IMG2IMG_STRENGTH` = 0.55; the studio's slider runs 0.3-0.9. |

**TODO-verify on the GPU**: Z-Image-Turbo at 8 steps with `denoise < 1` (the turbo schedule
is distilled for a full pass; a partial one may want fewer steps - `steps * strength` is the
usual rule and would be a one-line patch on the same node). The graph validates against
`/object_info` (every node and key is one `concept_pass.json` already sends); what is
unverified is the picture.

### Patch points

source image (`LoadImage`), positive prompt, seed + `denoise` (one node), checkpoint, Pixel-Art
LoRA `lora_name` + `strength_model` (node removed with no LoRA, cloned and chained with more
than one - `anim.comfy.chain_loras`), output prefix.

---

## `inpaint_pass.json` — the retouch feature (2026-09-22)

`img2img_pass.json` with one node added: the source is the attempt **as the editor left it**
(RGBA, what the owner painted opaque and what they rubbed out transparent), and core
`SetLatentNoiseMask` marks the erased region as the only part the sampler may denoise.

| `class_type` | Inputs used | Where / what |
|---|---|---|
| `LoadImage` (`PATCH source image`) | `image` | **two outputs**: `IMAGE` is the canvas's colours (the CPU flattens the holes onto white before upload, so the encoder never sees black), `MASK` is `1 - alpha` — exactly the erased region. One upload carries both. |
| `VAEEncode` → `SetLatentNoiseMask` | `samples`, `mask` | core ComfyUI (`nodes.py`, latent/inpaint); the unmasked latent is kept, the masked part is noised to `denoise`. No custom node, no inpainting checkpoint: Z-Image-Turbo is used as it is. |
| `KSampler` (`PATCH seed`) | `denoise` = strength | `config.CONCEPT_INPAINT_STRENGTH` 0.9 — a masked region is meant to be redrawn, not nudged. |

**The unmasked pixels are guaranteed on the CPU, not by the graph**: a VAE decode re-renders
the whole frame, so `concept.inpaint_attempt` pastes the model's output back through the mask
and everything outside it stays byte-identical to the canvas. That is the promise the feature
makes, and it does not depend on the sampler honouring the mask perfectly.

**TODO-verify on the GPU**: the turbo schedule at 8 steps with `denoise < 1` (the same open
question as `img2img_pass.json`), and whether a hard mask edge needs feathering — if seams
show, a `MaskBlur`-equivalent (or a Pillow blur of the mask before upload) is the one-line fix.

### Patch points

source image (`LoadImage`, RGBA — its alpha is the mask), positive prompt, seed + `denoise`,
checkpoint, Pixel-Art LoRA, output prefix.

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
| `clip_render.json` | reference image filename, control video filename, clip size (width/height/length), positive prompt, negative prompt, seed (+ steps/cfg on the same node), output prefix |
| `concept_pass.json` | checkpoint (`unet_name`), sketch filename (scribble control; node removed when there is no sketch), control `strength`, Pixel-Art LoRA `lora_name` + `strength_model` (node removed with no LoRA, chained with several), positive prompt, cell size (square), seed, output prefix |
| `img2img_pass.json` | checkpoint (`unet_name`), source image, Pixel-Art LoRA (as above), positive prompt, seed + `denoise`, output prefix |
| `inpaint_pass.json` | the same, where the source image is RGBA and its alpha is the noise mask (`SetLatentNoiseMask`) |
| `view_pass.json` | checkpoint (`model_name`), optional LoRA (bypassed at none), base image (image1, also the latent), positive prompt (the view sentence + parts phrase, or an edit instruction), seed, output prefix |

The checkpoint loaders are titled `PATCH checkpoint (...)` and never `PATCH model ...`:
`anim.comfy.find_node` matches a substring of the title, and the LoRA node's title already
contains `strength_model`.

`LoadImage` filenames resolve against the container's `/basedir/input/`, so the CPU
pipeline must copy its strips there before enqueuing.
