# How does LVCD keep a character's colours from one reference, and what can our sprite pipeline learn from it?

Follow-up to `2026-09-25-keyframe-first-sprite-animation.md`, which already covers LVCD's licence status and the in-betweening model landscape. This file is about technique only. Licence is out of scope here.

## Short answer

LVCD fine-tunes Stable Video Diffusion (SVD, 14 frames, 576×320). It adds a lineart ControlNet and a "Reference Attention" layer that lets every frame's spatial self-attention look anywhere in a colour reference frame, not just at the same pixel position. That global lookup is what carries colours through large motion. The lesson for us is to split "shape" (per-frame lineart or silhouette) from "colour" (one reference). The split only helps if the reference is matched to each frame by content, not by pixel position, and LVCD assumes a same-scene reference, so a turnaround-sheet reference is closer to AniDoc's setting than to LVCD's.

## Findings

### What LVCD is
- The paper is "LVCD: Reference-based Lineart Video Colorization with Diffusion Models" by Zhitong Huang, Mohan Zhang and Jing Liao, published in ACM TOG / SIGGRAPH Asia 2024 (arXiv 2409.12960). [arXiv abs](https://arxiv.org/abs/2409.12960), [ACM DL](https://dl.acm.org/doi/10.1145/3687910)
- It makes three contributions: a Sketch-guided ControlNet, Reference Attention, and a sequential sampling scheme (Overlapped Blending Module plus Prev-Reference Attention) for videos longer than the model's window. [arXiv abs](https://arxiv.org/abs/2409.12960), [project page](https://luckyhzt.github.io/lvcd)

### Base model, resolution, compute
- The base model is SVD img2vid, which generates fixed 14-frame sequences. SVD was pretrained at 576×320 and later fine-tuned at 1024×576. LVCD trains at 576×320 "due to limited computing resources" (Sec. 3). [arXiv HTML](https://arxiv.org/html/2409.12960)
- The repo loads `stabilityai/stable-video-diffusion-img2vid` plus LVCD weights from `luckyhzt/lvcd_pretrained_models`. [GitHub repo](https://github.com/luckyhzt/LVCD)
- The inference notebook uses resolution `[320, 576]`, 19 frames per window, 25 steps, overlap 4, `motion_bucket_id` 160 and `fps_id` 6. [sample.ipynb](https://github.com/luckyhzt/LVCD/blob/main/inference/sample.ipynb)
- Training data was 6,472 clips (average 58 frames) from six Miyazaki films, with lineart extracted by the method of Chan et al. 2022. Training ran for 18k steps at batch size 32 and lr 5e-5, taking 80 hours on 4× RTX A6000 (Sec. 5). [arXiv HTML](https://arxiv.org/html/2409.12960)
- Inference uses 25 Euler steps at about 2 s per frame on one RTX A6000 (Sec. 5). [arXiv HTML](https://arxiv.org/html/2409.12960)
- VRAM: the author says inference "normally … only requires around 20GB GPU memory". One user hit OOM on a 40 GB A100 (Colab, modified env) but ran fine on an 80 GB A100. The author blamed older PyTorch attention memory use. [issue #7](https://github.com/luckyhzt/LVCD/issues/7)
- In the config, only `TemporalAttention_Masked` and `ReferenceAttention` layers are trainable in the main UNet. The spatial self-attention type is swapped for `models.layers.ReferenceAttention`, and the ControlNet takes a 3-channel hint. [configs/lvcd.yaml](https://github.com/luckyhzt/LVCD/blob/main/configs/lvcd.yaml)

### Sketch-guided ControlNet
- The ControlNet clones the whole SVD UNet encoder, including temporal attention and 3D conv layers. Zero-initialised conv layers encode the lineart, which is concatenated to the cloned encoder's input. ControlNet outputs are added to the UNet decoder's skip connections (Sec. 4.1). [arXiv HTML](https://arxiv.org/html/2409.12960)
- Asked why the sketch is concatenated with the noised and reference latents, the author says the ControlNet copies the UNet encoder, so its inputs must match SVD's original inputs (noised latents plus reference frame). [issue #10](https://github.com/luckyhzt/LVCD/issues/10)

### Reference Attention: how one reference keeps colours identical
- **Why stock SVD fails on large motion.** SVD concatenates the reference latent with each noised latent along the channel axis. This "spatial alignment of the two inputs prevents effective long-range matching". SVD's temporal layers are 1-D and only relate "frames at the same spatial position" (Sec. 4.1). [arXiv HTML](https://arxiv.org/html/2409.12960)
- **The fix.** The reference input `[x0, xT0]` goes through a separate "reference path" with all temporal layers skipped. Its intermediate features are injected into each video frame's spatial self-attention as extra keys and values: `Attn(Qi,Ki,Vi) → Attn(Qi,[Ki,K'0],[Vi,V'0])`. This gives global attention from every frame position to every reference position (Sec. 4.1). [arXiv HTML](https://arxiv.org/html/2409.12960)
- **Training.** The reference is drawn at random from a candidate set, so the distance between reference and target frame varies. It is always from the same clip (Sec. 4.1/5). [arXiv HTML](https://arxiv.org/html/2409.12960)
- **At inference the reference is the coloured first frame of the shot.** The notebook encodes `inp.imgs[0]` as `first_cond`, and every frame's lineart is extracted from the ground-truth colour frames with `LineartDetector`. [sample_func.py](https://github.com/luckyhzt/LVCD/blob/main/inference/sample_func.py), [sample.ipynb](https://github.com/luckyhzt/LVCD/blob/main/inference/sample.ipynb), [issue #7](https://github.com/luckyhzt/LVCD/issues/7)
- **Ablation.** Removing Reference Attention worsens every metric (e.g. FVD 40.27→41.29, temporal consistency 1.08→1.16). Visually it gives "inconsistent and incorrect coloring of areas with significant motion" (Sec. 6.4). [arXiv HTML](https://arxiv.org/html/2409.12960)
- **Baselines.** LVCD beats ACOF, TCVC, ControlNet + Reference-only, EISAI and SEINE. For example, FID is 8.84 against 16.93 for ControlNet on the "similar" test set (Sec. 6.2). [arXiv HTML](https://arxiv.org/html/2409.12960)

### Long videos
- Segments overlap by o=4 frames. The Overlapped Blending Module replaces the overlapped frames' intermediate denoising results with those from the previous segment at every timestep. Prev-Reference Attention lets new frames attend to the shifted previous-segment frames as well as the reference: `Attn(Qi,[K(i−3),K'0],[V(i−3),V'0])` (Sec. 4.2). [arXiv HTML](https://arxiv.org/html/2409.12960)
- In code, later segments are conditioned on `[first_cond, prev_conds (last `overlap` frames of the previous x0), first_conds]`. Reference-attention weight is amplified 10× on the overlapped frames. [sample_func.py](https://github.com/luckyhzt/LVCD/blob/main/inference/sample_func.py)

### Limitations the authors state
- The VAE causes artifacts in tiny details. Partially visible new objects are coloured wrongly until the "character's full body is visible" (Sec. 6.7). [arXiv HTML](https://arxiv.org/html/2409.12960)

### Open code and ComfyUI port
- The official repo has inference (a Jupyter notebook) and training code (`python main.py --train --base configs/lvcd.yaml`). It has 200 stars and was last pushed 2025-01-06. [GitHub repo](https://github.com/luckyhzt/LVCD), [GitHub API](https://api.github.com/repos/luckyhzt/LVCD)
- **ComfyUI port:** `kijai/ComfyUI-LVCDWrapper` (Apache-2.0, 60 stars, last pushed 2024-09-30, not archived).
  - It needs an SVD checkpoint; the README says it "seems to work best with the original one" but also runs with 1.1 and XT.
  - It auto-downloads `Kijai/LVCD-pruned/lvcd-fp16.safetensors` to `ComfyUI/models/lvcd`.
  - [ComfyUI-LVCDWrapper](https://github.com/kijai/ComfyUI-LVCDWrapper), [GitHub API](https://api.github.com/repos/kijai/ComfyUI-LVCDWrapper)
- The wrapper has three nodes: `LoadLVCDModel`, `LVCDSampler` and `LVCDDecoder`.
  - `LVCDSampler` takes `ref_images` and `sketch_images` (IMAGE batches).
  - Its other inputs are `num_frames` (default 19, min 15), `overlap` (4), `prev_attn_steps`, `motion_bucket_id` and `fps_id`.
  - It hard-sets `ref_mode='prevref'`.
  - [nodes.py](https://github.com/kijai/ComfyUI-LVCDWrapper/blob/main/nodes.py)

### Closest relative that tolerates a pose-mismatched reference
- AniDoc, also video-diffusion lineart colourisation, colours sketch sequences "following the reference character specification". It uses "correspondence matching as an explicit guidance" for "robustness to the variations (e.g., posture) between the reference character and each line art frame". [AniDoc arXiv](https://arxiv.org/abs/2412.14173), [ComfyUI-AniDoc](https://github.com/LucipherDev/ComfyUI-AniDoc)

### How Wan in ComfyUI already sees a reference (for comparison)
- Wan's DiT uses "a full spatio-temporal attention mechanism" (Sec. 4.2.1). Unlike SVD's factorised same-position temporal layers, every token can already attend to every other token. [Wan arXiv HTML](https://arxiv.org/html/2503.20314)
- In ComfyUI, `WanVaceToVideo` VAE-encodes `reference_image` and prepends it along the time axis of the control latent, then trims it after sampling. The reference is an extra frame inside that global attention, which is structurally close to LVCD's extra K/V. [nodes_wan.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_wan.py)
- `Wan22FunControlToVideo` passes `ref_image` as `reference_latents` conditioning. `WanImageToVideo`, `WanFunControlToVideo` and `WanFirstLastFrameToVideo` only get a CLIP-vision embedding, plus channel-concatenated start/end frames. That is the SVD-style, position-aligned path LVCD identifies as weak under large motion. [nodes_wan.py](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_wan.py)

## Application to our problem

This section is our reasoning, built on the facts above. Each point names the fact it rests on.

1. **Diagnose which reference path the drifting workflow uses.**
   - If costume identity reaches Wan only through a start frame (channel concat) or a CLIP-vision embedding, that is the position-aligned or global-summary conditioning LVCD shows breaks under large motion (Sec. 4.1, 6.4 above).
   - Routing the costume through VACE's `reference_image` or Fun-Control's `ref_image` puts it inside full attention, which is the LVCD idea already built into Wan (nodes_wan.py, Wan Sec. 4.2.1). Test this before building anything new.
2. **Why pose+depth drives costume drift.**
   - LVCD's control signal is lineart of the *target character*, extracted from real frames (sample.ipynb). It carries costume silhouette and internal lines such as hood edge, belt and quiver.
   - A mannequin's pose+depth carries none of that. The model must invent costume shape per frame, and it invents it differently each time.
   - A "lineart per frame" split fixes this only if the lineart contains the costume, so it must be drawn *from the costumed character*, not the mannequin.
3. **Where LVCD-style "lineart first, colour second" fits the keyframe-first route.**
   - The keyframes come from Qwen-Image-Edit, from the facing view plus a skeleton. They are already coloured, so lineart adds nothing there.
   - The split helps the in-betweens. Extract lineart from the finished keyframes (LVCD uses a lineart detector for exactly this, sample.ipynb). Get in-between lineart by in-betweening *lineart*, not colour. Then colour the whole run from one coloured keyframe.
   - This is the LVCD setting: the reference is a coloured frame of the same shot, and every other frame is lineart (sample_func.py).
   - Colour identity then comes from a same-shot reference, which is the only regime LVCD was trained and tested on (Sec. 4.1/5).
4. **Do not expect LVCD to colour from a turnaround sheet.**
   - LVCD's reference is always a frame of the same clip, and it mis-colours parts not visible in the reference (Sec. 6.7).
   - A 4-direction sprite turn shows sides and backs the facing view lacks. Colour each direction's run from *that direction's* coloured keyframe, not from the front view.
   - If a sheet-style reference is unavoidable, AniDoc's correspondence matching is the documented mechanism for pose mismatch (AniDoc abstract).
5. **Resolution fit.**
   - LVCD runs at 576×320 in 14–19-frame windows with about 20 GB VRAM (Sec. 3, issue #7). That fits a 24 GB card.
   - SVD's VAE loses tiny details (Sec. 6.7), and on ~48 px sprites every detail is tiny. So the colouring pass should run at a large upscale and be palette-quantised afterwards, as already planned in the prior doc.
6. **Borrowable pieces with no model change.**
   - *Overlap blending:* re-denoise the last 4 frames of segment n−1 as the first of segment n (Sec. 4.2). This maps onto chaining Wan segments that share a keyframe.
   - *Re-inject the reference on the seam frames:* LVCD amplifies reference attention 10× there (sample_func.py). The Wan analogue is putting the reference image into every segment, not only the first.

## Sources

- https://arxiv.org/abs/2409.12960
- https://arxiv.org/html/2409.12960
- https://dl.acm.org/doi/10.1145/3687910
- https://luckyhzt.github.io/lvcd
- https://github.com/luckyhzt/LVCD
- https://api.github.com/repos/luckyhzt/LVCD
- https://github.com/luckyhzt/LVCD/blob/main/inference/sample.ipynb
- https://github.com/luckyhzt/LVCD/blob/main/inference/sample_func.py
- https://github.com/luckyhzt/LVCD/blob/main/configs/lvcd.yaml
- https://github.com/luckyhzt/LVCD/issues/7
- https://github.com/luckyhzt/LVCD/issues/10
- https://github.com/kijai/ComfyUI-LVCDWrapper
- https://github.com/kijai/ComfyUI-LVCDWrapper/blob/main/nodes.py
- https://api.github.com/repos/kijai/ComfyUI-LVCDWrapper
- https://arxiv.org/abs/2412.14173
- https://github.com/LucipherDev/ComfyUI-AniDoc
- https://arxiv.org/html/2503.20314
- https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_wan.py

## Blocked

- dl.acm.org (ACM TOG version) · not fetched, arXiv used instead · not needed unless the camera-ready differs from arXiv.
- LVCD supplementary (120 demo clips, project page) · videos, not text-readable · watch them manually if visual quality on large motion matters.
