# Can we move from Wan 2.2 Fun-Control 5B to a 14B model so the pix3lwalk sprite-walk LoRA works with pose control?

## Short answer
Yes. Wan 2.2 Fun-Control A14B exists (Apache-2.0) and ComfyUI has native, fp8, and GGUF files for it. Its transformer has the same width and depth as I2V-A14B, with extra input channels (in_dim 52 vs 36), so LoRAs on the attention and FFN layers should load. That is not verified for pix3lwalk: ComfyUI's own Fun-Control template loads the I2V lightx2v LoRAs on it, but no source tests a style LoRA. Most practical route: fp8 Fun-Control high+low (28.6 GB) + lightx2v 4-step LoRAs (2.5 GB) + pix3lwalk high+low (0.3 GB) + Wan 2.1 VAE (0.25 GB) = about 31.6 GB, plus the 6.7 GB fp8 UMT5 encoder if not already installed (about 38.3 GB).

## Findings

### 1. Fun-Control A14B
- Exists: `alibaba-pai/Wan2.2-Fun-A14B-Control`, license Apache-2.0, tagged base model Wan2.2-I2V-A14B, with separate `high_noise_model/` and `low_noise_model/` folders. [HF API](https://huggingface.co/api/models/alibaba-pai/Wan2.2-Fun-A14B-Control)
- Upstream card: control by Canny, Depth, Pose, MLSD, trajectory; the examples also take a reference image. Recommended 512/768/1024 resolution, 81 frames, trained at 16 fps. About 60 GB of weights in the original repo. No minimum VRAM stated (tested GPUs 12-80 GB, with fp8 and CPU-offload modes). [README_en](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control/raw/main/README_en.md)
- ComfyUI repackaged files (Comfy-Org/Wan_2.2_ComfyUI_Repackaged, `split_files/diffusion_models/`), sizes in bytes/decimal GB: [tree](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)
  - `wan2.2_fun_control_high_noise_14B_bf16.safetensors` 28.58 GB; `..._low_noise_14B_bf16` 28.58 GB
  - `wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors` 14.30 GB; `..._low_noise_14B_fp8_scaled` 14.30 GB
  - For comparison, the current 5B file `wan2.2_fun_control_5B_bf16` is 10.00 GB.
- GGUF (QuantStack/Wan2.2-Fun-A14B-Control-GGUF, `HighNoise/` and `LowNoise/`, identical sizes per quant, per expert): Q4_0 8.57 GB, Q4_K_M 9.66 GB, Q5_0 10.33 GB, Q5_K_M 10.80 GB, Q6_K 12.02 GB, Q8_0 15.42 GB (decimal GB; Q2/Q3 also exist). [tree](https://huggingface.co/api/models/QuantStack/Wan2.2-Fun-A14B-Control-GGUF/tree/main?recursive=true). The GGUF repo's license was not checked; it is derived from Apache-2.0 weights.
- ComfyUI-native workflow: template "Wan2.2 Fun Control" (`video_wan2_2_14B_fun_control.json`). Nodes: Load Diffusion Model x2, LoraLoaderModelOnly, Load Image (start frame), Load Video (already-preprocessed pose video), `Wan22FunControlToVideo`. Needs `wan_2.1_vae.safetensors` and `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (from the Wan 2.1 repackaged repo). Optional `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise/low_noise` LoRAs, enabled by default. [Comfy docs](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control)
- Comfy's test (RTX 4090D 24 GB, 640x640, 81 frames, fp8_scaled): 83% VRAM and about 520 s without the LoRA; 89% VRAM and about 79-138 s with the 4-step LoRA. [Comfy docs](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control)
- Control video must already be a pose render; only Canny is built into Comfy, other preprocessors need comfyui_controlnet_aux. This matches our skeleton-video approach. [Comfy docs](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control)

### 2. I2V-A14B LoRAs on Fun-Control A14B
- Fun-Control A14B config: dim 5120, ffn_dim 13824, 40 layers, 40 heads, model_type "i2v", but `in_dim` 52 (plus `add_ref_conv`, `in_dim_control_adapter` 24). [Fun-Control high_noise config](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control/raw/main/high_noise_model/config.json)
- I2V-A14B config: dim 5120, ffn_dim 13824, 40 layers, 40 heads, in_dim 36. [I2V config](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/raw/main/high_noise_model/config.json)
- So only the input patch embedding differs in shape. Inference: a LoRA that targets attention/FFN layers fits both, but this is my reading of the configs, not a documented test. Documented: ComfyUI's Fun-Control template ships and enables the I2V lightx2v 4-step LoRAs on the Fun-Control experts. [Comfy docs](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control)
- A community Fun-Control workflow also chains I2V LoRAs (high-noise first) with the lightx2v LoRAs. [MimicPC](https://www.mimicpc.com/workflows/wan-22-fun-control-i2v-and-t2v-with-video-reference-and-light2x)
- Not found: any report, issue, or doc testing a style LoRA (like pix3lwalk) on Fun-Control A14B. Treat as "likely loads, effect strength unverified"; a side-by-side run is needed. The GitHub issue search (VideoX-Fun) returned HTTP 504, so it was not checked.

### 3. pix3lwalk
- Civitai model 2172038, "wan2.2 lora, walk animation, side view sprite animation, pixel style"; creator aaaa_Aa; v1.0, Nov 28 2025; base Wan Video 2.2 I2V-A14B; license shown as Apache-2.0. [Civitai](https://civitai.com/models/2172038)
- Civitai holds only the high-noise file (146.34 MB). The low-noise file is on Hugging Face: `theamusing/wan2.2_pixel_walk_lora` with `pixel_walk_lora_v1_high_noise.safetensors` and `pixel_walk_lora_v1_low_noise.safetensors`, 153,453,624 bytes (153.5 MB) each; HF license MIT; README is empty. [HF tree](https://huggingface.co/api/models/theamusing/wan2.2_pixel_walk_lora/tree/main), [HF API](https://huggingface.co/api/models/theamusing/wan2.2_pixel_walk_lora)
- Usage per the author: prompt "pix3lwalk, side view character walk animation."; strength 0.7-0.9 (0.8 suggested); 24 frames at 16 fps; trained 3,000 steps; use both high and low LoRAs for better pixel results; the high one works on any art style. Showcase used downsampling and scaling to sharpen the pixel look. [Civitai](https://civitai.com/models/2172038)
- Input expectation: the page only specifies a text prompt; it says nothing about the input image (background, view, single sprite). The prompt implies a side-view character. SFW/NSFW rating was not shown on the fetched page. Both unverified.

### 4. Plain I2V-A14B sizes, VRAM, speed LoRA
- Comfy fp16: `wan2.2_i2v_high_noise_14B_fp16` and `low_noise` 28.58 GB each; fp8_scaled 14.29 GB each. [tree](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)
- GGUF (QuantStack/Wan2.2-I2V-A14B-GGUF), per expert: Q4_K_M 9.65 GB, Q5_0 10.31 GB, Q5_K_M 10.79 GB, Q8_0 15.41 GB (Q4_0 listed only for LowNoise at 8.56 GB). VAE 0.25 GB. [tree](https://huggingface.co/api/models/QuantStack/Wan2.2-I2V-A14B-GGUF/tree/main?recursive=true)
- VRAM: Comfy gives no I2V-A14B table. The only measured figure is the Fun-Control one above (24 GB card, 640x640x81 frames, fp8: 83-89%). 512px and 33-49 frames should need less; I found no primary figure for it. ComfyUI loads one expert at a time, so VRAM is set by one expert plus activations (inference, not documented).
- Speed LoRA: Comfy repackaged `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise/low_noise.safetensors`, 1.227 GB each. [tree](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/loras) Newer lightx2v/Kijai rank-64 versions are ~0.63 GB each. [Kijai tree](https://huggingface.co/api/models/Kijai/WanVideo_comfy/tree/main?recursive=true) Lightx2v card: Apache-2.0, 4 steps, strength 1.0, high-noise more creative, low-noise more stable; made for the I2V-A14B base. [lightx2v](https://huggingface.co/lightx2v/Wan2.2-Distill-Loras) Comfy notes it slightly lowers quality/motion. [Comfy docs](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control) Risk: a distill LoRA can damp walk motion; a user comment on the earlier release reports reduced movement (anecdotal, [search result](https://comfyui-wiki.com/en/tutorial/advanced/video/wan2.2/wan2-2-fun-control)).

### 5. Other 14B pose routes
- Wan 2.2 Animate 14B: Apache-2.0 base I2V-A14B, video-to-video; reference image + driving video; Comfy converts the video to pose with a DWPose node (no separate pose-video input in the template). Files: `wan2.2_animate_14B_bf16` 34.55 GB, int8 18.41 GB (Comfy repackage), GGUF (QuantStack) Q4_K_M 11.50 GB, Q8_0 18.72 GB; also needs clip_vision_h (1.26 GB), the lightx2v I2V LoRA, KJNodes and controlnet_aux. [HF model](https://huggingface.co/api/models/Wan-AI/Wan2.2-Animate-14B), [Comfy docs](https://docs.comfy.org/tutorials/video/wan/wan2-2-animate), [tree](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models), [GGUF](https://huggingface.co/api/models/QuantStack/Wan2.2-Animate-14B-GGUF/tree/main?recursive=true)
- Wan 2.2 Fun-VACE A14B (Comfy): fp8 17.35 GB per expert (bf16 34.68 GB). Kijai also has VACE modules (6.10 GB each) to add on top of a base. [tree](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models), [Kijai tree](https://huggingface.co/api/models/Kijai/WanVideo_comfy/tree/main?recursive=true)
- Wan 2.1 VACE 14B: `wan2.1_vace_14B_fp16` about 32 GB; reference-to-video and control video (the doc example is depth; pose not mentioned). 4090 timing: 640x640, 49 frames about 7 min. [Comfy docs](https://docs.comfy.org/tutorials/video/wan/vace) Note it is Wan 2.1, so I2V-A14B (2.2) LoRAs are a different generation.
- Whether I2V-A14B LoRAs apply to Animate or VACE: no primary source found. Animate and Fun-VACE are I2V-A14B derivatives, so the same config reasoning applies (inference only); Wan 2.1 VACE is not.

## Download total for the most practical route (Fun-Control A14B fp8 + lightx2v + pix3lwalk, ComfyUI native)
| Item | GB |
|---|---|
| Fun-Control high+low fp8_scaled | 28.59 |
| lightx2v 4-step I2V LoRA high+low (Comfy copy) | 2.45 |
| pix3lwalk high+low (HF) | 0.31 |
| Wan 2.1 VAE | 0.25 |
| Subtotal | 31.6 |
| umt5_xxl_fp8_e4m3fn_scaled (if not already installed) | 6.74 |
| Total with encoder | 38.3 |

Lower-VRAM alternative: GGUF Q4_K_M experts (2 x 9.66 = 19.33 GB) gives about 22.3 GB (29.1 with encoder); Q8_0 (2 x 15.42) gives about 33.8 GB (40.6 with encoder). Needs a GGUF loader node. Sizes computed from the listings above. The base Fun-Control 5B uses the Wan 2.2 VAE, so the Wan 2.1 VAE is likely a new download (not checked locally).

## Sources
- https://civitai.com/models/2172038
- https://huggingface.co/api/models/theamusing/wan2.2_pixel_walk_lora
- https://huggingface.co/api/models/theamusing/wan2.2_pixel_walk_lora/tree/main
- https://huggingface.co/api/models/alibaba-pai/Wan2.2-Fun-A14B-Control
- https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control/raw/main/README_en.md
- https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control/raw/main/high_noise_model/config.json
- https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/raw/main/high_noise_model/config.json
- https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models
- https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/loras
- https://huggingface.co/api/models/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files?recursive=true
- https://huggingface.co/api/models/QuantStack/Wan2.2-Fun-A14B-Control-GGUF/tree/main?recursive=true
- https://huggingface.co/api/models/QuantStack/Wan2.2-I2V-A14B-GGUF/tree/main?recursive=true
- https://huggingface.co/api/models/QuantStack/Wan2.2-Animate-14B-GGUF/tree/main?recursive=true
- https://huggingface.co/api/models/Kijai/WanVideo_comfy/tree/main?recursive=true
- https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
- https://huggingface.co/api/models/Wan-AI/Wan2.2-Animate-14B
- https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control
- https://docs.comfy.org/tutorials/video/wan/wan2-2-animate
- https://docs.comfy.org/tutorials/video/wan/vace
- https://docs.comfy.org/tutorials/video/wan/wan2_2
- https://blog.comfy.org/p/comfyui-wan22-fun-inp-support
- https://www.mimicpc.com/workflows/wan-22-fun-control-i2v-and-t2v-with-video-reference-and-light2x
- https://comfyui-wiki.com/en/tutorials/advanced/video/wan2.2/wan2-2-fun-control

## Blocked
- github.com/aigc-apps/VideoX-Fun issues · 504 timeout · retry later or run `gh search issues "lora fun control" --repo aigc-apps/VideoX-Fun`
- civitai.com/models/2172038 (SFW rating, image-input guidance, filename) · page lacks it · open the page in a logged-in browser and paste the description, comments, and rating
- huggingface.co/theamusing/wan2.2_pixel_walk_lora · empty README · none (author published no card)
- Reddit / community reports on style LoRAs on Fun-Control · search found none · paste any thread links, or test locally
