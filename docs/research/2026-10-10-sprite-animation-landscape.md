# What open-weight options exist (Oct 2026) to generate a 2D pixel-art character and animate it into a sprite sheet, and which have real sprite evidence, on one 24 GB RTX A5000 in ComfyUI?

## Short answer
The only routes with sprite-specific evidence are Wan 2.2 I2V-A14B with the theamusing/Teluv walk, attack, idle, run and jump LoRAs (video, then pixel-grid post-processing), and single-image sheet LoRAs on Qwen-Image-Edit, FLUX.2 Klein and Anima (a whole grid in one image). On the A5000 (Ampere, no fp8 compute) the best Wan 14B precision to test first is INT8-ConvRot (native in ComfyUI, 14.5 GB per expert) or GGUF Q8_0 (15.4 GB per expert); fp8_scaled only saves memory there, not time. All "evidence" is Civitai download counts and author showcase clips: no benchmark or user study of walk-cycle quality exists, so a side-by-side on our own 64px test sheet is still required.

## Findings

### A. Character generation (single sprite, side view)
- Flux.1 Kontext "make a pixel sprite" LoRA (Civitai 1875421): turns an input character into a chibi pixel sprite on a green background, facing right, 512x512, 175 MB, 1,029 downloads, trained on Guardian Tales assets (IP concern); Kontext-dev licence is non-commercial. [Civitai](https://civitai.com/models/1875421), [Kontext licence](https://huggingface.co/api/models/black-forest-labs/FLUX.1-Kontext-dev)
- FLUX.2 Klein 9B "Make Sprite" (2505873): photo or creature to retro sprite, trigger `M4k3spr1t3`, 81 MB, 352 downloads; Klein 9B is under the FLUX non-commercial licence. [Civitai](https://civitai.com/models/2505873), [licence](https://huggingface.co/api/models/black-forest-labs/FLUX.2-klein-base-9B)
- Z-Image-Turbo (Apache-2.0) has several pixel-style LoRAs: Hard Edge Pixel Art (681332, 4,500 downloads), Elusarca Detailed Pixel Art (2190363), Pixel Art Style (1770073, 3,382). Style only, no animation. [Civitai API](https://civitai.com/api/v1/models?query=pixel&types=LORA), [Z-Image licence](https://huggingface.co/api/models/Tongyi-MAI/Z-Image-Turbo)
- Krea 2 "Pixel Art Side-view (No fake pixel artifacts)" (2988089): 128x128 upscaled to 1024 with an 8x8 logical pixel grid, post-process with the `comfyui-pixelgrid-helpers` "Pixel Grid: Median Fixer" node; 223 MB, 536 downloads; the author says INT8, INT4 and GGUF stay usable. Krea 2's base licence was not checked. [Civitai](https://civitai.com/models/2988089)
- Pixel-art SDXL/Illustrious/Anima style LoRAs (Pixel Art XL 120096, 118k downloads; Anima RPG Character Sprite 1397312, 6.5k downloads) draw characters but not motion. [Civitai API](https://civitai.com/api/v1/models?query=pixel&types=LORA)

### B. Video models plus sprite LoRAs
- **Wan 2.2 I2V-A14B + pix3lwalk (2172038 / HF theamusing/wan2.2_pixel_walk_lora)**: base Wan 2.2 I2V-A14B (Apache-2.0); motion driven by text only (image start frame + prompt); 146 MB high-noise on Civitai, low-noise on HF; HF licence MIT, Civitai 945 downloads. Author guidance: weight 0.7-0.9, 24 frames at 16 fps. Evidence: author showcase only, no card on HF. Weaknesses not documented; known general risks are background, identity drift and loop closure. Already covered in the 14B note. [Civitai](https://civitai.com/models/2172038), [HF](https://huggingface.co/api/models/theamusing/wan2.2_pixel_walk_lora), [14B note](2026-10-10-wan22-14b-sprite-route.md)
- **theamusing/pixel_animation (HF, new, found this session)**: one repo with high+low pairs for attack, defeated, hit, idle, jump, run and walk; created 2026-01-27; licence CC-BY-NC-4.0 (non-commercial), README empty. Also `theamusing/idle_v2` (Apache-2.0, 2026-04-22). These widen the walk/idle/run/jump set on the same I2V-A14B base. Quality unverified, no card. [HF pixel_animation](https://huggingface.co/api/models/theamusing/pixel_animation), [HF idle_v2](https://huggingface.co/api/models/theamusing/idle_v2), [HF search](https://huggingface.co/api/models?search=theamusing)
- **Attack LoRAs**: 2085866 (theamusing, v2, three epoch checkpoints on HF, MIT) trained on about 30 melee attack animations at 512x512, 28 frames; input should be a ~64x64 pixel character resized to 512, facing right; mostly slashes, no stabs, no controllable combos; combine with lightx2v. Higher epochs give consistent slashes, lower epochs follow the input better. 1,023 downloads. 1924815 (Teluv, high-noise 150 MB, 3,000 steps, flagged "experimental, may be unstable", 1,312 downloads). [Civitai 2085866](https://civitai.com/models/2085866), [Civitai 1924815](https://civitai.com/models/1924815), [HF attack repo](https://huggingface.co/api/models/theamusing/wan2.2_pixel_attack_lora)
- **Wan 2.1 14B I2V pixel LoRAs**: Birdman-style pixel animation (1431085, 7,607 downloads, 351 MB, trigger `p1x3lStyl3`, Wan 2.1 I2V 480p, idle-loop example prompt; author says lower the output fps to about 11-14 so it reads as pixel art). "Pixel breathe" (1468622, 351 MB, idle breathing for mini characters only, 1,822 downloads). Wan 2.1 LoRAs are a different generation from Wan 2.2 A14B. [Civitai 1431085](https://civitai.com/models/1431085), [Civitai 1468622](https://civitai.com/models/1468622)
- Wan 2.1 1.3B: Pokemon Sprite Animation (1595383, 84 MB) and PixelArtRedmond Wan (2423593); small and fast but low quality tier; not 14B-compatible. [Civitai API](https://civitai.com/api/v1/models/1595383)
- **Wan 2.2 Fun-Control A14B (pose-driven)**: Apache-2.0, pose/depth/canny control, fp8 14.3 GB per expert, GGUF Q8_0 15.4 GB; no sprite LoRA tested on it by anyone I could find. **Wan 2.2 Animate 14B** and **Fun-VACE**: same story, pose or reference-video driven, no pixel-sprite evidence found (the search returned only general Animate/VACE material). See the 14B note for sizes. [14B note](2026-10-10-wan22-14b-sprite-route.md), [search result](https://www.runcomfy.com/models/community/wan-2-2/vace-fun/image-to-video)
- **Wan 2.2 5B**: no pixel or sprite LoRAs; only flat-colour and Hollow Knight style LoRAs. [5B note](2026-10-10-wan22-5b-loras.md)
- **HunyuanVideo (T2V)**: two pixel style LoRAs: "Hunyuan Video LoRA - PixelArt" (1424229, 315 MB, trigger `pixel art,`, trained on pixel animations and stills at ~768, 824 downloads, HF repo trojblue/HunyuanVideo-lora-PixelArt) and "Pixel style hunyuan video model" (1144459, 231 MB, 1,413 downloads). Style LoRAs on a T2V model, no sprite or walk-cycle framing. HunyuanVideo-I2V licence is the Tencent community licence (not Apache; excludes EU/UK/South Korea use). No Hunyuan I2V sprite LoRA found. [Civitai 1424229](https://civitai.com/models/1424229), [Civitai 1144459](https://civitai.com/models/1144459), [Hunyuan-I2V licence](https://huggingface.co/api/models/tencent/HunyuanVideo-I2V)
- **LTX-Video (LTX-2 / 2.3 / 2.5 now)**: Civitai lists many LTX LoRAs and workflows (including a "Pose Helper (I2V)" and an Animatediff LoRA) but my searches for pixel or sprite found nothing. LTX-Video original licence is "other" on HF. No sprite evidence. [Civitai search](https://civitai.com/api/v1/models?query=LTX+pixel), [HF](https://huggingface.co/api/models/Lightricks/LTX-Video)
- **CogVideoX / AnimateDiff**: Civitai searches for pixel/sprite returned no CogVideoX or AnimateDiff pixel or sprite motion module in the first results (search is not exhaustive). Treat as no evidence. [Civitai search](https://civitai.com/api/v1/models?query=AnimateDiff+pixel)
- Wan 2.2 wins on evidence simply because it is the only family with sprite-named, side-view, walk/attack/idle LoRAs.

### C. Image-edit and whole-grid sheet generation
- **FLUX.2 Klein 4B-base "Pixel Art Spritesheet - 4-Walk Small" (2356302)**: draws one 512x512 sheet of 32x32 characters with walk up/down/left/right, standing, jump and lying frames in a single image; 90 MB LoRA on a 7.75 GB base (Apache-2.0); use the base model, not the distilled one (the distilled one "ruins the quality of the pixels and the consistency"); downscale by 4 (k-centroid) to get pixel-perfect 128x128. Works in text-to-image and the edit workflow. 1,190 downloads, 86 thumbs up. Trained on a CC BY 4.0 template sheet by George Bailey; author links a sandbox game to test the sheets. This is the most directly "sprite sheet" model I found, but the fixed layout is 4-walk at 32x32 within a 512 image, not a 64px side-view cycle. [Civitai](https://civitai.com/models/2356302), [Klein 4B base](https://huggingface.co/api/models/black-forest-labs/FLUX.2-klein-base-4B)
- **Spritely "Posable Game Sprites" (2982439)** on Qwen-Image-Edit: you supply a skeleton chart (image 1) and your character on white (image 2); "Sheet v5" outputs a 1536x1024 sheet of 6 frames from a 3x2 chart of 6 skeletons (walk, run, attack), strength 1, 4 steps at CFG 1 with the Lightning LoRA or 25 steps at CFG 4; 576 MB; 846 starter skeleton charts and free ComfyUI workflows included; 8 directions, side-scroller, isometric and top-down. Pose-driven, so the walk shape is controlled, not left to the model. Only 92 downloads and 5 thumbs up (new); licence on Civitai is Image + RentCivit; also sold as a hosted service. Prompt text asks for "32-bit pixel art" and "stable identity and equipment". Evidence is the author's own. [Civitai](https://civitai.com/models/2982439)
- **[Lah] Animation Sheet (2177558)** on Qwen-Image-Edit 2509: 3x3 or 3x2 sheets for hit/kick/slash/stab, jump/die/dash and dual-sword skills; style is 2D anime, not pixel; 1,062 downloads, 136 thumbs. No walk. [Civitai](https://civitai.com/models/2177558)
- **Pose Transfer - Qwen Edit (1959609)**: 7,781 downloads, 288 MB, left half pose / right half subject, prompt "transfer the pose in the image on the left to the person in the image on the right", strength 1-1.25, works best with front-facing poses. Generic, one frame at a time. [Civitai](https://civitai.com/models/1959609)
- Qwen-Image-Edit sizes: 2509/2511 fp8 20.4-20.5 GB; GGUF 2509 Q4_K_M 13.07 GB, Q5_K_M 14.93 GB, Q6_K 16.82 GB, Q8_0 21.76 GB. 2511 is Apache-2.0. Fits 24 GB. [Comfy tree](https://huggingface.co/api/models/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models), [QuantStack GGUF](https://huggingface.co/api/models/QuantStack/Qwen-Image-Edit-2509-GGUF/tree/main), [licence](https://huggingface.co/api/models/Qwen/Qwen-Image-Edit-2511)
- **Anima sheet LoRAs** (Anima is a newer anime base with many pixel LoRAs): "RPG Character Sprite" (1397312, 3x4 grid of front/side/back views, RPG Maker style, 6,536 downloads, 795 thumbs, 90 MB) and "RPG Sideview Battle Actor Sprite" (2646716, 9x6 grid including walk, fight and idle at 2048x1360, 721 downloads). Large sprite-sheet grids but chibi RPG Maker layout, not freely posed. Anima base licence not checked. [Civitai 1397312](https://civitai.com/models/1397312), [Civitai 2646716](https://civitai.com/models/2646716)
- **SDXL/Illustrious "sprite sheet" LoRAs** (Space Candy Media Sprite Sheet Generator v2-4, 1.0-1.3k downloads; Walking spritesheet 1099272, 650 downloads). Older, draw a grid in one image; alignment between cells is not guaranteed. [Civitai API](https://civitai.com/api/v1/models?query=sprite+sheet&types=LORA)
- **FLUX.1 Kontext sheet LoRAs**: only a character turnaround sheet LoRA (1753109, 7,642 downloads) and the pixel-sprite one above; no Kontext walk-sheet LoRA found. [Civitai API](https://civitai.com/api/v1/models?query=character+sheet&types=LORA)

### D. Pixel-art-specific models and research
- **Retro Diffusion** (rd-animation, rd-fast, etc.): closed. Hosted on Replicate and its own API (animations $0.07-0.25 each), ComfyUI only through an API node needing a key; Astropulse's page says the code and models are owned by Astropulse LLC and not usable commercially, outputs are the user's. A local Aseprite extension exists, not ComfyUI. Not open-weight. Output formats advertised: Four Angle Walking 48x48, Small Sprites 32x32 (vendor blog claims, 16-frame walk sheet). [Replicate](https://replicate.com/retro-diffusion/rd-animation/examples), [ComfyUI API node](https://github.com/svntax/ComfyUI-RetroDiffusion-API-Node), [itch.io](https://astropulse.itch.io/retrodiffusionai), [Replicate blog](https://replicate.com/blog/retro-diffusions-pixel-art-models-are-now-on-replicate)
- **PixelLab** (Pixflux, Bitforge, skeleton animate): hosted commercial service, not open weights (per the search summary). [Sorceress overview](https://sorceress.games/blog/ai-sprite-generator-pixel-walk-cycles-idles-and-vfx)
- **Sprite Sheet Diffusion** (arXiv 2412.03685, v2 Mar 2025, CMU/DeepMind): Animate Anyone adapted to sprites (ReferenceNet on SD-1.5, Pose Guider, motion module); trained on 463 triplets from 55 sequences of 11 characters; small-scale baseline. The project page links https://github.com/chenganhsieh/Sprite-Sheet-Diffusion, but the repo gave HTTP 504, so code/weights availability is unconfirmed. SD-1.5-era; not practical at 64px. [arXiv](https://arxiv.org/html/2412.03685v2), [project page](https://chenganhsieh.github.io/spritesheet-diffusion/)
- No other 2024-2026 open pixel-sprite animation paper with released code turned up in my searches. Open pixel checkpoints are mostly style LoRAs (Pixel Art XL, 8bitdiffuser 64x on SD 1.5, 21k downloads). [Civitai API](https://civitai.com/api/v1/models?query=pixel+art+walk)

### E. Hybrid: 3D rig, render, pixelate
- Pipeline is established, not AI: Mixamo (In-Place walk) to Blender orthographic camera script to frames to sprite sheet plus JSON, then ImageMagick level curves for a pixel look. A freegamedev forum post describes a fully scripted Blender-to-sprite-sheet pipeline. Nothing found documents 64px, outlines or palette steps; Mixamo's licence terms were not checked. Gives exact loop closure, perfect grid alignment and identical identity across frames; the cost is a 3D character (or an AI 3D step) and a style step to make it look hand-drawn. Our existing pixel-grid post-processing and an img2img pass (e.g. Klein or Kontext pixel LoRA per frame) would sit at the end. [GameFromScratch](https://gamefromscratch.com/creating-a-walk-cycle-spritesheet-in-under-15-minutes/), [freegamedev](https://forum.freegamedev.net/viewtopic.php?p=72603), [BlenderNation](https://www.blendernation.com/2020/05/19/blender-secrets-mixamo-walk-cycle/)

### F. Precision on 24 GB for the Wan 14B sprite route
- ComfyUI only enables fp8 compute on NVIDIA GPUs with compute capability 8.9 or higher (major 8 with minor < 9 returns False; major >= 9 returns True). The RTX A5000 is Ampere (8.6, not confirmed from an NVIDIA page; see Blocked), so fp8_scaled weights are used for storage and upcast, giving no speed gain over bf16 compute. [ComfyUI source](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/model_management.py)
- ComfyUI now has `supports_int8_compute` (weight-only int8 via `torch._int_mm`), true on any CUDA device. Comfy-Org ships `wan2.2_animate_14B_int8_convrot` (18.41 GB); Winnougan/Wan2.2-INT8-Convrot (Apache-2.0) provides I2V-A14B high/low INT8-ConvRot at 14.54 GB each, a UMT5 int8 encoder (6.74 GB) and says "INT8 is significantly faster than FP8/FP4 on this generation (RTX 30XX)", with no measured numbers and no quality comparison; it uses the unchanged lightx2v 4-step LoRAs. Community pages say native INT8 merged in June 2026 (ComfyUI >= 0.27.0 per one) and that Wan 2.2 loads; no official Wan Ampere benchmark found. LoRA behaviour on int8 weights is not documented. [ComfyUI source](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/model_management.py), [HF repo](https://huggingface.co/Winnougan/Wan2.2-INT8-Convrot), [HF tree](https://huggingface.co/api/models/Winnougan/Wan2.2-INT8-Convrot/tree/main), [Comfy tree](https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models)
- GGUF (QuantStack I2V-A14B): Q4_K_M 9.65 GB, Q5_K_M 10.79 GB, Q8_0 15.41 GB per expert. ComfyUI-GGUF states LoRA loading is "experimental" with the built-in loader; the GGUF loader dequantizes on the fly (llama.cpp style). Quality: one HF card says a 24 GB card comfortably runs Q8 high + Q8 low; a blog claims Q8_0 is visually indistinguishable from fp8 (uncontrolled, SEO-type source, treat as unverified). No controlled fp8-vs-GGUF test on pixel art exists. [14B note](2026-10-10-wan22-14b-sprite-route.md), [ComfyUI-GGUF README](https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/README.md), [Wan27 guide](https://wan27.org/blog/wan-2-2-gguf-guide), [HF card](https://huggingface.co/tanishqw/Wan2.2-Custom-Models-GGUF)
- Fidelity note: Kijai says his fp8_scaled files are "better ... when measured against fp16" but the card gives no numbers. The Comfy docs warn the lightx2v 4-step LoRA "slightly" lowers quality; for crisp pixels, test it with and without. [Kijai card](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled), [Comfy docs](https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control)
- Answer: on Ampere, rank (inference from the above, not benchmarked): (1) GGUF Q8_0 as the safe fidelity choice (15.4 GB per expert, closest to bf16, loads only one expert at a time, but dequant overhead and "experimental" LoRA support), (2) INT8-ConvRot as the speed candidate (14.5 GB per expert, hardware-friendly on Ampere, but newest and least verified with LoRAs), (3) fp8_scaled (14.3 GB per expert, smallest download of the Comfy-native set but no compute advantage on this GPU), (4) Q4/Q5 only if VRAM is tight with long clips or 720p (quality loss unquantified for pixel edges). Run one fixed-seed A/B of Q8_0 vs INT8 vs fp8 with pix3lwalk at strength 0.8 on one 64px character before committing.

## Route comparison table (summary)
| Route | Base / licence | Motion driver | Size at 24 GB-fit | Evidence | Weak points |
|---|---|---|---|---|---|
| Wan 2.2 I2V-A14B + pix3lwalk | Wan 2.2, Apache-2.0; LoRA MIT | text only | 2x14.3-15.4 GB + 0.3 GB LoRA | author clips, 945 dl | drift, background, loop closure unverified |
| Wan 2.2 Fun-Control A14B + same LoRA | Apache-2.0 | pose video | 2x14.3 GB (fp8) | none for sprite LoRA | LoRA transfer unverified |
| Wan 2.2 Animate / VACE | Apache-2.0 | ref video / pose | 11.5-18.7 GB (GGUF/int8) | none for pixel | unknown |
| Wan 2.1 Birdman / breathe LoRAs | Wan 2.1 I2V 480p | text | 351 MB LoRA | 7.6k / 1.8k dl | Wan 2.1, idle-style, higher fps must be cut |
| HunyuanVideo pixel LoRAs | Tencent community licence | text | n/a | style only | no I2V sprite use |
| LTX / CogVideoX / AnimateDiff | various | n/a | n/a | none found | |
| Flux.2 Klein 4B 4-Walk sheet | Apache-2.0 | prompt | 7.75 GB + 90 MB | 1.2k dl, sandbox tool | fixed 32x32, 4 directions at 512px |
| Qwen-Image-Edit + Spritely Sheet | Apache-2.0 base | skeleton chart | 13-22 GB GGUF + 576 MB | author only, 92 dl | new, license terms, white bg |
| Retro Diffusion | closed | prompt | n/a | vendor | API only |
| 3D rig to pixelate | Blender, Mixamo | skeleton | n/a | long practice | needs 3D character, style step |

## Ranked shortlist: basic human adventurer, side-view walk cycle, 64px sheet
1. **Wan 2.2 I2V-A14B + pix3lwalk (and the pixel_animation walk pair), Q8_0 or INT8, with downscale to 64px.** Only family with a side-view walk LoRA aimed at exactly this; the first-frame image fixes identity; frames get pixel-grid snapped afterward. Risk: loop closure and drift need a selection or crossfade step.
2. **Qwen-Image-Edit 2511/2509 + Spritely Sheet v5 (skeleton-driven, 6 frames per sheet).** Pose chart controls the walk phases, so loop closure and consistent frames are better than text-only video, and the output is already a grid. Risk: newest, 92 downloads, "32-bit pixel art" not 64px exact, still needs grid snapping.
3. **FLUX.2 Klein 4B-base + 4-Walk Small sheet LoRA.** Apache-2.0 base, small, fast, designed as a sheet, with a downscale rule from the author. Risk: fixed 32x32 character size and 4-direction layout; it could only be a reference or style starting point for 64px.
4. **Wan 2.2 Fun-Control A14B with a skeleton video plus pix3lwalk.** Combines your pose control with the pixel LoRA; potentially the best identity-plus-motion result. Risk: LoRA on Fun-Control is unverified, so it ranks below the plain I2V route until a side-by-side run proves it.
5. **Hybrid 3D rig (Mixamo or own skeleton) in Blender, then pixelate and style-pass.** Most reliable loop and alignment; cost is the 3D character and a look-matching step. Use as the fallback when AI frames will not close the loop.

## Sources
- https://civitai.com/models/2172038
- https://civitai.com/models/2085866
- https://civitai.com/models/1924815
- https://civitai.com/models/1431085
- https://civitai.com/models/1468622
- https://civitai.com/models/1424229
- https://civitai.com/models/1144459
- https://civitai.com/models/2356302
- https://civitai.com/models/2982439
- https://civitai.com/models/2177558
- https://civitai.com/models/1959609
- https://civitai.com/models/1875421
- https://civitai.com/models/2505873
- https://civitai.com/models/2988089
- https://civitai.com/models/1397312
- https://civitai.com/models/2646716
- https://civitai.com/api/v1/models (various queries: pixel, sprite, sprite sheet, LTX, AnimateDiff)
- https://huggingface.co/api/models/theamusing/wan2.2_pixel_walk_lora
- https://huggingface.co/api/models/theamusing/wan2.2_pixel_attack_lora
- https://huggingface.co/api/models/theamusing/pixel_animation
- https://huggingface.co/api/models/theamusing/idle_v2
- https://huggingface.co/api/models?search=theamusing
- https://huggingface.co/Winnougan/Wan2.2-INT8-Convrot
- https://huggingface.co/api/models/Winnougan/Wan2.2-INT8-Convrot/tree/main
- https://huggingface.co/api/models/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models
- https://huggingface.co/api/models/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models
- https://huggingface.co/api/models/QuantStack/Qwen-Image-Edit-2509-GGUF/tree/main
- https://huggingface.co/api/models/black-forest-labs/FLUX.2-klein-base-4B
- https://huggingface.co/api/models/black-forest-labs/FLUX.2-klein-base-9B
- https://huggingface.co/api/models/black-forest-labs/FLUX.1-Kontext-dev
- https://huggingface.co/api/models/Qwen/Qwen-Image-Edit-2511
- https://huggingface.co/api/models/Tongyi-MAI/Z-Image-Turbo
- https://huggingface.co/api/models/tencent/HunyuanVideo-I2V
- https://huggingface.co/api/models/Lightricks/LTX-Video
- https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled
- https://huggingface.co/tanishqw/Wan2.2-Custom-Models-GGUF
- https://wan27.org/blog/wan-2-2-gguf-guide
- https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/model_management.py
- https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/README.md
- https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-control
- https://arxiv.org/html/2412.03685v2
- https://chenganhsieh.github.io/spritesheet-diffusion/
- https://replicate.com/retro-diffusion/rd-animation/examples
- https://replicate.com/blog/retro-diffusions-pixel-art-models-are-now-on-replicate
- https://astropulse.itch.io/retrodiffusionai
- https://github.com/svntax/ComfyUI-RetroDiffusion-API-Node
- https://sorceress.games/blog/ai-sprite-generator-pixel-walk-cycles-idles-and-vfx
- https://gamefromscratch.com/creating-a-walk-cycle-spritesheet-in-under-15-minutes/
- https://forum.freegamedev.net/viewtopic.php?p=72603
- https://www.blendernation.com/2020/05/19/blender-secrets-mixamo-walk-cycle/
- https://www.runcomfy.com/models/community/wan-2-2/vace-fun/image-to-video
- docs/research/2026-10-10-wan22-14b-sprite-route.md
- docs/research/2026-10-10-wan22-5b-loras.md

## Blocked
- github.com/chenganhsieh/Sprite-Sheet-Diffusion · 504 timeout · retry later or `gh repo view chenganhsieh/Sprite-Sheet-Diffusion`
- github.com/city96/ComfyUI-GGUF (HTML page) · 504 timeout · README read from raw.githubusercontent.com instead; open issues on Wan LoRA/speed not checked
- developer.nvidia.com/cuda-gpus · page text lacked RTX A5000 · confirm A5000 compute capability 8.6 locally with `nvidia-smi --query-gpu=compute_cap --format=csv`
- Reddit / r/StableDiffusion / r/comfyui threads on sprite results and fp8 vs GGUF · search returned none · paste thread links or run local A/B
- Civitai model pages (comments, example videos, full descriptions of pix3lwalk and Spritely) · API gave description only, no images · open in a browser and paste comments
- Wan 2.2 INT8-ConvRot Ampere speed and quality numbers · not published · local benchmark on the A5000
- Mixamo licence terms, Krea 2 and Anima base licences · not checked · read each vendor page
