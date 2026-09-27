# How are people generating pixel-art sprite animations with open models today, and which approach fits a local 24 GB A5000 + ComfyUI pipeline (~48 px, 8 directions, 12 fps)?

Date: 2026-09-25. Extends `docs/research/2026-09-25-keyframe-first-sprite-animation.md` (in-betweeners, Qwen keyframes, PixelLab/RD practice, Sprite Sheet Diffusion) and `docs/research/2026-09-20-pixel-art-loras.md` (Z-Image/Qwen/Wan pixel LoRAs). Neither is repeated here. Licences ignored per the brief. Web sources fetched 2026-09-25.

## Short answer

The open-model community has converged on four shapes: (1) **video-then-extract**, i.e. one image → Wan 2.2 I2V/FLF clip → pick frames → remove background → snap to a grid (most posted results); (2) **per-frame posed keyframes**, i.e. Qwen-Image-Edit-2509 with a pose skeleton, one frame at a time; (3) **whole sheet in one image**, from sprite-sheet LoRAs (fast, but only 3–4 frames and 4 directions); and (4) a **3D proxy** (generated mesh → rig → render 8 directions → re-render or pixelate), which holds identity best across directions. For the A5000 at 48 px / 8 directions / 12 fps, the best fit is route 2 for identity plus route 1 as an optional loop/in-between tool. Both need a post stage that detects the pixel grid and palette once per animation, not once per frame. Whole-sheet LoRAs are prototyping tools only. Nobody reports a clean, open, 8-direction, pixel-true result from any single model.

## Findings

### A. Video-then-extract (image → Wan clip → pick frames → pixelate)

- **Best-received open result:** SDXL Illustrious/NoobAI/Pony sprites animated with **Wan 2.2 FLF (first-last-frame to video), using the same image as the first and last frame**, so every clip loops. The poster used no in-between keyframes, only a prompt per animation ("Nope, only prompt for each animation"), on a 12 GB RTX 3060. Posts scored 362 and 282. [R1][R2]
- The shared workflow uses Wan 2.2 I2V A14B **Q4_K_S GGUF** high/low-noise experts, Lightning 4-step LoRAs (4 steps, CFG 1, `ModelSamplingSD3` shift 5), and `WanFirstLastFrameToVideo` at **448×576, 73 frames**, output at **24 fps**. The input image had already been cleaned by "unfake". The prompt is structured as GAME / SUBJECT ("low resolution Pixel art, pixelated game sprite…") / ACTION. [WF1]
- The same author later used MiniMax H3 with the loop "generate sprite → generate video → get key frames → auto background removal". They said "FFLF is better for looping" and used bgbye for background removal. [R3]
- **mor-o/comfyui-2d-character-pipeline** (MIT, 101 stars) runs Qwen-Image-Edit 2509 + InstantX ControlNet-Union to make a posed keyframe, then Wan 2.2 I2V 14B Q5_K_M + lightx2v 4-step, then BiRefNet background removal and a sheet. It is "tuned to peak at ~22.5 GB" on a 24 GB card [G1]. It measures **~18 GB peak at 512×512, 33 frames, 16 fps**, but only after forcibly unloading each ~10.8 GB expert between stages [G1a]. Frame counts must follow Wan's `4n+1` rule [G1a]. The author tried Wan 2.2 Animate for layered cosmetics, but it "didn't stay true to the original base animation", so they used VACE inpainting instead [R4].
- Civitai "Sprite sheet generator" workflow (Wan 2.2 I2V-A14B, first+last frame): 480×480 input, a looping clip, a fixed number of frames extracted, then a **Nearest-Exact downscale to 64×64**, green-screen removal and colour correction. [CV1]
- Civitai "Image to Sprite Sheet, LOW VRAM" (1,341 downloads): Flux Kontext poses the character (a "default pose" is generated first because Kontext alters anatomy), then Wan 2.1 seamless loop with first frame = last frame, then WebP → sheet. Poses need manual rescaling and all characters should face one side. [CV2]
- Wan pixel-sprite motion LoRAs not in the prior survey:
  - "wan2.2 pixel walk" (I2V-A14B): high- and low-noise pair, strength 0.7–0.9, **24 frames at 16 fps**, trigger `pix3lwalk`; the author downsamples afterwards [CV3][HF1].
  - "Wan_Video_Attack_Sprite_Animation" (I2V-A14B, side-view attacks, marked experimental) [CV4].
  - "Pokemon Sprite Animation" for Wan 1.3B/14B T2V, recommended for image-to-video via VACE at 768×768 [CV5].
- Wan 2.1 FLF2V Q4_0 GGUF on a 16 GB A4000: 97 frames at 512×512 took about **1,219 s at 86 % VRAM**, with 20 steps and no distillation LoRA. [CV6]
- **Failure mode, motion style:** video models animate pixel characters "in a 3d kind of way" [R5]. MiniMax "tries to animate pixel art by smoothly rotating limbs", which "doesn't work very well for pixel art" [R6]. A commercial tool builder using the same pose → I2V → sheet shape called pixel art "a whole beast" [R5].
- **Held-frame recovery:** the `Pixel Art Animation Pose Compositor` node splits a noisy video into held poses by frame difference (threshold 7.0; within-pose differences 0.04–1.98, transitions 13–30). It then takes a per-pixel mode across each hold, giving **one clean frame per distinct pose**. This turns "smooth video" into "limited-frame pixel animation" automatically. [G6]

### B. Per-frame posed keyframes (image edit + pose, one frame at a time)

- Civitai "Posed Sprite Sheet Generator" (Qwen-Image-Edit 2509 GGUF) takes a character plus a pose and outputs that character in that pose. About **70 s per image on a 10 GB RTX 3080**. [CV7]
- mor-o's W1 does the same with an OpenPose skeleton and the InstantX ControlNet-Union, and is used as the seed for the video stage. [G1]
- harumaxy/pixel-2d-game-art (SD1.5 + openpose/depth ControlNet from Blender-rendered Mixamo skeletons + IP-Adapter + character LoRA + AnimateDiff v3, 512 px → 64 px, 8 directions) warns that its quality is **not practical**. Walk, run and idle are usable, but aim, jump and roll "collapse into blobs". Its 8-frame walk is generated as 16 frames with only the even frames kept, because AnimateDiff's native length is 16. The author is moving to "image gen for character sheets only → Tripo 3D → retarget → 2D". [G2]
- Its pixelate stage is worth copying:
  - one scale across all motions of a character;
  - contrast-adaptive cell sampling;
  - Oklab palette snapping;
  - removal of stray specks under 4 px;
  - an optional 1 px darkest-colour outline. [G2]
- **8-direction views from one image:** the Qwen-Edit-2509 "Multiple-angles" LoRA takes prompts such as "Rotate the camera 45 degrees to the left", which matches the 8-direction step exactly. The author re-uploaded it in Nov 2025 after users reported "unstable consistency". [HF2] The Civitai "Character Turnaround Sheet" LoRA for Qwen Edit 2509/2511 notes that 2511 "always produced narrow characters" (the fix is a +20 % horizontal stretch). [CV8]

### C. Whole sheet in one image (sprite-sheet LoRAs)

- "Pixel Art Spritesheet – 4-Walk Small" for **FLUX.2 Klein 4B base** (1,127 downloads) outputs 32×32 characters: 4-direction walk, arms-up, jump and lying poses on a 512×512 sheet.
  - The card recommends a ×4 **k-centroid** downscale.
  - It says the distilled Klein "ruins the quality of the pixels and the consistency".
  - Reported failures: cut-off hair and headwear, poor non-human anatomy, and inconsistent back-view rows ("the werewolf sample is missing the red scarf"). [CV9]
- `Mystic07/flux-lora-spritesheet` (FLUX.2-klein-base-9B, trained on ~18.7k LPC-style sheets, trigger `gmsspritesheet`, 1024², 30 steps, CFG 4) promises "4 direction sprite sheet … walk cycle". [HF3]
- "[Lah] Animation Sheet" for **Qwen-Image-Edit 2509** turns an input character into a 3×3 or 3×2 sheet of hit, kick, jump, death, dash and several sword skills. The author admits some substituted actions "are not good". [CV10]
- Older SDXL-family sheet LoRAs:
  - "RPG Top Down 4 Direction Walk Cycle" (Pony): 384×640, downscale ×8 [CV11].
  - "RPG Character Sprite" (Illustrious/Anima): a 3×4 grid of front/side/back RPG Maker sheets [CV12].
  - "RPG Sideview Battle Actor" (Anima): a 9×6 grid [CV13].
  - A PonyXL sprite-sheet LoRA whose author reports that back views are unreliable and "left and right cannot [be] distinguish[ed]" [CV14].
  - The 2022 SD 1.5 "PixelArt SpriteSheet Generator" uses per-view triggers and suggests mirroring for left/right [HF4].
- Krea 2 pixel LoRAs (32/64/128 grids, 1024² generation) also try 4-view sheets. The author says "it doesn't know left from right". Asked about animation, the author said open solutions are lacking. [R6][HF5]
- Prompting a hosted model for an 8-direction sheet: the most upvoted replies point out that the left and right views are "99 % mirrored" (the staff switches hands) and back views go wrong. Several recommend converting to 3D (Meshy/Tripo) and rotating instead. [R7]

### D. 3D proxy (mesh → rig → render → optional re-render)

- An itch.io tutorial runs:
  1. Qwen Edit or Flux to create a T-pose plus front/back/side views;
  2. Hunyuan3D for the mesh;
  3. Mixamo or Blender for the rig;
  4. Mixamo or HY-Motion clips for animation;
  5. rendering the animation to sheets;
  6. ComfyUI re-render with depth + DWPose ControlNets.

  Batches are capped at about **6 sprites on 16 GB**. It recommends 512×512 frames and reports hand/body defects that need touch-ups. [IT1]
- Hunyuan3D-2.1 needs **10 GB for shape and 21 GB for texture (29 GB for both at once)**. Hunyuan3D-2 needs 6 GB for shape and 16 GB for shape + texture, and has a `--low_vram_mode`. [G7][G8]
- A 2023 Civitai pose pack of **8-direction** walk/run OpenPose skeletons, made by rotating animated 3D models, is still widely used (11.7k downloads) as ControlNet input. [CV15]

### E. Big open video model: MiniMax H3

- MiniMax H3 (open weights since 2 Aug 2026; 33B-parameter transformer; FL2VA with 0–2 frames or Ref2VA with up to 9 reference images; 24 fps; native ComfyUI nodes plus a Fun ControlNet Union with pose/depth). It gave the most-upvoted sprite-animation post of the year (867). [R3][HF6][DC1]
- **24 GB fit is poor:**
  - the pruned int8 or fp8 diffusion weights are 21.0 GB;
  - the Qwen3-VL-32B text encoder is 15.7 GB even at nvfp4;
  - the full int8 model is 34 GB. [HF7]
- The licence excludes the US, EU, UK and Korea from its territory. This is noted only because it blocks use outright. [HF8]

### F. Research papers and small open models

- SD-πXL (2024) makes palette-constrained low-resolution images of any H×W and n colours, using score distillation with Gumbel-softmax. It is still-image only. [P1]
- Two GAN papers target direction, not motion:
  - Pix2Pix front → side sprite translation on small datasets, with "varying degrees of generalization" [P2];
  - a missing-data-imputation GAN that generates the missing facing from the other three [P3].
- Sprite Sheet Diffusion v2 (Mar 2025) adds no new open weights beyond what the prior doc covered. [P4]
- Animator2D (BERT + convolutional generator; 64–128 px; the author calls it "early stage") [HF9], and pixel-forge (tiny 32×32 EDM models in Rust) [G9], are not competitive for 48 px characters.

### G. Turning AI output into true pixel art (the post stage)

- **Naive:** nearest-neighbour downscale by the known factor. Pixel Art XL says "downscale 8 times … Nearest Neighbors" [HF10], and the Wan sheet workflow uses Nearest-Exact to 64² [CV1]. Or use **k-centroid** (recommended on the Klein sheet LoRA [CV9]; available in PixelOE [G3]).
- **Contrast-aware:** PixelOE (Apache-2.0, 509 stars) expands outlines before downscaling, offers center/contrast/k-centroid modes and palette quantisation, and ships ComfyUI nodes. [G3]
- **Grid detection** (when the pseudo-pixel size is unknown or uneven):
  - unfake.js: Sobel + tiled voting, WuQuant palette to 8–32 colours, Rust core with CLI, MIT, 922 stars [G4][R8];
  - proper-pixel-art: Canny + Hough mesh, most common colour per cell. For **video/GIF it fits one mesh and one palette over sampled frames** and applies them to every frame, so the output doesn't flicker [G5];
  - perfectPixel: FFT grid detection with a ComfyUI node, 2.1k stars; by the same author as the Wan pixel-walk LoRA [G10][HF1];
  - Sprite Fusion Pixel Snapper: Rust CLI, batch directories, `--palette` hex constraint, 3.2k stars, used in a 64k-view tutorial [G11][YT1].
- **Animation-aware ComfyUI refiner:** `ComfyUI-Krea2-Pixel-Art-Refiner` has OKLab palette reduction, a modal-colour grid collapse, and a mesh detected **on the first frame and reused for every later frame** so sprite size and position stay stable. It also has the Pose Compositor from A. [G6]
- **Palette swap / fixed palettes:** ComfyUI-PixelArt-Detector (446 stars) provides palette loading, a palette generator, kmeans/quantize methods and dithering. [G12]
- **Other practical fixes** named by practitioners: chroma-key or AI background removal; a custom frame picker and frame aligner to stop "frame drift"; avoiding bleed between cells when slicing sheets. [YT1]

### H. YouTube

- The high-view tutorials almost all demonstrate closed tools: PixelLab (182k views), Sorceress/"Spritely" (105k), GPT Image 2 + Nano Banana 2 + Pixel Snapper (64k), GenioPlus (5 directions in one video + mirroring to make 8). [YT1][YT2][YT3][YT4]
- One open-stack example: Qwen Image Edit keyframes → Wan 2.2 FLF, with workflows on GitHub. [YT5]
- The only ComfyUI sprite-sheet workflow Comfy.org features uses Gemini (closed): it generates a 2×2 grid of 4 frames per action on green. [CW1]

## Ranking for the A5000 (24 GB Ampere) → ~48 px, 8 directions, 12 fps

The scale works out as follows: 48 px tall at ×8 means a character about 384 px tall inside a 448–512 px frame. That is the resolution every open workflow above already uses [WF1][G1a][CV1], so no approach needs an unusual canvas.

| Rank | Approach | Identity across frames / directions | Pixel-true path | Frames / fps | VRAM on 24 GB | Main failure modes |
|---|---|---|---|---|---|---|
| **1** | **Per-frame keyframes:** accepted view per direction (Qwen-Edit ±45° multi-angle, or the existing turnaround) → Qwen-Image-Edit-2509 + skeleton per frame → shared-grid/palette post | Each frame re-anchored on the view; directions anchored by the accepted views, not by the model's left/right sense [HF2][R7] | Generate about ×8, then downscale with k-centroid/PixelOE and snap to one palette per unit [G3][G6] | Exactly the frames you ask for (8–12 per action); 12 fps is native | Fits: 70 s/image even on 10 GB [CV7] | Pose flattening toward the reference (the project's own risk); Qwen Multi-angle "unstable consistency" [HF2]; unusual poses need good skeletons [G2] |
| **2** | **Video-then-extract:** Wan 2.2 I2V-A14B GGUF, FLF with the same image at both ends for loops (or a keyframe pair), plus a pixel walk/attack LoRA → pick holds or frames → shared-grid post | Strong within a clip (one start image); each direction is a separate clip, so cross-direction identity depends on the input views | Pose Compositor or picking + first-frame mesh reuse [G6]; proper-pixel-art video mode [G5] | 16/24 fps native, `4n+1` frames; subsample to 12 fps or collapse holds [G1a][G6] | ~18 GB peak Q5_K_M at 512², with expert eviction [G1a]; Q4_K_S ran on 12 GB [R1][WF1] | "3D-like" smooth motion instead of pixel timing [R5][R6]; one clip per action per direction (8× cost); Q/GGUF needs ComfyUI-GGUF |
| **3** | **3D proxy:** Hunyuan3D-2 mesh → auto-rig → Mixamo/Quaternius clip → render 8 yaws → pixelate directly, or re-render with depth+pose | Best across directions (true rotation; no mirrored-view errors) [R7][IT1] | Direct pixelation of renders, or ControlNet re-render then post [IT1][G2] | Any; rendered at exact instants | Hunyuan3D-2: 16 GB shape+texture; 2.1 texture 21 GB [G8][G7] | Looks "3D"; texture and mesh quality ("thin" textures showing the inside) [R7]; hand defects on re-render [IT1] |
| **4** | **Whole-sheet LoRAs** (Klein 4B 4-walk, Klein 9B LPC, Qwen-Edit "Animation Sheet", Pony/Illustrious RPG sheets) | Good within one sheet; back rows and left/right unreliable [CV9][CV14][R6] | ×4–×8 k-centroid or nearest [CV9][CV11] | 3–4 frames per direction, fixed layouts; no 8-direction sheet found | Small (Klein 4B, SDXL) | Fixed 32 px style, too few frames for 12 fps, only 4 directions |
| 5 | **MiniMax H3** FL2VA/Ref2VA | Strong reference handling (≤9 images) [HF6] | Same post as route 2 | 24 fps | 21 GB model + 15.7 GB text encoder → offload-bound [HF7] | Smoothly rotating limbs on pixel art [R6]; licence territory [HF8] |
| 6 | SD1.5 + AnimateDiff pose-guided; Sprite Sheet Diffusion; GAN papers | Weak outside locomotion [G2]; research-scale datasets [P2][P3] | — | 16-frame native (AnimateDiff) [G2] | Small | "Blobs" on uncommon poses [G2] |

**What this adds to the project's plan (route D in the prior doc):**
- Keep D as rank 1.
- Use route 2 only as an FLF loop or in-between tool.
- Borrow the post-stage ideas that nobody in the repo uses yet: fit **one grid and one palette over a whole animation**, or over all of a unit's animations [G5][G6][G2]; collapse held poses automatically [G6]; remove specks under 4 px and add an optional 1 px outline [G2].
- For directions, generate the 5 unique facings and mirror the other 3 (a common practice [YT4][HF4]). Otherwise, if asymmetric props matter, use a 3D proxy, because diffusion models produce "99 % mirrored" views [R7].

## Sources

- [R1] https://www.reddit.com/r/StableDiffusion/comments/1n51gg9/
- [R2] https://www.reddit.com/r/StableDiffusion/comments/1n87toz/
- [R3] https://www.reddit.com/r/StableDiffusion/comments/1vw755b/
- [R4] https://www.reddit.com/r/comfyui/comments/1sve5id/
- [R5] https://www.reddit.com/r/aigamedev/comments/1o8kzwh/
- [R6] https://www.reddit.com/r/StableDiffusion/comments/1wi0mpd/
- [R7] https://www.reddit.com/r/aigamedev/comments/1tlidho/
- [R8] https://www.reddit.com/r/StableDiffusion/comments/1m833n4/
- [WF1] https://pastebin.com/raw/hPLdGbAZ (workflow linked in R1/R2)
- [G1] https://github.com/mor-o/comfyui-2d-character-pipeline
- [G1a] https://github.com/mor-o/comfyui-2d-character-pipeline/blob/main/docs/workflow-2-video-gen.md and .../docs/workflow-3-spritesheet.md
- [G2] https://github.com/harumaxy/pixel-2d-game-art
- [G3] https://github.com/KohakuBlueleaf/PixelOE
- [G4] https://github.com/jenissimo/unfake.js
- [G5] https://github.com/KennethJAllen/proper-pixel-art
- [G6] https://github.com/envy-ai/ComfyUI-Krea2-Pixel-Art-Refiner
- [G7] https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- [G8] https://github.com/Tencent-Hunyuan/Hunyuan3D-2
- [G9] https://github.com/FromAriel/pixel-forge
- [G10] https://github.com/theamusing/perfectPixel
- [G11] https://github.com/Hugo-Dz/spritefusion-pixel-snapper
- [G12] https://github.com/dimtoneff/ComfyUI-PixelArt-Detector
- [CV1] https://civitai.com/models/2425415 (API: https://civitai.com/api/v1/models/2425415)
- [CV2] https://civitai.com/models/1779447
- [CV3] https://civitai.com/models/2172038
- [CV4] https://civitai.com/models/1924815
- [CV5] https://civitai.com/models/1595383
- [CV6] https://civitai.com/models/1475948
- [CV7] https://civitai.com/models/2206848
- [CV8] https://civitai.com/models/2149265
- [CV9] https://civitai.com/models/2356302
- [CV10] https://civitai.com/models/2177558
- [CV11] https://civitai.com/models/375001
- [CV12] https://civitai.com/models/1397312
- [CV13] https://civitai.com/models/2646716
- [CV14] https://civitai.com/models/810089
- [CV15] https://civitai.com/models/56307
- [HF1] https://huggingface.co/theamusing/wan2.2_pixel_walk_lora
- [HF2] https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles
- [HF3] https://huggingface.co/Mystic07/flux-lora-spritesheet
- [HF4] https://huggingface.co/Onodofthenorth/SD_PixelArt_SpriteSheet_Generator
- [HF5] https://huggingface.co/e-n-v-y/Krea-2-Pixel-Art
- [HF6] https://huggingface.co/MiniMaxAI/MiniMax-H3
- [HF7] https://huggingface.co/Comfy-Org/MiniMax-H3 (file tree via https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/tree/main/diffusion_models)
- [HF8] https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE
- [HF9] https://huggingface.co/Lod34/Animator2D-v1.0.0
- [HF10] https://huggingface.co/nerijs/pixel-art-xl
- [DC1] https://docs.comfy.org/tutorials/video/minimax/minimax-h3
- [IT1] https://rotinda-dag.itch.io/ai-gen-sprite-tutorial (redirected from tawusgames.itch.io)
- [P1] https://arxiv.org/abs/2410.06236
- [P2] https://arxiv.org/abs/2208.06413
- [P3] https://arxiv.org/abs/2409.10721
- [P4] https://arxiv.org/abs/2412.03685
- [YT1] https://www.youtube.com/watch?v=nIAIxvNUrdU
- [YT2] https://www.youtube.com/watch?v=7ChVezZPv64
- [YT3] https://www.youtube.com/watch?v=sTuzV9RxWro
- [YT4] https://www.youtube.com/watch?v=dudo8WjtSqM
- [YT5] https://www.youtube.com/watch?v=TWvN0p5qaog
- [CW1] https://comfy.org/workflows/templates-sprite_sheet-fe5600667e2c/

## Blocked

- reddit.com · bot check (FlareSolverr); old.reddit login required · resolved: the throwaway Playwright container loaded search pages and threads. Only the top ~14 comments per thread were read; open the threads in a browser for full comment trees.
- WebSearch on reddit.com · crawler blocked · resolved via Playwright search pages.
- youtube.com/watch?v=bfpHce8zThw ("Turn 1 Image into a Playable Character FOR FREE (Local AI Workflow)") · metadata not returned · open it in a browser and paste the description or workflow link.
- YouTube tutorial contents · descriptions only; videos not watched · paste transcripts if step-level detail is needed.
- huggingface.co/krea/Krea-2-Turbo · login required (gated) · log in and accept the gate to read the card (only needed if Krea 2 is considered as a base).
- MiniMax H3 VRAM on 24 GB · not documented by Comfy docs or the card · needs a local test.
- docs.comfy.org MiniMax H3 guide · no VRAM or quant guidance on the page · see previous line.
