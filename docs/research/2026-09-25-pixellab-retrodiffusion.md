# How do PixelLab and Retro Diffusion generate pixel-art character animations from a single sprite?

Date: 2026-09-25. Extends `docs/research/2026-09-25-keyframe-first-sprite-animation.md` (branch 3, sources [PL1–PL3], [RD1–RD2]); what that file already says is not repeated. Web sources fetched 2026-09-25. Items marked **(inference)** are my reading of the evidence, not a vendor statement.

## Short answer

PixelLab is not one method but a stack of engines behind one API: an old small-canvas diffusion model that paints a fixed window of 3–4 frames at a time (skeleton- or text-conditioned, with frozen frames passed in as inpainting), template skeletons from a 3D mannequin, newer "v3" start-frame (+ optional end-frame) video-style models including a "skeleton video model", a licensed third-party video model (MiniMax H3), and "Pro" modes that pack every frame (or every direction, plus the reference) into one grid image. Retro Diffusion uses per-action trained models ("their own specific model", not ControlNet) that animate an uploaded start frame into 4–16 frames at native 32–256 px, plus fixed-format prompt-only sheet styles (e.g. 48 px, 4 directions × 4 walk frames); its image models are named "Retro Diffusion FLUX", and the animation model is "absolutely massive" and cannot be distributed for licensing reasons. Retro Diffusion's terms say it does not claim ownership of your designs and you "retain full rights to them".

## Findings — PixelLab

### Inputs and engines (animation)

- **v3 text animation** (`POST /animate-with-text-v3`): inputs are one `first_frame` (≤256×256), an optional `last_frame`, and an `action` text; `frame_count` 4–16, even, default 8; total budget `width × height × frame_count ≤ 524,288`; typical 30–180 s. [PLAPI]
- v3 has a `drift_threshold` "colour de-flicker": "Frames whose foreground drifts from the first frame beyond this are corrected toward it" — identity is held partly by a post-pass that pulls colours back to frame 0. [PLAPI]
- v3 via the character system: `custom_start_frame` replaces the stored rotation image as the start; `end_frame` "enables interpolation mode: the model animates between the start frame … and this end frame"; `keep_first_frame` (default true) stores the reference as frame 0, so 8 requested frames store 9. [PLAPI]
- v3 cost is `ceil(canvas²·frames/65536)` per direction on the generated canvas, and "a 96px character commonly animates at ~132px" because the canvas grows to fit the motion. [PLMCP] **(inference)** A cost that scales with pixels × frames and a canvas that grows around the silhouette is the signature of a video-style model generating all frames jointly from the start frame, not per-frame image generation.
- **Skeleton v3** (`POST /animate-with-skeleton-v3`, beta): "Animate a sprite by posing it: one skeleton per frame, drawn from a single reference image"; 3–15 frames, canvas ≤256; skeletons are 18 humanoid joints as 0–1 fractions, with optional `depth` taken "from the standing pose of `template_id`" (mannequin, bear, cat, dog, horse, lion). [PLAPI]
- Skeleton v3 needs `first_frame_keypoints` — the skeleton of the pose the sprite is already in: "The model redraws this frame first and learns the background key and the colours from that redraw", and that frame is not returned. [PLAPI]
- The character endpoint describes the same engine as "the skeleton video model, which moves the character instead of redrawing it each frame — steadier identity and colours", 2–4 generations and ~3–5 min per direction. [PLAPI]
- **Legacy skeleton animation** (`POST /animate-with-skeleton`): sizes 16–256; `skeleton_keypoints` "Requires EXACTLY 3 frames — the model is a 3-frame window"; `inpainting_images` are "Images used for showing the model with connected skeleton"; plus `reference_image`, `init_images`, `mask_images`, `color_image` (forced palette), `guidance_scale`. [PLAPI] The v1 API summary called it "Generate 4 frames of an animation from skeleton poses". [PLV1]
- PixelLab's skeleton tutorial shows how the window is used: chapters "Freeze 1, Generate 3 frames" and "Freeze 3, Generate 1 frame", plus inpainting and init-image passes. [PLYT1] **(inference)** The legacy model is a fixed 4-slot sheet: frozen (already-good) frames sit in some slots as inpainting context and the model fills the rest, so long animations are built by sliding the window — per-frame generation with neighbours as the identity anchor.
- **Legacy text animation** (`POST /animate-with-text`): 64×64 only; `n_frames` 2–20 is the "Length of full animation (the model will always generate 4 frames)", with `start_frame_index` choosing which 4; `inpainting_images` and `mask_images` take 4 slots each. [PLAPI] The docs page says it can "generate 3 frames from one idle image", create one frame matching an existing flow, or "interpolate 2 frames between first and last positions". [PLDOC-ANIM]
- **Template animation** (character `mode: "template"`): "skeleton-based from template_animation_id (1 gen/direction)"; `ai_freedom` 0–900 controls deviation from the template skeleton ("0 = rigid template following"). [PLMCP] Templates are named motion clips with fixed frame counts, e.g. `walking-4-frames`, `walking-6-frames`, `walking-8-frames`, `running-4/6/8-frames`, `breathing-idle`, `fight-stance-idle-8-frames`, `cross-punch`, `fireball`. [PLMCP]
- **PixMiniMax** (`POST /animate-pixminimax`, beta): "Powered by MiniMax H3"; first frame + optional last frame + motion text; frame count a multiple of 4 from 4 to 40; frame 0 is "your input frame unchanged"; "sizes just above 64 or 128 pixels are cheaper than sizes just below them because the model works on an upscaled canvas". [PLAPI]
- **Pro text animation** (`POST /animate-with-text-v2`): 32 and 64 px return 16 frames; 128–256 px return 4 frames. [PLAPI] The docs page gives the layout: 32/64 px → "16 frames in a 4×4 grid", 65–256 px → "4 frames (2×2 grid)", 20–40 generations. [PLDOC-PRO] In character `pro` mode frame count "is fixed by character size (≤64px → 16 frames, >64px → 4 frames)". [PLMCP]
- Pro edit and outfit transfer state the grid mechanism outright: "All frames are packed into a single grid image for editing, so larger frames fill the fixed generation canvas faster"; limits 16 frames at 16–64 px (4×4), 9 at 65–80 px (3×3), 4 at 81–256 px (2×2). For outfit transfer "the reference uses one slot, so each tier holds one fewer frame" (15 / 8 / 3). [PLAPI] This is whole-sheet generation with the identity reference placed inside the same image.
- Pro multi-direction animation "generates directions sequentially, using completed sides as reference (20-40 gen/direction)". [PLAPI]
- **Interpolation (Pro)** (`POST /interpolation-v2`): start and end keyframe images + action text → "typically 4-8 frames", sizes 16–128 px. [PLAPI] Keyframe-then-interpolate exists but is a side tool, as the prior file noted.

### Rotation / directions

- Legacy `POST /rotate`: one image in, one image out, parameterised by `direction_change` (−180…180°) and `view_change` (−90…90°) or from/to direction and view; sizes 16–128. [PLAPI] The guide recommends either one camera-facing reference for all 7 other views, or the incremental method — "Update the reference image after each generation so that the rotation is always 45 degrees from the previous direction" — which is better per step but accumulates errors; it also suggests flipping a good opposite direction. [PLROT]
- v3 rotations (`POST /generate-8-rotations-v3`, `create-character-v3`): one reference in, 8 views out, ≤256 px; "The frontend Character Creator enforces [south-facing] and the v3 model is trained around a south-facing input." [PLAPI]
- Pro rotations: "all 8 rotational views … in one go … a single 3×3 grid (8 directions plus center reference)", ≤168×168 per view, 20–40 generations. [PLDOC-ROT] **(inference)** 3 × 168 ≈ 504, so the grid fits a ~512 px generation canvas with the reference in the centre cell.
- Standard character creation is "Template-based skeleton generation" from a 3D template; users may upload some directions, which are "used as-is (frozen)", and the rest are generated; `south` is required for bipeds. [PLAPI] After generation "the 3D template the skeleton estimator fits to the generated frames" is chosen by `template_id` "so the character can be animated". [PLAPI]
- `pixellab-code` maintains a fork of `mattloper/chumpy` (the autodiff library used by the SMPL body-model code). [PLGH] **(inference, weak)** This hints at SMPL-style 3D body fitting behind the mannequin template and skeleton estimation.

### Small sizes

- Image models are sized in true pixels: Pixflux 32×32 minimum area up to 400×400 area (tier-dependent), recommending 2:3 canvases such as 48×64 for characters; Pixen 16 px minimum side to 512×512 area, sides divisible by 4. [PLFLUX][PLAPI]
- Pro image generation returns more images at smaller sizes — ≤42 px: 64 images (8×8), 43–85 px: 16 (4×4), 86–170 px: 4 (2×2) — i.e. many small sprites tiled into one fixed canvas. [PLAPI] **(inference)** The Pro family generates at ~340–512 px and slices; native pixel look comes from the model and cleanup, not from a 32 px diffusion canvas.
- Cleanup endpoints exist for the grid problem: `unzoom` "detects the underlying pixel grid and downsamples back onto it"; `correct-pixelart` removes "stray and anti-aliased pixels" and tightens the palette while keeping the grid. [PLAPI]

### Base models

- PixelLab names its models (Pixflux, Bitforge, Pixen, v3, Pro, Pro Flash) but does not disclose bases; the only named third-party model is MiniMax H3 behind PixMiniMax. [PLAPI]
- The Pixflux docs page lives at the slug `/docs/tools/create-image-flux`. [PLFLUX] **(inference, weak)** Suggests a Flux-derived model; not confirmed anywhere.
- PixelLab was started by two brothers ("a project that me and my brother started"). [PLYT2] No founder posts on model internals were found.

## Findings — Retro Diffusion

### Inputs and engines (animation)

- **Advanced animations** (`rd_advanced_animation__*`): `input_image` required — "upload *any* pixel-art image as a start frame"; width/height match the frame, 32–256 px; `frames_duration` 4, 6, 8, 10, 12 or 16; output GIF or `return_spritesheet`. Actions: walking, idle, jump, crouch, attack, destroy, custom_action, subtle_motion, rotate. [RDGH]
- RD's own rules for it: "Send the native-resolution frame" (a 96 px sprite exported at 4× is rejected); "Give motion room" (sprites touching the edge animate badly); 8 frames for loops, 6 for a snappy action, 10–12 for ambient motion; "Animations fail/time out more often than stills". [RDGH] The style pages add "Best results when uploading a neutral pose." [RDSTY-WALK]
- **Prompt-driven sheet styles** (`rd_animation__*`) are fixed formats: `four_angle_walking` 48 px "4 direction, 4 frame long walking"; `battle_sprites` "4 direction sprites with 3-frame idle, 6-frame walk, 2-frame jump, and 3-frame attack"; `small_sprites` 32 px 4-direction with walking, attacking, looking around, surprised and laying down; `any_animation` 64 px; `big_animation` 128 px; `8_dir_rotation`; `vfx`. [RDGH][RDSTY-4W][RDSTY-BS][RDSTY-SS][RDSTY-ANY] Replicate's card: "Each style has a dedicated look and format, ensuring positioning and action consistency across generations." [RDREP] **(inference)** The prompt-only styles generate a whole fixed-layout sheet in one image.
- On Replicate, "Image references are not used as direct initial images for generation" — the reference conditions the model rather than seeding img2img. [RDREP]
- Method, from the developer (Astropulse) on itch.io: "The animations are not done using controlnet, they are made using their own specific model. This is also why other actions haven't been added yet, we dont have the data to train a model for it." [RDITCH] So each action is a trained capability, not a pose-control trick.
- Size and licence of that model: "it is absolutely massive, beyond what is even reasonable for a 5090 to run" and "licensing issues, as we cannot distribute the model for local use". [RDITCH] On in-betweens: "in-betweens are actually more complex than just regular animation … it will only be available on the site due to the massive size of the models involved." [RDITCH] **(inference)** A >32 GB model whose weights RD cannot redistribute fits a fine-tuned large video model or a FLUX.1-dev-class base under a commercial licence; RD does not say which.
- Astropulse's public `mixamotoopenpose` (MIT) converts Mixamo animations to OpenPose image sequences, GIFs or grid sheets, "for AI image generation and dataset creation", with X/Y/Z rotation for view angles. [RDMIX] **(inference)** A plausible way RD built per-action, per-direction pose data, but no source ties it to the web model.

### Rotation / directions

- `rd_advanced_animation__rotate` "generates the other seven directional views of the uploaded subject"; spritesheet output is "a 3×3 PNG sheet … each cell the input size, center cell empty", the same layout as `rd_animation__8_dir_rotation`; width/height multiples of 8. [RDGH]
- `rd_plus__character_turnaround` and `rd_fast__character_turnaround` are still-image turnaround styles. [RDGH]
- For stills, RD Pro takes up to 9 `reference_images` "to keep a character … consistent": generate the character once, then pass it back as a reference. [RDGH]

### Small sizes and palette

- Output is at native pixel size by default (`upscale_output_factor` 1); sizes 12–512 overall; `input_palette` "Constrain[s] output colors to a palette image" with `return_pre_palette` to get the un-quantised image too. [RDGH] **(inference)** `return_pre_palette` shows palette quantisation is a post-pass after generation.
- Astropulse's open tools show the grid technique: `pixeldetector` finds the pixel grid from peaks in horizontal/vertical colour differences, then downsizes with "kCentroid" (k-means per tile, keep the dominant colour); `K-Centroid-Aseprite` is "A new downsampling algorithm using k-means"; `sd-palettize` palettises generated images. [RDPD][RDKC][RDGHP]

### Base models

- The site names its image model "Retro Diffusion FLUX"; the animation set debuted with "Four Angle Walking". [RDLM]
- RD's first web Flux model launched in late 2024 per Astropulse's X post (title visible in search; post itself blocked). [RDX]
- The local Aseprite extension and the $50 standalone model are different, smaller models: "THESE ARE NOT THE MODELS USED ON RETRODIFFUSION.AI"; the standalone model is "fully compatible with most Stable Diffusion 1.5 based generators". [RDGUM] The extension needs ~12 GB VRAM, has ControlNet, and "can't handle any kind of animation". [RDITCH-P][RDITCH]
- Astropulse's only Hugging Face upload is `sd1-5_control_lora_pose_r_128`, an SD 1.5 pose control-LoRA (OpenRAIL-M) made with Stability's control-LoRA script. [RDHF] Hugging Face search finds only third-party `RetroDiffusion` re-uploads (not Astropulse's; not examined); Civitai search found nothing from RD. [RDHF]
- Training data: "trained on licensed assets from Astropulse and other pixel artists with their consent" (stated for the extension model). [RDITCH-P]

### Terms and output ownership (Astropulse LLC, last updated 19 Aug 2025)

- The terms grant a "non-exclusive, non-transferable, revocable license" to access the Services and "Create designs using Customer Material ("Designs") through the Services". [RDTOS]
- Section 7: "We do not claim ownership over your Designs or Customer Materials. You retain full rights to them." You are "solely responsible for your content". [RDTOS]
- Astropulse owns the site's Content and Marks; feedback "Submissions" are assigned to Astropulse; prohibited uses include "Using automated tools like bots or scrapers" and "Competing with us using the Services". [RDTOS]
- The terms contain no clause about training on user uploads or outputs. [RDTOS]
- The learn-more FAQ: "All art you generate is yours to use for personal or commercial projects." [RDLM]
- The local extension's licence is different: per Astropulse the code and models "are owned by Astropulse LLC, and not able to be used commercially" (the comment is cut off in the page text; the full licence ships inside the extension). [RDITCH]

## Comparison

| | PixelLab | Retro Diffusion |
|---|---|---|
| Input | 1 sprite + text (v3, PixMiniMax, Pro); 1 sprite + per-frame 18-joint skeletons (skeleton-v3, legacy); stored character + template id; 2 keyframes (interpolation, v3 `end_frame`) [PLAPI] | 1 start frame (advanced, 32–256 px) or prompt only (fixed-format sheets) [RDGH] |
| Method | Legacy: fixed 3–4-frame window, frozen frames as inpainting [PLAPI][PLYT1]. v3 / skeleton-v3: start-frame video-style models [PLAPI]. PixMiniMax: MiniMax H3 [PLAPI]. Pro: all frames in one grid [PLAPI][PLDOC-PRO] | Per-action trained model from a start frame, not ControlNet [RDITCH]; prompt styles output a fixed sheet [RDGH] |
| Frames | 3–15 (skeleton-v3), 4–16 even (v3), 4–40 (PixMiniMax), 16 or 4 (Pro), template-fixed [PLAPI][PLMCP] | 4/6/8/10/12/16 (advanced); fixed per style, e.g. 4×4 walk at 48 px [RDGH] |
| Sizes | ≤256 for most; Pro 32–256 with grid tiers; interpolation 16–128 [PLAPI] | 32–256 advanced; 32/48/64/128 fixed styles [RDGH] |
| Identity across frames | Redraw first frame from its skeleton to learn colours; colour de-flicker to frame 0; reference slot in the grid; frozen frames [PLAPI] | Start frame conditions the model (not img2img); palette lock [RDREP][RDGH] |
| Directions | v3: south-facing reference → 8 views; Pro: 3×3 grid with centre reference; standard: 3D template, uploaded views frozen; animation directions sequential with finished sides as reference (Pro) [PLAPI][PLDOC-ROT] | Rotate → 3×3 sheet, centre empty [RDGH] |
| Base models | Undisclosed except MiniMax H3 [PLAPI] | "Retro Diffusion FLUX" for images [RDLM]; animation model undisclosed, very large, non-redistributable [RDITCH] |

## Techniques worth borrowing (inference, for this repo)

- **The reference inside the sheet.** PixelLab's outfit transfer and Pro rotations spend one grid cell on the reference so the model sees it with shared attention [PLAPI][PLDOC-ROT]. With Qwen-Image-Edit-2509 this is a 2×2 or 3×3 canvas: reference in one cell, skeletons in the others. Caveat: the prior file's pose-flattening result on multi-cell strips (`DEVLOG.md:2429-2436`).
- **Redraw the reference pose first.** Skeleton-v3 regenerates frame 0 from its own skeleton to calibrate colours and background [PLAPI]. That gives a cheap drift test: if the model cannot redraw the accepted view in its own pose, no other frame will hold.
- **Colour de-flicker toward frame 0** after generation [PLAPI] matches this repo's post/palette snap and could run per frame before pick.
- **Canonical south-facing input for rotation** [PLAPI], and a fixed 3–4-frame window with frozen neighbours [PLAPI][PLYT1], are both simple to copy with the existing Qwen two-image graph.

## Sources

- [PLAPI] https://api.pixellab.ai/v2/openapi.json (endpoint descriptions and schemas; human view https://api.pixellab.ai/v2/docs); index https://api.pixellab.ai/v2/llms.txt
- [PLV1] https://api.pixellab.ai/v1/openapi.json
- [PLMCP] https://api.pixellab.ai/mcp/docs
- [PLROT] https://www.pixellab.ai/docs/guides/rotating-a-character
- [PLDOC-ROT] https://www.pixellab.ai/docs/tools/create-8-rotations-pro
- [PLDOC-PRO] https://www.pixellab.ai/docs/tools/animate-with-text-pro
- [PLDOC-ANIM] https://www.pixellab.ai/docs/tools/animation
- [PLFLUX] https://www.pixellab.ai/docs/tools/create-image-flux
- [PLGH] https://github.com/pixellab-code and https://github.com/pixellab-code/chumpy
- [PLYT1] https://www.youtube.com/watch?v=zBfVT5pwCSs (PixelLab, "Tutorial: How to animate pixel art using a skeleton", 2024-10-26, description)
- [PLYT2] https://www.youtube.com/watch?v=ufQ72nGORC0 (PixelLab, rotation tutorial, 2023-11-27, description)
- [RDGH] https://github.com/Retro-Diffusion/api-examples (README)
- [RDREP] https://replicate.com/retro-diffusion/rd-animation/readme
- [RDLM] https://retrodiffusion.ai/learn-more (rendered in Playwright)
- [RDSTY-4W] https://retrodiffusion.ai/styles/four-angle-walking/
- [RDSTY-BS] https://retrodiffusion.ai/styles/battle-sprites/
- [RDSTY-SS] https://retrodiffusion.ai/styles/small-sprites/
- [RDSTY-ANY] https://retrodiffusion.ai/styles/any-animation/
- [RDSTY-WALK] https://retrodiffusion.ai/styles/walking/
- [RDTOS] https://retrodiffusion.ai/terms → PDF https://retrodiffusion.ai/assets/terms-8312c717.pdf
- [RDITCH] https://astropulse.itch.io/retrodiffusion/comments (pages `?after=0`, `?after=17`)
- [RDITCH-P] https://astropulse.itch.io/retrodiffusion and https://astropulse.itch.io/retrodiffusionai
- [RDGUM] https://astropulse.gumroad.com/l/RetroDiffusionModel
- [RDMIX] https://github.com/Astropulse/mixamotoopenpose
- [RDPD] https://github.com/Astropulse/pixeldetector (README and `pixeldetector.py`)
- [RDKC] https://github.com/Astropulse/K-Centroid-Aseprite
- [RDGHP] https://github.com/Astropulse (repo list, incl. sd-palettize)
- [RDHF] https://huggingface.co/Astropulse/sd1-5_control_lora_pose_r_128; https://huggingface.co/api/models?search=retrodiffusion; https://civitai.com/api/v1/models?query=retro%20diffusion
- [RDX] https://x.com/RealAstropulse/status/1861791765851738590 (blocked; title seen in search results only)

## Blocked

- retrodiffusion.ai/app/guide/api (full API reference) · login required · log in and paste the animation section; not bypassed (the public GitHub README covered the parameters).
- x.com (Astropulse and @PixelLabAI posts) · 402 / login required · paste the posts, or give an X session; the RD Flux launch date rests on a search-result title.
- Retro Diffusion and PixelLab Discord announcements · login required · paste the #announcements posts on the animation and v3 / skeleton-v3 models.
- pixellab.ai/docs/tools/skeleton-animation · page content not served (marketing text only) · open in a browser and paste; the YouTube chapter list and API spec covered the window mechanics.
- Model internals for both vendors · not disclosed · ask on their Discords (PixelLab: base of v3 / Pro / skeleton video model; RD: base of the animation model).
- Retro Diffusion extension licence text · ships inside the paid extension · paste the licence file if the local model's terms matter.
