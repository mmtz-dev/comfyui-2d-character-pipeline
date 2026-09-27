# Keyframes first: is "image-model keyframes → video-model in-betweens" a viable replacement for route V?

Date: 2026-09-25. Four parallel research branches (in-betweening models, consistent keyframe generation, industry practice, codebase audit), merged here. Every claim carries its source; web sources were fetched 2026-09-25. Paths are absolute under `/home/workstation/Repos/game-dev/framework-ladder/tools/anim-pipeline/` unless stated.

## Verdict

Yes, keyframes-first is viable on the A5000 under Apache-2.0 weights, and it attacks the right problem: today's drift comes from Wan following the mannequin's depth/body channel, and a keyframe generated from the accepted facing view pins the costume in pixels at each phase instead. But the research changes the shape of the proposal. At 48 px and 6–12 frames per action, hand animators and the pixel-native AI tools (PixelLab, Retro Diffusion) treat every frame as a deliberately posed drawing, so there is almost nothing left to "tween"; a video in-betweener adds morphing risk on large pose gaps and a costly Qwen↔Wan model swap for little gain. The recommended route is therefore **generate every frame as a posed keyframe with Qwen-Image-Edit-2509 (already installed) conditioned on the facing view plus a rig-rendered skeleton**, and keep a Wan 2.2 Fun-InP / Wan 2.1 FLF2V in-betweener only as an optional tool for a gap that a single image pass cannot bridge. Whether this fixes drift is not yet measured: the one recorded Qwen reference+pose test flattened poses back toward the standing reference on a multi-cell strip (`DEVLOG.md:2429-2436`), and single-cell keyframes were never tried. That is the experiment below.

## What each branch found (one paragraph each)

**Branch 1, in-betweening models.** The permissive, 24 GB-capable in-betweeners are all Wan: Wan 2.1 FLF2V 14B, Wan 2.2 I2V-A14B first/last frame, Wan 2.2 Fun-InP (A14B and 5B) and Wan 2.1 VACE (14B GGUF or 1.3B), all Apache-2.0 with core ComfyUI nodes (`WanFirstLastFrameToVideo`, `WanFunInpaintToVideo`, `WanVaceToVideo`) [W1][W2][C1][C2][C3][C4]. They accept exactly two pins (first + last); a 3-keyframe attack means chaining two segments sharing the middle frame. ToonCrafter is Apache-2.0 and cartoon-tuned but fixed at 16 frames, 512×320 [T1][T2]. ToonComposer is the closest fit to "one colour reference + several keyframe sketches" but needs ~57 GB, so RunPod only [T3]. LTX-Video/LTX-2 (multi-keyframe) carry a custom licence with a revenue clause [L1][L2]; Framer, CogVideoX-Interpolation, AnimeInbet, LVCD and SVD-based work are non-commercial, unlicensed or use-restricted [F1][CG1][A1][A2][S1]. No primary source tests any of these on ~48 px pixel art. Published failure modes: the path between two distant keyframes is ambiguous and morphs (FCVG) [P1]; fine-tuned I2V models weakly obey the end frame (EF-VI) [P2]; cartoon motion breaks live-action priors (ToonCrafter's own motivation) [T1].

**Branch 2, consistent keyframes.** Qwen-Image-Edit-2509 (Apache-2.0, up to 3 input images, "native support for … keypoint maps", "pose transformations") is the best fit and is already installed [Q1][Q5]; the pattern is image1 = character, image2 = skeleton or depth. On 2511 a user reports the skeleton trick regressed [Q2][Q3], but an Apache-2.0 LoRA (`QIE2511_3Dpose2Chara`) takes a **3D mannequin render** as image2 and keeps "art style, face, hair and outfit" [Q6] — which is exactly what `rig` already produces. The AnyPose LoRA warns it struggles with "flat art styles (cel-shading, pixel art)", so generate at illustration resolution and pixelate afterwards [Q7]. Nearly every other reference-identity method (Flux Kontext dev, FLUX.2 dev, UNO, DreamO, InstantCharacter) sits on FLUX.1-dev or FLUX.2-dev, whose licences permit commercial *outputs* but restrict *model use* to non-commercial purposes [B1][B2][B3][U1][D1][I1]; Qwen-Image-2.1 is Qwen Research (non-commercial) [Q4]. FLUX.2 klein 4B is the one permissive Flux (Apache-2.0, ~13 GB) but has no pose input [B4]. Per-character LoRA: Qwen-Image trains on 24 GB at 3-bit (ai-toolkit, MIT) [K1]; Qwen-Image-Edit-2509 needs 32 GB, so RunPod [K2].

**Branch 3, how it is actually done.** Saint11's pixel tutorials: 6-frame walk at low resolution (contact, down, down+, prepare-passing, passing, up), 8-frame run (4 key poses × 2 legs), attack = 3 action frames + 4 recovery with *no* anticipation frame ("players will feel input lag"), idle 2–6 frames, jump split into short clips [S11a][S11b][S11c][S11d][S11e]. LPC: walk 8, run 8, slash 6, thrust 8, jump 6, idle 2 [LPC1]. Retro Diffusion's 48 px walking style uses 4 frames per direction [RD2]. Aseprite sets duration per frame, so "12 fps" really means holding key frames longer [AS1]. Tool approaches: PixelLab is skeleton-per-frame or whole-sequence from a start frame (4–16 frames, pixel-native model, closed) [PL1][PL2][PL3]; Retro Diffusion generates the whole 4–16-frame sequence from a start frame, palette-locked (closed) [RD1]; Ludo and Scenario run generic video models and pick frames — the same shape as route V [LU1][SC1]; God Mode rigs body-part layers [GM1]. Keyframe+interpolation exists only as a side option (PixelLab's old Interpolate, Ludo's 3 keyframes, ToonCrafter). The only open sprite-specific model, Sprite Sheet Diffusion, is pose-conditioned SD 1.5 (OpenRAIL-M, not Apache) trained on 619 triplets and showed identity drift [SSD1][SSD2].

**Branch 4, the codebase.** Views, `ACTION_PHASES`, `PhasePlan.keys()`/`phase_instants()`, `pick --phases`, post/parts/palette, the board, matrix batching, concept, retouch, outcome tracking and export all carry over with small changes (details below). Only one video model is installed — Wan 2.2 Fun-Control 5B — and its node has no start/end image input (`nodes_wan.py:119-135` in the ComfyUI checkout); ComfyUI 0.35.0 ships the FLF2V, Fun-InP, VACE and RIFE/FILM nodes but none of their weights are on disk (`gpu/models.lock:86-98`). The bake-off's route K was text+pose stills with rig-space tweening and **no reference image in the frames** (`DEVLOG.md:1563-1590`); it lost on identity ("three knights, not one", `DEVLOG.md:1628-1632`). The DEVLOG's "don't re-litigate keyframes-first" ruling (`DEVLOG.md:407-414`) rests on K and does not cover a route where every keyframe is conditioned on the accepted view.

## Comparison of candidate routes

Effort is relative to the current codebase. VRAM notes assume the A5000 is Ampere: INT4/INT8 only, fp8 weights load but give no speedup, and Qwen-Edit at fp8 did not fit beside anything else (`gpu/README.md:209-216`).

| Route | Identity hold (as reported) | Pixel-art suitability | VRAM on 24 GB | ComfyUI support | Licence | Effort to adopt |
|---|---|---|---|---|---|---|
| **V (current):** rig clip → Wan 2.2 Fun-Control 5B with ref + depth → pick → post | Holds at rest, drifts at extremes; the depth body wins over the reference (`DEVLOG.md:1623-1627, 1941-1946`) | Proven end to end; the post stage exists | Measured 17.6 GB resident (`DEVLOG.md:1609`) | Installed (`gpu/models.lock:96`) | Apache-2.0 | 0 — it is what runs today |
| **A. Keyframes (Qwen-Edit-2509 + skeleton) → Wan Fun-InP 5B / FLF2V 14B in-betweens → pick → post** | Costume pinned at the two ends of each segment; the middle can morph on big gaps [P1][P2]; a second pin per segment is all Wan offers [C1][C3] | Untested on pixel art by any source; generate ~480p, downscale as today | Fun-InP 5B ≈ installed 5B class (~10 GB bf16) [VF1]; Fun-InP A14B fp8 measured 83–89 % of a 24 GB 4090D at 640×640×81 [C3]; FLF2V 14B fp8 needs a local benchmark [W1] | Core nodes `WanFunInpaintToVideo` / `WanFirstLastFrameToVideo` (present in 0.35.0, `comfy_extras/nodes_wan.py:184,256`) | Apache-2.0 [W1][W2] | Medium: download weights, one new stage, a second `run_cell_chain`; Qwen↔Wan swap cost per cell (K measured 10.4–10.7 min from reloads, `DEVLOG.md:1603-1606`) unless work is batched model-by-model |
| **B. Keyframes → Wan VACE 14B GGUF (first/last) → pick → post** | Same two-pin limit; upstream notes "VACE treats `reference_image` as loose guidance, not strict visual preservation" (`docs/workflow-4-cosmetic-inpaint.md:200,211`) | As A | Q5_K_M 13 GB, Q6_K 14.5 GB [VG1]; 1.3B is 480p-only at ~7 GB [V1] | Core `WanVaceToVideo`; GGUF needs ComfyUI-GGUF, which the repo deliberately omits (`gpu/custom_nodes.txt:36-42`) | Apache-2.0 [V1] | Medium-high; mid-clip keyframes undocumented [V2] |
| **C. Keyframes → ToonCrafter** | "Success rate is not guaranteed" (authors) [T1]; cel-tuned, handles non-linear motion better than live-action priors | Best of the video models for cel style; fixed 16 frames at 512×320 (landscape canvas, portrait sprite) [T1] | ~24–27 GB official on A100; ~10–12 GB community fp16 builds [T1][T2] | Wrapper custom nodes only (ComfyUI-DynamiCrafterWrapper / ComfyUI-ToonCrafter) | Apache-2.0 [T2] | Medium; adds a custom node family and a fixed frame count that does not match `ACTION_PHASES` |
| **D. Keyframes-only: every frame is a posed image (Qwen-Edit-2509, image1 = view, image2 = rig skeleton at each frame instant); optional RIFE for tiny gaps** ← **recommended** | Every frame anchored on the accepted view; the open risk is pose flattening (`DEVLOG.md:2429-2436`) | Matches how pixel animators work (every frame a chosen pose) [S11a][S11c] and what PixelLab/RD ship [PL1][RD1]; generate illustrated, pixelate after (LoRA authors warn on flat styles [Q7]) | Already fits: Z-Image + Qwen measured 18.8 GB (`DEVLOG.md:1609`); Qwen-Edit INT4 12.65 GB (`gpu/models.lock:91`) | Core `TextEncodeQwenImageEditPlus` (image1..3) [Q5]; `rig(times=, yaw=)` already renders skeletons at exact instants (`anim/stages/rig.py:239-245`) | Apache-2.0 [Q1] | **Low**: no new weights; reuse `identity_pass.json`'s two-image graph shape (`DEVLOG.md:2429-2432`); ~11 s warm per image (`anim/config.py:852-853`) ⇒ ~2–3 min per 12-frame cell |
| **D′. D with Qwen-Edit-2511 + `3Dpose2Chara` LoRA (mannequin render as image2)** | Author claims style/face/hair/outfit kept, pose from image2 [Q6]; card claims "significantly improved" consistency [Q2] | As D | Needs a 2511 build that fits (no INT4 Nunchaku 2511 in the lock; check availability) | Core + LoRA loader; official 2511 workflow [Q3] | Apache-2.0 [Q2][Q6] | Low-medium; a second Qwen-Edit download |
| **E. Per-character LoRA (Qwen-Image 3-bit on 24 GB + InstantX ControlNet Union for pose; or Edit-2509 LoRA on RunPod 32 GB) + D** | Strongest identity lock in common practice (unmeasured here) [K1] | As D | Training 24 GB at 3-bit [K1]; Edit-2509 LoRA is 32 GB → RunPod (`docs/research/2026-09-24-runpod-comfyui.md`) [K2] | Core ControlNet nodes [Q8]; ai-toolkit outside ComfyUI | Apache-2.0 / MIT [K1][Q8] | High: a training step per unit; use only if D drifts |
| **F. Whole-sheet generation (all frames in one grid image)** | No open pixel-native model exists; Sprite Sheet Diffusion overfits (619 triplets, "robot-like features") [SSD1] | Native for PixelLab Pro / RD, but those are closed [PL3][RD1] | n/a | n/a | SSD is SD 1.5 OpenRAIL-M → flag [SSD2]; PixelLab/RD closed SaaS | Not viable as an open route today |
| **G. ToonComposer (one colour keyframe + several keyframe sketches)** | Designed for exactly this task [T3] | Cartoon / cel | ~57 GB → RunPod only [T3] | None found | MIT + Apache-2.0 [T3] | High; cloud fallback if every local option morphs |
| **Rejected on licence** | LTX-Video / LTX-2 (revenue-clause custom licence [L1][L2]); Framer (academic [F1]); CogVideoX-Interpolation (registration + visit cap [CG1]); AnimeInbet (non-commercial [A1]); LVCD (no licence [A2]); AniDoc/SVD (Stability Community [S1]); Flux Kontext dev / FLUX.2 dev / klein 9B / UNO / DreamO / InstantCharacter (non-commercial model use [B1][B2][B3][U1][D1][I1]); Qwen-Image-2.1 (Qwen Research [Q4]); `flux1-dev.safetensors` in the shared library is exposed via `gpu/extra_model_paths.yaml:27-31` but is not in the lock and carries the same non-commercial licence — do not use it | | | | | |

## Recommended route and the experiment

Proposed feature name: **the keyframe-first animation feature**.

Route D, with A as the optional in-betweener for a single large gap. The mannequin's role inverts: it supplies only a skeleton at each frame instant, and the accepted view supplies body and costume. Order work model-by-model (all Qwen passes for a batch of cells, then any Wan passes) so the cold-reload cost stays out of the per-cell time.

### Experiment on `hooded_archer`

Unit state on disk: base and all eight views accepted, turnaround 8/8 done, no action jobs yet (`work/hooded_archer_base/render/view_*.png`, `work/hooded_archer_curation.json`). Parts table: hood, gloves, boots, cloak (`DEVLOG.md:72-74, 95-99`).

**Step 1 — five keyframes, one facing (≈15 min GPU).**
- Action `attack` (budget 12: start 1 / wind-up 3 / action 4 / wind-down 2 / end 2, `README.md:166-173`; `PhasePlan.keys()` names one key slot per phase, `anim/stages/pick.py:276-288`).
- Facing: a drawn side view (`left`; `right` is a mirror, `README.md:627-656`).
- Run `rig(times=phase_instants(), yaw=…)` on the Quaternius attack clip to get five skeleton PNGs (`anim/stages/rig.py:239-245`). No depth.
- For each: Qwen-Image-Edit-2509, image1 = `view_left.png`, image2 = skeleton, prompt of the form "the character in Picture 1 in the exact pose of Picture 2; same hood, cloak, gloves, boots, bow; same proportions; plain background". Four seeds each. Graph: the two-image shape of `gpu/workflows/identity_pass.json`.
- **Pass:** for ≥4 of 5 phases at least one seed (a) keeps all four parts and the bow with no added pauldrons/sleeves/armour, (b) matches the skeleton — DWPose (`anim/stages/pose.py:140`) on the output within the board's existing pose tolerance, and (c) is not flattened back toward the standing view.
- **Fail → D′:** repeat with 2511 + `3Dpose2Chara`, image2 = the mannequin's full render instead of the skeleton [Q6]. Still failing → E (per-unit LoRA).

**Step 2 — all 12 frames, two arms (≈30 min GPU).**
- Arm D: rig at all 12 instants; one Qwen pass per frame, same seed as the winning keyframe seed.
- Arm A: install Wan 2.2 Fun-InP 5B [VF1]; chain segments between the accepted keyframes (frame counts 5/9/13, i.e. 4n+1 [C3]); resample 16 → 12 fps; `pick --phases` with pose tags written by the route itself (option B in `anim/stages/pick.py:646-648`).
- Write frames to `clip/frames/NNNN.png` + `picks.json`, exactly as route K did (`DEVLOG.md:1582-1590`), so `post` runs unchanged (`anim/stages/post.py:524-545`).
- **Pass:** after post at 48 px, the part palette snaps to the base `parts.json` colours on every frame, the board accepts every frame, and a flipbook at 12 fps reads as one character with no limb melting. Compare the two arms on part-colour drift and reject count; keep the arm that passes with fewer rejects, and note whether Arm A's middle frames morph.

**Step 3 — scale check (only if Step 2 passes).** Same for `walk` on two facings (drawn `left`, drawn `back_left`) to confirm identity holds across facings; then the 8-cell matrix cap (`anim/config.py:727`) with a phase-ordered chain.

## What carries over, what becomes dead (branch 4)

Carries over, with the change needed:
- **Views** — become the keyframe input instead of Wan's `ref_image`; no change (`anim/config.py:798-825`, `README.md:598-626`; `TURNAROUND_REFERENCE_BY_DIR` reusable as is, `anim/config.py:669-686`).
- **`ACTION_PHASES` / `PhasePlan.keys()` / `phase_instants()`** — say which keyframes to generate and how many frames each gap needs; no change (`anim/config.py:231-250`, `anim/stages/pick.py:276-288, 431-439`).
- **`pick --phases`** — hard-requires `pose/keypoints.json` (`anim/stages/pick.py:646-648`); either run DWPose over the generated frames or have the new route write tags/timing itself.
- **post / parts / palette** — unchanged if frames land in `clip/frames/` + `picks.json` (`anim/stages/post.py:524-545`); parts need per-frame keypoints (same fix as pick, `anim/stages/post.py:436-466`). Pre-existing: the archer collapses to three teals at the 355×725 → 24×48 downscale (`DEVLOG.md:129-134`) — not route-specific.
- **Curation board** — add the new GPU stage to "regenerate" and `BASE_GATED` (`anim/curation.py:280`, `README.md:774-778`); a per-keyframe accept before in-betweens are spent is a natural addition.
- **Matrix batching** — planning, gates, checkpoints, resume, seed rule and cap stay; `run_cell_chain` is hard-wired rig → clip → pick → post and needs a second chain (`anim/stages/matrix.py:482-539`).
- **Concept (Qwen-Image-Edit)** — the two-image `identity_pass.json` graph is the starting point (`DEVLOG.md:2429-2432`); `view_pass.json` uses image1 + latent (`anim/stages/concept.py:801`).
- **Retouch** — today concept targets only (`README.md:720-737`); pointing it at keyframe attempts lets a bad keyframe be fixed before it is tweened.
- **Run/outcome, export** — add the stage to `GPU_STAGES`; export needs `<unit>_<verb>[_<dir>]` names (`anim/outcome.py:57-69`, `anim/stages/export.py:129`).
- **Rig** — stays as the pose source at exact instants (`anim/stages/rig.py:239-245`, already used by K at `anim/stages/bakeoff.py:561-565`).

Dead or optional:
- **`clip` whole-clip Fun-Control render and the full-body depth blend** (`anim/stages/clip.py:318`, `anim/config.py:650-651, 705-706`) — depth is the documented cause of the body takeover (`DEVLOG.md:1623-1627`). Optional only as route A's in-betweener with a different model.
- **The mannequin as per-frame body shape** — skeleton only, at keyframe instants (`rigs/quaternius/UAL1_Standard.glb`, `rigs/kaykit-adventurers/Knight.glb`).
- **Turnaround** — optional; the accepted views become the identity gate, but its per-facing `parts.json` stays useful (`anim/config.py:639-651`, `README.md:910-912`).
- **Strip / render (Z-Image + identity pass)** — already superseded (`anim/stages/render.py:1-24`); only the graph shape is reused.
- **Face-point hiding, extract, pose** — only if skeletons are still used (`README.md:913-925`).

Models on disk vs needed: installed are Wan 2.2 Fun-Control 5B, Wan 2.2 VAE, umT5 fp8, Qwen-Image-Edit-2509 INT4 Nunchaku, Qwen2.5-VL fp8, Qwen VAE, Z-Image-Turbo int8 + ControlNet Union + pixel LoRA (`gpu/models.lock:86-98`). Route D needs nothing new. Route A needs Wan 2.2 Fun-InP 5B (or FLF2V 14B fp8) weights; route D′ needs a 2511 build plus the LoRA; RIFE/FILM checkpoints would go in a `frame_interpolation` folder for the core `FrameInterpolationModelLoader` node (`comfy_extras/nodes_frame_interpolation.py:16,75`). Two Apache-2.0 Wan 2.2 pixel-sprite LoRAs exist but target I2V-A14B, not the 5B (`docs/research/2026-09-20-pixel-art-loras.md:31-44`).

## Branch status

- Branch 1 (in-betweening models) — **done**; nothing blocked; open items need a local benchmark (VACE mid-clip keyframes, FLF2V 14B VRAM on Ampere, pixel-art behaviour).
- Branch 2 (consistent keyframes) — **done**; two minor blocks, both worked around.
- Branch 3 (industry practice) — **done, partial on licences**: output-licence terms for Retro Diffusion, Ludo and Scenario unverified; PixelLab/God Mode model internals undisclosed.
- Branch 4 (codebase audit) — **done**; read-only, nothing blocked.

## Blocked sources

- huggingface.co FLUX.1-dev `LICENSE.md` — gated (login) — resolved: same text read from the official GitHub repo [B1].
- docs.comfy.org Qwen-Image-Edit-2509 tutorial — 404 (guessed URL) — find the link in the docs sidebar; the 2511 page and core node source were used instead.
- retrodiffusion.ai home and learn-more — JS-rendered, title only — open in a browser and paste the animation and terms sections; RD output licence stays unverified.
- godmodeai.co sprite generator — method undisclosed — ask support or test the API.
- pixellab.ai — model architecture undisclosed — ask on their Discord; docs give inputs/outputs only.
- ludo.ai / scenario.com output licence — not in docs — read their ToS pages.
- Saint11 Patreon long-form tutorials — login required — not attempted; the free GIFs cover frame counts.
- LVCD licence — none in the repo — ask the authors.
- SketchKeyAnime (June 2026 paper) — no code or licence — nothing to unblock yet.

## Sources

Wan / VACE / ComfyUI
- [W1] https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P and https://github.com/Wan-Video/Wan2.1
- [W2] https://github.com/Wan-Video/Wan2.2
- [C1] https://docs.comfy.org/tutorials/video/wan/wan-flf
- [C2] https://docs.comfy.org/tutorials/video/wan/wan2_2
- [C3] https://docs.comfy.org/tutorials/video/wan/wan2-2-fun-inp
- [C4] https://docs.comfy.org/tutorials/video/wan/vace
- [VF1] https://github.com/aigc-apps/VideoX-Fun
- [V1] https://github.com/ali-vilab/VACE
- [V2] https://github.com/ali-vilab/VACE/blob/main/UserGuide.md
- [VG1] https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF

Other in-betweeners and papers
- [T1] https://github.com/ToonCrafter/ToonCrafter and https://arxiv.org/abs/2405.17933
- [T2] https://huggingface.co/Doubiiu/ToonCrafter
- [T3] https://github.com/TencentARC/ToonComposer, https://raw.githubusercontent.com/TencentARC/ToonComposer/main/LICENSE, https://arxiv.org/abs/2508.10881
- [L1] https://huggingface.co/Lightricks/LTX-Video/blob/main/LTX-Video-Open-Weights-License-0.X.txt and https://docs.comfy.org/tutorials/video/ltxv
- [L2] https://github.com/Lightricks/LTX-2, https://raw.githubusercontent.com/Lightricks/LTX-2/main/LICENSE-2_x, https://github.com/Lightricks/ComfyUI-LTXVideo
- [F1] https://github.com/aim-uofa/Framer
- [CG1] https://github.com/feizc/CogvideX-Interpolation and https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE
- [A1] https://github.com/lisiyao21/AnimeInbet
- [A2] https://github.com/luckyhzt/LVCD
- [S1] https://github.com/yihao-meng/AniDoc, https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt, https://stability.ai/license
- [R1] https://github.com/hzwer/ECCV2022-RIFE, https://github.com/google-research/frame-interpolation, https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
- [P1] FCVG, https://arxiv.org/abs/2412.11755
- [P2] EF-VI, https://arxiv.org/abs/2505.21205
- [P3] SketchKeyAnime, https://arxiv.org/abs/2606.19958
- [P4] Generative Inbetweening, https://arxiv.org/abs/2408.15239

Keyframe generation
- [Q1] https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- [Q2] https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- [Q3] https://huggingface.co/Qwen/Qwen-Image-Edit-2511/discussions/23 and https://docs.comfy.org/tutorials/image/qwen/qwen-image-edit-2511
- [Q4] https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE
- [Q5] https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_qwen.py; https://comfy.org/p/supported-models/qwen-image-edit-2509-fp8-e4m3fn/
- [Q6] https://huggingface.co/tori29umai/QwenImageEdit2511_LoRA
- [Q7] https://huggingface.co/lilylilith/AnyPose
- [Q8] https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union and https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_model_patch.py
- [B1] https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev
- [B2] https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
- [B3] https://huggingface.co/black-forest-labs/FLUX.2-dev and https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
- [B4] https://huggingface.co/black-forest-labs/FLUX.2-klein-4B and https://docs.comfy.org/tutorials/flux/flux-2-klein
- [O1] https://github.com/VectorSpaceLab/OmniGen2
- [U1] https://github.com/bytedance/UNO
- [D1] https://github.com/bytedance/DreamO
- [I1] https://github.com/Tencent-Hunyuan/InstantCharacter and https://raw.githubusercontent.com/Tencent-Hunyuan/InstantCharacter/main/License.txt
- [IP1] https://huggingface.co/h94/IP-Adapter and https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0
- [K1] https://github.com/ostris/ai-toolkit/blob/main/config/examples/train_lora_qwen_image_24gb.yaml
- [K2] https://github.com/ostris/ai-toolkit/blob/main/config/examples/train_lora_qwen_image_edit_2509_32gb.yaml
- [K3] https://github.com/kohya-ss/sd-scripts

Practice and tools
- [S11a] https://saint11.art/img/pixel-tutorials/Walk.gif (index: https://saint11.art/blog/pixel-art-tutorials/)
- [S11b] https://saint11.art/img/pixel-tutorials/RunCycleSimple.gif
- [S11c] https://saint11.art/img/pixel-tutorials/AttackSheet.gif
- [S11d] https://saint11.art/img/pixel-tutorials/characterIdle.gif
- [S11e] https://saint11.art/img/pixel-tutorials/Jump.gif
- [LPC1] https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator (sources/state/constants.ts)
- [AS1] https://www.aseprite.org/docs/frame-duration/
- [PL1] https://www.pixellab.ai/docs/tools/animate-with-skeleton
- [PL2] https://www.pixellab.ai/docs/tools/animate-with-text-new and https://www.pixellab.ai/docs/tools/interpolation
- [PL3] https://www.pixellab.ai/docs/tools/animate-with-text-pro; ToS https://www.pixellab.ai/termsofservice
- [RD1] https://github.com/Retro-Diffusion/api-examples and https://replicate.com/retro-diffusion/rd-animation
- [RD2] https://retrodiffusion.ai/styles/four-angle-walking/
- [LU1] https://ludo.ai/docs/sprite-generator
- [SC1] https://help.scenario.com/en/articles/create-spritesheets-with-scenario/ and https://www.scenario.com/apps/action-sprite-studio
- [GM1] https://www.godmodeai.co/ and https://www.godmodeai.co/get-api
- [SSD1] https://arxiv.org/html/2412.03685v1 and https://github.com/chenganhsieh/Sprite-Sheet-Diffusion
- [SSD2] https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 (OpenRAIL-M); https://github.com/MooreThreads/Moore-AnimateAnyone

Local
- `README.md`, `CONTEXT.md`, `DEVLOG.md`, `anim/config.py`, `anim/stages/{pick,post,rig,clip,render,matrix,bakeoff,concept,pose,export}.py`, `anim/curation.py`, `anim/outcome.py`, `gpu/models.lock`, `gpu/models.registry.tsv`, `gpu/custom_nodes.txt`, `gpu/extra_model_paths.yaml`, `gpu/README.md`, `docs/bake-off-2026-09-20.md`, `docs/models.md`, `docs/workflow-4-cosmetic-inpaint.md`, `docs/research/2026-09-20-pixel-art-loras.md`, `docs/research/2026-09-24-runpod-comfyui.md`; ComfyUI checkout `/mnt/private/AI Data/comfyui-run/ComfyUI/comfy_extras/{nodes_wan.py,nodes_frame_interpolation.py}`.
