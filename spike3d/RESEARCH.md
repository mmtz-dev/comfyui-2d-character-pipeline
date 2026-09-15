# the 3D-intermediate spike feature — research

Primary-source research for **Feature 9** of the pose-to-pixel animation pipeline plan.
Everything below was checked against the licence file, model card or source code itself on
**2026-09-13**, not against a blog post or another doc. Where a popular secondary source
contradicts the primary source it is called out.

**Verdict up front.** The spike is worth running, and the path is licence-clean, but three
things in the original plan are wrong and are corrected here:

1. TripoSG is the right model, but its own inference script downloads **BRIA RMBG-1.4**
   (non-commercial) at runtime. Feeding it an **RGBA** image with binary alpha skips that model
   entirely — verified in the source. See §1.4.
2. `mixamotoopenpose` does **not** emit JSON and does **not** render rtmlib-compatible
   skeletons. It renders the *canonical* OpenPose algorithm, which differs from rtmlib's in
   alpha, two limb colours and channel order. Its joint list must be re-rendered through
   rtmlib. That is what `render_openpose.py` in this directory does. See §3.
3. The "drive the rig from your filmed video later" follow-up is **licence-blocked** for
   3 of the 4 candidate mocap models, and the 4th needs the non-commercial SMPL body model.
   A commercially clean route exists but it is not any of them. See §2.4.

---

## 1. The image-to-3D model

### 1.1 Recommendation

**TripoSG** (VAST-AI-Research), 1.5B rectified-flow, run as a standalone container on the
native daemon.

| | |
|---|---|
| Code licence | **MIT** — <https://github.com/VAST-AI-Research/TripoSG/blob/main/LICENSE> (verified: file begins `# MIT License / # Copyright (c) 2025 VAST-AI-Research and contributors.`) |
| Weights licence | **MIT**, ungated — <https://huggingface.co/VAST-AI/TripoSG> (`cardData.license == "mit"`, `gated == false`, via `https://huggingface.co/api/models/VAST-AI/TripoSG`) |
| VRAM | **≥ 8 GB** — README §"System Requirements": *"CUDA-enabled GPU with at least 8GB VRAM"* |
| Weight size | **7.95 GB** total: transformer 5.76 GB + VAE 0.97 GB + DINOv2 image encoder 1.22 GB (`Content-Length` on each `resolve/main` URL) |
| dtype | `torch.float16`, `device="cuda"` hard-coded in `scripts/inference_triposg.py` |
| Output | GLB mesh, `--faces N` to decimate |
| Repo activity | last push **2025-04-18** — stable, not abandoned, but not moving |
| Stars | 1789 |

**Why this one, on this card.** It is the only candidate where *both* the code and the weights
are a real OSI licence, it fits in a third of the A5000's 24 GB (leaving room to keep ComfyUI
loaded), and 8 GB of headroom is what makes a 2-session time box realistic. Everything above it
in raw mesh quality costs either the licence or the VRAM.

### 1.2 Alternatives, with the reason each loses

| Model | Code licence | Weights licence | VRAM | Verdict |
|---|---|---|---|---|
| **TripoSG** | MIT | MIT | 8 GB | **chosen** |
| Hunyuan3D-2 | Tencent Hunyuan Community (`NOASSERTION`) | `tencent-hunyuan-community` | 6 GB shape / 16 GB shape+texture | **excluded — regional carve-out**, see §1.3 |
| Hunyuan3D-2.1 | same | same | 10 GB shape / 29 GB shape+texture | **excluded** — same licence, and 29 GB does not fit |
| TRELLIS | MIT | MIT (`microsoft/TRELLIS-image-large`) | **16 GB** | viable but heavier; MIT code pulls **NVIDIA-licensed** submodules (`nvdiffrast`, `kaolin`, modified FlexiCubes) — see §1.5 |
| **TRELLIS.2** (Dec 2025, 4B) | MIT | MIT | **24 GB minimum** | **excluded** — needs the whole card, verified on H100/A100 only, and `setup.sh` installs `nvdiffrast` |
| Hi3DGen / **Stable3DGen** | MIT | MIT (`Stable-X/trellis-normal-v0-1`) | not stated (TRELLIS-class, ~16 GB) | **best runner-up.** Its README says it *"specifically removed [TRELLIS's] dependencies on certain NVIDIA libraries (kaolin, nvdiffrast, flexicube) to ensure this adapted version can be used commercially"* — the cleanest licence story of any TRELLIS derivative. Use it if TripoSG's mesh is too soft for the auto-rigger. |
| Step1X-3D | Apache-2.0 | Apache-2.0 | **27–29 GB** (README inference table) | excluded — does not fit |
| Direct3D-S2 | MIT | MIT | 10 GB @ 512³, **~24 GB @ 1024³**; README says 512 quality is *"noticeably lower"* | excluded — the usable resolution needs the whole card |
| Sparc3D (`lizhihao6/Sparc3D`) | **none** | — | — | **excluded — no licence file at all.** Same reason the repo already refuses `Astropulse/sd-palettize`. |

URLs: <https://github.com/microsoft/TRELLIS> · <https://github.com/microsoft/TRELLIS.2> ·
<https://github.com/Stable-X/Stable3DGen> · <https://github.com/Tencent-Hunyuan/Hunyuan3D-2> ·
<https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1> · <https://github.com/stepfun-ai/Step1X-3D> ·
<https://github.com/DreamTechAI/Direct3D-S2> · <https://github.com/lizhihao6/Sparc3D>

### 1.3 The Hunyuan3D regional exclusion, quoted

From `LICENSE` in `Tencent-Hunyuan/Hunyuan3D-2`
(<https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE>), line 3:

> THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA
> AND IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.

Definition, §1(l):

> "Territory" shall mean the worldwide territory, excluding the territory of the European Union,
> United Kingdom and South Korea.

And §5(c) — the clause that actually bites a shipped game:

> You must not use, reproduce, modify, distribute, or display the Tencent Hunyuan 3D 2.0 Works,
> **Output or results** of the Tencent Hunyuan 3D 2.0 Works outside the Territory. Any such use
> outside the Territory is unlicensed and unauthorized under this Agreement.

"Output or results" is the problem. game-02 ships on stores that sell into the EU and UK. A
sprite whose geometry came out of Hunyuan3D is an *output*, so selling it in the EU/UK is
unlicensed — not a theoretical concern, a distribution blocker. **Hunyuan3D is out.**

> Secondary sources get this wrong. Several 2026 round-up articles state "Hunyuan 3D-2.0 is
> fully open source under the Apache 2.0 license". The licence file in the repo says otherwise.
> Tencent *has* shipped some other models under Apache-2.0; the 3D ones are not among them.

### 1.4 The TripoSG licence trap, and how to avoid it

`NOTICE` in the TripoSG repo (<https://github.com/VAST-AI-Research/TripoSG/blob/main/NOTICE>)
declares three inbound dependencies:

- **RMBG** — *"License: bria-rmbg-1.4, Source: https://huggingface.co/briaai/RMBG-1.4"*
- code derived from **HunyuanDiT**, *"licensed under the Tencent Hunyuan Community License Agreement"*
- code derived from **FlashVDM**, *"licensed under the Tencent Hunyuan FlashVDM Community License Agreement"*

The BRIA one is the live problem. `briaai/RMBG-1.4` is `license: other`,
`license_name: bria-rmbg-1.4`, and the model card states:

> The model is released under a Creative Commons license for non-commercial use. … Commercial
> use is subject to a commercial agreement with BRIA.

And `scripts/inference_triposg.py` downloads it unconditionally, line 85:

```python
snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
```

**The fix, verified in `scripts/image_process.py`.** `load_image()` only calls `rmbg()` when the
input has no usable alpha channel:

```python
elif num_channels == 4:
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    b, g, r, alpha = cv2.split(img)
    if not is_valid_alpha(alpha):
        alpha = None            # -> the `if alpha is None:` branch, which calls rmbg()
    else:
        alpha_gpu = ...         # -> the `else:` branch, which does NOT
```

`is_valid_alpha` passes when ≥1 % of pixels sit in the lowest histogram bin **and** ≥1 % in the
highest — i.e. exactly the strictly-binary alpha this pipeline already guarantees
(`anim.config.ALPHA_VALUES == (0, 255)`).

So: **feed TripoSG an RGBA PNG with binary alpha, never load `rmbg_net`, and RMBG-1.4 is never
downloaded or executed.** `spike3d/run_triposg.py` enforces this — it refuses a non-RGBA input
rather than silently falling back. The Tencent-derived *code* notices stay in `NOTICE` and are
reproduced in this directory's `NOTICE-3d.md`; they are attribution obligations on source files,
not a territory restriction on output.

### 1.5 The NVIDIA-submodule trap in the TRELLIS family

TRELLIS's own README §License says the model and *"the majority of the code"* are MIT, then:

> The following submodules may have different licenses: **diffoctreerast** … **Modified
> Flexicubes** …

and TRELLIS.2 adds:

> certain dependencies operate under separate license terms: **nvdiffrast** … **nvdiffrec** …

`nvdiffrast` and `nvdiffrec` are under the NVIDIA Source Code License, which is research-use.
They are used for *rendering* generated assets, so a pipeline that only needs the mesh can in
principle avoid them — but `TRELLIS.2/setup.sh`'s documented invocation installs them, so
"avoiding them" is a patch you maintain. Stable3DGen already did that work and says so
explicitly. This is the whole reason TRELLIS.2 (MIT, better meshes) is not the recommendation.

### 1.6 ComfyUI integration — don't

There is no maintained first-party local TripoSG node:

- `VAST-AI-Research/ComfyUI-Tripo` (MIT, 351★, active) is a client for the **paid Tripo cloud
  API**, not local inference. Wrong tool.
- Community local nodes exist but are unmaintained toys: `sdwebui-extensions/ComfyUI-TripoSG`
  (MIT, 2★, last push 2025-04), `axwf/ComfyUI-TripoSG` (MIT, 0★), `acechenee/comfyui-triposg`
  (no licence), `ohdaehwan/ComfyUI-TripoSG` (no licence). The node named in blog posts,
  `fredconex/ComfyUI-TripoSG`, **404s** — the repo is gone.
- `MrForExample/ComfyUI-3D-Pack` (MIT, 3864★, last push 2025-12) is the real suite, but it is a
  large build with its own CUDA extensions and it would have to go into the *shared* ComfyUI
  basedir, i.e. it changes the 2D chain's container. Out of scope for a time-boxed spike.

**Decision: run TripoSG in its own container** (`spike3d/compose.yml`, `spike3d/Dockerfile`) and
leave the ComfyUI service exactly as the 2D chain needs it. The two never run at the same time —
see the runbook's VRAM note.

---

## 2. Rig and pose

### 2.1 `Astropulse/mixamotoopenpose` — verified

<https://github.com/Astropulse/mixamotoopenpose> · **MIT** (`LICENSE` line 1: `MIT License`,
`Copyright (c) 2024 Astropulse`) · 114★ · last push 2024-11-10 · not archived.

| | |
|---|---|
| Input | **`.dae` (Collada) only.** README: *"Download an animation from Mixamo as a Collada .dae file … DO NOT reduce keyframes!"* `.fbx` is **not** supported — it parses the XML directly with `xml.etree.ElementTree` and looks for `<node id="mixamorig_Hips">`. |
| Python deps | **Pillow, NumPy, OpenCV** — nothing else. README: `pip install pillow numpy opencv-python`. **No Blender, no torch, no GPU.** |
| Emits | `--output_format` ∈ `GIF` (animated), `PNG` (folder of `0.png`, `1.png`, … zero-padded to the frame count), `SHEET` (one square-ish grid sheet) |
| Resolution | `-ow/--width` × `-oh/--height`, **default 512×512** (note the function default is `(320, 512)`; the CLI default is 512×512) |
| Useful flags | `-ry/--rotation_y` (rotate the pose to any viewing angle — **this is how you get a profile view**), `-os/--scale` (default 2.0), `-f/--max_frames` (even-distribution frame reduction), `-ifps` (default 30) |
| Thickness | `stickwidth = 4`, joint circles `radius=4`, limb overlay `cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)` |
| Joint set | `openpose_keypoints` = COCO-18 in exactly OpenPose order: `nose, neck, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle, right_eye, left_eye, right_ear, left_ear` |
| Bone mapping | `mixamorig_HeadTop_End → nose` (so the "nose" point is the **top of the head**, offset by `nose_offset=(0, 0.2, 0)`), `mixamorig_Neck → neck`, `mixamorig_{Right,Left}{Arm,ForeArm,Hand} → shoulder/elbow/wrist`, `mixamorig_{Right,Left}{UpLeg,Leg,Foot} → hip/knee/ankle`. Eyes and ears are **synthesised** from the head bone's rotation matrix with `eye_offset=(0.2, 0, -0.2)` / `ear_offset=(0.4, 0.02, -0.45)`, because Mixamo rigs usually have no eye/ear bones. |

**Two things the plan assumed that are not true:**

- **It never writes JSON.** `format_to_openpose()` returns an in-memory list of
  `{"candidate": [[x, y, c], …18], "subset": [[0..17 or -1]]}` — the `controlnet_aux`
  candidate/subset shape, *not* the OpenPose `people[].pose_keypoints_2d` shape the fork's own
  `tools/render_openpose.py` reads. Nothing writes it to disk. To get a joint list out you
  `import mixamo_to_openpose` and call `convert_dae_to_openpose(dae, image_size, rotation_angles,
  scale_factor)` yourself. `spike3d/render_openpose.py` accepts that shape directly.
- **Its PNGs are not rtmlib's PNGs.** See §3 — this is the single most important finding in this
  document.

### 2.2 Mixamo is an owner step, and its terms have a sharp edge

Mixamo (<https://www.mixamo.com/>) needs a **free Adobe account**; an agent cannot log in, so
downloading clips and running the auto-rigger are owner steps. As of 2026-09 the site, the
auto-rigger and the free tier are all still up.

The auto-rigger accepts **FBX, OBJ, or ZIP** (OBJ + MTL + textures), humanoid only, mesh in a
T-pose or near-T/A-pose with a distinguishable head, body, arms and legs; it then asks you to
place chin/wrist/elbow/knee/groin markers
(<https://helpx.adobe.com/creative-cloud/help/mixamo-rigging-animation.html>).
**TripoSG emits GLB**, so a GLB→OBJ/FBX conversion sits between the two — see the runbook.

Licence, per the Mixamo FAQ: characters and animations are free for *"unlimited commercial or
non commercial use"* with no royalty and no attribution requirement. Two restrictions matter
here:

- **No repackaging raw files.** You may not redistribute the raw `.dae`/`.fbx` as an asset pack
  or engine template. → **never commit the downloaded `.dae` to the repo.** `spike3d/.gitignore`
  enforces this.
- **"Mixamo content can't be used for training machine-learning models."** We are not training —
  the skeleton is a *conditioning input at inference time*, and what ships is a 64×64 pixel
  sprite, not the animation data. That is very likely outside the clause as written, but it is
  close enough to the line that the owner should make the call knowingly rather than discover it
  later. **If the answer is "don't risk it", §2.3 is the drop-in substitute** and the rest of the
  spike is unchanged.

### 2.3 The open, no-Adobe-login alternative

| Source | Licence (primary source) | Format | Sword/hit/death coverage |
|---|---|---|---|
| **CMU Graphics Lab Motion Capture Database** <https://mocap.cs.cmu.edu/> | site front page, verbatim: *"This data is free for use in research projects. You may include this data in commercially-sold products, but you may not resell this data directly, even in converted form."* Asks (not requires) an acknowledgement line. | ASF/AMC native; **BVH** via Bruce Hahn's conversion at cgspeed, linked from the "Resources" tab | broad everyday motion; combat is thin |
| **Quaternius Universal Animation Library** <https://quaternius.com/> | **CC0** | GLB/FBX | ~260 clips **including sword and bow combat** — the best free match for `strike` |
| `maxparata.itch.io/cc0-animations` | CC0 1.0 | — | 5 clips incl. 1 sword attack + 1 death |

**The catch.** `mixamotoopenpose` only reads Collada with `mixamorig_*` node ids. A BVH or a
Quaternius GLB therefore needs a retarget-and-export step onto a Mixamo-named rig — which *does*
need Blender headless, and is roughly a session on its own. So: **Mixamo for the spike** (it is
the fast path and it is what makes 2 sessions plausible), CC0 as the fallback if the owner
rejects the ML clause or Mixamo changes.

`VAST-AI-Research/UniRig` (**MIT** code, **MIT** weights at `VAST-AI/UniRig`, ≥ 8 GB VRAM) is the
open auto-rigger, and it is from the same lab as TripoSG — but it predicts an *arbitrary*
skeleton, not a `mixamorig_*` one, and it supplies no animation clips. It replaces Mixamo's
rigging step only, and then nothing downstream reads its output. Not for this spike.

### 2.4 The "pose from your filmed video" follow-up — scoped, and mostly licence-blocked

Not built now. This is the table for when it is.

| Model | Repo licence (verified) | Depends on SMPL/SMPL-X body model | VRAM | Usable in a commercial game? |
|---|---|---|---|---|
| **WHAM** <https://github.com/yohanshin/WHAM> | **MIT** | yes | not stated in README | **no** — the SMPL body model itself is a non-commercial research licence from MPI |
| **TRAM** <https://github.com/yufu-wang/tram> | **MIT** | yes — README: *"Register at SMPLify and SMPL … Note that thirdparty models have their own licenses."* | not stated | **no**, same reason |
| **GVHMR** <https://github.com/zju3dv/GVHMR> | **non-commercial**, verbatim: *"Permission to use, copy, modify and distribute this software and its documentation for educational, research and non-profit purposes only. Any modification based on this work must be open-source and prohibited for commercial use."* | yes | trained on 2×4090 | **no** |
| **SMPLer-X** <https://github.com/MotrixLab/SMPLer-X> | **S-Lab License 1.0** — *"Redistribution and use for non-commercial purpose…"* | yes | tested on RTX 3090 | **no** |

All four are out, and three are out on the repo licence alone before SMPL is even considered.

**The clean route instead: `rtmlib`'s own 3D estimator.** `rtmlib` is **Apache-2.0**
(<https://github.com/Tau-J/rtmlib>) and already ships `Wholebody3d` / `rtmpose3d` — it is
*already installed in the CPU pipeline container* (`rtmlib==0.0.16` in `requirements.lock`). It
returns 3D keypoints plus the 2D projection, no SMPL mesh anywhere:

```python
keypoints, scores, keypoints_simcc, keypoints_2d = wholebody3d(frame)
```

So the eventual video→rig path is: film → `rtmlib.Wholebody3d` → 3D joints → drive the rigged
TripoSG mesh by IK → re-render the 2D projection through the same `render_openpose.py`. Same
licence footing as the 2D chain, no new licence surface, no registration. That is the follow-up
worth scoping — not WHAM.

---

## 3. Skeleton compatibility — the finding that decides the spike

**Three renderers are in play and no two agree.**

| | mixamotoopenpose `draw_bodypose` | `controlnet_aux.util.draw_bodypose` (= the fork's `tools/render_openpose.py`) | **rtmlib `draw_skeleton(openpose_skeleton=True)`** (what the 2D chain feeds the ControlNet) |
|---|---|---|---|
| Joint order | COCO-18, OpenPose order | COCO-18, same order | COCO-18, same order (`openpose18.keypoint_info`) — **all three agree here** |
| Limb half-width | `stickwidth = 4` | `STICKWIDTH = 4` | `int(line_width)` where `draw_skeleton` passes `line_width * 2` → **4** at the default `line_width=2` |
| Limb alpha | **real 0.6** — `cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)` | **real 0.6** | **effectively 1.0.** `draw_polygons` does `cv2.addWeighted(img, 1-alpha, img, alpha, 0)` — it blends the image *with itself*, so the polygon lands fully opaque. Same no-op in `draw_circles`. |
| Joint radius | 4 | 4 | hard-coded **4**; the `radius` argument is ignored (the line that used it is commented out) |
| Limb colour 9 (neck→left_hip) | `[0,255,255]` | `[0,255,255]` | **`[0,255,225]`** ← off-by-30 in the green-blue |
| Limb colour 13 (nose→right_eye) | `[85,0,255]` | `[85,0,255]` | **`[255,0,170]`** |
| Joint colours | table value, then `BGR2RGB` at save | table value as RGB | **`kpt_info['color'][::-1]`** — reversed, while limb colours are not |
| Out-of-frame limbs | drawn | drawn | **silently skipped** if either endpoint has `x <= 0 or x >= w or y <= 0 or y >= h` |
| Confidence gate | `subset` index `-1` | `c <= 0` | `scores[j] < kpt_thr`, `kpt_thr=0.5` from `draw_skeleton` |

So dropping mixamotoopenpose's PNGs straight into the 2D chain's `strip` → `identity_pass`
stages would feed the ControlNet an image that differs from every skeleton it has seen from this
pipeline in **opacity, two limb hues and the joint-dot channel order**. Those are exactly the
kind of small distribution shifts that make a ControlNet quietly ignore its hint.

**Which makes this the rule for the spike:** the 3D path may use the rig only as a *source of
2D joint coordinates*. The image handed to `identity_pass.json` must be rendered by
`rtmlib.draw_skeleton(..., openpose_skeleton=True)` — the same function, in the same container,
at the same version (`rtmlib==0.0.16`) — so the 3D and 2D arms of the comparison differ in
*pose*, and in nothing else. That is what `spike3d/render_openpose.py` is for, and why it calls
rtmlib rather than reimplementing it.

### 3.1 The `#28` report, quoted and sized honestly

The plan cites "an unresolved report that hand-authored skeletons are ignored". The primary
source is Hugging Face discussion **#28** on
`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`
(<https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/discussions/28>),
titled *"Does not work with openpose? (3d pose editor, VNCCS)"*, opened by `jhj001105` on
**2026-04-03**, status **open**. Its entire body is one sentence:

> Everything else works fine, but the ones where I create poses or using openpose processor
> manually aren't being recognized at all.

Sized honestly: **one report, no maintainer reply, no reproduction, no attached images, still
open**. It is a reason to *verify early*, not a proven defect. Note what it does say, though —
the reporter's failing cases are a **3D pose editor** and a **manual** OpenPose processor, which
is precisely the shape of the 3D-intermediate path. The other reports on that repo
(#1, #12, #18, #23, #25 — several in Chinese, "controlnet does not work", "can't reproduce the
official examples") suggest the pose branch is generally fragile.

**Therefore the runbook's first GPU action is the cheap A/B**, before any 3D work: take one
skeleton that the 2D chain's `rtmlib` renderer produced and one rendered by
`spike3d/render_openpose.py` from a synthetic joint list, run both through `pose_pass.json` at
the same seed, and confirm the second conditions the model at all. If it does not, the 3D
intermediate cannot work through this ControlNet and the spike ends in 20 minutes instead of
2 sessions. That is the single highest-value thing in this document.

### 3.2 Channel order — a real ambiguity, settled explicitly

rtmlib writes limb colours into the array **unreversed** and joint colours **reversed**, so one
of the two is channel-swapped no matter how you save the array. `render_openpose.py` therefore
takes `--channel-order`:

- **`rgb` (default)** — save the array as RGB. Limb colours then match the canonical OpenPose
  table exactly (limbs are most of the ink, and this is the convention the colour tables are
  written in); joint dots come out reversed, which is rtmlib's own behaviour and therefore what
  the 2D chain's skeletons look like too.
- `bgr` — save via cv2 semantics; the reverse trade.

Whichever the 2D chain's `anim pose` ends up using, **both arms of the comparison must use the
same one.** The flag exists so that is a one-word change in one place, not a re-render.

---

## 4. Blender — needed, but only for format conversion

Blender itself is **GPL-2.0-or-later**. That governs Blender's *code*; meshes and images it
outputs are the user's own work, so using it as a converter puts no licence on the sprites.

| Image | Licence | Fit |
|---|---|---|
| `linuxserver/blender` <https://github.com/linuxserver/docker-blender> | **GPL-3.0-or-later** (the image's own scripts) | a KasmVNC **desktop** image — works headless via `blender --background` but pulls a whole web desktop for a 3-line conversion |
| `nytimes/blender`, `james-baber/rd-blender-docker` | varies, community | unmaintained / unpinned |
| *(no official Blender Foundation Docker image exists)* | — | — |

**Better: skip Blender for GLB→OBJ.** `trimesh` (**MIT**) loads GLB and writes OBJ in four lines,
and it is already a TripoSG dependency, so the conversion runs inside the spike's own container
with no extra image at all. Blender is only worth pulling in if the CC0/BVH route of §2.3 is
taken, where retargeting genuinely needs it. The runbook does it with `trimesh`.

---

## 5. The honest limitation

**This spike replaces "pose from your filmed video" with "pose from a rig."** The 2D chain's
whole premise is that the owner acts a motion out in profile and the pipeline follows *that*
performance. The 3D path does not do that. It takes a canned animation clip — Mixamo's or a
CC0 library's — and re-projects it. It will very likely hold `strike`, `hit` and `death` better,
because a 3D rig cannot produce an anatomically impossible limb the way a 2D ControlNet hint
can; but what it produces is *someone else's* motion, identical for every unit that uses the
same clip, and not adjustable by re-filming.

Closing that gap needs the video→3D-mocap step of §2.4, which is not in this spike's box and
whose obvious candidates are all licence-blocked. **If the spike wins at the anchor gate, the
`rtmlib.Wholebody3d` follow-up is the work that makes it match the 2D chain's original promise —
and it should be a named work package before the 3D route is adopted, not after.**

---

## 6. Sources I could not reach

| Source | Why | To unblock |
|---|---|---|
| <https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html> | **HTTP 403** to the fetcher (Adobe bot-blocks automated agents) | Owner opens it in a browser and pastes the Licensing section — needed to quote the ML-training clause of §2.2 from the primary source instead of from search-result summaries |
| <https://community.adobe.com/…/mixamo-faq-licensing…> | Fetcher was served an **unrelated Illustrator thread** (Adobe community SPA does not render for it) | same |
| <https://mocap.cs.cmu.edu/> | TLS: *"unable to verify the first certificate"* — expired/incomplete chain on CMU's host | **Recovered**: refetched with `curl -k`; the terms in §2.3 are quoted verbatim from the page body. Nothing outstanding. |
| <https://bria.ai/bria-huggingface-model-license-agreement/> | not fetched — the HF model card's own licence statement was sufficient and is the operative one | fetch it if a commercial BRIA licence is ever actually wanted |
| Step1X-3D / Direct3D-S2 / Stable3DGen exact VRAM at inference | READMEs give a training or a range figure, not a single inference number | only matters if TripoSG is rejected; measure on the card |

Marked **UNVERIFIED** anywhere it matters:

- **Stable3DGen's inference VRAM** is assumed TRELLIS-class (~16 GB) from its lineage; its README
  does not state a number. UNVERIFIED.
- **`Astropulse/mixamotoopenpose` against a 2026 Mixamo `.dae` export.** The script was last
  touched 2024-11-10 and parses Collada XML by hard-coded node id (`mixamorig_Hips`). If Adobe
  has changed its Collada export naming since, it breaks. UNVERIFIED — this is checkpoint 2 in
  the runbook, and it is checked before any GPU time is spent.
- **Mixamo's auto-rigger accepting a TripoSG mesh.** Single-image 3D from a 64 px side-view
  sprite produces a soft, possibly non-T-pose mesh; the auto-rigger wants clean humanoid
  geometry. UNVERIFIED, and the most likely place the spike fails. The runbook's mitigation is to
  feed it a front-facing A-pose reference generated by the 2D chain's own `base_sprite.json`
  rather than the 64 px side sprite.
