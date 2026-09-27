# ComfyUI 2D Character Animation Pipeline

---

# Game-Dev fork

> **Read this before the upstream README below.** This is a fork of
> [`mor-o/comfyui-2d-character-pipeline`](https://github.com/mor-o/comfyui-2d-character-pipeline)
> (MIT, credited in [`NOTICE`](NOTICE)), vendored into the Game-Dev monorepo at
> `tools/anim-pipeline/` and retargeted at a different job. Everything below the
> `Upstream README` heading is upstream's own text and describes upstream's five-workflow
> pipeline, **most of which is parked here**. Where the two disagree, this section wins.
>
> Fork: <https://github.com/mmtz-dev/comfyui-2d-character-pipeline> ·
> upstream tracked as the `upstream` git remote.

## Where this is going — the sprite studio feature (decided 2026-09-20)

This repo is becoming the engine of **the sprite studio**: a web pixel editor (a fork of
Piskel in a Vite + TypeScript shell, served by a FastAPI service on `:8720`) with this
pipeline as a panel inside it — text or sketch → base character → 8 side + 8 top-down
facings → idle / walk / run / jump / attack → 12 fps keyframe phases → curated,
palette-snapped sheets + Godot `SpriteFrames`. The rig → clip → post stages below stay the
engine and stay runnable headless; the API landed 2026-09-21 (`anim-api`, below) and the
`review` page on `:8710` is retired. Design: `games-design-wiki/shared/sprite-studio.md`. Work: the OpenProject project
[Sprite Studio](http://192.168.1.158:8700/projects/sprite-studio), printed into
[`TODO.md`](TODO.md) (generated — do not edit). Branch: `feat/sprite-studio`.

## What this fork is for

**The pose-to-pixel animation pipeline feature** for `game-02-tactical-rpg`: film a
motion in profile on a phone or Mac — or, since 2026-09-14, take it from a **CC0
rigged character** (the rig motion input feature, below) — click which key poses to
keep, and get palette-locked **64×64 pixel frames** that match the unit's own sprite
and play back in Godot on the combat choreography's timing.

Upstream produces smooth 512px greyscale sprite sheets with tintable cosmetic layers
for a runtime compositor, driven by a video model. This fork produces **2–6 hand-nudgeable
64px key-pose frames per action** in a fixed Lospec palette, driven by a **pose skeleton
extracted from real footage**. Same ComfyUI plumbing, different output contract.

| | Upstream | This fork |
|---|---|---|
| Motion source | Wan 2.2 image-to-video extrapolation | DWPose skeletons from a filmed clip (`rtmlib`, outside ComfyUI), **or** exact skeletons sampled from a CC0 rig clip (`anim rig`, KayKit Knight) |
| Frames per action | 33+, a smooth cycle | 2–6 auto-picked key poses, owner can override |
| Output | 512px greyscale RGBA sheet, runtime-tinted | 64×64 PNG per frame, palette-snapped, binary alpha, one sidecar per action |
| Layering | separate cosmetic sheets (hair/eyes/clothes) | none — one sprite per unit |
| Weights | ~120 GB (Wan 2.2, VACE, SAM3, GGUF) | 58 GiB, all Apache-2.0 (`gpu/models.lock`; 17 GiB of it is the clip render feature's Wan 2.2 Fun-Control 5B, 162 MB the character concept feature's `tarn59` pixel-art LoRA for Z-Image-Turbo) |
| Post-processing | inside ComfyUI (BiRefNet) | deterministic CPU chain outside ComfyUI (rembg → pixeloe → pixel-art-fixer → ImageMagick `-remap`) |

## The `anim` CLI — the whole flow

Everything CPU-side is one command. It runs on **Docker Desktop**; the only stages
that touch a GPU are `render`, `clip`, `base`, `concept`, `animate`, `bakeoff` and `matrix`, over HTTP.

```bash
cd tools/anim-pipeline

# 0. once: build the image
docker compose build anim

# 1. film in profile (art bible §9) and drop the clip in capture/
#    one action per clip, 3-5 s, hold the start pose, do it once, hold the end

# 2. the whole chain
docker compose run --rm anim run capture/knight_strike.mp4 \
    --unit knight --action strike
```

That is:

```
capture/knight_strike.mp4
   │
   ├─ extract ─► work/knight_strike/frames/0001.png …      ffmpeg, fps-capped at 12, rotation honoured
   ├─ pose    ─► work/…/pose/keypoints.json + skel_*.png   rtmlib DWPose on the CPU, cached per clip
   ├─ pick    ─► work/…/picks.json                         N key poses (art bible budget per action)
   ├─ strip   ─► work/…/strip/skeleton_strip.png           ONE 1216×320 strip, never square
   │             work/…/strip/reference_512.png            the unit's 64px sprite, nearest ×8
   │
   │   ┌──────────── GPU reachable? ────────────┐
   │   │  no  → stops here, says how to start it │   ← the normal case; nothing is lost
   │   └────────────────┬────────────────────────┘
   │                    yes
   ├─ render  ─► work/…/render/identity_pass.png           two /prompt jobs + provenance.json
   ├─ post    ─► work/…/post/knight_strike_0..3.png        palette-locked 64×64, binary alpha
   └─ publish ─► assets/2d/units/knight_strike_0..3.png    + ONE knight_strike.json sidecar
                                                           then tools/asset_audit.py must exit 0
      export  ─► work/_export/knight/{sheets/, atlas.json, knight.tres}   every (action × direction) of the UNIT
                                                           (the generic contract; publish is its game-02 adapter)
```

Any stage also runs on its own, in any order, any number of times — all state is
files under `work/<job>/`, so a re-run after a hand edit costs only that stage:

```bash
docker compose run --rm anim pose knight_strike
docker compose run --rm anim pick knight_strike -n 3
docker compose run --rm anim strip knight_strike
docker compose run --rm anim render knight_strike --seed 861234901
docker compose run --rm anim post knight_strike
docker compose run --rm anim publish knight_strike --dry-run
docker compose run --rm anim export knight              # sheets + atlas + SpriteFrames for the whole unit
```

| Stage | In | Out | Notes |
|---|---|---|---|
| `extract <clip>` | a filmed clip | `frames/%04d.png`, `job.json` | fps is a **cap**, not a target; a 10 fps clip stays at 10. Rotation is honoured by never passing `-noautorotate`. `--crop` (off by default) crops every frame to **one** person-centred box — use it when the figure sits small in the frame, because the skeleton is letterboxed into the cell and headroom is wasted resolution. |
| `pose <job>` | frames | `pose/keypoints.json`, `pose/skel_%04d.png` | rtmlib DWPose over onnxruntime, CPU. Inference happens **once per clip** and is cached; `pick`, `review` and `strip` re-read it. Rendered with rtmlib's own `draw_skeleton(openpose_skeleton=True)` at the cell size, line width scaled to the canvas. |
| `rig <glb> --clip <name>` | a rig clip | `job.json`, `frames/%04d.png`, `pose/keypoints.json`, `pose/skel_%04d.png`, `depth/%04d.png` | **replaces `extract` + `pose`** when the motion is a rig clip. Samples the clip at the extract fps, maps the bones to OpenPose-18, projects in profile **facing right by detection**, one scale per rig; the skeleton PNGs go through the very same `posekit.draw`, with the unit's props (sword, shield) drawn on as white outlines. Also writes a **depth frame** per sample — the skinned mesh plus the props as solids, z-buffered under the same projection (the rig props feature); `--depth props` draws the prop solids only, body-occluded, for a model that paints the mannequin when given the body; `--depth none` / `--no-depth` skips it. `--yaw 0|45|…|315` and `--elevation side|top` view the rig from any of the 16 direction cells (the camera direction feature) — the job is then `<unit>_<action>_<dir>`. `--list` prints the clips, `--props` overrides the kit. See "Rig input mode". |
| `pick <job>` | keypoints | `picks.json` | peaks of deviation-from-rest + reversals + both endpoints, de-duplicated in torso lengths. Also records the **timing shape** (`timing`: one weight per frame + the anchor frame — the strip timing feature): the per-verb template in `config.ACTION_TIMING` by default, or the clip's own frame gaps with `--timing clip`. **`--phases`** (the keyframe phase feature; the default for `attack`, `run`, `jump`) picks against the studio's template instead — start → wind-up → action → wind-down → end → loop, the per-action budget from `config.ACTION_PHASES` (cap 12), ease as denser sampling around the action, one anticipation frame for a one-shot, a loop-closure check for a cycle — and tags every pick with its phase; `--no-phases` is the 2–6 key-pose picker for any action. See "Keyframe phases" below. **Refuses to overwrite a hand-edited `picks.json`** (picks *or* timing) unless `--force`. |
| `review <job>` | frames + picks | rewrites `picks.json` | prints where the click-to-pick page is — it lives on the API now (`/jobs/<job>/review`), see "The review page" below. `ReviewApp` in the module is what the API drives. |
| `strip <job>` | picks + skeletons | `strip/skeleton_strip.png`, `reference_512.png`, `reference_strip.png`, `layout.json` | asserts the strip is non-square and both dimensions are multiples of 16. |
| `render <job>` | strip | `render/pose_pass.png`, `identity_pass.png`, `provenance.json` | **a GPU stage.** Two `/prompt` jobs, not one graph. Seed is derived from the clip hash + action + picks unless `--seed` is given, so a re-run reproduces. |
| `clip <job>` | every skeleton (+ `depth/` if `rig` wrote it) | `clip/frames/%04d.png`, `clip/control.mp4`, `clip/clip.json`, `clip/provenance.json` | **a GPU stage — the clip render feature.** Renders the whole rig clip as one Wan 2.2 Fun-Control 5B video with the unit's `base` render as the identity reference and the skeleton PNGs as the control video, one generated frame per rig frame (padded to 4n+1 by holding the last; `clip.json.frame_map`). `--control auto|pose|depth|blend` (auto = blend when `depth/` exists) and `--depth-gain` decide what the control video carries. Rig the job at the clip cell first: `rig … --width 512 --height 768` (768×768 for a lying death). Measured 2026-09-17: 9–55 s of GPU per action. Then `pick` as usual and `post --input clip`. See "The clip route" below. |
| `post <job>` | generated strip, or `--input clip` | `post/<slug>_<action>_<n>.png`, `report.json` | deterministic and entirely outside ComfyUI: matte → slice → grid repair → pixeloe → palette snap → recanvas → verify. A frame that fails verification is not written. **The snap is the only quantisation in the pipeline** (the palette feature): `-remap` to the project palette, or to the unit's sub-palette when `palette_units/<slug>.json` exists; `--no-snap` is a raw-colour preview (matted, downscaled, recanvassed, not remapped) that `publish` refuses. `--matte rembg` (default) is the real cut; `--matte chroma` is a deterministic flat-background key — use it on drawn or fixture strips, where `isnet-anime` has no character to find and returns an empty mask. `--input clip` takes the picked frames from `clip/frames/` one by one instead of slicing a strip; `report.json.route` records which. **`--parts`** (the part palette lock feature, off by default): SAM 2.1 masks each HD cell into the unit's parts (`config.UNIT_PARTS`, prompted from the rig's joints and prop outlines), and after the palette snap every pixel of a part is re-snapped to *that part's* colours on the base sprite. On a base job it EXTRACTS those palettes to `post/parts.json`; on an action it LOCKS to them (`--parts-from FILE` overrides the lookup: published `<slug>.json` sidecar, then `work/<slug>_base/post/parts.json`). `report.json` gets per-frame, per-part pixels / colours / `found` / `drift`, and `post/parts/` the HD overlays and 64 px label maps. See "The part palette lock" below. |
| `publish <job>` | finished frames | `assets/2d/units/…` + one sidecar | **the game-02 adapter** of the spritesheet export feature — what game-02 reads today, unchanged: copies **in place** (never rename), clears stale numbered frames, writes `contract.timing` + `generation.timing_source` (hand edit > what `pick` recorded > template > uniform, which writes no block; a phased pick adds `fps`, `loop`, `phases` and `durations_ms` to the block), then runs game-02's asset audit and fails loudly on a non-zero exit. `anim export --adapter game02` runs it per job. |
| `export <unit>` | every `post/` of `<unit>_<action>[_<dir>]` | `work/_export/<unit>/sheets/*.png`, `atlas.json`, `<unit>.tres`, `preview.html` | **the spritesheet export feature** — the generic contract. One sheet per direction with one row per action (`--sheet packed`: one row-major sheet), a schema-versioned atlas (rects, phase tag and `duration_ms` per frame, loop, pivot, source job + commit), and a Godot 4 `SpriteFrames` over the sheets with per-frame durations — checked by reading it back before the stage returns. The cell is `frame_size` from `work/project.json` (`--size N` / `WxH` overrides). Missing cells are listed warnings, not errors. `--adapter game02` (or `export.adapter` in the settings) also runs `publish` per job. See "Export" below. |
| `base --unit <slug>` | a canned standing skeleton, or `--rig <glb> --clip Idle --frame 0` | `assets/2d/units/<slug>.png` + sidecar | for a unit with no sprite yet: the pose stack alone, one 512×768 portrait cell, `contract.frames [1, 1]`. With `--rig` the standing pose is one sampled frame of a rig clip at the same facing and scale as that rig's action strips, and `--yaw` / `--elevation` view it from any direction cell (job `<slug>_base_<dir>`; the default `<slug>_base` is unchanged); `--skeleton-only` stops before the GPU, and `--repose` re-lays the canned skeleton on a base job that already exists (the front base pose feature — no GPU, and unlike a re-pick it leaves the render and the accepted views alone). `--parts` extracts the unit's part palettes from the finished sprite and publishes them in its sidecar (`generation.parts`), which is where every action of that unit then reads them from. |
| `concept --unit <slug> --prompt "…"` | a description, optionally `--sketch <png>`, `--existing [png]`, `--upload <png>`, `--model` / `--lora FILE[:STRENGTH]`; or `--img2img REF --strength` / `--edit REF --instruction` on one attempt | `work/<slug>_concept/candidates/NN.png` + `job.json` (every attempt kept); `--pick NN\|existing` → `work/<slug>_base/render/concept_pick.png` | **a GPU stage — the character concept feature.** Text (or a sketch) to `--count` candidate front-facing base characters: Z-Image-Turbo int8 + the `tarn59` pixel-art LoRA (trigger and strength in `config.CONCEPT_LORA`), one `/prompt` and one seed per candidate, 768×768 on white; a sketch goes through the Fun ControlNet Union as a scribble, and without one no ControlNet node is in the graph. `--existing` registers the unit's published sprite (or a given file) as candidate 00 so the owner compares against it. `--pick` makes a candidate the unit's base render — the slot `clip`, `post --parts` and `publish` read — without touching `anim base`. Measured 2026-09-20: ~9 s per image after the first. See "Character concept" below. |
| `bakeoff --unit <slug> --action attack` | the two rigs' attack clips | `work/bakeoff_<slug>_attack/{bakeoff.json,page/,report.md}` + one job per cell `work/bakeoff_<slug>_attack_<cell>/` | **a GPU stage — the frame route bake-off feature.** One action through both candidate routes for the in-between frames on both candidate rigs — four cells `V-kaykit`, `V-quaternius`, `K-kaykit`, `K-quaternius` — as ordinary `rig → clip → pick → post --parts` (V) and `rig → pick → strip → render → post --parts` (K) jobs called as a library, one seed, then one page with an animated 12 fps preview per row and one metric table. Resumable per cell (`--cells`, `--force`, `--page-only`); seconds per stage and VRAM per cell in `bakeoff.json`. A failed cell is recorded and the others still run — and `bakeoff.json`'s `summary` plus the stage's **exit code** say so (see "The outcome of a batch stage" below). The owner rules on the page; the stage picks nothing. See "The frame route bake-off" below. |
| `matrix <unit>` | the accepted base, its accepted views, its part palettes (`work/<unit>_base/post/parts.json`) and the rig's clip per action | `work/<unit>_<verb>[_<dir>]/` — an ordinary `rig → clip → pick --phases → post` job per cell — plus `work/<unit>_matrix.json`, the plan and every verdict | **a GPU stage — the action matrix feature (#278).** Every selected (action × direction) cell of `config.STUDIO_ACTIONS`, batched, route V as the bake-off ruled it (`config.MATRIX_*`) — skeleton over full-body depth at gain 0.35, every part locked to the **base's** part palettes in every direction, and `pick --phases` picks the five phases out of the whole clip rather than generating them one at a time. Job names use the pipeline verb, not the studio name (`attack` at `s45` is `knight_strike_s45`), so `export` finds them. **The hard prerequisite gate**: a direction with no accepted view on the base is blocked, not front-view-falled-back — there is no `--allow-fallback` — and so is every cell while the base has no measured part palettes (`anim post <unit>_base --parts`). Resumable per cell (checkpointed after every one, `--force` redoes a done one), one seed for the whole unit. `--actions`, `--dirs`, `--all-dirs`, `--dry-run` (the plan and every verdict, no GPU). Capped at `config.MATRIX_MAX_CELLS` (8) per run until the motion exaggeration feature (#277) lands; `--max-cells 0` accepts the re-render. `summary` counts `done` / `failed` / `blocked` / `of`, and the **exit code** is the tally, not the process result (see "The outcome of a batch stage" below). |
| `palette --lospec <slug>` \| `--file <path>` | a Lospec slug (`https://lospec.com/palette-list/<slug>.json`) or a `.hex` / `.gpl` / one-row `.png` | `palette.gpl` + `.hex` + `.png` + `.json` beside `ANIM_PALETTE` | **the palette feature.** Installs the project palette — the `-remap` source of truth `post` reads. `--unit <slug> --colours '#…,#…'` or `--from-sprite <png>` writes the unit's sub-palette to `palette_units/<slug>.json`; `--list` prints what is installed; `--into <path>` targets another `.gpl`. Same input, same bytes. See "The project palette" below. |
| `curate <unit>` | `work/<unit>_curation.json` + the unit's jobs | the same file | **the curation board feature**, from the shell. `--accept-cell attack/s0`, `--reject-cell`, `--pending-cell` (reopen a locked cell), `--accept-frame attack/s0/3`, `--reject-frame` (also drops the frame from that job's `picks.json`, byte for byte as the review page does), `--clear-frame`, `--accept-gate base`, `--pending-gate`, `--regenerate attack/s0 [--seed N]` (records the seed and prints the `anim clip … --seed N` line; never the GPU), `--mode gates\|every\|end` (the project setting). With no flags, prints the board and what is waiting on you; with no unit, the units. See "The curation board" below. |
| `run <clip>` | a filmed clip | all of the above | stops politely before `render` when the GPU is off. |

### The clip route — the clip render feature

```bash
anim rig rigs/quaternius/UAL1_Standard.glb --clip Sword_Attack --unit knight \
    --action strike --job knightq_strike_clip --width 512 --height 768
anim clip knightq_strike_clip            # ~20 s on the A5000 once the model is warm
anim pick knightq_strike_clip
anim post knightq_strike_clip --input clip
anim publish knightq_strike_clip         # sidecar generation.route = "clip"
```

Why: the strip route generates each action's poses with an image model and keeps
identity across actions by prompt + seed alone, which is where the anchor gate's
shield-colour and armour-tone drift came from. A reference-conditioned video model
denoises every frame of one clip against one reference latent, so a clip is one
character throughout; `pick` then chooses the 2–6 key poses from frames that already
agree. The strip pose pass stays as the seconds-long preview. Cell: the base render's
512×768 so reference and control share a scale; a lying death needs 768×768 (the rig
scales once per rig and a lying body is wider than 512). Wan has no square-aspect
rule — that one is Qwen-Image-Edit's. Models: `wan2.2_fun_control_5B_bf16`,
`umt5_xxl_fp8_e4m3fn_scaled`, `wan2.2_vae` in `gpu/models.lock` (all Apache-2.0).
Evidence and the depth-control trials: `http://192.168.1.158:8090/clip-render/`.

### Keyframe phases — the keyframe phase feature

Every studio clip is picked against one shape, whatever route the frames came from — a Wan
clip, a pose-pass strip, a filmed take ([#276](http://192.168.1.158:8700/work_packages/276)):

```
start → wind-up → action → wind-down → end → (loop)
```

```bash
anim pick knight_attack                 # attack: the template is the default
anim pick knight_strike --phases        # an art-bible verb, on request (the matrix does this)
anim pick knight_attack --no-phases     # the 2–6 key-pose picker instead
```

- **Budgets** (`config.ACTION_PHASES`, cap 12): idle 6 · walk 8 · run 8 · jump 10 · attack 12.
  Idle, walk and run loop; jump and attack are one-shots. `-n` overrides inside the cap.
- **The split and the span.** Each row divides the *budget* between the five phases (`split`)
  and, separately, the *source clip's time* (`span`). The action phase gets a third of the
  frames for a quarter of the clip; the rest phases get one or two frames for a fifth of it
  each. Frames per second of clip is the sample density, and dense-at-the-action /
  sparse-at-rest is what ease in / ease out means for a picker. Attack comes out
  `start 1 · wind-up 3 · action 4 · wind-down 2 · end 2`.
- **Anticipation.** A one-shot spends one frame of its budget on the wind-up beyond its
  split, so the beat before the action is never a single cell (`phases.anticipation` is that
  pick's index). Loops get none.
- **Loops.** A cycle's last pick stops short of the frame that is its first again (no doubled
  cell on the wrap), and the **loop-closure check** measures the last pick's distance back to
  the first — mean joint displacement in torso lengths from the keypoints, against
  `ANIM_LOOP_TOL` (0.12, the picker's own "same pose" threshold); the mean pixel difference of
  the two frames when a pose is missing, against `ANIM_LOOP_PIXEL_TOL` (0.08). The verdict is
  in `picks.json` (`phases.closure`) and on the review page; a cycle that does not close is
  reported, not fixed.
- **What lands in `picks.json`.** A `phases` block — the plan, one tag per pick, the
  anticipation index, the closure — and `timing` becomes the phase shape: one weight per
  frame in 12 fps ticks, the anchor on the first action frame, `phases` and `durations_ms`
  beside them. `publish` copies all of it into the sidecar's `contract.timing`.
- **The hold rule.** A clip lasts budget / 12 s. When a phase's window held fewer source frames
  than the phase was allotted (a 9-frame clip against a budget of 12 gives all nine), its
  frames hold longer — each weighs allotted / picked ticks — so the phase still lasts its
  share and the clip its budget. With every phase full, every weight is 1 (83 ms).
- **Per-project override.** `ANIM_TIMING` (default `/assets_src/timing.json`, beside the
  palette in the customer game's `assets_src/`): action rows merge over the defaults —
  `{"attack": {"budget": 10}}` changes the budget and keeps the split — and a
  `"units": {"hound": {"run": {…}}}` block overrides per unit. Keys may be the studio name or the
  pipeline verb (`attack` ↔ `strike`; `config.STUDIO_ACTIONS` also names each rig's clip).
- **Without `--phases`** an existing job's `picks.json` is byte-for-byte what it was (a golden
  file in `tests/fixtures/` pins it). The bake-off's two routes now go through the same picker
  with `spread="even"` — the phase labels and timing, the frames the owner judged.

### The part palette lock

The generation cannot promise that a shield stays the same grey from `strike` to `death`
(the anchor gate, 2026-09-15). `post --parts` is the deterministic safety net:

```bash
# once, on the accepted base sprite: extract the palettes and publish them
docker compose run --rm anim post knight_base --parts
docker compose run --rm anim publish knight_base          # -> knight.json generation.parts

# every action: lock to them
docker compose run --rm anim post knight_strike --parts
```

- **Masks** come from SAM 2.1 hiera-small on the CPU (Apache-2.0, pinned by sha256 in
  `config.SAM2_WEIGHTS`, fetched once into `.cache/sam2/`), prompted with **points from the
  rig** — the joints in `pose/keypoints.json` plus the sword and shield outlines the skeleton
  props feature draws — never with text (SAM 3 is gated and not Apache-2.0). The part table
  per unit is `config.UNIT_PARTS`; order is priority for overlaps.
- **The prompts must match the picture** (the front base pose feature). A base the `concept`
  stage picked has no rig: `concept --pick` lays down a canned skeleton purely so this stage
  has joints to prompt from, and since txt2img was asked for a *front* view it lays down
  `anim/data/stand_pose_front.json`, not the side profile `anim base` renders through the
  ControlNet. Prompting a front-facing figure from the side profile stacks every joint into a
  ~66 px column, and the parts then share each other's points and fail the contamination gate
  — measured on `hooded_archer` 2026-09-24, where `arms` / `torso` / `legs` all came back with
  0 px and every direction then failed on the missing palettes. A unit picked
  before this landed fixes its prompts with `anim base --unit <slug> --repose`, which rewrites
  only the pose (a re-pick would clear the base's accepted views).
- **The part table describes the art, not the skeleton.** `knight` is helmet / arms / torso /
  legs because plate armour separates that way; `hooded_archer` is hood / gloves / boots /
  cloak because one full-length cloak does not. Naming a region the art does not show asks SAM
  for a mask that shares every pixel with its neighbours, and the gate correctly refuses it.
- A part SAM does not find in a frame (a shield behind the body) is reported, not invented,
  and a part under `PART_MIN_PALETTE_PIXELS` on the base sprite yields no palette, so an
  action never locks a whole shield to three pixels.
- **Off by default.** `config.PART_LOCK_DEFAULT` turns it on per unit; `--no-parts` forces it
  off. With it off `post` produces the same bytes it did before the feature existed.
- Read `anim/partkit.py` for the measured reasons behind each choice (why negatives are used
  to choose SAM's candidate rather than fed to it, why no box, why Oklab).

### The project palette — the palette feature

One project palette, from Lospec, per-unit sub-palettes, **snapped at export**: `concept`,
`base`, `render` and `clip` generate in free colour, and `post` is the one place a pixel
is remapped (sprite-studio.md §2). [#282](http://192.168.1.158:8700/work_packages/282).

```bash
# the project palette - by Lospec slug (the JSON endpoint), or from a file
docker compose run --rm anim palette --lospec lospec-2000
docker compose run --rm anim palette --file /assets_src/palette_candidates/grim32/grim32.gpl
docker compose run --rm anim palette --list

# a unit's sub-palette: what the knight may use, out of the project's colours
docker compose run --rm anim palette --unit knight --from-sprite /assets/2d/units/knight.png
docker compose run --rm anim palette --unit knight --colours '#141415,#444443,#c9c2a3'
docker compose run --rm anim palette --unit knight --clear

# a raw-colour preview of a job, no snap (never publishable)
docker compose run --rm anim post knight_strike --no-snap
```

- **The files.** `anim palette` writes the same set game-02's `install_palette.py` does —
  `palette.gpl` (the human-readable source), `palette.hex` (Lospec's format, one `rrggbb`
  per line), `palette.png` (the one-row image `magick -dither None -remap` consumes) — plus
  `palette.json` (name, author, source slug / file, the colours as hex, `fetched_at`), all
  beside whatever `ANIM_PALETTE` names (`/assets_src/palette.gpl`). Written in place, and
  the `.gpl` carries no date, so re-running with the same input changes no bytes. Checked
  2026-09-21: `--lospec lospec-2000` reproduces the installed `palette.hex` and `palette.png`
  byte for byte; the `.gpl` differs only in its header comments.
- **Lospec.** `GET https://lospec.com/palette-list/<slug>.json` →
  `{"name", "author", "colors": ["rrggbb", …]}`; a wrong slug is a 404 and the stage says
  so, no network is a clear message with the `--file` fallback. `ANIM_LOSPEC_URL` overrides
  the template (the tests point it at a `file://` directory).
- **Sub-palettes** live one file per unit in `palette_units/<slug>.json` next to the
  project palette — with the other art decisions (the palette, `timing.json`), not in
  `work/`, because "the knight is these 56 of the 182" is a decision, not scratch. Every
  colour must be a member of the project palette: a typed list with an outsider is refused,
  a sprite's outsider snaps to its nearest member and the record says which. A unit with
  no file gets the whole project palette — exactly the path every job took before the
  feature, byte for byte.
- **What `post` does with it.** The `-remap` palette becomes the unit's subset, the
  verification checks against the same subset, and the part palette lock (which runs
  *after* the remap, order unchanged) is narrowed to it — a base sprite posted before the
  sub-palette existed may carry outsiders, and each moves to its nearest member so the
  lock cannot reintroduce them. `report.json.palette` records `colours` (what was snapped
  to), `project_colours`, `unit_palette` and `snap`. A sub-palette that names a colour the
  project palette no longer has stops `post` with the command that fixes it.
- **The API.** `GET /settings` shows the palette expanded (`{path, name, colours, source}`),
  `GET /palette` the full record, `POST /palette {"lospec": …}` / `{"file": …}` /
  `{"unit": …, "colours" | "from_sprite" | "clear"}` runs the stage through the run manager
  (a CPU run, `202` + the run), `GET /palette/units/{unit}` one unit's file.

### Export — the spritesheet export feature

The generic export contract ([#283](http://192.168.1.158:8700/work_packages/283); the
decision in `sprite-studio.md` §2: **generic + a game-02 adapter**). One command per unit:

```bash
docker compose run --rm anim export knight                       # work/_export/knight/
docker compose run --rm anim export knight --sheet packed --size 48
docker compose run --rm anim export knight --adapter game02      # + publish per job, + the audit
```

```
work/_export/knight/
  sheets/knight_s0.png     per direction: one row per action (alphabetical), frames left to right,
                           cell = frame size, transparent padding; or ONE sheets/knight.png (--sheet packed)
  atlas.json               anim.atlas/1 — unit, frame_size, fps, palette, per <action>_<dir>: sheet, frame
                           rects, phase tag + ticks + duration_ms per frame, loop, pivot, anchor,
                           source {job, picks, timing_source, seed, clip, motion_source}; tool {commit}
  knight.tres              Godot 4 SpriteFrames: one AtlasTexture per frame, one animation per
                           <action>_<dir>, speed = fps, per-frame duration = ticks, loop from the plan
  preview.html             the sheets at ×4 with the rects drawn, every animation playing at its durations
```

- **Which jobs.** The jobs named `<unit>_<action>` (direction `s0`) and `<unit>_<action>_<dir>`
  — the names `anim rig` makes — whose `job.json` agrees (slug, action, direction) and whose
  `post/` has frames. `knight_strike_phases` is skipped with a warning (not the matrix's job);
  a base or concept job is a still, not an animation; a `--no-snap` preview is refused. The
  matrix is the studio's five actions × the eight side cells plus whatever the unit has, and
  every empty cell is a **listed warning** (`atlas.json.warnings`, the CLI's stderr), never an
  error — today the knight has `s0` only.
- **Timing.** The same `publish.resolve_timing` the sidecar uses (hand edit > what `pick`
  recorded > template > uniform), in **ticks** at the animation's fps: a phased pick's weights,
  the strip template's holds, or 1 per frame. `duration_ms = ticks × 1000 / fps`; the `.tres`
  keeps the ticks as Godot's per-frame `duration` with `speed = fps`, so both play the same.
  Loop: the phase plan's flag, else the studio's row (idle / walk / run loop), else off.
- **The cell.** `frame_size` in `work/project.json` (`GET /settings`), default 64×64 — a
  project setting, not a constant. A smaller frame is padded bottom-centre; a larger one is an
  error. Pivot is the cell's bottom centre (`animations.*.pivot = [w/2, h]`).
- **Adapters.** `generic` writes the three files; `game02` also runs `publish` for every job
  it exported — the numbered PNGs, the sidecar and `asset_audit.py`, byte-for-byte what
  `anim publish` alone writes — so game-02 keeps working unchanged. `export.adapter` in the
  settings picks the default; `--adapter` overrides per run.
- **In Godot.** Drop the directory into the project (the `ext_resource` paths are relative to
  the `.tres`, so it works anywhere under `res://`), let the editor import the PNGs, then:

  ```gdscript
  $AnimatedSprite2D.sprite_frames = load("res://units/knight/knight.tres")
  $AnimatedSprite2D.centered = false
  $AnimatedSprite2D.offset = Vector2(-32, -64)   # the atlas pivot: bottom centre of a 64×64 cell
  $AnimatedSprite2D.play(&"strike_s0")
  ```

  Set `rendering/textures/canvas_textures/default_texture_filter` to Nearest for pixel art.
  Verified 2026-09-21 on Godot 4.6.3 headless: the knight's `.tres` loads with its three
  animations, per-frame durations and `AtlasTexture` regions intact, no errors.
- **The API.** `POST /jobs {"stage": "export", "args": {"unit": "knight"}}` runs it like any
  CPU stage; `GET /export/{unit}` lists the last export (files, animations, warnings, commit).

### The review page

The click-to-pick sheet is served by the pipeline API (below), not by a stage:

```bash
docker compose up -d anim-api
# then, from the Mac:
open http://192.168.1.158:8720/                      # every job, its stages, recent runs
open http://192.168.1.158:8720/jobs/knight_strike/review
```

A contact sheet of every extracted frame, picks highlighted, dark ground and
`image-rendering: pixelated`. Clicking a frame toggles it and `PUT`s the whole
list to `/jobs/<job>/picks`, which rewrites `picks.json` immediately through the
same `ReviewApp` the old page used — byte for byte the same file (a test checks).
The other picks keep their own frame numbers. Re-run `anim strip` afterwards
and the new picks are what gets composed.

**The "Request regenerate" button does not touch the GPU.** It sets
`rerender_requested` in `job.json` and stops. A page that could start a 24 GB
model load is a page a stray browser prefetch can fire, and the GPU service is
on the other Docker engine and usually not running at all. Starting a render on
purpose is `POST /jobs {"stage": "render", …}` — explicit, one at a time.

`anim review <job>` still exists and prints the URL; the stdlib server it used to
start on `:8710` is gone, as is the `review` compose service.

## The pipeline API — the pipeline API feature

FastAPI over `work/`, so the editor ([#280](http://192.168.1.158:8700/work_packages/280))
and the curation board ([#281](http://192.168.1.158:8700/work_packages/281)) drive the
same `work/<job>/` files the CLI writes. **No database**: a job is its directory, a run is
the CLI in a subprocess plus a log under `work/_runs/`, the project settings are
`work/project.json`.

```bash
docker compose up -d anim-api              # http://192.168.1.158:8720/  (docs at /docs)
curl -s http://192.168.1.158:8720/jobs | jq '.[].name'
curl -s -X POST http://192.168.1.158:8720/jobs \
     -H 'Content-Type: application/json' \
     -d '{"stage": "pick", "args": {"job": "knight_strike", "n": 3}}'
curl -N http://192.168.1.158:8720/runs/<id>/events      # live log as SSE
```

| Method, path | What |
|---|---|
| `GET /jobs` | every directory under `work/` with a `job.json`: name, unit, action, direction cell, stages done (derived from the files, not a status field), updated, newest first |
| `GET /jobs/{job}` | `job.json` as written, plus the stage list, the current picks and how many PNGs each set has |
| `GET /jobs/{job}/frames` | every PNG set — `frames`, `pose`, `depth`, `clip`, `control` (what `clip` sent as the control video), `strip`, `render`, `post`, `candidates` — each entry with its URL |
| `GET /jobs/{job}/frames/{set}/{n}.png` | the PNG; `n` is the index in that listing (0-based, the same numbers `picks.json` uses), never a filename from the URL. The `post` set is the `<stem>_<n>.png` frames in the order of `n` — `palette.png` and the `_hd` / `_labels` diagnostics of `post --parts` are not frames — so post frame `k` is pick `k` |
| `PUT /jobs/{job}/frames/{set}/{n}.png` | the editor's save: the body is the PNG (`Content-Type: image/png`), overwriting that frame **in place** (`write_bytes_in_place`, the inode rule). Same size as the file it replaces or `409`; not a PNG `415`; no such frame `404` — it never creates one. Alpha is rounded to the contract before writing (`>= 128` opaque, else the pixel becomes `(0,0,0,0)`, `imaging.binarise_alpha`); the reply says how many pixels needed it |
| `GET /jobs/{job}/picks` | the review state (`ReviewApp.state()`) |
| `PUT /jobs/{job}/picks` `{"picks": [...]}` | rewrite `picks.json` exactly as the review page did; sorted, de-duplicated, out-of-range dropped, a hand `timing` kept |
| `POST /jobs/{job}/regenerate` | flag `rerender_requested` in `job.json`. Never the GPU |
| `POST /jobs` `{"stage": "pick", "args": {...}}` | run a stage in the background → `202` + the run, `Location: /runs/{id}`. `args` keys are the stage's `--help` names (`job`, `n`, `force`, `slug`, `audit: false` → `--no-audit`); or `"argv": ["knight_strike", "-n", "3"]` verbatim. Unknown keys and bad values are a `400` in argparse's own words. **`423 Locked`** with the reason when the unit's curation gate blocks the stage (the curation board feature, below); `force: true` runs it anyway |
| `GET /runs` · `GET /runs/{id}?tail=N` | every run this API knows, newest first; one run with its log. `status` is `queued` \| `running` \| `done` \| `partial` \| `failed` \| `lost` \| `cancelled`, and `cells` is the batch stages' tally — `{done, failed, blocked, of}` for `matrix` and `bakeoff`, `null` for every other stage and for a run recorded before 2026-09-24 |
| `GET /runs/{id}/events` | `text/event-stream`: `log` per line, `progress` when the percent changes, one `done` (`status`, `returncode`, `percent`, `cells`). A stream opened after the run ended replays the whole log |
| `GET /settings` · `PUT /settings` | `work/project.json`: `palette`, `frame_budgets` (`{action: 1..12}`), `curation` (`gates` \| `every` \| `end` — the curation board's mode, read by every gate check), `frame_size` (`[w, h]`), `export` (`{"adapter": "generic" \| "game02"}`). `palette` is stored as a path (null = `ANIM_PALETTE`) and shown expanded — `{path, name, colours, source}` read from the palette files; a PUT takes the path string or that object back. The keyframe phase, curation and export features read the rest |
| `GET /palette` | the project palette record (`palette.json`, or one derived from a bare `.gpl`) plus `units`; `404` with the fix when nothing is installed |
| `POST /palette` `{"lospec": "endesga-32"}` \| `{"file": "/path.hex"}` \| `{"unit": "knight", "colours": [...] \| "from_sprite": "/path.png" \| "clear": true}` | run `anim palette …` in the background (a CPU run) → `202` + the run, `--into` the settings' palette path |
| `GET /palette/units` · `GET /palette/units/{unit}` | the units that have a sub-palette; one unit's `palette_units/<unit>.json` (`404`: the whole project palette applies) |
| `GET /curation` | the units that have a cell job or a curation record, and the project's curation mode |
| `GET /curation/{unit}` | the board: the gates, every cell (`attack/s0` …) with its job, status, frame marks, seed, the strip it shows (`post` when it matches the picks, else `clip` / `frames`) and the GPU stage a regenerate would run, plus `waiting` - what the owner has to decide |
| `PUT /curation/{unit}/cells/{action}/{dir}` `{"status": "accepted" \| "rejected" \| "pending"}` and/or `{"frames": {"3": "reject", "5": "accept", "7": null}}` | mark frames (a reject leaves that job's `picks.json` through `ReviewApp.set_picks`, a cleared reject comes back), then set the cell. An accepted cell is locked: `409` on a frame mark until it is set pending again |
| `POST /curation/{unit}/cells/{action}/{dir}/regenerate` `[{"seed": N}]` | record a new seed for the cell and run its GPU stage (`clip` on the clip route, else `render`) through the run manager → `202` + the cell with its run. A busy card is the `409`, a pending gate the `423`; a refused request records nothing |
| `PUT /curation/{unit}/gates/{gate}` `{"state": "accepted" \| "pending"}` | `base` (the one gate; any other name is a `404`); answers the board |
| `PUT /curation/{unit}/views/{view}` `{"state": "accepted" \| "rejected" \| "pending"}` | the views step under the base gate (the multi-view reference feature): one view of the base; the board's `views` row carries each view's state, file URL, `provisional`, and (the mirrored view feature) `mirror_of` — the partner it could be flipped from — with `mirrored` / `mirrored_from` when it already is one |
| `GET /export/{unit}` | the last `anim export` of a unit under `work/_export/<unit>/`: `created`, `commit`, `frame_size`, `sheet_mode`, `sheets`, `animations` (frames, loop, fps, sheet, job), `warnings`, `files` with sizes; `404` with the `POST /jobs` line when there is none (the spritesheet export feature) |
| `GET /matrix/{unit}` | the action matrix: every planned (action × direction) cell, its prerequisite verdict, its finished frames' URLs, the seed and the `#277` cap; a matrix **launches** through `POST /jobs {"stage": "matrix"}` like any stage — there is no `POST /matrix/…` |
| `GET /concept/{unit}` | the concept generation feature: the base attempts (`candidates`, each with its `url`, `source`, `prompt`, `seed`, `model`, `loras`, `parent`), the accepted `pick`, `front_url`; every view with its `prompt`, `default_prompt`, `group` (`main` \| `advanced`), attempts, `pick` and curation `state`; `locked` (why the views may not be drawn now, or null), `gates`, the unit's model `defaults`, `last_seed`, `views_stale`, `estimate_seconds` — the measured warm median seconds for one view render (this unit's own recorded renders, falling back to every unit's, then to a constant), read by the advanced views sweep feature's confirm |
| `POST /concept/{unit}/generate` | `{"op": "txt2img" \| "views" \| "img2img" \| "edit", …}` with that op's `anim concept` flags by name (`prompt`, `views`, `view_prompt`, `ref`, `strength`, `instruction`, `count` 1–4, `seed`, `model`, `loras`, `lora_strength`, `fresh`, `force`) → `anim concept …` in the background, `202` + the run; `423` while a view op waits for the base gate, `409` while the GPU is busy, `400` for an op missing its input |
| `PUT /concept/{unit}/upload` | the body is the PNG (`X-Filename` optional) → a base candidate of source `upload`; `415` when it is not a PNG. No run: the state comes back with `uploaded` |
| `POST /concept/{unit}/pick` `{"index", "parts"?, "avoid"?}` | accept that base candidate — `anim concept --pick`, the base gate; `400` no such candidate |
| `POST /concept/{unit}/views/{view}/pick` `{"index"}` \| `{"mirror": true}` | accept that view attempt (the matrix's reference for the facings that want it) — `--pick-view view=NN`; with `mirror` instead of an index it is the mirrored view feature's `--mirror-view view` — the horizontal flip of the view's accepted partner, no GPU, and the answer carries the handedness `warning`. `400` for a view with no partner (`back`, `top`), for a partner nobody has accepted, and for a body with neither field |
| `PUT /concept/{unit}/attempts/{target}/{index}/image` | the retouch feature: the body is a PNG of that attempt's own size, written over it (transparency flattened onto white); the row gains an `edited` block and a new `sha256`. `400` wrong size / hidden / deleted, `415` not a PNG |
| `POST /concept/{unit}/attempts/{target}/{index}/inpaint` | the body is the canvas as the editor left it (RGBA, erased = redraw); it is stashed under the job and redrawn by `anim concept --inpaint` through the run manager → `202` + the run. `?prompt=`, `?strength=`, `?seed=`, `?model=`. `400` when nothing is erased |
| `PUT /concept/{unit}/attempts/{target}/{index}` `{"hidden"}` | hide one attempt from the strips and the views page, or show it again; `target` is `base` or a view name. `400` on the accepted one with the reason, `404` on an unknown target |
| `DELETE /concept/{unit}/attempts/{target}/{index}` | delete a **hidden** attempt's PNG; the row stays as its record (its number is never reused). `400` when it is not hidden, or is the accepted one |
| `PUT /concept/{unit}/defaults` `{"family", "model", "loras"}` | the unit's default checkpoint + LoRAs for one family, checked against the registry (`400` otherwise) |
| `GET /animate/{unit}/{action}/{dir}` | the animate tab feature: the `compass` (the eight side directions, each with its own view's thumbnail `url` or the `blocked` reason, its `mirror_of`, and `approved` / `frames` for this action), the `actions` with their frame counts, the `frames` row — per frame `id`, `position`, `label`, `key`, `phase`, `at` (where in the clip), `status` (`empty` \| `generating` \| `review` \| `approved` \| `rejected`), `current`, `verdict`, `prompt` / `default_prompt`, `draw_from`, the `before` / `after` keyframes with `approved`, `default_skeleton_frame` / `skeleton_frame`, every attempt, `game_url` — plus `counts`, `next_step` (Generate all's step), `reference` (the direction's view), `locked`, the `rig` clip, `skeleton_url`, `mirror` (`partner`, `available`, the handedness `warning`) and `output` (`job`, `written`, `stale`, `ready`, `error`). `{dir}` is a cell or a view name (`left`); `404` for a top cell or an unknown action |
| `POST /animate/{unit}/{action}/{dir}/generate` | `{"frame"}` (id, number or phase) or `{"batch": "all" \| "keys" \| "rejected" \| "inbetweens"}`, with `prompt`, `draw_from` (`view` \| `keys` \| `both`), `skeleton` / `skeleton_frame`, `count` 1–4, `seed`, `model`, `force` → `anim animate …` in the background, `202` + the run; `423` while the base gate is pending, `409` while the GPU is busy; `400` **before the run** for a frame that may not be drawn (the view not accepted, a neighbouring keyframe not approved, four images) |
| `POST /animate/{unit}/{action}/{dir}/frames` `{"after"}` · `PUT …/frames/{id}` `{"key", "phase"}` \| `{"current"}` · `DELETE …/frames/{id}` | the row: insert an in-between after frame `after` (0 = first); mark / un-mark a keyframe with a phase label, or show another attempt; take a frame out (its attempts move to the record's `removed`) |
| `POST /animate/{unit}/{action}/{dir}/frames/{id}/approve` · `…/reject` `{"index"?}` · `POST …/approve-reviewed` | the verdict on the frame's current attempt (or `index`, which becomes current); every frame to review at once. The approval that completes the row writes the output; a refusal there lands in `output.error` |
| `POST /animate/{unit}/{action}/{dir}/mirror` | the partner direction's frames flipped horizontally into this one (right↔left, back_right↔back_left, front_left↔front_right), no GPU, each a new attempt **to review**; the answer carries `warning` (handedness). `400` for `s90` / `s270` or a partner with no pictures |
| `POST /animate/{unit}/{action}/{dir}/output` `{"replace_job"?}` | every approved frame through the post chain into `work/<unit>_<verb>[_<dir>]/`; `400` until every frame is approved, or when another route's job is there and `replace_job` is not set |
| `PUT /animate/{unit}/{action}/{dir}/attempts/{id}/{index}` `{"hidden"}` · `DELETE` · `…/image` · `POST …/inpaint` | hide / show; delete a hidden attempt's PNG; the editor's Save over it; the masked redraw (the canvas, erased = redraw → `anim animate --inpaint`, `202` + the run). An edit of the judged attempt clears the verdict |
| `GET /animate/{unit}/{action}/{dir}/skeleton/{n}.png` · `GET …/game/{id}.png` | scrub frame `n` of the rig clip at that direction's yaw (CPU, drawn on the first ask, cached); the frame's current attempt at game size (the post chain on the CPU, cached by hash) |
| `GET /models` | `gpu/models.registry.tsv` by family and role, each row `locked` (in the lock) and — with `?probe=1` — `installed` (ComfyUI lists the file; 3 s timeout); plus the ops' families and the workflows' own checkpoints |
| `POST /runs/{run}/cancel` | stop a queued or running stage: SIGTERM to its process group, SIGKILL after 5 s, ComfyUI `/interrupt` for a GPU stage; `202` + the run (it ends `cancelled`), `409` when it is over, `404` unknown |
| `GET /health` | `ok`, the work dir, the active GPU run; `?probe=1` also asks ComfyUI (3 s timeout) |
| `GET /` | the web pixel editor (`web/dist/`, see below), else a page listing the jobs and runs |
| `GET /jobs/{job}/review` | the click-to-pick page |

**The GPU rule.** `base`, `render`, `clip`, `concept`, `animate`, `bakeoff`, `matrix` and `run` (which renders
when it can) reach ComfyUI. They start **only from an explicit `POST /jobs`** — nothing in the
API polls or prefetches — and **one at a time**: a second is `409` with the id of the one
running. CPU stages run concurrently up to `ANIM_API_CPU_SLOTS` (default 2), then `409` too.
`review` is a page, not a stage, and cannot be posted.

**How a run works.** `POST /jobs` becomes `python -m anim <stage> <argv>` in a subprocess
with `ANIM_WORK_DIR` set to the API's `work/` and stdout streamed line by line into
`work/_runs/<id>.log`; `<id>.json` beside it is the status. So the API never re-implements a
stage's arguments or steps — a stage that gains a flag gains it here for free — and a stage
that dies hard (an ONNX segfault, a torch OOM) cannot take the API down. Percent comes from
lines a stage already prints: `[n/m] …` (what `anim run` prints per step) or `progress: 42%`.
After a restart the runs on disk are listed again; one that was still active belonged to a
process that no longer exists and is reported `lost`.

Port **8720** is the sprite studio's (`sprite-studio.md` §2); free on 2026-09-21 (`ss -ltn`;
the neighbours are 8700 openproject, 8770 research-digest, 8787 bg3-modded-wiki). Bound to
`0.0.0.0` because the owner opens it from the Mac — a `curl` from the Fedora host to its own
LAN IP routes via `lo` and proves nothing about LAN reach. Same image and mounts as the
`anim` service, so a run through the API writes exactly what `docker compose run --rm anim`
would.

### The web pixel editor — the web pixel editor feature

[#280](http://192.168.1.158:8700/work_packages/280). What `GET /` serves: **Piskel**, the
open-source pixel editor (layers, onion skin, timeline, palettes, pencil / bucket / shapes /
selection, PNG / GIF / sheet export), with the pipeline as a panel down the left side.

```
http://192.168.1.158:8720/          the editor (from the Mac; Alt+P folds the panel)
```

**Where Piskel comes from.** `web/vendor/piskel/` is upstream's `src/` at the commit pinned in
[`web/vendor/piskel/UPSTREAM`](web/vendor/piskel/UPSTREAM) (Apache-2.0; its LICENSE is beside
it, the line is in [`NOTICE`](NOTICE)). It is vendored as **source, not as a built bundle**,
because upstream's build is three small steps — join the script list, join the style list,
resolve the `@@include` partials of `index.html` — and [`web/piskel-plugin.mjs`](web/piskel-plugin.mjs)
does exactly those three with no dependencies, so the shell needs none of upstream's toolchain.
Every change to the vendored tree is listed in
[`web/vendor/piskel/CHANGES.md`](web/vendor/piskel/CHANGES.md) (one: a hook that lets the
timeline tiles carry a text tag); a re-pull is "copy upstream's `src/` over, re-apply that list".

**The shell** (`web/`, Vite + TypeScript, `web/src/`) boots Piskel and mounts the panel:

| Section | What it does today |
|---|---|
| Job | `GET /jobs`; Open loads the job's best PNG set — `post` (the 64×64 frames), else `clip`, else `frames` — as one animation, one layer, at the job's fps. The keyframe phase of each frame (`/jobs/{job}/picks`) shows on its timeline tile: `start`, `wind-up`, `action`… |
| Save | every frame, layers flattened, alpha rounded to 0/255, `PUT` back to the set it came from. Same frame count and size as the set, or it refuses and says why |
| Pipeline | `pick --phases` and `post` (CPU) start at a click; `clip` and `bakeoff` (GPU) ask first — "this starts a GPU job". A `409` reads "GPU busy with `<id>`" and the panel follows that run instead. The concept stage moved to its own tab |
| Run | the run's log live over `/runs/{id}/events` (SSE) with a progress bar; recent runs are one click to follow |
| Palette | the project palette from `GET /settings` (`palette.colours`, the palette feature's shape) goes into Piskel's palette list and is selected; a path-only or empty `palette` reads "no project palette" |
| Board (a second tab) | the curation board — see "The curation board" below. The tab widens the panel; the Run section shows on every tab |
| Concept (a third tab) | the concept generation feature — the base, its views, regenerate / img2img / edit per attempt, the model and LoRA picker, Cancel. See "The Concept tab" below. **Edit pixels** there opens one attempt here and Save writes it back over itself (the retouch feature) |
| Animate (a fourth tab) | the animate tab feature — every frame of one animation drawn by hand, direction by direction, approved one by one, played, and written as the cell's finished frames. See "The Animate tab" below. **Pixel edit** and **Inpaint** there open one frame attempt here; Save writes it back, Inpaint redraws what you rubbed out |
| Matrix (a fifth tab) | the action matrix feature (#278) — one unit, the five studio actions × the eight side directions as a grid. A press is `POST /jobs {"stage": "matrix"}`, same as `anim matrix` and every other stage's button, so a busy card's `409`, a blocked cell's `423` and the run's log come for free — there is no separate accept or reject here; a blocked cell carries the stage's own reason as its text and its title and wears the Concept tab's `.locked` styling |

**Build.** `web/dist/` is generated, never committed. The Dockerfile's node stage runs
`npm ci && npm run build` and bakes it into the image at `/opt/anim/web/dist`; the API
(`config.WEB_DIST`) prefers the checkout's own `web/dist` when one exists — so on the host:

```bash
cd web && npm ci && npm run build     # -> web/dist, served by the running anim-api at once
cd web && npm test                    # vitest: the panel's pure parts (frame sets, phase tags, SSE, palette, alpha, the board, the concept and animate tabs)
cd web && npm run dev                 # http://localhost:5173, API routes proxied to :8720
docker compose up -d --build anim-api # or rebuild the image and let it serve its own copy
```

**Not yet.** Adding or removing frames is not saved back (the set's count is the count,
including a matrix cell's — the frame add/remove round-trip); `render/` and `strip/` are
strips, not frames, so they do not open (strip-set slicing in the editor); the palette panel
reads the project palette but does not import one (the palette feature, #282). Beyond the
editor: the top-down pitch (`t0` … `t315`, #285), video intake (#286) and sheet slicing (#287)
are still unbuilt, and the action matrix renders each cell's whole clip once — the motion
exaggeration feature (#277) is the hold that keeps it from having to render twice. Piskel's
own Save / Export drawer still works — it writes to the browser or downloads, not to `work/`.

### Character concept — the character concept feature

The studio's first question is *who* the character is, before any pose ([#273](http://192.168.1.158:8700/work_packages/273)).
`concept` answers it with candidates the owner chooses from — generated ones and, side by
side, the art that already exists:

```bash
# four candidates from text, the published knight.png registered as 00 to compare against
docker compose run --rm anim concept --unit knight --count 4 --existing \
  --prompt "a plate-armoured knight with a closed great helm, a kite shield on the left arm and a longsword in the right hand"

# the same, steered by a sketch (any dark-on-light drawing; sent as a scribble condition)
docker compose run --rm anim concept --unit knight --sketch scans/knight.png --prompt "…"

# choose: the candidate becomes work/knight_base/render/concept_pick.png - and say which parts
# every clip must keep and what its negative names (the multi-view reference feature, #298)
docker compose run --rm anim concept --unit knight --pick 03 \
  --parts "closed great helm, kite shield, longsword, dark cloak" --avoid "bare head, uncovered face"
docker compose run --rm anim concept --unit knight --parts "…" --avoid "…"   # the phrases alone, no new pick
docker compose run --rm anim post knight_base --parts && docker compose run --rm anim publish knight_base

# the views: the accepted base seen from the other sides, drawn FROM it by Qwen-Image-Edit
docker compose run --rm anim concept --unit knight --views                 # back, left, right, top; 2 attempts each
docker compose run --rm anim concept --unit knight --views all --count 3   # + the four obliques
docker compose run --rm anim concept --unit knight --views advanced        # right + the four obliques only
docker compose run --rm anim concept --unit knight --views left --view-prompt "…"   # that view again, your sentence
mkdir -p ~/.cache/claude/visuals/sprite-studio/views && cp -r work/_views/knight/. ~/.cache/claude/visuals/sprite-studio/views/
docker compose run --rm anim concept --unit knight --pick-view back=01 left=00 top=01
docker compose run --rm anim concept --unit knight --mirror-view right   # the left view flipped: no GPU

# the concept generation feature: rework one attempt, choose the weights, keep everything
docker compose run --rm anim concept --unit knight --upload scans/knight_base.png        # your art as a candidate
docker compose run --rm anim concept --unit knight --img2img base:02 --strength 0.5      # classic redraw, composition kept
docker compose run --rm anim concept --unit knight --edit left:01 --instruction "make the helmet red"
docker compose run --rm anim concept --unit knight --prompt "…" --lora pixel_art_style_z_image_turbo.safetensors:0.6
docker compose run --rm anim concept --unit knight --prompt "…" --fresh                   # start the unit's attempts over
docker compose run --rm anim concept --unit knight --hide base:03                        # out of the strips; --show brings it back
docker compose run --rm anim concept --unit knight --delete-file base:03                 # the PNG of a hidden attempt; the row stays

# the retouch feature: change an attempt where it sits (the Concept tab does this with buttons)
docker compose run --rm anim concept --unit knight --replace base:02 --image fixed.png    # write over it by hand
docker compose run --rm anim concept --unit knight --inpaint base:02 --mask erased.png \
  --prompt "a closed great helm" --strength 0.9                                            # redraw only what was erased
docker compose run --rm anim concept --unit knight --edit base:02 --instruction "…" --in-place
```

Or none of that: the studio's **Concept tab** (<http://192.168.1.158:8720/> → Concept) is the
same stage with buttons — see "The Concept tab" below.

- **No skeleton.** `base` is the pose stack because a standing sprite has to match the rig;
  a concept is earlier than that, so it is plain text-to-image on the same int8 Z-Image with
  **`tarn59/pixel_art_style_lora_z_image_turbo`** (Apache-2.0, 162 MB, `gpu/models.lock`) at
  0.8 (`--lora-strength`; 0 drops the node). The prompt is `config.CONCEPT_PROMPT`: the
  LoRA's trigger ("Pixel art style."), your description, then the base pass's framing and
  flat-white-background rules so `post`'s matte has something to key on.
- **One `/prompt` and one seed per attempt** (first seed + index; random unless `--seed` is
  given, derived from unit + prompt + sketch on a `--fresh` run so a unit's first set
  reproduces), so any one attempt can be re-made alone. **Every attempt is kept** (schema
  `anim.concept/2`, the concept generation feature): a run appends to `candidates/` and
  `job.json`, `--fresh` is the old wipe, and each row says what it was made from and with —
  `source` (`txt2img` / `upload` / `existing` / `img2img` / `edit`), its own prompt, seed,
  `model`, `loras`, `parent` (the attempt it reworks), `strength` or `instruction`, the
  workflow hash as sent and the locked models. A `/1` job is migrated in place on first touch.
- **The sketch is a scribble.** The Fun ControlNet Union picks the condition from the image
  alone, so the sketch is sent as white strokes on black (a dark-on-light scan is inverted,
  transparency counts as paper), letterboxed into the cell, at `--sketch-strength` 0.6.
- **Where the pick lands.** Every later stage reads the unit's reference from the base job —
  `clip` takes `work/<slug>_base/render/<render.final>`, `post --parts` extracts the part
  palettes there, `post` + `publish` make the 64 px sprite — so `--pick` lays that job out
  (`base.prepare`, at the candidate's size) and writes `render/concept_pick.png` with
  `render.final` pointing at it and a provenance record from the concept job. `anim base` is
  untouched; an earlier base render's files stay on disk and the note says what was replaced;
  picking again changes the pick. The manual test page:
  <http://192.168.1.158:8090/sprite-studio/character-concept/>.
- **The parts phrase** (the multi-view reference feature, [#298](http://192.168.1.158:8700/work_packages/298)).
  A front-view reference cannot say what the back of a closed helm looks like, and Wan drew bare
  heads on the turned cells of the first identity gate. `--pick … --parts "…" --avoid "…"` (or
  `--parts` / `--avoid` alone) stores a `parts_phrase` and an `avoid_phrase` in the base's
  `concept` block — explicit strings, no parsing; with no `--parts` the phrase is the whole
  description. `clip` reads them off the base for **every** clip of the unit: the prompt becomes
  `config.CLIP_PARTS_PROMPT` (the view, the action, "with the same *parts* as the reference
  image", what that view shows of the head — `config.CLIP_IDENTITY_PHRASES`, "no face visible"
  on the turned-away cells — "exactly one figure"), and the negative is Wan's own plus "a second
  figure, two people, an extra person" plus the avoid phrase. A base without a parts phrase
  renders the old prompt word for word.
- **The views.** `--views [back,left,right | main | advanced | all]` renders the accepted base
  from the other sides through the identity edit path already in the container — Qwen-Image-Edit 2509 INT4,
  `gpu/workflows/view_pass.json`: the base is `image1` and the latent (so a view is the base's
  size), the prompt is `config.CONCEPT_VIEWS[view]` ("seen from directly behind, turned all the
  way round: the back of the head …") plus the parts phrase. Views are named for the **side of
  the character** the camera sees, the compass in `config.DIRECTIONS`: `right` is the right
  profile (yaw 0, facing screen right — the s0 cell), `left` yaw 180, `back` yaw 90, the obliques
  `back_right` / `back_left` / `front_left` / `front_right`; `front` is the base itself. Beside a
  comma list and `all`, `--views` takes a **group**: `main` is `config.CONCEPT_MAIN_VIEWS` (back,
  left, top) and `advanced` is `config.CONCEPT_ADVANCED_VIEWS` (right and the four obliques — what
  the advanced views sweep feature's button sends). A group may sit in a list with plain names
  (`--views main,front_right`), and the answer stays in table order with no duplicates. N
  candidates per view (`--count`, default 2; seed + 100 per view + 1 per candidate) land in
  `work/<unit>_concept/views/<view>_NN.png` with `views.json`, and the page
  (`work/_views/<unit>/`, copied to <http://192.168.1.158:8090/sprite-studio/views/>) shows the
  base beside each view's candidates. `--pick-view back=01 left=00 …` copies the chosen ones into
  the base as `render/view_<view>.png`, records them in the base's `concept.views` and accepts
  them in the curation record's **views** step (`anim curate knight --accept-view back` /
  `--reject-view`, or the Board's row under the gates); `--provisional` records a pick made for a
  rerun as pending. A new `--pick` clears the views — they were drawn from another base. The
  alternative conditioning — Z-Image + the Fun ControlNet with a rig-rendered depth of the
  mannequin at that yaw — holds the pose but carries identity in the prompt only, the very
  failure this fixes; it is the fallback if the edit path stops keeping the character.
  Since the concept generation feature a view asked for again gets **more** attempts (its
  earlier ones and its pick stay), `--view-prompt "…"` replaces the config sentence for a
  single view, and **`top`** (yaw 270 from above, `elevation: top`) is a view: the
  matrix's `t*` cells take it first (`config.matrix_reference_views`). The views
  **wait behind the base gate** — `anim concept --views` / `--edit <view>:NN` refuse with the
  reason (the API's `423`) until the base is accepted; `--force` runs anyway.
- **The mirrored view feature — three of the eight views need no GPU at all.** By the same
  compass, a horizontal flip of the picture takes a camera at yaw *t* to the camera at yaw
  *180 − t*, so three views are the same pixels the other way round: `right` (0) ↔ `left`
  (180), `back_right` (45) ↔ `back_left` (135), `front_right` (315) ↔ `front_left` (225).
  `--mirror-view right` derives that view by flipping its partner instead of spending a
  Qwen-Image-Edit round trip; it takes several names (`--mirror-view right back_right`) and
  either half of a pair may be the source. `back` (90), `top` and `front` look straight down
  the symmetry plane and map to themselves, so they have **no** partner and are refused by
  name, listing the six that are mirrorable. The pairs are computed from the yaws in
  `config.CONCEPT_VIEWS` (`config.CONCEPT_MIRROR_PAIRS`, `config.mirror_of`), not typed out a
  second time.
  - **The partner must be accepted, not merely drawn** — the matrix's strictness
    (`matrix.prerequisite`): a mirrored view is what a whole facing will
    be rendered against, and deriving it from an attempt nobody has judged would launder an
    unjudged image into an accepted one. The refusal says which command fixes it.
  - **It lands exactly where `--pick-view` lands** — `work/<unit>_base/render/view_<view>.png`,
    the base's `concept.views` entry, the provenance sidecar and the curation record's views
    step, accepted — so `matrix` and the Board read one shape and nothing
    downstream knows the difference. The entry adds `mirrored: true` and a `mirror` block
    naming the source view, the file it was flipped from and that file's sha256; the entry's
    own `sha256` is of the flipped bytes, so *(source hash, flip horizontally)* reproduces the
    picture exactly — which no seed can promise. `candidate` and `seed` are `null`: nothing was
    sampled for it.
  - **One warning, once per invocation.** When `config.UNIT_PROPS` lists props for the unit, or
    the base's parts phrase names handed gear (`concept.HANDED_GEAR_WORDS` — sword, shield, bow,
    …), the run prints one line saying the flip swaps handedness and names the gear. It does not
    refuse: the owner decides whether the sword hand matters for that view.
  - **Nothing is mirrored behind the owner's back.** A matrix cell never is; a whole Animate
    animation is only when the owner presses "Mirror from <partner>" for it, and the flipped
    frames come back to review (see "The Animate tab"). A left-handed swing read as a
    right-handed one is a bug nobody sees until the sheet ships, so it stays a person's choice.

### The Concept tab — the concept generation feature

The studio's Concept tab ([#300](http://192.168.1.158:8700/work_packages/300)) is `anim
concept` with buttons and a memory: the base, its views, and a way to fix one attempt without
redoing the rest.

- **The base.** A prompt and **Generate** (× 1–4 per press, `--count`), or **Upload PNG**
  (`--upload`: the file lands under `uploads/` as given and fitted into the cell as a
  candidate — no GPU, no prompt). Every attempt stays in the strip; click one and **Accept**
  it — that is `--pick`, and the base gate.
- **The views.** **All** (back, side, top-down), **Front** (the accepted base itself — never
  rendered), **Back**, **Side** (the left profile; the right is mirrored in a sheet),
  **Top-down**; the right profile and the four obliques under an *advanced* fold. Each view
  has its own prompt (seeded from `config.CONCEPT_VIEWS`, `--view-prompt`), its own strip of
  attempts, and one accepted (`--pick-view`). Locked, with the reason, until the base is
  accepted. A paired view also gets a **mirror from `<partner>`** button (the mirrored view
  feature) — disabled, with the reason in its title, until that partner's own view is accepted;
  a view that *is* a flip carries a `⇄ mirrored from …` marker instead of an outlined tile,
  here and on the Board's views ring. The advanced fold also has **All advanced (N)** (the
  advanced views sweep feature) — one press that sends `POST /concept/{unit}/generate
  {"op": "views", "views": [...]}` for whatever advanced views have no accepted pick yet, at
  the tab's own attempts-per-view count. `N` in the label is how many views it would render, so
  a view already settled is left out and a second press only tops up what is unfinished. It confirms first —
  unlike every other press on this tab — with the render count and a measured estimate (the
  warm median from this unit's own recorded view renders, or every unit's, or a constant; see
  `GET /concept/{unit}`'s `estimate_seconds`), plus the separate cold-start penalty when
  Qwen-Image-Edit has to page into VRAM. It does not mirror: a flip still needs its partner
  accepted first and can swap the handedness of handed gear, so mirroring stays the explicit
  per-view button.
- **Click an attempt** — a base candidate or a view attempt — for the action bar:
  **Accept** · **Regenerate** (that target again with its prompt; a new seed unless the seed
  is locked) · **img2img** (`--img2img REF --strength`: the attempt as the latent, redrawn at
  the slider's strength on the Z-Image stack — `gpu/workflows/img2img_pass.json`, 0.3–0.9,
  default `config.CONCEPT_IMG2IMG_STRENGTH` 0.55 — the composition kept; an optional prompt,
  else the parent's, or for a view the unit's description seen from that side) · **Edit**
  (`--edit REF --instruction "make the helmet red"`: the attempt as `image1` through
  Qwen-Image-Edit with `config.CONCEPT_EDIT_PROMPT`). The new attempt lands on the same
  target with `parent` = the one it reworks.
- **The checkpoint and the LoRAs.** The settings row has one picker per family — *base*
  (Z-Image: txt2img and img2img) and *views / edit* (Qwen-Image-Edit) — populated from
  **`gpu/models.registry.tsv`** (`GET /models`): a hand-curated sibling of `models.lock`
  (which is generated and fixed at 8 columns) saying which weight is a checkpoint / LoRA /
  ControlNet / text encoder / VAE and for which family (`zimage` | `qwen-edit` | `wan`), with
  a LoRA's trigger and default strength. Only the family's rows are offered, and the stage
  refuses anything else before ComfyUI would (`models_lock.resolve_model` /
  `resolve_loras`). LoRAs stack (`--lora FILE[:STRENGTH]` repeatable → chained
  `LoraLoaderModelOnly` nodes, `comfy.chain_loras`; none → the node is bypassed). A choice
  applies to that press; **save as unit default** writes `job.json["defaults"][family]`
  (`PUT /concept/{unit}/defaults`) and later presses without a choice use it.
- **What is registered is not always installed.** `?probe=1` marks each row `installed`
  from ComfyUI's own file lists — the tab always probes, so a missing file is greyed out
  and named before the press (the Qwen pixel LoRA is registered and deliberately *not*
  installed: `models.lock`'s KNOWN DISCREPANCY). The stage checks again before its first
  `/prompt` (`concept.check_installed`) and refuses with the filename and the
  `gpu/download_models.sh` line, because ComfyUI validates every model name against the
  loader's file list and would otherwise answer `400` on the whole graph — after the inputs
  had been uploaded — naming only a node id.
- **A failed run leaves nothing behind.** The attempts it planned and did not render are
  dropped (they carry the settings that failed), so the record never claims an attempt with
  no file and a rerun starts clean.
- **Seed.** Random per press; **lock seed** resends the last one so a prompt tweak changes
  only what you changed.
- **Retouch instead of rerolling — the retouch feature.** Three ways to change an attempt
  *where it sits*, so the strip does not grow. **Edit pixels** opens that one attempt in the
  studio's Piskel canvas and switches to the editor; **Save** writes it back over itself
  (`PUT /concept/{unit}/attempts/{target}/{index}/image`, `--replace REF --image FILE`) with
  any transparency flattened onto white, because a concept image is opaque on white and
  transparency here means *mask*. Rub an area out and press **Inpaint** instead: the canvas as
  you left it is the source — what you painted guides the model — and the hole is the region
  redrawn (`gpu/workflows/inpaint_pass.json`, Z-Image + core `SetLatentNoiseMask`, strength
  `config.CONCEPT_INPAINT_STRENGTH` 0.9; `--inpaint REF --mask FILE`). Everything outside the
  mask is pasted back on the CPU afterwards, so not one pixel you kept can change. And the
  **edit in place** tick in the settings row makes img2img and Edit overwrite the attempt you
  clicked instead of appending (`--in-place`).
  The row keeps its generation record — seed, model, LoRAs, prompt, now as the history of what
  the pixels grew out of — and gains an `edited` block (how and when); `sha256` is recomputed
  and `embedded_prompt` cleared, because a re-save drops ComfyUI's text chunks. A hidden or
  file-deleted attempt is refused. Editing the **accepted** attempt does not touch
  `work/<unit>_base/render/`: the tab flags *"edited since accepted — Accept again to push it
  through"* (the pick records the hash it copied) until you do.
- **Hide what you do not want to look at.** Under a selected attempt, **Hide** takes it out of
  the strip and off the views page (`--hide base:03`); **show hidden** brings the hidden ones
  back, greyed, with **Restore** (`--show`). Once hidden, **Delete file** removes the PNG from
  disk after a confirm (`--delete-file`) — but **the row stays**, with its prompt, seed, model,
  LoRAs and parent. That is not sentiment: an attempt's filename comes from its index and the
  next index is `max + 1`, so a row that left the list would hand its number *and* its filename
  to the next attempt, over a file that may still exist; and anything made from it by img2img
  or an edit still names it as its `parent`. The accepted attempt — the base pick, or a view's
  pick — refuses both and says to pick another first, because every later stage reads it.
- **The GPU is one card.** While a run holds it every generate button waits — a run started
  from the CLI or the Board too (`/health`'s `gpu_run`) — and **Cancel** ends it: `POST
  /runs/{id}/cancel` sends SIGTERM to the stage's process group (SIGKILL after 5 s) and
  ComfyUI `/interrupt`, the run ends `cancelled`, the slot is free. The static page
  `work/_views/<unit>/index.html` is still written — a printout of `views.json`, not the UI.

### The Animate tab — the animate tab feature

The owner directs every frame of an animation by hand, direction by direction, with approval as
the gate, and the approved frames ship to the game (the animate tab feature; it replaced the
Keyframes tab on 2026-09-26). One animation is one (unit, action, direction) cell. The tab is
laid out as the approved mockup, top to bottom:

1. **Direction** — a compass of the unit's eight side-on views, each a thumbnail of its
   accepted view (`s270` is the front: the accepted base itself). A direction whose view is not
   accepted is shown blocked with the reason; it is never drawn from another direction's view.
   Top-down is shown disabled — a later milestone.
2. **Animation** — the five studio actions, each with its frame count at this direction.
3. **Frames** — the row: `ACTION_PHASES[action]["budget"]` frames to start (attack 12), the
   phase keys marked as **keyframes** with gold labels (attack: frames 1 / 3 / 7 / 10 / 12 —
   Start / Wind-up / Action / Wind-down / End; the cycles get theirs from `PhasePlan.keys()`),
   the rest "Frame N". Each frame shows its status — empty, generating, to review, approved,
   rejected — and ↻ / ✓ / ✗. The toolbar: **Generate all** is two steps with approval between
   them — while any keyframe is unapproved it draws the keyframes that are empty or rejected,
   then stops; once every keyframe is approved the same button fills the in-betweens (its label
   says which: "Generate keyframes (5)", "Approve the keyframes to go on", "Fill the in-betweens
   (7)"). Frames waiting for review are never drawn again by it. Also **Generate keyframes
   only** (every unapproved keyframe), **Regenerate rejected**, **Approve all reviewed**, the
   mirror button, and "N / M approved".
4. **The selected frame** — the big picture beside the pose skeleton for this frame, every
   attempt (all kept; click one to make it current), the keyframe tick and its phase, **+ before**
   / **+ after** / **Remove frame** (a removed frame's attempts stay in the record), the prompt
   (pre-filled per phase for a keyframe, "moving from the wind-up toward the action" for an
   in-between), **draw from**, the **pose slider**, the model, and **Generate this frame** /
   **Approve** / **Reject** / **Pixel edit** / **Inpaint**.
5. **Play** — the frames looping at full size and at game size (the pipeline's own matte,
   downscale to the unit's height on the 64 px canvas — `hooded_archer`: 48 px — and palette
   snap, on the CPU, cached), a speed slider (12 fps, the timing template's), and the
   "approved frames only" and "loop" ticks.

```bash
# one keyframe: Wind-up of the attack at the left profile, from the left view + its skeleton
docker compose run --rm anim animate --unit hooded_archer --action attack --dir left \
  --frame wind-up --draw-from view --skeleton
docker compose run --rm anim animate --unit hooded_archer --action attack --dir left --frame 3 --approve
docker compose run --rm anim animate ... --batch all        # keyframes; once approved, the in-betweens
docker compose run --rm anim animate ... --batch keys       # | rejected | inbetweens
docker compose run --rm anim animate ... --approve-reviewed
docker compose run --rm anim animate ... --frame 4 --skeleton-frame 14 --prompt "..."   # an in-between
docker compose run --rm anim animate ... --frame 3 --reject          # --select NN makes NN current
docker compose run --rm anim animate ... --add-frame 5               # after frame 5; 0 = first
docker compose run --rm anim animate ... --frame 6 --key action      # --unkey; --remove-frame
docker compose run --rm anim animate --unit hooded_archer --action attack --dir right --mirror
docker compose run --rm anim animate ... --output [--replace-job]
docker compose run --rm anim animate ... --inpaint f03:01 --mask canvas.png   # --replace f03:01 --image x.png
docker compose run --rm anim animate ... --list          # the row, statuses, attempts
docker compose run --rm anim animate ... --skeletons     # draw the scrub skeletons (CPU)
```

`--dir` takes a cell (`s180`) or its view (`left`); `--frame` a number, an id (`f03`) or a
keyframe's phase (`wind-up`).

- **Draw from.** `view` — the direction's own accepted view. `keys` — the two nearest keyframes
  either side of the frame (a cycle wraps round), which must both be **approved**; otherwise the
  frame is refused with the reason, before any GPU call. `both` — the view and the two keys.
  Keyframes default to the view, in-betweens to the keys; overridable per frame and recorded
  per attempt with the images in the order Qwen-Image-Edit-2509 saw them and their hashes. The
  skeleton, when ticked, is the last image; Qwen takes three, so `both` + skeleton is refused
  rather than silently dropping one.
- **The pose slider** scrubs the rig clip the matrix rigs for the action
  (`config.STUDIO_ACTIONS`, `config.MATRIX_RIG` — the Quaternius UAL's `Sword_Attack` for attack)
  at 24 fps, posed at the direction's yaw, the skeleton re-drawn live (CPU, cached under
  `skeletons/`). It starts on the frame's template instant (`pick.phase_positions`); the instant
  chosen is recorded per attempt and remembered on the frame. The UAL has no whole jump, so the
  jump is drawn without a skeleton.
- **Model.** The installed `qwen-edit` checkpoints from `gpu/models.registry.tsv` (today one).
- **Approval.** Always a person's act. Every attempt is kept; showing another puts the frame
  back to review unless that attempt is the judged one. **Pixel edit** (the Piskel retouch
  path) and **Inpaint** (the retouch feature's masked redraw, `gpu/workflows/inpaint_pass.json`,
  Z-Image) edit the attempt where it sits — and an edit of the approved attempt clears the
  approval: the owner approved other pixels.
- **Mirror** — per animation, "Mirror from <partner>" for the three pairs (right↔left,
  back_right↔back_left, front_left↔front_right): every frame of the partner's cell flipped
  horizontally (its approved attempt, else its current one), no GPU, each landing as a new
  attempt **to review**, never approved, with a handedness warning. The row follows the
  partner's. Never automatic.
- **Output.** When every frame is approved (the approval that completes the row does it; the
  **Write to the game layout** button does it again) the frames go through the post chain into
  the cell's ordinary job, `work/<unit>_<verb>[_<dir>]/` — `job.json` (`motion_source.kind:
  animate`), `post/<unit>_<verb>_<n>.png`, `post/report.json` (palette snapped) and a
  `picks.json` whose timing block carries one tick per frame at 12 fps, each frame's phase and
  the keyframe positions — exactly what the Board and `anim export` read. The Board's cell is
  set accepted. A job there from another route (the video matrix) is refused unless
  `--replace-job` / the confirm moves it to `work/_replaced/` first.
- **Where it lands.** `work/<unit>_animate_<action>_<dir>/job.json` (schema `anim.animate/1`: the
  row, per frame every attempt, `current`, the `verdict`, the `removed` frames, the mirror and
  output blocks), `attempts/<id>_NN.png`, `skeletons/`, `game/`, `masks/`. The manifest says
  `unit` / `studio_action` / `cell`, so the Board never mistakes it for a cell.
- **The gates.** Every GPU line waits behind the base gate (the API's `423`); approvals, the
  row, mirror, output and the skeletons spend no GPU. Generation goes through the run manager —
  one at a time, `409` while busy, the log over SSE, `[n/m]` per frame and the batch `outcome:`
  tally, **Cancel**.

The matrix / video route stays until Animate has shipped one real animation; removing it is
its own work package. The old Keyframes tab's data (`work/hooded_archer_keyframes_attack_s270/`)
is left on disk; nothing reads it.

### The curation board — the curation board feature

Where the human says yes or no ([#281](http://192.168.1.158:8700/work_packages/281)). The
**Board** tab of the editor's panel is one unit's grid — rows the studio's five actions
(attack, idle, walk, run, jump), columns the directions the unit has a job in — and every cell
is that job's frame strip at ×2, nearest-neighbour. A cell is the job `rig` names for it
(`<unit>_<verb>[_<dir>]`: `knight_strike` is `attack/s0`); when two jobs land on one cell the
canonical name wins, then the newest.

```
http://192.168.1.158:8720/   → Board        (from the Mac)
docker compose run --rm anim curate knight                              # the same board, printed
docker compose run --rm anim curate knight --reject-frame attack/s0/5 --accept-cell attack/s0
docker compose run --rm anim curate knight --accept-view back --reject-view left   # the views step (#298)
```

- **Per frame:** click cycles accept → reject → clear. A rejected frame **leaves the job's
  `picks.json`** through the same `ReviewApp.set_picks` the review page uses (byte for byte the
  same file — a test checks), stays on the strip dimmed with a ✗ so you can take it back, and
  `strip` / `post` compose without it the next time they run. Clearing the reject puts it back.
- **Per cell:** accept locks the cell (its frames grey out until you reopen it); reject puts it
  on the "waiting on you" list until it is regenerated; **regenerate** records a new seed in the
  record and starts the cell's GPU stage — `clip` on the clip route, else `render` — through the
  run manager behind the same "this starts a GPU job" confirm as the pipeline buttons, so one
  at a time and a busy card is the `409` the panel follows. The cell goes back to pending with
  its marks cleared; the run shows in the Run section and the board reloads when it ends.
- **The record** is `work/<unit>_curation.json` (`anim.curation/1`): per cell `job`, `status`,
  `frames` (`{"5": "reject"}`), `seed` / `regenerated_from` / `decided_at`, and the two gates.
  One flat file beside the jobs rather than `work/<unit>/curation.json`, because a unit has no
  directory of its own and a `work/<unit>/` would list as a job with no manifest. The API and
  `anim curate` write it through one module, `anim/curation.py`.

**The gates are a project setting** — `curation` in `work/project.json`, the mode selector on
the board, `PUT /settings`, or `anim curate --mode …` (sprite-studio.md §2 "Curation"):

| Mode | What waits for what |
|---|---|
| `gates` (default) | `clip`, `render`, `bakeoff`, `matrix` and `run` for a unit wait until its **base** gate is accepted — the one gate. A matrix direction also needs its **view** accepted (a per-cell check in the stage, not a gate). Review the cells in a batch |
| `every` | the same gates, and a cell's GPU stage waits until **every earlier cell** that has a job is accepted — direction-major: every action of `s0`, then `s45`, … so the side view game-02 ships is finished before a second direction starts. The alternative, action-major (idle in every direction first), matches how a character sheet is drawn but would leave `s0` half-done while the top-down pitch is a later milestone |
| `end` | nothing blocks; the board is a batch review at the end |

CPU stages (`rig`, `pick`, `strip`, `post`) never wait: they make the frames there are to
judge. A blocked `POST /jobs` is **`423 Locked`** with the reason (the panel shows it); the
CLI prints the same sentence and exits 2, and `--force` on `clip`, `render`, `bakeoff`,
`matrix` or `run` runs anyway. **`anim concept --pick` accepts the base gate** (choosing
the base *is* the gate) and clears the views, which were drawn from the old base; the views
are then drawn again and accepted one by one on the Board. A record written before
2026-09-25 may still carry a `turnaround` gate; it is ignored (see "The approval gate").

**Legacy units are not gated.** A unit with no `work/<unit>_curation.json` — the knight and
every job that predates the board — answers "gates accepted" and nothing blocks, so `anim
render knight_strike` behaves as it did yesterday. Only `concept --pick` writes a real gate
record; a board edit on a legacy unit creates its record with the gate accepted and
`"origin": "legacy"`.

### The outcome of a batch stage

`matrix` and `bakeoff` are **batch** stages: they loop over cells, catch each
cell's exception, record it and go on to the next, because one bad cell must not cost the
other seven. Until 2026-09-24 that also meant they exited `0` whatever came of the run — so
a batch run on `hooded_archer`, whose eight directions all raised
`ValueError: no part table for unit 'hooded_archer'`, was recorded `status: done, percent: 100`
in `GET /runs`, in the panel's run line and in the Board. "The process exited cleanly" and
"the work succeeded" are two different facts, and only the first was being carried.

So each of them now ends by **reporting its tally** (`anim/outcome.py`):

```
outcome: done=0 failed=8 blocked=0 of=8
```

| Where | What it says |
|---|---|
| the state file | `summary` — `{done, failed, blocked, of}` over the cells **this** run planned (`selected`), not the whole grid the page draws |
| the **exit code** | `0` only when every planned cell is done · `3` partial, some are · `4` nothing, none are. `blocked` counts as not-done — a run of forty blocked cells asked for forty and got none |
| `GET /runs` | `cells`, the same tally, and the status `partial` beside `done` / `failed` / `lost` / `cancelled` |
| the panel, the Board, the Matrix tab | *"matrix: failed — 0 of 8 cells done, 8 failed"* rather than *"matrix: done"* |
| the bar | rebased at the end onto the cells that are actually done: `0%` for an all-failed run, `50%` for four of eight. The live `[n/m]` bar counts cells **finished**, failures included, so only its last value is corrected |

`--dry-run` and `--page-only` are asked to run no cell, so they report no tally and exit `0`.
A stage with no cells to count (`pick`, `post`, `render`, …) is unchanged: it raises or it
succeeds, `cells` is `null`, and a clean exit is still `done`. Run records written before this
existed load with `cells: null` — they are migrated in memory, never rewritten, because
back-filling a tally the old run never reported would be inventing it.

### The frame route bake-off — the frame route bake-off feature

Before the action matrix renders eighty clips, one question has to be answered on one
action ([#274](http://192.168.1.158:8700/work_packages/274)): where do the in-between frames
come from — and, on the same page, which rig (the ruling open since #261)?

```bash
docker compose run --rm anim bakeoff --unit knight --action attack --parts-from work/gate_parts.json
docker compose run --rm anim bakeoff --unit knight --action attack --cells K-kaykit --force   # redo one cell
docker compose run --rm anim bakeoff --unit knight --action attack --page-only                # rebuild the page
mkdir -p ~/.cache/claude/visuals/sprite-studio/bake-off && \
  cp -r work/bakeoff_knight_attack/page/. ~/.cache/claude/visuals/sprite-studio/bake-off/
```

- **Route V — rig → Wan video.** What `anim clip` does: the rig clip at 12 fps, skeleton over
  full-body depth at gain 0.35 (the second gate's variant (b)), Wan 2.2 Fun-Control 5B against the
  unit's base render, then the clip *downsampled* to the attack's 12-frame budget by an even
  spread of picks (`pick --phases` with `spread="even"` since the keyframe phase feature: the
  same frames, now tagged by phase), `post --input clip --parts`.
- **Route K — keyframe stills → rig-space tween.** The strip route's pose pass (Z-Image-Turbo +
  the Fun ControlNet Union), with the in-betweening done on the rig: the clip is sampled at
  exactly 12 instants (`rig(..., times=)`), so every frame is a still of the rig and "tween" means
  denser sampling rather than pixels invented between two renders. The 12 cells go through the
  pose pass as three 4-cell strips with one seed, at the anchor gate's control strength 1.0 and
  unit prompt (`bakeoff.POSE_PROMPTS`; the workflow's own prompt returns silhouettes) — a 12-cell
  strip breaks the pose pass, measured 2026-09-20 — joined into one `render/pose_pass.png` for
  the page and cut into `clip/frames/` so `post --input clip --parts` mattes each cell on its own
  (`rembg` on a 3648 px strip loses cells). The identity pass runs too and its strip is on the
  page; the frames are the pose pass, as every published strip has been. No RIFE / FILM: no
  interpolation node is pinned and this feature adds no weights. A pixel-space crossfade of the
  five phase stills is on the page as a display-only baseline.
- **The metric table** per cell: frames achieved against the budget; per part, mean pixels,
  drift (pixels the part palette lock moved) and the frames the part was not found in; and two
  proxies for the second gate's eye checks — *one sword in the sword hand* (one blob, nearest the
  right wrist) and *one figure* (one matte blob, at most one helmet) — computed on the HD cells
  from the same SAM 2.1 masks the lock used (`RecordingSegmenter`). The page says they are proxies.
- Cells named `<route>-<rig>`; `bakeoff.json` records seconds per stage (rig / GPU / post), the
  service's VRAM after the GPU stage, picks, sample instants, the chunk jobs and the metric block.
  `report.md` is the same table in markdown plus the four questions the owner has to answer; the
  committed copy is `docs/bake-off-2026-09-20.md`. Page:
  <http://192.168.1.158:8090/sprite-studio/bake-off/>.

### The approval gate — the views, then the keyframes

What a person approves before the card is spent on motion is **pictures of the character**, one
at a time: the base (the `base` gate, `anim concept --pick`), then **each view of the base**,
accepted on the Board (`anim curate <unit> --accept-view back`), and — going forward — **the
keyframes** of each action × direction cell (start, wind-up, action, wind-down, end; see
`CONTEXT.md`), approved per cell with a single-keyframe redo. There is no separate identity
check and no second gate: `base` is the only curation gate, and a direction of the action
matrix is runnable only once the view it wants is accepted (`matrix.prerequisite`). The
frames are drawn on the Animate tab (above): the owner rejected every arm of the keyframe route
bake-off and the Keyframes tab, and directs every frame by hand instead. The bake-off's folder
(`experiments/keyframe-bakeoff/`) is deleted and its three Wan weights' rows are out of the
registry and the lock. Until Animate ships one real animation, the matrix still renders whole
clips as below.

(There used to be a `turnaround` stage here — one idle still per direction through route V as
an identity gate. It was removed on 2026-09-25; `DEVLOG.md` says why. Its old output under
`work/<unit>_idle_<dir>/`, `work/<unit>_turnaround.json` and `work/_turnaround/` is left on
disk; nothing reads it, and a curation record's old `gates.turnaround` key is ignored.)

- **One reference per view** — the multi-view reference feature
  ([#298](http://192.168.1.158:8700/work_packages/298)). Each matrix direction hands `clip` the
  base's view that already shows that side — `config.MATRIX_REFERENCE_BY_DIR`, best first — and
  the view must be **accepted**; there is no fallback to the front:

  | cell | wants | then |
  |---|---|---|
  | s0 | `right` (the right profile) | front |
  | s45 / s135 | `back_right` / `back_left` | back, front |
  | s90 | `back` | front |
  | s180 | `left` | front |
  | s225 / s315 | `front_left` / `front_right` | front |
  | s270 | front (the base itself) | — |

  The turned-away obliques take the **back** (the 2026-09-22 failures were bare heads where the
  reference showed a face; the back view is the one that shows the back of the head), the
  turned-toward ones the **front**. A `top` cell tries the `top` view first, then its yaw's row.
- **One set of part palettes per unit.** Every matrix cell locks its parts to the base's
  measured palettes, `work/<unit>_base/post/parts.json` (`anim post <unit>_base --parts`, or the
  Board's "measure part palettes" button). A unit without them has every cell blocked with that
  reason, before any GPU call.
- **No face off-front.** A rig has no detector, so every joint carried confidence 1.0 in every
  cell — nose and eyes included on the back view. `rigkit.hidden_face_points` drops **every face
  point** (nose, both eyes, both ears; confidence 0.0, the frame's `hidden` list) once the *posed
  head* has turned past the profile away from the camera by more than half a direction step
  (`FACE_HIDE_TURN`, sin 22.5°, so a breathing head at s0 never flickers). s0 / s180 and s225 /
  s270 / s315 are byte-for-byte what they were; s45 / s90 / s135 lose the whole face — an ear is
  a landmark the pose ControlNet builds a face around.

## Rig input mode — the rig motion input feature

The other way in. Instead of filming, the motion comes from a **rigged, animated glTF
character** whose licence is CC0 — first the KayKit Adventurers Knight, vendored with its
licence under [`rigs/`](rigs/README.md). One stage replaces `extract` + `pose`; everything
after it is untouched and cannot tell the difference:

```bash
docker compose run --rm anim rig rigs/kaykit-adventurers/Knight.glb --list       # 76 clips
docker compose run --rm anim rig rigs/kaykit-adventurers/Knight.glb \
    --clip 1H_Melee_Attack_Chop --unit knight --action strike                    # -> work/knight_strike
docker compose run --rm anim pick knight_strike
docker compose run --rm anim strip knight_strike
# ... anim render knight_strike when the GPU is up, exactly as for a filmed job

# the standing sprite from the same rig, same facing, same scale:
docker compose run --rm anim base --unit knight \
    --rig rigs/kaykit-adventurers/Knight.glb --clip Idle --frame 0 --skeleton-only

# the same clip seen from the front, and from straight above facing south:
docker compose run --rm anim rig rigs/kaykit-adventurers/Knight.glb \
    --clip 1H_Melee_Attack_Chop --unit knight --action strike --yaw 270   # -> work/knight_strike_s270
docker compose run --rm anim rig rigs/kaykit-adventurers/Knight.glb \
    --clip 1H_Melee_Attack_Chop --unit knight --action strike \
    --yaw 270 --elevation top                                            # -> work/knight_strike_t270
```

What `anim rig` writes is byte-for-byte the layout `extract` + `pose` leave behind
(`job.json`, `frames/`, `pose/keypoints.json`, `pose/skel_*.png`), so `pick`, `review` and
`strip` run unchanged — plus one directory a filmed clip never gets, `depth/`, which only
`anim clip` reads. Six things are worth knowing:

- **Facing is detected, not assumed.** glTF says nothing about which way a character faces.
  The stage reads the rest pose's foot→toes vector and projects that direction onto +x, so the
  figure faces **right** (art bible §1: the party faces right; enemies are `flip_h`).
  `--mirror` flips the projection (the face points stay on the character's real front). The
  Knight faces +Z; `job.json.motion_source.forward` records what was found.
- **One scale per rig, never per clip.** The rest-pose mesh height maps to 78 % of the cell,
  with the floor on a fixed row (300 of 320). `hit` and `death` therefore come out the same size
  as `strike`, and a body on the floor lies on the row the standing feet stood on. Each frame is
  centred horizontally on its own extent, so a lunge or a lying chibi head cannot leave the cell
  (rtmlib drops any limb that touches the edge).
- **Provenance rides along.** `rigs/<pack>/rig.json` (pack, author, url, licence, sha256) is
  checked against the GLB and copied into `job.json.motion_source`; `anim publish` copies it
  into the sidecar's `generation.motion_source`, so the audit sees the motion's licence next to
  the pixels' seed and model hashes.
- **The unit's kit is drawn on (the skeleton props feature).** A stick figure does not say which
  hand holds the sword, and the pose ControlNet guessed — two swords on the recovery frame of
  `Sword_Attack`, and no prompt fixed it. So a knight's skeleton carries a blade off the right
  hand and a kite on the left forearm, computed in 3D on the rig's bones (palms-down rest
  convention: blade forward out of the fist, shield on the back of the forearm), projected with
  the joints and stroked in **white** — a colour DWPose never uses, read like a canny edge.
  `config.UNIT_PROPS` says who carries what; `--props sword,shield` / `--props none` overrides;
  `keypoints.json` records the outlines per frame and `job.json.motion_source.props` what was
  drawn. Filmed clips have no props and render exactly as before.
- **The posed mesh is rendered as depth (the rig props feature).** An outline tells the pose
  ControlNet which stick is the sword arm; the clip route's video model is conditioned on depth
  and needs the same fact as a *surface*. So the rig's skinned mesh is posed per frame in numpy
  (linear-blend skinning from the GLB's own joints, weights and inverse bind matrices — no
  Blender), the props are extruded into thin solids on the same bones, and the lot is z-buffered
  under the very same projection as the joints, so `depth/%04d.png` and `pose/skel_%04d.png`
  align pixel for pixel. The viewer stands on the character's **right**, so the sword hand is
  the near hand and the shield the far one. The Quaternius mannequin is one skinned mesh with
  no props (parsed 2026-09-16), which is why the props are added rather than read; the KayKit
  Knight's static helmet and its four shields and three swords sitting on the hand slots are
  skipped, and only its skinned body renders. `--depth props` draws the prop solids alone —
  still occluded by the body, so a kite strapped behind the torso stays hidden — for a video
  model that paints the mannequin when it is shown the body (the clip render feature found
  exactly that); `--depth none` (alias `--no-depth`) skips the whole thing.

  **The depth contract** — what `anim clip` reads, and what nothing else may change:

  | | |
  |---|---|
  | file | `work/<job>/depth/%04d.png`, one per sampled frame, the same index as `frames/%04d.png` and `pose/skel_%04d.png`, the same cell |
  | format | 8-bit grayscale (`L`) |
  | values | **near = 255, far = 48** (`rigkit.DEPTH_FAR_GREY`, never 0), **background = 0** — the MiDaS / ControlNet-depth convention |
  | range | one per **clip**, not per frame: `job.json.motion_source.depth_range = [far, near]` in rig units along the viewing axis, measured over every frame, so a still body does not flicker and a thrust toward the camera brightens |
  | mode | `full` (body + props, default) or `props` (prop solids only, body-occluded). One grey mapping for both — the range is read off body + props either way — so a `props` frame is the `full` frame with the body pixels blanked |
  | manifest | `motion_source.depth` (bool), `depth_mode` (`full` / `props` / `none`), `depth_range`, `depth_far_grey` |
  | optional | a bare animation library, a filmed clip, `--depth none`, or `--depth props` on a unit with no kit writes none; `job.depths()` is then `[]` |

  `anim base --rig` writes the same one frame as `depth/0001.png` at the portrait cell.

- **The rig can be looked at from sixteen places (the camera direction feature).** The rig
  route is the only motion source that gives direction control, so `--yaw` and `--elevation`
  on `anim rig` and `anim base` rotate the posed joints, the skinned mesh and the props into
  the view *before* the projection above — skeleton PNGs, keypoints and depth frames come out
  for every cell from the same code, no new model, no new stage. The table is
  `config.DIRECTIONS`: two pitches × eight facings at 45°, named by the pitch's initial and
  the yaw. Yaw is the figure's turn about its own up axis, counter-clockwise from above, so
  the compass reads the same in both pitches and matches a top-down game's eight directions:

  | yaw | faces | `side` shows | `top`: faces |
  |---|---|---|---|
  | `0` | screen right | the right profile — **today's output, byte for byte** | east |
  | `45` | away-right | back-right three-quarter | north-east |
  | `90` | away | the back | north (up-screen) |
  | `135` | away-left | back-left three-quarter | north-west |
  | `180` | screen left | the left profile — what `--mirror` gives | west |
  | `225` | toward-left | front-left three-quarter | south-west |
  | `270` | the viewer | the front | south (down-screen) |
  | `315` | toward-right | front-right three-quarter | south-east |

  `top` looks straight down with the figure's yaw-0 left side up-screen (a knight facing east
  has its shield arm to the north); the depth axis is then world up, so the head is the
  nearest surface. **Pivot and scale.** The scale is the rig's one scale in every cell. In
  `side` the floor row is absolute and each frame is centred horizontally on its own extent,
  exactly as before; in `top` both screen axes are ground axes, so both get the ground rule —
  centred on the joints' extent, pivot at the cell's centre — which is what keeps a lunge
  from translating at t90 but not at t0, and keeps a body lying full-length inside the cell
  at every yaw. (`anim post` trims and bottom-centres the pixels afterwards whichever rule
  placed them.) The default `s0` keeps its job name; any other cell suffixes it
  (`knight_strike_t270`) and is recorded in `job.json.motion_source` as `direction`, `yaw`,
  `elevation`, next to `mirror` and the rig's true `forward`. `--mirror` is the s0 profile
  reflected and predates the table; `--yaw 180` is the same picture, so the two are refused
  together. Only a rig can be turned: `anim base` without `--rig` refuses a direction. The
  OpenPose face points are hidden once the head faces away from the camera ("No face
  off-front" under "The approval gate" above): s45 / s90 / s135 draw no face points; the
  profile and the front cells draw what they always did.

**The manual test for the camera direction feature (#288) is one command.**
`docker compose run --rm anim-pipeline python -m anim.tools.direction_sheet knight Idle`
renders frame 0 of a clip in all sixteen cells plus the `--mirror` comparison and writes one
contact-sheet page — see [`anim/tools/direction_sheet.py`](anim/tools/direction_sheet.py) for
the five pass criteria and where the page lands.

The skeleton is a hint about limb angles only (art bible §3). The Knight is chibi — its head is
~45 % of its height, so the neck→nose limb is as long as torso + legs — and whether the pose
ControlNet honours that is the anchor-gate question; `rigs/quaternius/` holds a realistic-
proportion rig (Quaternius UAL, also CC0 by its zip's licence file) for the A/B.

### KayKit clip → action map

The art bible's vocabulary (§8) against the clips in `Knight.glb`. Staged for the anchor unit on
2026-09-14 are the first six rows; the rest are what to reach for next.

| Action | Clip | Why this one |
|---|---|---|
| *base sprite* | `Idle`, frame 0 (via `anim base --rig`) | the rest pose every other strip starts from |
| `strike` | **`1H_Melee_Attack_Chop`** | swings in the sagittal plane — overhead windup, down-and-forward cut, torso lean — which a side view shows in full. `_Slice_Horizontal` sweeps across the body and foreshortens to an arm wobble in profile; `_Stab` is a candidate for the pikeman. |
| `hit` | `Hit_A` | a backward flinch, 0.67 s; `Hit_B` is the alternative |
| `death` | `Death_A` | falls backward onto the floor, 0.8 s; `Death_B` is the 2.6 s version |
| `collapse` | `Lie_Down` | the downed-not-dead reaction |
| `revive` | `Lie_StandUp` | its reverse |
| `idle` | `Idle` (2 poses: breath extreme + loop point) | |
| `walk` | `Walking_A` (also `_B`, `_C`, `Running_A/B`) | |
| `shoot` | `1H_Ranged_Shoot` / `2H_Ranged_Shoot` (crossbow) | |
| `cast`, `bless`, `invoke` | `Spellcast_Shoot` / `Spellcast_Raise` / `Spellcast_Long` | |
| `shove` | `Unarmed_Melee_Attack_Punch_A` or `Block_Attack` | |
| `throw` | `Throw` | |
| `reach`, `pickup` | `Use_Item` / `PickUp` | |
| `dash` | `Dodge_Forward` | |
| `guard_set` | `Block` | |

The KayKit Skeletons pack shares this rig exactly (plus `Death_C_Skeletons_Resurrect` for the
barrow-dead's `revive`), so `wight` and friends need no new map; `Mage`, `Rogue`, `Barbarian`
cover the casters and the light units.

### Adding another rig

Put the GLB and its licence under `rigs/<pack>/` with a `NOTICE.md` and a `rig.json`
([`rigs/README.md`](rigs/README.md)). If the bone names are a known family — KayKit, or the
Unreal-mannequin names Quaternius ships — nothing else is needed. Otherwise add one `JointMap`
row to `RIGS` in [`anim/rigkit.py`](anim/rigkit.py): thirteen bone names for the body joints
(the neck is always the shoulder midpoint, because that is how rtmlib defines OpenPose's neck),
the head bone, the foot→toes pairs for facing, five face offsets in head lengths, and the bones
that identify the family. The module docstring is the full map for KayKit; the test suite
checks every row covers all 18 joints.

## Config values

Every path and number is an environment variable with a container default; no
stage hardcodes one. They all live in [`anim/config.py`](anim/config.py), and
`compose.yml` sets the ones that differ per machine.

| Variable | Default | What it is |
|---|---|---|
| `COMFYUI_URL` | `http://host.docker.internal:8188` | the GPU image service. From the host or the Mac: `http://192.168.1.158:8188` |
| `ANIM_CAPTURE_DIR` | `/work/capture` | the watched drop folder for filmed clips |
| `ANIM_WORK_DIR` | `/work/work` | per-job scratch; everything regenerable |
| `ANIM_WORKFLOW_DIR` | `/work/gpu/workflows` | the API-format workflows `render`, `clip`, `base` and `concept` POST |
| `ANIM_MODELS_LOCK` | `/work/gpu/models.lock` | repo + revision + sha256 per weight, copied into the sidecar |
| `ANIM_MODELS_REGISTRY` | `/work/gpu/models.registry.tsv` | which locked weight is a checkpoint / LoRA / … and for which family; the model picker's list (the concept generation feature) |
| `ANIM_ASSETS_DIR` | `/assets` | game-02's published assets (read-write) |
| `ANIM_GAME02_DIR` | `/game02` | the game-02 repo, read-only, so `publish` can run its audit here |
| **`ANIM_PALETTE`** | `/assets_src/palette.gpl` | the `-remap` source of truth — the project palette (Lospec 2000, 182 colours, installed 2026-09-14). `anim palette` rewrites it and its `.hex` / `.png` / `.json`; `palette_units/` beside it holds the unit sub-palettes. See "The project palette" |
| `ANIM_LOSPEC_URL` | `https://lospec.com/palette-list/{slug}.json` | where `anim palette --lospec` fetches from; the tests point it at a `file://` directory |
| `ANIM_EXTRACT_FPS` | `12` | the decode-rate cap (and the rate the game plays frames back at) |
| `ANIM_POSE_MODEL` | `wholebody` | `wholebody` is DWPose (134 joints); `body` is the lighter 18-joint rig |
| `ANIM_POSE_MODE` | `balanced` | `lightweight` \| `balanced` \| `performance` |
| `ANIM_POSE_KPT_THR` | `0.4` | below this a joint counts as absent, for the picker and the renderer |
| `ANIM_PICK_DEDUP` | `0.12` | minimum distance between two picks, **in torso lengths** |
| `ANIM_TIMING` | `/assets_src/timing.json` | the project's phase-template overrides (the keyframe phase feature); absent means the defaults |
| `ANIM_LOOP_TOL` | `0.12` | a loop's last pick must land within this of its first, in torso lengths |
| `ANIM_LOOP_PIXEL_TOL` | `0.08` | the same check as a mean pixel difference (0–1) when a pose is missing |
| `ANIM_API_PORT` | `8720` | the pipeline API's LAN port; keep in step with the published port in `compose.yml` |
| `ANIM_API_BIND` | `0.0.0.0` | the Mac has to reach it |
| `ANIM_API_CPU_SLOTS` | `2` | CPU stages the API runs at once; GPU stages are always one at a time |
| `ANIM_WEB_DIST` | `/work/web/dist` | the built editor, served at `/` when it exists |
| `ANIM_LAN_HOST` | `192.168.1.158` | only ever used to print a clickable URL |

Four tables in `config.py` are code rather than environment, because they are
design decisions and belong in review: `ACTION_FRAMES` (the art bible's key-pose
budget per action — strike 4, hit 2, walk 6, …), `ACTION_TIMING` (the hold
weights per `(action, n)` — the strip timing feature), `ACTION_PHASES` +
`STUDIO_ACTIONS` (the studio's phase template: budget, loop, split and span per
action, and each action's pipeline verb and rig clips — the keyframe phase
feature; `timing.json` overrides it per project) and `UNIT_HEIGHT` (the scale
ladder — `hound` 37 px, `knight` 53 px, `barrow_warden` 63 px).

## Manual steps — the one thing the pipeline cannot do for itself

**1. Somebody starts the GPU service.** Since 2026-09-25 this is rootless
Podman, not the native Docker daemon, so it needs no sudo and no TTY — an agent
or a script can do it:

```bash
podman-compose -f tools/anim-pipeline/gpu/compose.yml up -d   # GPU container ONLY
curl http://192.168.1.158:8188/system_stats                   # should name the A5000
```

Never `docker context use`, and never start `docker.service`. Everything else in
this repo stays on Docker Desktop.

**2. The owner picks the palette.** `assets_src/palette.gpl` is Lospec 2000 (182 colours,
installed 2026-09-14 by `game-02-tactical-rpg/assets_src/install_palette.py`, which the file's
own header says). Changing it is `anim palette --lospec <slug>` or `--file <path>` from this
repo (or `install_palette.py` from game-02's — both write the same `.gpl`/`.hex`/`.png` set in
place, and `anim palette` adds `palette.json`); frames do not re-snap themselves, so re-run
`anim post` + `anim publish` afterwards. To try a staged candidate without installing it,
point the pipeline at it:

```bash
ANIM_PALETTE=/assets_src/palette_candidates/grim32/grim32.gpl \
  docker compose run --rm anim post knight_strike
```

A `.gpl` or a one-row `.png` are both accepted; the `.gpl` stays the
human-editable source and the PNG `magick -remap` needs is derived from it.

## Tests

```bash
docker compose run --rm anim-pipeline pytest        # 728 passed, 4 skipped (2026-09-26)
cd web && npm test                                  # 129 passed - the editor panel's pure parts
```

No GPU, no model download, no network. The GPU is replaced by a threaded fake
ComfyUI (`tests/fixtures/fakecomfy.py`) that returns canned `/system_stats`,
`/prompt`, `/history` and `/view` responses; the clips and the "generated"
strips are drawn with Pillow; the sidecar is judged by game-02's **real**
`asset_audit.py`, copied into a temp tree. The rig tests (`test_rigkit.py`,
`test_rig.py`) use a two-bone GLB built in the test for the maths and the
vendored CC0 `Knight.glb` for the end-to-end run.

The skipped tests are opt-in. One needs the network on first run — it proves
rtmlib actually finds a person in a real photograph; one runs SAM 2.1 for real on
the knight base (downloads 184 MB once); two re-run `post` on the real
`knight_hit` / `knight_death` jobs and compare bytes with the published frames:

```bash
docker compose run --rm -e ANIM_POSE_SMOKE=1 anim-pipeline pytest tests/test_pose.py -k smoke
docker compose run --rm -e ANIM_SAM2_SMOKE=1 anim-pipeline pytest tests/test_partkit.py -k sam2
docker compose run --rm -e ANIM_POST_REGRESSION=1 anim-pipeline pytest tests/test_parts_post.py -k reproduces
```

Fixture sources and licences: [`tests/fixtures/FIXTURES_NOTICE.md`](tests/fixtures/FIXTURES_NOTICE.md).

## Stage mapping — what happened to W1…W5

Upstream's stages are **kept in the repo**, not deleted. Parked stages remain valid and
documented; they are the starting point for a later full-cycle animation tier.

| Upstream | Status here | Where it went |
|---|---|---|
| **W1** — Qwen-Image-Edit 2509 + ControlNet pose edit | **Adopted, rebuilt** | The direct basis for **`gpu/workflows/identity_pass.json`**. Kept: the two-`TextEncodeQwenImageEditPlus` (positive + empty negative) shape, `ModelSamplingAuraFlow` shift 3, `CFGNorm`, `VAEEncode`-as-latent, 4-step lightning at cfg 1.0, and the driver's match-nodes-by-title rule. Changed: GGUF loaders → `NunchakuQwenImageDiTLoader` (INT4 SVDQuant — the A5000 is Ampere and has no fp8 tensor cores); InstantX ControlNet-Union dropped, because pose now arrives as `image2`/`image3` on a strip that a **separate** Z-Image pass has already posed; 512×512 → a non-square 4×1 strip. |
| **W2** — Wan 2.2 i2v video generation | **Parked** | `workflows/w2-video-gen/`, `docs/workflow-2-video-gen.md`, `drivers/run_w2_video_gen.py` all kept. Weights **not downloaded** (~25 GB). This is the later "full cycles" tier; Wan-Animate-2 (Apache-2.0, in ComfyUI core) is the model to revisit it with. |
| **W3** — BiRefNet → sprite sheet | **Superseded, feeds the design** | Its strip-cell slicing and greyscale-RGBA sheet logic are the model for the CPU **post-chain** (`anim post`, `anim strip`). The BiRefNet node itself is replaced by `rembg -m isnet-anime -dc -ppm` on the CPU, because we need **binary** alpha (∈ {0,255}) for 64px sprites and the ComfyUI node exposes no threshold. `ComfyUI-RMBG` is therefore not installed. |
| **W4** — Wan 2.1 VACE cosmetic inpaint | **Parked** | Kept in-repo. No cosmetic layer system in game-02: units are one sprite. Weights not downloaded. |
| **W5** — SAM3 cosmetic sprite sheet | **Parked** | Same. Weights not downloaded. |
| Iris green-key script | **Parked** | Kept; no per-unit eye colour in game-02. |

**New in this fork, with no upstream equivalent:**

| Path | What it is |
|---|---|
| `gpu/` | **The GPU image service feature** — ComfyUI in a pinned container on the native Docker daemon, the only GPU service in the project. Compose file, `models.lock`, pinned custom nodes, the four workflows. Start here: [`gpu/README.md`](gpu/README.md). |
| `gpu/workflows/pose_pass.json` | Stage 1: Z-Image-Turbo (int8) + Z-Image Fun ControlNet Union 2.1 (full 2602 8-step; lite was ignoring the pose). Skeleton strip → posed strip, drawn at 2 px. No upstream counterpart; upstream had no pose-only stage. |
| `gpu/workflows/identity_pass.json` | Stage 2: the reworked W1. Re-skins the posed strip onto the unit's reference sprite. Two images in (reference, pose pass). **Off since the anchor gate** — it flattens poses; `post` takes the pose pass. |
| `gpu/workflows/base_sprite.json` | Generates a unit's standing sprite from a canned skeleton when no reference sprite exists yet. |
| `gpu/workflows/view_pass.json` | The multi-view reference feature: Qwen-Image-Edit 2509 (the identity pass's INT4 build) asked for the accepted base from another side — one image in, one out; `anim concept --views`. |
| `gpu/workflows/clip_render.json` | The clip render feature: Wan 2.2 Fun-Control 5B, reference image + control video → one PNG per frame (`SaveImage` on the decoded batch instead of the template's `SaveVideo`). Translated from the official `video_wan2_2_5B_fun_control` template. |
| `gpu/workflows/NOTES.md` | Which node names and input keys were verified against source, and what is still `TODO-verify`. |
| `anim/` + `Dockerfile` + `compose.yml` | The CPU pipeline on Docker Desktop: `anim extract pose rig pick review strip render post publish export base concept animate bakeoff matrix palette curate run`. |
| `anim/stages/export.py` | **The spritesheet export feature** — a unit's accepted frames packed into sheets + `atlas.json` + a Godot `SpriteFrames` `.tres` (read back and checked before the stage returns); `publish` is its game-02 adapter. |
| `anim/palettekit.py` + `anim/stages/palette.py` | **The palette feature** — the project palette from a Lospec slug or a `.hex` / `.gpl` file, the per-unit sub-palettes in `palette_units/`, and the one snap in `post`. |
| `anim/api/` | **The pipeline API feature** — FastAPI over `work/` on `:8720` (`anim-api` in `compose.yml`): jobs, frames, picks, runs with SSE progress, project settings; serves the review page and, once built, the editor. Deps in `requirements-api.txt`, their own Docker layer. |
| `anim/rigkit.py` + `anim/stages/rig.py` + `rigs/` | **The rig motion input feature** — a CC0 rigged character's clips as the motion source, sampled with `pygltflib` + numpy, mapped to OpenPose-18, projected in profile facing right. Replaces filming for the anchor unit; makes the spike's TripoSG / Mixamo path unnecessary. |

## Configuration — one knob for the ComfyUI URL

Upstream hardcoded `http://127.0.0.1:8000` in all five drivers and the docs. In this
fork the base URL is a config value read from the environment, everywhere:

```python
COMFY_URL = os.environ.get("COMFYUI_URL", "http://host.docker.internal:8188").rstrip("/")
```

| | Value |
|---|---|
| Env var | `COMFYUI_URL` |
| Default | `http://host.docker.internal:8188` |
| From the Fedora host or the Mac | `http://192.168.1.158:8188` |

The default is `host.docker.internal` because the CPU pipeline runs in a **Docker Desktop**
container while ComfyUI runs on the **native daemon** — two different engines on the same
host. That is also why `gpu/compose.yml` publishes `0.0.0.0:8188` and not `127.0.0.1:8188`:
a loopback bind would be invisible from the Docker Desktop VM.

Do not reintroduce a hardcoded URL. `grep -rn '127.0.0.1:8000' .` should stay empty.

## Two engines — the rule that matters most

| Engine | How | GPU | Runs |
|---|---|---|---|
| Docker Desktop (**default**) | context `desktop-linux` | no | the CPU pipeline, the model downloader, the Mac share |
| Native `docker.service` | context `docker-default` | yes | **nothing — off-limits on this host** |
| **Rootless Podman** | `podman-compose` | **yes**, via CDI | **only** `gpu/compose.yml` |

- Never run `docker context use`, and never start `docker.service`.
- `podman ps` and `docker ps` show different worlds. The GPU container is only
  in the first one.
- Podman gets the GPU through CDI (`/var/run/cdi/nvidia.yaml`, kept current by
  `nvidia-cdi-refresh.service`), which is why it needs no daemon and no sudo.
- Everything else stays on Docker Desktop. Nothing runs natively on the host — the host
  has Python 3.14 with no pip.

## Reproducibility

A frame must be re-creatable from its sidecar. Pinned, in order of how easily they drift:

1. Container image by **digest** (`gpu/compose.yml`).
2. Custom nodes by **commit** (`gpu/custom_nodes.txt`).
3. Model weights by **repo + revision + sha256** (`gpu/models.lock`, cross-checked
   against two HF endpoints by `gpu/relock_models.py`).
4. `batch_size: 1` and `sampler_name: euler` — never an ancestral sampler.
5. Seed, workflow hash, model SHA256s and clip hash recorded in the per-action sidecar.

## Licensing rules

Upstream is MIT and stays MIT; `LICENSE` is unmodified and `NOTICE` credits it.

game-02 ships commercially, so **every** asset and weight must be CC0 / Apache-2.0 / MIT.
`gpu/relock_models.py` refuses to lock a gated or non-commercial repo. Two things are
permanently excluded and must never be added:

- `comfy_pixelization` — non-commercial licence
- `Astropulse/sd-palettize` — no licence at all

Palette snapping is done deterministically on the CPU with ImageMagick
(`-dither None -remap palette.png`) instead.

---

# Upstream README


A set of ComfyUI workflows that turn a **single character image** into a full library of layered 2D animation sprite sheets — base motion plus separately-tintable cosmetic layers (hair, eyes, clothing, …) — pixel-aligned for runtime composition.

Designed for **24 GB VRAM** (e.g. RTX 3090). All workflows are tuned to peak at ~22.5 GB so the OS / browser / display compositor have headroom.

> **This repo is harness-driven** It's design to be run fully automatically by an AI agent such as Claude code or Cursor. The workflows are shipped in **API format** (the JSON ComfyUI's `/prompt` endpoint accepts) — they are **not** visual / UI workflows and will not appear as a laid-out node graph if you drag them into ComfyUI's editor.
>
> If you want to inspect or edit the graph visually, modern ComfyUI versions can auto-arrange nodes from API JSON, but layouts will be ad-hoc — these workflows were authored and maintained as API JSON, not in the UI.
>
> See [`docs/harness-integration.md`](docs/harness-integration.md) for the drive protocol and minimal Python / Node examples.

![pipeline overview](samples/base_animations/idle/idle.png)

 Idle base animation produced by W2 → W3 from the seed input at [`samples/inputs/source_idle_breathing.png`](samples/inputs/source_idle_breathing.png). Greyscale RGBA so the runtime can tint pixels at draw time. See [`samples/`](samples/) for the full sprite-sheet set this pipeline produced.

## What this pipeline produces

- **Base animations** — greyscale RGBA sprite sheets for idle, walking, jumping, …. Each sheet is a horizontal strip of equal-sized cells. Frame count and cell size are configurable per animation.
- **Cosmetic layers** — RGBA sprite sheets for hair, eyes, clothing, etc. Each cosmetic layer is pixel-identical in layout to the base animation it pairs with, so a runtime stacks `cosmetic.png` over `base.png` without re-alignment.
- **Eye / iris pairs** — the iris ships as a separate near-white greyscale sheet so the runtime can tint it to the player's chosen eye colour without affecting sclera, lashes, or specular highlights.

## Two pipelines, five workflows

```
                                 ┌──────────────────────────────┐
                                 │ Pipeline 1: BASE ANIMATIONS  │
                                 └──────────────────────────────┘
   character image  ─►  [W1 helper: pose edit]  ─►  posed keyframe
        │
        └──────────────────────────────────────────►  [W2: video gen]  ─►  base.mp4
                                                            │
                                                            └─►  [W3: spritesheet]  ─►  base sprite sheet  ✓


                                 ┌──────────────────────────────┐
                                 │ Pipeline 2: COSMETIC LAYERS  │
                                 └──────────────────────────────┘
   base.mp4
   cosmetic reference image  ─►  [W4: cosmetic inpaint, two-pass]  ─►  cosmetic.mp4
                                                                              │
                                                                              └─►  [W5: cosmetic spritesheet]  ─►  cosmetic sprite sheet  ✓
                                                                                                                  (+ iris green-key script for eye cosmetics)
```

| Workflow | Model | Purpose |
|---|---|---|
| [W1 — Pose Edit](docs/workflow-1-pose-edit.md) (helper) | Qwen-Image-Edit 2509 + InstantX ControlNet-Union | Pose a character from an OpenPose skeleton. Used to produce the seed image for W2 or a one-off posed keyframe. |
| [W2 — Video Generation](docs/workflow-2-video-gen.md) | WAN 2.2 i2v 14B (Q5_K_M GGUF) + Lightx2v 4-step distill | Image-to-video: extrapolate motion from a single frame. |
| [W3 — Sprite Sheet](docs/workflow-3-spritesheet.md) | BiRefNet (background removal) | Horizontal greyscale RGBA sprite sheet from the W2 mp4. |
| [W4 — Cosmetic Inpaint](docs/workflow-4-cosmetic-inpaint.md) | Wan 2.1 VACE 14B + CausVid V2 LoRA + SAM3 (mask) | Two-pass inpainting: paint the cosmetic onto the base motion while preserving non-cosmetic pixels bit-exact. |
| [W5 — Cosmetic Sprite Sheet](docs/workflow-5-cosmetic-spritesheet.md) | SAM3 (segmentation) | Per-frame text-prompted segmentation → cosmetic-only RGBA sprite sheet, pixel-aligned with the base sheet. |
| [Iris Green-Key](docs/iris-greenkey.md) (script) | — | Post-processing script for eye cosmetics: extracts a tintable iris-only sheet from the W4 pass-1 mp4 via colour-keying. |

For the full pipeline overview and design rationale, see **[`docs/pipeline-overview.md`](docs/pipeline-overview.md)**.

## Prerequisites

### Hardware

- **NVIDIA GPU with ≥ 24 GB VRAM** (tuned on RTX 3090). All workflows aim for ~22.5 GB peak.
- **≥ 32 GB system RAM**.
- **~120 GB free disk** for model weights (most are GGUF Q5_K_M variants).

### Software

1. **ComfyUI** — install per the [official ComfyUI README](https://github.com/comfyanonymous/ComfyUI). Launch with `--reserve-vram 1` so ~1 GB stays unallocated for ComfyUI overhead.

2. **Custom node packs** — install per [`docs/custom-nodes.md`](docs/custom-nodes.md):
   - `comfyui_vram_helpers` (this repo, [`custom_nodes/`](custom_nodes/comfyui_vram_helpers/)) — two-stage VRAM eviction nodes
   - `ComfyUI-GGUF` — GGUF model loaders
   - `comfyui_controlnet_aux` — ControlNet utilities
   - `ComfyUI-RMBG` — BiRefNet + SAM3 segmentation
   - `ComfyUI-Manager` (optional but recommended) — auto-installs missing nodes

3. **Models** — download per [`docs/models.md`](docs/models.md). Total ≈ 80 GB for the full pipeline (Q5_K_M variants of WAN 2.2, WAN 2.1 VACE, Qwen-Image-Edit; SAM3, BiRefNet, several VAEs and ControlNets).

4. **ComfyUI MCP** — required if you're driving the pipeline from an LLM coding agent (Claude Code, Cursor, etc.). This repo was developed and validated against **[`comfyui-mcp` by artokun](https://github.com/artokun/comfyui-mcp)** — install with:

   ```bash
   # Add to your MCP config (Claude Code: ~/.claude.json under "mcpServers")
   {
     "comfyui": {
       "type": "stdio",
       "command": "npx",
       "args": ["-y", "comfyui-mcp"],
       "env": {}
     }
   }
   ```

   That MCP exposes `enqueue_workflow`, `get_job_status`, `get_history`, `modify_workflow`, `start_comfyui` / `stop_comfyui`, `clear_vram`, `download_model`, and ~30 other tools that map directly to the operations these drivers perform. **Use the same MCP we use** — different ComfyUI MCPs expose different tool surfaces, and the prompt examples in [`docs/harness-integration.md`](docs/harness-integration.md) assume artokun's tool names.

## Quick start

1. **Install** ComfyUI, ComfyUI MCP, the custom nodes, and the models per the prerequisites above.

2. **Drop a character image** into ComfyUI's `input/` folder as `source_<animation>.png`. A working sample is at [`samples/inputs/source_idle_breathing.png`](samples/inputs/source_idle_breathing.png) — copy that to `<ComfyUI>/input/` to validate your install end-to-end.

3. **Launch ComfyUI** (`python main.py --reserve-vram 1`).

4. **Run W2 → W3** for a base animation, then **W4 → W5** for each cosmetic layer:

   ```bash
   python drivers/run_w2_video_gen.py        # produces base.mp4
   python drivers/run_w3_spritesheet.py      # produces base sprite sheet

   python drivers/run_w4_cosmetic_inpaint.py # produces cosmetic.mp4
   python drivers/run_w5_cosmetic_spritesheet.py  # produces cosmetic sprite sheet
   ```

   Each driver has a small block of per-run knobs at the top (animation name, seed, etc.) — edit and re-run.

5. **Compare against the samples** at [`samples/base_animations/`](samples/base_animations/) and [`samples/layers/`](samples/layers/). If your output looks close, your install is healthy.

## Repo layout

| Directory | Contents |
|---|---|
| [`workflows/`](workflows/) | One folder per workflow. Each contains the API-format JSON (what ComfyUI's `/prompt` endpoint accepts) plus a short README pointing at the full doc. |
| [`drivers/`](drivers/) | Python scripts that drive the workflows: edit knobs at the top, run the script, it POSTs the workflow to `$COMFYUI_URL/prompt` and waits for completion. Single-file, no dependencies beyond the standard library. |
| [`custom_nodes/comfyui_vram_helpers/`](custom_nodes/comfyui_vram_helpers/) | Two ComfyUI nodes (`VRAMUnloadModel`, `VRAMUnloadClip`) for targeted VRAM eviction in multi-stage workflows. Drop into `ComfyUI/custom_nodes/`. |
| [`docs/`](docs/) | Pipeline overview, per-workflow design docs, model and custom-node install guides, harness-integration guide. |
| [`samples/`](samples/) | Working seed input + the full sprite-sheet output the pipeline produced — useful for sanity-checking your install. |
| [`tools/`](tools/) | Standalone helpers (OpenPose skeleton renderer, depth renderer, frame inspector). |
| [`web/`](web/) | The web pixel editor: `vendor/piskel/` (upstream Piskel, pinned), `src/` (the studio panel, TypeScript), `piskel-plugin.mjs` (joins the vendored Piskel for Vite), `dist/` (the build; generated, in the image). |

## Workflow format — API only, harness-driven

Every JSON in [`workflows/`](workflows/) is **API format** — a flat `{node_id: {class_type, inputs, _meta}}` dict. This is the format ComfyUI's `/prompt` endpoint accepts and the format any automated harness (Python script, MCP-enabled coding agent, Node script, curl one-liner) consumes.

These workflows are **not** UI workflows. They were authored and maintained as API JSON — there is no saved node layout, no manually-arranged graph, no widget-position metadata. If you drag one into ComfyUI's editor, modern versions will auto-arrange the nodes so you can inspect the structure, but the result is ad-hoc and not optimised for human editing. **The intended way to use this repo is to POST workflows to ComfyUI from a harness, not to click "Queue" in the UI.**

The Python drivers in [`drivers/`](drivers/) are reference harnesses — they read the workflow JSON, patch a few inputs (animation name, seed, …), POST to `/prompt`, poll `/history`, and return the output paths. Each driver also handles cross-workflow gluing the workflows themselves can't express: passing W2's output mp4 path into W3's `LoadVideo` node, version-bumping output directories, sweeping seeds across re-runs, picking the latest mp4 from a versioned folder. **If you re-implement in another language, you re-implement that orchestration too.** See [`docs/harness-integration.md`](docs/harness-integration.md) for a minimal language-agnostic drive loop and Python / Node examples.

## Documentation

- **[`docs/pipeline-overview.md`](docs/pipeline-overview.md)** — full pipeline design, rationale, hardware tuning rules.
- **[`docs/workflow-1-pose-edit.md`](docs/workflow-1-pose-edit.md)** — W1 helper (Qwen-Image-Edit + ControlNet).
- **[`docs/workflow-2-video-gen.md`](docs/workflow-2-video-gen.md)** — W2 (WAN 2.2 i2v).
- **[`docs/workflow-3-spritesheet.md`](docs/workflow-3-spritesheet.md)** — W3 (BiRefNet → greyscale strip).
- **[`docs/workflow-4-cosmetic-inpaint.md`](docs/workflow-4-cosmetic-inpaint.md)** — W4 (VACE inpaint, two-pass bootstrap).
- **[`docs/workflow-5-cosmetic-spritesheet.md`](docs/workflow-5-cosmetic-spritesheet.md)** — W5 (SAM3 → cosmetic-only strip).
- **[`docs/iris-greenkey.md`](docs/iris-greenkey.md)** — eye cosmetic post-processing.
- **[`docs/models.md`](docs/models.md)** — full model list with download links and disk-path conventions.
- **[`docs/custom-nodes.md`](docs/custom-nodes.md)** — custom-node install instructions.
- **[`docs/harness-integration.md`](docs/harness-integration.md)** — driving these workflows from MCP / Python / Node / any harness.

## Contributing

Issues and PRs welcome. The pipeline has been validated on idle, walking, and in-air animations on a single base character — extending to new animations or new cosmetic categories should "just work" via the configuration knobs at the top of each driver, but feedback on edge cases is appreciated.

## License

[MIT](LICENSE) — use, modify, and redistribute freely. If this saves you time, a credit link back is appreciated but not required.
