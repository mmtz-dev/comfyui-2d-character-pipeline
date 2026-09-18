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
| Weights | ~120 GB (Wan 2.2, VACE, SAM3, GGUF) | 41 GiB, all Apache-2.0 (`gpu/models.lock`) |
| Post-processing | inside ComfyUI (BiRefNet) | deterministic CPU chain outside ComfyUI (rembg → pixeloe → pixel-art-fixer → ImageMagick `-remap`) |

## The `anim` CLI — the whole flow

Everything CPU-side is one command. It runs on **Docker Desktop**; the only stage
that touches a GPU is `render`, over HTTP.

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
```

| Stage | In | Out | Notes |
|---|---|---|---|
| `extract <clip>` | a filmed clip | `frames/%04d.png`, `job.json` | fps is a **cap**, not a target; a 10 fps clip stays at 10. Rotation is honoured by never passing `-noautorotate`. `--crop` (off by default) crops every frame to **one** person-centred box — use it when the figure sits small in the frame, because the skeleton is letterboxed into the cell and headroom is wasted resolution. |
| `pose <job>` | frames | `pose/keypoints.json`, `pose/skel_%04d.png` | rtmlib DWPose over onnxruntime, CPU. Inference happens **once per clip** and is cached; `pick`, `review` and `strip` re-read it. Rendered with rtmlib's own `draw_skeleton(openpose_skeleton=True)` at the cell size, line width scaled to the canvas. |
| `rig <glb> --clip <name>` | a rig clip | `job.json`, `frames/%04d.png`, `pose/keypoints.json`, `pose/skel_%04d.png` | **replaces `extract` + `pose`** when the motion is a rig clip. Samples the clip at the extract fps, maps the bones to OpenPose-18, projects in profile **facing right by detection**, one scale per rig; the skeleton PNGs go through the very same `posekit.draw`, with the unit's props (sword, shield) drawn on as white outlines. `--list` prints the clips, `--props` overrides the kit. See "Rig input mode". |
| `pick <job>` | keypoints | `picks.json` | peaks of deviation-from-rest + reversals + both endpoints, de-duplicated in torso lengths. Also records the **timing shape** (`timing`: one weight per frame + the anchor frame — the strip timing feature): the per-verb template in `config.ACTION_TIMING` by default, or the clip's own frame gaps with `--timing clip`. **Refuses to overwrite a hand-edited `picks.json`** (picks *or* timing) unless `--force`. |
| `review <job>` | frames + picks | rewrites `picks.json` | the click-to-pick page — see below. |
| `strip <job>` | picks + skeletons | `strip/skeleton_strip.png`, `reference_512.png`, `reference_strip.png`, `layout.json` | asserts the strip is non-square and both dimensions are multiples of 16. |
| `render <job>` | strip | `render/pose_pass.png`, `identity_pass.png`, `provenance.json` | **the only GPU stage.** Two `/prompt` jobs, not one graph. Seed is derived from the clip hash + action + picks unless `--seed` is given, so a re-run reproduces. |
| `post <job>` | generated strip | `post/<slug>_<action>_<n>.png`, `report.json` | deterministic and entirely outside ComfyUI: matte → slice → grid repair → pixeloe → palette snap → recanvas → verify. A frame that fails verification is not written. `--matte rembg` (default) is the real cut; `--matte chroma` is a deterministic flat-background key — use it on drawn or fixture strips, where `isnet-anime` has no character to find and returns an empty mask. |
| `publish <job>` | finished frames | `assets/2d/units/…` + one sidecar | copies **in place** (never rename), clears stale numbered frames, writes `contract.timing` + `generation.timing_source` (hand edit > what `pick` recorded > template > uniform, which writes no block), then runs game-02's asset audit and fails loudly on a non-zero exit. |
| `base --unit <slug>` | a canned standing skeleton, or `--rig <glb> --clip Idle --frame 0` | `assets/2d/units/<slug>.png` + sidecar | for a unit with no sprite yet: the pose stack alone, one 512×768 portrait cell, `contract.frames [1, 1]`. With `--rig` the standing pose is one sampled frame of a rig clip at the same facing and scale as that rig's action strips; `--skeleton-only` stops before the GPU. |
| `run <clip>` | a filmed clip | all of the above | stops politely before `render` when the GPU is off. |

### The review page

```bash
ANIM_JOB=knight_strike docker compose up review
# then, from the Mac:
open http://192.168.1.158:8710/
```

A contact sheet of every extracted frame, picks highlighted, dark ground and
`image-rendering: pixelated`. Clicking a frame toggles it and rewrites
`picks.json` immediately; the other picks keep their own frame numbers. Re-run
`anim strip` afterwards and the new picks are what gets composed.

**The "Request regenerate" button does not touch the GPU.** It sets
`rerender_requested` in `job.json` and stops. A page that could start a 24 GB
model load is a page a stray browser prefetch can fire, and the GPU service is
on the other Docker engine and usually not running at all.

Port **8710** was free on 2026-09-13 (`ss -tlnp` and `docker ps` both checked;
the occupied neighbours are 8700 openproject, 8770 research-digest, 8787
bg3-modded-wiki). It is bound to `0.0.0.0` because the owner opens it from the
Mac — a `curl` from the Fedora host to its own LAN IP routes via `lo` and proves
nothing about LAN reach.

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
```

What `anim rig` writes is byte-for-byte the layout `extract` + `pose` leave behind
(`job.json`, `frames/`, `pose/keypoints.json`, `pose/skel_*.png`), so `pick`, `review` and
`strip` run unchanged. Four things are worth knowing:

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
| `ANIM_WORKFLOW_DIR` | `/work/gpu/workflows` | the API-format workflows `render` POSTs |
| `ANIM_MODELS_LOCK` | `/work/gpu/models.lock` | repo + revision + sha256 per weight, copied into the sidecar |
| `ANIM_ASSETS_DIR` | `/assets` | game-02's published assets (read-write) |
| `ANIM_GAME02_DIR` | `/game02` | the game-02 repo, read-only, so `publish` can run its audit here |
| **`ANIM_PALETTE`** | `/assets_src/palette.gpl` | the `-remap` source of truth. Currently Resurrect 64 as a **provisional working default** — see below |
| `ANIM_EXTRACT_FPS` | `12` | the decode-rate cap (and the rate the game plays frames back at) |
| `ANIM_POSE_MODEL` | `wholebody` | `wholebody` is DWPose (134 joints); `body` is the lighter 18-joint rig |
| `ANIM_POSE_MODE` | `balanced` | `lightweight` \| `balanced` \| `performance` |
| `ANIM_POSE_KPT_THR` | `0.4` | below this a joint counts as absent, for the picker and the renderer |
| `ANIM_PICK_DEDUP` | `0.12` | minimum distance between two picks, **in torso lengths** |
| `ANIM_REVIEW_PORT` | `8710` | the LAN port; keep in step with the published port in `compose.yml` |
| `ANIM_REVIEW_BIND` | `0.0.0.0` | the Mac has to reach it |
| `ANIM_LAN_HOST` | `192.168.1.158` | only ever used to print a clickable URL |

Two tables in `config.py` are code rather than environment, because they are
design decisions and belong in review: `ACTION_FRAMES` (the art bible's key-pose
budget per action — strike 4, hit 2, walk 6, …) and `UNIT_HEIGHT` (the scale
ladder — `hound` 37 px, `knight` 53 px, `barrow_warden` 63 px).

## Manual steps — the two things the pipeline cannot do for itself

**1. The owner starts the GPU service.** `docker.service` is disabled and
starting it needs a real TTY for the sudo password, so no agent and no script
does it:

```bash
sudo systemctl start docker                                   # owner, in a terminal
docker --context docker-default compose \
    -f tools/anim-pipeline/gpu/compose.yml up -d              # GPU containers ONLY
curl http://192.168.1.158:8188/system_stats                   # should name the A5000
```

Never `docker context use`. Everything else in this repo stays on Docker Desktop.

**2. The owner picks the palette.** `assets_src/palette.gpl` exists (Resurrect 64,
installed 2026-09-13 as a *provisional* working default so the pipeline can run) but the
pick is still open: the owner is exploring with the art bible's palette explorer, and
`game-02-tactical-rpg/assets_src/install_palette.py <slug>` is how a new pick gets
installed (it rewrites `.gpl`/`.hex`/`.png` in place). To try a staged candidate without
installing it, point the pipeline at it:

```bash
ANIM_PALETTE=/assets_src/palette_candidates/grim32/grim32.gpl \
  docker compose run --rm anim post knight_strike
```

A `.gpl` or a one-row `.png` are both accepted; the `.gpl` stays the
human-editable source and the PNG `magick -remap` needs is derived from it.

## Tests

```bash
docker compose run --rm anim-pipeline pytest        # 207 passed, 1 skipped
```

No GPU, no model download, no network. The GPU is replaced by a threaded fake
ComfyUI (`tests/fixtures/fakecomfy.py`) that returns canned `/system_stats`,
`/prompt`, `/history` and `/view` responses; the clips and the "generated"
strips are drawn with Pillow; the sidecar is judged by game-02's **real**
`asset_audit.py`, copied into a temp tree. The rig tests (`test_rigkit.py`,
`test_rig.py`) use a two-bone GLB built in the test for the maths and the
vendored CC0 `Knight.glb` for the end-to-end run.

The one skipped test is opt-in and needs the network on first run — it proves
rtmlib actually finds a person in a real photograph:

```bash
docker compose run --rm -e ANIM_POSE_SMOKE=1 anim-pipeline pytest tests/test_pose.py -k smoke
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
| `gpu/` | **The GPU image service feature** — ComfyUI in a pinned container on the native Docker daemon, the only GPU service in the project. Compose file, `models.lock`, pinned custom nodes, the three workflows. Start here: [`gpu/README.md`](gpu/README.md). |
| `gpu/workflows/pose_pass.json` | Stage 1: Z-Image-Turbo (int8) + Z-Image Fun ControlNet Union 2.1 (full 2602 8-step; lite was ignoring the pose). Skeleton strip → posed strip, drawn at 2 px. No upstream counterpart; upstream had no pose-only stage. |
| `gpu/workflows/identity_pass.json` | Stage 2: the reworked W1. Re-skins the posed strip onto the unit's reference sprite. Two images in (reference, pose pass). **Off since the anchor gate** — it flattens poses; `post` takes the pose pass. |
| `gpu/workflows/base_sprite.json` | Generates a unit's standing sprite from a canned skeleton when no reference sprite exists yet. |
| `gpu/workflows/NOTES.md` | Which node names and input keys were verified against source, and what is still `TODO-verify`. |
| `anim/` + `Dockerfile` + `compose.yml` | The CPU pipeline on Docker Desktop: `anim extract pose rig pick review strip render post publish base`. |
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

| Engine | Context | GPU | Runs |
|---|---|---|---|
| Docker Desktop (**default**) | `desktop-linux` | no | the CPU pipeline, the model downloader, the Mac share |
| Native `docker.service` | `docker-default` | **yes** | **only** `gpu/compose.yml` |

- Never run `docker context use`. Target the GPU service explicitly: `docker --context docker-default …`.
- The native daemon is disabled/stopped by default and only the owner starts it
  (`sudo systemctl start docker`, in a real TTY).
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
