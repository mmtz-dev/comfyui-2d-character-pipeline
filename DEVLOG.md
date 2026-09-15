# DEVLOG — the pose-to-pixel animation pipeline feature

What changed and *why*. Status lives in OpenProject, never here.

---

## 2026-09-14 — the rig motion input feature: `anim rig`, the KayKit Knight as the anchor's motion

**What.** A CC0 rigged, animated glTF character is now a motion source on equal footing with
a filmed clip. `anim rig <glb> --clip <name> --unit <slug> --action <verb>` samples the clip,
maps the bones onto the 18-point OpenPose skeleton, projects it orthographically in profile,
and writes exactly the `work/<job>/` layout `extract` + `pose` would have — so `pick`,
`review`, `strip`, `render`, `post` and `publish` run unchanged. `anim base` gained
`--rig <glb> --clip Idle --frame 0` for the standing sprite. Suite: **207 passed, 1 skipped**
(was 154 + 1; 53 new tests). Vendored under `rigs/`: KayKit Adventurers `Knight.glb` (CC0, 76
clips) and Quaternius UAL `UAL1_Standard.glb` (CC0 by the zip's own `License.txt`, 43 clips).
`pygltflib` (MIT) added to `requirements.txt` and the lock; image rebuilt.

**Why.** The owner decided (2026-09-14, from the premade-assets research §5) to use the KayKit
Knight for the anchor unit instead of filming. It takes the owner's filming session, the Adobe
login, the Mixamo ML-clause decision, the TripoSG GPU build and the Mixamo auto-rigger — five
steps of the 3D spike, including the one the spike's own research called the most likely to
fail — off the critical path in one move, and gives the 2D chain exact profile skeletons with no
DWPose noise and no depth ambiguity. Filming stays the way to get the owner's *own* performance;
this is someone else's motion, and the honest limit RESEARCH.md §5 states still stands.

### Staged: six jobs in `work/`, ready for `anim render`

| Job | Clip | Frames | Picks (N) | Strip |
|---|---|---|---|---|
| `knight_base` | `Idle` #0 via `anim base --rig` | 1 | [0] (1) | 512×768 |
| `knight_strike` | `1H_Melee_Attack_Chop` | 14 | [0, 6, 9, 13] (4) | 1216×320 |
| `knight_hit` | `Hit_A` | 9 | [2, 8] (2) | 608×320 |
| `knight_death` | `Death_A` | 11 | [0, 5, 6, 10] (4) | 1216×320 |
| `knight_collapse` | `Lie_Down` | 37 | [0, 18, 36] (3) | 912×320 |
| `knight_revive` | `Lie_StandUp` | 29 | [0, 22, 28] (3) | 912×320 |

Contact sheets with the picks highlighted and the strips:
<http://192.168.1.158:8090/kaykit-knight-skeletons.html>.

**Chop over Slice_Horizontal for `strike`.** Both were sampled and picked side by side (the page
shows them). The chop swings in the sagittal plane — overhead windup, down-and-forward cut, a
torso lean — which a side view shows in full. The horizontal slice sweeps across the body, the
transverse plane, and in profile foreshortens to a small forward-back wobble of the arm; nine of
its fourteen frames are already the recovery. A profile sprite wants sagittal motion.

### Facing is detected from the rig, not assumed from glTF

glTF has no "forward". Blender exports usually face +Z; Unreal-style exports face +X or −Y;
"usually" is how a whole strip comes out facing left. So `rigkit.facing_direction` reads the
rest pose: the horizontal component of the foot→toes vector, both feet summed (`forward_hint`
in the joint map). That vector is the direction the character faces whatever the modelling
convention was, and the projection maps it onto +x so the figure faces **right** — the art
bible's party-faces-right rule. The Knight came out +Z; the Quaternius mannequin will be
detected the same way. `--mirror` flips the projection only: the synthesised face points stay
on the character's real front (the first cut mirrored the facing vector before synthesising the
face and put the nose behind the head — caught by the test that asserts a mirrored frame is an
exact reflection of the plain one).

The alternative — the head bone's local forward axis — is rig-specific knowledge (which local
axis is forward differs by exporter) and needs a second table; toes are a fact about the
character.

### One scale per rig, from the rest pose, never per clip

The rest-pose mesh extent gives the floor (min y) and the top (max y — the helmet on the
Knight); that standing height maps to 78 % of the cell with the floor on row 300 of 320. Every
clip of the rig projects at that one scale, so `hit` and `death` cannot come out a different
size from `strike`, and a body lying on the floor lies on the row the standing feet stood on.
Per-clip fitting (the spike's `fit_to_canvas`) would have made the death frames larger than the
strike frames because a lying figure is short. 78 % because a lying chibi is 250 px of a 304 px
cell and rtmlib silently drops any limb that touches the frame edge; every joint of every staged
clip is checked to be inside the cell.

Horizontally each frame is centred on its own extent. Hip-centring was tried first and put the
lying Knight's head 8 px from the cell edge (the head is ~45 % of the body). The frames' x
position carries nothing downstream — cells are independent and `post` bottom-centres — so the
safest choice is the right one.

### The joint map is a table, and the neck is the shoulder midpoint

`rigkit.RIGS` holds one `JointMap` per rig family, chosen by which bone names are present.
KayKit: `neck = midpoint(upperarm.l, upperarm.r)`, shoulders = `upperarm.*`, elbows =
`lowerarm.*`, wrists = `wrist.*`, hips = `upperleg.*`, knees = `lowerleg.*`, ankles = `foot.*`,
face from `head`. One deviation from the brief, which said neck = chest tip / head origin:
rtmlib's `convert_coco_to_openpose` defines the OpenPose neck as `np.mean(keypoints[:, [5, 6]])`
— the shoulder midpoint — so every DWPose skeleton the ControlNet was trained on has its neck on
the shoulder line, and a neck 0.13 units higher would be a neck it never saw. The head origin is
still the base for the five face points, which are offsets in the head bone's frame in head
lengths (as `mixamotoopenpose` does), the head length measured from the head origin to the mesh
top so a chibi head and a realistic head both get their nose halfway up the face.

**Quaternius is not `DEF-*`.** The research doc's Rigify bone names came from a 2025 community
mirror; the 2026 Standard zip's `Unreal-Godot/UAL1_Standard.glb` uses Unreal-mannequin names
(`pelvis, spine_01, neck_01, Head, clavicle_l, upperarm_l, lowerarm_l, hand_l, thigh_l,
calf_l, foot_l, ball_l`). The `quaternius-ual` map uses those; the test that loads the real
file caught it.

### The picker's two-pose case

Staging `knight_hit` exposed a picker gap: with N = 2 the "both endpoints always" rule returned
frames 0 and 8 of `Hit_A` — rest and rest, no flinch — and the same would hold for every
2-pose action (`idle`, `pickup`, `dash`, `equip`, `guard_set`). Now, for N == 2 only, the picks
are the deviation extreme and the last frame: the flinch then the recovery, the stoop then the
stand, the breath then the loop point. N ≥ 3 is unchanged and the existing tests still hold.

### Provenance: two of them, side by side

`rigs/<pack>/rig.json` (pack, author, url, licence, per-file sha256) is checked against the GLB
— a hash mismatch is an error, because a rig whose bytes changed is a rig whose licence has to
be re-read — and copied into `job.json.motion_source`. `publish` now carries that block into the
sidecar as `generation.motion_source`, distinct from the seed / workflow / model hashes, and
`source.author` says "rig clip" instead of "filmed reference clip". game-02's `asset_audit.py`
accepts the extra key (verified against the real script in the suite, and the same test proves
it still refuses a sidecar without a seed — nothing was weakened).

### What is left

- `anim render` on the six jobs, when the owner starts the GPU service. The skeletons are the
  Knight's real proportions; whether the ControlNet honours a ~2.2-head skeleton is the anchor
  gate's question, and the Quaternius rig is vendored for the A/B.
- The 3D spike's only open question is now whether a rendered depth / normal channel from the
  KayKit mesh adds anything over skeleton-only conditioning. TripoSG, Mixamo, `dae_to_joints`
  and `render_openpose.py` are superseded for the anchor unit (note in `spike3d/README.md`).
- Quaternius's *Ultimate Animated Character Pack* could not be fetched: its Google Drive folder
  answered "quota exceeded" on 2026-09-14 (`rigs/quaternius/NOTICE.md`). The UAL was the rig
  the research recommended for the A/B anyway.

---

## 2026-09-13 — the CPU stages: `anim extract … publish`, test-first

All ten stages implemented and green: `docker compose run --rm anim-pipeline
pytest` → **154 passed, 1 skipped** (the skip is opt-in and needs the network;
see below). The scaffold's nine stubs printed "not implemented" and returned 2;
`tests/test_cli.py::test_no_stage_is_still_a_stub` now asserts none of them do.

`run` is a tenth stage the scaffold did not have: the whole chain in one command,
which is how this will actually be used at 100+ animations.

### Shape: a pure function plus a thin CLI, and all state in files

Every stage module is `<verb>(...) -> result` (no argparse, no printing, no
globals) with a twenty-line `run_stage(args)` wrapper over it. That is what let
the suite drive `render` against a fake HTTP server, `publish` against a temp
assets tree, and `post` against a Pillow-drawn strip — none of which is possible
if the logic lives in the CLI handler.

There is no pipeline object handed from stage to stage. Every stage reads files
and writes files under `work/<job>/`, so any stage re-runs on its own, after a
hand edit, any number of times — and a job that failed at `post` does not cost
another GPU render. `anim/job.py` is the path arithmetic and nothing else.

### The picker heuristic — two signals, not one

The interesting design call. Working on **normalised** poses throughout
(root-centred on the pelvis, scaled so one torso length = 1.0), because a subject
who stepped closer to the camera has not changed pose and a pixel-space picker
would think they had. That normalisation is also what makes `ANIM_PICK_DEDUP`
a number with a meaning: 0.12 is "a limb moved about an eighth of a torso".

Two curves:

- **deviation** — distance from the rest pose (frame 0, which the art bible's
  filming rules make the rest pose by telling the owner to hold the start for half
  a second). Its *peaks* are the extremes of the motion: windup, contact,
  follow-through. These are the poses that read at 1/12 s each.
- **speed** — distance from the previous frame. Its *minima* are the held
  moments, where a limb reverses and the pose is momentarily still. A reversal is
  a key pose almost by definition.

Peaks of the first alone would miss a reversal that happens to sit on a plateau;
minima of the second alone are dominated by the still half-second at the start of
every take. Candidates are the union plus both endpoints, taken in order of
deviation, each admitted only if it is at least `dedup` away from everything
already chosen — otherwise four frames a tenth of a second apart in the middle of
one swing crowd out the recovery. The endpoints are always in (an action that
does not end where it started reads as a jump cut), and if the threshold is too
strict to reach N it relaxes and finally falls back to an even spread, because a
strip has N cells and a short strip is a broken strip.

Deterministic: same keypoints in, same picks out. Tested on a hand-written
13-frame swing where the answer is obvious by inspection (frame 6 is the extreme),
plus the degenerate cases — a take padded with a long still hold, a take with no
motion at all, a frame where no person was detected.

### The chroma-key palette remap — why not just quantize

Pillow's `quantize()` silently falls back to octree on RGBA and will not honour a
fixed palette, so the snap is ImageMagick's `-dither None -remap palette.png` on
flat RGB. That means flattening the alpha first, and *what* it is flattened onto
matters:

- onto **white** or onto any **palette colour**: the background fuses into the
  silhouette and `-remap` cannot tell the two apart at the edge.
- onto a colour the palette **cannot represent** — a chroma key — every background
  pixel is forced somewhere harmless, the binary mask goes back on afterwards, and
  the key never survives into the output.

Related, measured the same day: `--matte rembg` (`isnet-anime`) returns a clean
binary matte on a real photograph — 78% opaque, alpha strictly {0, 255} — and an
**empty** one on a strip of flat drawn shapes, because there is no anime
character in it to find. So `rembg` stays the default for real generations and
`--matte chroma` is what the tests and any drawn input use. The chroma keyer
samples the border ring rather than assuming white, so a strip generated on light
grey still keys.

`imaging.chroma_key_for()` picks magenta unless the palette contains it, then
walks the 6×6×6 cube for the first free colour, so it is deterministic and cannot
collide however exotic the palette. Transparent pixels are then zeroed to
`(0,0,0,0)` so the published bytes do not depend on which key was used, and
`verify_frame` only judges opaque pixels.

Every published frame is verified before it is written: 64×64, alpha strictly in
{0, 255}, every opaque colour a palette member. A frame that fails is not saved
and the stage raises with the reason.

### Measured: `pixelfixer` runs BEFORE `pixeloe`, not after

The plan's post-chain order was pixeloe → pixel-art-fixer. Measured on
2026-09-13, that is backwards. `pixelfixer process` looks for the block grid of an
*up-scaled* pixel image and reconstructs it at one pixel per block — so on a 51×54
image that pixeloe had already reduced to native scale it "found" a 5.77 px grid
and collapsed it to **9×10**. Run first, on the full-resolution 304 px cell, it
does the job it was built for.

It is guarded either way: an output shorter than the target height is rejected and
`post/report.json` records the rejection, so the chain cannot silently lose the
sprite. `--no-pixelfixer` skips it entirely.

Second measured pixeloe finding: its outline-expansion pass strides by
`(patch_size // 4) * 2`, so any `patch_size` below 4 gives torch a stride of zero
and blows up inside `im2col`. Nothing in pixeloe guards it; `post.MIN_PATCH = 4`
does.

### The scale ladder is applied here, not left to the model

`post` downscales each cell to the unit's band from the art bible's scale ladder
(`config.UNIT_HEIGHT`: hound 37 px, the standard soldiers 53, `barrow_warden` 63),
then recanvases bottom-center on 64×64. Asking a diffusion model to make a hound
smaller than a boss is a prompt-engineering problem; asking `Image.resize` is
arithmetic.

### The review page flags, it does not fire

`anim review` is stdlib `http.server` on **port 8710** (free on 2026-09-13;
`ss -tlnp` and `docker ps` both checked — the occupied neighbours are 8700
openproject, 8770 research-digest, 8787 bg3-modded-wiki), bound to `0.0.0.0`
because the owner opens it from the Mac.

Its "Request regenerate" button sets `rerender_requested` in `job.json` **and
stops**. A web page that could start a 24 GB model load is a page a stray browser
prefetch can fire, and the GPU lives on the other Docker engine behind a daemon
only the owner starts. The flag is a note-to-self that survives closing the tab;
the owner re-runs `anim strip`/`anim render` when they mean it.

All behaviour lives in a `ReviewApp` class that knows nothing about HTTP, so it is
unit-tested without a browser; the handler is a thin adapter, tested over a real
socket on an ephemeral port. Images are served by exact name matched against the
job's own listing, so no amount of `../` in a URL reaches outside `work/<job>/`.

### `picks.json` records both `auto` and `picks`

When they differ, a human changed it — in the review page or in an editor — and
re-running `pick` would throw that away. So it refuses and says `--force`. This is
the one file in the pipeline a person edits, and the one place an automated stage
could destroy work that cannot be regenerated.

### `render`: two `/prompt` jobs, patched by title, seeded by derivation

Two jobs rather than one graph, because Z-Image int8 (~6 GB) plus Qwen-Image-Edit
INT4 (~12 GB) plus two text encoders does not comfortably share 24 GB — and a bad
identity pass can then be re-rolled without paying for the pose pass again.

Nodes are patched by `_meta.title`, never by id, exactly as upstream's drivers do
and for the same reason: re-exporting a workflow renumbers nodes, and swapping the
reference sprite for the skeleton by accident would produce a beautifully posed
skeleton instead of a character. `comfy.patch()` also refuses to set an input the
node does not declare, so a renamed widget fails here rather than inside ComfyUI.

**The seed is derived, not random**: `sha256(clip_sha256 | action | slug | picks)`
unless `--seed` is passed. A sidecar that says "seed 861234901" is a promise the
frame can be re-made, and a random default quietly breaks that promise for anyone
who re-runs without reading the sidecar first.

`render/provenance.json` records the ComfyUI version and device from
`/system_stats`, the sha256 of each graph **as sent** (canonical JSON — sorted
keys, no incidental whitespace, so the hash cannot move because a dict reordered),
and the repo + revision + sha256 of every weight the graph names, read out of
`gpu/models.lock` rather than typed. Model rows are matched by scanning node
inputs for a locked filename — deliberately dumb, because a per-node-class table
of "which input holds a model name" goes stale the moment a workflow gains a node.

Tested against a threaded fake ComfyUI returning canned responses, which is enough
to assert the real thing that can break: that the posted graph carries seed 4242,
`sampler_name: euler` (never ancestral), a 1216×320 latent, and `image1` =
`reference_512.png` / `image3` = `pose_pass.png` and not the other way round.

### `publish`: in-place writes, and `source.license` is CC0, not Apache-2.0

Frames are written with `open(path, "wb")` and never renamed into place.
Docker Desktop's file share caches by inode, so the usual write-temp-then-rename
leaves a new inode behind the same name and a container holding the old one keeps
serving the old bytes — the Mac-edit-share quirk, from the other direction.
`tests/test_publish.py` asserts the inode survives a re-publish.

Stale numbered frames are deleted: re-publishing a 4-frame action over a 6-frame
one would otherwise leave `_4` and `_5`, which the audit correctly calls an error
(an unclaimed numbered PNG is silently unused art). Only the reserved `_<number>`
spelling is removed — `knight_strike_sneak.png` is a different asset, and there is
a test that says so.

**One deviation from the work package, and it is forced by the audit.** The brief
said the sidecar's `source.license` should be `"Apache-2.0"` (the model licences).
`asset_audit.py` rejects that: `ALLOWED_LICENCES` is `{CC0-1.0, CC0, PD, Public
Domain}`, and the audit's own test for an ai-generated sidecar
(`test_asset_audit.py`, `sidecar()`) writes `CC0-1.0`. Source wins over brief, so:

- `source.license` = **`CC0-1.0`** — the licence of the *output*, which is ours
  and is released CC0 with the rest of the assets.
- `source.model_license` = `Apache-2.0`, `source.models[].license` and
  `generation.models[].license` = the per-weight licence from `models.lock`, and
  `source.license_note` spells out that Apache-2.0 weights place no restriction on
  generated output, so the frames are commercially usable.

Nothing is lost — the weights' licence is recorded in three places — but **there
is a latent bug in the audit for its owner**: `_generated_sections()` pairs
`source.license` with each `generation.models[].repo`, so `CREDITS.md` will print
"Comfy-Org/z_image_turbo — License: CC0-1.0", which is not true of the weights. It
should read the per-model licence. Not fixed here: `asset_audit.py` belongs to the
frame-group asset audit feature, not to this one.

The audit itself is not optional. `publish` runs
`python3 /game02/tools/asset_audit.py` (the repo is mounted read-only at
`/game02` for exactly this; it is stdlib-only so it runs in the slim image) and
raises on a non-zero exit. Verified against the real tree: **OK — 180 asset(s)**
today, so a publish will not trip over pre-existing drift.

### Fixtures: everything drawn, except one CC0 photograph

Clips, strips and keypoints are all generated by `tests/fixtures/synthetic.py`
and `conftest.py` — no licence question, and the expected answers are obvious by
inspection. The canned standing skeleton `anim base` ships
(`anim/data/stand_pose.json`) is synthesised joint coordinates, not traced.

One real photograph is unavoidable: "does rtmlib actually find a person" can only
be answered by a person. It is
[a CC0 1.0 Flickr photo by D Coetzee](https://www.flickr.com/photos/29507259@N02/3624201561),
licence verified twice (Openverse reports `cc0`/`1.0`; the Flickr page itself
carries `"license":9`, Flickr's CC0 id), cropped and downscaled. Full provenance
in `tests/fixtures/FIXTURES_NOTICE.md`, including why the obvious public-domain
choice — Muybridge's *Animal locomotion* human plates — was rejected (they are the
"nude and semi-nude" volume).

The test that uses it is **opt-in** (`ANIM_POSE_SMOKE=1`) because it downloads
~300 MB of DWPose weights on first run, and a suite that needs the network goes
red for the wrong reasons. It was run and passed on 2026-09-13. The weights land
in the bind-mounted `.cache/rtmlib/` (the Dockerfile points `XDG_CACHE_HOME` at
`/work/.cache`), so they are fetched once per machine.

End-to-end proof on real inference, not just fixtures: `anim run` over a
30-frame synthetic clip found a person in all 15 decoded frames, picked
`[0, 8, 10, 14]`, composed a 1216×320 skeleton strip, then stopped before render
with the two commands needed to start the GPU. Feeding a drawn strip through
`post` produced four 64×64 sprites of 6–8 colours, all Grim32 members, alpha
binary, feet on row 63.

### Two small changes outside `anim/`

- `Dockerfile`: a second pip layer for `requirements-dev.txt` (pytest), placed
  *after* the expensive one so moving a test dependency never re-resolves torch +
  onnxruntime + the CUDA wheels; `XDG_CACHE_HOME=/work/.cache` so the model
  downloads persist; and `PYTHONPATH=/work` — without it the editable install
  resolves to the `COPY` at `/opt/anim`, not the mounted source, so editing
  `anim/` changed nothing until a rebuild.
- `compose.yml`: an `anim-pipeline` service whose entrypoint is cleared so
  `docker compose run --rm anim-pipeline pytest` works against the `anim`
  ENTRYPOINT image; the read-only `/game02` mount; `ANIM_PALETTE` defaulting to
  the (not yet existing) `palette.gpl` with the candidate override documented in
  place.

### Still open

- The palette. `assets_src/palette.gpl` does not exist — the owner has not picked
  a winner from the shootout. Everything runs today with
  `ANIM_PALETTE=/assets_src/palette_candidates/grim32/grim32.gpl`.
- `render` has never met the real ComfyUI. Every `TODO-verify` in
  `gpu/workflows/NOTES.md` is still open, and the first real render is where they
  get settled.
- The skeleton is letterboxed into the cell, so headroom in the source frame is
  wasted resolution. `--crop` exists and is off by default; whether it should be
  on is a question for the first real footage.

---

## 2026-09-13 — scaffold: the GPU image service feature + the pipeline skeleton

Stood up the two infrastructure features from the plan. No pipeline stage is
implemented yet; this is the frame everything hangs on.

### Started as a fork, not a rewrite

`gh repo fork mor-o/comfyui-2d-character-pipeline --clone` →
`mmtz-dev/comfyui-2d-character-pipeline`, cloned to `tools/anim-pipeline/` with
`upstream` as a second remote. MIT `LICENSE` untouched, credited in `NOTICE`.

**No upstream stage was deleted.** W2 (Wan 2.2 i2v), W4 (VACE inpaint) and W5
(SAM3 layers) are *parked* — workflows, drivers and docs all still in the repo,
weights not downloaded (~120 GB avoided). They are the starting point for a later
full-cycle animation tier. W1 (Qwen-Image-Edit + ControlNet pose edit) became the
basis for `gpu/workflows/identity_pass.json`; W3's strip slicing informed the CPU
post-chain. The full mapping is the "Stage mapping" table in `README.md`.

Deleting them would have been easy and wrong: the parked stages are the only
existing, working reference for the video route, and re-deriving them later costs
far more than carrying dead files.

### One knob for the ComfyUI URL

Upstream hardcoded `http://127.0.0.1:8000` in all five drivers and three docs.
Now every one reads `COMFYUI_URL`, defaulting to `http://host.docker.internal:8188`.
`grep -rn '127.0.0.1:8000' .` is empty and should stay that way.

The default is not localhost because **ComfyUI and the pipeline run on different
Docker engines**: ComfyUI on the native daemon (the only one with GPU access), the
pipeline on Docker Desktop (a Linux VM). From inside that VM, the host's published
port is reachable as `host.docker.internal`, never `127.0.0.1`. That is also why
`gpu/compose.yml` binds `0.0.0.0:8188` rather than loopback — a loopback bind would
be invisible both to the Desktop VM and to the owner's Mac.

### The GPU service is prepared, not started

`gpu/compose.yml` is written, validated (`docker --context desktop-linux compose
config` exits 0, project `anim-pipeline-gpu`) and **not run**. The native daemon
stays stopped; only the owner starts it, from a TTY. `gpu/README.md` has the exact
sequence.

Choices worth recording:

- **Image pinned by digest**, `mmartial/comfyui-nvidia-docker:latest@sha256:7a54a280…`
  (resolved from Docker Hub, tag last pushed 2026-09-13). A floating `:latest`
  would silently change torch, ComfyUI and node behaviour between renders.
- **A `run` volume as well as the basedir.** The image wants two mounts:
  `/comfy/mnt` (ComfyUI code + venv) and `/basedir` (BASE_DIRECTORY). The plan only
  named the basedir; without the `run` mount every recreate reinstalls torch into
  the container layer. Created `/mnt/private/AI Data/comfyui-run` for it.
- **`restart: "no"`.** The service holds ~20 GB of VRAM while loaded. It should not
  come back on boot.
- **`security_opt: label=type:nvidia_container_t`** — Fedora SELinux blocks the
  NVIDIA device nodes otherwise and ComfyUI silently starts CPU-only.
- **Ampere, so INT8/INT4.** The A5000 (sm_86) has no fp8/fp4 tensor cores; an fp8
  file saves VRAM but buys no speed, and nvfp4 will not run. Documented in
  `gpu/README.md` so nobody "upgrades" the weights into a wall. Qwen-Image-Edit at
  fp8 + fp8 TE needs ~27.9 GB and does not fit in 24 GB — that is the entire reason
  for the Nunchaku INT4 build.

### Nothing is duplicated on disk

`gpu/extra_model_paths.yaml` exposes the pre-existing
`/mnt/private/AI Data/models/Image Models/` read-only (`:ro` bind) so the SDXL /
Flux / Chroma / SD1.5 checkpoints already on this host are visible to ComfyUI
without being copied.

This surfaced a real hazard: **ComfyUI merges every configured path for a folder
type into one flat list keyed by filename.** The shared `vae/` already holds a Flux
`ae.safetensors`, and Z-Image's VAE is *also* called `ae.safetensors`. Two files,
one name, non-deterministic winner. So Z-Image's is installed as
`z_image_ae.safetensors` and the workflows reference that. The rule is written at
the top of both `extra_model_paths.yaml` and `models.lock`.

### models.lock is generated, and it caught a real bug

35.07 GiB across 8 files, all Apache-2.0 and ungated (verified via the HF API, not
by reading a model card). `gpu/relock_models.py` is the only writable source;
`models.lock` is a printout.

It cross-checks **two** HF endpoints and refuses to write on any disagreement.
That guard exists because the first version of the lock shipped an all-empty
sha256 column: `/api/models/<repo>?blobs=true` spells the hash
`siblings[].lfs.sha256`, while `paths-info` spells the same value `[].lfs.oid`. The
writer only read `oid`, got `""` for everything, and the downloader then "failed"
verification on a perfectly good 6 GiB file and deleted it. `hf_download.py` now
also refuses to run against a lock row whose hash is not 64 hex chars — the bug
class is closed at both ends.

Variant choices, from the model cards rather than guesswork:
- Fun ControlNet **`lite-2602-8steps`** — 2602 (2026-02-26) is the newest lite
  union build and supports Pose; the lite variants tolerate a higher control scale
  and suit lower-spec runs.
- Nunchaku **`lightning-251115/svdq-int4_r128 … 4steps`** — `lightning-251115` is
  the newest release, `int4` because Ampere, `r128` over `r32` for quality. The
  lightning LoRA is **fused into these weights**, so `identity_pass.json` has no
  separate lightning LoRA node.

### Downloads: 7 of 8, and an upstream integrity bug

Ran in a throwaway `python:3.12-slim` container on Docker Desktop (the host has
Python 3.14 and no pip; downloading weights needs no GPU, so this deliberately
never touches the native daemon). 34 GiB installed, every file sha256-verified.

**The optional Pixel-Art LoRA did not install.**
`prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA` advertises sha256 `2a2775ac…`
consistently — both endpoints, both `main` and the pinned revision — but the bytes
the CDN serves hash to `55300e5a…`. Confirmed with two independent fetches
(`huggingface_hub` and plain `curl` on the `resolve/` URL). The file is the right
size to the byte and parses as a well-formed Qwen-Image LoRA (1680 tensors,
dim 64 / alpha 32), so this is an upstream registry metadata bug, not a corrupt
download.

Left uninstalled deliberately. Accepting a hash that disagrees with the registry
would defeat the point of having the lock, and **nothing is blocked**: the LoRA is
optional and `identity_pass.json` bypasses it at `strength_model: 0.0`. The two
resolution paths are written up in the KNOWN DISCREPANCY block in `models.lock`;
it is an owner decision, not an agent one.

### Custom nodes pinned by commit, pre-cloned

`ComfyUI-nunchaku` @ `b600f879` (Apache-2.0, v1.2.1) and `ComfyUI-Image-Saver` @
`b4b349e5` (MIT, v1.24.1), cloned into the basedir's `custom_nodes/` by
`gpu/install_custom_nodes.sh` (idempotent). The compose file sets
`ALLOW_GIT_URL_INSTALL=false` / `ALLOW_PIP_INSTALL=false` so ComfyUI-Manager never
fetches code at runtime — a floating node HEAD changes render output silently.

Noted for later: `nunchaku-tech/ComfyUI-nunchaku` now redirects to
`nunchux-ai/ComfyUI-nunchaku` (org renamed). Also, the node pack imports the
`nunchaku` wheel and does not vendor it, so `NunchakuQwenImageDiTLoader` is absent
from `/object_info` until that wheel is installed into the container's venv —
`gpu/README.md` step 5, and the wheel filename belongs in this log when it lands.

`comfyui_vram_helpers` (upstream's own two-stage eviction nodes) is **not**
installed: our chain runs the pose pass and the identity pass as two *separate*
`/prompt` jobs, so ComfyUI's own model manager evicts between them. Simpler than
in-graph eviction, and a failed identity pass can be re-rolled without redoing the
pose pass.

### Workflows: hand-translated, and every node name verified against source

Three API-format workflows in `gpu/workflows/`. There are **no** API-format
templates upstream — all ~200 files in `Comfy-Org/workflow_templates` are UI-format
and both graphs we needed are wrapped in subgraphs, so they cannot be mechanically
flattened. Every `class_type` and input key was read from the ComfyUI source at a
recorded commit, not guessed. `gpu/workflows/NOTES.md` records what is CONFIRMED,
what is `TODO-verify`, and why each deviation was made.

Three findings that would have been silent bugs:

1. **`ModelPatchLoader`'s input key is `name`**, not `model_patch_name`.
2. **`CLIPLoader` has no `z_image` type.** Z-Image's text encoder is selected by
   auto-detecting the weights (`TEModel.QWEN3_4B`); the official template passes
   **`lumina2`**, which is what we pass.
3. **`ZImageFunControlnet` has no start/end percent or "control scale" input** —
   `strength` *is* the control scale. Set to 0.7 per the plan.

`base_sprite.json` is deliberately the pose stack **only**, one non-square portrait
cell (512×768). It exists for a unit that has no reference sprite yet, so there is
nothing for Qwen-Edit to take identity from — prompt and art bible are the only
style input. Running both models in one graph was also rejected on VRAM grounds.

Nothing is square anywhere (4×1 strip = 1216×320, cell 304×320; base = 512×768):
Qwen-Image-Edit degrades on square aspect ratios.

### CPU pipeline skeleton

`Dockerfile` + `compose.yml` (project `anim-pipeline`) on Docker Desktop, and an
`anim` CLI with nine stages — `extract pose pick review strip render post publish
base` — each a module exposing `NAME/HELP/add_arguments/run`. Every stage prints
`not implemented` and exits **2**, so a pipeline script cannot mistake a stub for
success. Each stub carries its intended arguments and the implementation
constraints in its docstring, which `anim <stage> --help` prints. A second agent
implements the bodies.

Built and verified on `desktop-linux`: `anim --help` runs, all of rtmlib,
onnxruntime 1.30, rembg 2.0.84, pixeloe (pinned git), Pillow 12.3, numpy 2.5,
cv2 5.0 and requests import, and `ffmpeg`/`magick`/`pixelfixer`/`rembg` are all on
PATH. onnxruntime reports CPU only, as intended — the GPU belongs to ComfyUI.

Two build problems worth recording, both fixed in-tree:

- **Debian bookworm ships ImageMagick 6**, which provides `convert` but not the v7
  `magick` entry point the post-chain calls. Switched both stages to
  `*-slim-trixie` (ImageMagick 7.1.1). The build now *asserts* v7 and runs the
  exact `magick -dither None -remap` invocation on a throwaway palette, so a base
  image regression fails the build instead of failing once per frame at runtime.
- **pixel-art-fixer does not compile on Linux.** It publishes no prebuilt binaries,
  so the Dockerfile compiles it — and its `peak_rss()` calls the Windows `kernel32`
  psapi API with no `#[cfg(windows)]` guard, so linking fails with
  `unable to find library -lkernel32`. `patches/pixelfixer-linux-peak-rss.patch`
  gates the Windows path behind `cfg(windows)` and adds a `getrusage(2)` version;
  it touches no algorithm code and is applied with `git apply` against the pinned
  commit, so it either applies exactly or the build fails loudly. Worth reporting
  upstream.

### Reproducibility chain, as it now stands

Container digest → custom-node commits → model repo+revision+sha256 →
`batch_size: 1` and `euler` (never ancestral) → seed and hashes in the per-action
sidecar. `pixeloe` and `pixel-art-fixer` are pinned to commits too, because a
change in either silently changes every published frame.

### Still open

- The `nunchaku` wheel (blocks `identity_pass.json` entirely).
- The seven `TODO-verify` items in `gpu/workflows/NOTES.md` — chiefly whether core
  `UNETLoader` accepts `z_image_turbo_int8_convrot.safetensors`, and `euler` vs the
  official template's `res_multistep` at 8 steps. All need a running GPU service.
- The Pixel-Art LoRA hash decision.
- Every pipeline stage body.
- Fixtures: a CC0 profile clip, the knight placeholder, a canned standing skeleton.
