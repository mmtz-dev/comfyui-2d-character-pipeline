# the 3D-intermediate spike feature — runbook

**The question this answers, and nothing else:** does routing the pose through a 3D rig hold
`strike` / `hit` / `death` better than the pure-2D chain, for a 64 px side-view sprite?

Two independent practitioners' pure-2D chains held up for walking and produced deformed attack,
jump and roll poses, and both moved to a 3D intermediate. That is the single biggest open risk in
the whole animation plan, and it is cheaper to answer now than after 100 animations are built the
wrong way.

**Time box: 2 sessions, ~2.5 h each.** Four checkpoints, each with an abort rule. Three of the
four are cheap and come before the expensive work. If the spike is going to fail, it should fail
in the first 40 minutes.

Research and licences behind every choice below: **[`RESEARCH.md`](RESEARCH.md)**.

> **Superseded for the anchor unit (2026-09-14).** The owner chose the CC0 **KayKit
> Adventurers Knight** as the anchor's motion source — *the rig motion input feature*,
> `anim rig` in the main pipeline (`../README.md`, "Rig input mode"). That removes the first
> five steps below (TripoSG, `glb_to_obj`, Mixamo login, auto-rig, `dae_to_joints`) and this
> directory's `render_openpose.py` from the critical path: `anim rig` renders its skeletons
> through `anim.posekit.draw`, the same rtmlib `draw_skeleton` this file wraps, so the "one
> renderer for both arms" rule is kept by construction. `knight_base/strike/hit/death` (plus
> `collapse`/`revive`) are staged in `../work/` up to the strip. **What is left of the spike:**
> one question — whether a rendered depth / normal channel from the KayKit mesh adds anything
> over skeleton-only conditioning. Everything below stays as the runbook for a rig that has to
> be *made* (TripoSG + auto-rig) rather than downloaded.

---

## The path

```
  reference sprite (RGBA, binary alpha, front-facing A-pose)
        │
        │  TripoSG  ── MIT, MIT weights, ~8 GB VRAM ── NATIVE daemon
        ▼
     knight.glb ──► glb_to_obj.py (trimesh, MIT, CPU) ──► knight.obj
        │
        │  Mixamo auto-rigger + 3 clips     ◄── OWNER, needs an Adobe login
        ▼
   slash.dae  hit.dae  death.dae
        │
        │  dae_to_joints.py   (vendored mixamotoopenpose, MIT, CPU)
        ▼
     joints.json   ── COCO-18 2D joint list, 18 points per frame
        │
        │  render_openpose.py  ◄── rtmlib's OWN draw_skeleton, the renderer the
        ▼                          2D chain feeds the ControlNet. Not negotiable.
   skeleton PNGs ──► make_strip.py ──► 1216×320 strip
        │
        ▼
   ┌─────────────────── FROM HERE ON, IDENTICAL TO THE 2D CHAIN ───────────────────┐
   │  gpu/workflows/identity_pass.json  →  anim post  →  64×64 frames + sidecar    │
   └──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   publish_compare.sh ──► http://192.168.1.158:8090/spike3d-compare.html
```

Everything runs on **Docker Desktop / CPU** except two GPU steps: TripoSG (`spike3d/compose.yml`)
and the identity pass (`../gpu/compose.yml`). They are ~8 GB and up to ~20 GB of VRAM, so on the
24 GB A5000 they **must not run at the same time** — TripoSG is a batch job that exits.

---

## Before session 1 — what only the owner can do

| # | Input | Why |
|---|---|---|
| 1 | **Start the native daemon**: `sudo systemctl start docker`, then bring ComfyUI up per [`../gpu/README.md`](../gpu/README.md) §2–4. | Needs a real TTY for the password. An agent cannot do it. |
| 2 | **An Adobe account for Mixamo** (free) at <https://www.mixamo.com/>. | The auto-rigger and the clip downloads are behind a login. |
| 3 | **Download 3 clips as Collada `.dae`** — *without skin*, and **do not** reduce keyframes in Mixamo's UI: a **sword slash**, a **hit reaction**, a **death/fall**. Drop them in `spike3d/mixamo/`. | These are the three actions the anchor gate judges. `.fbx` will not work — `mixamotoopenpose` parses Collada XML only. |
| 4 | **Decide the Mixamo question** in `RESEARCH.md` §2.2: Adobe's terms say Mixamo content may not be used to train ML models. We are conditioning at inference, not training, and we ship a pixel sprite — but it is your call, knowingly. If the answer is no, §2.3's CC0 route substitutes, at the cost of about a session for Blender retargeting. | It is cheaper to decide now than after the frames exist. |

**One prerequisite that is not an owner step:** the 2D chain's `strip`, `render`, `post` and
`publish` stages are still scaffolds (`anim/stages/*.py` print "not implemented" and exit 2).
This directory ships `make_strip.py` so the strip step is not blocked, but `render` and `post`
come from Feature 3. **If Feature 3's stages are not implemented yet, run sessions 1 and 2 up to
the strip and stop there** — a strip is enough to prove the pose holds, and the generation half
is shared with the 2D chain anyway.

---

## Session 1 — kill it cheaply, or earn session 2 (~2.5 h)

### Step 1 · Build the TripoSG image — start this first, it runs unattended (~15–25 min)

No GPU needed to build; `nvcc` comes from the base image and the CUDA extension (`diso`)
cross-compiles for sm_86.

```bash
cd /home/workstation/Repos/game-dev/framework-ladder
docker --context desktop-linux compose -f tools/anim-pipeline/spike3d/compose.yml build
```

Sanity-check the file first if you like — this parses it and starts nothing:

```bash
docker --context desktop-linux compose -f tools/anim-pipeline/spike3d/compose.yml config
```

### Step 2 · CHECKPOINT A — does the skeleton renderer work? (2 min, CPU)

Already passing as of 2026-09-13; re-run it, because everything downstream assumes it.

```bash
docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
  run --rm --entrypoint pytest anim spike3d/ -q
```

Expect **22 passed**.

Two things about that command, both of which will otherwise waste ten minutes:

- **`--entrypoint pytest` is required.** The image's `ENTRYPOINT` is `anim`, so
  `run --rm anim pytest …` would try to run a nonexistent `anim pytest` subcommand.
- **`spike3d/` is required.** `pyproject.toml` sets `testpaths = ["tests"]`, so a bare `pytest`
  collects the main suite only and reports nothing from here. An explicit path overrides it.

If the image predates `requirements-dev.txt` being wired into the `Dockerfile` (pytest lives
there now), either rebuild, or run the suite with no pytest at all — it has a standalone runner:

```bash
docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
  run --rm --entrypoint python anim spike3d/test_render_openpose.py
```

> **Abort rule:** if `test_render_is_exactly_rtmlibs_own_output` fails, rtmlib has changed and
> the two arms of the comparison are no longer pixel-comparable. Fix that before anything else.

### Step 3 · CHECKPOINT B — does the ControlNet respond to a rig-rendered skeleton at all? (20 min, GPU)

**This is the highest-value 20 minutes in the spike.** Hugging Face discussion #28 on the Z-Image
Fun ControlNet reports that manually created / 3D-pose-editor skeletons "aren't being recognized
at all" — one unanswered report, but it describes exactly this path (`RESEARCH.md` §3.1). If it
is real, the 3D intermediate cannot reach the model through this ControlNet and the spike is over
now, not in session 2.

```bash
# A synthetic joint list is already committed - no Mixamo, no mesh, no owner input needed.
docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
  run --rm --entrypoint python anim spike3d/render_openpose.py \
    spike3d/fixtures/strike_synthetic.json spike3d/work/checkB/cells \
    --width 304 --height 320

docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
  run --rm --entrypoint python anim spike3d/make_strip.py \
    spike3d/work/checkB/cells spike3d/work/checkB/strip.png     # needs 4 cells
```

Then push that strip and a DWPose-produced strip through `pose_pass.json` at the **same seed**
and compare. (Via `anim render --workflow pose_pass` once Feature 3 lands; by hand against
`POST /prompt` on `http://192.168.1.158:8188` until then — `../gpu/workflows/NOTES.md` has the
node names.)

> **Abort rule:** if the rig-rendered strip produces a figure unrelated to the skeleton while the
> DWPose strip conditions correctly, **stop**. Report discussion #28 as reproduced, and the answer
> to the spike is "not through this ControlNet". Total spend: ~40 minutes.

### Step 4 · CHECKPOINT C — does mixamotoopenpose read a 2026 Mixamo export? (10 min, CPU)

It was last touched **2024-11-10** and finds bones by hard-coded Collada node id
(`mixamorig_Hips`, `mixamorig_LeftForeArm`, …). If Adobe's exporter has been renamed since, it
returns nothing.

```bash
docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
  run --rm --entrypoint python anim spike3d/dae_to_joints.py \
    spike3d/mixamo/slash.dae spike3d/work/strike/joints.json \
    --rotation-y 90 --max-frames 4 --width 512 --height 512
```

Expect a frame count, then `joints -> …`. Render them and *look*:

```bash
docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
  run --rm --entrypoint python anim spike3d/render_openpose.py \
    spike3d/work/strike/joints.json spike3d/work/strike/preview --width 512 --height 512
```

> **Abort rule:** if the parse returns 0 frames or the rig is visibly scrambled, do **not** start
> debugging Collada. Fall back to `RESEARCH.md` §2.3 (CC0 clips + Blender retarget) and re-time-box
> — that is a different spike.

`--rotation-y` is the profile control. 90 or 270 depending on which way the clip faces; the art
bible wants the party facing **right**.

### Step 5 · Make the reference sprite TripoSG will actually cope with (30 min)

Do **not** feed TripoSG the 64 px side-view placeholder. Single-image 3D from a 64 px profile is
the weakest link in the whole path, and Mixamo's auto-rigger wants a humanoid in something near a
T- or A-pose with separable limbs.

Instead, generate a **front-facing A-pose** reference with the 2D chain's own
`gpu/workflows/base_sprite.json` driven by a canned front A-pose skeleton, at 1024 px, then cut a
**binary** alpha with the chain's MIT remover:

```bash
rembg i -m isnet-anime -dc -ppm knight_apose_raw.png spike3d/in/knight_apose.png
```

The RGBA + binary alpha is not cosmetic: it is what keeps the non-commercial BRIA background
remover out of TripoSG entirely (`RESEARCH.md` §1.4). `run_triposg.py` **refuses** any other input
rather than letting it silently fall back.

### Step 6 · Download the three clips (owner, 15 min)

Per the table above. `strike`, `hit`, `death` → `spike3d/mixamo/`.

**End of session 1.** You should have: a built image, checkpoints A/B/C green, a reference sprite,
and three `.dae` files.

---

## Session 2 — generate, compare, decide (~2.5 h)

### Step 7 · TripoSG: reference sprite → mesh (20 min, GPU, native daemon)

```bash
# Owner, real TTY - if it is not already running from session 1:
sudo systemctl start docker

# ComfyUI must NOT be holding VRAM while this runs.
docker --context docker-default compose -f tools/anim-pipeline/gpu/compose.yml down

docker --context docker-default compose -f tools/anim-pipeline/spike3d/compose.yml \
  run --rm triposg \
    --image-input /work/in/knight_apose.png \
    --output-path /work/out/knight.glb \
    --obj /work/out/knight.obj \
    --faces 20000 --seed 42
```

First run downloads ~8 GB of MIT weights to `/mnt/private/AI Data/models/3D Models/TripoSG`;
later runs reuse them (`--skip-download`). It prints the seed/steps/guidance line — **paste that
into `../DEVLOG.md`**, it is the reproducibility pin.

### Step 8 · CHECKPOINT D — does Mixamo's auto-rigger accept the mesh? (20 min, owner, web)

Upload `spike3d/out/knight.obj` at <https://www.mixamo.com/> → Upload Character. Place the
chin / wrist / elbow / knee / groin markers. If the mesh comes in lying down or absurdly scaled:

```bash
docker --context desktop-linux compose -f tools/anim-pipeline/spike3d/compose.yml \
  run --rm --entrypoint python triposg /opt/spike3d/glb_to_obj.py \
    /work/out/knight.glb /work/out/knight_fixed.obj --rotate-x 90 --center --scale 1.0
```

(That one has no GPU in it, so Docker Desktop is fine.)

Then apply each downloaded clip to the rigged character and re-export as **Collada `.dae`, without
skin**.

> **Abort rule:** two failed auto-rig attempts and the answer is "TripoSG's mesh is too soft for
> this rigger". Options, in order of cost: raise `--faces`; try **Stable3DGen / Hi3DGen** (MIT,
> commercially clean, sharper geometry — `RESEARCH.md` §1.2); or stop and report that the 3D route
> needs a better mesh model than fits in the box.

### Step 9 · Clips → skeleton strips (30 min, CPU)

For each of `strike`, `hit`, `death`:

```bash
A=strike   # then hit, then death
docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
  run --rm --entrypoint bash anim -lc "
    python spike3d/dae_to_joints.py spike3d/mixamo/\$A.dae spike3d/work/\$A/joints.json \
      --rotation-y 90 --max-frames 4 &&
    python spike3d/render_openpose.py spike3d/work/\$A/joints.json spike3d/work/\$A/cells \
      --width 304 --height 320 &&
    python spike3d/make_strip.py spike3d/work/\$A/cells spike3d/3d/\$A/skeleton_strip.png
  "
```

`--max-frames 4` matches the 4×1 strip and the art bible's 2–6 key poses. `render_openpose.py`
warns on stderr about any limb rtmlib will drop at the frame edge, and fits the pose into the
canvas by default to prevent it.

### Step 10 · The shared half — identity pass and post-chain (40 min, GPU)

TripoSG has exited, so bring ComfyUI back:

```bash
docker --context docker-default compose -f tools/anim-pipeline/gpu/compose.yml up -d
```

Then, per action, **the same commands the 2D chain uses** — same workflow, same seed, same
palette, same reference sprite:

```bash
docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
  run --rm anim render --workflow identity_pass --job "3d-$A" --seed 42
docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
  run --rm anim post --job "3d-$A"
```

Drop the results into `spike3d/3d/<action>/` as `frame_0.png` … `frame_3.png`, plus
`raw_strip.png`. Put the 2D chain's equivalents in `spike3d/2d/<action>/`.

> If Feature 3's `render`/`post` are still scaffolds, stop at step 9 and compare the **strips**.
> That still answers the pose question, which is the question.

### Step 11 · The deliverable (10 min)

```bash
tools/anim-pipeline/spike3d/publish_compare.sh --seed 42 --palette <palette-name> --frames 4
```

→ **<http://192.168.1.158:8090/spike3d-compare.html>** — open it on the Mac.

The page lays the knight's `strike`, `hit` and `death` side by side, 2D chain against 3D
intermediate, with the 64 px frames, an animated preview at adjustable fps, the conditioning
strips and the raw generations. Anything not generated yet shows as a labelled placeholder, so it
is worth opening early. The page also carries the judging criteria and the honest caveat below.

### Step 12 · Write it down

- `../DEVLOG.md` — what was run, the seeds, the model pins, what the comparison showed, **and the
  decision**. This is the gate's record.
- The work package in OpenProject (<http://192.168.1.158:8700>) — status lives there, never as a
  checkbox in a markdown file.
- `RESEARCH.md` §6 lists what is still UNVERIFIED; strike through whatever the spike settled.

---

## The honest limit — say this out loud at the gate

**This replaces "pose from your filmed video" with "pose from a rig."** The 2D chain's whole
premise is that the owner acts a motion out in profile and the pipeline follows *that*
performance. The 3D arm does not. It re-projects a canned Mixamo clip. It will probably hold
attack and death poses better — a rig cannot produce an impossible elbow — but what it produces
is someone else's motion, identical for every unit using the same clip, and not adjustable by
re-filming.

Closing that gap needs a video→3D-mocap step, which is **out of this spike's box** and whose
obvious candidates are licence-blocked for a commercial game: GVHMR and SMPLer-X are
non-commercial by their own licence text, and WHAM and TRAM are MIT repos that depend on the
non-commercial SMPL body model (`RESEARCH.md` §2.4). The clean route is
`rtmlib.Wholebody3d` — Apache-2.0, already installed in the CPU pipeline container, no SMPL.

**If the 3D route wins here, that follow-up should be a named work package before the route is
adopted, not after.**

---

## What is in this directory

| File | What it is |
|---|---|
| `RESEARCH.md` | Every model, licence, VRAM figure and renderer difference, checked against the primary source. Read §1.4, §2.2 and §3 before running anything. |
| `README.md` | This runbook. |
| `render_openpose.py` | **The load-bearing file.** A 2D joint list → a DWPose-style skeleton PNG drawn by *rtmlib's own* `draw_skeleton`, so the 3D arm's conditioning images are stylistically identical to the 2D arm's. |
| `test_render_openpose.py` | 22 tests, no GPU and no Mixamo needed. Includes the regression guard that our output is byte-identical to a direct rtmlib call. |
| `dae_to_joints.py` | Mixamo `.dae` → COCO-18 joints JSON. Loads the vendored `mixamotoopenpose` with its module-level CLI surgically removed (it calls `parse_args()` at import). |
| `make_strip.py` | N skeleton cells → one 1216×320 strip. A stand-in for `anim strip` while that is a scaffold; same geometry contract. |
| `run_triposg.py` | TripoSG inference with the non-commercial BRIA remover designed out. Refuses non-RGBA input rather than falling back. |
| `glb_to_obj.py` | GLB → OBJ for Mixamo, via `trimesh`. This is why no Blender image is needed. |
| `compose.yml` | TripoSG on the **native daemon** — `runtime: nvidia`, SELinux label, mirroring `../gpu/compose.yml`. Validated with `compose config`. |
| `Dockerfile` | The TripoSG image. Builds on Docker Desktop (compiling needs `nvcc`, not a GPU). Lints clean under `docker build --check`. |
| `compare.html` | The side-by-side deliverable page. Placeholders until frames exist. |
| `publish_compare.sh` | Copies frames + page to `~/.cache/claude/visuals/` and prints the LAN URL. Deletes before copying — the server pins a file after its first request. |
| `NOTICE-3d.md` | Third-party attribution carried by this spike. |
| `fixtures/` | A synthetic 3-frame joint list and rendered skeletons, so checkpoints A and B need no owner input. |
| `vendor/mixamotoopenpose/` | `Astropulse/mixamotoopenpose` @ `66ec14df`, MIT, with its LICENSE. Its **renderer is not used**. |
| `.gitignore` | Keeps meshes, weights and — importantly — Mixamo's raw `.dae`/`.fbx` out of git. Adobe's terms forbid redistributing them. |

## Rules this spike does not bend

- **Docker Desktop (`desktop-linux`) for everything** except the two GPU services. Never
  `docker context use`. The native daemon is targeted explicitly, per command, and only the owner
  starts it.
- **Commercially usable licences only.** No `briaai/RMBG-1.4`, no Hunyuan3D, no Sparc3D, no SMPL.
- **One renderer for both arms.** If the 2D chain's conditioning images change how they are drawn,
  `render_openpose.py` changes with them, or the comparison is void.
- **Status goes in OpenProject, explanation and history go in git.** Nothing here creates a
  checklist file.
