# DEVLOG — the pose-to-pixel animation pipeline feature

What changed and *why*. Status lives in OpenProject, never here.

---

## 2026-09-17 — the rig props feature: the posed mannequin and its kit, as depth

**What.** `anim rig` now writes `depth/%04d.png` beside the skeleton PNGs: the rig's skinned
mesh posed per frame, plus the unit's props as thin solids on the hand bones, z-buffered in the
same orthographic profile as the joints. One 8-bit grayscale frame per sample, near = 255,
far = 48, background 0, one range per clip. `anim base --rig` writes its one frame the same
way; `--no-depth` skips it. Filed as [#269](http://192.168.1.158:8700/work_packages/269); the
consumer is the clip render feature ([#268](http://192.168.1.158:8700/work_packages/268)),
built in parallel on its own branch against the contract in `rigkit`'s DEPTH section and
README "Rig input mode". Four Quaternius clips rendered, skeleton | depth side by side:
<http://192.168.1.158:8090/rig-props/>. Suite 257 passed / 1 skipped (was 238), +19.

**Why depth, when the outlines already fixed the recovery cell.** The outlines (2026-09-16,
below) work on the pose ControlNet because it reads white strokes as canny. The clip route's
video model (Wan 2.2 Fun-Control) is conditioned on pose *and* depth, and depth is the only
surveyed conditioning that says what a hand *holds* as a surface rather than a hint
(`ai-sprite-sheet-generators-research.md` §4.2). The owner chose the stronger route (Q15): make
which hand holds what a fact of the rig, carried by a channel the model was trained on.

**Skinning route: numpy, not Blender.** The CPU container has no `bpy` (cp311/cp313 wheels
only; the image is 3.12) and the GLB already carries everything glTF defines for a skinned
vertex — `JOINTS_0`, `WEIGHTS_0`, `inverseBindMatrices` — so linear-blend skinning is
`sum_k w_k · world[joint_k] @ ibm_k @ p`, one `einsum` per part. `load_rig` already evaluated
the joint world matrices per frame for the keypoints; `_mesh_parts` reads the primitives with
the same accessor reader (`ibm` transposed after the column-major read — the test pins the
bottom row to `[0 0 0 1]`). Proof it is right: in the bind pose the skinned vertices equal
`mesh_bounds`' extent to 1e-4, and a hand-built two-node part lands where glTF says.

**The z-buffer.** A software rasteriser over triangle *buckets*: triangles are grouped by
bounding-box size (2/4/8/16/32 px) and each bucket is one numpy pass — edge functions at pixel
centres, `np.maximum.at` for the nearest-wins scatter; the few larger triangles (a shield face,
a body triangle seen flat) take a per-triangle path. The mannequin's 13.7k triangles render
a 19-frame `Sword_Attack` in 0.33 s; a headless renderer would have cost a dependency for
nothing. Edges are inclusive so shared edges leave no pinholes.

**Conventions, and why.**

- *Viewer on the character's right.* The profile puts forward on +x and up on +y; a
  right-handed camera's axis is then forward × up — the character's right side. So the sword
  hand is the near hand, the shield the far one, which is how the published knight reads.
  `--mirror` flips projection and viewing side together. `Projection.near` / `depth_of()`.
- *Near = bright, far = 48, background 0.* MiDaS / ControlNet-depth convention. The far grey
  is not 0 so the far side of the body stays a surface, not background.
- *One range per clip.* Measured over every frame's vertices (mesh + props), recorded as
  `depth_range = [far, near]` in `job.json.motion_source`. A still body therefore does not
  flicker, and the range is what the clip actually needs: `Sword_Attack` spans [-1.31, 0.89]
  rig units because the sweep crosses the body, `Idle_Loop` only [-0.50, 0.34]. The price is
  that a still frame of a big-sweep clip uses the middle of the greys (119–192 on the rest
  frame of the strike) — visible in the evidence page; a per-*rig* range was considered and
  set aside until the clip render gate says whether contrast matters.
- *Props as extruded outlines.* `prop_polygons` already gives planar convex outlines in 3D;
  `extrude_polygon` makes each a closed solid along its plane normal, `PROP_SIZES[kind]
  ["thickness"]` × the standing height (sword 0.010, shield 0.018). So the depth and the
  drawn outline come from one geometry and cannot disagree. The shield edge-on is still a
  sliver (2026-09-16 "Open"), now a 4-px-thick one.
- *Only skinned primitives.* The KayKit Knight has its helmet, cape, four shields and three
  swords as static meshes on the head and hand-slot bones, all visible at once; rendering
  them would put every shield in the frame. Which kit a unit carries is `config.UNIT_PROPS`,
  and the generated props say it for every rig alike. So static meshes are skipped (the
  Knight renders as a bald body — it failed the gate anyway), `mesh_bounds` still reads
  them for the scale.

**Where it lives.** `rigkit`: `MeshPart`, `Rig.parts` / `.skinned`, `_mesh_parts`,
`skin_part`, `extrude_polygon`, `prop_meshes`, `posed_mesh`, `Projection.near` /
`depth_of`, `rasterize_depth`, `depth_to_grey`, `DepthClip`, `clip_depth`,
`DEPTH_FAR_GREY`. `stages/rig.py`: `write_depths`, `--no-depth`, `motion_source(...,
depth=)`. `stages/base.py`: `rig_payload` returns the depth frame too; `prepare` writes it.
`job.py`: `depth_dir`, `depths()`. Schema `anim.keypoints/1` untouched; the skeleton PNGs
are byte-identical to before.

**Open.** A per-rig depth range (above). Back-face culling would halve the triangle count
but is wrong for the open-sided prop solids as drawn, so it was not done. Whether pose and
depth are fed to Fun-Control stacked or as one control type is the clip render feature's
question, not this one's.

---

## 2026-09-16 — the skeleton props feature: the ControlNet is told which hand is the sword hand

**What.** `anim rig` now draws the unit's kit onto the skeleton. A knight's stick figure
carries a blade (with crossguard) off the right hand and a kite shield on the left forearm,
computed in 3D on the rig's `hand_r` / `lowerarm_l` bones, projected with the joints, and
stroked in white 2 px outlines by `posekit.draw`. `knight_strike` was re-run through
`rig → pick → strip → render → post` with it; **not published** — the frames are in
`work/knight_strike/post/` awaiting the owner's look. Before/after, the style trials and the
robustness renders: <http://192.168.1.158:8090/knight-two-swords/props.html>.

**Why this and not a prompt.** Yesterday's entry: 1 clean strip in 5 by prompting. The
prompt cannot name which stick is the sword arm; only the image can. Three prop styles
(single white line, white outline, filled white silhouette) were rendered on the published
picks at the published seed: **all three fixed the two-sword recovery cell** — the sword
went where the blade was drawn, the shield where the kite was. Outline kept the shield's
shape best and is what the union ControlNet already knows as canny, so that is what shipped.

**Where it lives.**

- `anim.rigkit`: `PROP_SIZES` (fractions of the standing height, so a chibi gets a chibi
  sword), `PROP_JOINT` (sword → right-wrist bone, shield → left-elbow bone), `PropFrame`
  on `RigFrame.props`, `prop_frame()` / `prop_polygons()`. One rest convention, stated in
  the module docstring: T/A-pose, palms down, so a gripped blade points FORWARD out of the
  fist, the guard runs along the fingers, and the back of the forearm — where the shield
  sits — faces UP. `Projection.anchor_for()` gives the joints' centre so the props are
  centred with the joints, not on their own extent (the test pins the guard's centre to
  the wrist keypoint in every frame).
- `keypoints.json`: per frame `props: [{kind, polygons: [[[x, y], …], …]}]` in cell
  pixels; `motion.props` lists the kinds. Schema still `anim.keypoints/1` — additive; a
  filmed clip never has it and the picker ignores it (drawing, not pose).
- `posekit.draw(props=…)`: `cv2.polylines` in `PROP_COLOUR` white after rtmlib's
  skeleton, same letterbox, same line width. `pose.render_skeletons` passes it through.
- `config.UNIT_PROPS` (`knight: sword, shield`) and `props_for_unit()`; `anim rig --props`
  and `anim base --props` override (`none` for a bare skeleton). `job.json.motion_source.
  props` records what was drawn; the previews get grey outlines.
- Tests: +15 (238 passed / 1 skipped, was 223). Geometry on the vendored Quaternius rig,
  renderer, stage defaults, the flag.

**The pick moved too, by hand.** With props the recovery cell was right in every trial,
but the follow-through cell (frame 6) still put the sword on the raised arm — that arm is
the *shield* arm, raised behind the head, and the real sword arm is crossed in front of
the chest at hip height, which a profile flattens to a stub. Nothing drawn on it survives
that reading. Frame 5 — the true contact, sword arm straight out, lost to 6 by 0.0007 of
deviation — renders clean at two seeds, so `picks.json` is hand-edited to `[0, 5, 10, 18]`
(rest, contact, recover, rest) with `timing {weights [2, 4, 3, 2], anchor 1, source
"hand"}`; `pick` will refuse to overwrite it without `--force`. The template-shaped
alternative `[0, 3, 5, 10]` (rest, wind-up, contact, recover) was tried and rejected: in
the wind-up the shield is edge-on, its outline a sliver, and the forward shield arm grew a
sword again.

**Rendered at the seed the other knight actions used** (`--seed 3139000483690707`), not
the derived one: the derived seed gave one sword per cell too but a different knight
(open-faced helm), and cross-action identity is prompt + seed only (2026-09-15, fix 4).

**Open.** An edge-on shield draws as a sliver; a minimum apparent width (turn the kite to
face the camera when its projected width drops under some fraction) would cover the
wind-up pose. The picker preferring 6 over 5 is a plateau tie, not a bug, but "the frame
after the speed peak is the contact" would have found 5 by itself. Neither built.

---

## 2026-09-16 — the two-sword strike: why, and why a prompt does not fix it

**What.** Owner feedback on the published `knight_strike`: the swing "spawns two swords".
Traced, then a prompt-only fix tried and rejected. No code changed; the published frames are
untouched. Sheets: <http://192.168.1.158:8090/knight-two-swords/> (cause) and
<http://192.168.1.158:8090/knight-two-swords/prompt-fix.html> (the experiment).

**Why it happens.** Three things stack.

1. **The skeleton has no props.** An OpenPose stick figure says where the wrists are, not
   what they hold; the prompt said "a sword and a kite shield" but not which hand, and the
   ControlNet's left/right colouring is a hint Z-Image mostly ignores — it already puts the
   sword in the *front* hand at rest, which on the rig is the left (shield) hand.
2. **Quaternius `Sword_Attack` is a horizontal slash with a 46° body twist, and the projection
   is a flat profile.** Read off the GLB: on the follow-through (frame 6) the sword hand has
   crossed to the character's left side at hip height (lateral +0.57 rig units) while the
   off-hand is raised high behind the head; the orthographic side view flattens that to
   "short stub forward-down, long arm raised behind", so the model paints the sword on the
   *shield* arm and the shield vanishes. On the recovery (frame 10) both arms are straight
   out, one forward one back — two straight arms plus "sword" in the prompt is a sword on
   each end.
3. The picks are `[0, 6, 10, 18]`: rest, follow-through, recovery, rest. The mid-swing
   contact (frame 5, deviation 0.8337) lost to frame 6 (0.8344) by 0.0007 and was deduped
   away, so the strip has no hit pose.

**The prompt experiment.** Same skeleton strip, strength 1.0, pose pass only. Two prompts
that spell out the hands ("sword in his right hand only … shield strapped to his left
forearm … left hand is empty and never holds a weapon … exactly one sword in each pose", and a
variant insisting the shield stays visible when that arm is raised or swung back), at the
original seed and two fresh ones. **1 clean strip out of 5.** Same seed → same composition,
two swords again; the one clean strip (A, seed 1001) got there by *not* following the
skeleton — the raised arm became a shield held up and cells 2 and 3 are the same lunge. No
negative prompt is available: the workflow runs Z-Image Turbo at cfg 1.0 with the negative
zeroed, so "no dual wield" cannot be said.

**Conclusion.** The prompt cannot name which stick is the sword arm; only the skeleton image
can. Next candidates, in order: draw the props onto the skeleton (a blade line off the right
wrist, a shield shape on the left forearm — the ControlNet is *told*, not asked); or pick a
clip that reads in profile (a vertical chop with no body twist) and pin frame 5 into the
picks. Neither is built; nothing in OpenProject was changed. Experiment renders in
`work/_exp/ks_prompt_*.png` (gitignored).

---

## 2026-09-15 — the strip timing feature: the sidecar carries the shape, `publish` writes it

**What.** Every published action now says how long each frame holds and which frame is the
contact. `picks.json` gains `timing` / `auto_timing` (`{"weights": [2, 3, 4, 3], "anchor": 2,
"source": "template:strike"}`), the sidecar gains `contract.timing` (weights + anchor only) and
`generation.timing_source`, and game-02's audit — which `publish` runs — checks the block against
`contract.frames`. The three knight jobs were re-published from `work/` with no render: same PNG
bytes, same inodes, sidecars now `strike [2, 3, 4, 3] / 2`, `hit [1, 3, 2, 2] / 0`,
`death [1, 2, 2, 4] / 0`. pytest: 208 → **223 passed**, 1 skipped.

**Why a template and not a derivation.** With 2–6 poses the roles are fixed by position — a
4-pose strike is rest / anticipation / contact / recover whatever the weapon — and the craft rule
is the same everywhere (hold the anticipation and the contact, rush the transit; Stardew's axe
150/40/40/170/75 ms, FE's `3-3-3-4 | 1-1-1-3-10`). So `config.ACTION_TIMING` is keyed by
`(action, n)` and tuned at the gate. Deriving the holds from `pick.signals` was considered: the
deviation peak is not reliably the contact pose (an overhead windup deviates more from rest than
the forward cut), so that stays UNVERIFIED and is not the default. `knight_hit` is a 4-frame
strip (picks `[0, 2, 3, 4]`), so the table has a `hit, 4` row beside the bible's `hit, 2`; `shove`
budgets 3 poses, so it has a `shove, 3` row (`2 4 3 / 1`) as well as the plan's `shove, 4`.

**`--timing clip`, the documented alternative.** Each weight is the clip frames from that pick to
the next, the last frame gets 1 (the tail after the last key pose is what the picker judged not
worth a cell), and the anchor still comes from the template row — the gaps say how long, not which
pose is the contact. Not the default because the half-second rest hold dominates: the knight's
Sword_Attack comes out `[6, 4, 8, 1]`. Clip mode with no template row is refused rather than
guessed.

**Precedence at publish, and why uniform writes nothing.** `publish.resolve_timing` takes a hand
edit (`timing` ≠ `auto_timing`) first, then whatever `pick` recorded, then the template row for
`(action, n)` when `picks.json` predates the feature (which is how the knight jobs got theirs),
else *uniform* — and uniform omits the block entirely, because the game already owns the even
weights and the default-anchor rule (last frame for an actor verb, frame 0 for a reaction) and a
second copy of that rule in the pipeline would drift. A hand `timing` whose length no longer
matches the frames (picks toggled in review after it was written) is dropped with a warning and
the template fills in, so the audit never sees a broken block.

**Weights, not seconds.** The game's `AttackChoreography` scales every duration by the speed
setting; seconds in a sidecar would fight it. The game turns the weights into per-frame durations
from its own windows — `game-02-tactical-rpg/DEVLOG.md` has that half, and the sequencing finding
(the even fit put the *recovered* pose on screen at impact) that made the anchor necessary.

**Also.** `review.py`'s empty record carries `timing: null` so a page opened before `pick` has the
keys; `toggle`/`set_picks` rewrite the record they read, so a hand shape survives a click (tested).
`pick.was_edited` now protects the timing pair the way it protects the picks pair. `anim pick` and
`anim publish` print the resolved shape, so `--dry-run` shows it before anything is written.

---

## 2026-09-15 — the anchor unit gate: first frames through the GPU, and what the ControlNet actually needed

**What.** The GPU image service came up for real (A5000, ComfyUI 0.35.0), the `nunchaku`
wheel went in, and the knight's `base`, `strike`, `hit`, `death` went through
`render → post → publish` into game-02: `knight.png` replaced the placeholder and three
frame groups (4 + 4 + 4) landed with sidecars, audit green (183 assets, 3 frame groups).
GDUnit4 1413/1413, this suite 208 passed / 1 skipped. Before/after in the running game and
every experiment below: <http://192.168.1.158:8090/anchor-gate/>.

**The gate, as judged from the frames.** Style passes with notes (bold outline, flat
shading, reads at 1×; dark steel sits close to the floor value). Proportions: **the
KayKit chibi rig fails** — the ControlNet does not read its collapsed profile skeleton as a
body and gives a different knight per cell — and **the Quaternius rig passes**: one knight,
four poses that follow the skeleton. So the published frames are Quaternius
(`Idle_Loop` #0, `Sword_Attack`, `Hit_Chest`, `Death01`). The owner's call on what that
means for the art bible (§3 names KayKit as the motion source, Quaternius as the A/B) and
for #267 is still open; nothing in the wiki or OpenProject was changed.

**Six workflow fixes, in the order they were found.** Each one was a real blocker, not a
tuning knob.

1. **`nunchaku` had no wheel for the image's torch.** The mmartial image installs the newest
   torch on every start (2.13.0); nunchaku publishes per torch minor and stops at 2.12
   (dev) / 2.11 (v1.2.1). `gpu/compose.yml` now pins `TORCH_LOCK=torch==2.11.0 …` and
   `gpu/userscripts/10-nunchaku.sh` (mounted at `/userscripts_dir`, run by the init inside
   the venv) installs `nunchaku-1.2.1+cu12.8torch2.11-cp312-cp312-linux_x86_64.whl`
   (sha256 `962eaabc…b32d4`) once and verifies on every later start. `piexif` for
   Image-Saver rides along. Nothing is installed by hand any more.
2. **The optional LoRA node is a 400, not a no-op.** NOTES.md said `load_lora` returns
   early at strength 0 — true, but ComfyUI validates `lora_name` against the file list
   before execution. `comfy.bypass()` removes the node unless `--lora-strength` is given.
3. **The chroma key leaked into every edge.** `post` flattened onto magenta *before* the
   downscale, so edge pixels blended with the key and Lospec 2000 has pinks to snap to
   (a 64-colour palette had hidden this). `imaging.bleed_edges()` pushes the sprite's own
   colours outward first. While there: pixeloe's `target_size` is not the output height
   (53 asked → 68 given → nearest-resampled), and pixelfixer "found" an 11.5×8.6 px grid
   on the base render and squeezed the knight to 23 px wide. **`--downscale box` is the
   default now** (pixeloe selectable), and a grid repair whose x/y pitches disagree is
   rejected. Comparison sheet on the gate page.
4. **The identity pass broke the strip.** Three images in (reference, skeleton, pose pass)
   → two zoomed figures of the wrong character. Two images (reference as `image1`, pose
   pass as `image2`, `VAEEncode` off `image2`) keep the layout; that is what `render.py`
   sends now. But even then Qwen-Image-Edit flattens the poses back toward the standing
   reference and turns a 2-cell strip into a close-up, so **every published strip is the
   pose pass** (`anim post --input render/pose_pass.png`). Cross-action identity is
   therefore prompt + rig + seed only, and it drifts (shield colour, armour tone). That is
   the gate's main open problem.
5. **The pose ControlNet was doing nothing — and it was the line width.** A lying-down
   death skeleton gave four standing knights at 0.7 and helmet close-ups at 1.5; the lite
   union model, the full union model, a single 512×768 cell — all the same. Canny worked,
   so the control path was fine. The model card's own pose conditions are DWPose renders
   with ~2 px sticks; ours had `line_width_for()` scaling limbs to 5–8 px "so the
   ControlNet can see them", and rtmlib draws each limb as an ellipse twice that. Fat
   blobs are not a pose. `posekit.LINE_WIDTH = 2`, and the same lunge that was ignored is
   nailed at strength 1.0. The full union model
   (`Union-2.1-2602-8steps`, 6.7 GB, 15 layers) was added to `models.lock` during the
   hunt and kept — the card says lite is "weaker control" — with the lite build left
   installed for A/B; `relock_models.py` now writes next to itself (run from the repo root
   it had written `./models.lock` and the downloader never saw the new row) and carries the
   KNOWN DISCREPANCY note so regeneration stops losing it.
6. **Sword tips cross cell boundaries.** A 4×1 strip has no wall between cells; the
   neighbour's blade lands as a floating shard. `imaging.keep_main_blobs()` drops islands
   under 10 % of the largest before the trim. And `rembg isnet-anime` returns an empty mask
   for a lying body, so `knight_death` is matted with `--matte chroma`.

**Also learned.** A 608×320 two-cell strip is unreliable (the model drew three figures);
`knight_hit` was re-picked at 4 with `anim pick -n 4 --force`. The base prompt's "four
heads tall" does not pull the Quaternius skeleton's ~5 heads down at strength 1.0 — the
skeleton wins, as §3 of the art bible predicted the other way round. `post` raising
mid-strip leaves finished frames behind that `publish` will happily ship as a shorter
group — it did once (a 2-frame death) before the matte fix; worth a guard.

**Verified.** TODO-verify #1, #2, #3, #5, #6, #7 in `gpu/workflows/NOTES.md` are settled
and moved; #4 (euler vs res_multistep) is the one left. Tests added: the LoRA bypass, the
two-image identity graph, the constant line width; the letterbox test now subtracts the
fixed stroke. `work/_exp/` holds every experiment render (gitignored).

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

---

## 2026-09-16 — the two open gate problems get a route: three features filed, no code

**What.** After the two-sword trace above, the owner asked how the hosted AI sprite-sheet tools
(pixelanimation.io) animate, and whether we can. Research and the grilled decisions are in
`game_core/discovery/game02/ai-sprite-sheet-generators-research.md`; the work is filed as
[#268 the clip render feature](http://192.168.1.158:8700/work_packages/268),
[#269 the rig props feature](http://192.168.1.158:8700/work_packages/269) and
[#270 the part palette lock feature](http://192.168.1.158:8700/work_packages/270). Nothing in this
repo changed; the published knight frames stay.

**Why this shape.** The hosted tools are closed models (Retro Diffusion behind pixelanimation.io;
PixelLab's own skeleton-window model), so "do it ourselves" means the approach, not the weights.
For **identity drift**, the survey's usable mechanism is a reference-image-conditioned video model —
Wan 2.2 Fun-Control (VACE as fallback; the forked pipeline already runs it at 22.5 GB) fed the
`base` pose-pass render and a control video rendered from every frame of the rig clip; `pick` then
chooses the 2–6 key poses from the generated frames, so `review`/`post`/`publish`, the sidecar and
the timing block are untouched, and the pose pass survives as the seconds-long preview. For
**props**, the DEVLOG entry above listed "draw the props onto the skeleton" first; the owner chose
the stronger route instead — prop meshes parented to the Quaternius hand bones in `anim rig`,
carried by a **depth** channel, because a depth frame from a rig that holds the sword is the only
conditioning any surveyed model has for "what the hand holds", and it is a fact of the rig rather
than a request in a prompt. First-hand: `rigs/quaternius/UAL1_Standard.glb` is a single `Mannequin`
mesh, so the props are added, not read. The **part palette lock** is the safety net the owner asked
for — SAM3 part masks at 512 px, each part snapped to the base sprite's colours — and its per-part
pixel counts are the measurable half of the second gate: one knight per frame; part colours within
tolerance across strike/hit/death; one sword, in the sword hand, in every strike frame. Budget: up
to ~5 min per action for a final render. The KayKit/Quaternius ruling waits for that gate.
