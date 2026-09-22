# DEVLOG — the pose-to-pixel animation pipeline feature

What changed and *why*. Status lives in OpenProject, never here.

---

## 2026-09-21 — the turnaround feature: eight idle stills, the identity gate, face points hidden on the far side

**What.** [#275](http://192.168.1.158:8700/work_packages/275), the fourth feature of the sprite
studio epic. `anim turnaround <unit> [--dirs s0,s45,…] [--elevation side|top] [--seed N]
[--force] [--page-only]` renders, from the accepted base, one standing still per direction —
the eight side facings for v1 — each as an ordinary `rig --yaw Y → clip → pick → post --input
clip --parts` job named `<unit>_idle_<dir>` (s0 included, so the Board's idle row, the matrix and
the export find them), writes **that direction's** `post/parts.json`, records seconds per stage,
VRAM, seed and metrics per direction in `work/<unit>_turnaround.json`, and builds the gate page
(`work/_turnaround/<unit>/index.html`, copied to
<http://192.168.1.158:8090/sprite-studio/turnaround/>): the eight stills ×4 in compass order, the
raw clip frame and the skeleton + depth control under each, the part table per direction, a
turntable, the reference, and the **identity table** — numbers, no verdict. New:
`anim/stages/turnaround.py`, `tests/test_turnaround.py` (16 cases), `GET /turnaround/{unit}`, a
`control` frame set on `GET /jobs/{job}/frames`, `rigkit.hidden_face_points`,
`config.clip_prompt(action, direction)` + the `TURNAROUND_*` block, and the Board shows the eight
stills under the gates with the question and the accept button (`board.ts`: `turnaroundTiles`,
`turnaroundText`; two vitest cases). The stage is base-gated and never waits for the turnaround
gate — it makes the stills that gate is about. pytest: 559 → **587 passed**, 4 skipped; vitest:
39 → **41 passed**. Two `test_parts_post.py` cases were pinned to a throwaway `assets_dir`: they
had been reading the owner's *real* published `knight.json` sidecar (republished today at 09:53
with the concept pick's palettes) and failing on the palette outside the fixture's — a
test-isolation bug that surfaced on the mount, not a code change.

**The route, as built — three decisions with their alternatives.**

- *Five frames, not one.* Wan's latent takes 4 frames per slot, so the shortest clip is 4n+1: n=0
  is a 1-frame video, legal but with nothing for the video model's temporal attention to denoise
  against; n=1 (5 frames, `TURNAROUND_CLIP_FRAMES`) costs the same second and gives it a window.
  `rig` samples frame 0 of the UAL's `Idle_Loop` five times at t=0 (`times=[0.0]*5`), so the
  control video is one held pose and `clip` pads nothing — no change to `clip`. `Idle_Loop` over
  `Sword_Idle` because it is the neutral stand the concept pick shows (the sheets are in
  `work/_exp/ta_sheets/`); `Sword_Idle` is a combat crouch with the blade raised.
- *The middle frame is the still, not the closest to the rig.* The five control frames are one
  pose, so "closest to the rig pose" would mean running DWPose on five near-identical generated
  frames per direction to pick among them — a model load for no information. The middle frame
  (`TURNAROUND_STILL_INDEX = 2`) is the one farthest from both temporal edges: the first frame of
  a Wan clip is encoded by the VAE on its own and leans hardest on the reference, the last is
  where padding would sit. `pick` is run for real (so `picks.json` has every field the Board, the
  export and `post` read) and then hand-set to `[2]`, the way the review page hand-sets one.
- *Lock to the base, then record what each direction shows.* `post --parts` on an action job
  locks every part to the base's palettes; the direction's `parts.json` is the colours each part
  shows *after* that lock (a locked part: members of the base's palette; a part the base has no
  palette for — the sword and shield, which SAM did not find on the front-view concept pick —
  keeps its snapped colours and the file says `locked: false`). So a back-view clip in the matrix
  can `--parts-from work/knight_idle_s90/post/parts.json` and lock the cloak to what the back
  view showed rather than to a front view that never saw it. The alternative — extract per
  direction with no lock — would let every facing drift its own way and leave nothing for the
  gate to measure.

**The face on the far side — decided, with the alternative.** The camera direction feature left
the OpenPose face points at confidence 1.0 in every cell, nose included on the back view.
`rigkit.hidden_face_points` now reads the *posed* head's forward vector against the view's near
axis: once the head has turned past the profile away from the camera by more than half a
direction step (`FACE_HIDE_TURN = sin 22.5°`, so a breathing head at s0 never flickers between
frames), the nose and both eyes get confidence 0.0 (listed in the frame's `hidden`), and the ear
on the far side by the same margin goes with them; from straight behind both ears stay. 0.0 is
what a filmed back view reads as too, so the renderer, the picker and the part prompts already
treat the point as absent — no second code path. s0 / s180 and every front cell are byte-for-byte
what they were (tested on the Knight for eight cells); s45 / s90 / s135 lose the face. The
alternative is a normal per point (nose forward, eyes a forward/lateral blend, ears lateral)
against the near axis: more honest geometry, but it also hides the far eye and the far ear at the
*exact* profile, which is what every published s0 job was rendered with and what DWPose keeps on
a profile shot; it is one threshold away in the same function if the gate wants it.

**The real run — the knight, eight side facings, seed 1625235789813781** (derived from the unit,
`Idle_Loop` and the base render's sha256, so a new pick is a new seed), reference
`work/knight_base/render/concept_pick.png` (the owner's concept pick 01 — `concept --pick` had
already run today at 09:45), part lock to the base's own `parts.json` (helmet / arms / torso /
legs; sword and shield have no palette there), matte `rembg`, ComfyUI 0.35.0 on the A5000:

| dir | still (w×h px) | rig s | GPU s (service) | post s | total s | face hidden |
|---|---|---|---|---|---|---|
| s0 | 35×53 | 4.3 | 153.9 (Wan paging in from cold) | 26.4 (SAM + rembg loading) | 185.2 | — |
| s45 | 39×53 | 4.2 | 9.0 | 2.2 | 16.7 | nose, eyes, left ear |
| s90 | 38×53 | 4.4 | 8.7 | 2.4 | 17.0 | nose, eyes |
| s135 | 32×53 | 4.2 | 8.9 | 2.7 | 17.1 | nose, eyes, right ear |
| s180 | 35×53 | 4.3 | 9.0 | 2.1 | 16.7 | — |
| s225 | 30×53 | 4.1 | 9.1 | 2.3 | 16.6 | — |
| s270 | 32×53 | 4.2 | 9.2 | 2.1 | 16.6 | — |
| s315 | 27×53 | 4.4 | 9.2 | 2.0 | 16.7 | — |

VRAM after every GPU stage: 17.6 GB (Wan resident). Warm, a facing is **~17 s wall, ~9 s of
GPU** — well inside the minute; nothing came near the 300 s budget. The identity table:

| part | s0 | s45 | s90 | s135 | s180 | s225 | s270 | s315 | spread | found | drift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sword | 60 | 35 | 73 | 37 | 34 | 44 | 61 | 17 | 56 | 8/8 | 0 (free) |
| shield | 99 | not found | 15 | 55 | 2 | 44 | 15 | 1 | 98 | 7/8 | 0 (free) |
| helmet | 74 | 81 | 62 | not found | 45 | 72 | 33 | 42 | 48 | 7/8 | 290 |
| arms | 50 | 121 | 93 | 47 | 169 | 123 | 119 | 84 | 122 | 8/8 | 522 |
| torso | 0 | 35 | 175 | 187 | 25 | 15 | 39 | 109 | 187 | 8/8 | 364 |
| legs | 363 | 119 | 479 | 325 | 442 | 210 | 487 | 237 | 368 | 8/8 | 1043 |

Silhouette: height 53 in every direction (the scale ladder holds), width 27–39 (spread 12),
opaque pixels 721–1019, colours 37–59.

**What I saw, per direction, honestly.** The scale, the dark plate with gold trim, the cloak
and a sword in the right hand hold in all eight; the **helmet does not**. s0 and s270 are the
knight: closed great helm, dark armour, the blade forward. s45, s135, s180, s225 and s315 have a
**bare human head** — black hair from behind at s45 / s135, a face with glasses at s180, an
orange-haired man at s225, a bald face at s315 — on the knight's body; the helmet row still
says "found" there because SAM's helmet prompt sits on the head and locks whatever is on it to
the helmet's greys (drift 290 px is that lock at work). s90 is the knight from behind with the
helm, **plus a second, white-shirted figure half behind it** — the one still that is not one
figure. So the risk named in `sprite-studio.md` §5 is real and it is not the back view alone:
the reference's helmet survives the front and the profile the reference was rendered from, and
is dropped whenever the head is turned, and where the face points are drawn the model paints a
face on it. Two readings for the owner: the depth channel's bare-headed Quaternius mannequin
(no helmet mesh) outweighs the reference at gain 0.35 on the turned heads — the KayKit chibi's
big helmet in the bake-off had the opposite problem; and the 768×768 concept pick letterboxed
into the 512×768 cell arrives at two thirds of its size. The sword reads as a plain stick
(orange at s0, red at s45) rather than a blade — the prop outline is followed, the reference's
sword is not. The gate stays **pending**: the page is at
<http://192.168.1.158:8090/sprite-studio/turnaround/>, the Board shows the same eight stills
under the gates, and `anim curate knight --accept-gate turnaround` (or the button) is the
owner's call. Levers that exist today without new weights, none pulled: `--depth-gain` lower
or `--control pose` for the turned heads; a helmet outline as a prop on the rig's head bone
(the skeleton props feature has the machinery); a per-direction `--seed` re-roll from the Board;
and the design doc's named fallback, a per-character LoRA.

**Left to the owner.** The gate. And, after any `concept --pick`, the one line:
`anim turnaround knight --force`.

---

## 2026-09-21 — the spritesheet export feature: sheets + atlas + SpriteFrames, publish becomes the game-02 adapter

**What.** [#283](http://192.168.1.158:8700/work_packages/283), the export decision from
`sprite-studio.md` §2: a **generic contract plus a game-02 adapter**. `anim export <unit>`
finds every `<unit>_<action>[_<dir>]` job with `post/` frames and writes
`work/_export/<unit>/`: `sheets/<unit>_<dir>.png` (one row per action, frames left to right,
transparent padding; `--sheet packed` gives one row-major sheet instead), `atlas.json`
(`anim.atlas/1`: unit, frame size, fps, palette name, per `<action>_<dir>` the sheet, the
frame rects, phase tag + ticks + `duration_ms` per frame, loop, pivot, anchor, the source job,
picks, timing source, seed and clip, and the pipeline commit), `<unit>.tres` (a Godot 4
`SpriteFrames`: one `AtlasTexture` per frame, one animation per cell, `speed = fps`, per-frame
`duration` in ticks, loop from the plan) and `preview.html`. `publish` is now documented and
named as **the game-02 adapter**: `--adapter game02` (or `export.adapter` in
`work/project.json`, new in the settings store) runs it per exported job and then the audit
once, and what it writes is byte-for-byte what `anim publish` alone writes — the tests compare
the sidecar and the frames against a second, direct publish. New API route
`GET /export/{unit}`; `POST /jobs {"stage": "export"}` needed nothing beyond registering the
stage (the run manager builds argv from the stage's own parser). pytest: 514 → **538 passed**,
4 skipped.

**Why the pieces are shaped this way.**

- *Timing in ticks, not milliseconds.* `picks.json` holds weights (ticks at the plan's fps) and
  `durations_ms` is derived from them; the export goes through the same
  `publish.resolve_timing` the sidecar uses, keeps the ticks, and derives the milliseconds
  itself (`round(ticks × 1000 / fps)`). Godot's per-frame `duration` is *relative* to
  `speed`, so with `speed = fps` the ticks go in unchanged — `[2, 3, 4, 3]` is
  `167 / 250 / 333 / 250 ms` in the atlas and `2.0 / 3.0 / 4.0 / 3.0` at 12 fps in the `.tres`,
  with no rounding drift between the two. The alternative, `speed = 1.0` and seconds per frame
  (what game-02's `apply_schedule` does at runtime), would have put `0.083` in the file.
- *Per-direction sheets by default, packed on request.* A row per action per direction is what
  a person reads and what a hand-rolled importer slices; the packed sheet is smaller and needs
  the atlas. The atlas says where every frame is in both modes, so the choice is only about
  the PNG a human opens. Rows are alphabetical — a canonical action order would be another
  table to keep true, and the atlas already names each row.
- *`work/_export/<unit>/`, not `work/<unit>_export/`.* Underscore directories are the existing
  "not a job" convention (`_runs`, `_exp`); `anim.api.jobs.valid_name` refuses them, so the API
  can never list an export as a job or a job scanner mistake it for `<unit>_<action>`. One
  place for every unit's export is also what a future `GET /export` (all units) would list.
  `--out` puts it anywhere.
- *Which jobs, by name.* The manifest alone would have made `knight_strike_phases` and the
  `knightq_*` gate jobs (slug `knight`, action `strike`) candidates for the same cell. The rule
  is the name `anim rig` makes — `<unit>_<action>` for `s0`, `<unit>_<action>_<dir>` otherwise
  — and the manifest must agree; anything else under the prefix is skipped with a reason.
- *Missing cells are warnings.* The matrix is the studio's five actions × the eight side
  cells (top-down is a later milestone), plus whatever else the unit has (`hit`, `death`).
  Today the knight has three `s0` cells, 53 missing-cell warnings and 3 skips; an error would have made v1
  unexportable.
- *The `.tres` is read back before the stage returns.* `parse_tres` is a small reader of the
  text-resource forms a `SpriteFrames` uses (also of the editor's own spelling, with `uid`s),
  and `verify_tres` checks animations, frame counts, durations, loop, speed, regions and sheet
  paths against the atlas. A `.tres` the game cannot load is the one thing this stage must
  never leave behind, and no test needs Godot for it.
- *Relative `ext_resource` paths.* Godot resolves a relative path against the resource's own
  directory, so the export directory drops into any project as one unit — no `res://` path
  baked in at export time.

**Verified.** `anim export knight` on the real `work/`: three animations (`death_s0`, `hit_s0`,
`strike_s0`; `collapse` and `revive` have no `post/`, `knight_strike_phases` skipped as
expected), sheet 256×192, atlas commit `4af2071`, palette Lospec 2000. **Godot 4.6.3 headless**
(`--headless --path <tmp project> --import`, then a `SceneTree` script that `load()`s the
`.tres`): loads without errors, three animations, four frames each, `speed 12`, durations
`[2, 4, 3, 2]` / `[1, 3, 2, 2]` / `[1, 2, 2, 4]`, every frame an `AtlasTexture` with a 64×64
region. The page is at `http://192.168.1.158:8090/sprite-studio/export/` (sheets at ×4 with
the rects, each animation playing at its durations; a headless Chromium screenshot confirmed
it renders).

**Not done.** No `GET /export/{unit}/files/...` route to serve the sheet over the API (the
files are on the `work/` mount; the board can read them there or a route can follow). The
`preview.html` is copied to the visuals dir by hand, as the direction sheet and the bake-off
pages are.

---

## 2026-09-21 — the curation board feature: accept / reject / regenerate per frame and per cell, gates as a project setting

**What.** [#281](http://192.168.1.158:8700/work_packages/281), the eighth feature of the sprite
studio epic — where the human says yes or no. A **Board** tab in the editor's panel: one unit's
grid, rows the five studio actions, columns the directions the unit has a job in, each cell that
job's frame strip at ×2 with a mark per frame (click cycles accept → reject → clear) and accept /
reject / regenerate for the cell; a header with the two gates, "waiting on you: …" and the mode
selector. One module, `anim/curation.py`, is the whole rule set — the record
`work/<unit>_curation.json`, the gate check, the board state — and both the API (`GET/PUT
/curation/…`, a `423 Locked` on `POST /jobs`) and a new `anim curate <unit>` CLI write through
it. `concept --pick` now accepts the base gate. `web/src/board.ts` is the pure half (grid,
reducer, what-is-waiting) and `board-view.ts` the DOM; `panel.ts` grew a Pipeline / Board tab bar
that widens the panel for the grid. pytest: 514 → **535 passed**, 4 skipped; vitest: 26 →
**39 passed**. Smoked in a headless Chromium (Playwright 1.63 from a scratch dir, the cached
chromium 1243) against the worktree's API on `:8721` with copies of `knight_strike` /
`knightq_strike` / `knight_concept`: boot, Board tab, the strip from `post/`, frame 5 rejected
and gone from `picks.json`, cell accepted and locked, mode → every, turnaround gate toggled,
regenerate → confirm → `render --seed N knight_strike` queued through the run manager (it then
hung on ComfyUI, which is off — the container was stopped). Screenshots stayed in the scratchpad.

**Curation is a project setting, and the gate is one function.** `sprite-studio.md` §2 says
so — default gates after the base and after the turnaround with a batch review at the end; also
every step, or end only — and the settings store already had `curation: gates | every | end`
waiting to be read. `curation.gate_blocks(unit, stage, action, direction)` is the one question
asked in three places: the API's `POST /jobs` (→ `423 Locked` with the reason, `gate: true`),
the board's regenerate (same, before the run manager's `409`), and `cli.main` for every stage
with `--force` unset (prints the sentence, exits 2) — one hook in `main` rather than one per
stage, so the action matrix is gated the day it lands (`matrix` is already in the set). In
`gates` mode `clip` / `render` / `bakeoff` / `run` wait for the base gate and any cell outside
`s0` for the turnaround gate; `every` adds "every earlier cell accepted"; `end` never blocks.
CPU stages never wait — they are what makes the frames there are to judge, and gating `pick`
on the previous cell would block the review it feeds. **`every` orders cells direction-major**
(every action of `s0`, then `s45`, …): the side view is what game-02 ships and the top-down
pitch is a later milestone, so finishing `s0` before opening a second direction is the order the
owner actually works in. The alternative, action-major (idle in all 16 directions first), is
how a turnaround is drawn and would fit the turnaround feature better; it would also leave
`s0`'s attack waiting behind fifteen idles nobody has asked for yet. Documented, not chosen.

**Legacy units are not gated.** Every knight job predates the board. A rule that refused
`anim render knight_strike` until someone accepted a base nobody was ever asked about would be
a regression dressed as a feature, so a unit with **no** `work/<unit>_curation.json` answers
"gates accepted" and nothing blocks; only `concept --pick` writes a real record (base accepted,
turnaround **pending** — the turnaround is drawn from the base, so a new base reopens it), and a
board edit on a legacy unit creates its record with both gates accepted and `"origin":
"legacy"`. The one thing that changes for the knight today is that `anim concept --unit knight
--pick NN` would, from then on, hold a `knight_walk_s45` clip until the turnaround gate is
accepted — which is the point.

**One flat file per unit, beside the jobs.** `work/<unit>_curation.json`, not
`work/<unit>/curation.json`: a unit has no directory of its own — its jobs are siblings named
`<unit>_<verb>[_<dir>]` — and a `work/<unit>/` would list on `GET /jobs` as a job with no
manifest. The flat file sits where `project.json` and `gate_parts.json` already sit and `ls
work/*_curation.json` is the list of units under review. The alternative was a block inside
each job's `job.json`; that scatters one decision (the gates) over N files and puts a cell's
seed next to the seed the stage recorded, two writers of one fact.

**A cell is a job; a rejected frame leaves `picks.json`.** The grid is `(studio action ×
direction)` and a cell is the job `rig` names for it (`knight_strike` is `attack/s0`; the
studio name and the pipeline verb are one key, `config.studio_action`). Two jobs on one cell —
`knight_strike` and `knight_strike_phases` — resolve to the record's own choice, then the
canonical name, then the newest, so a reload never swaps a cell's job. Frame marks are keyed by
**source frame number**, the number `picks.json` uses, and a reject is written through
`ReviewApp.set_picks` — the review page's own write, byte for byte the same file (a test twins
the job and compares bytes). The rejected frame stays on the strip dimmed with a ✗ so it can be
taken back: `post/` still holds its frame, and the strip's `post` mapping (post frame *k* is pick
*k*) is kept whole by counting the rejects back in; only when `post/` no longer matches the
picks at all (a re-pick since it ran) does the strip fall back to the source frames.
Accepting a cell locks its frames (`409` on a mark, the buttons grey) — a decision that can be
edited by accident is not a decision. Regenerate records the new seed and `regenerated_from`
(the seed the stage recorded, so a re-run reproduces either) and only **after** the run manager
accepted the run, so a refused request (`409` busy, `423` gated) records nothing; the CLI's
`--regenerate` records and prints the `anim clip … --seed N` line instead, because a shell
command that starts a 24 GB model load as a side effect is the thing the GPU rule exists to
prevent.

**Not in this feature.** The turnaround gate has no stage that accepts it yet (the turnaround
feature will); `anim curate` cannot start the GPU; the board shows one unit at a time; there is
no per-frame regenerate (a frame comes from a clip or a strip, so the cell is the unit of
re-rendering); `settings.curation` stays the string it already was rather than a
`{"mode": …}` object — the store, the tests and the README all had the string, and a second
shape for one setting is a second thing to keep true.

---

## 2026-09-21 — the web pixel editor feature: Piskel as the shell, the pipeline as a panel

**What.** [#280](http://192.168.1.158:8700/work_packages/280), the sixth feature of the sprite
studio epic — the studio's shell. `GET /` on `:8720` now serves a real pixel editor: Piskel
(layers, onion skin, timeline, palettes, the drawing tools, GIF / sheet export) in a Vite +
TypeScript shell under `web/`, with the pipeline as a panel down the left. The panel's v1: pick
a job, open its frames (the `post` 64×64 set, else `clip`, else `frames`) as one animation at
the job's fps with the keyframe phase on each timeline tile; edit; save every frame back
through a new `PUT /jobs/{job}/frames/{set}/{n}.png`; start `pick --phases` / `post` at a
click and `clip` / `bakeoff` / `concept` behind a "this starts a GPU job" confirm; follow the
run's log and progress live over the SSE route; and put the project palette from
`GET /settings` into Piskel's palette list. New in the API: the PUT route (`jobs.save_frame`:
same size as the file it replaces or `409`, alpha rounded to the contract, written in place),
and the `post` set listed as `<stem>_<n>.png` in numeric order — `job.post_frames()` sorts by
name, which put `_10` before `_2` and listed `palette.png` and the `_hd` / `_labels`
diagnostics as frames; the API's set is now what `publish` ships, so post frame *k* is pick
*k*. `config.WEB_DIST` resolves to the checkout's `web/dist`, then the image's
`/opt/anim/web/dist`. The Dockerfile grew a `node:22-slim` stage (`npm ci && npm run build`)
whose `dist/` is copied into the runtime image; `compose.yml` is unchanged. pytest: 483 →
**488 passed**, 4 skipped; `web/`: vitest, **26 passed**. The `anim-api` container was rebuilt
and recreated; `curl http://192.168.1.158:8720/` returns the editor. Verified in a headless
Chromium against the served build: boot, open a job, tags on the tiles, save (a pixel drawn on
frame 0 landed in `post/`, the other eleven re-encoded pixel-identical; the files were then
restored from a copy), `pick --phases` started and followed to `done`.

**Vendoring: source, not the bundle.** Piskel upstream (pinned at `bef52beb`, 2026-09-20)
dropped Grunt this year; its build is a Vite config with three home-made plugins that join
the script list, join the style list and resolve the `@@include` partials of `index.html`, plus
a spritesmith step for the icon sheet. That is small enough to do ourselves:
`web/piskel-plugin.mjs` (stdlib only) does those three joins into `public/piskel/` and drops
the body markup into our `index.html` at build time, so the shell's `devDependencies` are
vite, typescript and vitest and nothing of upstream's (no spritesmith, no terser, no karma, no
puppeteer downloading a browser inside the Docker build). The icon sheet is the one generated
thing committed into the vendored tree (`icons.png`, `icons@2x.png`, `icons.css`, made once with
upstream's `scripts/sprite.js`) — `CHANGES.md` says so and how to regenerate. The alternative
was vendoring upstream's built `dest/prod` bundle: one minified file, no build step, but then
every fix to Piskel is a patch against minified output and the fork's changes cannot be read as
a diff. Source keeps the fork a diff: one edit so far (`FramesListController`, a `tile-tag`
element per timeline tile fed by `pskl.app.frameTagProvider`), listed in
`web/vendor/piskel/CHANGES.md`. Note the brief called Piskel MIT; it is **Apache-2.0**, which
is what `NOTICE` and the vendored `LICENSE` say.

**Frames → Piskel: one layer, frame k = PNG k.** A job's set becomes one `pskl.model.Piskel`
with one layer, one Piskel frame per PNG, at `job.json`'s fps — so Piskel's timeline, onion
skin and preview all mean what the pipeline means by a frame, and the tile number is the
frame number `picks.json` and `publish` use. The alternative, one *layer* per pipeline frame,
would give per-frame opacity and blend but turn Piskel's animation into a stack, and every
tool that iterates frames (onion skin, export) would be wrong. Save flattens whatever layers
the owner added on top (`LayerUtils.flattenFrameAt` with opacity), rounds alpha to 0/255 and
zeroes transparent pixels (`web/src/pixels.ts`, the same rule as `imaging.binarise_alpha`,
which the API applies again — so the browser and the server agree on the bytes), and PUTs
each frame at the index it came from. The count and the size must match the set: a frame
added in the editor has no file to overwrite, and a resized frame is a different asset, so
both refuse with a sentence rather than write half a set.

**Where the phase tag comes from.** `/jobs/{job}/picks` tags *source* frames (`frames[i].phase`);
the `post` set is one frame per pick in ascending pick order (what `strip` and `post` build),
so post frame *k* carries the tag of source frame `picks[k]`; `clip` is 1:1 with the source.
Fewer or more post frames than picks — a partial `post`, a stale `picks.json` — means no tags
rather than wrong ones (`web/src/frames.ts`, tested).

**Layout without touching Piskel's CSS.** Piskel lays itself out `position: fixed` against
the viewport. The panel is a fixed 330 px column and one custom property (`--studio-w`)
shifts `#main-wrapper`, the tool column and the cheat-sheet link right by that much; Piskel's
`DrawingController` measures `#main-wrapper` and re-lays out through its own `ResizeObserver`,
so folding the panel (Alt+P) reflows the canvas with no fork change.

**What the panel cannot do yet.** Add or remove frames and save that back; open `render/`
or `strip/` (strips, not frames — slicing needs `layout.json`); import or fetch a palette
(the palette feature, #282 — the panel reads `settings.palette.colours` when that lands and says
"no project palette" until then); export to Godot or publish from the panel (the export
feature); edit picks (the click-to-pick page is still `/jobs/{job}/review`); anything
top-down. Piskel's own Save / Export drawer still writes to the browser or a download, not to
`work/`.

---

## 2026-09-21 — the palette feature: one project palette from Lospec, unit sub-palettes, snapped at export

**What.** `anim palette` — [#282](http://192.168.1.158:8700/work_packages/282), the sprite
studio's palette row (sprite-studio.md §2: *one project Lospec palette, per-character sub-picks,
snapped at export; file import and fetch by slug*). New: `anim/palettekit.py` (parse `.hex` /
`.gpl` / one-row `.png`, the `.gpl` writer, the Lospec fetch, the four-file install, the per-unit
files) and `anim/stages/palette.py` (`--lospec <slug>` / `--file <path>` / `--list` / `--into`;
`--unit <slug>` with `--colours` / `--from-sprite` / `--clear`). `config.LOSPEC_URL`
(`ANIM_LOSPEC_URL`). `post` gains the unit sub-palette (`palettekit.unit_colours`) and `--no-snap`;
`publish` refuses a `--no-snap` job; `imaging.verify_frame(palette=None)` skips the membership
check for that preview only. The API: `GET /settings` shows `palette` expanded (`{path, name,
colours, source}`) and `PUT` takes the path or that object back; `GET /palette`, `POST /palette`
(a CPU run through the run manager, `--into` the settings' path), `GET /palette/units[/{unit}]`.
pytest: 483 → **509 passed**, 4 skipped; the two opt-in `ANIM_POST_REGRESSION=1` tests
(`knight_hit` with rembg, `knight_death` with chroma, byte-compared with the published frames)
pass too, and a scratch run of the fixture job on this branch and on the branch point gives the
same frame hashes with both Grim32 and the installed palette. Nothing re-rendered, nothing
published, OpenProject and `TODO.md` untouched.

**The real fetch.** `anim palette --lospec lospec-2000` (the slug in the installed `.gpl`'s
`Name:` header, which `install_palette.py` wrote from a pasted hex list on 2026-09-14) into a
scratch dir: 182 colours, `palette.hex` and `palette.png` **byte-identical** to
`game-02-tactical-rpg/assets_src/`'s, the colour rows of `palette.gpl` identical, and the `.gpl`
header comments different — Lospec's JSON gives this palette an empty `author`, where the
hand-installed file says "Lospec community", and the pasted-list origin line is now a
`# Source: lospec lospec-2000` line and the page URL. So the installed palette *is* the Lospec
one, and the endpoint is `https://lospec.com/palette-list/<slug>.json` →
`{"name", "author", "colors": ["rrggbb", …]}` (a wrong slug is a 404 with
`{"error": "palette not found"}`; checked 2026-09-21). The README's "Resurrect 64, provisional"
paragraph was two palettes out of date — the source (the file's header) wins, and the README now
says Lospec 2000.

**Where a sub-palette lives: `palette_units/<slug>.json` beside the project palette, not
`parts.json`.** The brief offered `work/<unit>_base/post/parts.json` or a unit file. `parts.json`
is *derived* data — the colours SAM found inside each part of one base render, regenerated
whenever the base is re-posted, and per part, not per unit. A sub-palette is a *decision* —
"the knight is these 56 of the 182" — made before any base exists and outliving every re-render,
so it belongs with the other art decisions that already sit in `assets_src/` (the palette,
`timing.json`), one JSON per unit so two API runs on two units cannot race on one file. `post`
applies it in both places that matter: the unit-level `-remap` snaps to the subset, and the part
palettes are narrowed to it (an outsider moves to its nearest member in Oklab, the lock's own
metric — `post.narrow_part_palettes`), so the lock can only ever snap to the unit's own colours.
The alternative — writing the subset into `parts.json` as a synthetic part — would have made the
lock the only consumer and left the unit-level remap on the whole palette, which is the step that
actually decides most pixels. A unit with no file takes the pre-feature path exactly.

**Remap before the lock, unchanged.** `post` already ran `-remap` before the part lock; that order
stays. The lock's candidates are then members of whatever the remap used, and with a sub-palette
the remap already excludes the borrowed browns, so the lock has less to correct. The other order
(lock the HD cell, then remap) would have snapped part pixels to colours the remap could then move
again. Every existing job with no sub-palette and no flag produces the same bytes it did — that
is what the hash comparison against the branch point and the published-frame regression pin.

**Snap at export was already true; now it is checked.** `test_generation_never_quantises_only_post_does`
asserts no stage before `post` names `-remap` or `.quantize(`. `--no-snap` exists because the
studio's curation board will want to show the raw render beside the snapped one; the preview is
matted, downscaled and recanvassed so it is the same shape as the real thing, `report.json` and
`job.json` say `snap: false`, and `publish` refuses it rather than trusting the owner to remember.
`--no-snap --parts` is refused outright: the lock is a snap.

**Idempotent by leaving the date out.** `install_palette.py` stamps its `.gpl` with the day; this
writer does not, so `--lospec x` twice is a no-op in git and the test compares bytes. The one
moving field is `palette.json`'s `fetched_at`, which is data the API shows, not a diff on the
palette. The `.hex` and `.png` are byte-compatible with `install_palette.py`'s so either tool can
install and the other reads the result.

**`GET /settings` changes shape for `palette`.** It was a path string; the brief asks for
`{path, name, colours, source}`, so the store keeps the path and `load()` expands it from the
palette files. A client that GETs, edits another field and PUTs the whole document back would
otherwise have sent the object where a string was expected, so `validate()` accepts both (an
object needs a `path`). The web editor branch reads `settings.palette` — this is the one shape
change it has to absorb at merge.

**Not done.** No `anim palette` in `anim run` (the palette is a project setting, not a per-clip
step); no re-post of the existing jobs to a sub-palette (none is set — `--from-sprite` on the
published knight gives 56 colours, recorded here for the second gate to decide on); the
spritesheet export ([#283](http://192.168.1.158:8700/work_packages/283)) is where the snap gets
its generic-contract home.

---

## 2026-09-21 — the keyframe phase feature: pick against start → wind-up → action → wind-down → end → loop

**What.** `anim pick --phases` — [#276](http://192.168.1.158:8700/work_packages/276), the fifth
feature of the sprite studio epic. Every studio clip is now picked against one shape, whichever
route the frames came from (a Wan clip, a pose-pass strip, a filmed take): the owner's five
phases plus the loop flag, with the per-action budget from sprite-studio.md §2 (cap 12; idle 6 ·
walk 8 · run 8 · jump 10 · attack 12). New in `config.py`: `ACTION_PHASES` (budget, loop, `split`,
`span` per action), `STUDIO_ACTIONS` (the studio's five names against the pipeline's verbs —
`attack ↔ strike` — and each vendored rig's clip), `ANIM_TIMING` / `phases_for_action` (the
per-project `timing.json`), `ANIM_LOOP_TOL` / `ANIM_LOOP_PIXEL_TOL`. New in `pick.py`:
`PhasePlan`, `allocate`, `phase_plan`, `phase_positions` / `phase_frames` / `phase_instants`,
`phase_tags`, `loop_closure`, `phase_timing`, `phased_picks`; `pick()` grows `phases=` and a
library-only `spread=`. `picks.json` gains a `phases` block (plan, one tag per pick, the
anticipation index, the closure) and its `timing` becomes the phase shape; `publish` copies
`fps` / `loop` / `phases` / `durations_ms` into `contract.timing` beside the weights and the
anchor (game-02's audit checks those two and ignores the rest — run for real in the tests); the
review page shows the tag on each card the picker chose and the loop's verdict in the header.
`bakeoff.py`'s hand-spread of 12 is gone: both routes call `pick(..., phases=True,
spread="even")`, and `PHASES` / `PHASE_KEYS` / `BUDGET` are the attack row of the template.
pytest: 433 → **452 passed**, 4 skipped. Nothing re-rendered, nothing published, OpenProject
untouched.

**The table is `ACTION_PHASES`, not `ACTION_TIMING`.** The brief named the new table
`config.ACTION_TIMING`; that name already belongs to the strip timing feature's hold weights,
keyed by `(action, n)`, and is referenced by `pick`, `publish`, the README, this log and the
tests. Renaming it would have changed a shipped feature's vocabulary to make room for a new one,
so the phase table got its own name and the strip table kept its.

**Ease as density, not as a curve.** Each row carries two five-way shares: `split`, how the
*budget* is divided between the phases, and `span`, how the *source clip's time* is. The split
turns into whole frames by largest remainder (every phase at least one; ties to the earlier
phase), and the span cuts the clip into five windows; a phase's frames are spread evenly inside
its own window (the start pinned to the clip's first frame, a one-shot's end to its last). Frames
per second of clip therefore differs per phase — attack samples its action window at four frames
for a quarter of the clip and its rest windows at one or two frames for a fifth each — and that
non-uniform density *is* the ease in and out. The alternative was one easing curve (a smoothstep
inverse warping a linspace) with no per-phase knob: simpler, but it hides the split the owner
named and cannot say "the wind-up gets three frames". Two shares per row is the honest knob,
and `timing.json` can retune either. Attack comes out `1 / 3 / 4 / 2 / 2`, jump `1 / 3 / 3 / 2 / 1`,
walk and run `1 / 2 / 2 / 2 / 1`, idle `1 / 1 / 2 / 1 / 1`.

**Anticipation is one frame of the budget.** A one-shot allocates its split over budget − 1 and
gives the reserved frame to the wind-up, so the beat before the action is never a single cell
(`phases.anticipation` is that pick's index — attack's is 3, the last wind-up cell before the
first action cell). A loop has no beat to anticipate and gets none.

**Loops stop short, then get checked.** A cycle's `end` window is `[a, 1)`, its last pick centred
inside it rather than pinned to the clip's last frame — the frame after `end` is `start` again,
and a cycle whose last cell repeats its first hitches on the wrap. The loop-closure check then
measures the last pick's distance back to the first: `posekit.pose_distance` (mean joint
displacement in torso lengths) against `ANIM_LOOP_TOL`, set to **0.12 — `PICK_DEDUP`, the
picker's own "same pose" threshold** — and, when either frame has no pose, the mean absolute
pixel difference of the two frames against `ANIM_LOOP_PIXEL_TOL` 0.08; neither available, the
verdict is `null` and says so. The result is recorded and shown, never acted on: a cycle that
does not close is the owner's call (re-time, or pick a different clip).

**The hold rule.** A clip lasts budget / 12 s whatever the source. `phase_timing` gives every
frame a weight in 12 fps ticks (1 = 83 ms) and puts the anchor on the first action frame. When a
phase came back with fewer frames than it was allotted — a 9-frame clip against attack's 12
returns all nine, tagged by the window each falls in — each of its frames weighs allotted /
picked ticks, so the phase still lasts its share of the second and the clip its budget; a phase
with no frames at all lends its share to the phase before it (the one after, for `start`). With
every phase full, every weight is 1. `durations_ms` is the same shape in milliseconds for the
exporter and the editor; the game keeps reading weights (`PackedFloat32Array`, so a fractional
hold is fine).

**Where `timing.json` lives.** `ANIM_TIMING`, default `/assets_src/timing.json` — beside the
palette in the customer game's `assets_src/`, the one other art decision that already lives
there. Rows merge over the defaults (`{"attack": {"budget": 10}}` keeps attack's split), a
`units` block overrides per unit, and keys may be either spelling. The alternative was a
`timing.json` in the unit's job dir (`work/<unit>_base/`), which the brief's hint pointed at:
not chosen because `work/` is regenerable scratch and gitignored, and a project setting that
vanishes with `rm -rf work/` is not a project setting. Per-unit config in this repo is otherwise
a dict keyed by slug in `config.py` (`UNIT_HEIGHT`, `UNIT_PROPS`, `UNIT_PARTS`); the `units`
block is that pattern in a file.

**The default is on for `attack`, `run`, `jump` — not for `strike`, `idle`, `walk`.** The brief
said "default on for studio actions, off keeps today's behaviour byte-for-byte for existing
jobs", and those two clauses collide on `idle` and `walk`: both are art-bible verbs with 2- and
6-pose budgets and existing jobs (and `test_rig.py`'s anchor list). So `config.phases_default`
is on only for a studio name the art bible does not budget; `strike` / `idle` / `walk` stay on
the key-pose picker until `--phases` asks, and the action matrix (#278) passes `phases=True` for
every cell. A golden `tests/fixtures/picks_head_447ad25.json` — `picks.json` as HEAD wrote it
for the canned fixture — pins the byte-for-byte promise; `new_picks` adds the `phases` key only
when there is a block, so an old record keeps its exact keys.

**The bake-off keeps its numbers.** `even_spread` and `phase_times` are now
`pick.phase_frames(..., spread="even")` and `pick.phase_instants(..., spread="even")` — linspace
and the old `_even_spread` to the frame (tested against it for 12–60 frames) — so a cell re-run
would render the same instants the owner judged, and `PHASES` / `PHASE_KEYS` come from the
attack plan (`1 / 3 / 4 / 2 / 2` by column, keys `0, 2, 6, 9, 11`; the old hand tuple was
`1 / 3 / 3 / 3 / 2`, keys `0, 2, 5, 8, 11`). `bakeoff.json` on disk carries its own labels, so the
published page is unchanged until `--page-only` is run. The shaped spread for the routes is one
argument away; it was not switched on here because nothing was to be re-rendered.

**Not done.** `STUDIO_ACTIONS["jump"]["clips"]["quaternius"]` is `None`: the UAL ships the jump
as `Jump_Start` / `Jump_Loop` / `Jump_Land` and no whole clip, so the studio's jump on that rig
waits for a clip joiner. The `run` and `jump` verbs have no `ACTION_TIMING` hold row and no
`CLIP_ACTION_PHRASES` entry yet; the matrix feature adds the prompt phrases when it names its
cells.

---

## 2026-09-21 — the pipeline API feature: FastAPI on :8720 over work/, the review server retired

**What.** [#279](http://192.168.1.158:8700/work_packages/279), the sprite studio's eighth
feature: `anim/api/` — a FastAPI service (`anim-api` in `compose.yml`, port **8720**, same
image and mounts as the CLI, `uvicorn --factory anim.api.app:create_app`) that reads and
writes the same `work/<job>/` files the CLI does, so the editor (#280) and the curation board
(#281) have something to talk to. Endpoints: `GET /jobs`, `GET /jobs/{job}`, `/frames`,
`/frames/{set}/{n}.png`, `GET|PUT /jobs/{job}/picks`, `POST /jobs/{job}/regenerate`,
`POST /jobs` (run a stage), `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/events` (SSE),
`GET|PUT /settings`, `GET /health`, and `/` + `/jobs/{job}/review` as pages. The stdlib
`http.server` that `anim review` started on `:8710` is gone, and so is the `review` compose
service; `anim review <job>` now prints the API URL. `ReviewApp` stays — the API drives it.
New: `anim/api/{app,jobs,runs,settings,pages}.py`, `requirements-api.txt` (fastapi, uvicorn;
own Docker layer after the dev deps for the same reason pytest has one), `tests/test_api.py`
(TestClient, no socket; 26 functions, 36 cases with the parametrised ones; test_review.py lost its 7 HTTP-skin cases and gained 2 — 464 passed / 4 skipped, was 433 / 4). `httpx` in
`requirements-dev.txt` for the client. Config:
`REVIEW_PORT`/`REVIEW_BIND` → `API_PORT`/`API_BIND` + `API_CPU_SLOTS` + `WEB_DIST`.
Brought up once on Docker Desktop (`docker compose -p anim-api-wt up -d anim-api`), `/jobs`
and `/` curled from the host, taken down.

**A run is the CLI in a subprocess, not a thread calling `run_stage`.** `POST /jobs
{"stage": "pick", "args": {"job": "knight_strike", "n": 3}}` becomes
`python -m anim pick -n 3 knight_strike` with `ANIM_WORK_DIR` pointed at the API's `work/`,
stdout streamed line by line into `work/_runs/<id>.log`. The argv is built from the stage's
own argparse definition (`anim.api.runs.argv_for`: positionals in parser order, a true
`store_true` adds its flag, a false `store_false` adds its — `audit: false` → `--no-audit`)
and then parsed by the real parser, so a bad request is a `400` in argparse's words. That is
the "layer, never a copy" rule made concrete: the API knows no stage's arguments; a stage
that gains a flag gains it in the API with no change. The alternative — a thread calling
`run_stage(args)` in-process — was rejected because the stages *print* their progress (two
threads share one stdout), a stage that dies hard (an ONNX segfault, a torch OOM) would take
the API down with it, and a GPU stage holds a `requests` call open for minutes that a thread
cannot be told to abandon. The cost is a Python start per run (importing `anim.cli` pulls
every stage: torch, onnxruntime — a few seconds); acceptable against stages that take
seconds to minutes.

**The GPU rule** (sprite-studio.md §2 "Build base": explicit, one at a time). `base`,
`render`, `clip`, `concept`, `bakeoff` and `run` (it renders when it can) are the GPU set;
a second one while one is active is `409` with the running run's id. Nothing in the API
polls ComfyUI — even `/health` only probes it on `?probe=1`. CPU stages run concurrently up
to `ANIM_API_CPU_SLOTS` (2: `pose` and `post` each want every core), then `409` as well;
a queue would have hidden which request was waiting on what.

**SSE hand-rolled, not `sse-starlette`.** The stream is one sync generator over the
manager's `Condition`: `log` per line, `progress` when the percent changes, one `done`, a
`: keepalive` comment every 15 s of silence; a client that connects after the run ended
gets the whole log and then `done`, so a reloaded page misses nothing. `sse-starlette` would
have given the same three frames plus a dependency to licence-check and a ping thread of its
own; the whole thing is 30 lines here. **Percent** comes from lines the stages already print
— `[n/m] …` (what `anim run` prints per step) and `progress: 42%` — and nothing else, so a
line like "strength 80%" cannot move the bar. Most stages report no percent today; the
progress event is there for `run`, and for the action matrix (#278) when it prints its
cells.

**Runs persist as files, jobs stay files.** `work/_runs/<id>.json` + `.log`, written in
place like everything else; a run still `running` in a file at startup belonged to a process
that no longer exists and is reported `lost`, and does not hold the GPU. A directory is a
job only if it has a `job.json` — `_runs`, `_exp`, `sheets` and `gate_parts.json` are not
listed. Stage status is *derived* from the files (`picks.json` exists → `pick` ran), never
from a status field, so a job re-run from the CLI or hand-edited reads the same over the
API as on disk.

**Picks are byte-compatible with the old page** because they go through the same
`ReviewApp.set_picks`; `tests/test_api.py` drives a copied job through `ReviewApp` directly
and compares `picks.json` bytes with what the `PUT` wrote. The frame URLs index the job's
own listing (`/jobs/x/frames/frames/3.png` is `job.frames()[3]`), never a filename from the
URL, which is the old `image()` guard kept.

**Settings are just the store.** `work/project.json`: `palette`, `frame_budgets`
(`{action: 1..12}`, the bible's cap), `curation` (`gates | every | end`), `frame_size`. GET
merges the file over the defaults; PUT validates and writes the whole document. Nothing
reads it yet — the palette (#282), keyframe phase (#276), curation (#281) and export (#283)
features will; the CLI's `config.py` values stay in force until they do.

**Undone / for the next features.** No cancel endpoint (a run is a process; `kill` is a
one-liner when #281 needs it). No auth — LAN only, like everything on this host. The
fallback pages are the smallest client of the API and will be replaced by `web/dist/`
(#280); the static mount is already there.

---

## 2026-09-20 — the frame route bake-off feature: two routes × two rigs on the knight's attack, the page is the owner's

**What.** `anim bakeoff --unit knight --action attack` — [#274](http://192.168.1.158:8700/work_packages/274),
the third feature of the sprite studio epic. One character × one action through both candidate
routes for the in-between frames, on both candidate rigs, rendered for real on the A5000:
`V-kaykit`, `V-quaternius` (rig → Wan video), `K-kaykit`, `K-quaternius` (keyframe stills →
tween), one seed (`3139000483690707`, the second gate's), 12 frames each, `post --parts` locked to
`work/gate_parts.json`. Page: <http://192.168.1.158:8090/sprite-studio/bake-off/> (four rows, 64 px
post frames ×3 with the raw pre-post cell under each, a 12 fps CSS preview per row, a part-mask
toggle, the metric table); the same table in markdown with the owner's four questions:
`docs/bake-off-2026-09-20.md` (the committed copy of `work/bakeoff_knight_attack/report.md`). New:
`anim/stages/bakeoff.py`, `tests/test_bakeoff.py` (12 cases; 433 passed / 4 skipped, was 421 / 4), an optional `times=` on
`rig.rig()` (the CLI has no flag for it), README row + section. **No verdict here; the ruling is
the owner's.** Nothing published, nothing in OpenProject changed, no new weights.

**The two routes, as built.**

- *Route V* is `anim clip` as it stands: rig at 12 fps at the 512×768 clip cell, skeleton over
  full-body depth at gain 0.35 (the second gate's variant (b)), Wan 2.2 Fun-Control 5B against
  `work/knight_base/render/base_sprite.png`, then the clip *downsampled* to the budget — `pick`
  hand-set to an even spread of 12 over the 14 (KayKit) / 19 (Quaternius) rig frames, the
  picker's own 12 kept beside it in `picks.json` — and `post --input clip --parts`.
- *Route K*'s in-betweening is the **rig-space tween**: no RIFE / FILM node is pinned in
  `gpu/custom_nodes.txt` and this feature adds no weights, so the rig is sampled at exactly 12
  instants across the clip (`phase_times`, `rig(..., times=)`) and every frame — the five phase
  poses and the seven between — is a still of the rig through the strip route's pose pass. The
  pixel-space crossfade (the honest baseline) is on the page as a display-only strip under each
  K row: the five phase keys held, RGBA blends between, no palette snap, no metrics — it costs
  nothing and shows what a tween without the rig means. Three decisions inside K, each with the
  alternative:
  1. **Three 4-cell strips, one seed, not one 12-cell strip.** The 12-cell strip (3648×320) was
     tried first: the pose pass returned eight black silhouettes and four cells of noise
     (`work/_exp/bo_k12q`, image on the page's server as `exp-12cell-pose-pass.png`). The 4-cell
     strip is the unit the anchor gate verified. Per-frame single cells were the other option —
     untested on this model, and a 304×320 cell is near-square. The three strips are joined into
     one `render/pose_pass.png` (and `identity_pass.png`) for the page.
  2. **The anchor gate's strength and prompt.** `pose_pass.json`'s own prompt names no unit and
     returned black silhouettes on the first K run; the gate's strips were rendered at strength
     1.0 with the knight's description on the command line (read back from the `prompt` chunk of
     `work/knight_strike/render/pose_pass.png`). Both now live in the stage (`K_STRENGTH`,
     `POSE_PROMPTS`) — the first time that prompt is in code.
  3. **The frames go through `post --input clip`, not the strip slicer.** `rembg` on the whole
     3648 px strip returned an empty matte for one cell (isnet-anime works at ~1024 px; twelve
     cells wide leaves each knight ~85 px tall, and `post` refuses a transparent cell). So the 12
     pose-pass cells are also cut into `clip/frames/%04d.png`, 1:1 with the rig frames exactly as
     `anim clip` lays them out, and matted one by one — the very same `post` call Route V's
     frames take. `report.json.route` therefore says `clip` for both. The identity pass runs too
     (it is what `render` does) and its strip is on the page; the frames are the pose pass, as
     every published strip has been (2026-09-15, finding 4) — so **the base render is not in
     Route K's frames at all**: its identity is prompt + skeleton + seed, per strip.

**The metric table** (`docs/bake-off-2026-09-20.md`; the page has it per frame):

| cell | frames | one sword, sword hand | one figure | drift px, all parts, 12 frames | part-not-found cells | GPU s (service clock) | rig + post s | VRAM |
|---|---|---|---|---|---|---|---|---|
| V-kaykit | 12/12 | 10/12 | 12/12 | 3595 | 2 | 17.3 | 5.8 + 44.8 | 17.6 GB |
| V-quaternius | 12/12 | 11/12 | 12/12 | 2384 | 1 | 172.9 | 5.5 + 46.6 | 17.6 GB |
| K-kaykit | 12/12 | 7/12 | 12/12 | 4125 | 19 | 624.1 | 8.3 + 40.3 | 18.8 GB |
| K-quaternius | 12/12 | 11/12 | 12/12 | 5277 | 2 | 640.0 | 7.8 + 18.8 | 18.8 GB |

*Seconds.* V-quaternius's 172.9 s is one Wan job with the model paging in from disk (the second
gate measured 19.9 s warm); V-kaykit ran warm at 17.3 s. Each K cell is six `/prompt` jobs —
pose pass ~130 s and identity pass ~76 s each by the service's own stamps — because the two
passes alternate Z-Image and Qwen-Image-Edit and every swap reloads: **both K cells blew the
~5-minute budget** (10.4 / 10.7 min of GPU) and the run went on because the cost is the swap,
not the sampling. The first K-quaternius attempt was also killed mid-`post` by the caller's
timeout, which is why the stage now reuses a chunk's render on a re-run of a cell that is not
yet done (`_chunk_rendered`) and reads its seconds back from `/history`. VRAM is what
`/system_stats` reported after the GPU stage (Wan resident: 17.6 GB; Z-Image + Qwen: 18.8 GB).

*The two counts are proxies*, computed on the HD cells from the same SAM 2.1 masks the lock used
(`RecordingSegmenter` keeps what `post` asked and got): "one sword in the sword hand" = one sword
blob (islands under 15 % of the largest ignored) whose nearest pixel to the right wrist is within
12 % of the cell height and no nearer the left wrist; "one figure" = one matte blob and at most
one helmet mask. The second gate counted these by eye; the page says the owner's eye overrules.

**What I saw, per cell, honestly.**

- *V-quaternius* is the second gate's (b) render at 12 frames instead of 4: silver plate, dark
  cloak, red kite shield on the rest frames; frames 3–8 (the swing) go dark and painterly, the
  cloak a black mass, and the drift column says so (torso 0–128, shield 0–184 px/frame). Frame 10
  is the same two-blade recovery the gate saw. It plays as one knight swinging.
- *V-kaykit* is one figure in every frame and the sword stays in the near hand — but the model
  drew the **KayKit mannequin's proportions with a big blue helmet**, not the base render's knight:
  the chibi body in the depth channel won over the reference (helmet drift 2290 px is the lock
  fighting the blue). The shield is hidden behind the body in most frames. It is a consistent
  chibi; it is not the knight.
- *K-quaternius* follows the rig pose for pose and is the crispest pixel art of the four, but it
  is **three knights, not one**: each 4-cell strip renders its own (frames 8–11 a smaller,
  cloakless one), the sword sits in the far hand on frames 0, 1 and 3 and the shield changes from
  red-with-cross to a round buckler mid-swing. The drift total is the highest (5277) because the
  lock is pulling three renders toward one base.
- *K-kaykit* reproduces the anchor gate's finding: the chibi skeleton is barely read as a pose —
  the twelve cells are mostly front-facing shield-forward knights, the sword in the shield hand
  on frames 0–1 and 10–11 (7/12), 19 part-not-found cells (torso hidden behind the shield in 8
  frames, shield not found in 7). Three different knights across the three strips.

**Left to the owner** (the four questions in `docs/bake-off-2026-09-20.md`): the route; the rig;
whether the mid-swing drift needs a second reference at the contact pose or a Wan-side LoRA (the
two Wan 2.2 pixel LoRAs found target the 14B i2v — `docs/research/2026-09-20-pixel-art-loras.md`);
and whether exaggeration goes on the rig (#277 as planned) or into `post`. Also noted for the
next feature, not decided here: the phase labels on the page are the 12-budget template by
column (#276 makes them per-action); Route K without the identity pass would be three pose-pass
jobs and no model swap, the cheap variant this page did not measure; and `gate_parts.json` is
still the stopgap palette — the base re-render with props is the proper fix (2026-09-17).

---

## 2026-09-20 — the character concept feature: `anim concept`, Z-Image-Turbo + the pixel-art LoRA, sketch in, candidates out

**What.** `anim concept --unit knight --prompt "…" [--sketch png] [--count N] [--seed S]
[--existing [png]]` writes N candidate front-facing base characters to
`work/<unit>_concept/candidates/NN.png` with a `job.json` that records the prompt, one seed
per candidate, the LoRA and its strength, the sketch, the workflow hash and the locked models;
`anim concept --unit knight --pick NN` (or `--pick existing`) makes one of them the unit's base
render — [#273](http://192.168.1.158:8700/work_packages/273), the second feature of the sprite
studio epic. New: `anim/stages/concept.py`, `gpu/workflows/concept_pass.json`,
`tests/test_concept.py` (25 cases), a `--only` filter on `gpu/download_models.sh`,
`comfy.prune_orphans`, `Job.candidates_dir`, the `CONCEPT_*` block in `anim/config.py`.
421 passed / 4 skipped (was 393 / 4 before this and the direction sheet's file).

**The model side.** No skeleton: `base` is the pose stack because a standing sprite has to match
the rig's proportions, and a concept is earlier than that — there is only a description. So the
pass is Z-Image-Turbo text-to-image on the same int8 convrot UNET the pose stack loads, with
**`tarn59/pixel_art_style_lora_z_image_turbo`** through `LoraLoaderModelOnly` (the node
`identity_pass.json` already used; three keys, read from `/object_info` on 0.35.0). The LoRA is
the owner's pick from `docs/research/2026-09-20-pixel-art-loras.md`; it went into
`gpu/models.lock` through `relock_models.py` (repo, revision `0a5092d1`, 170,128,328 bytes, sha256
`09b1b45c…`, Apache-2.0, ungated — both HF endpoints agreed, the licence gate passed, the
parked `prithivMLmods` row is untouched) and onto disk through `download_models.sh --only
pixel_art_style` — the new flag, so one 162 MB row does not mean re-verifying 58 GiB or
re-fetching the Qwen LoRA that is known to fail. ComfyUI saw the file the moment it landed in
`/basedir/models/loras/` (its folder cache is mtime-based): **no container restart was needed**,
nothing on the native daemon was touched. Trigger phrase and strength live in
`config.CONCEPT_LORA` ("Pixel art style.", 0.8 — the card's band is 0.6–1.0); the prompt is
`config.CONCEPT_PROMPT` = trigger + the owner's description + the base pass's framing and
flat-white-background rules, so `post`'s matte has a field to key on. 768×768, square on
purpose: the square-aspect rule is Qwen-Image-Edit's, this pass never reaches that model, and
nothing downstream reads the size.

**The sketch — scribble, and why not canny.** The Fun ControlNet Union (the full 2602 build
already in the container) has no mode input: the card lists Canny / Depth / Pose / MLSD / HED /
Scribble / Gray and the node takes `image` + `strength`, so the condition is whatever the image
looks like. A hand sketch *is* strokes, so it is sent as the scribble condition — greyscale,
inverted when the scan is dark-on-light, transparency treated as paper, letterboxed into the
cell on black — at strength 0.6, the low end of the card's band, because the sketch is a
silhouette hint rather than a pose to trace. Canny was the alternative: run an edge detector on
a drawing that is already lines and hand the model a thinner, jaggier version of the same
thing. If a future sketch is a shaded drawing rather than line art, canny is the one-line swap
in `sketch_control`. With no sketch the ControlNet apply node is bypassed (as `render` does for
the LoRA) **and** the now-orphaned `ModelPatchLoader` + `LoadImage` are pruned
(`comfy.prune_orphans`), so the provenance hash and `models_lock.for_workflow` describe the graph
the image actually came from, and no never-uploaded filename reaches ComfyUI's validator.

**Where the pick lands — decided, with the alternative.** Every later stage already reads the
unit's reference from the base job: `clip.find_reference` takes
`work/<slug>_base/render/<render.final>`, `post --parts` extracts the part palettes there,
`post` + `publish` turn it into the 64 px sprite. So `--pick` lays that job out with
`base.prepare(slug, cell=<candidate size>)` and writes `render/concept_pick.png`, with
`render.final` pointing at it, a `render/provenance.json` built from the concept job's record
(seed, per-candidate workflow hash as sent, models, the prompt, the LoRA, the sketch; empty
passes for the existing art) and a `concept` block in `job.json`. `anim base` is unchanged and
`anim post knight_base` / `anim publish knight_base` run verbatim on the pick. An earlier base
render's manifest is rewritten but its files stay on disk, and the CLI names what it replaced;
picking again changes the pick. The alternative was a new first lookup in `clip.find_reference`
pointing at `work/<slug>_concept/pick.png` — it would have kept the base job pristine, but it
changes another stage, and every stage that follows (`post --parts`, `publish`, the turnaround)
would have needed the same new lookup; one slot is the point of the base job.

**The existing art as candidate 00.** `--existing` with no value registers the unit's published
sprite (`assets/2d/units/<slug>.png`, the same lookup `strip` uses) at the strip reference's ×8
nearest upscale, bottom-centred on white in the cell; `--existing <png>` registers any file the
same way. It is candidate 00 and `--pick existing` (or `--pick 00`) chooses it — so "keep the
hand art" is a pick like any other, with a provenance record that says it was not generated.

**The real run.** `anim concept --unit knight --count 4 --existing --prompt "a plate-armoured
knight with a closed great helm, a kite shield on the left arm and a longsword in the right
hand, dark cloak, silver plate with gold trim"` on ComfyUI 0.35.0 / the A5000: **109.9 s** for
the first image (Z-Image + the Qwen3-4B encoder paging in), then **8.7 / 8.8 / 8.9 s** per image
by the service's own `execution_start → execution_success` clock; `/system_stats` afterwards
reported 12.7 GB of 25.3 GB VRAM in use. All four are front-facing pixel-art knights on a clean
white field with helm, shield and sword where the prompt put them. A second job
(`sketchtest`, two candidates, a sketch made from candidate 04's edges) proved the scribble
wiring on the real service: the pose, sword hand and shield arm follow the lines across seeds,
10.9 s/image. The manual test surface for #273:
<http://192.168.1.158:8090/sprite-studio/character-concept/> (the five candidates, the sketch
pair, the job.json).

**Left for the owner.** The pick itself — nothing was picked; `knight_base` still holds the
2026-09-15 rig render and `work/knight_concept/` holds the five candidates. `--count 4` was the
brief's number; the base seed is derived from unit + prompt + sketch, so re-running the same
command reproduces the same four and `--seed` re-rolls. Two things to note before the turnaround
(#275): a front-facing pick as `clip`'s identity reference for side-view clips is untested (the
second gate used the profile base render), and `post --parts` on a front-facing pick prompts
SAM from the canned side-profile skeleton the base job is laid out with — the palettes it
extracts will be rough until the turnaround gives the base job a matching skeleton.

---

## 2026-09-20 — the camera direction sheet: one command for the manual test (#288)

**What.** `#288`'s manual test asked for one command and one page. `anim/tools/direction_sheet.py`
(new package `anim/tools/`) is that command: `python -m anim.tools.direction_sheet knight Idle`
renders frame 0 of the named clip in every cell of `config.DIRECTIONS` — sixteen skeleton +
depth pairs — plus the legacy `--mirror` render next to `s180`, and writes one static contact
sheet naming every cell against the five pass criteria in #288. Every cell goes through the exact
calls `anim rig` / `anim base` make (`rigkit.reference_frame`, `clip_keypoints`, `clip_depth`,
`posekit.draw`) at one sampled time instead of the whole clip — no second projection, nothing
written under `work/<job>/`. Ran green: 33 PNGs in 5.5 s, CPU only.

**The unit → rig lookup nothing else in the pipeline has.** Every stage takes a GLB path on the
command line; this tool's two positional args (`unit`, `clip`) needed a default, so
`DEFAULT_RIG_FOR_UNIT` lives in the new module rather than `anim/config.py`. It points `knight`
at the **KayKit** Knight, not Quaternius: both bodies are skinned, but the manual test's own
example clip is named `Idle`, which is a KayKit clip — Quaternius's UAL rig calls its nearest
equivalents `Idle_Loop` / `Sword_Idle` and has none spelled exactly `Idle`.

**Output has no mount to publish through.** `compose.yml` mounts `/assets`, `/assets_src` and
`/game02` into the container but nothing under `~/.cache/claude/visuals`, so the sheet writes to
`work/sheets/camera-direction/` (inside the repo, already mounted) and the CLI prints the one
`cp -r` line to publish it. Evidence: <http://192.168.1.158:8090/sprite-studio/camera-direction/>.

**Test, without a real rig.** `tests/test_direction_sheet.py` reuses `test_rigkit.py`'s two-bone
synthetic GLB (`write_synthetic_glb`) plus a throwaway `rig.json` sidecar (every rig here carries
a licence, even a fixture). Its bones match no `rigkit.JointMap` `detect_map` knows, so
`build_sheet` takes an optional `jmap` — test-only — that folds every OpenPose joint onto the
rig's two bones, geometrically meaningless but enough to prove all sixteen cells, both channels
and the mirror cell get written and named in `index.html`, with no network and no GPU. The bare
rig has no mesh, so `Rig.skinned` is False and every depth frame is a placeholder — the sheet
says so per cell (`no depth: unskinned rig`) rather than silently omitting the file, so the
16 × 2 + 1 file count holds for any rig, skinned or not. Full suite: 420 passed / 4 skipped / 1
failed (the one other
failure in the suite right now, `test_cli.py::test_every_stage_in_the_plan_is_registered`,
belongs to the concept-stage work landing in parallel on this branch — untouched here).

---

## 2026-09-20 — the camera direction feature: `--yaw` / `--elevation`, the rig rotated before projection

**What.** `anim rig` and `anim base` take `--yaw {0,45,…,315}` and `--elevation {side,top}`
— [#272](http://192.168.1.158:8700/work_packages/272), the first feature of the sprite studio
epic. The posed joints, the skinned mesh and the prop geometry are rotated into the chosen
view *before* the existing orthographic projection, so all sixteen cells come out with a
skeleton PNG, keypoints and a depth frame from the same code — no new model, no new stage,
nothing after `rig` knows which cell a job is. This is what lets the turnaround (#275) and
the action matrix (#278) render 8 side + 8 top-down facings from one set of rig clips.
393 passed / 4 skipped (was 312 / 4).

**Where.** The table is `config.DIRECTIONS` (two pitches × eight facings, cells `s0…s315`,
`t0…t315`, with `direction_name()` / `direction()` to go each way); the rotation is
`rigkit.direction_rotation` feeding `rigkit.Projection.rotation`, a 3×3 whose rows are the
view's right / up / near axes in world space, applied inside `Projection.view()` — the one
place every point (joint, vertex, prop outline) passes through on its way to `to_2d` and
`depth_of`. Yaw is the figure's turn about its own up axis, counter-clockwise seen from above,
so the compass reads the same in both pitches and matches a top-down game's eight directions:
0 faces screen right (the party rule), 90 away (north), 180 left, 270 the viewer (south).
`top` pitches the camera straight down about screen x, leaving the figure's yaw-0 left side
up-screen — a knight facing east has its shield to the north — and makes world up the depth
axis, so the head is the nearest surface.

**Byte-identical by construction, then checked.** The default cell builds exactly the basis
the profile always used (forward, world up, forward × up) and the side elevation reads screen
up straight from `y` rather than through a dot product, so no ulp can move. Verified: six jobs
(Knight hit / chop / death / mirrored idle / base, Quaternius strike) produced with the
pre-change code and with this one — every PNG identical, the JSON different only by the three
new keys. Existing job names are untouched; a non-default cell suffixes the default name
(`knight_strike_t270`, `knight_base_s270`) so sixteen views of one action never overwrite
each other, and the s0 base keeps the `<slug>_base` name `clip` and `post --parts` look for.
`job.json.motion_source` gains `direction`, `yaw`, `elevation` next to `mirror` and the rig's
true `forward`.

**The top-down pivot rule — decided, with the alternative.** The scale is the rig's one scale
in every cell, and `side` keeps its rule unchanged (absolute floor row, x centred per frame on
the joints' extent). For `top` I chose: **both screen axes are ground axes, so both get the
ground rule** — centred on the joints' extent in x *and* y, pivot at the cell's centre. Two
reasons. One rule for every yaw: a lunge is centred at t0 (it runs along x) and would have
*translated* at t90 (along y) under any absolute anchoring, so the token's motion would depend
on the cell. And it fits: a body lying full-length (`Death_A`, `Lie_Down`) spans ~250 px of a
320 px cell and is centred in every cell — tested for seven cells × four clips, every joint
inside the rtmlib edge margin. The alternative the brief suggested — anchor the posed
hips/root on a row set by the feet's rest-pose footprint, so a standing figure's near foot
sits on the floor row like the profile's does — gives a stabler root for upright clips but
puts a fallen body at t270 through the bottom edge, and the row it stabilises is discarded
anyway: `post` trims each frame to its opaque box and bottom-centres it in the 64 px canvas.
If the turnaround gate wants the root pinned for the idle stills, it is a per-clip row
shift in `Projection`, not a redesign.

**The mirror decision.** `--yaw 180` *is* the flipped profile (the reflection of a profile is
the other profile — tested equal to `--mirror` on the Knight to 0.02 px), so `--mirror` stays
as it was for the default cell and is **refused with any other cell**, at the CLI
(`SystemExit` before the rig is opened) and in the library (`reference_frame` raises).
Combining them would say "face left" twice in two vocabularies; one flag per fact.

**Left open, on purpose.** The OpenPose face points are drawn in every cell, nose included on
the back view, with confidence 1.0 as before — a real DWPose back view would report the nose
absent. Whether the pose ControlNet wants them hidden is a question for the turnaround gate
(#275) with real renders, not a geometry decision; the hook is `openpose_points`, which has
the head's forward vector and the view's near axis to hand. The top-down *skeleton* is, as
expected, a poor ControlNet image (an OpenPose figure seen from above is a T of limbs); the
top-down *depth* frame is the usable channel, which is why the top pitch is a later milestone
(#285) that rides the clip route.

---

## 2026-09-20 — the sprite studio feature: grilled, filed, the tree landed, no code

**What.** The owner asked for a new animation pipeline — a web pixel editor in the class of
LibreSprite / Piskel / PixelLab / Spritix / Spritely / Sorceress, with open models on the local
GPU behind it: base character → 8 side + 8 top-down facings → idle / walk / run / jump /
attack → 12 fps keyframe phases (start → wind-up → action → wind-down → end → loop) → curated,
palette-snapped sheets. A 27-question grilling session settled the design; it is written up
as `games-design-wiki/shared/sprite-studio.md` and filed as the new OpenProject project
**Sprite Studio** — epic [#271](http://192.168.1.158:8700/work_packages/271), features
#272–#287 — with `TODO.md` in this repo now generated from it (`09-export-todos.sh` maps
`tools/anim-pipeline:sprite-studio`; the pre-commit guard is installed here too). Nothing in
`anim/` changed.

**The reuse question, answered with two reports.** A pros/cons pass on this repo said "fresh
project, vendor the four proven pieces" — the job model is game-02-shaped (fixed 64×64,
`publish` shells into `asset_audit.py`), the tree was mid-rewrite, and the editor is new work
either way. A plan pass (Plan A extend vs Plan B fresh) said "extend" — the rig → clip → post
route is the only thing that actually answers *directions and actions* (rotate the rig before
projection and every view comes with skeleton + depth for free), and Plan B would rebuild ~8
of 15 features that exist and are tested, then rediscover the rig route because text-prompted
i2v gives no action or prop constancy (2026-09-16 research). **The owner chose Plan A
outright.** The pros/cons objections are met at the boundary instead: a generic export
contract with today's `publish` as the game-02 adapter, a FastAPI layer over `work/` that
retires the :8710 review server, and a project-level frame size.

**Landed to start clean.** The four stale `Repos/Game-Dev` paths committed (`8a3c619`); the
`feat/clip-render`, `feat/part-palette-lock` and `feat/rig-props` worktrees and branches
removed — all three were fully merged at `5ed567d`, zero commits ahead, clean; new branch
`feat/sprite-studio` off `feat/anim-pipeline`.

**Decisions that touch this repo directly.**

- The editor is the shell: a fork of **Piskel** (MIT) in a Vite + TypeScript shell under
  `web/`, served by a FastAPI service on **:8720** (Docker Desktop). Jobs stay file-based in
  `work/<job>/`; GPU jobs only on an explicit request, one at a time. The ComfyUI container
  stays the only GPU service.
- **The KayKit/Quaternius ruling stays open** and closes in the frame route bake-off feature
  (#274): knight × attack through rig → Wan video *and* keyframe stills → tween, both rigs,
  judged on a side-by-side page plus the part-colour drift table. The Wan-side pixel LoRA
  question is decided there too — both candidates found target the 14B i2v, not the 5B
  Fun-Control in `gpu/`.
- Pixel-art LoRA for the base character: **tarn59/pixel_art_style_lora_z_image_turbo**
  (Apache-2.0, ~162 MB), the owner's pick from
  `docs/research/2026-09-20-pixel-art-loras.md`. It goes into `gpu/models.lock` with the
  character concept feature (#273). The `prithivMLmods` Qwen LoRA row's hash mismatch was
  re-checked live: an upstream HF metadata bug (the file is valid, 1,179,885,016 bytes); it
  stays parked.
- Frame budget cap 12 at 12 fps — idle 6 · walk 8 · run 8 · jump 10 · attack 12 — replaces
  the "2–6 key poses per action" contract for studio output; `pick` gains phase templates
  (#276). Exaggeration (squash/stretch, smears, secondary) is applied on the rig before
  projection (#277), after the bake-off and before the 80-cell matrix (#278).
- v1 exit: **the knight**, side view, 8 × 5, imported and playing in game-02. Top-down,
  video intake and sheet slicing are a later milestone (#285–#287).

**Why no code today.** The order is camera → concept → bake-off → turnaround; the bake-off
decides the route and the rig before anything is built on top of either, so the first code
is the yaw/elevation rotation in `rigkit.py` (#272) — geometry on existing tests, not a bet.

---

## 2026-09-17 — the second gate: strike / hit / death through the clip route, three control variants

**What.** The three features merged (`5ed567d`, 312 passed / 4 skipped) and run end to end
on the Quaternius knight: `rig → clip → pick → post --input clip [--parts]` for `Sword_Attack`,
`Hit_Chest`, `Death01`, three control variants each — (a) pose-only, (b) skeleton over
full-body depth at gain 0.35, (c) skeleton over props-only depth at gain 1.0 — one seed
(`3139000483690707`) for all nine clips, strike picks hand-set to `[0, 5, 10, 18]` (the
contact, 2026-09-16), hit/death auto. Sheet, per-frame verdicts and per-part tables:
<http://192.168.1.158:8090/second-gate/>. Nothing published; nothing in OpenProject changed;
**the ruling is the owner's.** No code changed for the run — the merge held on real data.

**The three checks, honestly.**

| | (a) pose | (b) full depth 0.35 | (c) props depth 1.0 |
|---|---|---|---|
| 1 · one knight per frame | 10/10 | 10/10 | 10/10 |
| 2 · part colours across actions | by construction where found; drift 3–88 px/part | same; drift 0–128 | same; drift 1–151 |
| 3 · one sword, sword hand, every strike frame | 2/4 — frame 0 grows a second blade behind the legs, frame 10 shows a hilt and no blade | **4/4** | 2/4 — frame 0 second blade behind the legs, frame 5 a blade on the shield arm |
| seconds strike / hit / death | 20.1 / 9.5 / 51.7 | 19.9 / 9.0 / 51.0 | 21.0 / 8.9 / 51.7 |

**(b) is the variant to put in front of the owner**: the only one with one sword in the sword
hand on every strike frame, one knight everywhere, and the shield lands on the raised arm on
the contact frame and over the head on the recovery. (c), the props-only depth the rig props
follow-up added, did not beat it: the depth solids put the shield in front at rest (frame 18
is the best rest frame of the nine) but the model re-grew a blade on the shield arm mid-swing.
Gain 0.5 for (c) was not tried — the brief's trigger (grey props or a broken knight at 1.0)
did not fire.

**What check 2 actually measures now.** After the lock every colour a part has is a member of
that part's palette — true by construction wherever SAM found the part, so the table's
informative numbers are `drift` (pixels the lock had to move) and the "not found" cells. The
tolerance rule applied: exact set membership, no colour distance. A real tolerance (Oklab
distance, or a per-part drift ceiling) is still to be chosen.

**Two findings outside the checks.**

- **Identity holds at the rest poses and drifts at the extremes, in every variant.** Every rest
  frame is the base render's knight — silver plate, dark cloak, gold hands — and across actions
  those rest frames agree with each other, which the anchor gate's strips did not. The strike's
  frames 5 and 10 turn dark and painterly (a thin silhouette, the kite shield a cloak-like
  shape). Same character, a different rendering; the part lock trims the colours but cannot
  redraw a silhouette.
- **`rembg isnet-anime` still drops lying bodies** (death frames 2–3 matted to nothing, as on
  2026-09-15) and SAM on those empty mattes labelled the whole body "helmet" (1083 px). The
  death rows on the sheet are `--matte chroma`, which the clip frames' flat white background
  supports; with it the parts separate again. Mask quality follows matte quality.

**The part-palette stopgap.** The accepted base predates the skeleton props feature: its
keypoints carry no prop polygons, so SAM gets no sword/shield prompt and the base extract
(`work/knight_base_gate/post/parts.json`, on a copy of the job) has no palette for either.
`work/gate_parts.json` = that extract + the sword palette from strike (b) frame 0 (64 px, SAM
score 1.0) + the shield palette from strike (b) frame 2 (82 px, score 1.0) — highest score,
then most pixels, across the three strike variants. Gate-only. **Proper fix, owner's call:**
re-render the base with props/depth (`anim base --rig … --depth`), re-accept it,
`anim post knight_base --parts`, publish — then every action locks to the sprite's own sidecar.

**Left to the owner.** The ruling on the three checks; whether (b)'s mid-swing rendering is
acceptable or needs a next lever (a second reference at the contact pose, a per-unit LoRA —
Q8 allowed it, or A14B); the KayKit/Quaternius art-bible ruling (Q20); the base re-render.

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

**Same day, later — `--depth props`.** The clip render feature ran these frames through Wan
2.2 Fun-Control and reported back: full-strength body depth makes the model paint the
*mannequin* (thinner, painterly, the death loses its helm); at control gain 0.35 the knight
survives and the props land, but the death still loses the helm. It asked for a props-only
depth — the channel saying nothing but "what the hand holds". So `--depth` is now a mode,
`full | props | none` (`--no-depth` stays as an alias of `none`), recorded as
`job.json.motion_source.depth_mode`; `depth` stays the bool it was, so a reader of the first
cut still works. Same files, same contract. Two decisions inside it:

- *One grey mapping for both modes.* The range is still read off body + props, so a `props`
  frame is the `full` frame with the body blanked, pixel for pixel — the clip route can
  switch modes, or blend them, without the props changing brightness. `depth_range` is
  identical between `knightq_strike_depth` and `knightq_strike_depthprops`.
- *The body still occludes.* A first cut drew the prop solids alone and the kite strapped on
  the far forearm showed through the torso — which would tell the model there is a shield
  *in front of* the chest, the opposite of the truth. So `props` mode rasterises the full
  scene and keeps a pixel only where a prop is the nearest surface. A hidden shield stays
  hidden, as in the skeleton PNGs (where the outline is drawn but the body is not, this is
  the stricter of the two).

*Default stays `full`.* The finding is one run at one gain on one action; `props` is
untested on the model at the time of writing, and `full` at gain 0.35 already passes strike
and hit. The clip render gate decides which mode the route ships with; flipping the default
then is a one-line change plus a README row. `anim base --rig` takes the same mode.
`props` on a unit with no kit writes no depth (there is nothing to draw) and records
`depth_mode: none`. Props-only strike and death are on the evidence page under the full
renders. Suite 261 passed / 1 skipped, +4.
## 2026-09-17 — the clip render feature: `anim clip`, one Wan 2.2 Fun-Control video per action

**What.** A new GPU stage between `rig` and `pick`. `anim clip <job>` renders EVERY frame
of a rig clip as one short video — Wan 2.2 Fun-Control **5B** (the owner's pick over
A14B), the unit's `base` render as `ref_image`, the skeleton PNGs as the control video —
and writes one generated frame per rig frame to `clip/frames/`. `pick` runs unchanged on
the keypoints; `post --input clip` takes the picked frames one by one instead of slicing
a strip; `publish` records `generation.route: "clip"` and copies `clip/provenance.json`
instead of the strip render's. The knight's `Sword_Attack`, `Hit_Chest` and `Death01` are
rendered through it (jobs `knightq_{strike,hit,death}_clip`, **not published**). Evidence,
timings and the depth trials: <http://192.168.1.158:8090/clip-render/>. Work package
[#268](http://192.168.1.158:8700/work_packages/268); decisions in
`game_core/discovery/game02/ai-sprite-sheet-generators-research.md` §6/§7.1.

**Why.** The anchor gate published every action as a pose pass, so identity across
actions was prompt + seed alone and the shield/armour colours wandered. A video model
denoises all of a clip's frames against one reference latent — the cross-frame
mechanism the 2026-09-16 survey found — so at least within an action the character is
one character. First result: **it is.** All 19 strike frames are the base render's
knight (helm, cape, red shield with the yellow emblem, one sword). Cost: **20.2 s** for
strike, **8.8 s** hit, **48.7 s** death, by ComfyUI's own execution stamps, against a
5-minute budget; the first job after a model load adds ~65 s.

**What was built, where.**

- `gpu/workflows/clip_render.json` — translated from the official
  `video_wan2_2_5B_fun_control` template (ComfyUI 0.35.0, templates 0.1.78): `UNETLoader`
  → `ModelSamplingSD3` shift 8 → `KSampler` 20 steps / cfg 5 / uni_pc; `Wan22FunControlToVideo`
  with `ref_image` + `control_video`; `LoadVideo` + `GetVideoComponents` for the control;
  **`SaveImage` on the decoded batch** instead of `CreateVideo`/`SaveVideo`, so the CPU
  side gets PNGs and never decodes video. Every class and key read from source —
  `gpu/workflows/NOTES.md` has the rows.
- `gpu/models.lock` +3 rows via `relock_models.py` (which gained a 429 back-off — the HF
  API rate-limited an anonymous relock): `wan2.2_fun_control_5B_bf16` (9.3 GiB),
  `wan2.2_vae` (the 48-channel one the 5B line needs), `umt5_xxl_fp8_e4m3fn_scaled`
  (6.3 GiB). Comfy-Org repackaged, Apache-2.0, ungated. Downloaded and verified. No new
  custom node — core has everything. A14B is NOT locked; it is the fallback only.
- `anim/stages/clip.py` — control video assembly (pure, tested): pad to 4n+1 by holding
  the last frame with `frame_map` recorded; `--control auto|pose|depth|blend` and
  `--depth-gain`; lossless H.264 4:4:4 through ffmpeg so 2 px sticks survive; reference
  letterboxed to the cell's aspect on white before upload (the node centre-crops
  otherwise); seed derived from clip hash + action + unit + control mode (picks are not
  known yet); `clip.json` + `provenance.json` in the render's shape.
- `anim/comfy.py` `upload_image(name=)` — inputs go up under job-unique names because
  `LoadImage`/`LoadVideo` list only the input root.
- `post.py` split into `strip_cells` / `clip_cells`; `publish.py` route selection;
  `Job` gained `depth_dir`, `clip_dir`, `clip_frames_dir` and friends; `config.CLIP_*`.
- Tests: `tests/test_clip.py` (+17) against `FakeComfy`, which now answers a frame
  batch. **255 passed, 1 skipped** (was 238).

**Found on the GPU, recorded in NOTES.md.** `LoadVideo` reports its input preview
under `outputs[].images` with `type: "input"` (22 entries for 21 frames) — the stage
keeps `type == "output"` only. A **512×768 portrait cell clips a lying skeleton**: the
rig scales once per rig, a lying 1.83-unit body is ~600 px wide, the death skeleton's
bbox started at x = 0 and the model faithfully drew half a knight — death is rigged at
**768×768**; Wan has no square-aspect penalty, that rule is Qwen-Image-Edit's.

**The depth trials (later the same day).** The rig props feature landed on its branch
(`057df0b`) mid-build with the contract: `depth/%04d.png`, L, near 255 / far 48 /
background 0, same cell and index as the skeletons. Its committed `rig` was run with
this worktree's `work/` mounted over the container's, so nothing in its tree changed,
and the three actions were re-rendered with `blend` (skeleton over depth), `depth`
alone, and `blend` at `--depth-gain 0.35`:

- **Full-strength depth solves the props and breaks the identity.** Sword in the sword
  hand and the shield on the rig's plate in every frame — the two-sword strike is gone —
  but the model paints the *mannequin*: thinner, painterly, and on death a bare head
  with hair where the helm should be. Depth-only degrades to a grey mannequin by
  mid-clip.
- **Dimmed depth (gain 0.35) is the compromise for strike:** the knight stays the
  knight, sword and shield land where the rig put them. Death still loses the helmet —
  the mannequin's head silhouette is the one thing the reference cannot overrule.
- **Ask of the rig props feature:** a props-only depth (sword + shield, no body). That
  would carry "what the hand holds" without carrying "what the body looks like". Until
  then `auto` = blend at gain 1.0 remains the default so the behaviour is explicit;
  the owner picks per action.

**Open (TODO-verify in NOTES.md).** Pose+depth composition as above; 12 fps 1:1 versus
doubling to the model's native 24 fps; the prompt table is a first draft; the
styly-agents pixel-animate LoRA is A14B-only and moot for 5B. The `pick` result for
strike is the auto `[0, 6, 10, 18]` — the hand-pinned contact frame 5 of the strip
route was not carried over; the gate can re-pick from the clip.

## 2026-09-17 — the part palette lock feature: parts keep their colours, and the masks come from the rig

**What.** `anim post --parts` ([#270](http://192.168.1.158:8700/work_packages/270)). Each HD
cell is masked into the unit's parts — `config.UNIT_PARTS`, for the knight `sword, shield,
helmet, arms, torso, legs` in priority order — by **SAM 2.1 hiera-small on the CPU**, the masks
ride the same trim and downscale as the pixels (by coverage, so a one-pixel blade survives),
and after the unit-level `-remap` every pixel of a part is re-snapped to *that part's* palette.
The palettes come from the accepted base sprite: `post --parts` on a **base** job is the EXTRACT
mode and writes `post/parts.json`; `publish` carries it into the sprite's sidecar as
`generation.parts`; every action job then LOCKS to it (precedence: `--parts-from FILE` > the
published `<slug>.json` > `work/<slug>_base/post/parts.json`). `report.json` and the action's
sidecar record, per frame and per part: pixels, colours, `found`, and **`drift`** — how many
pixels were outside the part's palette before the lock — which is the measurable half of the
second gate. **Off by default**, and with it off `post` writes the 2026-09-15 bytes exactly
(opt-in `ANIM_POST_REGRESSION=1` re-runs `knight_hit` and `knight_death` and compares with
`/assets`; passed). Evidence: <http://192.168.1.158:8090/part-palette-lock/>.
Tests: 272 passed, 4 skipped (was 238 / 1); the four skips are opt-ins (rtmlib smoke, SAM 2.1
smoke, the two regression rows).

**Why SAM 2.1 with rig prompts, not SAM 3 with text.** SAM 3 is gated on the Hub (manual
approval) and `license: other` — it fails the repo's ungated / Apache-2.0 rule, checked
2026-09-17. SAM 2.1 is Apache-2.0 and ungated, takes points and boxes, and the pipeline already
knows where every part is: `pose/keypoints.json` carries the joints the ControlNet was driven
with and, since the skeleton props feature, the sword and shield outlines in the same pixels.
A part is therefore joints (and points along bones — the crown of the helmet is neck→nose at
1.35) or a prop kind whose polygon centroid is the point. Same prompts for the base and every
action, which is what makes the palettes comparable. `posekit.letterbox()` is now the one
transform from keypoint pixels to a cell, used by `draw` and by the prompts.

**What the prompts needed, measured on the knight base and `knight_strike`.**

- **Negatives collapse SAM 2.** Feeding the other parts' points as negative prompts gave
  predicted IoU 0.000 and a few hundred pixels for every part. So SAM gets the positives only,
  in multimask mode, and the negatives are used to **choose** among its three candidates
  (sub-part / part / whole): eligible = at least half the positives inside and at most one in
  four negatives inside (the occlusion budget — a raised shield covers the helmet's points in
  profile), and among the eligible SAM's own IoU picks. Nothing eligible → the part is "not
  found" for that frame and the report says so; nothing is locked to a guess.
- **A box from the prop outline hurts.** It pinned the raised shield to the kite's footprint
  and SAM returned the visor-sized blob inside it (864 px) instead of the shield (8,046 px);
  on the sword it collapsed one cell to nothing. `Part.box` exists and is off.
- **Contamination weighs double** in the fallback ranking: the base's rest-pose sword polygon
  put one of its two points on the torso, and at equal weight "torso and sword" beat "sword".
- **A base part under 8 px is not a palette.** The accepted base predates the props feature,
  so its shield prompt lands on the chest and finds 3 px of "shield"; locking the strike's
  raised shield to those three colours flattened it. `PART_MIN_PALETTE_PIXELS` rejects it (kept
  in `parts.json` under `rejected`, with the reason) and the actions leave the shield alone
  and say "no palette for this part".

**Colour space.** The unit-level snap stays ImageMagick `-remap` (sRGB). The within-part snap
is Oklab, in numpy: "which of the shield's greys is this pixel" is a question about how it
looks, and Oklab distances are perceptually uniform where sRGB's are not. Candidates are
palette members already, so the frame stays inside the unit palette whatever the metric.

**Where it stands.** The mechanism runs end to end and every locked part's colours are a subset
of its base palette. Mask quality tracks how well the render followed the rig: on strike cells
1–2 (props drawn, the model followed them) the sword and the raised shield are found cleanly;
on rest cells and on the 2026-09-15 renders (no props drawn) the sword prompt lands where the
rig holds the sword, not where the model painted it. The base palettes are the weak link today
— the knight base has to be re-run with `anim base --parts` once it is rendered with props /
depth (the clip route re-bases anyway), or `parts.json` hand-edited and passed with
`--parts-from`. Cost: SAM 2.1-small ≈ 1–1.5 s per HD frame on the CPU after a ~5 s one-time
load; a four-frame action went from 25 s to 31 s wall.

**Where it lives.** `anim/partkit.py` (Part, prompts, the SAM wrapper, candidate choice,
resolution, coverage downscale, Oklab, lock); `config.UNIT_PARTS` / `parts_for_unit()`,
`PART_LOCK_DEFAULT`, `PART_MIN_SCORE`, `PART_MIN_PALETTE_PIXELS`, `SAM2_WEIGHTS` (repo,
revision, file, size, sha256 read from the Hub API — `partkit.ensure_weights` refuses a
checkpoint that does not hash to it; lands in `.cache/sam2/`); `post.process_cell(part_masks=,
part_palettes=, parts_out=)`, `post.post(parts=, parts_from=, segmenter=)`, `post/parts/` HD
overlays and 64 px label maps; `anim base --parts`; `publish` → `generation.parts` and the
SAM row appended to `models`. The image gains `sam2==1.1.0` in its own Dockerfile step
(`--no-build-isolation`, `SAM2_BUILD_CUDA=0`; the sdist's setup.py imports torch, which pixeloe
already brought). Why no separate `anim parts` command: the extraction IS `post` on the base
job — same matte, same snap, same verification — and a second command would have been a
second copy of that chain.

**Open.** Frames rendered without props will keep mis-locating the props; the clip render
feature's depth-conditioned frames are the intended input. A rig-aware occlusion order (from
the rig props feature's depth) could replace the one-in-four occlusion budget. Arms and torso
split on the knight's pauldrons inconsistently between frames; a per-unit prompt tune is a
config edit, not code. `hound` has no humanoid part table.

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
