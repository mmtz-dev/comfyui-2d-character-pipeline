# Test fixtures — sources and licences

game-02 ships commercially and the repo's asset rule is **CC0 or public domain,
nothing else** (`game-02-tactical-rpg/tools/asset_audit.py`, `ALLOWED_LICENCES`).
Test fixtures are not shipped assets, but they are committed files, so they are
held to the same rule. Every fixture is listed here.

## Generated — no licence question at all

Most of this suite's inputs are **drawn or computed at test time**, which is the
cheapest way to have no licence problem:

| Fixture | Built by | What it is |
|---|---|---|
| the test clips | `tests/fixtures/synthetic.py` → `frame_series` + `write_clip` | a stick-and-blob figure against a plain wall, holding a start pose, swinging once, holding an end pose — the take the art bible's filming rules ask the owner to film. Encoded to a real H.264 file with ffmpeg so `anim extract` runs its real decode path, rotation metadata included. |
| the "generated" strips | `tests/fixtures/synthetic.py` → `write_strip` | an N × 1 strip of flat-shaded figures on white, standing in for what the GPU would return, so the whole post-chain can be tested without ComfyUI. |
| the canned keypoints | `tests/conftest.py` → `keypoints_payload` | 13 frames of an arm swinging out and back, written by hand so the picker's expected answer is obvious by inspection (frame 6 is the extreme). |
| the standing skeleton | `anim/data/stand_pose.json` | **synthesised joint coordinates, not traced from a photograph.** Ships in the package rather than in tests because `anim base` uses it in production. |
| the fake ComfyUI | `tests/fixtures/fakecomfy.py` | a threaded `http.server` returning canned `/system_stats`, `/prompt`, `/history` and `/view` responses and a Pillow-made PNG. |
| the two-bone rig | `tests/test_rigkit.py` → `write_synthetic_glb` | a GLB with a `foot` and a `toes` node and one clip (`Turn`), written with pygltflib at test time, so the interpolation and projection maths have closed-form answers. |

Nothing above is downloaded, traced, derived from a pack, or model output.

## The one real rig

`tests/test_rigkit.py` and `tests/test_rig.py` also run against the vendored
**KayKit Adventurers `Knight.glb`** (`rigs/kaykit-adventurers/`, CC0 1.0 by its
own `LICENSE.txt`) and detect the map of **Quaternius `UAL1_Standard.glb`**
(`rigs/quaternius/`, CC0 1.0 by the zip's `License.txt`). Neither lives under
`tests/` — they are production inputs of the rig motion input feature — and each
directory's `NOTICE.md` is the provenance record. Both tests skip cleanly when the
files are absent.

## The one real photograph

| File | `tests/fixtures/pose/cc0_performer.png` |
|---|---|
| Title | *Circus performer blonde man standing on ball and juggling* |
| Author | D Coetzee |
| Source | <https://www.flickr.com/photos/29507259@N02/3624201561> |
| Original file | <https://live.staticflickr.com/3390/3624201561_7d5a56313f_b.jpg> |
| Licence | **CC0 1.0 Universal (Public Domain Dedication)** — <https://creativecommons.org/publicdomain/zero/1.0/> |
| Licence verified | 2026-09-13, twice: the Openverse API reports `license: cc0`, `license_version: 1.0`, and the Flickr photo page itself carries `"license":9`, which is Flickr's CC0 1.0 id. Not taken on trust from an aggregator alone. |
| Modifications | cropped to the performer (x 95–680, y 36–1024 of the 680×1024 original) and downscaled to 303×512 with Lanczos. CC0 permits modification without condition. |

**Why a real photograph is needed at all.** Everything else in the pose stage —
the schema, the normaliser, the renderer's geometry, the caching rule — is tested
against synthetic keypoints. But "does rtmlib actually find a person" can only be
answered by a person. That is
`tests/test_pose.py::test_smoke_dwpose_finds_the_person_in_the_cc0_photo`, and it
is **opt-in**:

```bash
docker compose run --rm -e ANIM_POSE_SMOKE=1 anim-pipeline pytest tests/test_pose.py -k smoke
```

It is opt-in because it downloads the DWPose ONNX weights (~300 MB) on first
run, and a suite that needs the network to go green goes red for the wrong
reasons. It was run and **passed** on 2026-09-13; the weights then live in the
bind-mounted `.cache/rtmlib/`, so later runs are offline and take ~35 s.

**Why this photograph and not a cleaner studio shot.** Searched Wikimedia
Commons and Openverse for a CC0/PD full-body human against a plain background;
there is very little. The obvious public-domain candidate — Muybridge's *Animal
locomotion* plates, which are literally frames of a motion sequence — is
Volume VIII "Males and Females (nude and semi-nude)", which is not a file to
commit to a game repo. This one is a real person, head to feet, in daylight,
with a busy background; the crop keeps the performer and the downscale leaves
bystanders' faces around 20 px. DWPose finds several people in it, which is
useful in itself: it exercises `best_person`, the "which of these is the
subject" rule that a clip filmed anywhere but an empty room will need.

## Read from the mounts, never copied into the repo

These are inputs the container already has and the tests read rather than vendor.
Each test that needs one falls back to a generated stand-in (or skips with a
message) when run outside the container, so the suite never depends on a mount
being present.

| Path | Used for | Licence |
|---|---|---|
| `/assets_src/palette_candidates/grim32/grim32.gpl` | the palette the post-chain snaps to | Grim32 by Jake Stinnett, from Lospec; a palette is a list of numbers, and the candidates were staged by the art bible feature |
| `/assets/2d/units/knight.png` | the reference sprite `strip` upscales ×8 | CC0-1.0, Dungeon Crawl Stone Soup 32×32 tiles (its sidecar records it) |
| `/game02/tools/asset_audit.py` | judging the sidecar `publish` writes | this repo's own code |
