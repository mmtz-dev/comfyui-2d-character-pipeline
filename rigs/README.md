# rigs/ — CC0 rigged characters, the pipeline's second motion source

The rig motion input feature. A rigged, animated glTF character stands in for a filmed clip:
`anim rig <glb> --clip <name> --unit <slug> --action <verb>` samples the clip, maps the bones to
the 18-point OpenPose skeleton, projects it in profile facing right, and writes the same
`work/<job>/` layout `anim extract` + `anim pose` would have. See the fork section of
[`../README.md`](../README.md) ("Rig input mode").

## Rule: only CC0 goes in here, and every rig carries its licence

| Directory | Rig | Licence | Provenance |
|---|---|---|---|
| `kaykit-adventurers/` | `Knight.glb` — KayKit Character Pack: Adventurers 1.0 (Kay Lousberg), 41-bone rig, 76 clips | CC0 1.0 (`LICENSE.txt`) | [`NOTICE.md`](kaykit-adventurers/NOTICE.md) |
| `quaternius/` | `UAL1_Standard.glb` — Quaternius Universal Animation Library, Standard tier, Unreal-mannequin rig, 43 clips | CC0 1.0 (`License.txt` in the zip) | [`NOTICE.md`](quaternius/NOTICE.md) |

Each directory has three things, and `anim rig` refuses to run without the third:

1. the GLB(s), unmodified;
2. the pack's own licence file, unmodified, plus a human-readable `NOTICE.md` (pack, author,
   URL, licence quote, retrieval date, sha256, which files were taken and which were not);
3. **`rig.json`** — the machine-readable copy: `pack`, `author`, `url`, `license`, `retrieved`
   and `files: {<name>: <sha256>}`. `anim rig` hashes the GLB and compares; a mismatch is an
   error, because a rig whose bytes changed is a rig whose licence has to be re-read. The block
   ends up, verbatim, as `motion_source` in `job.json` and then in the published sidecar's
   `generation.motion_source`, so the asset audit can see the motion's provenance next to the
   pixels' provenance.

The GLBs are 3.7 MB and 7.6 MB. They are committed as-is for now (the audio milestone used Git
LFS for larger binaries; switch these over if the directory grows).

## Adding a rig

1. Check the licence **in the downloaded file**, not on the store page (Quaternius's site licence
   changed on 2026-08-28 while its pack pages still said CC0 — the zip's own `License.txt` is what
   counts). CC0 only.
2. `mkdir rigs/<pack>/`, copy in the GLB + licence file, write `NOTICE.md` and `rig.json`
   (`sha256sum` the GLB).
3. `anim rig rigs/<pack>/<file>.glb --list` — if the bone names match a known family
   (`kaykit`, `quaternius-ual`) you are done. KayKit Skeletons, Mage, Rogue, Barbarian all share
   the Adventurers rig and need nothing.
4. Otherwise add a `JointMap` row to `RIGS` in `anim/rigkit.py`: the thirteen body joints as bone
   names (a `(a, b)` pair means midpoint — the OpenPose neck is always the shoulder midpoint,
   because that is how rtmlib defines it), the head bone, the `forward_hint` pairs whose
   horizontal direction is "forward" (foot→toes), the five face offsets in head lengths, and the
   `detect` bones that identify the family. `tests/test_rigkit.py` checks every row covers all 18
   joints.

## What a rig is not

It is not art. Nothing in here is drawn, rendered or shipped; the mesh is read only to measure the
model's rest-pose height (the per-rig scale) and head length (for the synthesised nose / eyes /
ears). The pixels come from the GPU stages, re-skinned onto the unit's own reference sprite, and
the art bible's proportions win over the rig's.
