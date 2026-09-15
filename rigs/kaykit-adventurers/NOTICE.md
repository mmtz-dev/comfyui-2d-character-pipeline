# NOTICE — KayKit Character Pack: Adventurers (1.0)

| | |
|---|---|
| Pack | **KayKit : Adventurers Character Pack (1.0)** |
| Author | Kay Lousberg — <https://www.kaylousberg.com> |
| Source | GitHub mirror, no login: <https://github.com/KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0> (itch page: <https://kaylousberg.itch.io/kaykit-adventurers>) |
| Commit fetched | `672074b73ba276876a19e8816ecdc5241817ab47` (2023-09-16), branch `main` |
| Retrieved | 2026-09-14, via `raw.githubusercontent.com` |
| Licence | **CC0 1.0 Universal** (`LICENSE.txt`, unmodified copy beside this file) |

The licence file says, verbatim:

> License: (Creative Commons Zero, CC0)
> http://creativecommons.org/publicdomain/zero/1.0/
>
> This content is free to use in personal, educational and commercial projects.
>
> Support me by using a brand resource provided in this pack or by crediting Kay Lousberg, www.kaylousberg.com (this is not mandatory)

The repository README adds: *"Free for personal and commercial use, no attribution required. (CC0 Licensed)"*
and *"Licensed under CC0 1.0 Universal, see LICENSE.txt for more information."* The same
`LICENSE.txt` sits at the repo root and inside
`addons/kaykit_character_pack_adventures/`; the two are byte-identical.

## Files taken

| File | From | sha256 |
|---|---|---|
| `Knight.glb` (3,659,532 bytes) | `addons/kaykit_character_pack_adventures/Characters/gltf/Knight.glb` | `60428e3abc09ba83e595d256e3af8c5c976b46cdae599f0802fc82b4a3445168` |
| `LICENSE.txt` (891 bytes) | repo root | — |

Nothing else from the pack (no FBX, no textures, no other characters) is vendored. The GLB
carries the rig, 76 animation clips and the Knight's meshes; `anim rig` reads the rig and the
clips and uses the meshes only to measure the model's height and head length in the rest pose.
No mesh, texture or clip is shipped in any game — the rig is a **motion source** for the
pose-to-pixel pipeline, and the pixels it eventually produces are our own generation.

`rig.json` beside this file is the machine-readable copy of this table; `anim rig` refuses to
run if the GLB's hash no longer matches it.
