# NOTICE — Quaternius: Universal Animation Library (Standard tier)

| | |
|---|---|
| Pack | **Universal Animation Library [Standard]** (UAL 1, the free tier; zip 15,904,933 bytes, sha256 `cc73fc4e495b82958207316596317a3f40b9fa38065bde1027937452da537724`) |
| Author | Quaternius — <https://quaternius.com> |
| Download URL | <https://quaternius.itch.io/universal-animation-library> → "Download Now" → "No thanks, just take me to the downloads" → `Universal Animation Library[Standard].zip`. Free; no login; itch's name-your-own-price flow (upload id 17958403). |
| Retrieved | 2026-09-14 |
| Licence **inside the zip** | **CC0 1.0 Universal** — see below |

## What the downloaded zip's licence file says, verbatim

`Universal Animation Library[Standard]/License.txt` (332 bytes, unmodified copy beside this file):

```
-------------------------------------------------------
License:
CC0 1.0 Universal (CC0 1.0) 
Public Domain Dedication
https://creativecommons.org/publicdomain/zero/1.0/

------------------------------------------------------
Models by @Quaternius
Consider supporting me on Patreon!

https://www.patreon.com/quaternius
```

`README.txt` in the same zip repeats the identical licence block and adds the root-motion note
(*"the one ending in _RM has root motion baked into every animation, while the other has root
motion disabled"*).

The itch page on the retrieval date reads *"Free to use in personal, educational and commercial
projects. (CC0 License)"* and lists the asset licence as *"Creative Commons Zero v1.0 Universal"*.

## Why it was downloaded now

<https://quaternius.com/license.html> changed on **2026-08-28** to the *Quaternius Asset License
(QAL) v1.0*: *"You can use these assets, free of charge, in personal, educational, and commercial
games and other projects, with no credit required. You just can't resell or redistribute the
assets themselves as assets."* Its §3(a) forbids redistributing the assets *"as a standalone
asset, asset pack, stock file, template, or similar product, whether for free or for payment"*;
§7 says changes are not retroactive and *"the version in effect at the time you obtained the
Assets governs."* The zip obtained on 2026-09-14 carries the CC0 file above, so the vendored copy
is CC0 by its own licence text. Either licence permits this use — the rig never ships, only the
pixels generated from its poses do — but CC0 is what lets the GLB sit in this repo.

## Files taken

| File | From | sha256 |
|---|---|---|
| `UAL1_Standard.glb` (7,618,436 bytes) | `Universal Animation Library[Standard]/Unreal-Godot/UAL1_Standard.glb` | `69591853d817488edaa8fd9bf8fc1d821eaeaf789f8627b3cd23b41c4ed67997` |
| `License.txt` | zip root | — |
| `README.txt` | zip root | — |

Not taken: `UAL1_Standard_RM.glb` (root-motion variant), the two Unity `.fbx` files, the setup
PNGs. The GLB is a yellow mannequin with **43 clips** (`A_TPose, Sword_Attack, Sword_Idle,
Hit_Chest, Hit_Head, Death01, Idle_Loop, Walk_Loop, Jog_Fwd_Loop, Punch_Jab, Punch_Cross,
Spell_Simple_*, Pistol_*, Roll, ...`) on an **Unreal-mannequin-named** skeleton (`pelvis,
spine_01..03, neck_01, Head, clavicle_l, upperarm_l, lowerarm_l, hand_l, thigh_l, calf_l,
foot_l, ball_l, ...`, 67 nodes, Blender I/O 4.5.48). The research doc's `DEF-*` Rigify names were
from a 2025 community mirror and are **not** what this zip ships; `anim/rigkit.py`'s
`quaternius-ual` map uses the names above.

## What could NOT be fetched

The **Ultimate Animated Character Pack** (2019, 52 characters incl. `Knight_Male`) that the task
named first: its only download is a public Google Drive folder
(`https://drive.google.com/drive/folders/1sNi1AfenfPRrvRt5yfaj5QMMd6KKcUJ5`), which on 2026-09-14
returned *"Quota exceeded — Too many users have viewed or downloaded this file recently"* for
`License.txt` and for `glTF/Knight_Male.gltf` (tried both `drive.google.com/uc` and
`drive.usercontent.google.com`). The pack page itself still reads *"License CC0"*. Retry from a
browser later, or from another network; the UAL above is the rig the research recommended as the
realistic-proportion A/B anyway, and it has the sword / hit / death clips the anchor gate needs.
