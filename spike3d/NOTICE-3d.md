# Third-party notices — the 3D-intermediate spike feature

Attribution obligations carried by the code this spike vendors or calls. Kept alongside the
repo's existing `../NOTICE`; fold these in if the spike is adopted.

## TripoSG — MIT
`VAST-AI-Research/TripoSG`, pinned at `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c` (2025-04-18).
Weights: `VAST-AI/TripoSG` on Hugging Face, MIT, ungated.

Its own `NOTICE` declares three inbound dependencies, reproduced here verbatim because they are
attribution obligations on the source files:

> **RMBG** — Copyright (c) BRIA AI — License: bria-rmbg-1.4 —
> Source: https://huggingface.co/briaai/RMBG-1.4
>
> This software contains code derived from 🤗 Diffusers
> (https://github.com/huggingface/diffusers), available under the Apache License 2.0.
>
> This software contains code derived from HunyuanDiT (https://github.com/Tencent/HunyuanDiT).
> Notice: Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement,
> Copyright © 2024 Tencent. All Rights Reserved. The trademark rights of "Tencent Hunyuan" are
> owned by Tencent or its affiliate.
>
> This software contains code derived from FlashVDM (https://github.com/Tencent/FlashVDM).
> Notice: Tencent Hunyuan FlashVDM is licensed under the Tencent Hunyuan FlashVDM Community
> License Agreement, Copyright © 2025 Tencent. All Rights Reserved.

**The BRIA one is not carried.** `briaai/RMBG-1.4` is non-commercial. `spike3d/run_triposg.py`
never imports `briarmbg`, never downloads those weights, and refuses any input that would make
TripoSG reach for them. See `RESEARCH.md` §1.4.

The two Tencent notices are attribution on *derived source files*, not a territory restriction
on generated output — that restriction lives in the Hunyuan3D model licence, which is exactly
why Hunyuan3D itself was rejected (`RESEARCH.md` §1.3).

## Astropulse/mixamotoopenpose — MIT
`Copyright (c) 2024 Astropulse`. Used to project a Mixamo `.dae` to COCO-18 2D joints. Its
renderer is **not** used — see `RESEARCH.md` §3.

## rtmlib — Apache-2.0
`Tau-J/rtmlib`, already pinned at `0.0.16` in `../requirements.lock`. `spike3d/render_openpose.py`
calls its `draw_skeleton`; none of it is reimplemented.

## Mixamo content — Adobe
Animations and characters are free for unlimited commercial use, no royalty, no attribution
required. The raw `.dae`/`.fbx` must not be redistributed (see `.gitignore`), and Adobe's terms
state Mixamo content may not be used to train machine-learning models — discussed in
`RESEARCH.md` §2.2.

## CMU Graphics Lab Motion Capture Database — if the CC0 route is taken
Their requested acknowledgement, verbatim:

> The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with
> funding from NSF EIA-0196217.
