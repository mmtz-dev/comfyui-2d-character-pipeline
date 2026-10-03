# The GPU image service feature

ComfyUI headless, serving the two generation passes of **the pose-to-pixel
animation pipeline feature** over HTTP on `0.0.0.0:8188`.

This is the **only** part of this project that runs under Podman. Everything else
— the CPU pipeline, the model downloader, the Mac share — runs on Docker Desktop
(context `desktop-linux`).

---

## Why Podman, and only here

This host runs three container engines, and only one of them can give a
container the RTX A5000:

| Engine | How | GPU | Used for |
|---|---|---|---|
| Docker Desktop (**the default**) | context `desktop-linux` | **no** — runs in a Linux VM | everything except this |
| Native `docker.service` | context `docker-default` | yes, nvidia runtime | **nothing — off-limits** |
| **Rootless Podman** | `podman-compose` | **yes**, via CDI | this service |

ComfyUI needs real CUDA. The native Docker daemon can provide it but is off
limits on this host — it breaks the owner's setup, and every render needed
`sudo systemctl start docker` from a real TTY first, which an agent cannot do.
Rootless Podman reaches the same GPU through **CDI** with no daemon and no sudo,
so this service moved there on 2026-09-25 (see `DEVLOG.md`).

**How the GPU gets in:** `nvidia-ctk` writes a CDI spec describing the device
nodes and driver libraries to inject; Podman reads it by name when a service
asks for `devices: ["nvidia.com/gpu=all"]`. The spec lives at
`/var/run/cdi/nvidia.yaml` and `nvidia-cdi-refresh.service` (enabled) keeps it
current across driver updates, so there is nothing to maintain.

**Rules, not suggestions:**

- Never run `docker context use`, and never start `docker.service`. The active
  Docker context stays `desktop-linux` so a bare `docker ps` always shows the
  normal Docker Desktop stack.
- `podman` and `docker` see **different** images and containers. `podman ps`
  shows this service; `docker ps` never will.
- Rootless means no sudo, which means an agent can start and stop this service
  itself. That is the whole point of the move.
- `label=disable` and `userns: keep-id` in the compose file are load-bearing on
  this host — see the comments there before changing either.

---

## Owner-run sequence

Run these from the repo root (`/home/workstation/Repos/game-dev/framework-ladder`).

### 1. Pre-flight (no daemon needed)

```bash
# The A5000 and a driver >= 580 (the CUDA 13 floor for this image)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# The compose file parses (starts nothing)
podman-compose -f tools/anim-pipeline/gpu/compose.yml config

# Podman can really see the GPU (pulls a small CUDA image the first time)
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  docker.io/nvidia/cuda:13.0.3-runtime-ubuntu24.04 nvidia-smi

# The basedir exists and is owned by uid/gid 1000 (WANTED_UID/WANTED_GID)
ls -ld "/mnt/private/AI Data/comfyui-run" "/mnt/private/AI Data/comfyui-basedir"

# The weights are in place and match models.lock
tools/anim-pipeline/gpu/download_models.sh --verify
```

### 2. Bring the service up

No daemon to start first, and no sudo:

```bash
podman-compose -f tools/anim-pipeline/gpu/compose.yml up -d
```

**The first start is slow** (10–30 min): the image builds a python venv and
installs torch into `/mnt/private/AI Data/comfyui-run`. Later starts reuse it.
Watch it:

```bash
podman-compose -f tools/anim-pipeline/gpu/compose.yml logs -f
```

It is ready when the log says `To see the GUI go to: http://0.0.0.0:8188`.

### 3. Smoke test — the GPU is really there

```bash
curl -s http://192.168.1.158:8188/system_stats | python3 -m json.tool
```

Expect a `devices[0].name` containing **NVIDIA RTX A5000** and a `vram_total`
around 25 GB. If it names a CPU device instead, the nvidia runtime or the
SELinux label did not take — check `security_opt: [label=disable]` and that
`devices: ["nvidia.com/gpu=all"]` resolved (`ls /var/run/cdi/nvidia.yaml`).

Two more checks worth doing on the first run:

```bash
# The nodes the workflows need are actually loaded
curl -s http://192.168.1.158:8188/object_info | python3 -c '
import json,sys
have=json.load(sys.stdin)
for n in ["ModelPatchLoader","ZImageFunControlnet","TextEncodeQwenImageEditPlus",
          "NunchakuQwenImageDiTLoader","CLIPLoader","KSampler","SaveImage"]:
    print(("OK   " if n in have else "MISS "), n)'

# The weights are visible to ComfyUI (the read-only shared library)
curl -s http://192.168.1.158:8188/object_info/CLIPLoader | python3 -m json.tool | head -30
```

### 4. The `nunchaku` wheel — installed for you, on every start

`ComfyUI-nunchaku` is a thin wrapper; it imports the `nunchaku` Python package,
which the node pack does not vendor. Without it `NunchakuQwenImageDiTLoader` is
missing from `/object_info` and the identity pass cannot load the INT4 weights.

Two things in the repo handle it, and they move together:

- `compose.yml` sets `TORCH_LOCK=torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0`.
  The image otherwise installs the newest torch on every start (2.13.0 on
  2026-09-15), and nunchaku publishes wheels per torch minor version — none for
  2.13. 2.11 is the newest with a stable v1.2.1 wheel, the same version as the
  pinned node pack.
- `userscripts/10-nunchaku.sh` is mounted at `/userscripts_dir` and run by the
  image's init inside the venv. It downloads
  `nunchaku-1.2.1+cu12.8torch2.11-cp312-cp312-linux_x86_64.whl` (sha256
  `962eaabc7be74299678d83719e3589e97136c91752d61227ef1f55a3b50b32d4`) into
  `/comfy/mnt/wheels/` once, verifies it, installs it, and exits in a second on
  every later start. It also installs `piexif`, the one requirement
  `ComfyUI-Image-Saver` needs to import.

Confirm after the first start:

```bash
podman logs anim-pipeline-comfyui 2>&1 | grep "== nunchaku"
# == nunchaku 1.2.1+cu12.8torch2.11 | torch 2.11.0+cu129 | piexif 1.1.3
```

The A5000 is Ampere (sm_86) → the **INT4** kernels, not NVFP4 (Blackwell-only).
Moving the pin means changing the wheel name, URL and sha256 in the script, the
`TORCH_LOCK` line, and this paragraph, together.

### 5. Generate one sprite

```bash
# From the CPU pipeline container (it reaches this service via host.docker.internal)
cd tools/anim-pipeline
docker compose run --rm anim base --unit knight \
  --rig rigs/quaternius/UAL1_Standard.glb --clip Idle_Loop --frame 0 --strength 1.0 \
  --prompt "a single side-view pixel art character sprite of a plate-armoured knight ..."
```

First render is slow (the models page in from disk: ~2.5 min); later ones ~20 s
per pass.

### 6. Shut down when done

The service holds ~20 GB of VRAM while loaded. `restart: "no"` means it never
comes back on its own.

```bash
podman-compose -f tools/anim-pipeline/gpu/compose.yml down
sudo systemctl stop docker      # TTY again; optional
```

---

## What is in this directory

| File | What it is |
|---|---|
| `compose.yml` | The service. Project name `anim-pipeline-gpu`, image pinned by digest. |
| `extra_model_paths.yaml` | Exposes the pre-existing checkpoint library read-only so nothing is duplicated. |
| `models.lock` | **Resolved lockfile** — repo, revision, filename, size, sha256, licence for all 13 weights (58.38 GiB). Regenerate with `relock_models.py` (it writes next to itself, from any CWD); do not hand-edit — the KNOWN DISCREPANCY note is part of the generator. |
| `userscripts/` | Start-up scripts the image runs inside the venv on every start (`10-nunchaku.sh`). |
| `models.registry.tsv` | **Hand-curated**, beside the lock: which locked weight is a checkpoint / LoRA / ControlNet / text encoder / VAE and for which workflow family (`zimage` \| `qwen-edit` \| `wan`), with a LoRA's trigger and default strength. The concept generation feature's model picker (`GET /models`, `anim concept --model` / `--lora`) offers only these rows, per family — a Z-Image graph is never handed a Qwen file. Add a row when you add a weight; the lock keeps the hashes, this keeps the meaning. |
| `relock_models.py` | The only writable source for `models.lock`. Cross-checks two HF endpoints and refuses to write on any disagreement, gated repo, or non-commercial licence. |
| `download_models.sh` | Downloads the weights **on Docker Desktop** (no GPU needed). Idempotent, resumable, verifies every file against `models.lock`. `--only <text>` restricts it to the rows whose installed filename contains the text — one new weight without walking the other 58 GiB. |
| `hf_download.py` | Runs inside the throwaway `python:3.12-slim` container that `download_models.sh` starts. |
| `custom_nodes.txt` | The two custom node packs with pinned commits, plus the list of packs deliberately **not** installed. |
| `install_custom_nodes.sh` | Clones/verifies them into the basedir at those pins. Idempotent. |
| `workflows/` | The five API-format ComfyUI workflows. See `workflows/NOTES.md`. |

## Layout on disk

```
/mnt/private/AI Data/
├── comfyui-run/                     -> container /comfy/mnt   (ComfyUI code + venv)
├── comfyui-basedir/                 -> container /basedir     (BASE_DIRECTORY)
│   ├── models/                      empty since 2026-10-02 (weights moved to the shared library)
│   ├── hf-cache/                    download_models.sh's blob cache (deletable)
│   ├── custom_nodes/                ComfyUI-nunchaku, ComfyUI-Image-Saver (pinned)
│   ├── input/  output/  user/
└── models/Image Models/             -> container /shared-models  (READ-ONLY)
    ├── diffusion_models/            z_image_turbo_int8_convrot, nunchaku qwen-edit int4, wan2.2 fun control
    ├── text_encoders/               qwen_3_4b_fp8_mixed, qwen_2.5_vl_7b_fp8_scaled, umt5_xxl
    ├── vae/                         z_image_ae, qwen_image_vae, wan2.2_vae (+ Forge's own VAEs)
    ├── model_patches/               Z-Image Fun ControlNet Union 2.1 2602 (full) + lite
    └── lora/_Z_Image/               pixel_art_style_z_image_turbo (the character concept feature); the Qwen-Image-2512 LoRA is parked (see Known issues)

Every image model on the machine lives in the shared library, each with a `<model>.md`
usage note and listed in its `INDEX.md` (the model-notes tool in `ai-media/image-generator`).
`download_models.sh` installs there; the container still mounts it read-only.
```

## Hardware notes

The A5000 is **Ampere** (sm_86). It has no FP8 or FP4 tensor cores, so an `fp8`
weight file saves VRAM but buys no speed. The speed path on this card is the
quantised weights already in `models.lock`: `*_int8_convrot` for Z-Image and
Nunchaku **INT4** (SVDQuant) for Qwen-Image-Edit. Do not "upgrade" these to
fp8/nvfp4 builds — nvfp4 will not run at all.

Qwen-Image-Edit at fp8 with an fp8 text encoder needs ~27.9 GB and does not fit
in 24 GB; that is the whole reason for the INT4 build.

## Weights by feature

| Feature | Files | Size |
|---|---|---|
| the pose pass (`pose_pass.json`, `base_sprite.json`) | Z-Image-Turbo int8 convrot, Qwen3-4B fp8 mixed, Z-Image VAE, Fun ControlNet Union 2.1 2602 full (+ lite, for A/B) | 19.5 GiB |
| the identity pass (`identity_pass.json`, currently unused) | Nunchaku Qwen-Image-Edit-2509 INT4 r128 lightning, Qwen2.5-VL-7B fp8, Qwen-Image VAE | 20.8 GiB |
| the character concept feature (`concept_pass.json`) | the pose pass's Z-Image stack + **`tarn59/pixel_art_style_lora_z_image_turbo`** (`lora/_Z_Image/pixel_art_style_z_image_turbo.safetensors`, 162 MB, Apache-2.0; trigger "Pixel art style.", strength 0.6–1.0, default 0.8 in `anim/config.py`) | +0.16 GiB |
| the clip render feature (`clip_render.json`) | Wan 2.2 Fun-Control 5B bf16, Wan 2.2 VAE, umT5-XXL fp8 | 16.9 GiB |
| parked | prithivMLmods Qwen-Image-2512 Pixel-Art LoRA (fails verification, see below) | 1.1 GiB, not installed |

Install one new row without re-walking the rest: `gpu/download_models.sh --only <filename part>`.

## Known issues

**The optional Qwen Pixel-Art LoRA does not install.** (And ComfyUI would reject a graph
that names it while it is missing, even at strength 0 — so `anim render` leaves the
node out unless `--lora-strength` is given.) `prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA`
advertises one sha256 through the HF API but serves bytes that hash to another. Verified
2026-09-13 with two independent fetches; the file itself is a valid, complete Qwen-Image
LoRA of exactly the advertised size, so this is an upstream registry metadata bug, not a
corrupt download. `download_models.sh` therefore refuses it and reports
`sha256 mismatch`. **Nothing is blocked** — the LoRA is optional and
`anim render` bypasses the node. The full detail, and the two
ways to resolve it, are in the KNOWN DISCREPANCY block at the top of `models.lock`.
The other 7 files (34 GiB) installed and verified clean.

**The Hugging Face blob cache doubles the disk cost.** `download_models.sh` keeps its
cache at `<basedir>/hf-cache` so it is on the big disk and a re-run does not
re-download. That means ~35 GiB of models plus ~35 GiB of cache. Once the weights are
verified and ComfyUI has loaded them once, the cache can go:

```bash
rm -rf "/mnt/private/AI Data/comfyui-basedir/hf-cache"
```

Deleting it only costs bandwidth on a future re-download; `models.lock` remains the
source of truth for what should be there.

## Licences

All thirteen weight files are **Apache-2.0** and ungated — checked against the HF
API by `relock_models.py`, which refuses to write a lock row for anything else.
Both custom node packs are Apache-2.0 / MIT. The container image is MIT.

Never add `comfy_pixelization` (non-commercial) or `Astropulse/sd-palettize`
(no licence). Palette snapping is done deterministically in the CPU post-chain
with ImageMagick instead.
