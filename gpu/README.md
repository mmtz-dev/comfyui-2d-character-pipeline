# The GPU image service feature

ComfyUI headless, serving the two generation passes of **the pose-to-pixel
animation pipeline feature** over HTTP on `0.0.0.0:8188`.

This is the **only** part of this project that runs on the native Docker daemon.
Everything else — the CPU pipeline, the model downloader, the Mac share — runs on
Docker Desktop (context `desktop-linux`).

---

## Why the native daemon, and only here

This host runs two Docker engines:

| Engine | Context | Socket | GPU |
|---|---|---|---|
| Docker Desktop (**the default**) | `desktop-linux` | `~/.docker/desktop/docker.sock` | **no** — runs in a Linux VM |
| Native `docker.service` | `docker-default` | `/var/run/docker.sock` | **yes** — nvidia runtime |

ComfyUI needs real CUDA on the RTX A5000, so it is the one service that must run
on the native daemon. The native daemon is `disabled` and normally stopped.

**Rules, not suggestions:**

- Never run `docker context use`. The active context stays `desktop-linux` so a
  bare `docker ps` always shows the normal Docker Desktop stack.
- Target this service explicitly, every time: `docker --context docker-default …`.
- Starting the native daemon does **not** disturb the running Docker Desktop
  containers; they are on different sockets.
- `sudo systemctl start docker` needs a real TTY for the password. Only the owner
  can run it — an agent cannot, and should not ask for it to be left running.

---

## Owner-run sequence

Run these from the repo root (`/home/workstation/Repos/Game-Dev`).

### 1. Pre-flight (no daemon needed)

```bash
# The A5000 and a driver >= 580 (the CUDA 13 floor for this image)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# The compose file parses (this runs on Docker Desktop and starts nothing)
docker --context desktop-linux compose -f tools/anim-pipeline/gpu/compose.yml config

# The basedir exists and is owned by uid/gid 1000 (WANTED_UID/WANTED_GID)
ls -ld "/mnt/private/AI Data/comfyui-run" "/mnt/private/AI Data/comfyui-basedir"

# The weights are in place and match models.lock
tools/anim-pipeline/gpu/download_models.sh --verify
```

### 2. Start the native daemon — **in a real TTY**

```bash
sudo systemctl start docker
```

This asks for a password, so it cannot be run through an agent. Confirm it came
up, and confirm the nvidia runtime is registered:

```bash
docker --context docker-default info --format '{{.ServerVersion}} runtimes={{.Runtimes}}'
```

### 3. Bring the service up

```bash
docker --context docker-default compose -f tools/anim-pipeline/gpu/compose.yml up -d
```

**The first start is slow** (10–30 min): the image builds a python venv and
installs torch into `/mnt/private/AI Data/comfyui-run`. Later starts reuse it.
Watch it:

```bash
docker --context docker-default compose -f tools/anim-pipeline/gpu/compose.yml logs -f
```

It is ready when the log says `To see the GUI go to: http://0.0.0.0:8188`.

### 4. Smoke test — the GPU is really there

```bash
curl -s http://192.168.1.158:8188/system_stats | python3 -m json.tool
```

Expect a `devices[0].name` containing **NVIDIA RTX A5000** and a `vram_total`
around 25 GB. If it names a CPU device instead, the nvidia runtime or the
SELinux label did not take — check `security_opt` and `runtime: nvidia`.

Two more checks worth doing on the first run:

```bash
# The nodes the workflows need are actually loaded
curl -s http://192.168.1.158:8188/object_info | python3 -c '
import json,sys
have=json.load(sys.stdin)
for n in ["ModelPatchLoader","ZImageFunControlnet","TextEncodeQwenImageEditPlus",
          "NunchakuQwenImageDiTLoader","CLIPLoader","KSampler","SaveImage"]:
    print(("OK   " if n in have else "MISS "), n)'

# The weights are visible to ComfyUI (basedir + the read-only shared library)
curl -s http://192.168.1.158:8188/object_info/CLIPLoader | python3 -m json.tool | head -30
```

### 5. The `nunchaku` wheel — installed for you, on every start

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
docker --context docker-default logs anim-pipeline-comfyui 2>&1 | grep "== nunchaku"
# == nunchaku 1.2.1+cu12.8torch2.11 | torch 2.11.0+cu129 | piexif 1.1.3
```

The A5000 is Ampere (sm_86) → the **INT4** kernels, not NVFP4 (Blackwell-only).
Moving the pin means changing the wheel name, URL and sha256 in the script, the
`TORCH_LOCK` line, and this paragraph, together.

### 6. Generate one sprite

```bash
# From the CPU pipeline container (it reaches this service via host.docker.internal)
cd tools/anim-pipeline
docker compose run --rm anim base --unit knight \
  --rig rigs/quaternius/UAL1_Standard.glb --clip Idle_Loop --frame 0 --strength 1.0 \
  --prompt "a single side-view pixel art character sprite of a plate-armoured knight ..."
```

First render is slow (the models page in from disk: ~2.5 min); later ones ~20 s
per pass.

### 7. Shut down when done

The service holds ~20 GB of VRAM while loaded. `restart: "no"` means it never
comes back on its own.

```bash
docker --context docker-default compose -f tools/anim-pipeline/gpu/compose.yml down
sudo systemctl stop docker      # TTY again; optional
```

---

## What is in this directory

| File | What it is |
|---|---|
| `compose.yml` | The service. Project name `anim-pipeline-gpu`, image pinned by digest. |
| `extra_model_paths.yaml` | Exposes the pre-existing checkpoint library read-only so nothing is duplicated. |
| `models.lock` | **Resolved lockfile** — repo, revision, filename, size, sha256, licence for all 12 weights (58.23 GiB). Regenerate with `relock_models.py` (it writes next to itself, from any CWD); do not hand-edit — the KNOWN DISCREPANCY note is part of the generator. |
| `userscripts/` | Start-up scripts the image runs inside the venv on every start (`10-nunchaku.sh`). |
| `relock_models.py` | The only writable source for `models.lock`. Cross-checks two HF endpoints and refuses to write on any disagreement, gated repo, or non-commercial licence. |
| `download_models.sh` | Downloads the weights **on Docker Desktop** (no GPU needed). Idempotent, resumable, verifies every file against `models.lock`. |
| `hf_download.py` | Runs inside the throwaway `python:3.12-slim` container that `download_models.sh` starts. |
| `custom_nodes.txt` | The two custom node packs with pinned commits, plus the list of packs deliberately **not** installed. |
| `install_custom_nodes.sh` | Clones/verifies them into the basedir at those pins. Idempotent. |
| `workflows/` | The three API-format ComfyUI workflows. See `workflows/NOTES.md`. |

## Layout on disk

```
/mnt/private/AI Data/
├── comfyui-run/                     -> container /comfy/mnt   (ComfyUI code + venv)
├── comfyui-basedir/                 -> container /basedir     (BASE_DIRECTORY)
│   ├── models/
│   │   ├── diffusion_models/        z_image_turbo_int8_convrot, nunchaku qwen-edit int4
│   │   ├── text_encoders/           qwen_3_4b_fp8_mixed, qwen_2.5_vl_7b_fp8_scaled
│   │   ├── vae/                     z_image_ae, qwen_image_vae
│   │   ├── model_patches/           Z-Image Fun ControlNet Union 2.1 2602 (full) + lite
│   │   └── loras/                   Qwen-Image-2512 Pixel-Art LoRA
│   ├── custom_nodes/                ComfyUI-nunchaku, ComfyUI-Image-Saver (pinned)
│   ├── input/  output/  user/
└── models/Image Models/             -> container /shared-models  (READ-ONLY)
```

## Hardware notes

The A5000 is **Ampere** (sm_86). It has no FP8 or FP4 tensor cores, so an `fp8`
weight file saves VRAM but buys no speed. The speed path on this card is the
quantised weights already in `models.lock`: `*_int8_convrot` for Z-Image and
Nunchaku **INT4** (SVDQuant) for Qwen-Image-Edit. Do not "upgrade" these to
fp8/nvfp4 builds — nvfp4 will not run at all.

Qwen-Image-Edit at fp8 with an fp8 text encoder needs ~27.9 GB and does not fit
in 24 GB; that is the whole reason for the INT4 build.

## Known issues

**The optional Pixel-Art LoRA does not install.** (And ComfyUI would reject a graph
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
cache at `<basedir>/models/.hf-cache` so it is on the big disk and a re-run does not
re-download. That means ~35 GiB of models plus ~35 GiB of cache. Once the weights are
verified and ComfyUI has loaded them once, the cache can go:

```bash
rm -rf "/mnt/private/AI Data/comfyui-basedir/models/.hf-cache"
```

Deleting it only costs bandwidth on a future re-download; `models.lock` remains the
source of truth for what should be there.

## Licences

All nine weight files are **Apache-2.0** and ungated — checked against the HF
API by `relock_models.py`, which refuses to write a lock row for anything else.
Both custom node packs are Apache-2.0 / MIT. The container image is MIT.

Never add `comfy_pixelization` (non-commercial) or `Astropulse/sd-palettize`
(no licence). Palette snapping is done deterministically in the CPU post-chain
with ImageMagick instead.
