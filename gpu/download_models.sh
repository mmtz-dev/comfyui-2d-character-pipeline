#!/usr/bin/env bash
# Download the models named in gpu/models.lock into the shared image-model library
# (the one Forge, clone-studio and this pipeline's ComfyUI all read; merged there
# 2026-10-02 so every image model lives in one folder with a usage note beside it).
#
# Runs in a throwaway python:3.12-slim container on DOCKER DESKTOP
# (context desktop-linux). The Fedora host has Python 3.14 with no pip, and
# nothing in this project runs natively on the host. No GPU is needed to
# download weights, so this deliberately does NOT touch the native daemon.
#
#   ./download_models.sh                # download what is missing
#   ./download_models.sh --verify       # also sha256 files already present
#   ./download_models.sh --dry-run      # print the plan only
#   ./download_models.sh --only pixel_art_style   # one row by filename
#
# Idempotent and resumable: re-run it after an interruption.
set -euo pipefail

GPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASEDIR="${COMFYUI_BASEDIR:-/mnt/private/AI Data/comfyui-basedir}"
MODELS_DIR="${IMAGE_MODELS_DIR:-/mnt/virt-containers/ai-models/image}"
# Blob cache outside the library, so it never shows up as models; safe to delete
# once the weights are verified (see gpu/README.md).
CACHE_DIR="$BASEDIR/hf-cache"
DOCKER_CONTEXT_NAME="desktop-linux"   # never the native daemon; downloads need no GPU
PY_IMAGE="python:3.12-slim"

mkdir -p "$MODELS_DIR"/{diffusion_models,text_encoders,vae,model_patches,lora} "$CACHE_DIR"

echo "== free space =="
df -h "$MODELS_DIR"
echo

exec docker --context "$DOCKER_CONTEXT_NAME" run --rm \
  --name anim-pipeline-model-download \
  -v "$MODELS_DIR:/models" \
  -v "$CACHE_DIR:/hf-cache" \
  -v "$GPU_DIR:/spec:ro" \
  -e HF_HOME=/hf-cache \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  "$PY_IMAGE" \
  bash -c 'set -e
    pip install --quiet --no-cache-dir "huggingface_hub>=0.26"
    exec python /spec/hf_download.py "$@"' -- "$@"
