#!/usr/bin/env bash
# Download the models named in gpu/models.lock into the ComfyUI basedir.
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
MODELS_DIR="$BASEDIR/models"
DOCKER_CONTEXT_NAME="desktop-linux"   # never the native daemon; downloads need no GPU
PY_IMAGE="python:3.12-slim"

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "creating $MODELS_DIR"
  mkdir -p "$MODELS_DIR"/{diffusion_models,text_encoders,vae,model_patches,loras}
fi

echo "== free space =="
df -h "$BASEDIR"
echo

exec docker --context "$DOCKER_CONTEXT_NAME" run --rm \
  --name anim-pipeline-model-download \
  -v "$MODELS_DIR:/models" \
  -v "$GPU_DIR:/spec:ro" \
  -e HF_HOME=/models/.hf-cache \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  "$PY_IMAGE" \
  bash -c 'set -e
    pip install --quiet --no-cache-dir "huggingface_hub>=0.26"
    exec python /spec/hf_download.py "$@"' -- "$@"
