#!/usr/bin/env bash
# =============================================================================
# the GPU image service feature - install the `nunchaku` wheel into the venv
# =============================================================================
# Run by the mmartial/comfyui-nvidia-docker init script on EVERY container start
# (it runs /userscripts_dir/*.sh in name order, inside the activated venv, as
# the comfy user, with PIP3_CMD exported). Idempotent: exits 0 in a second when
# the pinned wheel is already installed.
#
# WHY A WHEEL AT ALL. ComfyUI-nunchaku (gpu/custom_nodes.txt) is a thin node
# pack that `import nunchaku` at load time and does not vendor it. Without the
# wheel, NunchakuQwenImageDiTLoader is absent from /object_info and the
# identity pass cannot load the INT4 Qwen-Image-Edit weights.
#
# WHY THIS EXACT WHEEL. It must match the container's torch minor version and
# python ABI. nunchaku publishes wheels per (cuda, torch, cpXX); no release
# builds against torch 2.13 (the image's default as of 2026-09-15, see DEVLOG),
# so compose.yml pins torch to 2.11.0 with TORCH_LOCK and this wheel is the
# v1.2.1 build for torch 2.11 - the same version as the pinned node pack.
# `cu12.8` is the toolkit it was compiled with; it runs on torch's cu129
# runtime (CUDA minor-version compatibility). INT4 kernels, not NVFP4: the
# A5000 is Ampere (sm_86).
#
# Moving the pin: change all three values below together, re-record the wheel
# filename + sha256 in DEVLOG.md, and keep TORCH_LOCK in compose.yml in step.
# =============================================================================
set -euo pipefail

NUNCHAKU_VERSION="1.2.1"
WHEEL_NAME="nunchaku-1.2.1+cu12.8torch2.11-cp312-cp312-linux_x86_64.whl"
WHEEL_URL="https://github.com/nunchux-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1%2Bcu12.8torch2.11-cp312-cp312-linux_x86_64.whl"
WHEEL_SHA256="962eaabc7be74299678d83719e3589e97136c91752d61227ef1f55a3b50b32d4"
EXPECT_TORCH="2.11."

# Persistent, outside the repo: /comfy/mnt is the comfyui-run bind mount, so a
# container recreate does not re-download 150 MB.
WHEEL_DIR="${COMFYUSER_DIR:-/comfy}/mnt/wheels"
PIP="${PIP3_CMD:-uv pip install}"

torch_ver="$(python3 -c 'import torch; print(torch.__version__)')"
case "${torch_ver}" in
  ${EXPECT_TORCH}*) ;;
  *) echo "!! 10-nunchaku.sh: torch is ${torch_ver}, wheel needs ${EXPECT_TORCH}x - is TORCH_LOCK set in gpu/compose.yml?"; exit 1 ;;
esac

have="$(python3 -c 'from importlib.metadata import version; print(version("nunchaku"))' 2>/dev/null || true)"
if [ "${have}" = "${NUNCHAKU_VERSION}+cu12.8torch2.11" ]; then
  echo "== nunchaku ${have} already installed (torch ${torch_ver})"
else
  mkdir -p "${WHEEL_DIR}"
  wheel="${WHEEL_DIR}/${WHEEL_NAME}"
  if [ ! -f "${wheel}" ] || ! echo "${WHEEL_SHA256}  ${wheel}" | sha256sum -c --quiet; then
    echo "== downloading ${WHEEL_NAME}"
    curl -fsSL --retry 3 -o "${wheel}.part" "${WHEEL_URL}"
    echo "${WHEEL_SHA256}  ${wheel}.part" | sha256sum -c --quiet
    mv -f "${wheel}.part" "${wheel}"
  fi
  echo "== installing ${WHEEL_NAME} (was: ${have:-absent})"
  ${PIP} "${wheel}"
fi

# ComfyUI-Image-Saver's one requirement; without it the pack fails to import.
python3 -c 'import piexif' 2>/dev/null || ${PIP} piexif

python3 - <<'PY'
from importlib.metadata import version
import torch
print(f"== nunchaku {version('nunchaku')} | torch {torch.__version__} | piexif {version('piexif')}")
PY
