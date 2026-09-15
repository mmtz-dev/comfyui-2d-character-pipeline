#!/usr/bin/env bash
# Clone (or verify) every custom node in custom_nodes.txt at its pinned commit,
# straight into the ComfyUI basedir. Pure git on the host - no Docker, no GPU.
# Idempotent: an existing clone is fetched and re-checked-out at the pin.
set -euo pipefail

GPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASEDIR="${COMFYUI_BASEDIR:-/mnt/private/AI Data/comfyui-basedir}"
DEST="$BASEDIR/custom_nodes"
mkdir -p "$DEST"

grep -v '^[[:space:]]*#' "$GPU_DIR/custom_nodes.txt" | grep -v '^[[:space:]]*$' | \
while read -r name url sha _rest; do
  target="$DEST/$name"
  if [[ -d "$target/.git" ]]; then
    echo "== $name: present, pinning to ${sha:0:8}"
    git -C "$target" fetch --quiet origin
  else
    echo "== $name: cloning"
    git clone --quiet "$url" "$target"
  fi
  git -C "$target" checkout --quiet "$sha"
  echo "   $(git -C "$target" log -1 --format='%H %ci %s')"
done

echo
echo "custom_nodes in $DEST:"
ls -1 "$DEST"
