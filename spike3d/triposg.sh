#!/usr/bin/env bash
# Run TripoSG (image -> 3D mesh), a GPU batch job, under rootless Podman + CDI.
#
#   spike3d/triposg.sh build [podman-compose build args]    build localhost/spike3d-triposg:local
#   spike3d/triposg.sh [run_triposg.py args]                one run, container removed after
#   spike3d/triposg.sh --dry-run --image-input /work/in/x.png   input + GPU + lease check, no model
#
# Paths inside the container: /work is this directory, /weights the model disk.
# The run books its GPU memory from the shared GPU gatekeeper before the model loads
# (profile anim-triposg) and waits in its queue while the card is full.
set -euo pipefail
cd "$(dirname "$0")"
IMAGE=localhost/spike3d-triposg:local

if [[ "${1:-}" == "build" ]]; then
  shift
  exec podman-compose -f compose.yml build "$@"
fi

if ! podman image exists "$IMAGE"; then
  echo "no $IMAGE yet; build it first (~15-25 min, no GPU needed): $0 build" >&2
  exit 2
fi
mkdir -p in out "/mnt/private/AI Data/models/3D Models"
exec podman-compose -f compose.yml run --rm triposg "$@"
