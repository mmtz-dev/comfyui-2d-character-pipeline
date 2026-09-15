#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Publish the side-by-side comparison page to the claude-visuals LAN server.
#
# the 3D-intermediate spike feature (Feature 9) - the anchor-gate deliverable.
#
#   tools/anim-pipeline/spike3d/publish_compare.sh [--seed 42] [--palette barrow-32]
#
# Serves at http://192.168.1.158:8090/spike3d-compare.html
#
# Layout it expects (missing files render as labelled placeholders, so this is
# safe to run at any point, including before anything has been generated):
#
#   spike3d/2d/<action>/frame_0..N.png      published 64x64 frames, 2D chain
#   spike3d/2d/<action>/skeleton_strip.png  the rtmlib-rendered conditioning strip
#   spike3d/2d/<action>/raw_strip.png       identity_pass output before `anim post`
#   spike3d/3d/<action>/...                 same four, from the rig
#
# <action> is strike, hit, death.
#
# NOTE the delete-then-copy. The claude-visuals container pins a file after its
# first request; overwriting in place serves the old bytes.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIS="${HOME}/.cache/claude/visuals"
PAGE="spike3d-compare.html"
LAN="http://192.168.1.158:8090"

SEED="${SEED:-unset}"
PALETTE="${PALETTE:-unset}"
FRAMES="${FRAMES:-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)    SEED="$2"; shift 2 ;;
    --palette) PALETTE="$2"; shift 2 ;;
    --frames)  FRAMES="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$VIS" ]]; then
  echo "claude-visuals dir not found: $VIS" >&2
  echo "It is served by the 'claude-visuals' container; create the dir and retry." >&2
  exit 1
fi

# --- assets ---------------------------------------------------------------
rm -rf "${VIS:?}/spike3d"
mkdir -p "$VIS/spike3d"
copied=0
for arm in 2d 3d; do
  for action in strike hit death; do
    src="$HERE/$arm/$action"
    [[ -d "$src" ]] || continue
    mkdir -p "$VIS/spike3d/$arm/$action"
    for f in "$src"/*.png; do
      [[ -e "$f" ]] || continue
      cp -f "$f" "$VIS/spike3d/$arm/$action/"
      copied=$((copied + 1))
    done
  done
done

# --- page -----------------------------------------------------------------
# Rewrite the manifest block, then delete-and-recreate so the server re-reads it.
python3 - "$HERE/compare.html" "$VIS/$PAGE.tmp" "$SEED" "$PALETTE" "$FRAMES" <<'PY'
import datetime, pathlib, re, sys

src, dst, seed, palette, frames = sys.argv[1:6]
html = pathlib.Path(src).read_text(encoding="utf-8")
today = datetime.date.today().isoformat()

def q(v):
    return '"' + str(v).replace('"', '\\"') + '"'

for key, value in (("seed", seed), ("palette", palette), ("date", today)):
    html = re.sub(rf'(\n  {key}: ).*?,', lambda m: m.group(1) + q(value) + ",", html, count=1)
html = re.sub(r'(\n  frameCount: )\d+,', lambda m: m.group(1) + str(int(frames)) + ",", html, count=1)

pathlib.Path(dst).write_text(html, encoding="utf-8")
PY

rm -f "$VIS/$PAGE"
mv "$VIS/$PAGE.tmp" "$VIS/$PAGE"

echo "copied ${copied} image(s)"
echo "seed=${SEED} palette=${PALETTE} frames=${FRAMES}"
echo
echo "  ${LAN}/${PAGE}"
