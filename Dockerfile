# =============================================================================
# CPU pipeline image - the pose-to-pixel animation pipeline feature
# =============================================================================
# Runs on DOCKER DESKTOP (context desktop-linux). No GPU, no CUDA: the only GPU
# workload in this project is ComfyUI on the native daemon (gpu/compose.yml),
# which this container talks to over HTTP at $COMFYUI_URL.
#
# Build:
#   docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml build
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: build pixel-art-fixer (Retro-Diffusion/pixel-art-fixer, MIT).
#
# It is a Rust CLI and publishes NO prebuilt release binaries - checked
# 2026-09-13, the repo has no releases at all - so it has to be compiled here
# rather than fetched. Pinned to a commit so the grid-repair behaviour cannot
# change under us between builds. The repo also ships a pure-Python
# implementation under python/; the Rust one is used because the post-chain runs
# once per frame across 100+ animations.
# -----------------------------------------------------------------------------
FROM rust:1-slim-trixie AS pixelfixer-build

ARG PIXEL_ART_FIXER_REF=ef376e57e1c272633ca2dbf5f29ec3fcf6596465

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --quiet https://github.com/Retro-Diffusion/pixel-art-fixer.git . \
 && git checkout --quiet "${PIXEL_ART_FIXER_REF}" \
 && git log -1 --format='pixel-art-fixer pinned at %H (%ci)'

# Upstream's peak_rss() calls the Windows psapi API with no #[cfg(windows)]
# guard, so cargo cannot link it on Linux ("unable to find library -lkernel32").
# The patch gates that behind cfg(windows) and adds a getrusage(2) version.
# `git apply` against the pinned commit either applies exactly or fails the
# build - it cannot silently half-apply. See the patch header for detail.
COPY patches/pixelfixer-linux-peak-rss.patch /tmp/
RUN git apply --verbose /tmp/pixelfixer-linux-peak-rss.patch

RUN cd rust && cargo build --release && ls -l target/release/pixelfixer

# -----------------------------------------------------------------------------
# Stage 2: build the web pixel editor (web/ - Piskel in a Vite + TS shell).
#
# `npm ci` against the committed package-lock.json, then `vite build` ->
# web/dist. The vendored Piskel (web/vendor/piskel) is joined by the shell's
# own plugin, so the only packages this stage needs are vite, typescript and
# vitest - no spritesmith, no puppeteer, nothing that downloads a browser.
# Node 22 LTS: Vite 7 wants 20.19+ / 22.12+. The stage is independent of the
# Rust one, so BuildKit builds them side by side.
# -----------------------------------------------------------------------------
FROM node:22-slim AS web-build

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build \
 && test -f dist/index.html \
 && test -f dist/piskel/piskel.js \
 && echo "web pixel editor built: $(du -sh dist | cut -f1)"

# -----------------------------------------------------------------------------
# Stage 3: the runtime image.
#
# Python is pinned to 3.12 - the host's 3.14 is not used for anything, and
# onnxruntime / rembg wheels lag new CPython releases.
#
# Debian TRIXIE, not bookworm, for one specific reason: bookworm's `imagemagick`
# package is ImageMagick 6.9, which provides `convert`/`mogrify` but NOT the v7
# `magick` entry point the post-chain calls. Trixie ships ImageMagick 7.1. Both
# stages use trixie so the glibc the pixelfixer binary was linked against matches
# the runtime's.
# -----------------------------------------------------------------------------
FROM python:3.12-slim-trixie

# imagemagick on trixie is v7, so `magick` exists. The build asserts that below -
# if a future base image regresses to v6 the build fails here rather than the
# post-chain failing per-frame at runtime.
#
# ffmpeg: stage 1 (`anim extract`) decodes the filmed clip.
# libgl1 / libglib2.0-0: opencv-python-headless still needs these at import.
# git: pip installs pixeloe from a git ref.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      imagemagick \
      libgl1 \
      libglib2.0-0 \
      git \
      ca-certificates \
      curl \
 && rm -rf /var/lib/apt/lists/* \
 && magick -version | head -1 \
 && magick -version | grep -q "ImageMagick 7" \
      || (echo "FATAL: expected ImageMagick 7 (the 'magick' CLI); got:" \
          && magick -version && exit 1)

# Prove the exact post-chain invocation works in this image, at build time, on a
# throwaway 2-colour palette. Catches a policy.xml coder restriction or a missing
# PNG delegate now instead of once per frame later.
RUN cd /tmp \
 && magick -size 8x8 xc:red png:probe_in.png \
 && magick -size 2x1 xc:red png:probe_pal.png \
 && magick probe_in.png -dither None -remap probe_pal.png png:probe_out.png \
 && echo "ImageMagick -remap probe OK" \
 && rm -f probe_in.png probe_pal.png probe_out.png

COPY --from=pixelfixer-build /src/rust/target/release/pixelfixer /usr/local/bin/pixelfixer
RUN pixelfixer --help >/dev/null 2>&1 || pixelfixer help >/dev/null 2>&1 || true

# The built editor. anim/config.py serves the checkout's /work/web/dist when
# the host built one (the bind mount), else this copy - so a fresh clone gets
# the editor from the image with no node on the host. ANIM_WEB_DIST overrides.
COPY --from=web-build /web/dist /opt/anim/web/dist

WORKDIR /work

# Dependencies first so a code change does not re-resolve the whole tree.
COPY requirements.txt /opt/anim/requirements.txt
RUN pip install --no-cache-dir -r /opt/anim/requirements.txt

# SAM 2.1 for the part palette lock feature - its own layer, no build isolation
# (setup.py imports torch, which the layer above already installed), CUDA
# extension off (optional; no nvcc here). See requirements.txt for the why.
RUN SAM2_BUILD_CUDA=0 pip install --no-cache-dir --no-build-isolation "sam2==1.1.0" \
 && python -c "from sam2.build_sam import build_sam2; print('sam2 importable')"

# Test-only deps in their own layer, AFTER the expensive one, so moving pytest
# never re-resolves torch + onnxruntime + the CUDA wheels.
COPY requirements-dev.txt /opt/anim/requirements-dev.txt
RUN pip install --no-cache-dir -r /opt/anim/requirements-dev.txt

# The pipeline API (fastapi + uvicorn), same reasoning: its own cheap layer so
# a framework bump never touches the torch layer.
COPY requirements-api.txt /opt/anim/requirements-api.txt
RUN pip install --no-cache-dir -r /opt/anim/requirements-api.txt

# The CLI package. Installed with -e so a second agent can edit the mounted
# source and re-run without rebuilding the image.
COPY pyproject.toml /opt/anim/pyproject.toml
COPY anim /opt/anim/anim
RUN pip install --no-cache-dir -e /opt/anim \
 && anim --help >/dev/null \
 && echo "anim CLI installed"

# rembg downloads its ONNX model on first use; keep it on the mounted cache so
# it survives a container recreate.
# rembg downloads its ONNX model on first use and rtmlib downloads DWPose's
# under $XDG_CACHE_HOME/rtmlib; both land on the bind-mounted /work/.cache so
# the ~400 MB is fetched once per machine, not once per container.
#
# PYTHONPATH=/work puts the MOUNTED source ahead of the copy `pip install -e`
# made at /opt/anim. Without it, editing anim/ on the host changes nothing until
# the image is rebuilt - the editable install points at the COPY, not the mount.
ENV U2NET_HOME=/work/.cache/rembg \
    XDG_CACHE_HOME=/work/.cache \
    NUMBA_CACHE_DIR=/tmp/numba \
    PYTHONPATH=/work \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    COMFYUI_URL=http://host.docker.internal:8188

ENTRYPOINT ["anim"]
CMD ["--help"]
