# =============================================================================
# the 3D-intermediate spike feature - TripoSG (image -> 3D mesh)
# =============================================================================
# Feature 9 of the pose-to-pixel animation pipeline plan. Single-image 3D, MIT
# code and MIT weights, ~8 GB VRAM. See RESEARCH.md §1 for why this model and
# not Hunyuan3D / TRELLIS.2 / Sparc3D.
#
# BUILD ON DOCKER DESKTOP. Compiling the CUDA extension needs `nvcc`, which is
# in the base image; it does NOT need a GPU to be present. So:
#
#   docker --context desktop-linux compose -f tools/anim-pipeline/spike3d/compose.yml build
#
# and only the RUN needs the native daemon. Building ahead of time is what keeps
# the spike inside its 2-session box - see README.md, session 1 step 1.
# =============================================================================

# CUDA 12.4 devel: ships nvcc, which `diso` needs to compile. The A5000's driver
# (580.159) is well past the CUDA 12.4 floor. A `-runtime` tag would not work -
# there would be no compiler.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Ampere / A5000 is sm_86. Pinning the arch list keeps the diso build to one
# target instead of compiling every architecture PyTorch knows about, which is
# the difference between a few minutes and half an hour.
ENV TORCH_CUDA_ARCH_LIST=8.6 \
    FORCE_CUDA=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/weights/.hf \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      ca-certificates \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# TripoSG at a pinned commit.
#
# The repo has no tags and no releases, so a commit is the only pin available.
# fc5c4099 is `main` as of 2025-04-18T06:58:51Z, which is also the repo's last
# push - it is stable, not stale-by-neglect-mid-development.
# -----------------------------------------------------------------------------
ARG TRIPOSG_REF=fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c

WORKDIR /opt
RUN git clone --quiet https://github.com/VAST-AI-Research/TripoSG.git triposg \
 && cd triposg \
 && git checkout --quiet "${TRIPOSG_REF}" \
 && git log -1 --format='TripoSG pinned at %H (%ci)'

WORKDIR /opt/triposg

# -----------------------------------------------------------------------------
# Dependencies.
#
# NOT `pip install -r requirements.txt` verbatim, for two reasons:
#
#  1. upstream pins `numpy==1.22.3`, which predates this Python and has no wheel
#     for it - the install simply fails. 1.26.x is the last numpy 1.x and is what
#     diffusers/trimesh/pymeshlab expect. Do NOT "upgrade" this to numpy 2: the
#     pinned TripoSG commit predates the numpy 2 ABI break.
#  2. `torch` / `torchvision` already come from the base image at the exact CUDA
#     build we want; letting pip resolve them again would swap in a different one.
#
# `diso` is the one that compiles (differentiable dual marching cubes, imported
# unconditionally by triposg/inference_utils.py). It is the slow part of this
# build and the reason for TORCH_CUDA_ARCH_LIST above.
#
# `briaai/RMBG-1.4` is deliberately NOT fetched here, and run_triposg.py never
# loads it: its licence is Creative Commons non-commercial, and game-02 ships
# commercially. RESEARCH.md §1.4 shows the code path that makes it unnecessary
# (an RGBA input with binary alpha skips it), and run_triposg.py enforces that
# input rather than silently falling back.
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
      "numpy==1.26.4" \
      "diffusers>=0.31" \
      "transformers>=4.44" \
      einops \
      huggingface_hub \
      opencv-python-headless \
      trimesh \
      omegaconf \
      scikit-image \
      peft \
      jaxtyping \
      typeguard \
      pymeshlab \
 && pip install --no-cache-dir diso

# Fail the build here rather than 20 minutes into a GPU session if the CUDA
# extension did not actually compile.
RUN python -c "import diso, trimesh, pymeshlab, diffusers; print('diso/trimesh/pymeshlab/diffusers OK')"

COPY run_triposg.py /opt/spike3d/run_triposg.py
COPY glb_to_obj.py /opt/spike3d/glb_to_obj.py

ENV PYTHONPATH=/opt/triposg:/opt/triposg/scripts

WORKDIR /work
ENTRYPOINT ["python", "/opt/spike3d/run_triposg.py"]
CMD ["--help"]
