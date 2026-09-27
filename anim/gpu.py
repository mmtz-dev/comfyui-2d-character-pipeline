"""Booking the GPU through the shared GPU gatekeeper (~/Repos/gpu-gatekeeper, Feature 6).

The A5000 is shared with clone-studio and other projects. Before a GPU stage sends its first
prompt to ComfyUI it books memory from the gatekeeper (http://192.168.1.158:7886), which queues
it while the card is full and frees idle ComfyUIs (this project's and clone-studio's) when it
has to. The stage holds the lease until its process ends.

How it is wired, in three calls:

  1. `anim.cli.main` calls `stage(name)` for a stage in STAGE_PROFILES: it picks the stage's
     job profile and releases the lease when the stage returns (also on Cancel's SIGTERM).
  2. `ComfyClient.post_prompt` calls `ensure_lease()`: the lease is booked lazily, on the first
     prompt, so `run` (which renders only when it can) books nothing for its CPU steps, and a
     stage refused by a gate books nothing at all.
  3. `ComfyClient.wait` calls `loaded()` when a prompt has finished: the model is on the GPU,
     which lets the gatekeeper learn the stage's real size.

Nothing happens outside a stage (tests, one-off scripts): no profile, no lease. If the
gatekeeper can't be reached the stage runs anyway after ~5 s (the client's fail-open) and
says so. URL: env GPU_GATEKEEPER_URL (compose.yml sets http://host.docker.internal:7886).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
from typing import Callable, Iterator

from anim import gpu_lease

# Stage -> gatekeeper job profile, sized for the heaviest model the stage loads. The seed sizes
# live in gpu-gatekeeper's config/gatekeeper.toml; the gatekeeper learns the real ones.
#   anim-zimage     Z-Image Turbo INT8 + ControlNet (base)
#   anim-qwen-edit  Qwen-Image-Edit-2509 INT4 + 7B text encoder (render's identity pass,
#                   animate)
#   anim-concept    concept: Z-Image passes or the Qwen-Edit view pass. Its own profile so
#                   its lighter Z-Image runs never teach anim-qwen-edit a size too small
#   anim-wan        Wan 2.2 Fun-Control 5B bf16 (clip, and matrix / bakeoff, which clip)
# Keep the keys equal to anim.api.runs.GPU_STAGES (tests/test_gpu_lease.py checks it).
STAGE_PROFILES: dict[str, str] = {
    "base": "anim-zimage",
    "render": "anim-qwen-edit",
    "run": "anim-qwen-edit",
    "concept": "anim-concept",
    "animate": "anim-qwen-edit",
    "clip": "anim-wan",
    "matrix": "anim-wan",
    "bakeoff": "anim-wan",
}
SERVICE = "anim-pipeline-comfyui"  # its name on the gatekeeper's allowlist

# Swapped in tests for a fake; the real one is the synced gatekeeper client.
ACQUIRE: Callable[..., gpu_lease.Lease] = gpu_lease.acquire

_lock = threading.Lock()
_state: dict = {"stage": None, "profile": None, "lease": None, "loaded": False}


def url() -> str:
    return os.environ.get("GPU_GATEKEEPER_URL") or gpu_lease.DEFAULT_URL


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    """Run a stage: GPU stages book a lease on their first prompt; released on exit."""
    profile = STAGE_PROFILES.get(name)
    with _lock:
        _state.update(stage=name, profile=profile, lease=None, loaded=False)
    try:
        yield
    finally:
        release()
        with _lock:
            _state.update(stage=None, profile=None)


def ensure_lease() -> None:
    """Book the stage's memory before its first prompt (no-op outside a GPU stage or once held)."""
    with _lock:
        profile, held, name = _state["profile"], _state["lease"], _state["stage"]
    if profile is None or held is not None:
        return
    print(f"GPU: booking {profile} from the GPU gatekeeper ({url()})", flush=True)
    lease = ACQUIRE(profile, holder=f"anim-pipeline {name}", service=SERVICE, url=url(), cap_torch=False)
    if lease.guarded:
        print(f"GPU: lease {lease.id} granted, {lease.gb} GB", flush=True)
    else:
        print("GPU: gatekeeper unreachable, running without a lease", flush=True)
    with _lock:
        _state["lease"] = lease


def loaded() -> None:
    """A prompt finished, so the model is on the GPU: tell the gatekeeper once."""
    with _lock:
        lease, done = _state["lease"], _state["loaded"]
        if lease is None or done:
            return
        _state["loaded"] = True
    lease.mark_loaded()


def release() -> None:
    with _lock:
        lease, _state["lease"] = _state["lease"], None
    if lease is not None:
        lease.release()


atexit.register(release)  # a stage that ends through sys.exit still gives the memory back
