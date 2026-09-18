"""Paths and settings shared by every stage.

Everything here is read from the environment with a container-appropriate
default, so nothing is hardcoded to one machine. A stage NEVER hardcodes a path
or a number that appears here - the whole point is that the owner can retarget
the pipeline (different palette, different cell size, different action budget)
without editing a stage.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# The GPU image service feature
# ---------------------------------------------------------------------------
# ComfyUI runs on the NATIVE Docker daemon; this package runs on Docker Desktop.
# Two engines on one host, so the hostname that works from inside this container
# is host.docker.internal, not localhost.
#   from the Fedora host or the Mac:  http://192.168.1.158:8188
COMFYUI_URL: str = os.environ.get(
    "COMFYUI_URL", "http://host.docker.internal:8188"
).rstrip("/")

# Where ComfyUI reads LoadImage filenames from and writes SaveImage output to,
# as seen from THIS container. The GPU container's /basedir/input and
# /basedir/output are bind-mounted here read-write by compose.yml so a strip can
# be handed over without an HTTP upload. `render` prefers the HTTP /upload/image
# endpoint (it works even when the share is not mounted) and falls back to a
# file copy here; both are exercised by the fake-server tests.
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/comfy/input"))
COMFY_OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT_DIR", "/comfy/output"))

# ---------------------------------------------------------------------------
# Pipeline directories (mounted by compose.yml)
# ---------------------------------------------------------------------------
# Watched drop folder: filmed clips land here (profile view, plain background,
# full body in frame, one action per clip).
CAPTURE_DIR = Path(os.environ.get("ANIM_CAPTURE_DIR", "/work/capture"))
# Per-job scratch: frames, keypoints.json, skeleton PNGs, picks.json, strips,
# raw generations, post-chain intermediates. Regenerable; gitignored.
WORK_DIR = Path(os.environ.get("ANIM_WORK_DIR", "/work/work"))
# The workflows POSTed by `render`.
WORKFLOW_DIR = Path(os.environ.get("ANIM_WORKFLOW_DIR", "/work/gpu/workflows"))
# repo + revision + sha256 of every weight, written by gpu/relock_models.py.
# `render` copies the rows for the files a workflow names into provenance.json,
# and `publish` copies them again into the sidecar's generation.models.
MODELS_LOCK = Path(os.environ.get("ANIM_MODELS_LOCK", "/work/gpu/models.lock"))
# game-02's published assets. `publish` writes here; the Mac edits the same
# bytes over the Samba share.
ASSETS_DIR = Path(os.environ.get("ANIM_ASSETS_DIR", "/assets"))
# The game-02 repo, mounted read-only, purely so `publish` can run its
# stdlib-only asset audit in this container instead of asking the owner to run
# it on the host. Absent -> `publish` says so and refuses (--no-audit skips).
GAME02_DIR = Path(os.environ.get("ANIM_GAME02_DIR", "/game02"))
AUDIT_SCRIPT = Path(os.environ.get(
    "ANIM_AUDIT_SCRIPT", str(GAME02_DIR / "tools" / "asset_audit.py")))

# ---------------------------------------------------------------------------
# Palette - the art bible's single source of truth for the colour snap
# ---------------------------------------------------------------------------
# Defaults to the .gpl the art bible names as the winner of the palette
# shootout. THAT FILE DOES NOT EXIST YET (the owner has not picked); until it
# does, point ANIM_PALETTE at one of the candidates, e.g.
#   ANIM_PALETTE=/assets_src/palette_candidates/grim32/grim32.gpl
# Either a GIMP .gpl or a one-row .png is accepted; `post` converts whichever it
# gets into the one-row PNG that `magick -remap` needs, so the .gpl stays the
# human-editable source and the PNG is always derived.
PALETTE = Path(os.environ.get("ANIM_PALETTE", "/assets_src/palette.gpl"))

# ---------------------------------------------------------------------------
# Fixed contract with game-02 (see games-design-wiki .../art-bible.md)
# ---------------------------------------------------------------------------
# One 64x64 PNG per frame, bottom-center pivot, side view, party faces right.
CANVAS = (64, 64)
# Alpha must be strictly binary - no soft edges on a 64px sprite.
ALPHA_VALUES = (0, 255)
# Default strip geometry: NxN1, NON-SQUARE on purpose. A square canvas triggers
# Qwen-Image-Edit's square-aspect degradation (upstream issue #243).
STRIP_CELLS = 4
STRIP_CELL = (304, 320)  # -> 1216x320; both dims divisible by 16 for the latent
# EmptySD3LatentImage has step 16 on both dimensions, so every generated canvas
# must be a multiple of this. `strip` asserts it rather than discovering it in
# a ComfyUI traceback twenty minutes into a batch.
LATENT_STEP = 16
# `base` generates a standing sprite at one portrait cell - also non-square.
BASE_CELL = (512, 768)

# `extract` caps the decode rate; 12 fps is plenty to find key poses, and it is
# the same 12 the game plays frames back at (render/unit_sprites.gd DEFAULT_FPS).
EXTRACT_FPS = int(os.environ.get("ANIM_EXTRACT_FPS", "12"))

# ---------------------------------------------------------------------------
# Frame budget per action (art bible S7 "Frame budget" + S8 "Action vocabulary")
# ---------------------------------------------------------------------------
# Key poses, not a smooth cycle. Every value is inside the art bible's 2-6 band.
# Two rows deviate from the bible's table on purpose and are called out here so
# the difference is never a surprise:
#   death  bible says 3, this says 4 - a fatal beat is 0.22s and the extra pose
#          buys the "settle" that reads as weight; revisit at the anchor gate.
#   walk   bible says 4, this says 6 - walk is a continuous LOOP at its own
#          frame rate rather than a per-hop cycle, so it can afford more.
# Anything not listed falls back to DEFAULT_FRAMES.
DEFAULT_FRAMES = 4
ACTION_FRAMES: dict[str, int] = {
    # combat verbs (AttackChoreography.Verb, lowercased)
    "strike": 4,
    "shoot": 4,
    "cast": 4,
    "shove": 3,
    "bless": 3,
    "conjure": 3,
    "invoke": 3,
    "guard_set": 2,
    "reach": 3,
    "throw": 3,
    "pickup": 2,
    "dash": 2,
    "equip": 2,
    "rescue": 3,
    # reactions
    "hit": 2,
    "death": 4,
    "collapse": 3,
    "revive": 3,
    # the two that belong to no verb
    "idle": 2,
    "walk": 6,
}

# Actions the art bible says have NO motion, so nothing should be filmed for
# them. `pick` refuses rather than quietly producing frames nobody plays.
NO_MOTION_ACTIONS = {"none", "wait"}


def frames_for_action(action: str) -> int:
    """The art bible's key-pose budget for one action."""
    return ACTION_FRAMES.get(action.strip().lower(), DEFAULT_FRAMES)


# ---------------------------------------------------------------------------
# Timing shape per action (the strip timing feature; art bible S7)
# ---------------------------------------------------------------------------
# Same frame count, different weight, purely from holds: industry strips hold
# the anticipation and the contact pose and rush the transit (Stardew's axe is
# 150/40/40/170/75 ms; a GBA Fire Emblem swing is 3-3-3-4 | 1-1-1-3-10 ticks).
# Keyed by (action, n) because with 2-6 poses the ROLES are fixed by position -
# a 4-pose strike is rest / anticipation / contact / recover whatever the weapon,
# and Stardew's axe and Slynyrd's sword differ only in the hold numbers.
#
# `weights` are dimensionless ticks, one per frame, never seconds: the game's
# AttackChoreography already scales every duration by the speed setting, and
# seconds in a sidecar would fight it. `anchor` is the frame whose START lands
# on the plan's anchor instant - impact for a strike, the launch for a shot,
# frame 0 for a reaction - and the game holds it through the hit-stop.
#
# A template rather than something derived from pick.signals: the deviation
# peak is not reliably the contact pose (an overhead windup deviates more from
# rest than the forward cut), so deriving is UNVERIFIED and hand edits in
# picks.json win instead. Tuned at the anchor gate; an (action, n) with no row
# publishes no `contract.timing` and the game plays it uniform.
ACTION_TIMING: dict[tuple[str, int], tuple[list[int], int]] = {
    # rest, anticipation, contact, recover
    ("strike", 4): ([2, 3, 4, 3], 2),
    ("shove", 4): ([2, 3, 4, 3], 2),
    # rest, push, recover (ACTION_FRAMES gives shove 3)
    ("shove", 3): ([2, 4, 3], 1),
    # rest, anticipation, smear, contact, recover - the smear slot (deferred)
    ("strike", 5): ([2, 3, 1, 4, 3], 3),
    # rest, draw, release, (recover)
    ("shoot", 3): ([2, 4, 3], 2), ("shoot", 4): ([2, 4, 3, 2], 2),
    ("cast", 3): ([2, 4, 3], 2), ("cast", 4): ([2, 4, 3, 2], 2),
    ("throw", 3): ([2, 4, 3], 2), ("throw", 4): ([2, 4, 3, 2], 2),
    # flinch, recover / rest, flinch, ease, recover
    ("hit", 2): ([3, 2], 0), ("hit", 4): ([1, 3, 2, 2], 0),
    # standing, struck, falling, lying
    ("death", 4): ([1, 2, 2, 4], 0),
}


def timing_for_action(action: str, n: int) -> tuple[list[int], int] | None:
    """The template shape for `n` poses of one action, or None when there is no row."""
    row = ACTION_TIMING.get((action.strip().lower(), int(n)))
    return (list(row[0]), row[1]) if row else None


# ---------------------------------------------------------------------------
# The scale ladder (art bible S3) - sprite height inside the 64px canvas
# ---------------------------------------------------------------------------
# `post` scales each unit to its band before the bottom-center recanvas, which
# is how a hound stops being the same height as the boss. Revisit at the
# anchor-unit gate; these are the midpoints of the bible's bands.
DEFAULT_UNIT_HEIGHT = 53
UNIT_HEIGHT: dict[str, int] = {
    # low / bestial 34-40
    "hound": 37,
    # light 46-50
    "archer": 48, "chanter": 48, "lurker": 48, "mage": 48,
    # standard 50-56
    "knight": 53, "pikeman": 53, "raider": 53, "ranger": 53, "shaman": 53,
    "vanguard": 53, "warden": 53, "wight": 53, "wizard": 53,
    # boss 62-64, and visibly the broadest
    "barrow_warden": 63,
    # objective object - fills its hex, not on the human ladder
    "barrow_heart": 64,
}


def height_for_unit(slug: str) -> int:
    """Target sprite height in the 64px canvas for one unit slug."""
    return UNIT_HEIGHT.get(slug.strip().lower(), DEFAULT_UNIT_HEIGHT)


# ---------------------------------------------------------------------------
# The skeleton props feature - what each unit holds, drawn onto its skeleton
# ---------------------------------------------------------------------------
# `anim rig` draws these onto the ControlNet image so the model is TOLD which
# hand is the sword hand (anim.rigkit, PROPS). Kinds are rigkit.PROP_KINDS.
# Only units whose kit is settled in the art bible are listed; anything else
# gets no props unless `--props` says otherwise.
UNIT_PROPS: dict[str, tuple[str, ...]] = {
    "knight": ("sword", "shield"),
}


def props_for_unit(slug: str) -> tuple[str, ...]:
    return UNIT_PROPS.get(slug.strip().lower(), ())


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------
# rtmlib model family. 'wholebody' is DWPose (133 keypoints -> openpose134 when
# to_openpose=True); 'body' is the lighter 17 -> openpose18 body rig. Both are
# drawn with rtmlib's OWN draw_skeleton(openpose_skeleton=True) - never a
# hand-authored rig, which Z-Image's ControlNet rejects (gpu/workflows/NOTES.md).
POSE_MODEL = os.environ.get("ANIM_POSE_MODEL", "wholebody")
POSE_MODE = os.environ.get("ANIM_POSE_MODE", "balanced")  # lightweight|balanced|performance
# Below this a joint is treated as absent - by the picker and by the renderer.
POSE_KPT_THR = float(os.environ.get("ANIM_POSE_KPT_THR", "0.4"))
# rtmlib downloads its ONNX weights under $XDG_CACHE_HOME/rtmlib. The Dockerfile
# points XDG_CACHE_HOME at /work/.cache, which is the bind-mounted repo, so the
# ~200 MB of models download once and survive a container recreate.

# ---------------------------------------------------------------------------
# The key-pose picker
# ---------------------------------------------------------------------------
# Two normalised poses closer than this are "the same pose" and only one is
# kept. Units are torso lengths (shoulder-centre to hip-centre == 1.0), so the
# number means something: 0.12 is a limb moving about an eighth of a torso.
PICK_DEDUP = float(os.environ.get("ANIM_PICK_DEDUP", "0.12"))

# ---------------------------------------------------------------------------
# The review page
# ---------------------------------------------------------------------------
# LAN port for the click-to-pick page. 8710 was free on 2026-09-13 and is not in
# the host's port table (8700 openproject, 8770 research-digest, 8787 bg3-wiki
# are the neighbours). Change here, not in the stage - and update compose.yml's
# published port to match.
REVIEW_PORT = int(os.environ.get("ANIM_REVIEW_PORT", "8710"))
# Bind every interface: the owner opens this from the Mac, not from this host.
REVIEW_BIND = os.environ.get("ANIM_REVIEW_BIND", "0.0.0.0")
# Only ever used to PRINT a URL the owner can click.
LAN_HOST = os.environ.get("ANIM_LAN_HOST", "192.168.1.158")
