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
from typing import Any

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
# The project palette (the palette feature): the .gpl the art bible names as
# its pick - Lospec 2000 since 2026-09-14, installed by game-02's
# install_palette.py. `anim palette --lospec <slug>` / `--file <path>` rewrites
# it (with the .hex / .png / .json beside it) and this path's directory is
# also where the per-unit sub-palettes live (palette_units/<slug>.json). To
# try a candidate without installing it, point ANIM_PALETTE at it, e.g.
#   ANIM_PALETTE=/assets_src/palette_candidates/grim32/grim32.gpl
# Either a GIMP .gpl or a one-row .png is accepted; `post` converts whichever it
# gets into the one-row PNG that `magick -remap` needs, so the .gpl stays the
# human-editable source and the PNG is always derived.
PALETTE = Path(os.environ.get("ANIM_PALETTE", "/assets_src/palette.gpl"))
# Lospec's JSON endpoint for one palette, with the slug as the template's one
# field (anim.palettekit.fetch_lospec). Overridable so a test can serve the
# API's `POST /palette` subprocess from a file:// directory instead of the web.
LOSPEC_URL = os.environ.get("ANIM_LOSPEC_URL",
                            "https://lospec.com/palette-list/{slug}.json")

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
# The keyframe phase feature - the studio's phase template and frame budgets
# ---------------------------------------------------------------------------
# sprite-studio.md S2 "Animation": 12 fps, cap 12 frames, defaults idle 6 /
# walk 8 / run 8 / jump 10 / attack 12, and every clip picked against ONE
# shape - start -> wind-up -> action -> wind-down -> end -> (loop). This is the
# studio's contract for a clip; ACTION_FRAMES above stays the art bible's 2-6
# key-pose contract for the strips game-02 plays today. `pick --phases` reads
# this table, `pick` without it reads that one.
#
# A row has: `budget` (frames in the finished clip, at most PHASE_CAP), `loop`
# (a cycle whose last frame leads back into its first, or a one-shot that
# starts and ends at rest), `split` (the share of the budget each phase gets,
# in PHASE_NAMES order) and `span` (the share of the SOURCE clip's duration
# each phase covers, same order). Split and span are two different things on
# purpose - the ease. A one-shot rig clip holds the rest pose for its first
# fifth and recovers slowly over its last, so `start` and `end` cover a lot of
# the clip's TIME and get few FRAMES; the action covers a quarter of the time
# and gets a third of the frames. Frames per unit of clip time is the sample
# density, and dense-at-the-action / sparse-at-rest is what "ease in, ease
# out" means for a picker. A one-shot also spends ONE frame of its budget on
# anticipation: the wind-up gets an extra frame on top of its split, so the
# beat before the action is never a single cell. Loops get no anticipation
# frame; a cycle has no beat to anticipate.
#
# The five names are the studio's, and each maps to the pipeline verb the
# rest of the stages, the sidecar and game-02 know (attack <-> strike; the
# other four are the same word) plus the clip each vendored rig ships for it.
PHASE_NAMES: tuple[str, ...] = ("start", "wind-up", "action", "wind-down", "end")
PHASE_FPS = 12
PHASE_CAP = 12
ACTION_PHASES: dict[str, dict[str, Any]] = {
    "idle": {"budget": 6, "loop": True,
             "split": (0.15, 0.20, 0.30, 0.20, 0.15),
             "span": (0.20, 0.20, 0.20, 0.20, 0.20)},
    "walk": {"budget": 8, "loop": True,
             "split": (0.10, 0.25, 0.30, 0.25, 0.10),
             "span": (0.20, 0.20, 0.20, 0.20, 0.20)},
    "run": {"budget": 8, "loop": True,
            "split": (0.10, 0.25, 0.30, 0.25, 0.10),
            "span": (0.20, 0.20, 0.20, 0.20, 0.20)},
    "jump": {"budget": 10, "loop": False,
             "split": (0.08, 0.22, 0.35, 0.25, 0.10),
             "span": (0.15, 0.20, 0.30, 0.20, 0.15)},
    "attack": {"budget": 12, "loop": False,
               "split": (0.08, 0.20, 0.37, 0.20, 0.15),
               "span": (0.20, 0.20, 0.25, 0.20, 0.15)},
}
STUDIO_ACTIONS: dict[str, dict[str, Any]] = {
    "attack": {"verb": "strike",
               "clips": {"kaykit": "1H_Melee_Attack_Chop", "quaternius": "Sword_Attack"}},
    # `Idle_Loop` is the neutral stand - sword hanging, shield at the side -
    # which is the pose the concept pick shows and so the one the turnaround
    # feature's identity gate compares like with like; the UAL's other idle,
    # `Sword_Idle`, is a combat crouch with the blade raised.
    "idle": {"verb": "idle", "clips": {"kaykit": "Idle", "quaternius": "Idle_Loop"}},
    "walk": {"verb": "walk", "clips": {"kaykit": "Walking_A", "quaternius": "Walk_Loop"}},
    "run": {"verb": "run", "clips": {"kaykit": "Running_A", "quaternius": "Jog_Fwd_Loop"}},
    # The UAL ships the jump as three clips (Jump_Start / Jump_Loop / Jump_Land)
    # and no whole one; the studio's jump on that rig waits for a clip joiner.
    "jump": {"verb": "jump", "clips": {"kaykit": "Jump_Full_Short", "quaternius": None}},
}
# Per-project override, beside the palette in the customer game's assets_src/
# (the one other art decision that lives there). Rows merge over the defaults
# - `{"attack": {"budget": 10}}` changes the budget and keeps the split - and
# an optional `"units": {"<slug>": {"<action>": {...}}}` block overrides per
# unit. Keys may be the studio name or the pipeline verb. Missing file: the
# defaults above.
TIMING_FILE = Path(os.environ.get("ANIM_TIMING", "/assets_src/timing.json"))
# The loop-closure tolerances: a loop's last pick must land within this of its
# first - mean joint displacement in torso lengths when the keypoints are
# there (0.12, the picker's own "same pose" threshold, PICK_DEDUP), else the
# mean absolute pixel difference of the two frames as a fraction of full
# scale.
LOOP_TOLERANCE = float(os.environ.get("ANIM_LOOP_TOL", "0.12"))
LOOP_PIXEL_TOLERANCE = float(os.environ.get("ANIM_LOOP_PIXEL_TOL", "0.08"))


def studio_action(action: str) -> str | None:
    """The studio's name for an action given either spelling ("strike" and
    "attack" both give "attack"), or None when the studio has no such action."""
    key = action.strip().lower()
    if key in STUDIO_ACTIONS:
        return key
    for name, row in STUDIO_ACTIONS.items():
        if row["verb"] == key:
            return name
    return None


def phases_default(action: str) -> bool:
    """Whether `pick` picks against the phase template unless told otherwise.

    On only for a studio action the art bible does NOT budget - attack, run,
    jump. `strike`, `idle` and `walk` are art-bible verbs with existing jobs
    and 2-6 pose budgets, and those must keep producing the same bytes, so
    they stay on the old picker until `--phases` asks (the action matrix asks
    for every cell).
    """
    key = action.strip().lower()
    return studio_action(key) is not None and key not in ACTION_FRAMES


def _phase_vector(value: Any, what: str) -> tuple[float, ...]:
    if isinstance(value, dict):
        missing = [n for n in PHASE_NAMES if n not in value]
        if missing:
            raise ValueError(f"timing.json {what} is missing {missing}")
        value = [value[n] for n in PHASE_NAMES]
    values = tuple(float(v) for v in value)
    if len(values) != len(PHASE_NAMES) or any(v < 0 for v in values) or sum(values) <= 0:
        raise ValueError(f"timing.json {what} must be {len(PHASE_NAMES)} non-negative "
                         f"shares, one per phase {list(PHASE_NAMES)}")
    return values


def _merge_phase_row(base: dict[str, Any], override: dict[str, Any] | None,
                     what: str) -> dict[str, Any]:
    row = dict(base)
    for key, value in (override or {}).items():
        if key == "budget":
            row["budget"] = int(value)
        elif key == "loop":
            row["loop"] = bool(value)
        elif key in ("split", "span"):
            row[key] = _phase_vector(value, f"{what}.{key}")
        else:
            raise ValueError(f"timing.json {what}: unknown key {key!r}; one of "
                             f"budget, loop, split, span")
    return row


def read_timing_file(path: Path | None = None) -> dict[str, Any]:
    """The project's timing.json, or {} when there is none."""
    path = Path(path) if path is not None else TIMING_FILE
    if not path.exists():
        return {}
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold an object of action rows")
    return data


def phases_for_action(action: str, *, slug: str | None = None,
                      timing_file: Path | None = None) -> dict[str, Any] | None:
    """The phase row for one action - defaults, then the project's timing.json
    action row, then its unit row - or None when the studio has no such action.

    The row that comes back names the studio action (`action`) and the
    pipeline verb (`verb`) beside budget / loop / split / span. The budget is
    checked against PHASE_CAP here, so an over-budget override fails at
    lookup and not twelve frames into a render.
    """
    name = studio_action(action)
    if name is None:
        return None
    verb = STUDIO_ACTIONS[name]["verb"]
    row = dict(ACTION_PHASES[name])
    data = read_timing_file(timing_file)
    source = "template"
    for key in (name, verb):
        if key in data and key != "units":
            row = _merge_phase_row(row, data[key], key)
            source = "timing.json"
    units = data.get("units") or {}
    if slug:
        unit = units.get(slug.strip().lower()) or {}
        for key in (name, verb):
            if key in unit:
                row = _merge_phase_row(row, unit[key], f"units.{slug}.{key}")
                source = f"timing.json:{slug.strip().lower()}"
    if not 1 <= row["budget"] <= PHASE_CAP:
        raise ValueError(f"{name}: budget {row['budget']} is outside 1..{PHASE_CAP} "
                         f"(the studio caps a clip at {PHASE_CAP} frames)")
    return {"action": name, "verb": verb, "source": source, **row}


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
# The camera direction feature - the 16 views the rig route can render
# ---------------------------------------------------------------------------
# Two pitches x eight facings, 45 degrees apart (sprite-studio.md S2). A cell
# is named by the pitch's initial and the yaw: s0 .. s315, t0 .. t315.
#
# YAW is the character's turn about its own up axis, counter-clockwise seen
# from above, from the art bible's rest facing - so the compass reads the same
# in both pitches and matches a top-down game's eight directions:
#
#   yaw    faces          side view shows        top view: faces
#     0    screen right   the right profile      east  (the party rule: s0 IS today's output)
#    90    away           the back               north (up-screen)
#   180    screen left    the left profile       west  (what --mirror used to give)
#   270    the viewer     the front              south (down-screen)
#
# ELEVATION `side` is the profile camera every stage has used so far; `top`
# looks straight down, with the character's yaw-0 left side up-screen, so a
# figure facing east at t0 has its shield arm to the north. anim.rigkit
# rotates the rig into the view before its projection; nothing after the rig
# stage knows or cares which cell a job is.
DIRECTION_YAWS: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
DIRECTION_ELEVATIONS: tuple[str, ...] = ("side", "top")
DEFAULT_YAW = 0
DEFAULT_ELEVATION = "side"
DIRECTIONS: dict[str, tuple[int, str]] = {
    f"{elevation[0]}{yaw}": (yaw, elevation)
    for elevation in DIRECTION_ELEVATIONS for yaw in DIRECTION_YAWS}
DEFAULT_DIRECTION = f"{DEFAULT_ELEVATION[0]}{DEFAULT_YAW}"


def direction_name(yaw: int = DEFAULT_YAW, elevation: str = DEFAULT_ELEVATION) -> str:
    """The cell name for a (yaw, elevation) pair: (45, "side") -> "s45".

    Refuses anything outside the table, so a stage cannot write a direction
    the turnaround and the action matrix will never ask for.
    """
    elevation = str(elevation).strip().lower()
    if elevation not in DIRECTION_ELEVATIONS:
        raise ValueError(f"unknown elevation {elevation!r}; one of "
                         f"{list(DIRECTION_ELEVATIONS)}")
    try:
        yaw_value = int(yaw)
    except (TypeError, ValueError):
        raise ValueError(f"yaw must be one of {list(DIRECTION_YAWS)}, not {yaw!r}") from None
    if yaw_value != yaw or yaw_value not in DIRECTION_YAWS:
        raise ValueError(f"yaw {yaw!r} is not in the direction table; one of "
                         f"{list(DIRECTION_YAWS)} (45-degree steps)")
    return f"{elevation[0]}{yaw_value}"


def direction(name: str) -> tuple[int, str]:
    """A cell name back to its (yaw, elevation): "t135" -> (135, "top")."""
    key = str(name).strip().lower()
    if key not in DIRECTIONS:
        raise ValueError(f"unknown direction {name!r}; one of {list(DIRECTIONS)}")
    return DIRECTIONS[key]


def is_default_direction(yaw: int = DEFAULT_YAW,
                         elevation: str = DEFAULT_ELEVATION) -> bool:
    """True for the view every job had before the camera direction feature."""
    return direction_name(yaw, elevation) == DEFAULT_DIRECTION


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
# The clip render feature - `anim clip`, Wan 2.2 Fun-Control
# ---------------------------------------------------------------------------
# The generated clip is a video: every rig frame in, every frame out, and the
# canvas is the base render's portrait cell so the reference and the control
# video share one scale. Wan's latent takes 4 frames per step, so the control
# video is padded to 4n+1 by holding the last frame (`clip` records the map).
CLIP_CELL = BASE_CELL
CLIP_FRAME_STEP = 4
# One prompt for every action; the action-specific phrase is looked up below
# and the unit is the reference image, not the words. `--prompt` replaces it.
# The view phrase names the direction cell (the turnaround feature): the
# default cell's is the sentence every s0 clip was rendered with, word for
# word, and any other cell says which way the figure faces instead of
# "facing right".
CLIP_PROMPT = (
    "a single {view} pixel art character sprite, full body in "
    "frame, {phrase}, the same character and outfit as the reference image in "
    "every frame, plain flat white background, no ground shadow, no floor")
CLIP_VIEW_PHRASES: dict[str, str] = {
    "s0": "side-view, facing right",
    "s45": "three-quarter back view, seen from behind and to the right, facing away-right",
    "s90": "back view, seen from directly behind, facing away from the viewer",
    "s135": "three-quarter back view, seen from behind and to the left, facing away-left",
    "s180": "side-view, facing left",
    "s225": "three-quarter front view, facing toward the viewer's left",
    "s270": "front view, facing the viewer",
    "s315": "three-quarter front view, facing toward the viewer's right",
}
CLIP_TOP_VIEW = "top-down view, seen from directly above, {facing}"
CLIP_TOP_FACINGS: dict[int, str] = {
    0: "facing east (screen right)", 45: "facing north-east", 90: "facing north (up-screen)",
    135: "facing north-west", 180: "facing west (screen left)", 225: "facing south-west",
    270: "facing south (down-screen)", 315: "facing south-east",
}
CLIP_ACTION_PHRASES: dict[str, str] = {
    "strike": "swinging a sword in a melee attack",
    "hit": "flinching from a hit",
    "death": "struck down, falling and collapsing to the ground",
    "idle": "standing at rest, breathing",
    "walk": "walking",
    "shoot": "drawing and loosing a bow",
    "cast": "casting a spell",
}


def clip_view_phrase(direction_cell: str = DEFAULT_DIRECTION) -> str:
    """The camera phrase for one cell of the direction table."""
    yaw, elevation = direction(direction_cell)
    if elevation == "top":
        return CLIP_TOP_VIEW.format(facing=CLIP_TOP_FACINGS[yaw])
    return CLIP_VIEW_PHRASES[f"s{yaw}"]


def clip_prompt(action: str, direction_cell: str = DEFAULT_DIRECTION) -> str:
    action = action.strip().lower()
    phrase = CLIP_ACTION_PHRASES.get(action) or (f"performing a {action}" if action
                                               else "standing at rest")
    return CLIP_PROMPT.format(view=clip_view_phrase(direction_cell), phrase=phrase)


# ---------------------------------------------------------------------------
# The turnaround feature - `anim turnaround`, one idle still per direction
# ---------------------------------------------------------------------------
# sprite-studio.md S2: from the accepted base, one idle still per direction,
# and the identity gate sits on those stills before any clip is spent. v1 is
# the eight side facings; `--elevation top` is accepted by the code (the same
# eight yaws) and left to the top-down pitch feature (#285) to run.
TURNAROUND_ACTION = "idle"
TURNAROUND_RIG = "quaternius"          # the bake-off ruling (2026-09-21): Quaternius UAL
# The still is frame 0 of the idle clip, sampled TURNAROUND_CLIP_FRAMES times
# at t=0 so `anim clip` renders a 4n+1 video (Wan's latent holds 4 frames per
# slot; 5 is n=1, the shortest clip with a temporal window) of one held pose,
# and the frame in the middle of it becomes the still - the first frame of a
# Wan clip is the one the VAE encodes on its own and the one that leans on
# the reference most, the last is the padding end.
TURNAROUND_CLIP_FRAMES = 5
TURNAROUND_STILL_INDEX = TURNAROUND_CLIP_FRAMES // 2
# Route V as the bake-off ran it: skeleton over full-body depth at gain 0.35.
TURNAROUND_CONTROL = "blend"
TURNAROUND_DEPTH_GAIN = 0.35
TURNAROUND_CELL = CLIP_CELL
# A direction that has not answered in this long is recorded as failed and
# the run goes on to the next (the brief's budget: ~1 min of GPU, 5 at most).
TURNAROUND_TIMEOUT_S = float(os.environ.get("ANIM_TURNAROUND_TIMEOUT", "300"))


# ---------------------------------------------------------------------------
# The character concept feature - `anim concept`, Z-Image-Turbo + the pixel LoRA
# ---------------------------------------------------------------------------
# Text (or a sketch) to N candidate front-facing base characters, one of which
# the owner picks as the unit's reference (sprite-studio.md S2 "Base
# character"). No skeleton: the concept is a text-to-image pass with the
# pixel-art style LoRA on the same int8 Z-Image the pose stack loads, and an
# optional sketch through the Fun ControlNet Union as a scribble. Square, and
# the base pass's tall side, so the candidate is as big as a base render:
# nothing downstream reads this size (the pick lands as a base render and
# `post` trims to the opaque box), so a square is free here - the square-aspect
# rule is Qwen-Image-Edit's, which this pass never touches.
CONCEPT_CELL = (768, 768)
CONCEPT_COUNT = 4
# tarn59/pixel_art_style_lora_z_image_turbo (gpu/models.lock; Apache-2.0).
# Trigger and strength band from the model card and its Civitai mirror
# (docs/research/2026-09-20-pixel-art-loras.md): "Pixel art style.", 0.6-1.0.
# Strength 0 drops the LoRA node from the graph (ComfyUI rejects a missing file
# even at 0 - see anim.comfy.bypass).
CONCEPT_LORA: dict[str, Any] = {
    "repo": "tarn59/pixel_art_style_lora_z_image_turbo",
    "file": "pixel_art_style_z_image_turbo.safetensors",
    "trigger": "Pixel art style.",
    "strength": 0.8,
}
# The Union model picks the condition from the image alone (no mode input):
# a sketch goes in as white strokes on black, which is its scribble condition.
# 0.6 is the low end of the card's 0.65-1.0 band on purpose - a sketch is a
# hint for the silhouette, not a pose to trace.
CONCEPT_SKETCH_STRENGTH = 0.6
# One prompt shape for every unit: the trigger first, the owner's description
# in the middle, the base pass's framing and background rules after it so the
# matte in `post` has a flat white field to key on.
CONCEPT_PROMPT = (
    "{trigger} a single front-view pixel art character sprite of {description}, "
    "facing the viewer, standing at rest, full body in frame, feet at the bottom "
    "edge, chunky readable silhouette, bold dark outline, flat shaded, plain flat "
    "white background, no ground shadow, no floor")


def concept_prompt(description: str, trigger: str | None = None) -> str:
    trigger = CONCEPT_LORA["trigger"] if trigger is None else trigger
    return CONCEPT_PROMPT.format(trigger=trigger, description=description.strip()) \
        .replace("  ", " ").strip()


# The part palette lock feature - what each unit is made of, and where it is
# ---------------------------------------------------------------------------
# `anim post --parts` masks each finished frame into these parts (SAM 2.1,
# prompted from the rig's joints and prop outlines - anim.partkit) and snaps
# every pixel of a part to the colours that part has on the unit's accepted
# base sprite. Order is priority: an overlap goes to the earlier row, so props
# come first. Joint names are OpenPose-18; `mids` are (from, to, t) points
# along a bone, t past 1 extrapolating (the crown of the helmet). Only units
# whose kit is settled are listed; a unit with no row cannot be locked.
from anim.partkit import Part  # noqa: E402  (a frozen dataclass; no cycle)

UNIT_PARTS: dict[str, tuple[Part, ...]] = {
    "knight": (
        Part("sword", prop="sword"),
        Part("shield", prop="shield"),
        Part("helmet", joints=("nose", "right_eye", "left_eye", "right_ear",
                               "left_ear"),
             mids=(("neck", "nose", 1.35),)),
        Part("arms", joints=("right_elbow", "left_elbow"),
             mids=(("right_shoulder", "right_elbow", 0.5),
                   ("right_elbow", "right_wrist", 0.5),
                   ("left_shoulder", "left_elbow", 0.5),
                   ("left_elbow", "left_wrist", 0.5))),
        Part("torso", joints=("neck",),
             mids=(("neck", "right_hip", 0.5), ("neck", "left_hip", 0.5),
                   ("neck", "right_hip", 0.8), ("neck", "left_hip", 0.8))),
        Part("legs", joints=("right_knee", "left_knee"),
             mids=(("right_hip", "right_knee", 0.5),
                   ("right_knee", "right_ankle", 0.5),
                   ("left_hip", "left_knee", 0.5),
                   ("left_knee", "left_ankle", 0.5),
                   ("right_knee", "right_ankle", 0.95),
                   ("left_knee", "left_ankle", 0.95))),
    ),
}


def parts_for_unit(slug: str) -> tuple[Part, ...]:
    return UNIT_PARTS.get(slug.strip().lower(), ())


# Off by default for every unit: `post` with the lock off must reproduce the
# bytes it produced before the feature existed. Turn it on per unit here once
# its base sprite has part palettes, or per run with `anim post --parts`.
PART_LOCK_DEFAULT: dict[str, bool] = {}


def part_lock_default(slug: str) -> bool:
    return PART_LOCK_DEFAULT.get(slug.strip().lower(), False)


# A part whose chosen mask scores below this CONSISTENCY (its positive prompt
# points inside minus twice the other parts' points inside; anim.partkit
# .choose_candidate, which returns a negative value when no candidate held
# the prompt) is "not found": skipped for that frame, said so in the report,
# never locked to a guess. 0 = "a candidate that fit the prompt existed".
PART_MIN_SCORE = float(os.environ.get("ANIM_PART_MIN_SCORE", "0"))

# A part with fewer opaque pixels than this on the base sprite yields no
# palette (recorded as rejected in parts.json) - so an action never locks a
# whole shield to the three pixels of shield the standing pose happens to show.
PART_MIN_PALETTE_PIXELS = int(os.environ.get("ANIM_PART_MIN_PALETTE_PIXELS", "8"))

# SAM 2.1 hiera-small: Apache-2.0, ungated. Pinned like gpu/models.lock pins
# the GPU weights - repo, revision, file, size, sha256 read from the Hub API on
# 2026-09-17 (both ?blobs=true and the revision's paths-info agreed) - and
# verified by anim.partkit.ensure_weights before it is ever loaded. The
# checkpoint lands on the bind-mounted cache like rtmlib's and rembg's.
SAM2_WEIGHTS: dict[str, Any] = {
    "repo": "facebook/sam2.1-hiera-small",
    "revision": "ee5bba1d82bb8749febdf90f45e84b687142ba03",
    "repo_file": "sam2.1_hiera_small.pt",
    "file": "sam2.1_hiera_small.pt",
    "size": 184416285,
    "sha256": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    "license": "apache-2.0",
    # The hydra config name inside the `sam2` package that matches the file.
    "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
}
SAM2_CACHE_DIR = Path(os.environ.get(
    "ANIM_SAM2_CACHE", str(Path(os.environ.get("XDG_CACHE_HOME", "/work/.cache")) / "sam2")))

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
# The pipeline API feature - FastAPI over work/, on the LAN
# ---------------------------------------------------------------------------
# LAN port for the API (and the pages it serves). 8720 is the sprite studio's
# port (sprite-studio.md S2 "Build base"); it was free on 2026-09-21 (`ss -ltn`:
# the neighbours are 8700 openproject, 8770 research-digest, 8787 bg3-wiki). It
# replaces the review page's 8710. Change here, not in the app - and update
# compose.yml's published port to match.
API_PORT = int(os.environ.get("ANIM_API_PORT", "8720"))
# Bind every interface: the owner opens this from the Mac, not from this host.
API_BIND = os.environ.get("ANIM_API_BIND", "0.0.0.0")
# CPU stages the API will run at the same time. Small on purpose: `pose` and
# `post` each want every core, and two of them at once are slower than one
# after the other. GPU stages are never concurrent - see anim.api.runs.
API_CPU_SLOTS = int(os.environ.get("ANIM_API_CPU_SLOTS", "2"))
# The built editor (the web pixel editor feature): `web/dist/`, served at `/`
# by the API. Two places it can be, tried in order when ANIM_WEB_DIST is not
# set: the checkout's own `web/dist` (what `npm run build` on the host makes;
# `/work/web/dist` in the container through the bind mount) and the copy the
# Dockerfile's node stage bakes into the image at /opt/anim/web/dist. The
# checkout wins so a host rebuild shows up without a `docker compose build`;
# the image copy means a fresh clone gets the editor from the image alone.
# Neither there: the API serves its own minimal jobs + review pages.
_WEB_DIST_CANDIDATES = (
    Path(__file__).resolve().parents[1] / "web" / "dist",
    Path("/opt/anim/web/dist"),
)
WEB_DIST = Path(os.environ["ANIM_WEB_DIST"]) if os.environ.get("ANIM_WEB_DIST") else next(
    (p for p in _WEB_DIST_CANDIDATES if (p / "index.html").exists()),
    _WEB_DIST_CANDIDATES[0])
# Only ever used to PRINT a URL the owner can click.
LAN_HOST = os.environ.get("ANIM_LAN_HOST", "192.168.1.158")
