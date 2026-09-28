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
from typing import Any, Sequence

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
# The concept generation feature: which locked weight is a checkpoint, a LoRA,
# a text encoder... and for which workflow family (zimage | qwen-edit | wan).
# Hand-curated beside the lock (the lock is generated and fixed at 8 columns);
# `GET /models` and the studio's model / LoRA picker read it.
MODELS_REGISTRY = Path(os.environ.get("ANIM_MODELS_REGISTRY",
                                      "/work/gpu/models.registry.tsv"))
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
#
# `view_keys` on: the action starts (and, unless it loops, ends) standing as
# the direction's view shows, so those keyframes are a copy of the view by
# default rather than a render - `view_key_phases`, the view keys feature.
PHASE_NAMES: tuple[str, ...] = ("start", "wind-up", "action", "wind-down", "end")
PHASE_FPS = 12
PHASE_CAP = 12
ACTION_PHASES: dict[str, dict[str, Any]] = {
    "idle": {"budget": 6, "loop": True,
             "split": (0.15, 0.20, 0.30, 0.20, 0.15),
             "span": (0.20, 0.20, 0.20, 0.20, 0.20), "view_keys": True},
    "walk": {"budget": 8, "loop": True,
             "split": (0.10, 0.25, 0.30, 0.25, 0.10),
             "span": (0.20, 0.20, 0.20, 0.20, 0.20), "view_keys": True},
    "run": {"budget": 8, "loop": True,
            "split": (0.10, 0.25, 0.30, 0.25, 0.10),
            "span": (0.20, 0.20, 0.20, 0.20, 0.20), "view_keys": True},
    "jump": {"budget": 10, "loop": False,
             "split": (0.08, 0.22, 0.35, 0.25, 0.10),
             "span": (0.15, 0.20, 0.30, 0.20, 0.15), "view_keys": True},
    "attack": {"budget": 12, "loop": False,
               "split": (0.08, 0.20, 0.37, 0.20, 0.15),
               "span": (0.20, 0.20, 0.25, 0.20, 0.15), "view_keys": True},
}
STUDIO_ACTIONS: dict[str, dict[str, Any]] = {
    "attack": {"verb": "strike",
               "clips": {"kaykit": "1H_Melee_Attack_Chop", "quaternius": "Sword_Attack"}},
    # `Idle_Loop` is the neutral stand - sword hanging, shield at the side -
    # which is the pose the concept pick shows, so an idle cell compares like
    # with like against its reference view; the UAL's other idle,
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
        elif key in ("loop", "view_keys"):
            row[key] = bool(value)
        elif key in ("split", "span"):
            row[key] = _phase_vector(value, f"{what}.{key}")
        else:
            raise ValueError(f"timing.json {what}: unknown key {key!r}; one of "
                             f"budget, loop, split, span, view_keys")
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


def view_key_phases(action: str, *, slug: str | None = None,
                    timing_file: Path | None = None) -> tuple[str, ...]:
    """The keyframe phases whose picture is, by default, a copy of the
    direction's view (the view keys feature): the start, and the end too
    unless the action loops - a loop's end has to flow back into its start,
    and two copies of the view in a row would hold at the seam. Empty when the
    action's `view_keys` is off, or the studio has no such action."""
    row = phases_for_action(action, slug=slug, timing_file=timing_file)
    if row is None or not row.get("view_keys"):
        return ()
    return ("start",) if row["loop"] else ("start", "end")


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
    # light 46-50. `hooded_archer` is spelled out beside `archer` because
    # `height_for_unit` matches the whole slug, not a word in it: the studio's
    # first archer is named `hooded_archer` and was silently taking
    # DEFAULT_UNIT_HEIGHT (53, the standard body) until 2026-09-24.
    "archer": 48, "hooded_archer": 48,
    "chanter": 48, "lurker": 48, "mage": 48,
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
    the action matrix will never ask for.
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
# The view phrase names the direction cell (the camera direction feature): the
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
    # The action matrix feature (#278). Two of the five studio actions had no
    # row, so `clip_prompt` fell through to "performing a run" / "performing a
    # jump" - grammatical, and a weaker description than every other action
    # gets. The matrix renders all five, so the two gaps stop being theoretical.
    # Keyed by the PIPELINE VERB, like every row above (`run` and `jump` happen
    # to be the same word as their studio action; `attack` is `strike`).
    # Wording follows the rows above: a present participle naming the motion,
    # no camera and no style - the view phrase and the style come from
    # CLIP_PROMPT, and a phrase that repeated them would fight the view
    # sentence on a back-facing cell.
    "run": "running at full stride",
    "jump": "leaping, feet off the ground",
}


# The multi-view reference feature (#298): when the unit's base carries a
# parts phrase (`anim concept --pick ... --parts`, or `--parts` alone), the
# prompt also NAMES the parts that must survive the turn - the reference
# shows a front view and cannot say what the back of a closed helm looks
# like - and says, per view, what that side of the head and body shows. The
# negative gets the one thing no sprite may have (a second figure) and the
# unit's own `--avoid` phrase ("bare head, uncovered face" for a helmeted
# knight; a bare-headed unit says nothing). Without a parts phrase the
# prompt is the sentence above, word for word, so every job rendered before
# this feature reproduces.
CLIP_PARTS_PROMPT = (
    "a single {view} pixel art character sprite, full body in frame, {phrase}, "
    "the same character and outfit as the reference image in every frame, "
    "with the same {parts} as the reference image, {identity}, exactly one figure, "
    "plain flat white background, no ground shadow, no floor")
# What each side facing shows of the head and body - generic geometry; the
# unit's parts phrase says what is on the head.
CLIP_IDENTITY_PHRASES: dict[str, str] = {
    "s0": "seen in profile: the side of the head and body, the head covered exactly as in the reference",
    "s45": "seen from behind at an angle: the back and one side of the head and body, "
           "the head covered exactly as in the reference, no face visible",
    "s90": "seen from directly behind: the back of the head and the back of the body, "
           "what hangs from the shoulders, the head covered exactly as in the reference, no face visible",
    "s135": "seen from behind at an angle: the back and one side of the head and body, "
            "the head covered exactly as in the reference, no face visible",
    "s180": "seen in profile: the side of the head and body, the head covered exactly as in the reference",
    "s225": "seen from the front at an angle: the front and one side of the head and body, "
            "the head covered exactly as in the reference",
    "s270": "seen from the front: the front of the head and body, the head covered exactly as in the reference",
    "s315": "seen from the front at an angle: the front and one side of the head and body, "
            "the head covered exactly as in the reference",
}
CLIP_TOP_IDENTITY = ("seen from above: the top of the head and the shoulders, the head "
                     "covered exactly as in the reference")
# Appended to the workflow's own negative (Wan's standard one) whenever the
# parts prompt is used; `--avoid` adds the unit's phrase after it.
CLIP_NEGATIVE_EXTRA = "a second figure, two people, an extra person"


def clip_view_phrase(direction_cell: str = DEFAULT_DIRECTION) -> str:
    """The camera phrase for one cell of the direction table."""
    yaw, elevation = direction(direction_cell)
    if elevation == "top":
        return CLIP_TOP_VIEW.format(facing=CLIP_TOP_FACINGS[yaw])
    return CLIP_VIEW_PHRASES[f"s{yaw}"]


def clip_identity_phrase(direction_cell: str = DEFAULT_DIRECTION) -> str:
    """What this cell shows of the head and body (the parts prompt only)."""
    yaw, elevation = direction(direction_cell)
    if elevation == "top":
        return CLIP_TOP_IDENTITY
    return CLIP_IDENTITY_PHRASES[f"s{yaw}"]


def clip_prompt(action: str, direction_cell: str = DEFAULT_DIRECTION,
                parts: str | None = None) -> str:
    """The clip's positive prompt. With no `parts` it is the sentence every
    clip before the multi-view reference feature was rendered with; with the
    unit's parts phrase it names them and the view (CLIP_PARTS_PROMPT)."""
    action = action.strip().lower()
    phrase = CLIP_ACTION_PHRASES.get(action) or (f"performing a {action}" if action
                                               else "standing at rest")
    parts = (parts or "").strip().rstrip(",.")
    if not parts:
        return CLIP_PROMPT.format(view=clip_view_phrase(direction_cell), phrase=phrase)
    return CLIP_PARTS_PROMPT.format(view=clip_view_phrase(direction_cell), phrase=phrase,
                                    parts=parts, identity=clip_identity_phrase(direction_cell))


def clip_negative(template: str, avoid: str | None = None) -> str:
    """The workflow's negative with the feature's extras after it."""
    extras = [CLIP_NEGATIVE_EXTRA]
    avoid = (avoid or "").strip().rstrip(",.")
    if avoid:
        extras.append(avoid)
    return ", ".join([template.strip().rstrip(",")] + extras) if template.strip() \
        else ", ".join(extras)


# ---------------------------------------------------------------------------
# The rig the Animate tab poses from, and the view each direction wants
# ---------------------------------------------------------------------------
# The skeleton under a frame comes from this rig's clip for the action (the
# bake-off's ruling, 2026-09-21: the Quaternius UAL rig).
ANIMATE_RIG = "quaternius"
# The multi-view reference feature (#298): which of the base's views each
# direction wants, best first ("front" is the base itself). By the compass
# above: yaw 0 shows the character's RIGHT profile, 180 the left, 90 the
# back, 270 the front. The Animate tab draws a direction from its OWN view
# only (`animate.reference`); this table is what the views ring reports as
# `wanted_by` - which facing goes without a reference while a view is missing.
REFERENCE_BY_DIR: dict[str, tuple[str, ...]] = {
    "s0": ("right", "front"),
    "s45": ("back_right", "back", "front"),
    "s90": ("back", "front"),
    "s135": ("back_left", "back", "front"),
    "s180": ("left", "front"),
    "s225": ("front_left", "front"),
    "s270": ("front",),
    "s315": ("front_right", "front"),
}


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


# The multi-view reference feature (#298): `anim concept --views` renders the
# accepted base from other sides so the matrix can hand Wan a reference
# that already shows that side. The views are named for the SIDE OF THE
# CHARACTER the camera sees, which is what the compass in DIRECTIONS calls
# them: "right" is the right profile (yaw 0, facing screen right), "left" the
# left profile (yaw 180), "back" yaw 90, and the obliques by the two sides
# they show. "front" is the base itself and is never rendered.
# `elevation` is "side" for the seven eye-level views and "top" for the
# top-down one (the concept generation feature, #285): the top cells of the
# matrix take the top view first, then their yaw's row.
CONCEPT_VIEWS: dict[str, dict[str, Any]] = {
    "back": {"yaw": 90, "elevation": "side",
             "sentence": "from directly behind, turned all the way round: "
                         "the back of the head, the back of the body, whatever "
                         "hangs from the shoulders; no face visible"},
    "left": {"yaw": 180, "elevation": "side",
             "sentence": "from the character's left side, in full profile, "
                         "facing screen left"},
    "right": {"yaw": 0, "elevation": "side",
              "sentence": "from the character's right side, in full profile, "
                          "facing screen right"},
    "top": {"yaw": 270, "elevation": "top",
            "sentence": "from directly above, a top-down view looking straight down: "
                        "the top of the head and the shoulders, the character facing "
                        "the bottom of the frame, the feet foreshortened beneath them"},
    "back_right": {"yaw": 45, "elevation": "side",
                   "sentence": "from behind and to the character's right, a "
                               "three-quarter back view; no face visible"},
    "back_left": {"yaw": 135, "elevation": "side",
                  "sentence": "from behind and to the character's left, a "
                              "three-quarter back view; no face visible"},
    "front_left": {"yaw": 225, "elevation": "side",
                   "sentence": "from the front and to the character's left, "
                               "a three-quarter front view"},
    "front_right": {"yaw": 315, "elevation": "side",
                    "sentence": "from the front and to the character's right, "
                                "a three-quarter front view"},
}
# `--views` with no list; and how the studio's Concept tab groups them: the
# main buttons (Front is the base itself, Side is the left profile - the
# right is mirrored in a sprite sheet) and the advanced group behind a fold.
CONCEPT_CORE_VIEWS: tuple[str, ...] = ("back", "left", "right", "top")
CONCEPT_MAIN_VIEWS: tuple[str, ...] = ("back", "left", "top")
CONCEPT_ADVANCED_VIEWS: tuple[str, ...] = ("right", "back_right", "back_left",
                                           "front_left", "front_right")
CONCEPT_VIEW_COUNT = 2
FRONT_VIEW = "front"
# The `--views` keywords that name a whole group, so a script and the Concept
# tab's buttons can say the same thing: `all` (every view), `main` and
# `advanced` (the two the tab groups by). `advanced` is what the tab's sweep
# button sends. Note `right` is in CORE - the no-arg default - AND in ADVANCED:
# the two splits answer different questions and are deliberately not the same
# set. See CONTEXT.md, "core / main / advanced".
CONCEPT_VIEW_GROUPS: dict[str, tuple[str, ...]] = {
    "main": CONCEPT_MAIN_VIEWS,
    "advanced": CONCEPT_ADVANCED_VIEWS,
}
# How long one view render takes, for the estimate the Concept tab quotes
# before a sweep. Measured on the A5000 over the 32 view renders recorded in
# work/*_concept/views/views.json (2026-09-24): a warm render is ~10.8 s
# median, a second candidate of the same view ~8 s, and the FIRST render of a
# cold run is 164-201 s while Qwen-Image-Edit pages into VRAM. A recorded
# figure above the cutoff is that cold first render and is dropped from the
# median rather than dragging it up by a factor of three.
CONCEPT_VIEW_SECONDS = 11.0
CONCEPT_VIEW_COLD_SECONDS = 190.0
CONCEPT_WARM_CUTOFF_S = 60.0
# The edit prompt for Qwen-Image-Edit (gpu/workflows/view_pass.json): image1
# is the base; the output is the same character, same size, from the view.
CONCEPT_VIEW_PROMPT = (
    "Show the same pixel art character from image1 seen {sentence}. The same character: "
    "the same {parts}, the same colours, the same outline weight, the same proportions, "
    "the same size, standing at rest with the feet at the bottom edge, one figure only, "
    "plain flat white background, no ground shadow, no floor.")


def concept_view_names(spec: str | Sequence[str] | None) -> list[str]:
    """`--views back,left,right` | `all` | `main` | `advanced` | None (the
    core four) -> names in table order; an unknown name is refused. A group
    keyword may sit in a list with plain names (`main,front_right`)."""
    if spec is None or (isinstance(spec, str) and spec.strip() == ""):
        wanted = list(CONCEPT_CORE_VIEWS)
    elif isinstance(spec, str):
        wanted = [v.strip().lower() for v in spec.split(",") if v.strip()]
    else:
        wanted = [str(v).strip().lower() for v in spec if str(v).strip()]
    if "all" in wanted:
        return list(CONCEPT_VIEWS)
    wanted = [name for v in wanted for name in CONCEPT_VIEW_GROUPS.get(v, (v,))]
    unknown = [v for v in wanted if v not in CONCEPT_VIEWS]
    if unknown:
        raise ValueError(f"unknown view(s) {unknown}; one of {list(CONCEPT_VIEWS)}, "
                         f"a group ({', '.join(CONCEPT_VIEW_GROUPS)}) or 'all'")
    return [v for v in CONCEPT_VIEWS if v in wanted]


def concept_view_seconds(recorded: Sequence[float]) -> float:
    """The warm median of recorded view-render seconds, cold first renders
    dropped; CONCEPT_VIEW_SECONDS when nothing warm was recorded."""
    warm = sorted(s for s in recorded if 0 < s < CONCEPT_WARM_CUTOFF_S)
    if not warm:
        return CONCEPT_VIEW_SECONDS
    middle = len(warm) // 2
    if len(warm) % 2:
        return round(warm[middle], 1)
    return round((warm[middle - 1] + warm[middle]) / 2, 1)


def concept_view_prompt(view: str, parts: str | None) -> str:
    sentence = CONCEPT_VIEWS[view]["sentence"]
    parts = (parts or "").strip().rstrip(",.") or "outfit"
    return CONCEPT_VIEW_PROMPT.format(sentence=sentence, parts=parts)


# ---------------------------------------------------------------------------
# The mirrored view feature - three of the eight views are a flip of another
# ---------------------------------------------------------------------------
# Every view above costs one Qwen-Image-Edit round trip. Three of them need
# not: by the compass this file has used since the camera direction feature
# (DIRECTIONS, and REFERENCE_BY_DIR's own note) yaw 0 shows the
# character's RIGHT profile, 180 the left, 90 the back and 270 the front. A
# horizontal flip of the picture is a reflection through the vertical plane
# that contains the camera axis, and it takes a camera at yaw t to the camera
# at yaw 180 - t. So:
#
#     right (0)        <-> left (180)
#     back_right (45)  <-> back_left (135)
#     front_right(315) <-> front_left (225)
#
# and `back` (90), `top` (yaw 270 from above) and `front` (270, the base
# itself) map to THEMSELVES - they sit on the symmetry plane and have no
# partner. They must never be offered as mirrorable: flipping the back view
# gives the back view, so the "derived" image would be a copy claiming to be
# a different facing.
#
# DERIVED, NOT LISTED. The pairs below are computed from the yaws in
# CONCEPT_VIEWS rather than typed out, so a view added or re-aimed there
# cannot leave a stale second table behind; the self-mapping ones drop out on
# their own, which is why `back` / `top` need no special case. `elevation` is
# part of the key because a top-down camera flipped horizontally is still a
# top-down camera - a flip never changes the pitch.
#
# The rejected alternative was to mirror at RENDER time instead - have the
# matrix ask for "left, flipped" and never store the file. That would
# save a few hundred KB and cost the thing this codebase is built on: a view
# on disk with a sha256 and a source. A stored flip is as traceable as a
# generated one, and everything downstream (`reference_for`, the matrix's
# prerequisite, the board's ring) keeps reading one shape.
def _mirror_pairs() -> dict[str, str]:
    by_camera = {(spec["yaw"], spec.get("elevation", "side")): name
                 for name, spec in CONCEPT_VIEWS.items()}
    pairs: dict[str, str] = {}
    for name, spec in CONCEPT_VIEWS.items():
        partner = by_camera.get(((180 - int(spec["yaw"])) % 360,
                                 spec.get("elevation", "side")))
        if partner is not None and partner != name:
            pairs[name] = partner
    # The pairing has to be an involution - if `right` mirrors from `left` then
    # `left` mirrors from `right` - or one of the two would derive from a view
    # that cannot derive back, and the UI would offer a one-way button. Checked
    # on import rather than at the first flip, so a bad yaw in CONCEPT_VIEWS
    # fails here and not in front of the owner.
    for view, partner in pairs.items():
        if pairs.get(partner) != view:
            raise ValueError(f"CONCEPT_VIEWS: {view!r} mirrors to {partner!r} but not back; "
                             f"a view's yaw must have its 180 - yaw twin in the table")
    return pairs


CONCEPT_MIRROR_PAIRS: dict[str, str] = _mirror_pairs()


def mirror_of(view: str) -> str | None:
    """The view this one is the horizontal flip of, or None when it has no
    partner - `back`, `top` and `front` sit on the symmetry plane, and so
    does any name that is not a view at all."""
    return CONCEPT_MIRROR_PAIRS.get(str(view or "").strip().lower())


def mirrorable_views() -> tuple[str, ...]:
    """The views that can be derived from another, in table order - what a
    refusal names so the owner does not have to guess."""
    return tuple(v for v in CONCEPT_VIEWS if v in CONCEPT_MIRROR_PAIRS)


# The concept generation feature: an instruction edit of one attempt through
# the same Qwen-Image-Edit path ("make the helmet red"). The owner's words
# first; the rest keeps the figure a drop-in for the one it edits.
CONCEPT_EDIT_PROMPT = (
    "{instruction}. Keep the same pixel art character from image1 otherwise: the same "
    "{parts}, the same colours, the same outline weight, the same proportions, the same "
    "size and pose, one figure only, plain flat white background, no ground shadow, no floor.")


def concept_edit_prompt(instruction: str, parts: str | None) -> str:
    instruction = " ".join(str(instruction or "").split()).strip().rstrip(".")
    if not instruction:
        raise ValueError("--instruction is required for an edit: what should change")
    parts = (parts or "").strip().rstrip(",.") or "outfit"
    return CONCEPT_EDIT_PROMPT.format(instruction=instruction, parts=parts)


# Classic img2img of a VIEW attempt runs on the Z-Image stack, whose prompt
# is a description, not an edit instruction: the unit seen from that side.
CONCEPT_VIEW_IMG2IMG_PROMPT = (
    "{trigger} a single pixel art character sprite of {description}, seen {sentence}, "
    "standing at rest, full body in frame, feet at the bottom edge, chunky readable "
    "silhouette, bold dark outline, flat shaded, the same {parts}, plain flat white "
    "background, no ground shadow, no floor")


def concept_view_img2img_prompt(description: str, view: str, parts: str | None,
                                trigger: str | None = None) -> str:
    trigger = CONCEPT_LORA["trigger"] if trigger is None else trigger
    parts = (parts or "").strip().rstrip(",.") or "outfit"
    return CONCEPT_VIEW_IMG2IMG_PROMPT.format(
        trigger=trigger, description=description.strip(),
        sentence=CONCEPT_VIEWS[view]["sentence"], parts=parts).replace("  ", " ").strip()


# Which registry family each concept op draws from (gpu/models.registry.tsv),
# and the checkpoint each workflow ships with - the default when no --model.
CONCEPT_FAMILIES: dict[str, str] = {"txt2img": "zimage", "img2img": "zimage",
                                    "views": "qwen-edit", "edit": "qwen-edit"}
CONCEPT_MODELS: dict[str, str] = {
    "zimage": "z_image_turbo_int8_convrot.safetensors",
    "qwen-edit": "svdq-int4_r128-qwen-image-edit-2509-lightning-4steps-251115.safetensors",
}
# Classic img2img: how much of the source is redrawn (KSampler denoise).
CONCEPT_IMG2IMG_STRENGTH = 0.55
# The retouch feature's masked redraw (gpu/workflows/inpaint_pass.json): the
# same slider, but a masked region is meant to be redrawn rather than nudged,
# so it starts near the top of the band.
CONCEPT_INPAINT_STRENGTH = 0.9


# ---------------------------------------------------------------------------
# The animate tab feature - the owner directs every frame, direction by direction
# ---------------------------------------------------------------------------
# One animation is one (unit, action, direction) cell: a row of frames the
# owner generates, judges and fixes one at a time, with approval as the gate
# and the approved frames written as the cell's finished frames. The row
# starts as the phase template (`ACTION_PHASES[action]["budget"]` frames, the
# phase keys marked as keyframes at `PhasePlan.keys()`) and the owner adds,
# removes and marks frames from there.
#
# Side-on only for now: the eight side cells, each drawn from its OWN
# accepted view (s270 is the front, the accepted base itself). Top-down is a
# later milestone and is refused rather than half-supported.
ANIMATE_ELEVATION = "side"
# What a frame attempt is drawn from, besides the optional skeleton:
# `view` - the direction's accepted view; `keys` - the two nearest APPROVED
# keyframes either side of the frame; `both` - the view and the two keys.
# Keyframes default to `view`, in-betweens to `keys`. `copy` is not a drawing:
# the keyframe's picture is the view itself, no GPU - the default for a view
# key (`view_key_phases`) that has no copy yet (the view keys feature).
ANIMATE_DRAW_FROM: tuple[str, ...] = ("view", "keys", "both", "copy")
# Qwen-Image-Edit-2509's TextEncodeQwenImageEditPlus takes image1..image3.
ANIMATE_MAX_IMAGES = 3
# The pose slider scrubs the rig clip at this rate - finer than the 12 fps the
# animation plays at, so a pose between two game frames can still be chosen.
# Rendered on the CPU once per (unit, action, direction) and cached.
ANIMATE_SCRUB_FPS = 24
# How many frames a row may grow to. The studio's contract caps a finished
# clip at PHASE_CAP; the owner may go past it by hand, but not without limit.
ANIMATE_MAX_FRAMES = 24
# The play-back rate and the one tick every frame holds in the output timing.
ANIMATE_FPS = PHASE_FPS
# The keyframe sweep feature: one press draws the keyframes (or, once they are
# approved, the in-betweens) of several animations as ONE run. Capped so one
# press cannot book the card for hours: 8 cells x ~4 keys is ~30 renders.
ANIMATE_SWEEP_MAX_CELLS = int(os.environ.get("ANIM_SWEEP_MAX_CELLS", "8"))
# Per action, per phase: what the pose is, in the owner's words. Pre-fills a
# keyframe's prompt; the owner edits it and each attempt remembers its own.
ANIMATE_PHASE_PROMPTS: dict[str, dict[str, str]] = {
    "attack": {
        "start": "standing ready at rest, the sword lowered, weight even on both feet",
        "wind-up": "winding up a sword swing, the sword pulled back over the shoulder, "
                   "weight on the back foot",
        "action": "mid-swing, the sword slashing forward across the body, weight "
                  "lunging onto the front foot",
        "wind-down": "recovering after the swing, the sword lowered and following "
                     "through, settling back toward a stand",
        "end": "back to standing ready at rest, the sword lowered",
    },
    "idle": {
        "start": "standing relaxed at rest, the weapon held loosely",
        "wind-up": "breathing in at rest, the chest lifting a little, the shoulders slightly raised",
        "action": "at the top of the breath, standing relaxed, the weapon held loosely",
        "wind-down": "breathing out at rest, the shoulders settling back down",
        "end": "standing relaxed at rest again, about to breathe in",
    },
    "walk": {
        "start": "walking, the contact pose, the right foot forward and the left arm swinging forward",
        "wind-up": "walking, mid-stride, the left foot stepping forward and the right arm swinging forward",
        "action": "walking, the passing pose, one foot under the body and the other lifting past it",
        "wind-down": "walking, mid-stride, the right foot stepping forward and the left arm swinging forward",
        "end": "walking, about to repeat the contact pose, the right foot reaching forward",
    },
    "run": {
        "start": "running, the contact pose, leaning forward, the right foot striking the ground",
        "wind-up": "running, leaning forward, the left knee driving up and the right arm pumping forward",
        "action": "running, both feet off the ground, the body leaning forward",
        "wind-down": "running, leaning forward, the right knee driving up and the left arm pumping forward",
        "end": "running, about to strike the ground again with the right foot",
    },
    "jump": {
        "start": "standing ready to jump, the knees slightly bent",
        "wind-up": "crouching to jump, the knees bent deep and the arms swung back",
        "action": "in the air at the top of a jump, the legs tucked and the arms raised",
        "wind-down": "landing from a jump, the knees bent to take the impact, the arms out for balance",
        "end": "standing again after the landing, the weight settled",
    },
}
# The whole Qwen-Image-Edit prompt around the owner's pose sentence. Which
# image is which depends on what the attempt was drawn from, so the roles are
# a sentence of their own (`animate_prompt`).
ANIMATE_PROMPT = (
    "Redraw the same pixel art character from image1, {pose}. {roles}The same character: "
    "the same {parts}, the same colours, the same outline weight, the same proportions and "
    "the same size, seen from the same side as image1. One figure only, full body in frame, "
    "feet at the bottom edge, plain flat white background, no ground shadow, no floor"
    "{no_lines}.")


def phase_label(phase: str | None) -> str:
    """"wind-up" -> "Wind-up"; what a keyframe is called on the tab."""
    text = str(phase or "")
    return text[:1].upper() + text[1:]


# A unit whose art does not fit the default sentences gets its own, per action
# and phase; a phase it leaves out falls back to ANIMATE_PHASE_PROMPTS. The
# archer's art carries no weapon at all (no bow on the base or any view - see
# UNIT_PARTS), so the owner ruled its attack a spell cast from the hands rather
# than a sword the design does not have (2026-09-26).
ANIMATE_UNIT_PHASE_PROMPTS: dict[str, dict[str, dict[str, str]]] = {
    "hooded_archer": {
        "attack": {
            "start": "standing ready at rest, both hands low at the sides, empty-handed",
            "wind-up": "gathering a spell, both hands drawn back to the chest, the body "
                       "coiled and leaning back, weight on the back foot, empty-handed",
            "action": "casting a spell, both arms thrust forward with the palms out and a "
                      "small burst of green magic at the hands, weight lunging onto the "
                      "front foot",
            "wind-down": "recovering after the cast, the arms lowering, the magic gone, "
                         "settling back toward a stand, empty-handed",
            "end": "back to standing ready at rest, both hands low at the sides, "
                   "empty-handed",
        },
    },
}


def animate_phase_prompt(action: str, phase: str, unit: str | None = None) -> str:
    """A keyframe's pre-filled pose sentence - the unit's own when it has one
    for that action and phase; a phase with none gets its name."""
    name = studio_action(action) or action
    own = ((ANIMATE_UNIT_PHASE_PROMPTS.get(unit or "") or {}).get(name) or {}).get(phase)
    return (own or (ANIMATE_PHASE_PROMPTS.get(name) or {}).get(phase)
            or f"{phase.replace('-', ' ')} of the {name}")


def animate_inbetween_prompt(action: str, before: str | None, after: str | None) -> str:
    """An in-between's pre-filled pose sentence: moving from the keyframe
    before it toward the keyframe after it."""
    name = studio_action(action) or action
    if before and after and before != after:
        return (f"{name}, in between two key poses, moving from the {before} toward "
                f"the {after}")
    if before or after:
        return f"{name}, in between two key poses, moving on from the {before or after}"
    return f"{name}, in motion"


# Camera words the unit's description may carry ("facing forward" - it was
# written for the front view the concept tab rendered). Pasted into a frame
# drawn at another direction they contradict image1, so a frame's copy of the
# description drops them; the direction is image1's to say.
ANIMATE_CAMERA_WORDS = (
    r"(?:facing|turned|looking)\s+(?:forward|forwards|the\s+viewer|the\s+camera|front|left|right|away|back|backwards)",
    r"(?:front|side|back|rear|profile|three[-\s]quarter)\s+view",
    r"(?:seen|viewed)\s+from\s+(?:the\s+)?(?:front|side|back|behind|left|right|above)",
    r"from\s+(?:the\s+)?(?:front|behind)",
    r"in\s+profile",
)


def strip_camera_words(text: str | None) -> str:
    """The description with its camera words and the empty sentences they
    leave taken out: "a hooded green archer. a fantasy ranger. facing forward"
    -> "a hooded green archer, a fantasy ranger"."""
    import re

    out = str(text or "")
    for pattern in ANIMATE_CAMERA_WORDS:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE)
    pieces = [" ".join(p.split()).strip(" ,") for p in re.split(r"[.;]", out)]
    return ", ".join(p for p in pieces if p)


def animate_prompt(pose: str, roles: Sequence[str], parts: str | None,
                   avoid: str | None = None) -> str:
    """The prompt as sent: the pose sentence, what each image is (`roles`, one
    of `view` / `before` / `after` / `skeleton` per image, in image order),
    the parts (camera words stripped), and the unit's avoid phrase as a
    sentence of its own - the Qwen-Image-Edit graph runs at cfg 1, where a
    negative prompt does nothing, so "what must not appear" has to be said in
    the positive one."""
    pose = " ".join(str(pose or "").split()).strip().rstrip(".")
    if not pose:
        raise ValueError("a frame needs a prompt: what the pose is")
    said: list[str] = []
    roles = list(roles)
    if "before" in roles and "after" in roles:
        b, a = roles.index("before") + 1, roles.index("after") + 1
        said.append(f"image{b} is the same character a moment earlier and image{a} a moment "
                    f"later in the same motion; draw the pose in between them.")
    elif "before" in roles or "after" in roles:
        n = (roles.index("before") if "before" in roles else roles.index("after")) + 1
        said.append(f"image{n} is the same character a moment "
                    f"{'earlier' if 'before' in roles else 'later'} in the same motion; carry "
                    f"the motion on from it.")
    if "skeleton" in roles:
        n = roles.index("skeleton") + 1
        said.append(f"Put the character in the exact body pose of the stick-figure skeleton "
                    f"in image{n}: follow every limb - the arms, the legs, the torso lean and "
                    f"the head.")
    parts = strip_camera_words(parts).rstrip(",.") or "outfit"
    avoid = " ".join(str(avoid or "").split()).strip().rstrip(",.")
    text = ANIMATE_PROMPT.format(
        pose=pose, roles="".join(f"{s} " for s in said), parts=parts,
        no_lines=", no skeleton lines" if "skeleton" in roles else "")
    return f"{text} Do not add: {avoid}." if avoid else text


def view_for_direction(direction_cell: str) -> str:
    """The view a side direction is drawn from: its own camera. s270 is the
    front - the accepted base itself."""
    yaw, elevation = direction(direction_cell)
    if yaw == 270 and elevation == "side":
        return FRONT_VIEW
    for view, spec in CONCEPT_VIEWS.items():
        if int(spec["yaw"]) == yaw and spec.get("elevation", "side") == elevation:
            return view
    raise ValueError(f"no view looks from {direction_cell!r}")


def mirror_direction(direction_cell: str) -> str | None:
    """The direction whose animation flips into this one - the camera at
    yaw 180 - t, the same pairs as the views (right<->left, back_right<->
    back_left, front_left<->front_right) - or None for s90 and s270, which
    look straight down the symmetry plane."""
    yaw, elevation = direction(direction_cell)
    partner = (180 - yaw) % 360
    if partner == yaw:
        return None
    return direction_name(partner, elevation)


# ---------------------------------------------------------------------------
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
    # The studio's second unit. NOT the knight's regions: this one wears one
    # full-length cloak, so `arms` / `torso` / `legs` name nothing a segmenter
    # can find - measured on the accepted base 2026-09-24, every candidate that
    # held one of the three held the other two as well and all three scored
    # below the contamination gate. What the art actually separates into is the
    # hood, the two gloves, the two boots and the cloak that covers the rest;
    # the cloak is last because priority resolves an overlap to the earlier row
    # and it is the one that would otherwise swallow the small parts. There is
    # deliberately no bow: a `prop=` part is a rigkit polygon drawn from
    # UNIT_PROPS onto the skeleton, and the ruled Quaternius rig carries a
    # sword, so a bow part would ask for a region that is never in the control
    # image. The bow's colours ride in `gloves` until a rig ships one.
    "hooded_archer": (
        # The hood is the whole mass over the skull, its dark face opening
        # included - the crown at 2.0 rather than the knight's 1.35 because the
        # opening is large enough that 1.35 lands inside it, and SAM then
        # returns the opening instead of the hood (2026-09-24: an 11k px face
        # blob beat the 26k px hood, and the palette came out all shadow).
        Part("hood", joints=("nose", "right_eye", "left_eye", "right_ear",
                             "left_ear"),
             mids=(("neck", "nose", 2.0),
                   ("neck", "right_ear", 1.35), ("neck", "left_ear", 1.35))),
        Part("gloves", joints=("right_wrist", "left_wrist"),
             mids=(("right_elbow", "right_wrist", 1.12),
                   ("left_elbow", "left_wrist", 1.12))),
        Part("boots", joints=("right_ankle", "left_ankle"),
             mids=(("right_knee", "right_ankle", 1.06),
                   ("left_knee", "left_ankle", 1.06))),
        Part("cloak", joints=("neck", "right_hip", "left_hip"),
             mids=(("neck", "right_hip", 0.5), ("neck", "left_hip", 0.5),
                   ("right_hip", "right_knee", 0.6),
                   ("left_hip", "left_knee", 0.6))),
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
