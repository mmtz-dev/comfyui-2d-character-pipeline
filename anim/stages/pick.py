"""Stage: `anim pick` - keypoints -> picks.json

Choose the N key poses that carry the action, out of the 30-60 frames a 3-5
second clip produces.

    anim pick knight_strike            # N from the art bible's budget
    anim pick knight_strike -n 3       # override

THE HEURISTIC
-------------
Work on NORMALISED poses (root-centred on the pelvis, scaled so one torso
length == 1.0), never on pixels: a subject who stepped 30 cm nearer the camera
has not changed pose, and a pixel-space picker would think they had.

Two signals, because one is not enough:

  deviation  distance from the rest pose (frame 0 - the art bible tells the
             owner to hold the start pose for half a second, so frame 0 IS the
             rest pose). Its PEAKS are the extremes of the motion: the windup,
             the contact, the follow-through. These are the poses that read.

  speed      distance from the previous frame. Its local MINIMA are the held
             moments - where a limb reverses direction and the pose is, for an
             instant, still. A reversal is a key pose almost by definition.

Candidates are both of those plus the two endpoints. They are then taken in
order of deviation, each admitted only if it is at least `--dedup` torso lengths
away from every pose already chosen, so four candidates a frame apart in the
middle of a swing cannot crowd out the recovery. The first and last frames are
always in - an action that does not start and end where it started reads as a
jump cut - and if the threshold is too strict to reach N it is relaxed, then
finally filled with an even spread, so the stage cannot return fewer poses than
asked.

THE TWO-POSE CASE is the one exception to "both endpoints". With N == 2 the
endpoints alone would be the rest pose twice (a hit reaction, a pickup, an idle
breath all start and end at rest), which is no animation at all. So a 2-pose
action gets the EXTREME and the last frame: the flinch, then the recovery; the
stoop, then the stand; the breath in, then the loop point. The last frame stays
so the action still lands where the next state begins.

Deterministic: same keypoints in, same picks out, no randomness anywhere.

THE TIMING SHAPE (the strip timing feature)
-------------------------------------------
Picking WHICH poses is half of a strip; how long each one holds is the other
half, and it is what turns a slideshow into a swing. picks.json therefore also
records `timing`: `{"weights": [2, 3, 4, 3], "anchor": 2, "source": ...}` -
one dimensionless weight per frame and the index of the frame whose START lands
on the choreography's anchor instant (the contact for a strike). `publish`
copies it into the sidecar's `contract.timing`.

    --timing template   (default) the per-verb shape in config.ACTION_TIMING,
                        keyed by (action, n); no row -> `timing` is null and the
                        game plays the strip uniform.
    --timing clip       the clip's own rhythm: each weight is the number of
                        clip frames between that pick and the next (the last
                        frame gets 1). Documented alternative, not the default,
                        because a filmed or rig clip holds the rest pose for
                        half a second and that hold dominates the shape - the
                        knight's Sword_Attack comes out [6, 4, 8, 1].

`timing` and `auto_timing` are written in step, like `picks` and `auto`, so a
hand edit of the weights or the anchor in an editor is detectable - and at
publish the hand edit wins over either mode.

THE PHASE TEMPLATE (the keyframe phase feature)
------------------------------------------------
The studio's clips are picked against one shape, whatever the route the
frames came from (a Wan clip, a pose-pass strip, a filmed take):

    start -> wind-up -> action -> wind-down -> end -> (loop)

    anim pick knight_attack                 # attack: the template is the default
    anim pick knight_strike --phases        # an art-bible verb, on request
    anim pick knight_attack --no-phases     # the 2-6 key-pose picker instead

`config.ACTION_PHASES` gives each action its budget (cap 12), whether it
loops, and two five-way shares: `split`, how the BUDGET is divided between
the phases, and `span`, how the SOURCE CLIP'S TIME is. `phase_plan` turns the
split into whole frames (largest remainder, every phase at least one) and,
for a one-shot, moves one more frame of the budget into the wind-up - the
ANTICIPATION frame, so the beat before the action is never one cell. The
span cuts the source clip into five windows; each phase's frames are spread
evenly inside its own window (the start pinned to the clip's first frame,
the end of a one-shot to its last). Frames per second of clip therefore
differs per phase - many in the action window, few in the rest windows -
and that non-uniform density IS the ease in and out. `spread="even"` keeps
the labels and drops the shaping (linspace over the clip): the bake-off's
two routes use it so their numbers stay what the owner judged.

A loop's `end` window stops short of the clip's last frame - the frame after
`end` is `start` again, and a cycle whose last cell repeats its first hitches
on the wrap. Instead the LOOP-CLOSURE check measures the distance from the
last pick back to the first (mean joint displacement in torso lengths from
the keypoints; the mean pixel difference of the two frames when a pose is
missing) against `config.LOOP_TOLERANCE`, records it in `picks.json` and the
review page shows it. A cycle that does not close is reported, not fixed.

picks.json gains a `phases` block - the plan, one tag per pick, the index of
the anticipation frame and the closure - and `timing` becomes the phase
shape: one weight per frame in 12 fps ticks, the anchor on the first action
frame, and `phases` / `durations_ms` beside them for the sidecar. THE HOLD
RULE: a clip lasts budget / 12 s. When a phase's window held fewer source
frames than the phase was allotted (a 9-frame clip against a budget of 12),
its frames hold longer - each weighs allotted / picked ticks - so the phase
still lasts its share and the clip its budget. With every phase full, every
weight is 1.

NEVER CLOBBERS A HUMAN EDIT
---------------------------
picks.json records both what the picker chose (`auto`) and what is in force
(`picks`), and the same pair for the timing shape. When either pair differs, a
person changed it - in the review page or in an editor - and re-running `pick`
would throw that away. So it refuses, and says how to override (`--force`).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from anim import config, posekit
from anim.job import Job
from anim.util import read_json, write_json_in_place

NAME = "pick"
HELP = "keypoints -> picks.json"

PICKS_SCHEMA = "anim.picks/1"
TIMING_MODES = ("template", "clip")
SPREADS = ("phase", "even")


# ---------------------------------------------------------------------------
# The picker
# ---------------------------------------------------------------------------
def _local_maxima(values: Sequence[float]) -> list[int]:
    """Indices strictly higher than both neighbours (plateau -> its first index)."""
    out: list[int] = []
    n = len(values)
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            out.append(i)
    return out


def _local_minima(values: Sequence[float]) -> list[int]:
    return [i for i in _local_maxima([-v for v in values])]


def signals(poses: Sequence[np.ndarray | None]) -> dict[str, list[float]]:
    """Deviation-from-rest and frame-to-frame speed, both in torso lengths.

    Missing frames get 0.0 rather than NaN so the curves stay plottable and the
    peak finder stays simple; they are excluded from the candidate set
    separately.
    """
    rest = next((p for p in poses if p is not None), None)
    deviation: list[float] = []
    speed: list[float] = []
    previous: np.ndarray | None = None
    for pose in poses:
        d = posekit.pose_distance(pose, rest)
        deviation.append(0.0 if math.isinf(d) else d)
        s = posekit.pose_distance(pose, previous)
        speed.append(0.0 if math.isinf(s) else s)
        if pose is not None:
            previous = pose
    return {"deviation": deviation, "speed": speed}


def auto_picks(poses: Sequence[np.ndarray | None], n: int,
               dedup: float = None) -> list[int]:
    """N frame indices: the key poses. See this module's docstring."""
    dedup = config.PICK_DEDUP if dedup is None else dedup
    usable = [i for i, p in enumerate(poses) if p is not None]
    if not usable:
        raise ValueError("no frame has a usable pose")
    if n >= len(usable):
        return usable
    if n <= 1:
        return usable[:1]

    curves = signals(poses)
    deviation, speed = curves["deviation"], curves["speed"]
    first, last = usable[0], usable[-1]

    peak_max = max(deviation) or 1.0
    candidates = {first, last}
    candidates.update(i for i in _local_maxima(deviation) if poses[i] is not None)
    # Reversals only count once the body has actually left the rest pose; the
    # half-second hold at the start of a take is full of near-zero minima.
    candidates.update(i for i in _local_minima(speed)
                      if poses[i] is not None and deviation[i] > 0.1 * peak_max)

    # Every frame is a fallback candidate, so a clip with no clean peaks (a slow
    # idle, say) still fills its budget instead of returning two poses.
    ordered = sorted(candidates, key=lambda i: (-deviation[i], i))
    ordered += [i for i in sorted(usable, key=lambda i: (-deviation[i], i))
                if i not in candidates]

    chosen: list[int] = [first]
    if last != first:
        chosen.append(last)
    if n == 2:
        # The extreme and the recovery, not rest twice - see the docstring.
        peak = max(usable, key=lambda i: (deviation[i], -i))
        if peak not in (first, last) and deviation[peak] > 0.0:
            chosen = [peak, last]
        return sorted(chosen[:n])

    threshold = dedup
    while len(chosen) < n:
        added = False
        for index in ordered:
            if len(chosen) >= n:
                break
            if index in chosen:
                continue
            if all(posekit.pose_distance(poses[index], poses[c]) >= threshold
                   for c in chosen):
                chosen.append(index)
                added = True
        if len(chosen) >= n:
            break
        if not added:
            if threshold <= 1e-4:
                # Everything left is a duplicate of something chosen. Spread the
                # remainder evenly rather than returning short: the strip has N
                # cells and a short strip is a broken strip.
                for index in _even_spread(usable, n):
                    if index not in chosen:
                        chosen.append(index)
                    if len(chosen) >= n:
                        break
                break
            threshold *= 0.5
    return sorted(chosen[:n])


def _even_spread(usable: Sequence[int], n: int) -> list[int]:
    if n <= 1:
        return [usable[0]]
    step = (len(usable) - 1) / (n - 1)
    return [usable[int(round(i * step))] for i in range(n)]


# ---------------------------------------------------------------------------
# The phase template (the keyframe phase feature)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhasePlan:
    """One action's template made concrete: whole frames per phase, the
    windows of the source clip they are picked from, and whether it loops."""

    action: str                       # the studio's name (attack, idle, ...)
    verb: str                         # the pipeline's (strike, idle, ...)
    budget: int
    loop: bool
    frames: dict[str, int]            # phase -> frames of the budget
    span: dict[str, float]            # phase -> share of the source clip
    anticipation: bool                # a wind-up frame came from the budget
    source: str                       # "template" | "timing.json[:<slug>]"

    def tags(self) -> list[str]:
        """One phase name per budget slot, in order."""
        return [name for name in config.PHASE_NAMES
                for _ in range(self.frames[name])]

    def keys(self) -> list[int]:
        """The slot that stands for each phase: the first of `start`, the last
        of `end`, the middle of the rest - what a crossfade blends between."""
        out: list[int] = []
        offset = 0
        for name in config.PHASE_NAMES:
            count = self.frames[name]
            if count:
                out.append(offset if name == "start" else
                           offset + count - 1 if name == "end" else
                           offset + count // 2)
            offset += count
        return out

    def anticipation_slot(self) -> int | None:
        """The last wind-up slot - the frame the budget lent - or None."""
        if not self.anticipation or not self.frames["wind-up"]:
            return None
        return self.frames["start"] + self.frames["wind-up"] - 1

    def as_record(self) -> dict[str, Any]:
        return {"action": self.action, "verb": self.verb, "budget": self.budget,
                "loop": self.loop, "frames": dict(self.frames),
                "span": {k: round(v, 4) for k, v in self.span.items()},
                "anticipation": self.anticipation, "source": self.source}


def allocate(budget: int, split: Sequence[float], *, anticipation: bool) -> list[int]:
    """Whole frames per phase from the shares: largest remainder over the
    budget (less one, when a one-shot reserves the anticipation frame), every
    phase at least one while the budget allows, ties to the earlier phase;
    then the reserved frame goes to the wind-up."""
    names = config.PHASE_NAMES
    total = sum(split)
    shares = [s / total for s in split]
    reserve = 1 if anticipation and budget > len(names) else 0
    pool = budget - reserve
    if pool <= len(names):
        # Too few frames to give every phase one: in split order of size,
        # earliest first on ties, until the pool is spent.
        order = sorted(range(len(names)), key=lambda i: (-shares[i], i))
        counts = [0] * len(names)
        for i in order[:pool]:
            counts[i] = 1
        return [c + (reserve if names[i] == "wind-up" else 0) for i, c in enumerate(counts)]
    raw = [s * pool for s in shares]
    counts = [max(1, int(math.floor(r))) for r in raw]
    while sum(counts) > pool:                     # the floor-of-one overshot
        i = max(range(len(names)), key=lambda i: (counts[i] - raw[i], -i))
        counts[i] -= 1
    remainders = [r - c for r, c in zip(raw, counts)]
    while sum(counts) < pool:
        i = max(range(len(names)), key=lambda i: (remainders[i], -i))
        counts[i] += 1
        remainders[i] -= 1.0
    if reserve:
        counts[names.index("wind-up")] += reserve
    return counts


def phase_plan(action: str, *, budget: int | None = None, slug: str | None = None,
               timing_file: Path | None = None) -> PhasePlan:
    """The plan for one action from config (and the project's timing.json).
    `budget` overrides the row's, inside the cap."""
    row = config.phases_for_action(action, slug=slug, timing_file=timing_file)
    if row is None:
        raise ValueError(
            f"{action!r} is not a studio action; the phase template knows "
            f"{sorted(config.STUDIO_ACTIONS)} (and their pipeline verbs). Pick "
            f"without --phases for an art-bible verb.")
    count = int(row["budget"] if budget is None else budget)
    if not 1 <= count <= config.PHASE_CAP:
        raise ValueError(f"a phase budget must be 1..{config.PHASE_CAP} frames, "
                         f"not {count} (the studio caps a clip at {config.PHASE_CAP})")
    anticipation = not row["loop"]
    counts = allocate(count, row["split"], anticipation=anticipation)
    total_span = sum(row["span"])
    return PhasePlan(
        action=row["action"], verb=row["verb"], budget=count, loop=bool(row["loop"]),
        frames=dict(zip(config.PHASE_NAMES, counts)),
        span={n: v / total_span for n, v in zip(config.PHASE_NAMES, row["span"])},
        anticipation=anticipation and counts[config.PHASE_NAMES.index("wind-up")] > 0,
        source=row["source"])


def phase_windows(plan: PhasePlan) -> dict[str, tuple[float, float]]:
    """Each phase's window of the source clip as [a, b) fractions; a loop's
    clip is a circle, so its last window stops short of 1.0 and the frame
    after `end` is `start` again."""
    out: dict[str, tuple[float, float]] = {}
    at = 0.0
    for name in config.PHASE_NAMES:
        end = 1.0 if name == config.PHASE_NAMES[-1] else min(1.0, at + plan.span[name])
        out[name] = (at, end)
        at = end
    return out


def phase_positions(plan: PhasePlan, spread: str = "phase") -> list[float]:
    """Where in the source clip each budget slot samples, as fractions 0..1
    in slot order. `phase`: evenly inside each phase's window - the start
    pinned to 0, a one-shot's end pinned to 1, the rest centred, so the
    density follows split / span (the ease). `even`: linspace, labels only."""
    if spread not in SPREADS:
        raise ValueError(f"spread must be one of {SPREADS}, not {spread!r}")
    if plan.budget <= 1:
        return [0.0]
    if spread == "even":
        return [float(u) for u in np.linspace(0.0, 1.0, plan.budget)]
    windows = phase_windows(plan)
    out: list[float] = []
    for name in config.PHASE_NAMES:
        count = plan.frames[name]
        if not count:
            continue
        a, b = windows[name]
        width = b - a
        if name == "start":
            out += [a + width * j / count for j in range(count)]
        elif name == "end" and not plan.loop:
            out += [b - width * (count - 1 - j) / count for j in range(count)]
        else:
            out += [a + width * (j + 0.5) / count for j in range(count)]
    return [min(1.0, max(0.0, u)) for u in out]


def _strictly_increasing(wanted: list[int], last: int) -> list[int]:
    """Nudge duplicate or out-of-order indices apart inside 0..last, keeping
    slot order; needs len(wanted) <= last + 1."""
    out = list(wanted)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + 1)
    out[-1] = min(out[-1], last)
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1] - 1)
    return out


def phase_frames(usable: Sequence[int] | int, plan: PhasePlan,
                 spread: str = "phase") -> list[int]:
    """Frame indices for the plan out of `usable` (a count, or the indices
    that have a pose), strictly increasing. Fewer usable frames than the
    budget: all of them."""
    indices = list(range(usable)) if isinstance(usable, int) else list(usable)
    if not indices:
        raise ValueError("no frame has a usable pose")
    if plan.budget >= len(indices):
        return indices
    if spread == "even":
        return _even_spread(indices, plan.budget)
    last = len(indices) - 1
    wanted = [int(round(u * last)) for u in phase_positions(plan, spread)]
    return [indices[i] for i in _strictly_increasing(wanted, last)]


def phase_instants(duration: float, plan: PhasePlan, spread: str = "phase") -> list[float]:
    """Sample instants (seconds) for a rig clip of `duration` - what the K
    route hands `rig(..., times=)` so the rig itself is sampled in shape."""
    if plan.budget < 2:
        return [0.0]
    duration = max(0.0, float(duration))
    if spread == "even":
        return [round(float(t), 6) for t in np.linspace(0.0, duration, plan.budget)]
    return [round(u * duration, 6) for u in phase_positions(plan, spread)]


def phase_tags(plan: PhasePlan, picks: Sequence[int], total: int) -> list[str]:
    """One tag per pick. With the budget met the tags are the plan's, in
    order; a short clip's picks are tagged by the window each falls in, so a
    phase can come back with fewer frames than its share (or none)."""
    picks = list(picks)
    if len(picks) == plan.budget:
        return plan.tags()
    windows = phase_windows(plan)
    last = max(1, total - 1)
    out: list[str] = []
    for index in picks:
        u = index / last
        name = next((n for n, (a, b) in windows.items() if a <= u < b), config.PHASE_NAMES[-1])
        out.append(name)
    return out


def loop_closure(poses: Sequence[np.ndarray | None], first: int, last: int, *,
                 frames_dir: Path | None = None,
                 tolerance: float | None = None,
                 pixel_tolerance: float | None = None) -> dict[str, Any]:
    """Does the last pick lead back into the first? Pose distance when both
    frames have a pose, else the pixel difference of the two frames when
    they are on disk, else "unknown" - said so, never guessed."""
    tolerance = config.LOOP_TOLERANCE if tolerance is None else tolerance
    pixel_tolerance = (config.LOOP_PIXEL_TOLERANCE if pixel_tolerance is None
                       else pixel_tolerance)
    out: dict[str, Any] = {"first": int(first), "last": int(last)}
    distance = posekit.pose_distance(poses[last], poses[first]) \
        if 0 <= first < len(poses) and 0 <= last < len(poses) else math.inf
    if math.isfinite(distance):
        out.update(metric="pose", distance=round(distance, 4), tolerance=tolerance,
                   ok=bool(distance <= tolerance))
        return out
    if frames_dir is not None:
        a, b = (Path(frames_dir) / f"{first + 1:04d}.png"), (Path(frames_dir) / f"{last + 1:04d}.png")
        if a.exists() and b.exists():
            from PIL import Image

            with Image.open(a) as ia, Image.open(b) as ib:
                x = np.asarray(ia.convert("L"), dtype=np.float32)
                y = np.asarray(ib.convert("L").resize(ia.size), dtype=np.float32)
            diff = float(np.abs(x - y).mean() / 255.0)
            out.update(metric="pixel", distance=round(diff, 4), tolerance=pixel_tolerance,
                       ok=bool(diff <= pixel_tolerance))
            return out
    out.update(metric="none", distance=None, tolerance=tolerance, ok=None)
    return out


def phase_timing(plan: PhasePlan, tags: Sequence[str], *,
                 fps: int | None = None) -> dict[str, Any]:
    """The timing shape for a phased pick: one weight per frame in fps ticks,
    the anchor on the first action frame. THE HOLD RULE - a phase that came
    back with fewer frames than it was allotted holds each of them longer
    (allotted / picked ticks) so the phase still lasts its share of the
    budget and the clip lasts budget / fps s; a phase with no frames at all
    lends its share to the phase before it (the one after, for `start`)."""
    fps = config.PHASE_FPS if fps is None else int(fps)
    tags = list(tags)
    if not tags:
        raise ValueError("a timing shape needs at least one frame")
    names = list(config.PHASE_NAMES)
    counts = {n: tags.count(n) for n in names}
    allotted = dict(plan.frames)
    # Empty phases lend their time to a neighbour that has frames.
    for i, name in enumerate(names):
        if counts[name] == 0 and allotted[name]:
            neighbours = [n for n in names[:i][::-1] if counts[n]] + \
                         [n for n in names[i + 1:] if counts[n]]
            allotted[neighbours[0]] += allotted[name]
            allotted[name] = 0
    weights: list[float] = []
    for tag in tags:
        w = allotted[tag] / counts[tag]
        weights.append(int(w) if float(w).is_integer() else round(w, 4))
    tick_ms = 1000.0 / fps
    anchor = tags.index("action") if "action" in tags else 0
    return {"weights": weights, "anchor": anchor, "source": f"phases:{plan.action}",
            "fps": fps, "loop": plan.loop, "phases": tags,
            "durations_ms": [int(round(w * tick_ms)) for w in weights]}


# ---------------------------------------------------------------------------
# The timing shape
# ---------------------------------------------------------------------------
def clip_timing(picks: Sequence[int], total_frames: int) -> list[int]:
    """One weight per pick: the clip frames from that pick to the next.

    The last pick has no successor, so it gets 1 - the clip's tail after the
    last key pose is by construction the frames the picker judged not worth a
    cell, and giving the recovered pose that whole hold would freeze the strip
    on its least interesting frame. `total_frames` is kept in the signature so
    a pick past the clip end is caught here rather than as a negative weight.
    """
    picks = list(picks)
    if any(p < 0 or p >= total_frames for p in picks):
        raise ValueError(f"picks {picks} fall outside the clip's {total_frames} frames")
    gaps = [max(1, b - a) for a, b in zip(picks, picks[1:])]
    return gaps + [1]


def auto_timing(action: str, picks: Sequence[int], total_frames: int,
                mode: str = "template") -> dict[str, Any] | None:
    """The shape `pick` records, or None when template mode has no row.

    Clip mode still takes its anchor from the template row: the gaps say how
    long each pose holds, not which pose is the contact, and that is the one
    thing the game must be told. With no row to place the anchor, clip mode is
    refused rather than guessed.
    """
    if mode not in TIMING_MODES:
        raise ValueError(f"--timing must be one of {TIMING_MODES}, not {mode!r}")
    template = config.timing_for_action(action, len(picks))
    if mode == "template":
        if template is None:
            return None
        weights, anchor = template
        return {"weights": weights, "anchor": anchor, "source": f"template:{action}"}
    if template is None:
        raise ValueError(
            f"--timing clip needs a template row for ({action!r}, {len(picks)}) to "
            f"place the anchor; add one to config.ACTION_TIMING or hand-edit "
            f"picks.json")
    return {"weights": clip_timing(picks, total_frames), "anchor": template[1],
            "source": "clip"}


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def read_picks(path: Path) -> dict[str, Any] | None:
    return read_json(path) if Path(path).exists() else None


def was_edited(existing: dict[str, Any] | None) -> bool:
    """True when a person changed picks.json after the picker wrote it - the
    picks OR the timing shape; either is work that must not be thrown away."""
    if not existing:
        return False
    if list(existing.get("picks") or []) != list(existing.get("auto") or []):
        return True
    return existing.get("timing") != existing.get("auto_timing")


def new_picks(picks: list[int], auto: list[int], *, n: int, action: str,
              dedup: float, curves: dict[str, list[float]],
              timing: dict[str, Any] | None = None,
              phases: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "schema": PICKS_SCHEMA,
        "n": n,
        "action": action,
        "dedup": dedup,
        "picks": list(picks),
        "auto": list(auto),
        # Two copies on purpose, like picks/auto: `was_edited` compares them.
        "timing": dict(timing) if timing else None,
        "auto_timing": dict(timing) if timing else None,
        # Kept so the review page can show WHY a frame was chosen instead of
        # presenting the picker as an oracle.
        "signals": {k: [round(v, 4) for v in vs] for k, vs in curves.items()},
    }
    if phases is not None:
        # Only ever present for a phased pick: a record from the old picker
        # keeps its exact keys, so an existing job's picks.json is unchanged.
        record["phases"] = dict(phases)
    return record


def phased_picks(poses: Sequence[np.ndarray | None], plan: PhasePlan, *,
                 spread: str = "phase", frames_dir: Path | None = None
                 ) -> tuple[list[int], dict[str, Any]]:
    """The picks for a plan and the `phases` block that describes them."""
    usable = [i for i, p in enumerate(poses) if p is not None]
    picks = phase_frames(usable, plan, spread)
    tags = phase_tags(plan, picks, len(poses))
    slot = plan.anticipation_slot()
    block: dict[str, Any] = {
        "plan": plan.as_record(),
        "spread": spread,
        "tags": tags,
        "anticipation": slot if slot is not None and slot < len(picks) else None,
        "closure": (loop_closure(poses, picks[0], picks[-1], frames_dir=frames_dir)
                    if plan.loop and len(picks) > 1 else None),
    }
    return picks, block


def pick(job_name: str, *, n: int | None = None, action: str | None = None,
         dedup: float | None = None, force: bool = False,
         timing: str = "template", phases: bool | None = None,
         spread: str = "phase", work_dir: Path | None = None) -> dict[str, Any]:
    """`phases` None: the template for a studio action the art bible does
    not budget (config.phases_default), the key-pose picker otherwise.
    `spread` (library only): "even" keeps the phase labels but samples the
    clip uniformly - the bake-off's routes."""
    job = Job.open(job_name, work_dir)
    manifest = job.read_manifest()
    action = (action or manifest.get("action") or "").strip().lower()
    if action in config.NO_MOTION_ACTIONS:
        raise ValueError(
            f"the art bible gives {action!r} no motion, so it gets no frames "
            f"(section 8). Nothing to pick.")
    if not job.keypoints_path.exists():
        raise FileNotFoundError(
            f"no {job.keypoints_path} - run `anim pose` first")
    if spread not in SPREADS:
        raise ValueError(f"spread must be one of {SPREADS}, not {spread!r}")

    payload = read_json(job.keypoints_path)
    poses, _ = posekit.series_from(payload)
    phased = config.phases_default(action) if phases is None else bool(phases)
    plan = phase_plan(action, budget=n, slug=manifest.get("slug")) if phased else None
    count = plan.budget if plan else (n if n is not None else config.frames_for_action(action))
    dedup = config.PICK_DEDUP if dedup is None else dedup

    existing = read_picks(job.picks_path)
    if was_edited(existing) and not force:
        if list(existing.get("picks") or []) != list(existing.get("auto") or []):
            what = f"picks {existing['picks']} differ from auto {existing['auto']}"
        else:
            what = (f"timing {existing.get('timing')} differs from auto_timing "
                    f"{existing.get('auto_timing')}")
        raise RuntimeError(
            f"{job.picks_path} was edited by hand ({what}). Refusing to "
            f"overwrite; pass --force to re-pick from scratch.")

    block = None
    if plan is not None:
        auto, block = phased_picks(poses, plan, spread=spread, frames_dir=job.frames_dir)
        shape = phase_timing(plan, block["tags"], fps=manifest.get("fps"))
        if timing == "clip":
            shape = {**shape, "weights": clip_timing(auto, len(poses)), "source": "clip",
                     "durations_ms": None}
        elif timing not in TIMING_MODES:
            raise ValueError(f"--timing must be one of {TIMING_MODES}, not {timing!r}")
    else:
        auto = auto_picks(poses, count, dedup)
        shape = auto_timing(action, auto, len(poses), timing)
    record = new_picks(auto, auto, n=count, action=action, dedup=dedup,
                       curves=signals(poses), timing=shape, phases=block)
    write_json_in_place(job.picks_path, record)
    job.update_manifest(picks=auto)
    return record


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job", help="job name, e.g. knight_strike")
    parser.add_argument("-n", type=int, default=None,
                        help="number of key poses (default: the art bible's "
                             "budget for the action)")
    parser.add_argument("--action", default=None,
                        help="override the action recorded in job.json")
    parser.add_argument("--dedup", type=float, default=config.PICK_DEDUP,
                        help=f"minimum pose distance between picks, in torso "
                             f"lengths (default {config.PICK_DEDUP})")
    parser.add_argument("--force", action="store_true",
                        help="overwrite a hand-edited picks.json")
    parser.add_argument("--timing", choices=TIMING_MODES, default="template",
                        help="where the timing shape comes from: the per-verb "
                             "template (default) or the clip's own frame gaps")
    parser.add_argument("--phases", dest="phases", action="store_true", default=None,
                        help="pick against the studio's phase template (start, "
                             "wind-up, action, wind-down, end, loop) and its "
                             "frame budget - the default for attack, run, jump")
    parser.add_argument("--no-phases", dest="phases", action="store_false",
                        help="the 2-6 key-pose picker, whatever the action")


def run_stage(args: argparse.Namespace) -> int:
    record = pick(args.job, n=args.n, action=args.action, dedup=args.dedup,
                  force=args.force, timing=args.timing, phases=args.phases)
    print(f"pick: {record['n']} key pose(s) for {record['action']}: "
          f"{record['picks']}")
    block = record.get("phases")
    if block:
        plan = block["plan"]
        print(f"  phases: {plan['action']} budget {plan['budget']} "
              f"({'loop' if plan['loop'] else 'one-shot'}, {plan['source']}): "
              + " ".join(f"{k} {v}" for k, v in plan["frames"].items()))
        closure = block.get("closure")
        if closure:
            verdict = ("closes" if closure["ok"] else
                       "does NOT close" if closure["ok"] is False else "closure unknown")
            print(f"  loop {verdict}: {closure['metric']} distance {closure['distance']} "
                  f"(tolerance {closure['tolerance']})")
    shape = record.get("timing")
    if shape:
        print(f"  timing: {shape['weights']} anchor {shape['anchor']} ({shape['source']})")
    else:
        print("  timing: none (no template row for this action and count; "
              "the game plays it uniform)")
    return 0


run = run_stage
