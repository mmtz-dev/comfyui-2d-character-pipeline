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
              timing: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
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


def pick(job_name: str, *, n: int | None = None, action: str | None = None,
         dedup: float | None = None, force: bool = False,
         timing: str = "template", work_dir: Path | None = None) -> dict[str, Any]:
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

    payload = read_json(job.keypoints_path)
    poses, _ = posekit.series_from(payload)
    count = n if n is not None else config.frames_for_action(action)
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

    auto = auto_picks(poses, count, dedup)
    shape = auto_timing(action, auto, len(poses), timing)
    record = new_picks(auto, auto, n=count, action=action, dedup=dedup,
                       curves=signals(poses), timing=shape)
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


def run_stage(args: argparse.Namespace) -> int:
    record = pick(args.job, n=args.n, action=args.action, dedup=args.dedup,
                  force=args.force, timing=args.timing)
    print(f"pick: {record['n']} key pose(s) for {record['action']}: "
          f"{record['picks']}")
    shape = record.get("timing")
    if shape:
        print(f"  timing: {shape['weights']} anchor {shape['anchor']} ({shape['source']})")
    else:
        print("  timing: none (no template row for this action and count; "
              "the game plays it uniform)")
    return 0


run = run_stage
