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

NEVER CLOBBERS A HUMAN EDIT
---------------------------
picks.json records both what the picker chose (`auto`) and what is in force
(`picks`). When they differ, a person changed it - in the review page or in an
editor - and re-running `pick` would throw that away. So it refuses, and says
how to override (`--force`).
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
# Stage entry point
# ---------------------------------------------------------------------------
def read_picks(path: Path) -> dict[str, Any] | None:
    return read_json(path) if Path(path).exists() else None


def was_edited(existing: dict[str, Any] | None) -> bool:
    """True when a person changed picks.json after the picker wrote it."""
    if not existing:
        return False
    return list(existing.get("picks") or []) != list(existing.get("auto") or [])


def new_picks(picks: list[int], auto: list[int], *, n: int, action: str,
              dedup: float, curves: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "schema": PICKS_SCHEMA,
        "n": n,
        "action": action,
        "dedup": dedup,
        "picks": list(picks),
        "auto": list(auto),
        # Kept so the review page can show WHY a frame was chosen instead of
        # presenting the picker as an oracle.
        "signals": {k: [round(v, 4) for v in vs] for k, vs in curves.items()},
    }


def pick(job_name: str, *, n: int | None = None, action: str | None = None,
         dedup: float | None = None, force: bool = False,
         work_dir: Path | None = None) -> dict[str, Any]:
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
        raise RuntimeError(
            f"{job.picks_path} was edited by hand (picks {existing['picks']} "
            f"differ from auto {existing['auto']}). Refusing to overwrite; "
            f"pass --force to re-pick from scratch.")

    auto = auto_picks(poses, count, dedup)
    record = new_picks(auto, auto, n=count, action=action, dedup=dedup,
                       curves=signals(poses))
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


def run_stage(args: argparse.Namespace) -> int:
    record = pick(args.job, n=args.n, action=args.action, dedup=args.dedup,
                  force=args.force)
    print(f"pick: {record['n']} key pose(s) for {record['action']}: "
          f"{record['picks']}")
    return 0


run = run_stage
