"""Keypoints: the schema on disk, normalisation, distance, and the renderer.

Split out of the `pose` stage so that everything except the actual inference can
be used - and tested - without downloading a 200 MB ONNX model. `pick` needs the
maths, `strip` needs the renderer, and neither should pull in onnxruntime.

THE SCHEMA (pose/keypoints.json)
--------------------------------
    {
      "schema": "anim.keypoints/1",
      "rig": "openpose134",          # rtmlib skeleton name; names[i] is joint i
      "names": ["nose", "neck", ...],
      "model": "wholebody",          # rtmlib family that produced this
      "mode": "balanced",
      "kpt_thr": 0.4,
      "source_size": [1080, 1920],   # the frame the pixel coords are in
      "frames": [
        {"index": 0,                 # 0-based, matches frames/0001.png
         "file": "0001.png",
         "keypoints": [[x, y], ...], # PIXELS in the source frame, one per joint
         "scores": [0.93, ...],      # 0..1 confidence, same length
         "person": {"bbox": [x1, y1, x2, y2], "score": 0.99}},
        ...
      ]
    }

`keypoints` is null for a frame where no person was found - the picker skips
those rather than inventing a pose. One person per frame: these are single-actor
clips by the art bible's filming rules, and the largest/highest-scoring
detection wins.

A rig job may add, per frame, `"props": [{"kind": "sword", "polygons": [[[x, y],
...], ...]}, ...]` - closed outlines in the same pixels as `keypoints` (the
skeleton props feature, `anim.rigkit`). Filmed clips never have it, and the
picker ignores it: props are drawing, not pose.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

KEYPOINTS_SCHEMA = "anim.keypoints/1"

# Joint names used by the normaliser. The first 18 entries of openpose134 are
# exactly openpose18, so the same names resolve in both rigs - which is why the
# maths below is written against NAMES and never against indices.
HIPS = ("left_hip", "right_hip")
SHOULDERS = ("left_shoulder", "right_shoulder")


def rig_names(rig: str) -> list[str]:
    """Joint names for an rtmlib skeleton, in index order.

    Read out of rtmlib's own skeleton tables rather than copied here, so a
    keypoints.json can never disagree with the renderer that drew it.
    """
    from rtmlib.visualization import skeleton as sk

    table = getattr(sk, rig, None)
    if table is None:
        raise ValueError(f"unknown rig {rig!r}; rtmlib knows "
                         f"{[n for n in dir(sk) if not n.startswith('_')]}")
    info = table["keypoint_info"]
    return [info[i]["name"] for i in range(len(info))]


def rig_for(num_keypoints: int) -> str:
    """The openpose-style rig rtmlib's draw_skeleton picks for this joint count."""
    return {17: "animal17", 18: "openpose18",
            26: "halpe26", 134: "openpose134"}.get(num_keypoints, "")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalise(keypoints: Sequence[Sequence[float]], scores: Sequence[float],
              names: Sequence[str], kpt_thr: float = 0.4) -> np.ndarray | None:
    """Root-centred, torso-scaled pose. Low-confidence joints become NaN.

    Two frames of the same action filmed a step nearer the camera are the same
    pose; in pixels they are wildly different. So:
      - root   = the midpoint of the hips (the pelvis, i.e. the pivot the game
                 also uses - bottom-center is the feet, but the pelvis is what
                 stays put through a swing)
      - scale  = torso length, shoulder-centre to hip-centre. One torso == 1.0,
                 which is what makes config.PICK_DEDUP a number with a meaning.

    Falls back to the keypoint spread when the torso is degenerate (a profile
    crouch can collapse it), and returns None when there is no usable root -
    the caller drops the frame rather than guessing.
    """
    kp = np.asarray(keypoints, dtype=float)
    sc = np.asarray(scores, dtype=float)
    if kp.ndim != 2 or kp.shape[1] != 2 or kp.shape[0] != sc.shape[0]:
        raise ValueError("keypoints must be (K, 2) and scores (K,)")
    index = {name: i for i, name in enumerate(names)}
    valid = sc >= kpt_thr

    def centre(pair: Iterable[str]) -> np.ndarray | None:
        points = [kp[index[n]] for n in pair
                  if n in index and valid[index[n]]]
        if not points:
            return None
        return np.mean(points, axis=0)

    root = centre(HIPS)
    neck = centre(SHOULDERS)
    if root is None:
        root = neck
    if root is None:
        return None

    scale = 0.0
    if neck is not None:
        scale = float(np.linalg.norm(neck - root))
    if scale < 1e-6:
        spread = kp[valid]
        if len(spread) >= 2:
            scale = float(np.linalg.norm(spread.max(axis=0) - spread.min(axis=0))) / 4.0
    if scale < 1e-6:
        return None

    out = (kp - root) / scale
    out[~valid] = np.nan
    return out


def pose_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean joint displacement between two normalised poses, in torso lengths.

    Only joints present in BOTH poses count; a joint the detector lost is not
    evidence of movement. Returns inf when they share nothing, so a caller
    ranking candidates treats "no comparison possible" as "maximally different"
    rather than as "identical".
    """
    if a is None or b is None:
        return math.inf
    both = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    if not both.any():
        return math.inf
    return float(np.linalg.norm(a[both] - b[both], axis=1).mean())


def series_from(payload: dict[str, Any]) -> tuple[list[np.ndarray | None], list[str]]:
    """Normalised pose per frame from a keypoints.json payload, plus joint names."""
    names = payload.get("names") or rig_names(payload["rig"])
    thr = float(payload.get("kpt_thr", 0.4))
    poses: list[np.ndarray | None] = []
    for frame in payload["frames"]:
        kp, sc = frame.get("keypoints"), frame.get("scores")
        poses.append(None if not kp else normalise(kp, sc, names, thr))
    return poses, list(names)


# ---------------------------------------------------------------------------
# Rendering - ALWAYS rtmlib's own DWPose renderer
# ---------------------------------------------------------------------------
LINE_WIDTH = 2  # rtmlib's default: the DWPose renders the ControlNet was trained on


def line_width_for(size: tuple[int, int]) -> int:
    """Limb thickness. Always rtmlib's default, whatever the canvas.

    The first cut scaled this with the canvas (1.6% of the short edge: 5 px at
    304, 8 px at 512) on the theory that a hairline rig is one the ControlNet
    cannot see. Measured on the anchor gate (2026-09-15) the opposite is true:
    rtmlib draws each limb as an ellipse whose minor axis is 2 x line_width, so
    at 5-8 px a 190-700 px figure is a chain of fat blobs, and the Fun ControlNet
    Union 2.1 (full and lite) treated it as no pose at all - a lying-down
    skeleton gave standing knights, a lunge gave a guard stance, strength 1.5
    gave helmet close-ups. The same lunge drawn at 2 px was followed at 0.8 and
    nailed at 1.0. The model card's own pose conditions are DWPose renders with
    ~2 px sticks on a 1024 px figure; the thickness is part of the style the
    ControlNet learned, not geometry we get to choose. `size` is kept in the
    signature so callers do not change.
    """
    return LINE_WIDTH


# Props are stroked in white: a colour DWPose never uses, so a blade cannot be
# mistaken for a limb, and the colour of the canny edges the same union
# ControlNet was trained on. Measured 2026-09-16 (DEVLOG): outline, single
# line and filled silhouette all put the sword in the drawn hand; outline kept
# the shield's shape best.
PROP_COLOUR = (255, 255, 255)


def letterbox(source_size: tuple[int, int],
              out_size: tuple[int, int]) -> tuple[float, float, float]:
    """(scale, off_x, off_y) that fits `source_size` inside `out_size`, centred.

    The one transform between keypoints.json pixels and a rendered cell. `draw`
    uses it for the skeleton and the prop outlines; the part palette lock
    feature uses it to put its SAM prompts on the generated figure - so a
    prompt lands exactly where the stick it was derived from was drawn.
    """
    src_w, src_h = source_size
    out_w, out_h = out_size
    scale = min(out_w / src_w, out_h / src_h)
    return scale, (out_w - src_w * scale) / 2.0, (out_h - src_h * scale) / 2.0


def draw(keypoints: Sequence[Sequence[float]], scores: Sequence[float],
         source_size: tuple[int, int], out_size: tuple[int, int],
         kpt_thr: float = 0.4,
         props: Sequence[dict[str, Any]] | None = None) -> np.ndarray:
    """Render one OpenPose-style skeleton onto a black canvas of `out_size`.

    Always `rtmlib.draw_skeleton(..., openpose_skeleton=True)` - never a
    hand-authored rig. Z-Image's Fun ControlNet was trained on DWPose's exact
    colour convention and limb-as-ellipse style, and rejects anything else
    (gpu/workflows/NOTES.md, upstream discussion #28). The only liberties taken
    are the canvas size and the line width, which are geometry, not style.

    Coordinates are letterboxed rather than stretched: a skeleton squashed to a
    different aspect ratio is a skeleton with the wrong proportions, and the
    reference sprite - not the rig - is supposed to own proportions.

    `props` is a frame's `props` list from the keypoints payload (the skeleton
    props feature): closed outlines in the same source pixels as the joints,
    stroked over the skeleton at the same line width.
    """
    import cv2
    from rtmlib import draw_skeleton

    out_w, out_h = out_size
    scale, off_x, off_y = letterbox(source_size, out_size)

    kp = np.asarray(keypoints, dtype=float)
    placed = np.empty_like(kp)
    placed[:, 0] = kp[:, 0] * scale + off_x
    placed[:, 1] = kp[:, 1] * scale + off_y

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas = draw_skeleton(canvas,
                           placed[None, :, :],
                           np.asarray(scores, dtype=float)[None, :],
                           openpose_skeleton=True,
                           kpt_thr=kpt_thr,
                           line_width=line_width_for(out_size))
    for prop in props or ():
        for polygon in prop.get("polygons") or ():
            pts = np.asarray(polygon, dtype=float)
            pts[:, 0] = pts[:, 0] * scale + off_x
            pts[:, 1] = pts[:, 1] * scale + off_y
            cv2.polylines(canvas, [np.round(pts).astype(np.int32)], True,
                          PROP_COLOUR, thickness=line_width_for(out_size))
    return canvas
