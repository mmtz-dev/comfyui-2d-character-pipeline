#!/usr/bin/env python3
"""Render a 2D joint list as a DWPose-style skeleton PNG, using rtmlib itself.

the 3D-intermediate spike feature (Feature 9)
=============================================

Why this file exists
--------------------
The 3D arm of the spike gets its poses from a rig (a Mixamo clip projected to 2D by
``Astropulse/mixamotoopenpose``) instead of from DWPose running on a filmed clip. Both arms
then go through the SAME generation stages (``gpu/workflows/identity_pass.json``, ``anim post``),
so the ONLY difference between them must be the pose. Anything else that differs turns the
comparison into a comparison of renderers.

And the renderers really do differ. ``mixamotoopenpose`` ships the canonical OpenPose
``draw_bodypose``; the 2D chain feeds the ControlNet ``rtmlib``'s
``draw_skeleton(openpose_skeleton=True)``. Measured differences (see ``RESEARCH.md`` §3):

  * limb alpha: canonical blends at 0.6; rtmlib's blend is a no-op
    (``cv2.addWeighted(img, 1-alpha, img, alpha, 0)`` blends the image with ITSELF), so rtmlib's
    limbs land fully opaque;
  * limb colour 9 (neck->left_hip) is ``[0, 255, 225]`` in rtmlib, ``[0, 255, 255]`` canonically;
  * limb colour 13 (nose->right_eye) is ``[255, 0, 170]`` in rtmlib, ``[85, 0, 255]`` canonically;
  * rtmlib reverses JOINT colours (``kpt_info['color'][::-1]``) but not LIMB colours;
  * rtmlib silently drops any limb with an endpoint at or past the frame edge.

So this module does not reimplement anything. It normalises whatever joint list it is handed
into the ``(1, 18, 2)`` + ``(1, 18)`` arrays rtmlib wants and calls rtmlib. If rtmlib is
upgraded, both arms of the comparison move together, which is the point.

Inputs it accepts
-----------------
1. ``mixamotoopenpose`` frames — what ``convert_dae_to_openpose()`` returns in memory. It never
   writes these to disk, so dump them yourself::

       import json, mixamo_to_openpose as m2o
       frames = m2o.convert_dae_to_openpose("slash.dae", image_size=(512, 512),
                                            rotation_angles=(0, 90, 0), scale_factor=2.0)
       json.dump(frames, open("slash.json", "w"))

   Shape: ``[{"candidate": [[x, y, c], ... x18], "subset": [[0..17 or -1, ...]]}, ...]``

2. OpenPose standard JSON — ``{"canvas_width", "canvas_height", "people":
   [{"pose_keypoints_2d": [x0,y0,c0, ... x17,y17,c17]}]}``. Same shape the fork's
   ``tools/render_openpose.py`` reads, so hand-authored poses from an OpenPose editor work.

3. A bare list of 18 ``[x, y, c]`` triples, or ``[x, y]`` pairs (confidence defaults to 1.0).

COCO-18 order, in all three (this is also ``rtmlib.openpose18``'s order and
``mixamotoopenpose``'s ``openpose_keypoints`` order — all three agree)::

    0 nose        4 r_wrist     8  r_hip     12 l_knee     16 r_ear
    1 neck        5 l_shoulder  9  r_knee    13 l_ankle    17 l_ear
    2 r_shoulder  6 l_elbow     10 r_ankle   14 r_eye
    3 r_elbow     7 l_wrist     11 l_hip     15 l_eye

CLI
---
    python3 render_openpose.py POSE.json OUT_DIR [--width 512] [--height 512] ...
    python3 render_openpose.py POSE.json OUT.png --frame 0

Run it inside the CPU pipeline container, which already has ``rtmlib==0.0.16``:

    docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
      run --rm --entrypoint python anim spike3d/render_openpose.py ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants that describe rtmlib's behaviour. Read from rtmlib where possible so
# a version bump cannot silently invalidate them.
# ---------------------------------------------------------------------------

NUM_KEYPOINTS = 18

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
)

# rtmlib's draw_skeleton() doubles what it is given before handing it to draw_openpose, where it
# becomes the ellipse's minor axis. line_width=2 therefore yields a half-width of 4 px, which is
# the same ink as canonical OpenPose's stickwidth=4. 512 px is the reference canvas that value
# was tuned for, so thickness is scaled off it.
REFERENCE_EDGE = 512
DEFAULT_LINE_WIDTH = 2

# draw_openpose() hard-codes the joint-dot radius at 4 px and ignores the `radius` argument
# entirely (the line that used it is commented out upstream). Recorded so the docs and the
# thickness maths do not pretend otherwise.
RTMLIB_JOINT_RADIUS_IS_FIXED_AT = 4

# draw_skeleton() forwards this to draw_openpose(); a joint scoring below it is not drawn, and a
# limb with either endpoint below it is not drawn.
DEFAULT_KPT_THR = 0.5


def _rtmlib_draw_skeleton():
    """Import lazily so ``--help`` and the pure-geometry helpers work without rtmlib."""
    try:
        from rtmlib import draw_skeleton  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            "rtmlib is not importable. This module must run inside the CPU pipeline container "
            "(rtmlib==0.0.16 is in requirements.lock):\n"
            "  docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \\\n"
            "    run --rm --entrypoint python anim spike3d/render_openpose.py --help"
        ) from exc
    return draw_skeleton


def rtmlib_openpose18_limbs() -> list[tuple[int, int, tuple[int, int, int]]]:
    """The limb list rtmlib will actually draw, straight from rtmlib's own tables.

    Returns ``(joint_a, joint_b, colour)`` triples in rtmlib's draw order. Used by
    :func:`limbs_dropped_at_border` and by the tests, so neither hard-codes a copy of the table.
    """
    from rtmlib.visualization.skeleton import openpose18  # type: ignore

    name_to_id = {k["name"]: k["id"] for k in openpose18["keypoint_info"].values()}
    limbs: list[tuple[int, int, tuple[int, int, int]]] = []
    for _, info in sorted(openpose18["skeleton_info"].items()):
        a, b = info["link"]
        limbs.append((name_to_id[a], name_to_id[b], tuple(info["color"])))
    return limbs


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _as_triples(raw: Iterable[Sequence[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for kp in raw:
        vals = list(kp)
        if len(vals) == 2:
            vals.append(1.0)
        if len(vals) != 3:
            raise ValueError(f"keypoint must be [x, y] or [x, y, confidence], got {kp!r}")
        out.append([float(vals[0]), float(vals[1]), float(vals[2])])
    if len(out) != NUM_KEYPOINTS:
        raise ValueError(f"expected {NUM_KEYPOINTS} keypoints (COCO-18), got {len(out)}")
    return out


def frames_from_mixamotoopenpose(frames: Sequence[dict[str, Any]]) -> list[list[list[float]]]:
    """``[{"candidate": [[x,y,c]...], "subset": [[...]]}, ...]`` -> list of COCO-18 triples.

    ``mixamotoopenpose`` marks a missing joint by putting ``-1`` in ``subset`` AND ``[0,0,0]`` in
    ``candidate``; both are honoured, and a missing joint is emitted with confidence 0.0 so
    rtmlib's ``kpt_thr`` gate drops it.
    """
    out: list[list[list[float]]] = []
    for i, frame in enumerate(frames):
        try:
            candidate = frame["candidate"]
            subset = frame["subset"]
        except (TypeError, KeyError) as exc:
            raise ValueError(f"frame {i}: not a mixamotoopenpose frame ({exc})") from exc
        if not subset:
            raise ValueError(f"frame {i}: empty subset")
        person = subset[0]
        if len(person) != NUM_KEYPOINTS:
            raise ValueError(
                f"frame {i}: subset[0] has {len(person)} entries, expected {NUM_KEYPOINTS}"
            )
        triples: list[list[float]] = []
        for j in range(NUM_KEYPOINTS):
            idx = int(person[j])
            if idx < 0 or idx >= len(candidate):
                triples.append([0.0, 0.0, 0.0])
                continue
            x, y, c = candidate[idx][0], candidate[idx][1], candidate[idx][2]
            triples.append([float(x), float(y), float(c)])
        out.append(triples)
    return out


def frames_from_openpose_json(data: dict[str, Any]) -> tuple[list[list[list[float]]], int, int]:
    """OpenPose standard JSON -> (frames, canvas_width, canvas_height)."""
    w = int(data.get("canvas_width", REFERENCE_EDGE))
    h = int(data.get("canvas_height", REFERENCE_EDGE))
    people = data.get("people") or []
    if not people:
        raise ValueError("OpenPose JSON has no people")
    frames: list[list[list[float]]] = []
    for i, person in enumerate(people):
        flat = person["pose_keypoints_2d"]
        if len(flat) != NUM_KEYPOINTS * 3:
            raise ValueError(
                f"people[{i}]: expected {NUM_KEYPOINTS * 3} values, got {len(flat)}"
            )
        frames.append(_as_triples([flat[k * 3 : k * 3 + 3] for k in range(NUM_KEYPOINTS)]))
    return frames, w, h


def load_frames(path: Path) -> tuple[list[list[list[float]]], int | None, int | None]:
    """Sniff the JSON shape and return ``(frames, canvas_w_or_None, canvas_h_or_None)``."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "people" in data:
        frames, w, h = frames_from_openpose_json(data)
        return frames, w, h

    if isinstance(data, dict) and "candidate" in data:
        data = [data]

    if isinstance(data, list) and data and isinstance(data[0], dict) and "candidate" in data[0]:
        return frames_from_mixamotoopenpose(data), None, None

    # A bare list of 18 keypoints, or a list of such lists.
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
        if len(data) == NUM_KEYPOINTS and isinstance(data[0][0], (int, float)):
            return [_as_triples(data)], None, None
        return [_as_triples(f) for f in data], None, None

    raise ValueError(
        f"{path}: unrecognised pose JSON. Expected mixamotoopenpose candidate/subset frames, "
        "OpenPose standard JSON, or a list of 18 [x, y, c] keypoints."
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def fit_to_canvas(
    triples: Sequence[Sequence[float]], width: int, height: int, margin: int = 8
) -> list[list[float]]:
    """Uniformly scale + translate the visible joints to sit inside the canvas with a margin.

    Two reasons this is not cosmetic:

    * rtmlib drops a limb outright when an endpoint satisfies ``x <= 0 or x >= w or y <= 0 or
      y >= h``. A pose touching the edge loses whole limbs with no warning.
    * ``mixamotoopenpose`` centres and scales for ITS canvas; asking for a different cell size
      (the 2D chain's cell is 304x320, not 512x512) can push the pose out of frame.

    Aspect ratio is preserved, so the pose is never stretched. Joints with confidence <= 0 are
    moved with the rest but stay at confidence 0.
    """
    vis = [t for t in triples if t[2] > 0]
    if not vis:
        return [list(t) for t in triples]

    xs = [t[0] for t in vis]
    ys = [t[1] for t in vis]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    avail_w = max(1, width - 2 * margin)
    avail_h = max(1, height - 2 * margin)
    scale = min(
        avail_w / span_x if span_x > 0 else float("inf"),
        avail_h / span_y if span_y > 0 else float("inf"),
    )
    if scale == float("inf"):
        scale = 1.0

    cx_src = (max(xs) + min(xs)) / 2.0
    cy_src = (max(ys) + min(ys)) / 2.0

    out: list[list[float]] = []
    for x, y, c in triples:
        out.append(
            [
                (x - cx_src) * scale + width / 2.0,
                (y - cy_src) * scale + height / 2.0,
                float(c),
            ]
        )
    return out


def limbs_dropped_at_border(
    triples: Sequence[Sequence[float]], width: int, height: int, kpt_thr: float = DEFAULT_KPT_THR
) -> list[tuple[str, str]]:
    """Which limbs rtmlib will silently skip, by name. Mirrors rtmlib's own guard exactly."""

    def out_of_frame(kp: Sequence[float]) -> bool:
        x, y = kp[0], kp[1]
        return x <= 0 or x >= width or y <= 0 or y >= height

    dropped: list[tuple[str, str]] = []
    for a, b, _colour in rtmlib_openpose18_limbs():
        ka, kb = triples[a], triples[b]
        if (
            out_of_frame(ka)
            or out_of_frame(kb)
            or ka[2] < kpt_thr
            or kb[2] < kpt_thr
        ):
            dropped.append((KEYPOINT_NAMES[a], KEYPOINT_NAMES[b]))
    return dropped


def line_width_for(width: int, height: int, base: int = DEFAULT_LINE_WIDTH) -> int:
    """Scale rtmlib's ``line_width`` off the 512 px canvas its default was tuned for.

    A 304x320 cell at the default gives a hairline rig; a 1024 px canvas gives a rig too thin
    relative to the figure. rtmlib's joint radius is fixed at 4 px and cannot be scaled, so
    below roughly 256 px the dots start to dominate the limbs -- render large and let
    ``anim strip``/``anim post`` do the downscaling.
    """
    scale = min(width, height) / REFERENCE_EDGE
    return max(1, int(round(base * scale)))


# ---------------------------------------------------------------------------
# The render
# ---------------------------------------------------------------------------


def render(
    triples: Sequence[Sequence[float]],
    width: int = REFERENCE_EDGE,
    height: int = REFERENCE_EDGE,
    line_width: int | None = None,
    kpt_thr: float = DEFAULT_KPT_THR,
) -> np.ndarray:
    """COCO-18 ``[x, y, c]`` triples -> ``(height, width, 3)`` uint8 array, drawn by rtmlib.

    The returned array carries rtmlib's raw channel order: limb colours as written in
    ``openpose18['skeleton_info']``, joint colours reversed (rtmlib's own quirk). See
    :func:`save_png` for how that is committed to a file.
    """
    triples = _as_triples(triples)
    draw_skeleton = _rtmlib_draw_skeleton()

    keypoints = np.array([[[t[0], t[1]] for t in triples]], dtype=np.float32)
    scores = np.array([[t[2] for t in triples]], dtype=np.float32)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if line_width is None:
        line_width = line_width_for(width, height)

    return draw_skeleton(
        canvas,
        keypoints,
        scores,
        openpose_skeleton=True,
        kpt_thr=kpt_thr,
        line_width=line_width,
    )


def save_png(array: np.ndarray, path: Path, channel_order: str = "rgb") -> None:
    """Write the array, committing to a channel order.

    rtmlib writes limb colours unreversed and joint colours reversed, so exactly one of the two
    is channel-swapped whichever way the array is saved:

    ``rgb`` (default)
        Save the array as-is. Limb colours then match the canonical OpenPose table exactly --
        limbs are most of the ink, and the tables are written in RGB. Joint dots come out
        reversed, which is what rtmlib produces in the 2D chain too.
    ``bgr``
        Save with cv2 semantics (channels reversed on the way out): joint dots canonical, limbs
        swapped.

    Both arms of the spike MUST use the same setting, or the comparison measures the renderer
    instead of the pose. RESEARCH.md §3.2 has the full argument.
    """
    if channel_order not in ("rgb", "bgr"):
        raise ValueError(f"channel_order must be 'rgb' or 'bgr', got {channel_order!r}")
    from PIL import Image

    out = array if channel_order == "rgb" else array[..., ::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(out), mode="RGB").save(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="render_openpose.py",
        description=(
            "Render a 2D joint list as a DWPose-style skeleton PNG by calling rtmlib's own "
            "draw_skeleton(openpose_skeleton=True) -- the same renderer the 2D chain feeds the "
            "Z-Image Fun ControlNet."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("pose_json", type=Path, help="mixamotoopenpose frames, OpenPose JSON, or 18 keypoints")
    p.add_argument("out", type=Path, help="output PNG, or a directory when there are many frames")
    p.add_argument("--width", type=int, default=REFERENCE_EDGE)
    p.add_argument("--height", type=int, default=REFERENCE_EDGE)
    p.add_argument("--frame", type=int, default=None, help="render only this frame index")
    p.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="rtmlib line_width; default scales off a 512px reference canvas",
    )
    p.add_argument("--kpt-thr", type=float, default=DEFAULT_KPT_THR)
    p.add_argument(
        "--no-fit",
        action="store_true",
        help="do not rescale the pose into the canvas (fit is on by default; rtmlib drops "
        "limbs that touch the frame edge)",
    )
    p.add_argument("--margin", type=int, default=8, help="border kept clear by --fit")
    p.add_argument(
        "--channel-order",
        choices=("rgb", "bgr"),
        default="rgb",
        help="how the array is committed to PNG; see save_png (default: rgb)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    frames, json_w, json_h = load_frames(args.pose_json)
    width = args.width
    height = args.height
    # A pose authored for an explicit canvas keeps it unless the caller overrode the default.
    if json_w and args.width == REFERENCE_EDGE:
        width = json_w
    if json_h and args.height == REFERENCE_EDGE:
        height = json_h

    if args.frame is not None:
        if not 0 <= args.frame < len(frames):
            print(f"frame {args.frame} out of range (0..{len(frames) - 1})", file=sys.stderr)
            return 2
        frames = [frames[args.frame]]

    many = len(frames) > 1
    if many and args.out.suffix.lower() == ".png":
        print(
            f"{len(frames)} frames but --out looks like a single file; pass a directory",
            file=sys.stderr,
        )
        return 2

    pad = len(str(max(0, len(frames) - 1)))
    written = 0
    for i, triples in enumerate(frames):
        if not args.no_fit:
            triples = fit_to_canvas(triples, width, height, margin=args.margin)

        dropped = limbs_dropped_at_border(triples, width, height, kpt_thr=args.kpt_thr)
        if dropped:
            print(
                f"frame {i}: rtmlib will skip {len(dropped)} limb(s): "
                + ", ".join(f"{a}->{b}" for a, b in dropped),
                file=sys.stderr,
            )

        img = render(
            triples,
            width=width,
            height=height,
            line_width=args.line_width,
            kpt_thr=args.kpt_thr,
        )
        out_path = args.out if not many else args.out / f"{i:0{pad}d}.png"
        save_png(img, out_path, channel_order=args.channel_order)
        print(f"  {out_path}  {width}x{height}  line_width="
              f"{args.line_width if args.line_width is not None else line_width_for(width, height)}")
        written += 1

    print(f"{written} skeleton PNG(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
