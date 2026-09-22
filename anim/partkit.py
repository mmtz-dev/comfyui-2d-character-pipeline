"""The part palette lock feature: part masks from the rig, palettes per part.

A unit's parts - shield, sword, helmet, torso, arms, legs - must keep the same
colours in every action. The generation cannot promise that (the anchor gate,
2026-09-15: shield colour and armour tone wander between `strike`, `hit` and
`death`), so `post` gets a deterministic safety net:

    HD frame  ->  SAM 2.1 masks, one per part, PROMPTED FROM THE RIG
              ->  masks downscaled with the frame (coverage, not nearest)
              ->  every pixel inside a part snapped to THAT PART'S palette

and the part palettes are read ONCE from the accepted base sprite, by running
the same masks over the base render and collecting the colours inside each.

WHY THE PROMPTS COME FROM THE RIG AND NOT FROM TEXT
---------------------------------------------------
SAM 3 is text-promptable but gated on the Hub with `license: other`, which
fails the repo's ungated / Apache-2.0 rule (checked 2026-09-17). SAM 2.1 is
Apache-2.0 and ungated, and takes points and boxes - and the pipeline already
knows where every part is: `pose/keypoints.json` carries the COCO-18 joints
the ControlNet was driven with, plus - since the skeleton props feature - the
sword and shield outlines in the same pixels. A part is therefore a list of
joints (and midpoints between joints) that lie on it, or a prop kind whose
polygon is on it. Positive points inside the part, the other parts' points as
negatives, and the prop's polygon bounds as a box. No text, no guessing what
"shield" means to a model, and the same prompts for the base sprite and every
action, which is what makes the palettes comparable.

WHY MASKS ARE DOWNSCALED BY COVERAGE
------------------------------------
The frame goes from ~250 px to ~53 px tall. A sword is 6 px wide at HD and one
pixel at 64 px; nearest sampling of its mask lands between the columns and the
sword vanishes from its own part. So each HD mask is box-filtered to the
sprite grid (a coverage fraction per output pixel) and each output pixel goes
to the part that covers most of it, priority breaking ties - which is also
how overlaps are resolved: props over body, earlier rows over later.

COLOUR SPACE
------------
The unit-level snap is ImageMagick's `-remap`, which measures in sRGB. The
within-part re-snap uses Oklab: the question it answers is "which of the
shield's four greys is this pixel", which is a question about how it LOOKS,
and Oklab distances are perceptually uniform where sRGB distances are not.
The candidates are all palette members already, so the result stays inside
the unit palette whatever the metric.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from PIL import Image

from anim import posekit

# `config` imports `Part` from here for its UNIT_PARTS table, so config is
# imported lazily inside the functions that read it - never at module level.

PARTS_SCHEMA = "anim.parts/1"

RGB = tuple[int, int, int]
Point = tuple[float, float]


# ---------------------------------------------------------------------------
# The part table entry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Part:
    """One nameable region of a unit and where its prompts come from.

    `joints` are OpenPose-18 names that lie on the part; `mids` are
    (from, to, t) - the point at fraction t along from->to, t outside [0, 1]
    extrapolates (the crown of the head is neck->nose at 1.3). `prop` names a
    rigkit prop kind whose polygon(s) ARE the part: centroid as the positive
    point, bounds as the box. Order in the unit's table is priority.
    """

    name: str
    joints: tuple[str, ...] = ()
    mids: tuple[tuple[str, str, float], ...] = ()
    prop: str | None = None
    # Also hand SAM the prop polygon's bounds as a box. Off by default:
    # measured on knight_strike (2026-09-17) the box pinned the raised shield
    # to the kite outline's footprint and SAM returned the visor-sized blob
    # inside it (864 px) instead of the shield (8046 px); on the sword it
    # collapsed one cell to nothing. Points alone found both. Turn on for a
    # rig whose props the render follows to the pixel (the clip route).
    box: bool = False


@dataclass
class Prompt:
    """What one part asks SAM for, in frame pixels."""

    part: str
    points: list[Point] = field(default_factory=list)   # positives
    negatives: list[Point] = field(default_factory=list)
    box: tuple[float, float, float, float] | None = None

    @property
    def usable(self) -> bool:
        return bool(self.points) or self.box is not None


class Segmenter(Protocol):
    """One image in, one binary mask + confidence per prompt out."""

    def segment(self, rgb: np.ndarray,
                prompts: Sequence[Prompt]) -> list[tuple[np.ndarray, float] | None]:
        ...


# ---------------------------------------------------------------------------
# Prompts from the rig
# ---------------------------------------------------------------------------
def _joint_points(entry: dict[str, Any], names: Sequence[str],
                  kpt_thr: float) -> dict[str, Point]:
    """The joints a prompt may sit on: every detected one, plus the ones a
    RIG placed and then marked `hidden` (rigkit, THE FACE ON THE FAR SIDE).
    A hidden face point has confidence 0.0 so the pose ControlNet draws no
    face on the back of a head, but the rig still put it where the head is
    - the orthographic projection lands the nose inside the helm seen from
    behind - so for SAM it is as good a prompt as it was from the front. A
    filmed payload has no `hidden`, so its low-confidence guesses stay out
    as before (the multi-view reference feature, 2026-09-22: hiding every
    face point off-front had cost the helmet its prompt on s45 / s90 / s135)."""
    kp = entry.get("keypoints") or []
    sc = entry.get("scores") or []
    hidden = set(entry.get("hidden") or ())
    out: dict[str, Point] = {}
    for i, name in enumerate(names):
        if i < len(kp) and i < len(sc) and (sc[i] >= kpt_thr or name in hidden):
            out[name] = (float(kp[i][0]), float(kp[i][1]))
    return out


def _polygon_centroid(polygon: Sequence[Sequence[float]]) -> Point:
    pts = np.asarray(polygon, dtype=float)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def prompts_for(parts: Sequence[Part], entry: dict[str, Any],
                names: Sequence[str], source_size: tuple[int, int],
                cell: tuple[int, int], kpt_thr: float = 0.4,
                box_pad: float = 0.08) -> list[Prompt]:
    """One Prompt per part for one keypoints frame, in cell pixels.

    Joints and prop polygons are in `source_size` pixels; the generated cell
    was drawn from the same skeleton letterboxed into `cell`, so the same
    letterbox (posekit.letterbox) puts the prompts on the generated figure.
    Every other part's positive points become this part's negatives.
    """
    scale, off_x, off_y = posekit.letterbox(source_size, cell)

    def place(p: Point) -> Point:
        return (p[0] * scale + off_x, p[1] * scale + off_y)

    joints = _joint_points(entry, names, kpt_thr)
    props = {p["kind"]: p.get("polygons") or []
             for p in (entry.get("props") or ())}

    prompts: list[Prompt] = []
    for part in parts:
        prompt = Prompt(part=part.name)
        if part.prop is not None:
            polys = props.get(part.prop) or []
            pts = [np.asarray(poly, dtype=float) for poly in polys if len(poly) >= 3]
            if pts:
                for poly in pts:
                    prompt.points.append(place(_polygon_centroid(poly)))
            if pts and part.box:
                allpts = np.concatenate(pts)
                x0, y0 = allpts.min(axis=0)
                x1, y1 = allpts.max(axis=0)
                pad = box_pad * max(x1 - x0, y1 - y0)
                bx0, by0 = place((x0 - pad, y0 - pad))
                bx1, by1 = place((x1 + pad, y1 + pad))
                prompt.box = (bx0, by0, bx1, by1)
        for name in part.joints:
            if name in joints:
                prompt.points.append(place(joints[name]))
        for a, b, t in part.mids:
            if a in joints and b in joints:
                pa, pb = joints[a], joints[b]
                prompt.points.append(place((pa[0] + (pb[0] - pa[0]) * t,
                                            pa[1] + (pb[1] - pa[1]) * t)))
        prompts.append(prompt)

    for prompt in prompts:
        for other in prompts:
            if other is not prompt:
                prompt.negatives.extend(other.points)
    return prompts


# ---------------------------------------------------------------------------
# SAM 2.1
# ---------------------------------------------------------------------------
def weights_path(cache_dir: Path | None = None) -> Path:
    from anim import config

    base = Path(cache_dir) if cache_dir is not None else config.SAM2_CACHE_DIR
    return base / config.SAM2_WEIGHTS["file"]


def ensure_weights(cache_dir: Path | None = None,
                   fetch: Callable[[str, Path], None] | None = None) -> Path:
    """The pinned SAM 2.1 checkpoint on disk, sha256-verified.

    Downloaded from the Hub at the pinned revision on first use, like rtmlib
    and rembg fetch theirs - but unlike those, refused if the bytes do not
    hash to `config.SAM2_WEIGHTS["sha256"]`. A frame's sidecar names this hash.
    """
    from anim import config

    spec = config.SAM2_WEIGHTS
    path = weights_path(cache_dir)
    if path.exists() and _sha256(path) == spec["sha256"]:
        return path
    url = (f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/"
           f"{spec['repo_file']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    (fetch or _download)(url, path)
    got = _sha256(path)
    if got != spec["sha256"]:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SAM 2.1 weights from {url} hash to {got}, expected "
            f"{spec['sha256']} (config.SAM2_WEIGHTS). Refusing to use them.")
    return path


def _download(url: str, dest: Path) -> None:
    import requests

    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(1 << 20):
                handle.write(chunk)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def model_entry() -> dict[str, Any]:
    """The SAM 2.1 row in the shape the sidecar's `models` lists use."""
    from anim import config

    spec = config.SAM2_WEIGHTS
    return {"repo": spec["repo"], "file": spec["file"], "sha256": spec["sha256"],
            "license": spec["license"], "revision": spec["revision"]}


class Sam2Segmenter:
    """SAM 2.1 image predictor on the CPU, prompted with points and boxes."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.checkpoint = ensure_weights(cache_dir)
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            import torch
            from anim import config
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            torch.manual_seed(0)
            model = build_sam2(config.SAM2_WEIGHTS["config"], str(self.checkpoint),
                               device="cpu")
            self._predictor = SAM2ImagePredictor(model)
        return self._predictor

    def segment(self, rgb: np.ndarray,
                prompts: Sequence[Prompt]) -> list[tuple[np.ndarray, float] | None]:
        """One (mask, consistency) per prompt.

        SAM is asked with the POSITIVE points and the box only, in multimask
        mode, and the negatives are used to CHOOSE among its three candidates
        rather than fed to it. Measured on the knight base (2026-09-17):
        feeding the other parts' points as negatives collapses every mask
        (predicted IoU 0.000, a few hundred pixels), while the three
        multimask candidates - sub-part, part, whole - reliably contain the
        right one. The pick is the candidate with the most positives inside
        and the fewest negatives inside (`consistency`, in -1..1), predicted
        IoU breaking ties; that consistency is the score the caller
        thresholds. Whole-figure candidates lose because they hold every
        other part's points.
        """
        import torch

        predictor = self._load()
        with torch.inference_mode():
            predictor.set_image(np.ascontiguousarray(rgb[:, :, :3]))
            out: list[tuple[np.ndarray, float] | None] = []
            for prompt in prompts:
                if not prompt.usable:
                    out.append(None)
                    continue
                kwargs: dict[str, Any] = {"multimask_output": True}
                if prompt.points:
                    kwargs["point_coords"] = np.asarray(prompt.points, dtype=np.float32)
                    kwargs["point_labels"] = np.ones(len(prompt.points), dtype=np.int32)
                if prompt.box is not None:
                    kwargs["box"] = np.asarray(prompt.box, dtype=np.float32)
                masks, scores, _ = predictor.predict(**kwargs)
                out.append(choose_candidate(
                    [(np.asarray(m) > 0.5, float(s)) for m, s in zip(masks, scores)],
                    prompt))
        return out


def _fraction_inside(mask: np.ndarray, points: Sequence[Point]) -> float:
    if not points:
        return 0.0
    h, w = mask.shape
    hits = 0
    for px, py in points:
        x = min(w - 1, max(0, int(px)))
        y = min(h - 1, max(0, int(py)))
        hits += bool(mask[y, x])
    return hits / len(points)


def choose_candidate(candidates: Sequence[tuple[np.ndarray, float]],
                     prompt: Prompt) -> tuple[np.ndarray, float] | None:
    """The candidate that holds the prompt's positives and not its negatives.

    Returns (mask, consistency), consistency = positives inside -
    CONTAMINATION_WEIGHT x negatives inside. Among the ELIGIBLE candidates -
    at least half the positives inside, at most MAX_CONTAMINATION of the
    negatives - the one SAM itself rates highest wins; SAM's predicted IoU is
    a good judge of "is this one whole thing", the prompts only say which
    thing. With nothing eligible the least-bad candidate comes back with a
    consistency below zero, which `resolve` treats as not found.

    Why a contamination allowance rather than zero: a raised shield covers the
    helmet in profile, so the helmet's points fall inside the shield's true
    mask (knight_strike cells 1-2, 2026-09-17). One point in four is the
    occlusion budget; more than that is a mask that swallowed a neighbour.
    Why contamination weighs double in the fallback: on the knight base the
    sword's rest polygon put one of its two points on the torso, and at equal
    weight the torso-and-sword candidate beat the sword alone.
    """
    scored: list[tuple[tuple[float, float], tuple[np.ndarray, float]]] = []
    eligible: list[tuple[float, tuple[np.ndarray, float]]] = []
    for mask, iou in candidates:
        pos = _fraction_inside(mask, prompt.points) if prompt.points else 1.0
        neg = _fraction_inside(mask, prompt.negatives)
        consistency = round(pos - CONTAMINATION_WEIGHT * neg, 4)
        scored.append(((consistency, iou), (mask, consistency)))
        if pos >= 0.5 and neg <= MAX_CONTAMINATION:
            eligible.append((iou, (mask, consistency)))
    if eligible:
        return max(eligible, key=lambda e: e[0])[1]
    if scored:
        # Ineligible: one full unit below, so it reads as "not found" at any
        # threshold in 0..1 while still ranking the least-bad candidate.
        best = max(scored, key=lambda e: e[0])[1]
        return best[0], round(best[1] - 1.0, 4)
    return None


CONTAMINATION_WEIGHT = 2.0
MAX_CONTAMINATION = 0.25


# ---------------------------------------------------------------------------
# Resolving masks into one label map
# ---------------------------------------------------------------------------
@dataclass
class PartMasks:
    """Per-part HD masks after overlap resolution, plus what was not found."""

    names: list[str]
    labels: np.ndarray          # HxW int16, -1 = no part, else index into names
    found: dict[str, bool]
    scores: dict[str, float]
    reasons: dict[str, str]

    def mask(self, name: str) -> np.ndarray:
        return self.labels == self.names.index(name)

    def image(self, name: str) -> Image.Image:
        return Image.fromarray((self.mask(name) * 255).astype(np.uint8), "L")


def resolve(parts: Sequence[Part], prompts: Sequence[Prompt],
            results: Sequence[tuple[np.ndarray, float] | None],
            alpha: np.ndarray, min_score: float = 0.0,
            min_pixels: int = 1) -> PartMasks:
    """Priority-resolve raw masks into one label map, opaque pixels only.

    A part whose prompt had nothing to say (no joints above threshold, no
    polygon this frame - a shield hidden behind the body) or whose mask came
    back empty or below `min_score` is reported absent, never invented.
    """
    names = [p.name for p in parts]
    labels = np.full(alpha.shape, -1, dtype=np.int16)
    found: dict[str, bool] = {}
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    opaque = alpha > 0
    for index, (part, prompt, result) in enumerate(zip(parts, prompts, results)):
        if not prompt.usable:
            found[part.name], reasons[part.name] = False, "no prompt this frame"
            continue
        if result is None:
            found[part.name], reasons[part.name] = False, "segmenter returned nothing"
            continue
        mask, score = result
        scores[part.name] = round(float(score), 4)
        if score < min_score:
            found[part.name] = False
            reasons[part.name] = f"score {score:.3f} below {min_score}"
            continue
        free = mask & opaque & (labels < 0)
        if int(free.sum()) < min_pixels:
            found[part.name] = False
            reasons[part.name] = "mask empty after priority"
            continue
        labels[free] = index
        found[part.name] = True
    return PartMasks(names, labels, found, scores, reasons)


def downscale_labels(labels: np.ndarray, size: tuple[int, int],
                     box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Label map -> `size` (w, h) by coverage: each output pixel takes the part
    covering most of it; the lower index wins a tie. -1 where no part covers."""
    if box is not None:
        x0, y0, x1, y1 = box
        labels = labels[y0:y1, x0:x1]
    count = int(labels.max()) + 1 if labels.size else 0
    if count <= 0:
        return np.full((size[1], size[0]), -1, dtype=np.int16)
    best = np.full((size[1], size[0]), -1, dtype=np.int16)
    best_cov = np.zeros((size[1], size[0]), dtype=np.float32)
    for index in range(count):
        mask = Image.fromarray(((labels == index) * 255).astype(np.uint8), "L")
        cov = np.asarray(mask.resize(size, Image.BOX), dtype=np.float32) / 255.0
        take = (cov > 0) & (cov > best_cov)
        best[take] = index
        best_cov[take] = cov[take]
    return best


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) 0..255 sRGB -> (N, 3) Oklab. Björn Ottosson's reference transform."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[:, 0], lin[:, 1], lin[:, 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ], axis=1)


def nearest_colours(pixels: np.ndarray, palette: Sequence[RGB]) -> np.ndarray:
    """Each (N, 3) pixel -> the nearest palette member in Oklab. (N, 3) uint8."""
    if not len(pixels):
        return np.zeros((0, 3), dtype=np.uint8)
    pal = np.asarray(palette, dtype=np.uint8)
    d = srgb_to_oklab(pixels)[:, None, :] - srgb_to_oklab(pal)[None, :, :]
    return pal[np.argmin((d * d).sum(axis=2), axis=1)]


def colours_in(frame: Image.Image, mask: np.ndarray) -> list[RGB]:
    """Sorted distinct opaque colours of `frame` under a boolean mask."""
    arr = np.asarray(frame.convert("RGBA"))
    sel = arr[mask & (arr[:, :, 3] > 0)][:, :3]
    return sorted({tuple(int(v) for v in px) for px in sel})


def extract_palettes(frame: Image.Image, small_labels: np.ndarray,
                     names: Sequence[str]) -> dict[str, list[RGB]]:
    """Part -> the colours the finished (snapped) frame shows inside that part."""
    return {name: colours_in(frame, small_labels == i) for i, name in enumerate(names)}


def lock(frame: Image.Image, small_labels: np.ndarray, names: Sequence[str],
         palettes: dict[str, Sequence[RGB]]) -> tuple[Image.Image, dict[str, dict[str, Any]]]:
    """Snap every opaque pixel of each part to that part's palette.

    Returns the frame and, per part: pixel count, colours after, and `drift` -
    the pixels that were NOT in the part's palette before the snap, which is
    the number the gate reads. A part with no palette (not found on the base
    sprite) is left as it is and says so.
    """
    arr = np.asarray(frame.convert("RGBA")).copy()
    opaque = arr[:, :, 3] > 0
    report: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        sel = (small_labels == index) & opaque
        pixels = int(sel.sum())
        palette = list(palettes.get(name) or [])
        if not pixels:
            report[name] = {"pixels": 0, "drift": 0, "colours": [], "locked": False}
            continue
        if not palette:
            report[name] = {"pixels": pixels, "drift": None,
                            "colours": [list(c) for c in colours_in(frame, sel)],
                            "locked": False, "reason": "no palette for this part"}
            continue
        before = arr[sel][:, :3]
        members = {tuple(c) for c in palette}
        drift = int(sum(tuple(int(v) for v in px) not in members for px in before))
        arr[sel, :3] = nearest_colours(before, palette)
        after = Image.fromarray(arr, "RGBA")
        report[name] = {"pixels": pixels, "drift": drift,
                        "colours": [list(c) for c in colours_in(after, sel)],
                        "locked": True}
    return Image.fromarray(arr, "RGBA"), report


def overlay(rgb: Image.Image, labels: np.ndarray, names: Sequence[str],
            alpha: float = 0.55) -> Image.Image:
    """The label map tinted over the frame, for the report and the review page."""
    base = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    out = base.copy()
    for index in range(len(names)):
        colour = np.asarray(PART_COLOURS[index % len(PART_COLOURS)], dtype=np.float32)
        sel = labels == index
        out[sel] = base[sel] * (1 - alpha) + colour * alpha
    return Image.fromarray(out.round().clip(0, 255).astype(np.uint8), "RGB")


PART_COLOURS: tuple[RGB, ...] = (
    (230, 60, 60), (60, 120, 230), (240, 200, 40), (60, 200, 90),
    (200, 80, 220), (255, 140, 0), (0, 200, 200), (150, 150, 150),
)
