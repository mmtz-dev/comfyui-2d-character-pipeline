"""The pixel grid downscale feature: one art grid per post job.

A model asked for pixel art draws *fake* pixel art: every art pixel is a block
several full-resolution pixels wide, sitting on a lattice - the **art grid**,
whose block size is the **pitch** (CONTEXT.md). Recovering the sprite means
taking one colour per block, at the sprite's true size.

**Measured once per job, from every frame together.** Re-detecting the grid
per frame gives a different grid each frame - a different scale, so the figure
breathes. proper-pixel-art, tried first, was worse than that on the wanderer
walk (22 Wan 14B frames, 2026-10-11): its per-frame widths ran from 5 to 13.5,
and their median (6.3) was wrong too. What works is the period of the edges:
each frame's gradient profile along x and y, autocorrelated (which does not
care where the grid starts, so a figure that moves does not blur it), summed
over the whole job, and read at sub-pixel precision. On the wanderer that
gave 5.7-6.0 on every single frame and 5.78 for the job - the true block,
checked by eye on an 8x crop. (#350 had said ~7.5: proper-pixel-art's reading
of frame 1. Sampled at 7.5 the sprite fits the canvas but drops the eye, the
belt buckle and the coat seams.) Each frame's own period is still reported, and
frames spread wider than `config.GRID_SPREAD_TOLERANCE` are flagged.

**One offset per job.** `shared_box` is the union of every frame's figure and
`place` puts that box bottom-centre on the canvas, so a figure that moves
inside the clip still moves on the canvas and a still one stays put - unlike
the per-frame trim + recanvas of the `box` route.

**True size.** The sprite comes out as tall as the art grid says, not squeezed
to the unit's band on the scale ladder (squeezing is the blur this feature
removes). `anim post` warns when the two differ by more than
`config.GRID_HEIGHT_TOLERANCE` and refuses a figure the canvas cannot hold;
the fix is framing the figure smaller at generation. `fit_to_height` is the
opt-in escape hatch (`post --fit band`): a coarser grid sized to the band,
crisp but one colour per *group* of art pixels, so detail is lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from PIL import Image

from anim import config

# Pitches searched, in full-resolution pixels. A 64 px sprite from a 512 px or
# larger frame has blocks of 5+; under 4 the search finds half-pitch echoes
# (the wanderer's 5.65 read as 2.9 on six frames, 2026-10-11), not art pixels.
MIN_PITCH = 4.0
MAX_PITCH = 24.0
PITCH_STEP = 0.01
# Lags read when scoring a pitch: enough multiples to tell 5.75 from 6.
MAX_LAG = 96
# A job whose best pitch scores under this has no grid to speak of.
MIN_SCORE = 0.15
# A whole fraction of the best pitch scoring at least this share of it is the
# real grid, and the best was one of its multiples. A wrong fraction (half the
# true pitch) scores far lower: every other one of its lags falls between
# grid lines, where the autocorrelation is negative.
HARMONIC_SHARE = 0.6
MAX_HARMONIC = 8
# A non-integer pitch puts its block edges a whole pixel apart unevenly (7, 8,
# 7, 8 for 7.5), splitting each autocorrelation peak in two while every
# second multiple lands exactly. Blurring the autocorrelation by this sigma
# merges a split peak back into one, so 7.5 is not outscored by 15.
SMOOTH_SIGMA = 1.0
# Colours vote in bins this wide, so a model's colour noise inside one block
# still counts as one colour; the winner's pixels are then averaged.
VOTE_BIN = 8


@dataclass(frozen=True)
class ArtGrid:
    """The one grid a job is sampled on."""
    pitch: float
    origin: tuple[float, float]
    pitches: list[float] = field(default_factory=list)
    measured: int = 0

    @property
    def spread(self) -> float:
        """How far the frames disagreed: (max - min) / the job's pitch."""
        if len(self.pitches) < 2:
            return 0.0
        return (max(self.pitches) - min(self.pitches)) / self.pitch

    @property
    def flagged(self) -> bool:
        return self.spread > config.GRID_SPREAD_TOLERANCE

    def report(self) -> dict[str, Any]:
        return {"pitch": round(self.pitch, 3),
                "origin": [round(v, 3) for v in self.origin],
                "per_frame": [round(p, 3) for p in self.pitches],
                "measured": self.measured, "spread": round(self.spread, 4),
                "flagged": self.flagged}


def _grey(img: Image.Image) -> np.ndarray:
    """Luma, with transparent pixels flattened to black so a matted cell's
    background is flat."""
    rgba = np.asarray(img.convert("RGBA"), dtype=np.float64)
    grey = rgba[..., :3] @ np.array([0.299, 0.587, 0.114])
    return np.where(rgba[..., 3] >= 128, grey, 0.0)


def edge_profiles(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Edge energy at every column and row boundary: index i is the step
    between pixel i - 1 and pixel i, summed down the column (along the row)."""
    g = _grey(img)
    px = np.zeros(g.shape[1])
    py = np.zeros(g.shape[0])
    px[1:] = np.abs(np.diff(g, axis=1)).sum(axis=0)
    py[1:] = np.abs(np.diff(g, axis=0)).sum(axis=1)
    return px, py


def _autocorr(profile: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of the profile's busy span (flat background
    either side cut off, so it does not dilute the period)."""
    busy = np.flatnonzero(profile > 0.01 * (profile.max() or 1))
    out = np.zeros(MAX_LAG + 1)
    if len(busy) < 4:
        return out
    signal = profile[busy[0]:busy[-1] + 1]
    signal = signal - signal.mean()
    n = signal.size
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    spectrum = np.fft.rfft(signal, size)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    if corr[0] <= 0:
        return out
    corr = corr / corr[0]
    out[:min(n, MAX_LAG + 1)] = corr[:MAX_LAG + 1]
    return _smooth(out)


def _smooth(values: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Gaussian blur, edges held."""
    radius = int(math.ceil(3 * sigma))
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, taps / taps.sum(), mode="valid")


def _score(corr: np.ndarray, pitch: float) -> float:
    """How strongly the edges repeat every `pitch` pixels: the mean
    autocorrelation at its multiples, read between whole lags."""
    lags = np.arange(1, int(MAX_LAG / pitch) + 1) * pitch
    return float(np.interp(lags, np.arange(len(corr)), corr).mean()) if len(lags) else 0.0


def _period(corr: np.ndarray) -> tuple[float, float]:
    """(pitch, score): the best-scoring pitch, divided down to the smallest
    whole fraction of it (1/2 ... 1/8) that scores nearly as well - every
    multiple of the true pitch scores well too, since its lags are a subset of
    the true pitch's, so the raw best is often 3x or 4x the grid."""
    pitches = np.arange(MIN_PITCH, MAX_PITCH, PITCH_STEP)
    scores = np.array([_score(corr, p) for p in pitches])
    best, score = float(pitches[scores.argmax()]), float(scores.max())
    pitch = best
    for k in range(2, MAX_HARMONIC + 1):
        candidate = best / k
        if candidate < MIN_PITCH:
            break
        # Re-centre on the nearest local best: best / k is only as exact as
        # best was, and that error grows k times smaller, not zero.
        near = np.arange(max(MIN_PITCH, candidate - 0.1), candidate + 0.1, PITCH_STEP)
        near_scores = [_score(corr, q) for q in near]
        if max(near_scores) >= HARMONIC_SHARE * score:
            pitch = float(near[int(np.argmax(near_scores))])
    return pitch, score


def _phase(profile: np.ndarray, pitch: float) -> float:
    """Where the grid lines fall on a `pitch`-wide cycle: the circular mean of
    the boundary positions, weighted by their edge energy."""
    if not profile.any():
        return 0.0
    angles = np.arange(len(profile)) / pitch * 2 * math.pi
    mean = math.atan2(float((profile * np.sin(angles)).sum()),
                      float((profile * np.cos(angles)).sum()))
    return (mean / (2 * math.pi) * pitch) % pitch


def measure(images: Sequence[Image.Image]) -> ArtGrid | None:
    """One grid for every frame of a job, or None when the job has none.

    The pitch is the period of the edges summed over all frames; the position
    is where the summed edges fall on that period (the camera is fixed, so the
    art grid is too); `pitches` is each frame's own period, for the report."""
    if not images:
        return None
    profiles = [edge_profiles(img) for img in images]
    corrs = [_autocorr(px) + _autocorr(py) for px, py in profiles]
    job_corr = np.sum(corrs, axis=0) / (2 * len(corrs))
    pitch, score = _period(job_corr)
    if score < MIN_SCORE:
        return None
    total_x = np.sum([px for px, _ in profiles], axis=0)
    total_y = np.sum([py for _, py in profiles], axis=0)
    return ArtGrid(pitch=pitch, origin=(_phase(total_x, pitch), _phase(total_y, pitch)),
                   pitches=[_period(c / 2)[0] for c in corrs], measured=len(images))


def fit_to_height(grid: ArtGrid, box: tuple[int, int, int, int],
                  height: int) -> ArtGrid:
    """`post --fit band`: a coarser grid on which the figure that spans
    `box` (in cells of `grid`) is exactly `height` cells tall, starting at the
    figure's top edge. Every cell still gets one colour - but now one colour
    for a group of art pixels, so detail is dropped. Opt-in only."""
    true_height = box[3] - box[1]
    pitch = grid.pitch * true_height / height
    top = grid.origin[1] + _first_index(grid.origin[1], grid.pitch) * grid.pitch \
        + box[1] * grid.pitch
    left = grid.origin[0] + _first_index(grid.origin[0], grid.pitch) * grid.pitch \
        + box[0] * grid.pitch
    return ArtGrid(pitch=pitch, origin=(left % pitch, top % pitch),
                   pitches=list(grid.pitches), measured=grid.measured)


def _first_index(origin: float, pitch: float) -> int:
    """The grid index of the first (possibly partial) cell of an axis."""
    return math.floor(-origin / pitch)


def _cell_spans(length: int, pitch: float, origin: float) -> list[tuple[int, int]]:
    """The central half of every grid cell along one axis, clipped to the
    image: [a, b) pixel spans, empty when the centre falls outside."""
    first = _first_index(origin, pitch)
    last = math.floor((length - 1 - origin) / pitch)
    spans = []
    for index in range(first, last + 1):
        start = origin + index * pitch
        a = max(0, int(math.floor(start + pitch * 0.25)))
        b = min(length, max(a + 1, int(math.ceil(start + pitch * 0.75))))
        spans.append((a, b) if a < length and b > 0 and a < b else (0, 0))
    return spans


def sample(img: Image.Image, grid: ArtGrid) -> Image.Image:
    """One colour per art pixel: the dominant colour in the central half of
    each grid cell (edges, where a block blends into its neighbour, never
    vote), opaque when most of that centre is. RGBA in, RGBA out."""
    rgba = np.asarray(img.convert("RGBA"), dtype=np.int32)
    xs = _cell_spans(rgba.shape[1], grid.pitch, grid.origin[0])
    ys = _cell_spans(rgba.shape[0], grid.pitch, grid.origin[1])
    out = np.zeros((len(ys), len(xs), 4), dtype=np.uint8)
    for r, (y0, y1) in enumerate(ys):
        for c, (x0, x1) in enumerate(xs):
            block = rgba[y0:y1, x0:x1].reshape(-1, 4)
            if not len(block):
                continue
            opaque = block[block[:, 3] >= 128]
            if len(opaque) * 2 <= len(block):
                continue
            keys = opaque[:, :3] // VOTE_BIN
            values, inverse, counts = np.unique(keys, axis=0, return_inverse=True,
                                                return_counts=True)
            winner = opaque[inverse.reshape(-1) == counts.argmax(), :3]
            out[r, c, :3] = np.round(winner.mean(axis=0)).astype(np.uint8)
            out[r, c, 3] = 255
    return Image.fromarray(out, "RGBA")


def shared_box(frames: Sequence[Image.Image]) -> tuple[int, int, int, int]:
    """The union of every frame's opaque box: (left, top, right, bottom)."""
    union = np.zeros(np.asarray(frames[0]).shape[:2], dtype=bool)
    for frame in frames:
        union |= np.asarray(frame.convert("RGBA"))[..., 3] > 0
    if not union.any():
        raise ValueError("every frame is empty after sampling - nothing to place")
    rows = np.where(union.any(axis=1))[0]
    cols = np.where(union.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def place(frame: Image.Image, box: tuple[int, int, int, int],
          canvas: tuple[int, int] = config.CANVAS) -> Image.Image:
    """`frame` cropped to the job's shared box and put bottom-centre: one
    offset for every frame of the job."""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if width > canvas[0] or height > canvas[1]:
        raise ValueError(f"the figure is {width}x{height} art pixels and does not "
                         f"fit the {canvas[0]}x{canvas[1]} canvas - frame it "
                         f"smaller at generation, or post with --downscale box")
    out = Image.new("RGBA", canvas, (0, 0, 0, 0))
    out.paste(frame.crop(box), ((canvas[0] - width) // 2, canvas[1] - height))
    return out
