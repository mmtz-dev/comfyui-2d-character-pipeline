"""A stand-in segmenter for the part palette lock feature's tests.

Masks by flat colour instead of by SAM 2.1, so the mask-priority, downscale,
palette-extraction and lock logic run without a 184 MB checkpoint. The drawn
strips (`synthetic.write_strip`) are flat-shaded, so "every pixel near this
colour" is an exact part mask - which is the point: the tests judge what
`post` does WITH a mask, not how good the mask is. The real segmenter has its
own opt-in smoke test.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from anim.partkit import Prompt

# synthetic.write_strip's colours
STRIP_COLOURS = {
    "torso": (60, 78, 128),
    "head": (196, 152, 118),
    "legs": (40, 42, 52),
    "sword": (150, 150, 158),
}


class FakeSegmenter:
    """`segment()` returns one flat-colour mask per prompt, keyed by part name.

    A prompt with nothing usable (no joints, no polygon) gets None, exactly as
    the SAM wrapper does, so the "reported, not invented" path is exercised.
    Records every call so a test can see the prompts `post` built.
    """

    def __init__(self, colours: dict[str, tuple[int, int, int]] | None = None,
                 tolerance: int = 12, score: float = 1.0) -> None:
        self.colours = dict(colours or STRIP_COLOURS)
        self.tolerance = tolerance
        self.score = score
        self.calls: list[list[Prompt]] = []

    def segment(self, rgb: np.ndarray,
                prompts: Sequence[Prompt]) -> list[tuple[np.ndarray, float] | None]:
        self.calls.append(list(prompts))
        arr = np.asarray(rgb, dtype=np.int16)[:, :, :3]
        out: list[tuple[np.ndarray, float] | None] = []
        for prompt in prompts:
            colour = self.colours.get(prompt.part)
            if not prompt.usable or colour is None:
                out.append(None)
                continue
            dist = np.abs(arr - np.asarray(colour, dtype=np.int16)).sum(axis=2)
            out.append((dist <= self.tolerance, self.score))
        return out
