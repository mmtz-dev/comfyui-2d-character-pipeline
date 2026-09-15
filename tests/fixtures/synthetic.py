"""Draw the pipeline's test inputs with Pillow, so no fixture has a licence.

Three things get built here:

  `figure_frame`   one frame of a stick-and-blob person against a plain wall,
                   posed by an arm angle and a stride. Deliberately obeys the
                   art bible's filming rules (profile, plain background, full
                   body, feet visible) so the crop and pose code sees the shape
                   it was written for.
  `write_clip`     those frames encoded to a real video file with ffmpeg, so
                   `anim extract` runs the same ffmpeg path it runs in anger -
                   including the rotation metadata, which is the part most
                   likely to break on a phone clip.
  `write_strip`    a fake "generated" N x 1 strip for the post-chain: flat
                   colour blobs on white, no model required.

Nothing here is random. Same arguments in, same pixels out.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

WALL = (214, 210, 202)
SKIN = (198, 150, 116)
SHIRT = (58, 74, 120)
TROUSERS = (44, 46, 54)


def figure_frame(size: tuple[int, int] = (360, 640), *, arm: float = 0.0,
                 stride: float = 0.0, offset: int = 0) -> Image.Image:
    """One frame. `arm` is the swing angle in radians, `stride` the leg spread."""
    width, height = size
    img = Image.new("RGB", size, WALL)
    draw = ImageDraw.Draw(img)
    cx = width // 2 + offset
    ground = height - 40
    hip_y = ground - 210
    shoulder_y = hip_y - 150
    head_r = 34

    # legs
    for direction in (-1, 1):
        knee = (cx + direction * (14 + stride * 30), hip_y + 105)
        foot = (cx + direction * (18 + stride * 55), ground)
        draw.line([(cx, hip_y), knee], fill=TROUSERS, width=26)
        draw.line([knee, foot], fill=TROUSERS, width=22)
        draw.ellipse([foot[0] - 16, ground - 10, foot[0] + 16, ground + 6],
                     fill=TROUSERS)
    # torso
    draw.line([(cx, hip_y), (cx, shoulder_y)], fill=SHIRT, width=54)
    # arms - one swings, one hangs
    for direction, angle in ((1, arm), (-1, -arm * 0.35)):
        elbow = (cx + direction * 30 + int(70 * math.sin(angle)),
                 shoulder_y + 70 - int(50 * math.sin(angle)))
        hand = (elbow[0] + direction * 20 + int(70 * math.sin(angle)),
                elbow[1] + 60 - int(70 * math.sin(angle)))
        draw.line([(cx + direction * 22, shoulder_y + 8), elbow],
                  fill=SHIRT, width=22)
        draw.line([elbow, hand], fill=SKIN, width=18)
    # head
    draw.ellipse([cx - head_r, shoulder_y - 2 * head_r,
                  cx + head_r, shoulder_y], fill=SKIN)
    return img


def frame_series(count: int = 24, size: tuple[int, int] = (360, 640),
                 offset: int = 0) -> list[Image.Image]:
    """A hold, a swing, a hold - the take the art bible asks the owner to film.

    Frames 0..3 and the last three are still, so the picker's endpoints and its
    "reversal" minima have something real to find, and the middle is one clean
    arc so the deviation curve has a single unambiguous peak.
    """
    frames: list[Image.Image] = []
    hold = max(1, count // 6)
    for index in range(count):
        if index < hold:
            phase = 0.0
        elif index >= count - hold:
            phase = 1.0
        else:
            phase = (index - hold) / max(1, count - 2 * hold - 1)
        angle = math.sin(phase * math.pi) * 1.15
        frames.append(figure_frame(size, arm=angle, stride=phase * 0.6,
                                   offset=offset))
    return frames


def write_clip(dest: Path, frames: list[Image.Image], *, fps: int = 24,
               rotation: int | None = None) -> Path:
    """Encode frames to a real video file. Returns the path.

    `rotation` writes a display matrix, which is what a phone does when it films
    with the handset on its side - and what `anim extract` has to honour.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f"_{dest.stem}_frames"
    staging.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(staging / f"{index:04d}.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", str(fps), "-i", str(staging / "%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)],
                   check=True, capture_output=True, text=True)
    if rotation is not None:
        # The display matrix has to be stamped by a REMUX with
        # `-display_rotation` on the input. `-metadata:s:v rotate=N` is the
        # old spelling and ffmpeg 7 silently drops it (measured 2026-09-13),
        # which would make the rotation test pass by never testing anything.
        rotated = dest.with_name(f"{dest.stem}_rot{dest.suffix}")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-display_rotation", str(rotation), "-i", str(dest),
                        "-c", "copy", str(rotated)],
                       check=True, capture_output=True, text=True)
        rotated.replace(dest)
    for stale in staging.glob("*.png"):
        stale.unlink()
    staging.rmdir()
    return dest


def write_strip(dest: Path, cells: int = 4, cell: tuple[int, int] = (304, 320),
                background=(255, 255, 255)) -> Path:
    """A stand-in for what the GPU would return: N flat-shaded figures on white.

    Flat blocks of a handful of colours, because that is what the post-chain has
    to survive - it snaps to a palette, and a photographic gradient would only
    test the quantiser, not the chain.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (cell[0] * cells, cell[1]), background)
    draw = ImageDraw.Draw(img)
    for index in range(cells):
        x0 = index * cell[0]
        cx = x0 + cell[0] // 2
        ground = cell[1] - 20
        lean = (index - (cells - 1) / 2) * 18
        draw.rectangle([cx - 44, ground - 190, cx + 44, ground - 40],
                       fill=(60, 78, 128))
        draw.ellipse([cx - 40, ground - 270, cx + 40, ground - 190],
                     fill=(196, 152, 118))
        draw.rectangle([cx - 34, ground - 40, cx - 6, ground], fill=(40, 42, 52))
        draw.rectangle([cx + 6, ground - 40, cx + 34, ground], fill=(40, 42, 52))
        draw.line([(cx + 30, ground - 170), (cx + 30 + lean, ground - 240)],
                  fill=(150, 150, 158), width=16)
    img.save(dest)
    return dest
