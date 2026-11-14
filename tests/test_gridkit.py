"""The pixel grid downscale feature: one art grid per post job.

Every test here draws its own "fake pixel art": a small native sprite blown up
by a known, deliberately non-integer pitch at a known offset, with colour noise
like a model's. So the right answer is always known exactly.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from anim import config, gridkit

RNG_SEED = 350
COLOURS = np.array([(40, 52, 90), (196, 152, 118), (120, 30, 40), (230, 220, 120)],
                   dtype=np.int32)


def native_sprite(cols: int, rows: int, *, seed: int = RNG_SEED,
                  margin: int = 2) -> np.ndarray:
    """A cols x rows RGBA sprite: random palette blocks inside a clear margin."""
    rng = np.random.default_rng(seed)
    out = np.zeros((rows, cols, 4), dtype=np.uint8)
    body = COLOURS[rng.integers(0, len(COLOURS), size=(rows - 2 * margin,
                                                      cols - 2 * margin))]
    out[margin:rows - margin, margin:cols - margin, :3] = body
    out[margin:rows - margin, margin:cols - margin, 3] = 255
    return out


def fake_pixel_art(native: np.ndarray, *, pitch: float, origin: tuple[float, float],
                   size: tuple[int, int], noise: int = 6, seed: int = RNG_SEED
                   ) -> Image.Image:
    """`native` drawn the way a model draws pixel art: every art pixel a
    pitch-wide block, the grid starting at `origin`, plus colour noise."""
    width, height = size
    xs = np.floor((np.arange(width) - origin[0]) / pitch).astype(int)
    ys = np.floor((np.arange(height) - origin[1]) / pitch).astype(int)
    inside = ((xs[None, :] >= 0) & (xs[None, :] < native.shape[1])
              & (ys[:, None] >= 0) & (ys[:, None] < native.shape[0]))
    hd = np.zeros((height, width, 4), dtype=np.int32)
    yy = np.clip(ys, 0, native.shape[0] - 1)[:, None]
    xx = np.clip(xs, 0, native.shape[1] - 1)[None, :]
    hd[inside] = native[yy, xx][inside]
    rng = np.random.default_rng(seed + 1)
    hd[..., :3] += rng.integers(-noise, noise + 1, size=hd[..., :3].shape)
    return Image.fromarray(np.clip(hd, 0, 255).astype(np.uint8), "RGBA")


def fixed(pitch: float, origin: tuple[float, float]):
    """A job measurer that always finds the same grid."""
    return lambda _images: gridkit.ArtGrid(pitch=pitch, origin=origin,
                                           pitches=[pitch], measured=1)


def walk(pitch: float, origin: tuple[float, float], *, frames: int = 4,
         size: tuple[int, int] = (300, 420)) -> tuple[np.ndarray, list[Image.Image]]:
    """A figure that steps one art pixel right per frame, on one fixed grid."""
    native = native_sprite(24, 40)
    return native, [fake_pixel_art(native, pitch=pitch,
                                   origin=(origin[0] + i * pitch, origin[1]),
                                   size=size, seed=RNG_SEED + i)
                    for i in range(frames)]


# ---------------------------------------------------------------------------
# Measuring: once per job, from every frame together
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pitch", [5.75, 7.5, 9.0])
def test_the_job_pitch_is_found_to_a_fraction_of_a_pixel(pitch):
    _native, frames = walk(pitch, (2.0, 3.0))
    grid = gridkit.measure(frames)
    assert grid is not None
    assert grid.pitch == pytest.approx(pitch, abs=0.05)
    assert grid.measured == len(frames)
    assert not grid.flagged


@pytest.mark.parametrize("pitch,origin", [(7.5, (3.0, 5.0)), (5.75, (1.0, 4.0))])
def test_the_measured_grid_recovers_every_frame_exactly(pitch, origin):
    native, frames = walk(pitch, origin)
    grid = gridkit.measure(frames)
    for shift, frame in enumerate(frames):
        small = np.asarray(gridkit.sample(frame, grid))
        rows = np.where(small[..., 3].any(1))[0]
        cols = np.where(small[..., 3].any(0))[0]
        window = small[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        body = native[2:-2, 2:-2]
        assert window.shape[:2] == body.shape[:2], f"frame {shift}"
        assert np.abs(window[..., :3].astype(int) - body[..., :3].astype(int)).max() <= 8


def test_frames_that_disagree_on_the_pitch_are_flagged():
    native = native_sprite(24, 40)
    frames = [fake_pixel_art(native, pitch=p, origin=(0.0, 0.0), size=(300, 420))
              for p in (6.0, 6.0, 8.0)]
    grid = gridkit.measure(frames)
    assert grid.flagged
    assert max(grid.pitches) - min(grid.pitches) == pytest.approx(2.0, abs=0.1)


def test_a_job_with_no_grid_measures_none():
    flat = Image.new("RGBA", (200, 200), (120, 30, 40, 255))
    assert gridkit.measure([flat, flat]) is None
    assert gridkit.measure([]) is None


def test_fit_to_height_makes_the_figure_exactly_the_band():
    native, frames = walk(5.75, (1.0, 4.0), frames=1)
    grid = gridkit.measure(frames)
    small = gridkit.sample(frames[0], grid)
    box = gridkit.shared_box([small])
    assert box[3] - box[1] == 36                      # true size
    fitted = gridkit.fit_to_height(grid, box, 27)
    assert fitted.pitch == pytest.approx(grid.pitch * 36 / 27)
    coarse = gridkit.sample(frames[0], fitted)
    coarse_box = gridkit.shared_box([coarse])
    assert coarse_box[3] - coarse_box[1] == 27


# ---------------------------------------------------------------------------
# Sampling: one colour per art pixel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pitch,origin", [(7.5, (3.0, 5.0)), (6.0, (0.0, 0.0)),
                                          (8.25, (7.9, 1.2))])
def test_sampling_recovers_the_native_sprite(pitch, origin):
    native = native_sprite(20, 30)
    hd = fake_pixel_art(native, pitch=pitch, origin=origin, size=(240, 300))
    grid = gridkit.ArtGrid(pitch=pitch, origin=origin, pitches=[pitch], measured=1)
    small = np.asarray(gridkit.sample(hd, grid))
    # The sample covers the whole frame; the sprite sits where the grid put it.
    col0 = math.floor(-origin[0] / pitch) * -1
    row0 = math.floor(-origin[1] / pitch) * -1
    window = small[row0:row0 + 30, col0:col0 + 20]
    assert (window[..., 3] == native[..., 3]).all()
    opaque = native[..., 3] > 0
    assert np.abs(window[..., :3][opaque].astype(int)
                  - native[..., :3][opaque].astype(int)).max() <= 8


def test_a_transparent_art_pixel_stays_transparent():
    native = native_sprite(10, 10)
    hd = fake_pixel_art(native, pitch=7.0, origin=(0.0, 0.0), size=(70, 70))
    grid = gridkit.ArtGrid(pitch=7.0, origin=(0.0, 0.0), pitches=[7.0], measured=1)
    small = np.asarray(gridkit.sample(hd, grid))
    assert small.shape[:2] == (10, 10)
    assert (small[..., 3] == native[..., 3]).all()
    assert set(np.unique(small[..., 3])) <= {0, 255}


# ---------------------------------------------------------------------------
# Placing: one offset for the whole job
# ---------------------------------------------------------------------------
def test_every_frame_shares_one_offset_so_a_moving_figure_moves():
    a = np.zeros((40, 40, 4), dtype=np.uint8)
    b = np.zeros((40, 40, 4), dtype=np.uint8)
    a[10:30, 5:15] = 255     # figure at the left...
    b[10:30, 9:19] = 255     # ...four art pixels further right
    frames = [Image.fromarray(a, "RGBA"), Image.fromarray(b, "RGBA")]
    box = gridkit.shared_box(frames)
    assert box == (5, 10, 19, 30)
    placed = [np.asarray(gridkit.place(f, box, (64, 64))) for f in frames]
    lefts = [np.where(p[..., 3].any(0))[0][0] for p in placed]
    bottoms = [np.where(p[..., 3].any(1))[0][-1] for p in placed]
    assert lefts[1] - lefts[0] == 4          # the motion survives
    assert bottoms == [63, 63]               # feet on the bottom edge
    assert lefts[0] == (64 - 14) // 2        # the union box is centred


def test_a_figure_too_big_for_the_canvas_is_refused():
    big = np.zeros((80, 20, 4), dtype=np.uint8)
    big[:, :] = 255
    frame = Image.fromarray(big, "RGBA")
    with pytest.raises(ValueError, match="does not fit"):
        gridkit.place(frame, gridkit.shared_box([frame]), (64, 64))


# ---------------------------------------------------------------------------
# Through `anim post`
# ---------------------------------------------------------------------------
def test_post_downscale_grid_samples_every_frame_on_one_grid(clipped_job, palette):
    from anim.stages import post as post_stage

    picks = json.loads(clipped_job.picks_path.read_text())["picks"]
    with Image.open(clipped_job.clip_frames_dir / f"{picks[0] + 1:04d}.png") as first:
        size = first.size
    pitch = round(size[1] / 90, 2)
    native = native_sprite(22, 50, margin=0)
    for shift, frame_index in enumerate(picks):
        art = fake_pixel_art(native, pitch=pitch,
                             origin=(size[0] / 3 + shift * pitch, size[1] / 4),
                             size=size, seed=RNG_SEED + shift)
        flat = Image.new("RGB", size, (255, 255, 255))
        flat.paste(art, (0, 0), art)
        flat.save(clipped_job.clip_frames_dir / f"{frame_index + 1:04d}.png")

    report = post_stage.post(clipped_job.name, palette=palette, matte="chroma",
                             source="clip", downscale="grid",
                             grid_measure=fixed(pitch, (size[0] / 3 % pitch,
                                                        size[1] / 4 % pitch)),
                             work_dir=clipped_job.root.parent)
    assert report["downscale"] == "grid"
    assert report["grid"]["pitch"] == pytest.approx(pitch)
    assert report["grid"]["native_height"] == 50
    assert report["grid"]["target_height"] == config.height_for_unit("knight")
    lefts, heights = [], []
    for record in report["frames"]:
        assert not record["problems"], record
        with Image.open(clipped_job.post_dir / record["file"]) as frame:
            alpha = np.asarray(frame)[..., 3]
        cols = np.where(alpha.any(0))[0]
        rows = np.where(alpha.any(1))[0]
        lefts.append(cols[0])
        heights.append(rows[-1] - rows[0] + 1)
    assert heights == [50] * len(picks)                      # true size, every frame
    assert lefts == [lefts[0] + i for i in range(len(picks))]  # one shared offset


def test_post_grid_warns_when_the_true_height_misses_the_ladder(clipped_job, palette):
    from anim.stages import post as post_stage

    picks = json.loads(clipped_job.picks_path.read_text())["picks"]
    with Image.open(clipped_job.clip_frames_dir / f"{picks[0] + 1:04d}.png") as first:
        size = first.size
    pitch = round(size[1] / 60, 2)
    native = native_sprite(16, 30, margin=0)     # 30 tall against the knight's 53
    for frame_index in picks:
        art = fake_pixel_art(native, pitch=pitch, origin=(pitch, pitch), size=size)
        flat = Image.new("RGB", size, (255, 255, 255))
        flat.paste(art, (0, 0), art)
        flat.save(clipped_job.clip_frames_dir / f"{frame_index + 1:04d}.png")
    report = post_stage.post(clipped_job.name, palette=palette, matte="chroma",
                             source="clip", downscale="grid",
                             grid_measure=fixed(pitch, (0.0, 0.0)),
                             work_dir=clipped_job.root.parent)
    assert report["grid"]["native_height"] == 30
    assert "height" in " ".join(report["grid"]["warnings"])


def test_post_grid_falls_back_to_box_when_no_grid_is_found(clipped_job, palette):
    from anim.stages import post as post_stage

    report = post_stage.post(clipped_job.name, palette=palette, matte="chroma",
                             source="clip", downscale="grid",
                             grid_measure=lambda _images: None,
                             work_dir=clipped_job.root.parent)
    assert report["grid"]["fallback"] == "box"
    assert all(not r["problems"] for r in report["frames"])


def test_post_grid_refuses_the_part_lock_for_now(clipped_job, palette):
    from anim.stages import post as post_stage

    with pytest.raises(ValueError, match="--parts"):
        post_stage.post(clipped_job.name, palette=palette, matte="chroma",
                        source="clip", downscale="grid", parts=True,
                        work_dir=clipped_job.root.parent)


def test_post_grid_refuses_a_figure_too_tall_and_fit_band_fits_it(clipped_job, palette):
    from anim.stages import post as post_stage

    picks = json.loads(clipped_job.picks_path.read_text())["picks"]
    with Image.open(clipped_job.clip_frames_dir / f"{picks[0] + 1:04d}.png") as first:
        size = first.size
    pitch = round(size[1] / 90, 2)
    native = native_sprite(20, 80, margin=0)     # 80 art pixels: no 64 canvas holds it
    for frame_index in picks:
        art = fake_pixel_art(native, pitch=pitch, origin=(pitch, pitch), size=size)
        flat = Image.new("RGB", size, (255, 255, 255))
        flat.paste(art, (0, 0), art)
        flat.save(clipped_job.clip_frames_dir / f"{frame_index + 1:04d}.png")
    kwargs = dict(palette=palette, matte="chroma", source="clip", downscale="grid",
                  grid_measure=fixed(pitch, (0.0, 0.0)), work_dir=clipped_job.root.parent)
    with pytest.raises(ValueError, match="--fit band"):
        post_stage.post(clipped_job.name, **kwargs)
    report = post_stage.post(clipped_job.name, fit="band", **kwargs)
    assert report["grid"]["fit"] == "band"
    assert report["grid"]["true_height"] == 80
    assert report["grid"]["native_height"] == config.height_for_unit("knight")
    assert all(not r["problems"] for r in report["frames"])
