"""`anim post`: the deterministic palette-locking chain.

Fed a Pillow-drawn strip, so no GPU and no model download are involved. The
matte is the `chroma` keyer for the same reason - `rembg` would want to fetch
isnet-anime the first time it ran.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from anim import config, imaging
from anim.stages import pick as pick_stage
from anim.stages import post as post_stage
from anim.stages import strip as strip_stage


@pytest.fixture
def rendered_job(posed_job, reference_sprite, strip_png):
    """A job as it is the moment `render` finishes - with a fake strip."""
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    strip_stage.strip(posed_job.name, reference=reference_sprite,
                      work_dir=posed_job.root.parent)
    posed_job.render_dir.mkdir(parents=True, exist_ok=True)
    (posed_job.render_dir / "identity_pass.png").write_bytes(strip_png.read_bytes())
    posed_job.update_manifest(render={"seed": 4242, "passes": ["pose_pass",
                                                               "identity_pass"],
                                      "final": "identity_pass.png"})
    return posed_job


# ---------------------------------------------------------------------------
# Matting
# ---------------------------------------------------------------------------
def test_the_chroma_keyer_cuts_a_flat_background(strip_png):
    with Image.open(strip_png) as img:
        cut = post_stage.matte_chroma(img)
    arr = np.asarray(cut)
    assert arr[2, 2, 3] == 0, "a corner of a plain background must be cut"
    assert arr[:, :, 3].max() == 255, "the figure must survive"
    assert set(np.unique(arr[:, :, 3])) <= set(config.ALPHA_VALUES)


def test_the_keyer_samples_the_border_rather_than_assuming_white(tmp_path):
    from fixtures import synthetic

    grey = synthetic.write_strip(tmp_path / "grey.png", cells=2,
                                 background=(180, 180, 180))
    with Image.open(grey) as img:
        cut = post_stage.matte_chroma(img)
    assert np.asarray(cut)[2, 2, 3] == 0


def test_matte_none_only_binarises_what_is_already_there():
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[1, 1] = (10, 20, 30, 100)
    out = np.asarray(post_stage.matte_keep(Image.fromarray(arr, "RGBA")))
    assert set(np.unique(out[:, :, 3])) <= set(config.ALPHA_VALUES)


# ---------------------------------------------------------------------------
# The per-cell chain
# ---------------------------------------------------------------------------
def test_a_cell_comes_out_as_a_verified_64x64_frame(strip_png, palette, tmp_path):
    colours = imaging.read_palette(palette)
    palette_file = imaging.palette_png(colours, tmp_path / "palette.png")
    with Image.open(strip_png) as img:
        cut = post_stage.matte_chroma(img)
    cell = cut.crop((0, 0, 304, 320))
    frame, steps = post_stage.process_cell(
        cell, target_height=53, palette_png=palette_file,
        chroma=imaging.chroma_key_for(colours))

    assert imaging.verify_frame(frame, colours) == []
    assert frame.size == (64, 64)
    assert imaging.alpha_is_binary(frame)
    assert imaging.opaque_colours(frame) <= set(colours)
    assert any("recanvas" in s for s in steps)


def test_the_scale_ladder_decides_how_tall_the_sprite_is(strip_png, palette,
                                                         tmp_path):
    """A hound must not come out the same size as the boss (art bible S3)."""
    colours = imaging.read_palette(palette)
    palette_file = imaging.palette_png(colours, tmp_path / "palette.png")
    with Image.open(strip_png) as img:
        cell = post_stage.matte_chroma(img).crop((0, 0, 304, 320))
    heights = {}
    for target in (37, 63):
        frame, _ = post_stage.process_cell(
            cell, target_height=target, palette_png=palette_file,
            chroma=imaging.chroma_key_for(colours))
        rows = np.flatnonzero(np.asarray(frame)[:, :, 3].any(axis=1))
        heights[target] = rows[-1] - rows[0] + 1
    assert heights[37] < heights[63]
    assert heights[63] <= 64


def test_the_feet_land_on_the_last_row(strip_png, palette, tmp_path):
    colours = imaging.read_palette(palette)
    palette_file = imaging.palette_png(colours, tmp_path / "palette.png")
    with Image.open(strip_png) as img:
        cell = post_stage.matte_chroma(img).crop((0, 0, 304, 320))
    frame, _ = post_stage.process_cell(cell, target_height=53,
                                       palette_png=palette_file,
                                       chroma=imaging.chroma_key_for(colours))
    assert np.asarray(frame)[63, :, 3].any(), "bottom-center pivot: feet on row 63"


def test_a_fully_transparent_cell_is_an_error_not_an_empty_png(palette, tmp_path):
    colours = imaging.read_palette(palette)
    palette_file = imaging.palette_png(colours, tmp_path / "palette.png")
    blank = Image.new("RGBA", (304, 320), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="fully transparent"):
        post_stage.process_cell(blank, target_height=53,
                                palette_png=palette_file,
                                chroma=imaging.chroma_key_for(colours))


def test_grid_repair_is_rejected_when_it_collapses_the_sprite():
    """Measured 2026-09-13: pixelfixer reduced a 51x54 image to 9x10.

    It reconstructs at one pixel per detected block, so on an image already at
    native pixel scale it destroys the sprite. The guard is what makes running
    it safe at all.
    """
    from fixtures import synthetic

    tiny = synthetic.figure_frame((51, 54))
    out, note = post_stage.grid_repair(tiny, target_height=53)
    assert out.size == tiny.size
    assert "rejected" in note or "skipped" in note


def test_grid_repair_is_skippable():
    from fixtures import synthetic

    cell = synthetic.figure_frame((304, 320))
    out, note = post_stage.grid_repair(cell, target_height=53)
    assert out.size[0] > 0 and note


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------
def test_post_writes_a_frame_per_cell_and_a_report(rendered_job, palette):
    report = post_stage.post(rendered_job.name, palette=palette, matte="chroma",
                             work_dir=rendered_job.root.parent)
    assert len(report["frames"]) == 4
    files = sorted(p.name for p in rendered_job.post_dir.glob("knight_strike_*.png"))
    assert files == [f"knight_strike_{i}.png" for i in range(4)]
    assert all(record["problems"] == [] for record in report["frames"])
    assert report["matte"] == "chroma"
    assert report["target_height"] == config.height_for_unit("knight")
    assert json.loads(rendered_job.report_path.read_text()) == report


def test_every_published_frame_passes_verification(rendered_job, palette):
    post_stage.post(rendered_job.name, palette=palette, matte="chroma",
                    work_dir=rendered_job.root.parent)
    colours = imaging.read_palette(palette)
    for path in sorted(rendered_job.post_dir.glob("knight_strike_*.png")):
        with Image.open(path) as img:
            assert imaging.verify_frame(img.convert("RGBA"), colours) == []


def test_post_is_deterministic(rendered_job, palette):
    post_stage.post(rendered_job.name, palette=palette, matte="chroma",
                    work_dir=rendered_job.root.parent)
    first = (rendered_job.post_dir / "knight_strike_0.png").read_bytes()
    post_stage.post(rendered_job.name, palette=palette, matte="chroma",
                    work_dir=rendered_job.root.parent)
    assert (rendered_job.post_dir / "knight_strike_0.png").read_bytes() == first


def test_stale_frames_from_a_longer_take_are_cleared(rendered_job, palette):
    (rendered_job.post_dir).mkdir(parents=True, exist_ok=True)
    (rendered_job.post_dir / "knight_strike_5.png").write_bytes(b"stale")
    post_stage.post(rendered_job.name, palette=palette, matte="chroma",
                    work_dir=rendered_job.root.parent)
    assert not (rendered_job.post_dir / "knight_strike_5.png").exists()


def test_the_report_names_the_palette_and_the_chroma_key(rendered_job, palette):
    report = post_stage.post(rendered_job.name, palette=palette, matte="chroma",
                             work_dir=rendered_job.root.parent)
    colours = imaging.read_palette(palette)
    assert report["palette"]["colours"] == len(colours)
    assert tuple(report["palette"]["chroma_key"]) not in set(colours)
    assert "ImageMagick 7" in report["tools"]["magick"]


def test_post_without_a_render_says_to_run_render(posed_job, reference_sprite,
                                                  palette):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    strip_stage.strip(posed_job.name, reference=reference_sprite,
                      work_dir=posed_job.root.parent)
    with pytest.raises(FileNotFoundError, match="anim render"):
        post_stage.post(posed_job.name, palette=palette, matte="chroma",
                        work_dir=posed_job.root.parent)


def test_an_unknown_matte_is_refused(rendered_job, palette):
    with pytest.raises(ValueError, match="unknown matte"):
        post_stage.post(rendered_job.name, palette=palette, matte="magic",
                        work_dir=rendered_job.root.parent)
