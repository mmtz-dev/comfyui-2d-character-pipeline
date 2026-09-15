"""The deterministic image rules: palettes, alpha, the 64x64 canvas contract."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from anim import config, imaging


def test_a_gpl_palette_reads_back_in_file_order(tmp_path):
    path = tmp_path / "p.gpl"
    path.write_text("GIMP Palette\nName: T\nColumns: 4\n# comment\n"
                    "  8  22  17\t#081611\n 15  30  29\t#0f1e1d\n",
                    encoding="utf-8")
    assert imaging.read_palette(path) == [(8, 22, 17), (15, 30, 29)]


def test_a_one_row_png_palette_reads_back(tmp_path):
    path = tmp_path / "p.png"
    Image.fromarray(np.array([[(10, 20, 30), (40, 50, 60)]], dtype=np.uint8),
                    "RGB").save(path)
    assert imaging.read_palette(path) == [(10, 20, 30), (40, 50, 60)]


def test_a_duplicate_colour_appears_once(tmp_path):
    path = tmp_path / "p.gpl"
    path.write_text("GIMP Palette\n 1 1 1\n 1 1 1\n 2 2 2\n", encoding="utf-8")
    assert imaging.read_palette(path) == [(1, 1, 1), (2, 2, 2)]


def test_a_missing_palette_says_where_the_candidates_are(tmp_path):
    with pytest.raises(FileNotFoundError, match="palette_candidates"):
        imaging.read_palette(tmp_path / "nope.gpl")


def test_the_chroma_key_is_never_a_palette_colour():
    palette = [(255, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    assert imaging.chroma_key_for(palette) not in palette


def test_binarise_alpha_zeroes_the_rgb_under_transparent_pixels():
    arr = np.zeros((2, 2, 4), dtype=np.uint8)
    arr[0, 0] = (200, 10, 10, 255)
    arr[0, 1] = (200, 10, 10, 40)   # nearly transparent -> dropped
    arr[1, 0] = (200, 10, 10, 200)  # nearly opaque -> kept
    out = np.asarray(imaging.binarise_alpha(Image.fromarray(arr, "RGBA")))
    assert set(np.unique(out[:, :, 3])) <= set(config.ALPHA_VALUES)
    assert tuple(out[0, 1]) == (0, 0, 0, 0)
    assert tuple(out[1, 0]) == (200, 10, 10, 255)


def test_recanvas_puts_the_feet_on_the_last_row():
    sprite = Image.new("RGBA", (10, 20), (255, 0, 0, 255))
    out = imaging.recanvas(sprite, (64, 64))
    assert out.size == (64, 64)
    arr = np.asarray(out)
    assert arr[63, 32, 3] == 255, "bottom row must be opaque - feet flush"
    assert arr[0, 32, 3] == 0, "top must be empty"
    # horizontally centred: the same margin either side
    columns = np.flatnonzero(arr[:, :, 3].any(axis=0))
    assert columns[0] == 64 - 1 - columns[-1]


def test_scale_to_height_keeps_the_aspect_and_invents_no_colour():
    sprite = Image.new("RGBA", (40, 80), (10, 20, 30, 255))
    out = imaging.scale_to_height(sprite, 53)
    assert out.height == 53
    assert out.width == round(40 * 53 / 80)
    assert imaging.opaque_colours(out) == {(10, 20, 30)}


def test_largest_integer_scale_never_overflows_the_cell():
    sprite = Image.new("RGBA", (64, 64))
    scale = imaging.largest_integer_scale(sprite, (304, 320))
    assert scale == 4
    assert 64 * scale <= 304 and 64 * scale <= 320


def test_verify_frame_is_silent_on_a_good_frame(palette):
    colours = imaging.read_palette(palette)
    img = imaging.recanvas(Image.new("RGBA", (10, 20), (*colours[3], 255)))
    assert imaging.verify_frame(img, colours) == []


def test_verify_frame_names_every_way_a_frame_can_be_wrong(palette):
    colours = imaging.read_palette(palette)
    stray = (1, 2, 3)
    assert stray not in colours
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[0, 0] = (*stray, 128)  # off-palette AND a partial alpha
    problems = imaging.verify_frame(Image.fromarray(arr, "RGBA"), colours)
    assert any("dimensions" in p for p in problems)
    assert any("alpha" in p for p in problems)
    assert any("outside the palette" in p for p in problems)
