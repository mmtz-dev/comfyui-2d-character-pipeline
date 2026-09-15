"""`anim strip`: the N x 1 strip, the never-square rule, and the reference."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from anim import config
from anim.stages import pick as pick_stage
from anim.stages import strip as strip_stage


def test_a_square_strip_is_refused():
    """Qwen-Image-Edit degrades on square aspects (upstream issue #243)."""
    with pytest.raises(ValueError, match="square"):
        strip_stage.check_strip_size(320, 320)


def test_a_strip_off_the_latent_grid_is_refused():
    with pytest.raises(ValueError, match="multiples of 16"):
        strip_stage.check_strip_size(1215, 320)


def test_the_default_geometry_is_legal():
    width = config.STRIP_CELL[0] * config.STRIP_CELLS
    strip_stage.check_strip_size(width, config.STRIP_CELL[1])
    assert (width, config.STRIP_CELL[1]) == (1216, 320)


def test_compose_lays_cells_left_to_right():
    cell = (304, 320)
    tiles = [Image.new("RGB", cell, colour)
             for colour in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]]
    sheet = strip_stage.compose(tiles, cell)
    assert sheet.size == (1216, 320)
    for index, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                                    (255, 255, 0)]):
        assert sheet.getpixel((index * 304 + 150, 160)) == colour


def test_strip_writes_the_sheet_the_reference_and_the_layout(posed_job,
                                                             reference_sprite):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    layout = strip_stage.strip(posed_job.name, reference=reference_sprite,
                               work_dir=posed_job.root.parent)

    sheet = posed_job.strip_dir / "skeleton_strip.png"
    assert sheet.exists()
    with Image.open(sheet) as img:
        assert img.size == (304 * 4, 320)
    assert layout["strip"]["cells"] == 4
    assert [c["rect"] for c in layout["cells"]] == [
        [0, 0, 304, 320], [304, 0, 304, 320],
        [608, 0, 304, 320], [912, 0, 304, 320]]
    assert json.loads(posed_job.layout_path.read_text()) == layout


def test_the_reference_is_upscaled_by_eight_with_nearest(posed_job,
                                                         reference_sprite):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    layout = strip_stage.strip(posed_job.name, reference=reference_sprite,
                               work_dir=posed_job.root.parent)
    assert layout["reference"]["upscale"] == 8
    with Image.open(posed_job.strip_dir / "reference_512.png") as img:
        assert img.size == (512, 512), "identity_pass's ImageScale expects 512"
        # Nearest upscale of a 64px sprite: every 8x8 block is one colour.
        assert img.getpixel((0, 0)) == img.getpixel((7, 7))


def test_the_reference_strip_shares_the_skeletons_cell_grid(posed_job,
                                                            reference_sprite):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    layout = strip_stage.strip(posed_job.name, reference=reference_sprite,
                               work_dir=posed_job.root.parent)
    assert layout["reference"]["strip"]["size"] == layout["strip"]["size"]
    assert layout["reference"]["strip"]["cell_upscale"] == 4, \
        "the biggest whole-number scale of 64px that fits a 304x320 cell"


def test_a_pick_with_no_skeleton_is_explained_not_crashed(posed_job,
                                                          reference_sprite):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    picks = json.loads(posed_job.picks_path.read_text())["picks"]
    (posed_job.pose_dir / f"skel_{picks[1] + 1:04d}.png").unlink()
    with pytest.raises(FileNotFoundError, match="no person was detected"):
        strip_stage.strip(posed_job.name, reference=reference_sprite,
                          work_dir=posed_job.root.parent)


def test_three_picks_make_a_three_cell_strip(posed_job, reference_sprite):
    pick_stage.pick(posed_job.name, n=3, work_dir=posed_job.root.parent)
    layout = strip_stage.strip(posed_job.name, reference=reference_sprite,
                               work_dir=posed_job.root.parent)
    assert layout["strip"]["size"] == [304 * 3, 320]


def test_strip_without_picks_says_to_run_pick(posed_job, reference_sprite):
    with pytest.raises(FileNotFoundError, match="anim pick"):
        strip_stage.strip(posed_job.name, reference=reference_sprite,
                          work_dir=posed_job.root.parent)


def test_a_missing_reference_sprite_points_at_anim_base(posed_job, tmp_path):
    pick_stage.pick(posed_job.name, work_dir=posed_job.root.parent)
    with pytest.raises(FileNotFoundError, match="anim base"):
        strip_stage.strip(posed_job.name, assets_dir=tmp_path / "empty",
                          work_dir=posed_job.root.parent)
