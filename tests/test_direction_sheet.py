"""`anim.tools.direction_sheet` - the camera direction sheet (#288).

Runs against the same tiny two-bone synthetic GLB `test_rigkit.py` builds for
the projection maths (`write_synthetic_glb`: a `foot` root and a `toes` child,
one clip called `Turn`). That rig's bones match no `rigkit.JointMap`
`detect_map` knows, so the test hands `build_sheet` a throwaway `JointMap`
that folds every OpenPose joint onto `foot`/`toes` - geometrically
meaningless, but enough to prove the sheet writes every cell, both channels,
and the mirror comparison without crashing, with no network and no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anim import config, rigkit
from anim.tools import direction_sheet
from anim.util import sha256_file, write_json_in_place
from test_rigkit import TWO_BONE_MAP, write_synthetic_glb

CELL = (64, 80)


@pytest.fixture
def two_bone_glb(tmp_path: Path) -> Path:
    """The synthetic GLB plus the `rig.json` sidecar `rig_provenance` needs -
    every rig here carries its licence, even a fixture (rigs/README.md)."""
    glb = write_synthetic_glb(tmp_path / "two_bones.glb")
    write_json_in_place(glb.parent / "rig.json", {
        "pack": "test fixture", "author": "test", "url": "http://example.invalid",
        "license": "CC0-1.0", "retrieved": "2026-09-20",
        "files": {glb.name: sha256_file(glb)},
    })
    return glb


def test_writes_every_cell_both_channels_and_the_mirror(two_bone_glb, tmp_path):
    out_dir = tmp_path / "sheet"
    index_path = direction_sheet.build_sheet(
        "testunit", "Turn", rig=two_bone_glb, cell=CELL, out_dir=out_dir,
        jmap=TWO_BONE_MAP)

    assert index_path == out_dir / "index.html"
    img_dir = out_dir / "img"

    for name in config.DIRECTIONS:
        skel = img_dir / f"{name}_skeleton.png"
        depth = img_dir / f"{name}_depth.png"
        assert skel.is_file(), f"missing skeleton for {name}"
        assert depth.is_file(), f"missing depth for {name}"

    mirror = img_dir / "mirror.png"
    assert mirror.is_file()

    # 16 cells x 2 channels + the mirror cell.
    pngs = list(img_dir.glob("*.png"))
    assert len(pngs) == len(config.DIRECTIONS) * 2 + 1

    html = index_path.read_text(encoding="utf-8")
    for name in config.DIRECTIONS:
        assert name in html, f"{name} not named in the sheet"
    assert "mirror.png" in html
    assert "testunit" in html and "Turn" in html
    for criterion in direction_sheet.PASS_CRITERIA:
        assert criterion in html


def test_the_bare_rig_gets_a_depth_placeholder_not_a_crash(two_bone_glb, tmp_path):
    """The two-bone rig has no mesh (`rigkit.Rig.skinned` is False), so every
    depth frame is the placeholder, not a real z-buffer - and the sheet says so."""
    out_dir = tmp_path / "sheet"
    direction_sheet.build_sheet("testunit", "Turn", rig=two_bone_glb, cell=CELL,
                               out_dir=out_dir, jmap=TWO_BONE_MAP)
    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "no depth: unskinned rig" in html


def test_rig_for_unit_needs_rig_or_a_known_unit():
    with pytest.raises(SystemExit):
        direction_sheet.rig_for_unit("no-such-unit", None)
    assert direction_sheet.rig_for_unit("KNIGHT", None) == \
        direction_sheet.DEFAULT_RIG_FOR_UNIT["knight"]
    explicit = Path("/tmp/whatever.glb")
    assert direction_sheet.rig_for_unit("knight", explicit) == explicit
