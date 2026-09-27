"""anim.partkit: prompts from the rig, mask resolution, palettes, the lock.

No SAM here - the segmenter is a Protocol and these tests hand in masks by
hand, because what is under test is what the pipeline DOES with a mask: where
the prompts land, which part wins an overlap, how a one-pixel sword survives
the downscale, and that a locked pixel ends up in its part's palette. The
real SAM 2.1 wrapper has one opt-in smoke test at the bottom.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anim import config, partkit, posekit
from anim.partkit import Part, Prompt

NAMES = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
         "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
         "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
         "left_eye", "right_ear", "left_ear"]


def frame_entry(props: bool = True) -> dict:
    kp = [[100, 20], [100, 60], [110, 62], [120, 100], [130, 140],
          [90, 62], [80, 100], [70, 140], [108, 150], [110, 210], [112, 270],
          [92, 150], [90, 210], [88, 270], [104, 16], [96, 16], [108, 20],
          [92, 20]]
    entry = {"index": 0, "keypoints": kp, "scores": [0.9] * 18}
    if props:
        entry["props"] = [
            {"kind": "sword", "polygons": [[[130, 140], [190, 100], [194, 106],
                                            [134, 146]]]},
            {"kind": "shield", "polygons": [[[60, 90], [80, 90], [80, 130],
                                             [60, 130]]]},
        ]
    return entry


TABLE = (Part("sword", prop="sword"), Part("shield", prop="shield"),
         Part("head", joints=("nose",), mids=(("neck", "nose", 1.5),)),
         Part("torso", joints=("neck",), mids=(("neck", "right_hip", 0.5),)))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def test_letterbox_is_the_one_transform_draw_uses():
    scale, off_x, off_y = posekit.letterbox((360, 420), (304, 320))
    assert scale == pytest.approx(min(304 / 360, 320 / 420))
    assert off_x == pytest.approx((304 - 360 * scale) / 2)
    assert off_y == pytest.approx((320 - 420 * scale) / 2)


def test_prompts_land_where_the_skeleton_was_drawn():
    prompts = partkit.prompts_for(TABLE, frame_entry(), NAMES, (200, 300),
                                  (400, 600))
    by = {p.part: p for p in prompts}
    # 200x300 into 400x600 is exactly x2, no offset
    assert by["head"].points[0] == (200.0, 40.0)
    # the extrapolated crown: neck (100,60) -> nose (100,20) at t=1.5 = (100,0)
    assert by["head"].points[1] == (200.0, 0.0)
    assert by["torso"].points == [(200.0, 120.0), (208.0, 210.0)]


def test_a_rigs_hidden_face_point_still_prompts_but_a_filmed_low_score_does_not():
    """The multi-view reference feature: a rig hides the face off-front
    (score 0.0, listed in `hidden`) for the ControlNet's sake, but the point
    is placed on the head and stays a prompt for SAM; a filmed payload's
    low-confidence nose (no `hidden`) is a guess and stays out as before."""
    entry = frame_entry()
    entry["scores"][NAMES.index("nose")] = 0.0
    filmed = {p.part: p for p in partkit.prompts_for(TABLE, entry, NAMES, (200, 300), (400, 600))}
    assert filmed["head"].points == [], "no nose, no crown"
    rig = dict(entry, hidden=["nose"])
    rigged = {p.part: p for p in partkit.prompts_for(TABLE, rig, NAMES, (200, 300), (400, 600))}
    assert rigged["head"].points == [(200.0, 40.0), (200.0, 0.0)]
    assert (200.0, 40.0) in rigged["torso"].negatives, "and it is the other parts' negative"


def test_a_prop_prompt_is_its_polygon_centroid_and_no_box_by_default():
    prompts = partkit.prompts_for(TABLE, frame_entry(), NAMES, (200, 300),
                                  (200, 300))
    by = {p.part: p for p in prompts}
    assert by["sword"].points == [(162.0, 123.0)]
    assert by["sword"].box is None
    boxed = partkit.prompts_for((Part("sword", prop="sword", box=True),),
                                frame_entry(), NAMES, (200, 300), (200, 300))
    x0, y0, x1, y1 = boxed[0].box
    assert x0 < 130 and y0 < 100 and x1 > 194 and y1 > 146


def test_every_other_parts_positives_are_my_negatives():
    prompts = partkit.prompts_for(TABLE, frame_entry(), NAMES, (200, 300),
                                  (200, 300))
    by = {p.part: p for p in prompts}
    assert set(by["sword"].negatives) == set(
        by["shield"].points + by["head"].points + by["torso"].points)
    assert by["sword"].points[0] not in by["sword"].negatives


def test_a_part_with_nothing_to_point_at_is_unusable_not_guessed():
    prompts = partkit.prompts_for(TABLE, frame_entry(props=False), NAMES,
                                  (200, 300), (200, 300))
    by = {p.part: p for p in prompts}
    assert not by["sword"].usable and not by["shield"].usable
    assert by["head"].usable
    low = frame_entry()
    low["scores"] = [0.1] * 18
    prompts = partkit.prompts_for(TABLE, low, NAMES, (200, 300), (200, 300))
    assert not {p.part: p for p in prompts}["head"].usable


# ---------------------------------------------------------------------------
# Choosing among SAM's candidates
# ---------------------------------------------------------------------------
def _mask(size=(20, 20), box=None):
    m = np.zeros(size, bool)
    if box:
        x0, y0, x1, y1 = box
        m[y0:y1, x0:x1] = True
    return m


def test_the_candidate_that_holds_the_positives_and_not_the_negatives_wins():
    prompt = Prompt("shield", points=[(5.0, 5.0)], negatives=[(15.0, 15.0)])
    sub = (_mask(box=(4, 4, 7, 7)), 0.3)           # holds the point, low IoU
    part = (_mask(box=(0, 0, 10, 10)), 0.9)        # holds the point, best IoU
    whole = (_mask(box=(0, 0, 20, 20)), 0.95)      # swallows the negative
    mask, score = partkit.choose_candidate([sub, part, whole], prompt)
    assert mask is part[0]
    assert score == 1.0


def test_one_occluded_point_in_four_is_allowed_but_more_is_not():
    negs = [(15.0, 15.0), (16.0, 16.0), (17.0, 17.0), (1.0, 19.0)]
    prompt = Prompt("shield", points=[(5.0, 5.0)], negatives=negs)
    ok = (_mask(box=(0, 0, 10, 20)), 0.8)          # holds (1,19): 1 of 4 inside
    bad = (_mask(box=(0, 0, 20, 20)), 0.99)        # holds all four
    mask, score = partkit.choose_candidate([ok, bad], prompt)
    assert mask is ok[0]
    assert score == pytest.approx(1 - 2 * 0.25)


def test_nothing_eligible_comes_back_below_zero_so_resolve_skips_it():
    prompt = Prompt("sword", points=[(5.0, 5.0), (15.0, 15.0)],
                    negatives=[(15.0, 15.0)])
    only = (_mask(box=(10, 10, 20, 20)), 0.9)      # misses one positive, holds the negative
    mask, score = partkit.choose_candidate([only], prompt)
    assert score == pytest.approx(0.5 - 2 * 1.0 - 1.0)
    parts = (Part("sword", prop="sword"),)
    resolved = partkit.resolve(parts, [prompt], [(mask, score)],
                               np.full((20, 20), 255, np.uint8), min_score=0.0)
    assert resolved.found == {"sword": False}
    assert "below" in resolved.reasons["sword"]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def test_an_overlap_goes_to_the_earlier_part_and_only_opaque_pixels_count():
    parts = (Part("sword", prop="sword"), Part("torso", joints=("neck",)))
    prompts = [Prompt("sword", points=[(1.0, 1.0)]), Prompt("torso", points=[(1.0, 1.0)])]
    sword = _mask(box=(0, 0, 10, 10))
    torso = _mask(box=(0, 0, 20, 20))
    alpha = np.full((20, 20), 255, np.uint8)
    alpha[:, 15:] = 0
    out = partkit.resolve(parts, prompts, [(sword, 1.0), (torso, 1.0)], alpha)
    assert out.mask("sword").sum() == 100
    assert out.mask("torso").sum() == 20 * 15 - 100
    assert not (out.labels[:, 15:] >= 0).any()
    assert out.found == {"sword": True, "torso": True}


def test_a_part_with_no_prompt_or_no_mask_is_reported_absent():
    parts = (Part("sword", prop="sword"), Part("shield", prop="shield"),
             Part("head", joints=("nose",)))
    prompts = [Prompt("sword"), Prompt("shield", points=[(1.0, 1.0)]),
               Prompt("head", points=[(1.0, 1.0)])]
    alpha = np.full((8, 8), 255, np.uint8)
    out = partkit.resolve(parts, prompts, [None, None, (_mask((8, 8)), 0.9)], alpha)
    assert out.found == {"sword": False, "shield": False, "head": False}
    assert out.reasons["sword"] == "no prompt this frame"
    assert out.reasons["shield"] == "segmenter returned nothing"
    assert out.reasons["head"] == "mask empty after priority"


# ---------------------------------------------------------------------------
# Downscale by coverage
# ---------------------------------------------------------------------------
def test_a_one_pixel_sword_survives_the_downscale():
    """Nearest sampling of a 6px-wide blade at 1/8 lands between columns and
    the sword vanishes; coverage keeps it."""
    labels = np.full((80, 80), -1, np.int16)
    labels[:, 33:39] = 0          # a 6px vertical blade, off the 8px grid
    labels[40:80, 0:80] = np.where(labels[40:80, 0:80] == 0, 0, 1)  # body below
    small = partkit.downscale_labels(labels, (10, 10))
    assert (small[:5] == 0).any(), "the blade must appear in the top half"
    assert (small[5:] == 1).any()


def test_the_bigger_cover_wins_and_a_tie_goes_to_the_lower_index():
    labels = np.full((8, 8), -1, np.int16)
    labels[:, :3] = 1             # 3 columns of part 1
    labels[:, 3:8] = 0            # 5 columns of part 0
    assert partkit.downscale_labels(labels, (1, 1))[0, 0] == 0
    labels[:, :4] = 1
    labels[:, 4:] = 0
    assert partkit.downscale_labels(labels, (1, 1))[0, 0] == 0
    assert partkit.downscale_labels(np.full((8, 8), -1, np.int16), (2, 2)).tolist() \
        == [[-1, -1], [-1, -1]]


def test_the_trim_box_crops_the_labels_like_the_pixels():
    labels = np.full((10, 10), -1, np.int16)
    labels[2:4, 2:4] = 0
    small = partkit.downscale_labels(labels, (2, 2), box=(2, 2, 4, 4))
    assert (small == 0).all()


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
def test_oklab_puts_white_at_one_and_black_at_zero():
    lab = partkit.srgb_to_oklab(np.array([[255, 255, 255], [0, 0, 0]]))
    assert lab[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert lab[1, 0] == pytest.approx(0.0, abs=1e-6)


def test_nearest_keeps_a_member_and_picks_the_closer_tone():
    palette = [(180, 180, 180), (60, 60, 60), (200, 40, 40)]
    pixels = np.array([[60, 60, 60], [200, 200, 200], [170, 50, 50]])
    out = [tuple(int(v) for v in c) for c in partkit.nearest_colours(pixels, palette)]
    assert out == [(60, 60, 60), (180, 180, 180), (200, 40, 40)]
    assert partkit.nearest_colours(np.zeros((0, 3)), palette).shape == (0, 3)


def test_lock_snaps_only_the_part_and_counts_the_drift():
    frame = Image.new("RGBA", (4, 2), (0, 0, 0, 0))
    px = frame.load()
    for x in range(4):
        px[x, 0] = (200, 40, 40, 255)      # part 0, off-palette red
        px[x, 1] = (10, 10, 200, 255)      # part 1, left alone
    labels = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], np.int16)
    palettes = {"shield": [(60, 60, 60), (220, 220, 220)]}
    out, report = partkit.lock(frame, labels, ["shield", "legs"], palettes)
    arr = np.asarray(out)
    assert {tuple(p[:3]) for p in arr[0]} <= {(60, 60, 60), (220, 220, 220)}
    assert [tuple(p[:3]) for p in arr[1]] == [(10, 10, 200)] * 4
    assert report["shield"] == {"pixels": 4, "drift": 4, "colours": [[60, 60, 60]],
                                "locked": True}
    assert report["legs"]["locked"] is False
    assert report["legs"]["reason"] == "no palette for this part"


def test_a_pixel_already_in_its_palette_is_not_drift():
    frame = Image.new("RGBA", (2, 1), (60, 60, 60, 255))
    labels = np.zeros((1, 2), np.int16)
    _, report = partkit.lock(frame, labels, ["shield"],
                             {"shield": [[60, 60, 60]]})
    assert report["shield"]["drift"] == 0


def test_extract_reads_the_colours_under_each_mask():
    frame = Image.new("RGBA", (2, 1))
    frame.putpixel((0, 0), (1, 2, 3, 255))
    frame.putpixel((1, 0), (4, 5, 6, 255))
    pal = partkit.extract_palettes(frame, np.array([[0, 1]], np.int16), ["a", "b"])
    assert pal == {"a": [(1, 2, 3)], "b": [(4, 5, 6)]}


# ---------------------------------------------------------------------------
# The weights pin
# ---------------------------------------------------------------------------
def test_weights_that_do_not_hash_to_the_pin_are_refused(tmp_path, monkeypatch):
    def fetch(url, dest):
        dest.write_bytes(b"not the checkpoint")
    with pytest.raises(RuntimeError, match="Refusing"):
        partkit.ensure_weights(tmp_path, fetch=fetch)
    assert not partkit.weights_path(tmp_path).exists()


def test_weights_that_hash_to_the_pin_are_kept_and_not_refetched(tmp_path, monkeypatch):
    blob = b"pretend checkpoint"
    spec = dict(config.SAM2_WEIGHTS, sha256=hashlib.sha256(blob).hexdigest())
    monkeypatch.setattr(config, "SAM2_WEIGHTS", spec)
    calls = []

    def fetch(url, dest):
        calls.append(url)
        dest.write_bytes(blob)
    path = partkit.ensure_weights(tmp_path, fetch=fetch)
    assert path.read_bytes() == blob
    assert calls == [f"https://huggingface.co/{spec['repo']}/resolve/"
                     f"{spec['revision']}/{spec['repo_file']}"]
    partkit.ensure_weights(tmp_path, fetch=fetch)
    assert len(calls) == 1


def test_the_model_entry_is_the_pinned_row():
    entry = partkit.model_entry()
    assert entry["repo"] == "facebook/sam2.1-hiera-small"
    assert entry["license"] == "apache-2.0"
    assert len(entry["sha256"]) == 64


# ---------------------------------------------------------------------------
# Opt-in: the real SAM 2.1 on the real knight base
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("ANIM_SAM2_SMOKE"),
                    reason="set ANIM_SAM2_SMOKE=1 to run SAM 2.1 for real "
                           "(downloads 184 MB on first use)")
def test_sam2_finds_the_knights_helmet_and_legs_on_the_base_render():
    base = Path("/work/work/knight_base")
    from anim import posekit
    from anim.util import read_json

    if not (base / "job.json").exists():
        pytest.skip("no work/knight_base in this checkout")
    # `render.final` is base_sprite.png for a rig/canned base and
    # concept_pick.png for one the concept stage picked - ask the manifest.
    final = (read_json(base / "job.json").get("render") or {}).get("final")
    render = base / "render" / (final or "base_sprite.png")
    if not render.exists():
        pytest.skip(f"no {render.name} in this checkout")

    payload = read_json(base / "pose" / "keypoints.json")
    with Image.open(render) as img:
        rgb = np.asarray(img.convert("RGB"))
    # A canned payload carries no `names`; resolve them from the rig the way
    # post._mask_cells does, or this raises TypeError instead of running.
    names = list(payload.get("names") or posekit.rig_names(payload["rig"]))
    prompts = partkit.prompts_for(config.parts_for_unit("knight"),
                                  payload["frames"][0], names,
                                  tuple(payload["source_size"]), rgb.shape[1::-1])
    results = partkit.Sam2Segmenter().segment(rgb, prompts)
    alpha = np.full(rgb.shape[:2], 255, np.uint8)
    out = partkit.resolve(config.parts_for_unit("knight"), prompts, results, alpha)
    assert out.found["helmet"] and out.found["legs"]
    assert out.mask("helmet").sum() < out.mask("legs").sum()


# ---------------------------------------------------------------------------
# The part tables describe the art, not the rig
# ---------------------------------------------------------------------------
def test_the_archer_is_not_cut_up_like_the_knight():
    """The archer wears one full-length cloak. Naming arms/torso/legs on it
    asks SAM for three regions that share every pixel, and on 2026-09-24 all
    three came back below the contamination gate with 0 px, which failed
    all eight directions."""
    names = [p.name for p in config.parts_for_unit("hooded_archer")]
    assert names == ["hood", "gloves", "boots", "cloak"]
    assert not {"arms", "torso", "legs"} & set(names)
    assert names[-1] == "cloak", ("priority resolves an overlap to the earlier "
                                  "row, so the part that covers the others is last")


def test_every_part_table_prompts_from_joints_the_rig_has():
    """A typo in a joint name is silent - the part just never gets a prompt."""
    from anim import rigkit

    known = set(rigkit.OPENPOSE18)
    for slug, table in config.UNIT_PARTS.items():
        for part in table:
            for joint in part.joints:
                assert joint in known, f"{slug}.{part.name}: unknown joint {joint}"
            for a, b, _t in part.mids:
                assert a in known, f"{slug}.{part.name}: unknown joint {a}"
                assert b in known, f"{slug}.{part.name}: unknown joint {b}"
            assert part.joints or part.mids or part.prop, \
                f"{slug}.{part.name} can never be prompted"


def test_the_archers_hood_reaches_past_its_face_opening():
    """The crown point is what stops SAM returning the dark opening instead of
    the hood: at the knight's 1.35 it lands inside the opening, and the palette
    came out all shadow (2026-09-24, an 11k px blob beating the 26k px hood)."""
    hood = next(p for p in config.parts_for_unit("hooded_archer")
                if p.name == "hood")
    crowns = [t for a, b, t in hood.mids if (a, b) == ("neck", "nose")]
    assert crowns and max(crowns) >= 2.0
