"""Unit tests for spike3d/render_openpose.py, against a synthetic joint list.

No GPU, no Mixamo download, no .dae file: every test builds its joints in Python. The point is
to prove -- before any GPU session is spent -- that the 3D arm of the spike can produce a
skeleton image that is byte-identical in style to what the 2D chain feeds the ControlNet.

Run inside the CPU pipeline container (it has rtmlib==0.0.16 and, via requirements-dev.txt,
pytest):

    docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
      run --rm --entrypoint pytest anim spike3d/ -q

``--entrypoint pytest`` because the image's ENTRYPOINT is ``anim``; the explicit ``spike3d/``
because ``pyproject.toml`` sets ``testpaths = ["tests"]`` and a bare ``pytest`` would not collect
this file. On an image built before pytest was added, this file also runs standalone:

    docker --context desktop-linux compose -f tools/anim-pipeline/compose.yml \
      run --rm --entrypoint python anim spike3d/test_render_openpose.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import render_openpose as ro  # noqa: E402

W = H = 512

# A side-view sword-strike pose: figure faces right, right arm raised and forward, left leg
# back. COCO-18 order, [x, y, confidence]. Every joint is comfortably inside the frame so no
# limb hits rtmlib's border guard.
STRIKE: list[list[float]] = [
    [262, 70, 1.0],    # 0  nose  (mixamotoopenpose maps this from HeadTop_End)
    [256, 110, 1.0],   # 1  neck
    [250, 125, 1.0],   # 2  right_shoulder
    [300, 105, 1.0],   # 3  right_elbow
    [350, 70, 1.0],    # 4  right_wrist
    [262, 125, 1.0],   # 5  left_shoulder
    [300, 150, 1.0],   # 6  left_elbow
    [330, 130, 1.0],   # 7  left_wrist
    [250, 230, 1.0],   # 8  right_hip
    [265, 320, 1.0],   # 9  right_knee
    [255, 410, 1.0],   # 10 right_ankle
    [262, 230, 1.0],   # 11 left_hip
    [230, 320, 1.0],   # 12 left_knee
    [220, 410, 1.0],   # 13 left_ankle
    [270, 80, 1.0],    # 14 right_eye
    [272, 82, 1.0],    # 15 left_eye
    [255, 85, 1.0],    # 16 right_ear
    [257, 87, 1.0],    # 17 left_ear
]


# Probe points, precomputed from STRIKE. rtmlib draws limbs first, then joint dots on top, so a
# joint's centre pixel carries the JOINT colour and a limb's midpoint carries the LIMB colour.
#
# These are located-pixel probes rather than "is this colour anywhere in the image" checks, for a
# reason worth writing down: rtmlib's openpose18 colour table is a rainbow ramp that is CLOSED
# UNDER CHANNEL REVERSAL -- [255,0,0] and [0,0,255] are both in it, [255,85,0] and [0,85,255] are
# both in it, and so on for every entry. So the SET of colours in the image is identical whether
# joint colours are reversed or not, and identical whether the file is saved RGB or BGR. Set
# membership proves nothing here. Only a pixel at a known location does.
P_LIMB3_MID = (325, 87)    # right_elbow -> right_wrist, limb colour [255,255,0]
P_LIMB9_MID = (259, 170)   # neck -> left_hip,          limb colour [0,255,225] (rtmlib's typo)
P_LIMB6_MID = (253, 170)   # neck -> right_hip,         limb colour [0,255,0]
P_JOINT13 = (220, 410)     # left_ankle dot, table colour [85,0,255]
P_JOINT0 = (262, 70)       # nose dot,       table colour [255,0,0]
P_JOINT4 = (350, 70)       # right_wrist dot


def _px(img: np.ndarray, point: tuple[int, int]) -> tuple[int, int, int]:
    """Colour at (x, y). numpy is row-major, so this exists to stop the axes being swapped."""
    x, y = point
    return tuple(int(c) for c in img[y, x])


# ---------------------------------------------------------------------------
# The joint set really is the one rtmlib and mixamotoopenpose both use
# ---------------------------------------------------------------------------


def test_keypoint_order_matches_rtmlib_openpose18() -> None:
    """Our names are rtmlib's, in rtmlib's id order -- not a hand-typed copy that can drift."""
    from rtmlib.visualization.skeleton import openpose18

    rtm = [openpose18["keypoint_info"][i]["name"] for i in range(ro.NUM_KEYPOINTS)]
    assert rtm == list(ro.KEYPOINT_NAMES)


def test_keypoint_order_matches_mixamotoopenpose() -> None:
    """The order Astropulse/mixamotoopenpose emits, transcribed from its source.

    If these ever disagree, its candidate/subset arrays would silently land on the wrong joints
    -- a left leg drawn as a right arm, and nobody would see an error.
    """
    astropulse_order = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
        "left_eye", "right_ear", "left_ear",
    ]
    assert astropulse_order == list(ro.KEYPOINT_NAMES)


# ---------------------------------------------------------------------------
# The render itself
# ---------------------------------------------------------------------------


def test_synthetic_pose_renders_a_skeleton() -> None:
    img = ro.render(STRIKE, width=W, height=H)
    assert img.shape == (H, W, 3)
    assert img.dtype == np.uint8
    # Background stays black; the rig is a small fraction of a 512x512 canvas but far from empty.
    ink = int(np.count_nonzero(img.any(axis=2)))
    assert ink > 2000, f"only {ink} non-black pixels - the rig did not draw"
    assert ink < W * H * 0.25, f"{ink} non-black pixels - that is not a stick figure"


def test_render_is_exactly_rtmlibs_own_output() -> None:
    """The regression guard: we call rtmlib, we do not reimplement it.

    If someone later "fixes" render() by hand-drawing limbs, this fails, because the 2D and 3D
    arms of the comparison would stop being pixel-comparable.
    """
    from rtmlib import draw_skeleton

    keypoints = np.array([[[t[0], t[1]] for t in STRIKE]], dtype=np.float32)
    scores = np.array([[t[2] for t in STRIKE]], dtype=np.float32)
    expected = draw_skeleton(
        np.zeros((H, W, 3), dtype=np.uint8),
        keypoints,
        scores,
        openpose_skeleton=True,
        kpt_thr=ro.DEFAULT_KPT_THR,
        line_width=ro.line_width_for(W, H),
    )
    assert np.array_equal(ro.render(STRIKE, width=W, height=H), expected)


def test_limb_colours_are_rtmlibs_table_including_its_deviations() -> None:
    """rtmlib's openpose18 table differs from canonical OpenPose in two limbs.

    Documented in RESEARCH.md §3. Asserted here so the spike cannot quietly start matching the
    canonical table instead -- which is what mixamotoopenpose's own renderer would give us, and
    which would make the two arms of the comparison incomparable.
    """
    img = ro.render(STRIKE, width=W, height=H)

    # neck -> left_hip: rtmlib says [0,255,225]; canonical OpenPose says [0,255,255].
    assert _px(img, P_LIMB9_MID) == (0, 255, 225)
    assert _px(img, P_LIMB9_MID) != (0, 255, 255)
    # The limbs where rtmlib and canonical agree still land on the shared value.
    assert _px(img, P_LIMB6_MID) == (0, 255, 0)
    assert _px(img, P_LIMB3_MID) == (255, 255, 0)


def test_joint_dots_use_rtmlibs_reversed_colour() -> None:
    """rtmlib reverses JOINT colours (``kpt_info['color'][::-1]``) but not LIMB colours.

    left_ankle's table colour is [85,0,255]; its dot is drawn (255,0,85). nose is [255,0,0] and
    its dot is (0,0,255). Both checked at the dot centre, because the palette is reversal-closed
    and so a colour-set check cannot see this at all.
    """
    from rtmlib.visualization.skeleton import openpose18

    img = ro.render(STRIKE, width=W, height=H)

    table_13 = tuple(openpose18["keypoint_info"][13]["color"])
    table_0 = tuple(openpose18["keypoint_info"][0]["color"])
    assert _px(img, P_JOINT13) == table_13[::-1] == (255, 0, 85)
    assert _px(img, P_JOINT13) != table_13
    assert _px(img, P_JOINT0) == table_0[::-1] == (0, 0, 255)


def test_low_confidence_joint_is_dropped() -> None:
    """rtmlib's kpt_thr gate: a joint below threshold takes its dot and its limbs with it."""
    faded = [list(t) for t in STRIKE]
    faded[4][2] = 0.1  # right_wrist
    img = ro.render(faded, width=W, height=H)

    assert _px(img, P_LIMB3_MID) == (0, 0, 0), "right_elbow->right_wrist drew below kpt_thr"
    assert _px(img, P_JOINT4) == (0, 0, 0), "the right_wrist dot drew below kpt_thr"
    assert _px(img, P_LIMB9_MID) == (0, 255, 225), "the rest of the rig should be untouched"


def test_border_drop_is_detected_and_real() -> None:
    """A joint on the frame edge loses limbs with no error from rtmlib. We report it.

    rtmlib's guard is ``x <= 0 or x >= w or y <= 0 or y >= h`` -- note ``<=``, so a joint at
    exactly x == 0 counts as out of frame. This is why fit_to_canvas() keeps a margin.
    """
    edged = [list(t) for t in STRIKE]
    edged[4] = [0.0, 70.0, 1.0]  # right_wrist pinned to x == 0

    dropped = ro.limbs_dropped_at_border(edged, W, H)
    assert ("right_elbow", "right_wrist") in dropped

    img = ro.render(edged, width=W, height=H)
    assert _px(img, P_LIMB3_MID) == (0, 0, 0), "limb drew despite the border guard"
    assert _px(img, P_LIMB9_MID) == (0, 255, 225), "unrelated limbs should be unaffected"


def test_fit_to_canvas_removes_border_drops() -> None:
    """An out-of-frame pose is brought in, aspect preserved, and then loses no limbs."""
    huge = [[t[0] * 3 - 300, t[1] * 3 - 300, t[2]] for t in STRIKE]
    assert ro.limbs_dropped_at_border(huge, W, H), "fixture should start out of frame"

    fitted = ro.fit_to_canvas(huge, W, H, margin=8)
    assert ro.limbs_dropped_at_border(fitted, W, H) == []

    xs = [t[0] for t in fitted]
    ys = [t[1] for t in fitted]
    assert min(xs) >= 8 - 1e-6 and max(xs) <= W - 8 + 1e-6
    assert min(ys) >= 8 - 1e-6 and max(ys) <= H - 8 + 1e-6

    # Aspect preserved: the same uniform scale on both axes.
    sx = (max(xs) - min(xs)) / (max(t[0] for t in huge) - min(t[0] for t in huge))
    sy = (max(ys) - min(ys)) / (max(t[1] for t in huge) - min(t[1] for t in huge))
    assert abs(sx - sy) < 1e-6


def test_fit_keeps_invisible_joints_invisible() -> None:
    partial = [list(t) for t in STRIKE]
    partial[16] = [0.0, 0.0, 0.0]  # right_ear missing, as Mixamo rigs often are
    fitted = ro.fit_to_canvas(partial, W, H)
    assert fitted[16][2] == 0.0


# ---------------------------------------------------------------------------
# Input shapes
# ---------------------------------------------------------------------------


def test_mixamotoopenpose_candidate_subset_shape() -> None:
    """The in-memory shape convert_dae_to_openpose() returns. It never writes JSON itself."""
    candidate = [[t[0], t[1], t[2]] for t in STRIKE]
    subset = [[i for i in range(ro.NUM_KEYPOINTS)]]
    frames = ro.frames_from_mixamotoopenpose([{"candidate": candidate, "subset": subset}])
    assert len(frames) == 1
    assert frames[0] == [[float(a), float(b), float(c)] for a, b, c in STRIKE]


def test_mixamotoopenpose_missing_joint_becomes_zero_confidence() -> None:
    """-1 in subset means "this bone was not in the .dae" - eyes and ears, usually."""
    candidate = [[t[0], t[1], t[2]] for t in STRIKE]
    subset = [[i for i in range(ro.NUM_KEYPOINTS)]]
    subset[0][16] = -1
    frames = ro.frames_from_mixamotoopenpose([{"candidate": candidate, "subset": subset}])
    assert frames[0][16] == [0.0, 0.0, 0.0]


def test_openpose_standard_json_shape(tmp_path: Path) -> None:
    flat: list[float] = []
    for x, y, c in STRIKE:
        flat += [float(x), float(y), float(c)]
    p = tmp_path / "pose.json"
    p.write_text(
        json.dumps({"canvas_width": 640, "canvas_height": 480, "people": [{"pose_keypoints_2d": flat}]})
    )
    frames, w, h = ro.load_frames(p)
    assert (w, h) == (640, 480)
    assert frames[0][0] == [262.0, 70.0, 1.0]


def test_bare_keypoint_list(tmp_path: Path) -> None:
    p = tmp_path / "bare.json"
    p.write_text(json.dumps([[t[0], t[1]] for t in STRIKE]))  # x,y only -> confidence 1.0
    frames, w, h = ro.load_frames(p)
    assert (w, h) == (None, None)
    assert frames[0][0][2] == 1.0


def test_wrong_keypoint_count_is_rejected() -> None:
    try:
        ro.render(STRIKE[:12], width=W, height=H)
    except ValueError as exc:
        assert "18" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a 12-keypoint pose should not render")


# ---------------------------------------------------------------------------
# Thickness and channel order
# ---------------------------------------------------------------------------


def test_line_width_scales_off_the_512px_reference() -> None:
    assert ro.line_width_for(512, 512) == 2      # rtmlib's default == canonical stickwidth 4
    assert ro.line_width_for(1024, 1024) == 4
    assert ro.line_width_for(304, 320) == 1      # the 2D chain's strip cell
    assert ro.line_width_for(32, 32) == 1        # never zero


def test_thicker_line_width_puts_down_more_ink() -> None:
    thin = ro.render(STRIKE, width=W, height=H, line_width=1)
    thick = ro.render(STRIKE, width=W, height=H, line_width=6)
    assert int(np.count_nonzero(thick.any(axis=2))) > int(np.count_nonzero(thin.any(axis=2)))


def test_channel_order_is_a_deliberate_choice(tmp_path: Path) -> None:
    """The file, not the array, is where the channel order gets committed.

    Checked at a located pixel: the palette is reversal-closed, so comparing colour SETS between
    an RGB and a BGR save would find them identical and prove nothing.
    """
    from PIL import Image

    img = ro.render(STRIKE, width=W, height=H)
    rgb_path, bgr_path = tmp_path / "rgb.png", tmp_path / "bgr.png"
    ro.save_png(img, rgb_path, channel_order="rgb")
    ro.save_png(img, bgr_path, channel_order="bgr")

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    bgr = np.array(Image.open(bgr_path).convert("RGB"))
    assert not np.array_equal(rgb, bgr), "the two orders must actually differ"
    assert np.array_equal(rgb[..., ::-1], bgr)
    # Default 'rgb' keeps LIMB colours on rtmlib's own table, which is the canonical OpenPose
    # table apart from the two deviations above.
    assert _px(rgb, P_LIMB9_MID) == (0, 255, 225)
    assert _px(bgr, P_LIMB9_MID) == (225, 255, 0)


def test_save_png_rejects_a_nonsense_order(tmp_path: Path) -> None:
    img = ro.render(STRIKE, width=64, height=64)
    try:
        ro.save_png(img, tmp_path / "x.png", channel_order="rbg")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("channel_order should be validated")


# ---------------------------------------------------------------------------
# The vendored mixamotoopenpose loads with its module-level CLI removed
# ---------------------------------------------------------------------------


def test_vendored_mixamotoopenpose_loads_without_running_its_cli() -> None:
    """mixamo_to_openpose.py calls parse_args() AND dispatches a conversion at module level.

    A plain ``import`` would parse pytest's argv and then crash or silently convert something.
    dae_to_joints.load_upstream() strips those two module-level pieces. This proves the surgery
    still applies to the pinned source (66ec14df) and that the functions survive it -- checked
    here rather than discovered mid-session with a .dae in hand.
    """
    import dae_to_joints

    m2o = dae_to_joints.load_upstream()
    assert callable(m2o.convert_dae_to_openpose)
    assert callable(m2o.reduce_frames)
    assert m2o.reduce_frames(list(range(30)), 4) == [0, 8, 15, 22]


def test_vendored_joint_order_is_the_one_we_render() -> None:
    """The live check behind test_keypoint_order_matches_mixamotoopenpose's transcription."""
    import dae_to_joints

    assert list(dae_to_joints.load_upstream().openpose_keypoints) == list(ro.KEYPOINT_NAMES)


def test_surgery_anchors_are_asserted_not_assumed(tmp_path: Path) -> None:
    """If upstream is re-pinned and the anchors move, we must fail loudly, not half-patch."""
    import dae_to_joints

    broken = tmp_path / "mixamo_to_openpose.py"
    broken.write_text("args = vars(ap.parse_args())\n")  # has one anchor, not the other
    try:
        dae_to_joints.load_upstream(broken)
    except SystemExit as exc:
        assert 'if os.path.isfile(args["input"]):' in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a source missing the dispatch anchor should be refused")


# ---------------------------------------------------------------------------
# Standalone runner, so the suite works without pytest in the image
# ---------------------------------------------------------------------------


def _run_standalone() -> int:
    import inspect
    import tempfile
    import traceback

    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as td:
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                kwargs["tmp_path"] = Path(td)
            try:
                fn(**kwargs)
                print(f"PASS  {name}")
            except Exception:
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
