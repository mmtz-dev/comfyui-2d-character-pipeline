"""`anim.rigkit`: sampling, the hierarchy, facing, scale and the 18-point map.

Two kinds of fixture. A tiny synthetic GLB (two bones, one clip, built here
with pygltflib) whose answers can be worked out by hand - that is where the
interpolation and the projection maths are pinned. And the real vendored
KayKit Knight (`rigs/kaykit-adventurers/Knight.glb`, CC0), because the joint
map, the facing detection and the per-rig scale only mean anything against a
real rig.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from anim import posekit, rigkit

REPO = Path(__file__).resolve().parent.parent
KNIGHT = REPO / "rigs" / "kaykit-adventurers" / "Knight.glb"
UAL = REPO / "rigs" / "quaternius" / "UAL1_Standard.glb"
CELL = (304, 320)


# ---------------------------------------------------------------------------
# The synthetic rig: foot -> toes, one clip called Turn
# ---------------------------------------------------------------------------
def write_synthetic_glb(path: Path, *, toes_z: float = 1.0) -> Path:
    """Two bones and one clip.

        foot  (root, at the origin)
          toes  (child, at (0, 0, toes_z))

        Turn, 1 s:
          foot.rotation     LINEAR  identity -> 90 deg about +Y
          foot.scale        LINEAR  1 -> 2
          toes.translation  STEP    (0,0,toes_z) -> (0,0,2*toes_z)

    Every world position is therefore known in closed form.
    """
    import pygltflib as g

    half = math.sqrt(0.5)
    times = np.array([0.0, 1.0], dtype=np.float32)
    rot = np.array([[0, 0, 0, 1], [0, half, 0, half]], dtype=np.float32)
    scl = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)
    tra = np.array([[0, 0, toes_z], [0, 0, 2 * toes_z]], dtype=np.float32)

    blob = b""
    views, accessors = [], []

    def add(arr: np.ndarray, kind: str) -> int:
        nonlocal blob
        raw = np.ascontiguousarray(arr).tobytes()
        while len(blob) % 4:
            blob += b"\0"
        views.append(g.BufferView(buffer=0, byteOffset=len(blob), byteLength=len(raw)))
        flat = arr.reshape(len(arr), -1) if arr.ndim > 1 else arr.reshape(len(arr), 1)
        accessors.append(g.Accessor(
            bufferView=len(views) - 1, componentType=g.FLOAT, count=len(arr),
            type=kind, max=flat.max(axis=0).tolist(), min=flat.min(axis=0).tolist()))
        blob += raw
        return len(accessors) - 1

    a_t = add(times, "SCALAR")
    a_rot = add(rot, "VEC4")
    a_scl = add(scl, "VEC3")
    a_tra = add(tra, "VEC3")

    gltf = g.GLTF2(
        asset=g.Asset(version="2.0", generator="tests/test_rigkit.py"),
        scene=0,
        scenes=[g.Scene(nodes=[0])],
        nodes=[g.Node(name="foot", children=[1]),
               g.Node(name="toes", translation=[0.0, 0.0, toes_z])],
        animations=[g.Animation(
            name="Turn",
            samplers=[g.AnimationSampler(input=a_t, output=a_rot, interpolation="LINEAR"),
                      g.AnimationSampler(input=a_t, output=a_scl, interpolation="LINEAR"),
                      g.AnimationSampler(input=a_t, output=a_tra, interpolation="STEP")],
            channels=[g.AnimationChannel(sampler=0, target=g.AnimationChannelTarget(node=0, path="rotation")),
                      g.AnimationChannel(sampler=1, target=g.AnimationChannelTarget(node=0, path="scale")),
                      g.AnimationChannel(sampler=2, target=g.AnimationChannelTarget(node=1, path="translation"))])],
        buffers=[g.Buffer(byteLength=len(blob))],
        bufferViews=views,
        accessors=accessors,
    )
    gltf.set_binary_blob(blob)
    gltf.save(str(path))
    return path


@pytest.fixture
def synthetic(tmp_path: Path) -> rigkit.Rig:
    return rigkit.load_rig(write_synthetic_glb(tmp_path / "two_bones.glb"))


@pytest.fixture
def synthetic_mirrored(tmp_path: Path) -> rigkit.Rig:
    return rigkit.load_rig(write_synthetic_glb(tmp_path / "mirrored.glb", toes_z=-1.0))


@pytest.fixture(scope="module")
def knight() -> rigkit.Rig:
    if not KNIGHT.exists():
        pytest.skip("rigs/kaykit-adventurers/Knight.glb is not vendored")
    return rigkit.load_rig(KNIGHT)


def toes_at(rig: rigkit.Rig, t: float) -> np.ndarray:
    return rig.pose_at(rig.clip("Turn"), t)[rig.index("toes")][:3, 3]


# ---------------------------------------------------------------------------
# Sampling and the hierarchy
# ---------------------------------------------------------------------------
def test_the_synthetic_rig_loads_with_its_clip(synthetic):
    assert synthetic.names == ["foot", "toes"]
    assert synthetic.parents == [None, 0]
    assert list(synthetic.clips) == ["Turn"]
    assert synthetic.clip("Turn").duration == pytest.approx(1.0)
    assert synthetic.mesh_bounds is None, "no mesh in the file"


def test_the_rest_pose_puts_the_toes_forward(synthetic):
    world = synthetic.rest_world()
    assert world[synthetic.index("toes")][:3, 3] == pytest.approx([0, 0, 1])


def test_linear_rotation_is_slerped_and_scale_lerped_through_the_hierarchy(synthetic):
    """At t=0.5: 45 deg about Y and scale 1.5, both applied to the child."""
    half = math.sqrt(0.5)
    assert toes_at(synthetic, 0.5) == pytest.approx([1.5 * half, 0.0, 1.5 * half], abs=1e-6)


def test_the_endpoints_are_exact(synthetic):
    assert toes_at(synthetic, 0.0) == pytest.approx([0, 0, 1], abs=1e-6)
    # 90 deg about +Y takes +Z to +X; scale 2; STEP translation has reached 2.
    assert toes_at(synthetic, 1.0) == pytest.approx([4, 0, 0], abs=1e-6)


def test_step_interpolation_holds_the_previous_key(synthetic):
    """Just before the end the STEP channel is still on its first key."""
    clip = synthetic.clip("Turn")
    step = next(c for c in clip.channels if c.interpolation == "STEP")
    assert rigkit.sample_channel(step, 0.999) == pytest.approx([0, 0, 1])
    assert rigkit.sample_channel(step, 1.0) == pytest.approx([0, 0, 2])


def test_sampling_is_clamped_outside_the_key_range(synthetic):
    assert toes_at(synthetic, -5.0) == pytest.approx(toes_at(synthetic, 0.0))
    assert toes_at(synthetic, 50.0) == pytest.approx(toes_at(synthetic, 1.0))


def test_slerp_takes_the_short_arc():
    q0 = np.array([0, 0, 0, 1.0])
    q1 = -np.array([0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)])  # 90 deg, negated
    mid = rigkit.slerp(q0, q1, 0.5)
    # Halfway to a 90 deg turn is 45 deg, whichever sign the input used.
    m = rigkit.quat_to_matrix(mid)
    assert m @ np.array([0, 0, 1.0]) == pytest.approx([math.sin(math.pi / 4), 0, math.cos(math.pi / 4)], abs=1e-6)


def test_cubicspline_passes_through_its_keys():
    times = np.array([0.0, 1.0, 2.0])
    values = np.zeros((3, 3, 3))
    values[:, 1] = [[0, 0, 0], [1, 2, 3], [4, 4, 4]]     # the values; tangents zero
    ch = rigkit.Channel(node=0, path="translation", interpolation="CUBICSPLINE",
                        times=times, values=values)
    assert rigkit.sample_channel(ch, 1.0) == pytest.approx([1, 2, 3])
    assert rigkit.sample_channel(ch, 0.5) == pytest.approx([0.5, 1.0, 1.5])


def test_sample_times_include_both_ends_at_about_the_fps():
    ts = rigkit.sample_times(1.0667, 12)
    assert ts[0] == 0.0 and ts[-1] == pytest.approx(1.0667)
    assert len(ts) == 14
    assert np.diff(ts).max() == pytest.approx(1 / 12, abs=0.01)


# ---------------------------------------------------------------------------
# Facing and the projection
# ---------------------------------------------------------------------------
def test_facing_is_read_off_the_toes(synthetic, synthetic_mirrored):
    hint = (("foot", "toes"),)
    fwd = rigkit.facing_direction(rigkit.positions_by_name(synthetic, synthetic.rest_world()), hint)
    back = rigkit.facing_direction(rigkit.positions_by_name(synthetic_mirrored, synthetic_mirrored.rest_world()), hint)
    assert fwd == pytest.approx([0, 0, 1])
    assert back == pytest.approx([0, 0, -1])


def test_a_mirrored_rig_projects_flipped(synthetic, synthetic_mirrored):
    """The same world point lands on the opposite side once forward flips."""
    hint = (("foot", "toes"),)
    frames = []
    for rig in (synthetic, synthetic_mirrored):
        fwd = rigkit.facing_direction(rigkit.positions_by_name(rig, rig.rest_world()), hint)
        frames.append(rigkit.Projection(forward=fwd, floor_y=0.0, scale=100.0,
                                        cell=CELL, baseline=300.0))
    points = np.array([[0, 0, 0], [0, 0, 1.0]])       # origin, and one unit "+z"
    a = frames[0].to_2d(points, anchor=np.zeros(3))
    b = frames[1].to_2d(points, anchor=np.zeros(3))
    assert a[1, 0] - a[0, 0] == pytest.approx(100.0), "+z is forward: goes right"
    assert b[1, 0] - b[0, 0] == pytest.approx(-100.0), "mirrored: goes left"
    assert a[:, 1] == pytest.approx(b[:, 1]), "vertical is untouched by facing"


def test_the_floor_lands_on_the_baseline_and_up_is_screen_up():
    proj = rigkit.Projection(forward=np.array([1.0, 0, 0]), floor_y=0.0,
                             scale=100.0, cell=CELL, baseline=300.0)
    xy = proj.to_2d(np.array([[0, 0, 0], [0, 1.0, 0]]), anchor=np.zeros(3))
    assert xy[0] == pytest.approx([152, 300])
    assert xy[1] == pytest.approx([152, 200]), "one unit up is 100 px higher"


def test_facing_with_vertical_toes_is_refused():
    with pytest.raises(ValueError, match="cannot detect facing"):
        rigkit.facing_direction({"a": np.zeros(3), "b": np.array([0, 1.0, 0])},
                                (("a", "b"),))


# ---------------------------------------------------------------------------
# The 18-point map
# ---------------------------------------------------------------------------
def test_openpose18_is_rtmlibs_own_order():
    assert list(rigkit.OPENPOSE18) == posekit.rig_names("openpose18")


def test_the_preview_limbs_are_rtmlibs_openpose18_links():
    from rtmlib.visualization.skeleton import openpose18

    ours = {frozenset(pair) for pair in rigkit.LIMBS}
    theirs = {frozenset(info["link"]) for info in openpose18["skeleton_info"].values()}
    assert ours == theirs


@pytest.mark.parametrize("jmap", list(rigkit.RIGS.values()), ids=lambda m: m.name)
def test_every_map_covers_all_18_joints(jmap):
    covered = set(jmap.joints) | set(jmap.face)
    assert covered == set(rigkit.OPENPOSE18), sorted(set(rigkit.OPENPOSE18) - covered)
    assert not set(jmap.joints) & set(jmap.face), "a joint is a bone or a face point, not both"


def test_the_knight_is_detected_as_kaykit_and_lacks_nothing(knight):
    jmap = rigkit.detect_map(knight.names)
    assert jmap.name == "kaykit"
    assert jmap.missing(knight.names) == []


def test_the_quaternius_library_is_detected_as_its_own_map():
    if not UAL.exists():
        pytest.skip("rigs/quaternius/UAL1_Standard.glb is not vendored")
    rig = rigkit.load_rig(UAL)
    jmap = rigkit.detect_map(rig.names)
    assert jmap.name == "quaternius-ual"
    assert jmap.missing(rig.names) == []


def test_an_unknown_rig_is_refused_with_the_known_names():
    with pytest.raises(ValueError, match="kaykit"):
        rigkit.detect_map(["Bip01", "Bip01_Spine"])


# ---------------------------------------------------------------------------
# The real Knight: facing, scale, the payload
# ---------------------------------------------------------------------------
def test_the_knight_faces_right_by_detection_not_assumption(knight):
    frame = rigkit.reference_frame(knight)
    assert frame.forward == pytest.approx([0, 0, 1], abs=1e-4), "KayKit faces +Z"
    kp = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL)
    names = kp["names"]
    pts = kp["frames"][0]["keypoints"]
    assert pts[names.index("nose")][0] > pts[names.index("neck")][0], "nose ahead of the neck: facing right"


def test_mirror_flips_the_knight_but_keeps_its_face_in_front(knight):
    kp = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL, mirror=True)
    names = kp["names"]
    pts = kp["frames"][0]["keypoints"]
    assert pts[names.index("nose")][0] < pts[names.index("neck")][0], "faces left"
    plain = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL)["frames"][0]["keypoints"]
    # Mirroring is a reflection about the cell's centre column, nothing else.
    for (x, y), (px, py) in zip(pts, plain):
        assert y == pytest.approx(py, abs=0.02)
        assert x + px == pytest.approx(CELL[0], abs=0.05)
    assert kp["motion"]["forward"] == pytest.approx([0, 0, 1], abs=1e-3)
    assert kp["motion"]["mirror"] is True


def test_one_scale_serves_every_clip_of_the_rig(knight):
    scales = set()
    ankle_rows = set()
    for clip in ("Idle", "1H_Melee_Attack_Slice_Horizontal", "Hit_A", "Death_A"):
        kp = rigkit.clip_keypoints(knight, clip, fps=12, cell=CELL)
        scales.add(kp["motion"]["scale_px_per_unit"])
        first = kp["frames"][0]["keypoints"]
        ankle_rows.add(round(first[kp["names"].index("left_ankle")][1]))
    assert len(scales) == 1, f"scale drifted per clip: {scales}"
    assert len(ankle_rows) == 1, f"the standing ankle moved between clips: {ankle_rows}"


def test_the_standing_height_is_the_agreed_fraction_of_the_cell(knight):
    frame = rigkit.reference_frame(knight)
    kp = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL)
    expected = rigkit.HEIGHT_FRACTION * CELL[1] / frame.height
    assert kp["motion"]["scale_px_per_unit"] == pytest.approx(expected, rel=1e-3)
    assert kp["motion"]["baseline_row"] == pytest.approx(rigkit.BASELINE_FRACTION * CELL[1])


@pytest.mark.parametrize("clip", ["Idle", "1H_Melee_Attack_Slice_Horizontal",
                                  "Hit_A", "Death_A", "Lie_Down", "Lie_StandUp"])
def test_every_joint_stays_inside_the_cell(knight, clip):
    """rtmlib silently drops any limb whose endpoint touches the frame edge."""
    kp = rigkit.clip_keypoints(knight, clip, fps=12, cell=CELL)
    for entry in kp["frames"]:
        for x, y in entry["keypoints"]:
            assert 1 < x < CELL[0] - 1 and 1 < y < CELL[1] - 1, (clip, entry["index"], x, y)


def test_the_payload_is_the_pose_stages_schema(knight, keypoints_payload):
    kp = rigkit.clip_keypoints(knight, "Hit_A", fps=12, cell=CELL)
    assert kp["schema"] == keypoints_payload["schema"] == posekit.KEYPOINTS_SCHEMA
    assert kp["rig"] == "openpose18"
    assert set(keypoints_payload) <= set(kp), "every key `anim pose` writes is present"
    assert kp["source_size"] == list(CELL)
    assert len(kp["frames"]) == round(0.6667 * 12) + 1
    for entry in kp["frames"]:
        assert set(entry) >= {"index", "file", "keypoints", "scores", "person"}
        assert len(entry["keypoints"]) == len(entry["scores"]) == 18
        assert entry["scores"] == [1.0] * 18, "a rig has no detector"
        assert entry["file"] == f"{entry['index'] + 1:04d}.png"
    # And the normaliser the picker uses accepts it whole.
    poses, _ = posekit.series_from(kp)
    assert all(p is not None for p in poses)


def test_the_feet_stand_on_the_floor_in_idle(knight):
    kp = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL)
    names = kp["names"]
    lowest = max(y for x, y in kp["frames"][0]["keypoints"])
    baseline = kp["motion"]["baseline_row"]
    assert baseline - 25 < lowest < baseline, "ankles a little above the floor row"
    assert lowest == kp["frames"][0]["keypoints"][names.index("left_ankle")][1] or \
        lowest == kp["frames"][0]["keypoints"][names.index("right_ankle")][1]


# ---------------------------------------------------------------------------
# Props - the skeleton props feature
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ual() -> rigkit.Rig:
    if not UAL.exists():
        pytest.skip("rigs/quaternius/UAL1_Standard.glb is not vendored")
    return rigkit.load_rig(UAL)


def _outline(kind: str, polys: list) -> np.ndarray:
    return np.asarray(polys[0], dtype=float)


def test_a_rig_without_props_writes_no_props_key(knight):
    kp = rigkit.clip_keypoints(knight, "Hit_A", fps=12, cell=CELL)
    assert kp["motion"]["props"] == []
    assert all("props" not in entry for entry in kp["frames"])


def test_an_unknown_prop_is_refused_with_the_known_kinds(ual):
    with pytest.raises(ValueError, match="sword"):
        rigkit.reference_frame(ual, props=("halberd",))


def test_props_ride_the_hand_and_forearm_bones(ual):
    frame = rigkit.reference_frame(ual, props=("sword", "shield"))
    assert [(p.kind, p.bone, p.end_bone) for p in frame.props] == [
        ("sword", "hand_r", None), ("shield", "lowerarm_l", "hand_l")]


def test_at_rest_the_blade_points_forward_and_the_shield_faces_up(ual):
    """The palms-down convention: T-pose, a gripped blade points forward out
    of the fist, the back of the forearm (and so the shield) faces up."""
    frame = rigkit.reference_frame(ual, props=("sword", "shield"))
    world = ual.rest_world()
    props = dict(rigkit.prop_polygons(ual, world, frame))
    hand = world[ual.index("hand_r")][:3, 3]
    blade = props["sword"][0]
    tip = blade[2]
    reach = tip - hand
    assert reach @ frame.forward == pytest.approx(
        rigkit.PROP_SIZES["sword"]["blade"] * frame.height, rel=1e-3)
    assert abs(reach[1]) < 1e-6, "level, at rest"
    kite = props["shield"][0]
    normal = np.cross(kite[1] - kite[0], kite[3] - kite[0])
    normal /= np.linalg.norm(normal)
    assert abs(normal[1]) == pytest.approx(1.0, abs=1e-3), "face up (or down) at rest"
    forearm_mid = (world[ual.index("lowerarm_l")][:3, 3] + world[ual.index("hand_l")][:3, 3]) / 2
    assert np.linalg.norm(kite.mean(axis=0) - forearm_mid) < 0.15 * frame.height


def test_props_are_sized_by_the_standing_height(ual):
    frame = rigkit.reference_frame(ual, props=("sword",))
    blade = dict(rigkit.prop_polygons(ual, ual.rest_world(), frame))["sword"][0]
    length = np.linalg.norm(blade[2] - (blade[0] + blade[4]) / 2)
    sizes = rigkit.PROP_SIZES["sword"]
    assert length == pytest.approx((sizes["blade"] + sizes["pommel"]) * frame.height, rel=1e-3)


def test_props_project_with_the_joints_centre_not_their_own(ual):
    """The guard is centred on the hand, so its 2D centre must land on the
    right-wrist keypoint in every frame - which only holds if the props were
    centred on the joints' extent, not on the props' own."""
    frame = rigkit.reference_frame(ual, props=("sword", "shield"))
    kp = rigkit.clip_keypoints(ual, "Sword_Attack", fps=12, cell=CELL, frame=frame)
    assert kp["motion"]["props"] == ["sword", "shield"]
    wrist = kp["names"].index("right_wrist")
    for entry in kp["frames"]:
        kinds = [p["kind"] for p in entry["props"]]
        assert kinds == ["sword", "shield"]
        guard = np.asarray(entry["props"][0]["polygons"][1], dtype=float)
        assert guard.mean(axis=0) == pytest.approx(entry["keypoints"][wrist], abs=0.05)
        assert len(entry["props"][1]["polygons"][0]) == 5, "a kite"


def test_props_do_not_move_the_joints(ual):
    plain = rigkit.clip_keypoints(ual, "Sword_Attack", fps=12, cell=CELL)
    frame = rigkit.reference_frame(ual, props=("sword", "shield"))
    armed = rigkit.clip_keypoints(ual, "Sword_Attack", fps=12, cell=CELL, frame=frame)
    assert [e["keypoints"] for e in plain["frames"]] == [e["keypoints"] for e in armed["frames"]]
    poses, _ = posekit.series_from(armed)
    assert all(p is not None for p in poses), "the picker still reads it"


def test_the_blade_follows_the_swing(ual):
    """Sword_Attack is a horizontal slash: on the recovery (frame 10) the sword
    hand is out front and the blade reaches further forward still."""
    frame = rigkit.reference_frame(ual, props=("sword",))
    kp = rigkit.clip_keypoints(ual, "Sword_Attack", fps=12, cell=CELL, frame=frame)
    wrist = kp["names"].index("right_wrist")
    entry = kp["frames"][10]
    tip_x = entry["props"][0]["polygons"][0][2][0]
    assert tip_x > entry["keypoints"][wrist][0] + 40
