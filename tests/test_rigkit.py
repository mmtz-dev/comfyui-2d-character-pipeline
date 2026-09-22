"""`anim.rigkit`: sampling, the hierarchy, facing, scale and the 18-point map.

Two kinds of fixture. A tiny synthetic GLB (two bones, one clip, built here
with pygltflib) whose answers can be worked out by hand - that is where the
interpolation and the projection maths are pinned. And the real vendored
KayKit Knight (`rigs/kaykit-adventurers/Knight.glb`, CC0), because the joint
map, the facing detection and the per-rig scale only mean anything against a
real rig.
"""

from __future__ import annotations

import json
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


# Every OpenPose body joint onto the rig's only two bones, so
# `rigkit.reference_frame` / `clip_keypoints` have something to resolve. The
# face offsets are copied from `rigkit.KAYKIT` - the numbers don't matter here,
# only that every face joint has one. Shared with test_direction_sheet.py and
# test_turnaround.py.
TWO_BONE_MAP = rigkit.JointMap(
    name="two-bone-test",
    joints={
        "neck": "foot",
        "right_shoulder": "foot", "right_elbow": "foot", "right_wrist": "foot",
        "left_shoulder": "foot", "left_elbow": "foot", "left_wrist": "foot",
        "right_hip": "foot", "right_knee": "foot", "right_ankle": "foot",
        "left_hip": "foot", "left_knee": "foot", "left_ankle": "foot",
    },
    head="toes",
    forward_hint=(("foot", "toes"),),
    face={
        "nose": (0.48, 0.42, 0.0),
        "right_eye": (0.58, 0.34, -0.16), "left_eye": (0.58, 0.34, 0.16),
        "right_ear": (0.55, 0.02, -0.40), "left_ear": (0.55, 0.02, 0.40),
    },
    detect=("foot", "toes"),
    note="test fixture only - not a real rig family, never added to RIGS",
)


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


# ---------------------------------------------------------------------------
# Depth - the rig props feature
# ---------------------------------------------------------------------------
def _two_bone_world(toes_at: np.ndarray) -> np.ndarray:
    """World matrices for a two-node rig: node 0 at the origin, node 1 at `toes_at`."""
    world = np.stack([np.eye(4), np.eye(4)])
    world[1, :3, 3] = toes_at
    return world


def test_skinning_is_glTFs_joint_times_inverse_bind():
    """A vertex bound wholly to node 1 rides that node: rest at its origin,
    posed at its origin. Half-and-half lands halfway. The mesh node's own
    transform never enters."""
    rest = _two_bone_world(np.array([0.0, 0.0, 1.0]))
    ibm = np.linalg.inv(rest)
    part = rigkit.MeshPart(
        name="t", positions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        triangles=np.array([[0, 1, 2]]),
        joints=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
        weights=np.array([[1.0, 0, 0, 0], [0.5, 0.5, 0, 0], [1.0, 0, 0, 0]]),
        skin_joints=(0, 1), ibm=ibm)
    assert rigkit.skin_part(part, rest) == pytest.approx(part.positions)
    posed = _two_bone_world(np.array([4.0, 0.0, 0.0]))
    out = rigkit.skin_part(part, posed)
    assert out[0] == pytest.approx([4.0, 0.0, 0.0])
    assert out[1] == pytest.approx([2.0, 0.0, 0.5]), "half on node 0 (fixed), half on node 1"
    assert out[2] == pytest.approx([0.0, 0.0, 0.0])


def test_the_zbuffer_keeps_the_nearest_surface_and_leaves_the_rest_empty():
    """Two triangles over the same pixels: the nearer (larger depth) wins;
    pixels nobody covers stay -inf; a pixel centre exactly on an edge counts."""
    pts = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0],     # near, z = 2
                    [0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])    # far, z = 1
    tris = np.array([[3, 4, 5], [0, 1, 2]])                 # far drawn first
    z = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    zbuf = rigkit.rasterize_depth(pts, z, tris, cell=(10, 10))
    assert zbuf[0, 0] == 2.0
    assert zbuf[3, 3] == 2.0
    assert np.isinf(zbuf[9, 9]) and zbuf[9, 9] < 0
    assert np.isinf(zbuf[0, 9])
    assert np.isfinite(zbuf).sum() == 36, "the lower-left half of an 8x8 square"


def test_the_zbuffer_interpolates_depth_across_a_triangle():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    z = np.array([0.0, 10.0, 0.0])
    zbuf = rigkit.rasterize_depth(pts, z, np.array([[0, 1, 2]]), cell=(12, 12))
    assert zbuf[0, 4] == pytest.approx(4.5), "depth follows x at the centre of pixel 4"


def test_a_big_triangle_takes_the_slow_path_and_still_fills():
    pts = np.array([[0.0, 0.0], [300.0, 0.0], [0.0, 300.0]])
    zbuf = rigkit.rasterize_depth(pts, np.ones(3), np.array([[0, 1, 2]]), cell=(304, 320))
    assert np.isfinite(zbuf).sum() > 44000


def test_the_grey_mapping_is_near_bright_far_dark_background_black():
    zbuf = np.array([[-np.inf, 0.0], [0.5, 1.0]])
    grey = rigkit.depth_to_grey(zbuf, far=0.0, near=1.0)
    assert grey.dtype == np.uint8
    assert grey[0, 0] == 0, "background"
    assert grey[0, 1] == rigkit.DEPTH_FAR_GREY, "the farthest surface is not black"
    assert grey[1, 1] == 255, "the nearest is white"
    assert grey[0, 1] < grey[1, 0] < grey[1, 1]


def test_extrusion_makes_a_closed_solid_around_the_outline():
    square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    verts, tris = rigkit.extrude_polygon(square, thickness=0.2)
    assert verts.shape == (8, 3) and tris.shape == (2 * 2 + 4 * 2, 3)
    assert verts[:, 1].min() == pytest.approx(-0.1) and verts[:, 1].max() == pytest.approx(0.1)
    assert verts.mean(axis=0) == pytest.approx(square.mean(axis=0)), "centred on the outline"


def test_the_viewer_stands_on_the_characters_right(ual):
    """With forward on +x and up on +y the camera axis is forward x up: the
    sword hand is the near hand, the shield hand the far one. `mirror` swaps."""
    frame = rigkit.reference_frame(ual)
    proj = rigkit.Projection.for_rig(frame, CELL)
    assert proj.near == pytest.approx(np.cross(frame.forward, [0, 1, 0]))
    world = ual.rest_world()
    hand_r = world[ual.index("hand_r")][:3, 3]
    hand_l = world[ual.index("hand_l")][:3, 3]
    assert proj.depth_of(hand_r[None])[0] > proj.depth_of(hand_l[None])[0]
    mirrored = rigkit.Projection.for_rig(rigkit.reference_frame(ual, mirror=True), CELL)
    assert mirrored.depth_of(hand_r[None])[0] < mirrored.depth_of(hand_l[None])[0]


def test_the_quaternius_mannequin_loads_as_skinned_parts(ual, knight):
    assert ual.skinned
    assert [p.name for p in ual.parts] == ["Mannequin#0", "Mannequin#1"]
    for part in ual.parts:
        assert part.joints.max() < len(part.skin_joints) == 65
        assert part.weights.sum(axis=1) == pytest.approx(np.ones(len(part.weights)), abs=1e-3)
        assert part.triangles.max() < len(part.positions)
        assert part.ibm.shape == (65, 4, 4)
        assert part.ibm[:, 3, :] == pytest.approx(np.tile([0, 0, 0, 1], (65, 1))), \
            "row-major after the column-major read"
    assert knight.skinned, "the KayKit body is skinned too"
    assert len(knight.parts) == 6, "six skinned body parts; the static helmet, cape, shields and swords are not parts"


def test_a_bare_rig_has_no_parts_and_renders_no_depth(synthetic):
    """The two-bone rig has no mesh and no joint map, so a hand-made frame
    with no props stands in: nothing to skin, nothing to draw."""
    assert not synthetic.skinned
    bare = rigkit.RigFrame(map=rigkit.KAYKIT, forward=np.array([0, 0, 1.0]),
                           mirror=False, floor_y=0.0, top_y=1.0, head_length=1.0,
                           head_forward_local=np.array([0, 0, 1.0]),
                           head_up_local=np.array([0, 1.0, 0]))
    verts, tris = rigkit.posed_mesh(synthetic, synthetic.rest_world(), bare)
    assert len(verts) == 0 and len(tris) == 0


def test_the_skinned_rest_pose_matches_the_files_own_extent(ual):
    """In the bind pose joint x inverse-bind is the identity, so the skinned
    vertices ARE the file's vertices - the extent `mesh_bounds` read."""
    world = ual.rest_world()
    verts = np.concatenate([rigkit.skin_part(p, world) for p in ual.parts])
    lo, hi = ual.mesh_bounds
    assert verts.min(axis=0) == pytest.approx(lo, abs=1e-4)
    assert verts.max(axis=0) == pytest.approx(hi, abs=1e-4)


def test_the_guards_depth_centroid_lands_on_the_wrist_keypoint(ual):
    """The sword's crossguard is centred on the hand, so the pixels its depth
    solid covers must centre on the right-wrist keypoint in every frame of
    the swing - the proof that depth and skeleton share one projection."""
    frame = rigkit.reference_frame(ual, props=("sword",))
    kp = rigkit.clip_keypoints(ual, "Sword_Attack", fps=12, cell=CELL, frame=frame)
    proj = rigkit.Projection.for_rig(frame, CELL)
    wrist = kp["names"].index("right_wrist")
    for entry, t in zip(kp["frames"], rigkit.sample_times(ual.clip("Sword_Attack").duration, 12)):
        world = ual.pose_at(ual.clip("Sword_Attack"), float(t))
        joints = rigkit.openpose_points(ual, world, frame)
        (kind, verts, tris), = rigkit.prop_meshes(ual, world, frame)
        assert kind == "sword"
        guard_verts = verts[10:18]                        # outline 2 of 2, extruded: 8 verts
        guard_tris = tris[(tris >= 10).all(axis=1)] - 10
        zbuf = rigkit.rasterize_depth(proj.to_2d(guard_verts, anchor=proj.anchor_for(joints)),
                                      proj.depth_of(guard_verts), guard_tris, CELL)
        ys, xs = np.nonzero(np.isfinite(zbuf))
        assert len(xs) > 0
        centroid = [xs.mean() + 0.5, ys.mean() + 0.5]
        assert centroid == pytest.approx(entry["keypoints"][wrist], abs=1.5)


def test_clip_depth_is_one_frame_per_sample_with_one_range(ual):
    frame = rigkit.reference_frame(ual, props=("sword", "shield"))
    dc = rigkit.clip_depth(ual, "Sword_Attack", fps=12, cell=CELL, frame=frame)
    n = len(rigkit.sample_times(ual.clip("Sword_Attack").duration, 12))
    assert len(dc.zbufs) == n
    assert dc.far < dc.near
    frames = dc.frames()
    assert all(f.shape == (CELL[1], CELL[0]) and f.dtype == np.uint8 for f in frames)
    drawn = np.concatenate([f[f > 0] for f in frames])
    assert drawn.min() <= rigkit.DEPTH_FAR_GREY + 2 and drawn.max() >= 253, \
        "the clip's range spans far..near (pixel centres, so within a step of each)"
    body = frames[0][frames[0] > 0]
    assert body.min() > rigkit.DEPTH_FAR_GREY + 10 and body.max() < 255, \
        "at rest neither extreme is reached: the swing sets the range, not the frame"


def _probe_on_the_blade(blade: np.ndarray, back: float = 40.0) -> tuple[int, int]:
    """A cell pixel on the blade `back` px short of the tip - the tip itself
    may leave the cell on a big swing."""
    axis = blade[2] - (blade[0] + blade[4]) / 2
    x, y = blade[2] - back * axis / np.linalg.norm(axis)
    return int(x), int(y)


def test_the_sword_shows_in_depth_only_when_the_unit_carries_it(ual):
    """Pixels at the blade tip (on the recovery, where the blade is out in
    front of the body) are drawn with the sword, empty without."""
    armed = rigkit.reference_frame(ual, props=("sword",))
    kp = rigkit.clip_keypoints(ual, "Sword_Attack", fps=12, cell=CELL, frame=armed)
    entry = kp["frames"][10]                                 # recovery: blade out front
    blade = np.asarray(entry["props"][0]["polygons"][0], dtype=float)
    x, y = _probe_on_the_blade(blade)
    t = entry["time"]
    with_sword = rigkit.clip_depth(ual, "Sword_Attack", fps=12, cell=CELL, frame=armed,
                                   times=[t]).frames()[0]
    bare = rigkit.clip_depth(ual, "Sword_Attack", fps=12, cell=CELL,
                             frame=rigkit.reference_frame(ual), times=[t]).frames()[0]
    window = (slice(y - 2, y + 3), slice(x - 2, x + 3))     # on the blade, near the tip
    assert with_sword[window].max() > 0
    assert bare[window].max() == 0


def test_depth_modes_are_normalised_from_flag_or_bool():
    assert rigkit.depth_mode(None) == rigkit.depth_mode(True) == "full"
    assert rigkit.depth_mode(False) == rigkit.depth_mode("none") == "none"
    assert rigkit.depth_mode(" Props ") == "props"
    with pytest.raises(ValueError, match="halberd"):
        rigkit.depth_mode("halberd")
    with pytest.raises(ValueError, match="none"):
        rigkit.clip_depth(rigkit.load_rig(UAL), "Idle_Loop", fps=12, cell=CELL,
                          frame=rigkit.reference_frame(rigkit.load_rig(UAL)), mode="none")


def test_props_only_depth_is_the_full_frame_minus_the_body(ual):
    """Same greys where the props are (one range for both modes, read off
    body + props), nothing where the body was, and the body still occludes."""
    frame = rigkit.reference_frame(ual, props=("sword", "shield"))
    kp = rigkit.clip_keypoints(ual, "Sword_Attack", fps=12, cell=CELL, frame=frame)
    full = rigkit.clip_depth(ual, "Sword_Attack", fps=12, cell=CELL, frame=frame)
    props = rigkit.clip_depth(ual, "Sword_Attack", fps=12, cell=CELL, frame=frame,
                              mode="props")
    assert props.mode == "props" and full.mode == "full"
    assert (props.far, props.near) == (full.far, full.near), "one range, both modes"
    entry = kp["frames"][10]
    x, y = _probe_on_the_blade(np.asarray(entry["props"][0]["polygons"][0], dtype=float))
    f10, p10 = full.frames()[10], props.frames()[10]
    assert p10[y, x] > 0 and p10[y, x] == f10[y, x], "the blade, same grey"
    hip = np.asarray(entry["keypoints"][kp["names"].index("right_hip")]).astype(int)
    assert f10[hip[1], hip[0]] > 0 and p10[hip[1], hip[0]] == 0, "the body is gone"
    assert 0 < (p10 > 0).sum() < 0.25 * (f10 > 0).sum()
    # Occluded by the body: every props pixel is a full-frame pixel of the
    # same grey, and on the rest frame the kite behind the torso is mostly
    # hidden - fewer pixels than the kite's own solid would cover.
    assert np.array_equal(p10[p10 > 0], f10[p10 > 0])
    f0, p0 = full.frames()[0], props.frames()[0]
    assert np.array_equal(p0[p0 > 0], f0[p0 > 0])
    unoccluded = rigkit.rasterize_depth(
        *_props_2d(ual, "Sword_Attack", 0.0, frame), CELL)
    assert (p0 > 0).sum() < 0.8 * np.isfinite(unoccluded).sum(), "the body hides part of the kite"


def _props_2d(rig, clip_name, t, frame):
    proj = rigkit.Projection.for_rig(frame, CELL)
    world = rig.pose_at(rig.clip(clip_name), t)
    joints = rigkit.openpose_points(rig, world, frame)
    pv, pt = rigkit.posed_mesh(rig, world, frame, body=False)
    return proj.to_2d(pv, anchor=proj.anchor_for(joints)), proj.depth_of(pv), pt


def test_props_only_depth_on_a_bare_unit_draws_nothing(ual):
    dc = rigkit.clip_depth(ual, "Idle_Loop", fps=12, cell=CELL,
                           frame=rigkit.reference_frame(ual), mode="props", times=[0.0])
    assert dc.frames()[0].max() == 0


# ---------------------------------------------------------------------------
# Direction - the camera direction feature
# ---------------------------------------------------------------------------
from anim import config  # noqa: E402

TWO_BONES = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # foot at the origin, toes forward
FORWARD_Z = np.array([0.0, 0.0, 1.0])


def _profile(yaw: int = 0, elevation: str = "side") -> rigkit.Projection:
    return rigkit.Projection(forward=FORWARD_Z, floor_y=0.0, scale=100.0,
                             cell=CELL, baseline=300.0, yaw=yaw, elevation=elevation)


def test_the_direction_table_is_two_pitches_by_eight_facings():
    assert len(config.DIRECTIONS) == 16
    assert list(config.DIRECTIONS)[:8] == ["s0", "s45", "s90", "s135", "s180", "s225", "s270", "s315"]
    assert list(config.DIRECTIONS)[8:] == ["t0", "t45", "t90", "t135", "t180", "t225", "t270", "t315"]
    assert config.DEFAULT_DIRECTION == "s0"


@pytest.mark.parametrize("name", list(config.DIRECTIONS))
def test_a_direction_round_trips_through_its_name(name):
    yaw, elevation = config.direction(name)
    assert config.direction_name(yaw, elevation) == name
    assert config.is_default_direction(yaw, elevation) == (name == "s0")


def test_a_yaw_off_the_table_is_refused_with_the_table():
    with pytest.raises(ValueError, match="45-degree"):
        config.direction_name(30)
    with pytest.raises(ValueError, match="elevation"):
        config.direction_name(0, "front")
    with pytest.raises(ValueError, match="unknown direction"):
        config.direction("x0")
    with pytest.raises(ValueError, match="45-degree"):
        rigkit.Projection(forward=FORWARD_Z, floor_y=0.0, scale=1.0, cell=CELL,
                          baseline=1.0, yaw=30)


def test_the_default_cell_is_the_profile_basis_to_the_bit():
    """Yaw 0 / side projects through exactly the axes the profile always used:
    forward, world up, forward x up - so an old job's bytes cannot move."""
    proj = _profile()
    assert proj.direction == "s0"
    assert np.array_equal(proj.rotation[0], FORWARD_Z)
    assert np.array_equal(proj.rotation[1], [0.0, 1.0, 0.0])
    assert np.array_equal(proj.rotation[2], np.cross(FORWARD_Z, [0.0, 1.0, 0.0]))
    assert np.array_equal(proj.near, np.cross(FORWARD_Z, [0.0, 1.0, 0.0]))


def test_yaw_180_is_the_profile_reflected_about_the_cell_centre():
    a = _profile(0).to_2d(TWO_BONES)
    b = _profile(180).to_2d(TWO_BONES)
    assert a[:, 0] + b[:, 0] == pytest.approx([CELL[0], CELL[0]], abs=1e-9)
    assert a[:, 1] == pytest.approx(b[:, 1])
    # And the same picture `mirror` makes: the reflected profile.
    mirrored = rigkit.Projection(forward=-FORWARD_Z, floor_y=0.0, scale=100.0,
                                 cell=CELL, baseline=300.0).to_2d(TWO_BONES)
    assert b == pytest.approx(mirrored, abs=1e-9)
    assert _profile(180).depth_of(TWO_BONES) == pytest.approx(-_profile(0).depth_of(TWO_BONES))


def test_yaw_90_puts_the_two_joints_on_one_column_facing_away():
    xy = _profile(90).to_2d(TWO_BONES)
    assert xy[0, 0] == pytest.approx(xy[1, 0]) == pytest.approx(CELL[0] / 2)
    assert xy[:, 1] == pytest.approx([300.0, 300.0]), "both on the floor"
    depth = _profile(90).depth_of(TWO_BONES)
    assert depth[1] < depth[0], "the toes point away from the viewer: the back view"
    front = _profile(270).depth_of(TWO_BONES)
    assert front[1] > front[0], "yaw 270 faces the viewer"


def test_yaw_45_foreshortens_by_cos_45():
    xy = _profile(45).to_2d(TWO_BONES, anchor=np.zeros(3))
    assert xy[1, 0] - xy[0, 0] == pytest.approx(100.0 * math.cos(math.radians(45)))


def test_the_top_view_puts_the_head_over_the_root_and_nearest():
    root_and_head = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    top = _profile(0, "top")
    xy = top.to_2d(root_and_head)
    assert xy[0] == pytest.approx(xy[1]), "one point from above"
    assert xy[0] == pytest.approx([CELL[0] / 2, CELL[1] / 2]), "centred, no floor row"
    depth = top.depth_of(root_and_head)
    assert depth[1] > depth[0], "the head is the nearest surface"
    assert top.near == pytest.approx([0.0, 1.0, 0.0])


def test_the_top_view_has_the_characters_left_up_screen():
    """Facing east (yaw 0), the left hand is to the north; at t90 (facing
    up-screen) it is to the west; at t270 (facing the viewer) to the east."""
    left = np.cross([0.0, 1.0, 0.0], FORWARD_Z)             # the character's LEFT
    pts = np.stack([np.zeros(3), left])
    for yaw, expect_dx, expect_dy in ((0, 0.0, -100.0), (90, -100.0, 0.0),
                                      (180, 0.0, 100.0), (270, 100.0, 0.0)):
        xy = _profile(yaw, "top").to_2d(pts, anchor=np.zeros(3))
        assert xy[1] - xy[0] == pytest.approx([expect_dx, expect_dy], abs=1e-9), yaw


def test_the_anchor_is_the_point_the_extent_would_centre_on():
    for yaw in (0, 45, 90, 135):
        for elevation in ("side", "top"):
            proj = _profile(yaw, elevation)
            pts = np.array([[0.3, 0.2, -0.4], [1.0, 0.9, 0.5], [-0.2, 0.1, 0.8]])
            anchored = proj.to_2d(pts, anchor=proj.anchor_for(pts))
            assert anchored == pytest.approx(proj.to_2d(pts)), (yaw, elevation)


def test_mirror_is_refused_with_any_other_cell(knight):
    with pytest.raises(ValueError, match="--yaw 180"):
        rigkit.reference_frame(knight, mirror=True, yaw=180)
    with pytest.raises(ValueError, match="mirror"):
        rigkit.reference_frame(knight, mirror=True, elevation="top")
    assert rigkit.reference_frame(knight, mirror=True).mirror is True, "alone it still works"


@pytest.mark.parametrize("name", list(config.DIRECTIONS))
def test_every_cell_keeps_the_rigs_scale_and_floor_row(knight, name):
    yaw, elevation = config.direction(name)
    plain = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL)
    kp = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL, yaw=yaw, elevation=elevation)
    assert kp["motion"]["direction"] == name
    assert (kp["motion"]["yaw"], kp["motion"]["elevation"]) == (yaw, elevation)
    assert kp["motion"]["scale_px_per_unit"] == plain["motion"]["scale_px_per_unit"]
    assert kp["motion"]["baseline_row"] == plain["motion"]["baseline_row"]
    assert kp["motion"]["floor_y"] == plain["motion"]["floor_y"]
    assert kp["motion"]["forward"] == plain["motion"]["forward"], "the true facing is recorded, not the view"
    if elevation == "side":
        # The floor is absolute in every profile: the ankles sit where s0 puts them.
        for joint in ("left_ankle", "right_ankle"):
            i = kp["names"].index(joint)
            assert kp["frames"][0]["keypoints"][i][1] == pytest.approx(
                plain["frames"][0]["keypoints"][i][1], abs=0.02)


def test_the_knights_yaw_180_is_its_mirror(knight):
    turned = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL, yaw=180)
    mirrored = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL, mirror=True)
    for a, b in zip(turned["frames"], mirrored["frames"]):
        assert np.asarray(a["keypoints"]) == pytest.approx(np.asarray(b["keypoints"]), abs=0.02)


@pytest.mark.parametrize("clip", ["Idle", "1H_Melee_Attack_Chop", "Death_A", "Lie_Down"])
@pytest.mark.parametrize("name", ["s45", "s90", "s270", "t0", "t90", "t225", "t270"])
def test_every_joint_stays_inside_the_cell_in_every_direction(knight, clip, name):
    """The top view centres both axes, so a body lying full-length at t270
    fits just as it does in profile - rtmlib drops any edge-touching limb."""
    yaw, elevation = config.direction(name)
    frame = rigkit.reference_frame(knight, yaw=yaw, elevation=elevation, props=("sword", "shield"))
    kp = rigkit.clip_keypoints(knight, clip, fps=12, cell=CELL, frame=frame)
    for entry in kp["frames"]:
        for x, y in entry["keypoints"]:
            assert 1 < x < CELL[0] - 1 and 1 < y < CELL[1] - 1, (clip, name, entry["index"], x, y)


def test_the_top_views_depth_range_is_the_standing_height_whatever_the_yaw(knight):
    """From above the depth axis is world up, so the range is the clip's
    vertical extent and identical in all eight top cells."""
    ranges = set()
    for yaw in config.DIRECTION_YAWS:
        frame = rigkit.reference_frame(knight, yaw=yaw, elevation="top")
        dc = rigkit.clip_depth(knight, "Idle", fps=12, cell=CELL, frame=frame, times=[0.0])
        ranges.add((round(dc.far, 4), round(dc.near, 4)))
        grey = dc.frames()[0]
        assert grey.max() == 255 and (grey > 0).any()
    assert len(ranges) == 1, ranges
    far, near = ranges.pop()
    # The skinned body only: the Knight's static helmet is not rendered, so
    # the span is a little under the rest-pose mesh height.
    assert 0.8 * frame.height < near - far <= frame.height + 0.05, "floor to the top of the body"


def test_the_shield_is_the_far_hand_in_profile_and_the_near_hand_turned_round(ual):
    """The viewer stands on the right at s0; at s180 on the left, so the
    shield hand is nearer. Depth and skeleton go through one rotation."""
    hand_r = ual.rest_world()[ual.index("hand_r")][:3, 3][None]
    hand_l = ual.rest_world()[ual.index("hand_l")][:3, 3][None]
    for yaw, nearer in ((0, "r"), (180, "l"), (270, None)):
        proj = rigkit.Projection.for_rig(rigkit.reference_frame(ual, yaw=yaw), CELL)
        r, l = proj.depth_of(hand_r)[0], proj.depth_of(hand_l)[0]
        if nearer == "r":
            assert r > l
        elif nearer == "l":
            assert l > r
        else:
            assert r == pytest.approx(l, abs=0.02), "facing the viewer, both hands level"


# ---------------------------------------------------------------------------
# The face on the far side (the turnaround feature)
# ---------------------------------------------------------------------------
FACE = ("nose", "right_eye", "left_eye", "right_ear", "left_ear")


def _scores(kp: dict, joints: tuple[str, ...] = FACE) -> dict[str, float]:
    entry = kp["frames"][0]
    return {j: entry["scores"][kp["names"].index(j)] for j in joints}


def test_the_two_bone_head_hides_its_face_as_it_turns_away(synthetic):
    """The Turn clip swings the head 90 degrees about +Y over one second: at
    t=0 it faces +z (the profile, viewer on its right), at t=0.5 it has
    turned 45 degrees away, at t=1 it faces straight away. The cell is s0 in
    every case - the HEAD moved, and the rule reads the posed head."""
    frame = rigkit.reference_frame(synthetic, TWO_BONE_MAP)
    proj = rigkit.Projection.for_rig(frame, CELL)
    clip = synthetic.clip("Turn")
    every = list(FACE)
    assert rigkit.hidden_face_points(synthetic, synthetic.pose_at(clip, 0.0), frame, proj.near) == []
    assert rigkit.hidden_face_points(synthetic, synthetic.pose_at(clip, 0.5), frame, proj.near) \
        == every, "45 away: every face point, the ears included (the multi-view reference feature)"
    assert rigkit.hidden_face_points(synthetic, synthetic.pose_at(clip, 1.0), frame, proj.near) \
        == every, "straight away: no face point at all"
    # Half a direction step is the threshold: 20 degrees is still the profile.
    assert rigkit.hidden_face_points(synthetic, synthetic.pose_at(clip, 20 / 90), frame,
                                     proj.near) == []
    assert rigkit.hidden_face_points(synthetic, synthetic.pose_at(clip, 25 / 90), frame,
                                     proj.near) != []
    kp = rigkit.clip_keypoints(synthetic, "Turn", fps=2, cell=CELL, frame=frame)
    assert "hidden" not in kp["frames"][0] and all(s == 1.0 for s in kp["frames"][0]["scores"])
    assert kp["frames"][1]["hidden"] == every
    assert _scores(kp | {"frames": kp["frames"][1:]}) == {j: 0.0 for j in FACE}
    assert len(kp["frames"][1]["keypoints"]) == 18, "the points are still there, only unseen"


@pytest.mark.parametrize("name", ["s0", "s180", "s225", "s270", "s315", "t0", "t90", "t270"])
def test_the_profile_and_the_front_cells_keep_every_face_point(knight, name):
    yaw, elevation = config.direction(name)
    kp = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL, yaw=yaw, elevation=elevation)
    for entry in kp["frames"]:
        assert "hidden" not in entry, (name, entry["index"])
        assert all(s == 1.0 for s in entry["scores"])


@pytest.mark.parametrize("name", ["s0", "s180", "s225", "s270", "s315"])
def test_the_profile_and_front_cells_are_byte_identical_to_a_rig_with_no_face_rule(
        knight, name, monkeypatch):
    """The multi-view reference feature tightened the rule for the turned-away
    cells only: the payload `rig` writes for s0 (every published job's cell),
    s180 and the front cells is the same JSON, byte for byte, as a rig that
    hides nothing - so an existing s0 job re-rigged today reproduces."""
    yaw, elevation = config.direction(name)
    kp = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL, yaw=yaw, elevation=elevation)
    monkeypatch.setattr(rigkit, "hidden_face_points", lambda *a, **k: [])
    bare = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL, yaw=yaw, elevation=elevation)
    assert json.dumps(kp, sort_keys=True) == json.dumps(bare, sort_keys=True)


@pytest.mark.parametrize("name, expect", [
    ("s45", list(FACE)),
    ("s90", list(FACE)),
    ("s135", list(FACE)),
])
def test_the_back_cells_hide_every_face_point(knight, name, expect):
    """Turned away past the profile - s45, s90, s135 - no face point is
    drawn: nose, both eyes AND both ears (the turnaround feature kept the
    near ear and, from behind, both; the multi-view reference feature hides
    them too, an ear being a landmark the pose ControlNet builds a face on)."""
    yaw, elevation = config.direction(name)
    kp = rigkit.clip_keypoints(knight, "Idle", fps=12, cell=CELL, yaw=yaw, elevation=elevation)
    for entry in kp["frames"]:
        assert entry["hidden"] == expect, (name, entry["index"])
    assert _scores(kp) == {j: (0.0 if j in expect else 1.0) for j in FACE}
    # And the skeleton PNG no longer draws them: fewer lit pixels than with
    # every score forced back to 1.0.
    entry = kp["frames"][0]
    drawn = posekit.draw(entry["keypoints"], entry["scores"], tuple(kp["source_size"]), CELL)
    forced = posekit.draw(entry["keypoints"], [1.0] * 18, tuple(kp["source_size"]), CELL)
    assert (drawn.max(axis=2) > 16).sum() < (forced.max(axis=2) > 16).sum()
