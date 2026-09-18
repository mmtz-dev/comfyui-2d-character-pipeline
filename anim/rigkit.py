"""Rig clips -> OpenPose-18 keypoints. The rig motion input feature.

A rigged, animated glTF character (KayKit's CC0 Knight, first) replaces the
filmed clip as the pipeline's motion source: every clip embedded in the GLB is
sampled at N fps, each joint's world position is worked out through the node
hierarchy, thirteen of those joints plus five synthesised face points are
mapped onto the 18-point OpenPose order the rest of the pipeline speaks, and
the result is projected orthographically in profile into the generation cell -
the same `anim.keypoints/1` payload `anim pose` writes, so `pick`, `review`,
`strip` and everything after them run unchanged.

Pure functions over numpy. pygltflib only parses the container and the
accessors; sampling, interpolation, the hierarchy and the projection are here,
because they are where the decisions live and where the tests point.

THE JOINT MAP - OpenPose-18 <- rig bones
----------------------------------------
OpenPose-18 (rtmlib's `openpose18`, the order `anim pose` produces):

    0 nose        4 r_wrist     8  r_hip     12 l_knee     16 r_ear
    1 neck        5 l_shoulder  9  r_knee    13 l_ankle    17 l_ear
    2 r_shoulder  6 l_elbow     10 r_ankle   14 r_eye
    3 r_elbow     7 l_wrist     11 l_hip     15 l_eye

KayKit rigs (Adventurers, Skeletons - one 41-bone rig with custom names, NO neck
or shoulder bone) map as follows. A bone name means "that bone's origin"; a
`midpoint(a, b)` means the mean of two origins.

    | OpenPose joint      | KayKit bone                        |
    |---------------------|------------------------------------|
    | neck                | midpoint(upperarm.l, upperarm.r)   |
    | r/l shoulder        | upperarm.r / upperarm.l            |
    | r/l elbow           | lowerarm.r / lowerarm.l            |
    | r/l wrist           | wrist.r / wrist.l                  |
    | r/l hip             | upperleg.r / upperleg.l            |
    | r/l knee            | lowerleg.r / lowerleg.l            |
    | r/l ankle           | foot.r / foot.l                    |
    | nose, eyes, ears    | synthesised from the `head` bone   |

WHY THE NECK IS THE SHOULDER MIDPOINT and not the `head` origin (the chest tip):
rtmlib's own `convert_coco_to_openpose` defines the OpenPose neck as
`np.mean(keypoints[:, [5, 6]])` - the mean of the two shoulders - so every
DWPose skeleton the ControlNet was trained on has its neck exactly on the
shoulder line. Putting it at the head origin (0.13 rig units higher on the
Knight) would draw a neck the ControlNet never saw. The `head` origin is still
used, as the base of the head for the face points.

THE FACE POINTS: no rig has a nose bone. As `mixamotoopenpose` does, the five
face joints are offsets in the head bone's own frame - up along the head bone,
forward along the character's facing, lateral for the pairs - scaled by the
HEAD LENGTH, which is measured from the head origin to the top of the mesh in
the rest pose (the helmet top on the Knight), so a chibi head and a realistic
head both get their nose halfway up the face. The offsets are part of the map
(`face`), in head lengths, as (up, forward, lateral); the right-hand point of
each pair uses -lateral.

WHY IT IS A TABLE: `RIGS` holds one `JointMap` per rig family, chosen by which
bone names are present (`detect_map`). KayKit Skeletons share the Adventurers
rig, so the same map serves hero and barrow-dead; Quaternius' Universal
Animation Library (the Unreal-Godot GLB in the 2026 Standard zip) is an
Unreal-mannequin rig - `pelvis, spine_01, neck_01, Head, clavicle_l,
upperarm_l, lowerarm_l, hand_l, thigh_l, calf_l, foot_l, ball_l` - with a real
neck, and is a second row. (The research doc's `DEF-*` Rigify names came from
a 2025 mirror and are not what the current zip ships.) Adding a rig is adding
a row, not code.

FACING - detected, never assumed
--------------------------------
glTF says nothing about which way a character faces. Blender exports usually
face +Z, but "usually" is how a whole strip comes out facing left. So the
direction is read off the rig itself: in the rest pose, the horizontal component
of the foot->toes vector (both feet, summed - `forward_hint` in the map) is
"forward". The profile projection then puts forward on +x, which is the art
bible's rule: the party faces RIGHT; enemies are the same art with `flip_h`.
`mirror=True` flips the PROJECTION only - the face points stay on the
character's real front - for a rig whose toes lie or a clip that should read
the other way.

SCALE - per rig, never per clip
-------------------------------
One scale for every clip of a rig, taken from the rest pose: the model's
standing height (mesh floor to mesh top) maps to `HEIGHT_FRACTION` of the cell
height, with the floor on a fixed baseline row. `hit` and `death` therefore
cannot come out a different size from `strike`, and a body lying on the floor
lies on the same row the standing feet stood on. Horizontally each frame is
centred on its own extent: the choreography moves the token, the frames do not
translate, and neither a lunge nor a chibi head on a body lying full-length
can walk out of the cell (rtmlib drops any limb that touches the edge).

PROFILE AND THE FAR SIDE: a true profile puts the far-side limbs behind the
near-side ones, and both project to nearly the same 2D points. Both are kept -
the renderer draws both, as DWPose does on a real profile shot - and every
joint carries confidence 1.0, because a rig has no detection uncertainty.

PROPS - the skeleton props feature
----------------------------------
A stick figure says where the wrists are, not what they hold. Asked for "a
sword and a kite shield", the ControlNet put the sword in whichever hand looked
like a sword hand - both of them, on the recovery frame of `Sword_Attack` -
and no prompt fixed it (DEVLOG 2026-09-16). So the props are drawn onto the
skeleton: a blade off the right hand, a kite on the left forearm, as white
outlines the ControlNet reads like canny edges. They are computed in 3D and
projected with the joints, so a swing that crosses the body foreshortens the
blade exactly as it foreshortens the arm.

Each prop rides a bone (`PROPS`: the sword the right-wrist bone, the shield the
left-elbow bone, i.e. the forearm) and is laid out in that bone's own frame at
REST under one convention: the rest pose is a T- or A-pose with the palms down,
so a gripped blade points FORWARD out of the fist (the thumb side), its guard
runs along the fingers, and the back of the forearm - where the shield is
strapped - faces UP. Sizes are fractions of the standing height, per rig, never
per clip. `prop_polygons()` returns the outlines in world space; `clip_keypoints`
writes them per frame as `props: [{kind, polygons: [[[x, y], ...], ...]}]` in
cell pixels, and `anim.posekit.draw` strokes them. A job without props writes
no `props` key and renders exactly as before.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# OpenPose-18, spelled out so this module needs no rtmlib. tests/test_rigkit.py
# asserts it equals rtmlib's openpose18 table.
# ---------------------------------------------------------------------------
OPENPOSE18: tuple[str, ...] = (
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
)
FACE_JOINTS = ("nose", "right_eye", "left_eye", "right_ear", "left_ear")
# Limb pairs, for the preview renderer only. The ControlNet image is always
# drawn by rtmlib (anim.posekit.draw); this list just has to look like a person.
LIMBS: tuple[tuple[str, str], ...] = (
    ("neck", "right_shoulder"), ("neck", "left_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("neck", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("neck", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("neck", "nose"), ("nose", "right_eye"), ("right_eye", "right_ear"),
    ("nose", "left_eye"), ("left_eye", "left_ear"),
)

# The model's standing height as a fraction of the cell height, and the row the
# floor sits on. 0.78 leaves room for arms raised over the head and keeps a body
# lying full-length inside a 304-wide cell (250 px of 304); rtmlib drops any
# limb that touches the frame edge, so margin is not cosmetic.
HEIGHT_FRACTION = 0.78
BASELINE_FRACTION = 0.9375
# When a GLB carries no mesh (a bare animation library, or the synthetic rig in
# the tests) the head length is this many neck-to-head distances instead.
HEAD_LENGTH_FALLBACK = 1.5

# ---------------------------------------------------------------------------
# Props - sizes as fractions of the standing height (a 1.83 m mannequin gets a
# 0.85 m blade; a chibi gets one that fits a chibi)
# ---------------------------------------------------------------------------
PROP_SIZES: dict[str, dict[str, float]] = {
    "sword": {"blade": 0.465, "pommel": 0.066, "guard": 0.120,
              "blade_width": 0.027, "guard_width": 0.019},
    # A kite: flat top toward the elbow, point toward the hand, standing a
    # little off the back of the forearm.
    "shield": {"length": 0.44, "width": 0.23, "offset": 0.055},
}
# Which OpenPose joint's bone each prop rides. Both are single bones in every
# map (the midpoint joints are the neck only).
PROP_JOINT: dict[str, str] = {"sword": "right_wrist", "shield": "left_elbow"}
PROP_KINDS: tuple[str, ...] = tuple(PROP_SIZES)


# ---------------------------------------------------------------------------
# The joint map table
# ---------------------------------------------------------------------------
Midpoint = tuple[str, str]


@dataclass(frozen=True)
class JointMap:
    """How one rig family's bones become OpenPose-18 joints."""

    name: str
    # OpenPose joint -> bone name, or a (bone_a, bone_b) pair meaning midpoint.
    joints: dict[str, str | Midpoint]
    head: str
    # (from_bone, to_bone) pairs whose summed horizontal direction is "forward".
    forward_hint: tuple[tuple[str, str], ...]
    # Face offsets in head lengths: joint -> (up, forward, lateral). The
    # right-hand member of a pair uses -lateral.
    face: dict[str, tuple[float, float, float]]
    # Bones that must all be present for detect_map() to choose this row.
    detect: tuple[str, ...]
    # Human-readable, for docs and job.json.
    note: str = ""

    def bones(self) -> set[str]:
        out: set[str] = {self.head}
        for spec in self.joints.values():
            out.update(spec if isinstance(spec, tuple) else (spec,))
        for a, b in self.forward_hint:
            out.update((a, b))
        return out

    def missing(self, node_names: Iterable[str]) -> list[str]:
        have = set(node_names)
        return sorted(b for b in self.bones() if b not in have)


KAYKIT = JointMap(
    name="kaykit",
    joints={
        "neck": ("upperarm.l", "upperarm.r"),
        "right_shoulder": "upperarm.r", "right_elbow": "lowerarm.r",
        "right_wrist": "wrist.r",
        "left_shoulder": "upperarm.l", "left_elbow": "lowerarm.l",
        "left_wrist": "wrist.l",
        "right_hip": "upperleg.r", "right_knee": "lowerleg.r",
        "right_ankle": "foot.r",
        "left_hip": "upperleg.l", "left_knee": "lowerleg.l",
        "left_ankle": "foot.l",
    },
    head="head",
    forward_hint=(("foot.l", "toes.l"), ("foot.r", "toes.r")),
    # Chibi: the head is ~45% of the standing height and wears a helmet, so the
    # face sits low on the head bone and well forward of its origin.
    face={
        "nose": (0.48, 0.42, 0.0),
        "right_eye": (0.58, 0.34, -0.16), "left_eye": (0.58, 0.34, 0.16),
        "right_ear": (0.55, 0.02, -0.40), "left_ear": (0.55, 0.02, 0.40),
    },
    detect=("hips", "chest", "head", "upperarm.l", "lowerarm.l", "wrist.l",
            "upperleg.l", "lowerleg.l", "foot.l", "toes.l"),
    note="KayKit Adventurers / Skeletons: one 41-bone rig, custom names, no "
         "neck or shoulder bone.",
)

QUATERNIUS_UAL = JointMap(
    name="quaternius-ual",
    joints={
        # A real neck bone (neck_01) exists, but the OpenPose neck is still the
        # shoulder midpoint - see the module docstring. The clavicles sit at the
        # sternum; the upperarm origins are the shoulder joints.
        "neck": ("upperarm_l", "upperarm_r"),
        "right_shoulder": "upperarm_r", "right_elbow": "lowerarm_r",
        "right_wrist": "hand_r",
        "left_shoulder": "upperarm_l", "left_elbow": "lowerarm_l",
        "left_wrist": "hand_l",
        "right_hip": "thigh_r", "right_knee": "calf_r", "right_ankle": "foot_r",
        "left_hip": "thigh_l", "left_knee": "calf_l", "left_ankle": "foot_l",
    },
    head="Head",
    forward_hint=(("foot_l", "ball_l"), ("foot_r", "ball_r")),
    # Realistic ~7-head proportions: the face fills the front of the head.
    face={
        "nose": (0.55, 0.50, 0.0),
        "right_eye": (0.68, 0.42, -0.18), "left_eye": (0.68, 0.42, 0.18),
        "right_ear": (0.62, 0.05, -0.45), "left_ear": (0.62, 0.05, 0.45),
    },
    detect=("pelvis", "spine_01", "neck_01", "Head", "clavicle_l", "upperarm_l",
            "lowerarm_l", "hand_l", "thigh_l", "calf_l", "foot_l", "ball_l"),
    note="Quaternius Universal Animation Library (Standard, Unreal-Godot GLB): "
         "Unreal-mannequin bone names, realistic proportions. The "
         "realistic-skeleton A/B for the anchor gate.",
)

RIGS: dict[str, JointMap] = {m.name: m for m in (KAYKIT, QUATERNIUS_UAL)}


def detect_map(node_names: Iterable[str]) -> JointMap:
    """The JointMap whose `detect` bones are all present, else a clear error."""
    have = set(node_names)
    for jmap in RIGS.values():
        if all(b in have for b in jmap.detect):
            return jmap
    known = ", ".join(f"{m.name} (needs {m.detect[0]}...)" for m in RIGS.values())
    raise ValueError(
        f"no joint map matches this rig's bones. Known rigs: {known}. Add a "
        f"JointMap row to anim/rigkit.py for this skeleton. Bones seen: "
        f"{sorted(have)[:12]}...")


# ---------------------------------------------------------------------------
# Quaternions and matrices (glTF: quaternion is [x, y, z, w]; matrices are
# column-major)
# ---------------------------------------------------------------------------
def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def trs_matrix(t: Sequence[float], r: Sequence[float], s: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(r) * np.asarray(s, dtype=float)[None, :]
    m[:3, 3] = np.asarray(t, dtype=float)
    return m


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Shortest-arc spherical interpolation, as the glTF spec requires."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = q0 + u * (q1 - q0)
        return out / (np.linalg.norm(out) or 1.0)
    theta0 = math.acos(min(1.0, d))
    theta = theta0 * u
    sin0 = math.sin(theta0)
    s0 = math.cos(theta) - d * math.sin(theta) / sin0
    s1 = math.sin(theta) / sin0
    return s0 * q0 + s1 * q1


def rotation_part(world: np.ndarray) -> np.ndarray:
    """The 3x3 rotation of a world matrix with any scale divided out."""
    r = world[:3, :3].copy()
    for col in range(3):
        n = np.linalg.norm(r[:, col])
        if n > 1e-12:
            r[:, col] /= n
    return r


# ---------------------------------------------------------------------------
# Channel sampling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Channel:
    node: int
    path: str                  # translation | rotation | scale
    interpolation: str         # LINEAR | STEP | CUBICSPLINE
    times: np.ndarray          # (K,)
    values: np.ndarray         # (K, C) or (K, 3, C) for CUBICSPLINE


def sample_channel(channel: Channel, t: float) -> np.ndarray:
    """The channel's value at time t. Clamped outside the keyframe range.

    LINEAR is lerp for translation/scale and slerp for rotation; STEP holds the
    previous key; CUBICSPLINE is the spec's cubic Hermite over (in-tangent,
    value, out-tangent) triples, tangents scaled by the segment length.
    """
    times, values = channel.times, channel.values
    cubic = channel.interpolation == "CUBICSPLINE"

    def value_at(k: int) -> np.ndarray:
        return values[k, 1] if cubic else values[k]

    if len(times) == 1 or t <= times[0]:
        return value_at(0)
    if t >= times[-1]:
        return value_at(len(times) - 1)
    i = int(np.searchsorted(times, t, side="right")) - 1
    t0, t1 = float(times[i]), float(times[i + 1])
    dt = t1 - t0
    u = 0.0 if dt <= 0 else (t - t0) / dt

    if channel.interpolation == "STEP":
        return value_at(i)
    if cubic:
        p0, m0 = values[i, 1], values[i, 2] * dt
        p1, m1 = values[i + 1, 1], values[i + 1, 0] * dt
        u2, u3 = u * u, u * u * u
        out = ((2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0
               + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1)
        if channel.path == "rotation":
            out = out / (np.linalg.norm(out) or 1.0)
        return out
    if channel.path == "rotation":
        return slerp(values[i], values[i + 1], u)
    return (1.0 - u) * values[i] + u * values[i + 1]


# ---------------------------------------------------------------------------
# The rig: nodes, hierarchy, clips
# ---------------------------------------------------------------------------
@dataclass
class Clip:
    name: str
    channels: list[Channel]

    @property
    def duration(self) -> float:
        return max((float(c.times[-1]) for c in self.channels), default=0.0)

    @property
    def keyframes(self) -> int:
        return max((len(c.times) for c in self.channels), default=0)

    def animated_nodes(self) -> set[int]:
        return {c.node for c in self.channels}


@dataclass
class Rig:
    path: Path
    names: list[str]                 # node index -> name
    parents: list[int | None]
    local: list[np.ndarray]          # static local matrix per node (rest pose)
    trs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    order: list[int]                 # parents before children
    clips: dict[str, Clip]
    # Mesh vertex extent in the rest pose, world space: (min_xyz, max_xyz), or
    # None when the file has no mesh at all.
    mesh_bounds: tuple[np.ndarray, np.ndarray] | None = None
    generator: str = ""

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"rig has no node named {name!r}") from exc

    def clip(self, name: str) -> Clip:
        if name not in self.clips:
            close = [c for c in self.clips if name.lower() in c.lower()]
            hint = f" Did you mean {close}?" if close else ""
            raise KeyError(f"{self.path.name} has no clip {name!r}.{hint} "
                           f"`anim rig --list` prints them all.")
        return self.clips[name]

    # --- poses -------------------------------------------------------------
    def world_matrices(self, local: Sequence[np.ndarray]) -> np.ndarray:
        """Compose local matrices down the hierarchy. (N, 4, 4)."""
        world = np.empty((len(self.names), 4, 4))
        for i in self.order:
            p = self.parents[i]
            world[i] = local[i] if p is None else world[p] @ local[i]
        return world

    def rest_world(self) -> np.ndarray:
        return self.world_matrices(self.local)

    def pose_at(self, clip: Clip, t: float) -> np.ndarray:
        """World matrices for every node at time t of a clip. (N, 4, 4)."""
        trs = {i: [np.array(tr), np.array(ro), np.array(sc)]
               for i, (tr, ro, sc) in enumerate(self.trs)}
        for channel in clip.channels:
            slot = {"translation": 0, "rotation": 1, "scale": 2}[channel.path]
            trs[channel.node][slot] = sample_channel(channel, t)
        local = [trs_matrix(*trs[i]) if i in trs else self.local[i]
                 for i in range(len(self.names))]
        return self.world_matrices(local)

    def sample(self, clip: Clip, times: Sequence[float]) -> np.ndarray:
        """(T, N, 4, 4) world matrices, one pose per requested time."""
        return np.stack([self.pose_at(clip, float(t)) for t in times])


# ---------------------------------------------------------------------------
# Loading with pygltflib
# ---------------------------------------------------------------------------
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9,
          "MAT4": 16}


def _buffer_bytes(gltf: Any, path: Path, index: int) -> bytes:
    buffer = gltf.buffers[index]
    uri = buffer.uri
    if uri is None:
        blob = gltf.binary_blob()
        if blob is None:
            raise ValueError(f"{path}: buffer {index} has no data")
        return blob
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    return (path.parent / uri).read_bytes()


def read_accessor(gltf: Any, path: Path, index: int,
                  buffers: dict[int, bytes]) -> np.ndarray:
    """An accessor as a float array, (count, components). Strided views honoured."""
    acc = gltf.accessors[index]
    if acc.sparse is not None:
        raise ValueError(f"{path}: sparse accessor {index} is not supported")
    if acc.bufferView is None:
        return np.zeros((acc.count, _COUNT[acc.type]), dtype=float)
    view = gltf.bufferViews[acc.bufferView]
    if view.buffer not in buffers:
        buffers[view.buffer] = _buffer_bytes(gltf, path, view.buffer)
    data = buffers[view.buffer]
    dtype = np.dtype(_COMPONENT[acc.componentType])
    width = _COUNT[acc.type]
    offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
    packed = dtype.itemsize * width
    stride = view.byteStride or packed
    if stride == packed:
        arr = np.frombuffer(data, dtype=dtype, count=acc.count * width,
                            offset=offset).reshape(acc.count, width)
    else:
        raw = np.frombuffer(data, dtype=np.uint8,
                            count=stride * (acc.count - 1) + packed, offset=offset)
        rows = np.lib.stride_tricks.as_strided(
            raw, shape=(acc.count, packed), strides=(stride, 1))
        arr = np.ascontiguousarray(rows).view(dtype).reshape(acc.count, width)
    out = arr.astype(float)
    if acc.normalized and dtype.kind in "iu":
        out = out / float(np.iinfo(dtype).max)
    return out


def _node_local(node: Any) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if node.matrix:
        m = np.array(node.matrix, dtype=float).reshape(4, 4, order="F")
        t = m[:3, 3].copy()
        s = np.linalg.norm(m[:3, :3], axis=0)
        r3 = m[:3, :3] / np.where(s > 1e-12, s, 1.0)[None, :]
        return m, (t, _matrix_to_quat(r3), s)
    t = np.array(node.translation or [0.0, 0.0, 0.0], dtype=float)
    r = np.array(node.rotation or [0.0, 0.0, 0.0, 1.0], dtype=float)
    s = np.array(node.scale or [1.0, 1.0, 1.0], dtype=float)
    return trs_matrix(t, r, s), (t, r, s)


def _matrix_to_quat(m: np.ndarray) -> np.ndarray:
    tr = float(np.trace(m))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q


def _topological_order(parents: Sequence[int | None]) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()

    def visit(i: int) -> None:
        if i in seen:
            return
        p = parents[i]
        if p is not None:
            visit(p)
        seen.add(i)
        order.append(i)

    for i in range(len(parents)):
        visit(i)
    return order


def load_rig(path: Path) -> Rig:
    """Parse a .glb/.gltf into a Rig. Meshes are read only for their extent."""
    import pygltflib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no rig at {path}")
    gltf = pygltflib.GLTF2().load(str(path))
    buffers: dict[int, bytes] = {}

    names = [n.name or f"node{i}" for i, n in enumerate(gltf.nodes)]
    parents: list[int | None] = [None] * len(gltf.nodes)
    for i, node in enumerate(gltf.nodes):
        for child in node.children or []:
            parents[child] = i
    local, trs = [], []
    for node in gltf.nodes:
        m, parts = _node_local(node)
        local.append(m)
        trs.append(parts)

    clips: dict[str, Clip] = {}
    for anim in gltf.animations:
        channels: list[Channel] = []
        for ch in anim.channels:
            if ch.target.node is None or ch.target.path not in (
                    "translation", "rotation", "scale"):
                continue
            sampler = anim.samplers[ch.sampler]
            times = read_accessor(gltf, path, sampler.input, buffers)[:, 0]
            values = read_accessor(gltf, path, sampler.output, buffers)
            interp = sampler.interpolation or "LINEAR"
            if interp == "CUBICSPLINE":
                values = values.reshape(len(times), 3, -1)
            channels.append(Channel(node=ch.target.node, path=ch.target.path,
                                    interpolation=interp, times=times,
                                    values=values))
        name = anim.name or f"clip{len(clips)}"
        clips[name] = Clip(name=name, channels=channels)

    rig = Rig(path=path, names=names, parents=parents, local=local, trs=trs,
              order=_topological_order(parents), clips=clips,
              generator=(gltf.asset.generator or "") if gltf.asset else "")
    rig.mesh_bounds = _mesh_bounds(gltf, path, rig, buffers)
    return rig


def _mesh_bounds(gltf: Any, path: Path, rig: Rig,
                 buffers: dict[int, bytes]) -> tuple[np.ndarray, np.ndarray] | None:
    """Rest-pose extent of every mesh, world space.

    A skinned mesh's vertices are, by the definition of the bind pose, already
    in world space (joint transform x inverse bind matrix == identity there), so
    they are taken as-is. A static mesh parented to a bone - the Knight's helmet
    and sword - is carried through its node's rest world matrix.
    """
    world = rig.rest_world()
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    found = False
    for i, node in enumerate(gltf.nodes):
        if node.mesh is None:
            continue
        for prim in gltf.meshes[node.mesh].primitives:
            pos_index = getattr(prim.attributes, "POSITION", None)
            if pos_index is None:
                continue
            verts = read_accessor(gltf, path, pos_index, buffers)[:, :3]
            if node.skin is None:
                verts = verts @ world[i][:3, :3].T + world[i][:3, 3]
            lo = np.minimum(lo, verts.min(axis=0))
            hi = np.maximum(hi, verts.max(axis=0))
            found = True
    return (lo, hi) if found else None


def list_clips(rig: Rig) -> list[dict[str, Any]]:
    return [{"name": c.name, "duration": round(c.duration, 4),
             "keyframes": c.keyframes, "nodes": len(c.animated_nodes())}
            for c in rig.clips.values()]


# ---------------------------------------------------------------------------
# Rig facts: facing, floor, height, head length
# ---------------------------------------------------------------------------
def positions_by_name(rig: Rig, world: np.ndarray) -> dict[str, np.ndarray]:
    return {name: world[i][:3, 3] for i, name in enumerate(rig.names)}


def facing_direction(positions: dict[str, np.ndarray],
                     hint: Sequence[tuple[str, str]]) -> np.ndarray:
    """Unit horizontal vector the character faces, from foot->toes-style pairs."""
    total = np.zeros(3)
    for a, b in hint:
        if a not in positions or b not in positions:
            raise KeyError(f"facing hint needs bones {a!r} and {b!r}")
        total += positions[b] - positions[a]
    total[1] = 0.0
    norm = float(np.linalg.norm(total))
    if norm < 1e-9:
        raise ValueError(
            "cannot detect facing: the forward-hint bones lie on a vertical "
            "line in the rest pose. Pass --mirror/--forward to set it by hand.")
    return total / norm


def resolve_joint(spec: str | Midpoint, positions: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(spec, tuple):
        return (positions[spec[0]] + positions[spec[1]]) / 2.0
    return positions[spec]


@dataclass(frozen=True)
class PropFrame:
    """One prop's attachment: the bone it rides and its axes in that bone's
    rest frame, so `bone_rotation @ axis` is the axis in any pose."""

    kind: str
    bone: str
    axes: dict[str, np.ndarray]
    # The shield sits on the forearm's midpoint, which needs the wrist bone too.
    end_bone: str | None = None


@dataclass(frozen=True)
class RigFrame:
    """Everything about a rig that is fixed across its clips."""

    map: JointMap
    forward: np.ndarray          # unit, horizontal, world space: the TRUE facing
    mirror: bool                 # project facing left instead of right
    floor_y: float
    top_y: float
    head_length: float
    head_forward_local: np.ndarray   # forward expressed in the head bone's frame
    head_up_local: np.ndarray
    props: tuple[PropFrame, ...] = ()

    @property
    def height(self) -> float:
        return self.top_y - self.floor_y


def prop_frame(kind: str, rig: Rig, world: np.ndarray, jmap: JointMap,
               forward: np.ndarray) -> PropFrame:
    """Attach one prop under the palms-down rest convention (module docstring)."""
    if kind not in PROP_SIZES:
        raise ValueError(f"unknown prop {kind!r}; known: {list(PROP_SIZES)}")
    bone = jmap.joints[PROP_JOINT[kind]]
    if isinstance(bone, tuple):
        raise ValueError(f"prop {kind!r} needs a single bone for "
                         f"{PROP_JOINT[kind]}, not a midpoint")
    up = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(up, forward)                  # the character's LEFT
    r_bone = rotation_part(world[rig.index(bone)])
    pos = positions_by_name(rig, world)
    if kind == "sword":
        # Blade out of the fist along forward; the guard along the fingers,
        # which in a T/A-pose is the arm's own direction, elbow -> wrist.
        elbow = jmap.joints["right_elbow"]
        along = pos[bone] - pos[elbow]
        along[1] = 0.0
        along = along / (np.linalg.norm(along) or 1.0)
        return PropFrame(kind, bone, {"blade": r_bone.T @ forward,
                                      "guard": r_bone.T @ along})
    wrist = jmap.joints["left_wrist"]
    axis = pos[wrist] - pos[bone]
    axis = axis / (np.linalg.norm(axis) or 1.0)
    return PropFrame(kind, bone, {"axis": r_bone.T @ axis,
                                  "normal": r_bone.T @ up,
                                  "width": r_bone.T @ np.cross(up, axis)},
                     end_bone=wrist)


def reference_frame(rig: Rig, jmap: JointMap | None = None,
                    mirror: bool = False,
                    props: Sequence[str] = ()) -> RigFrame:
    """Facing, floor, standing height and head length, all from the rest pose.

    `props` names the props to attach (`PROP_KINDS`); they are laid out in the
    same rest pose the facing is read from.
    """
    jmap = jmap or detect_map(rig.names)
    missing = jmap.missing(rig.names)
    if missing:
        raise ValueError(f"rig {rig.path.name} matched map {jmap.name!r} but "
                         f"lacks bones {missing}")
    world = rig.rest_world()
    pos = positions_by_name(rig, world)
    forward = facing_direction(pos, jmap.forward_hint)

    head = pos[jmap.head]
    joints = np.stack([resolve_joint(jmap.joints[j], pos)
                       for j in OPENPOSE18 if j in jmap.joints])
    if rig.mesh_bounds is not None:
        lo, hi = rig.mesh_bounds
        floor_y, top_y = float(lo[1]), float(hi[1])
    else:
        neck = resolve_joint(jmap.joints["neck"], pos)
        head_len = HEAD_LENGTH_FALLBACK * float(np.linalg.norm(head - neck))
        floor_y = float(min(joints[:, 1].min(), head[1]))
        top_y = float(head[1] + head_len)
    head_length = max(top_y - float(head[1]), 1e-6)

    r_head = rotation_part(world[rig.index(jmap.head)])
    up = np.array([0.0, 1.0, 0.0])
    return RigFrame(map=jmap, forward=forward, mirror=bool(mirror),
                    floor_y=floor_y, top_y=top_y, head_length=head_length,
                    head_forward_local=r_head.T @ forward,
                    head_up_local=r_head.T @ up,
                    props=tuple(prop_frame(k, rig, world, jmap, forward)
                                for k in props))


def prop_polygons(rig: Rig, world: np.ndarray, frame: RigFrame
                  ) -> list[tuple[str, list[np.ndarray]]]:
    """[(kind, [(N, 3) closed outlines in world space, ...]), ...] for one pose.

    A sword is two outlines - the blade (with its tip) and the crossguard - and
    a shield is one kite. Sizes come from `PROP_SIZES` x the standing height.
    """
    out: list[tuple[str, list[np.ndarray]]] = []
    for prop in frame.props:
        size = {k: v * frame.height for k, v in PROP_SIZES[prop.kind].items()}
        bone_world = world[rig.index(prop.bone)]
        r_bone = rotation_part(bone_world)
        origin = bone_world[:3, 3]
        ax = {k: r_bone @ v for k, v in prop.axes.items()}
        if prop.kind == "sword":
            blade, guard = ax["blade"], ax["guard"]
            half = guard * size["blade_width"] / 2.0
            tip = origin + blade * size["blade"]
            shoulder = origin + blade * (size["blade"] - size["blade_width"] * 1.5)
            pommel = origin - blade * size["pommel"]
            gh = guard * size["guard"] / 2.0
            gt = blade * size["guard_width"] / 2.0
            out.append((prop.kind, [
                np.stack([pommel + half, shoulder + half, tip, shoulder - half, pommel - half]),
                np.stack([origin - gh + gt, origin + gh + gt, origin + gh - gt, origin - gh - gt]),
            ]))
        else:
            end = world[rig.index(prop.end_bone)][:3, 3]
            centre = (origin + end) / 2.0 + ax["normal"] * size["offset"]
            a, w = ax["axis"], ax["width"]
            length, width = size["length"], size["width"]
            out.append((prop.kind, [np.stack([
                centre - a * length * 0.45 - w * width * 0.35,
                centre - a * length * 0.45 + w * width * 0.35,
                centre - a * length * 0.15 + w * width * 0.5,
                centre + a * length * 0.55,
                centre - a * length * 0.15 - w * width * 0.5,
            ])]))
    return out


# ---------------------------------------------------------------------------
# 3D joints -> 2D cell
# ---------------------------------------------------------------------------
def openpose_points(rig: Rig, world: np.ndarray, frame: RigFrame) -> np.ndarray:
    """(18, 3) world-space points in OpenPose order for one posed rig."""
    pos = positions_by_name(rig, world)
    jmap = frame.map
    head_world = world[rig.index(jmap.head)]
    head = head_world[:3, 3]
    r_head = rotation_part(head_world)
    fwd = r_head @ frame.head_forward_local
    up = r_head @ frame.head_up_local
    lateral = np.cross(up, fwd)          # the character's LEFT
    n = float(np.linalg.norm(lateral))
    lateral = lateral / n if n > 1e-9 else np.zeros(3)

    out = np.empty((len(OPENPOSE18), 3))
    for k, joint in enumerate(OPENPOSE18):
        if joint in jmap.face:
            a_up, a_fwd, a_lat = jmap.face[joint]
            out[k] = head + frame.head_length * (a_up * up + a_fwd * fwd
                                                 + a_lat * lateral)
        else:
            out[k] = resolve_joint(jmap.joints[joint], pos)
    return out


@dataclass(frozen=True)
class Projection:
    """Orthographic profile: forward -> +x (right), up -> -y (screen), floor on a row."""

    forward: np.ndarray
    floor_y: float
    scale: float                 # cell pixels per rig unit
    cell: tuple[int, int]
    baseline: float              # cell row of the floor

    @classmethod
    def for_rig(cls, frame: RigFrame, cell: tuple[int, int],
                height_fraction: float = HEIGHT_FRACTION,
                baseline_fraction: float = BASELINE_FRACTION) -> "Projection":
        width, height = cell
        return cls(forward=-frame.forward if frame.mirror else frame.forward,
                   floor_y=frame.floor_y,
                   scale=height_fraction * height / frame.height,
                   cell=(int(width), int(height)),
                   baseline=baseline_fraction * height)

    def anchor_for(self, points: np.ndarray) -> np.ndarray:
        """The world point `to_2d` would centre these points on - so props can
        be projected with the JOINTS' centre, not their own."""
        along = np.asarray(points, dtype=float) @ self.forward
        return self.forward * (float(along.min() + along.max()) / 2.0)

    def to_2d(self, points: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
        """(K, 3) world -> (K, 2) cell pixels.

        Horizontal: the points' own extent is centred in the cell, or, when
        `anchor` is given, that world point's forward-coordinate is. Vertical is
        absolute: the rig's floor is the baseline row whatever the pose.
        """
        pts = np.asarray(points, dtype=float)
        along = pts @ self.forward
        centre = float(np.asarray(anchor) @ self.forward) if anchor is not None \
            else float(along.min() + along.max()) / 2.0
        x = (along - centre) * self.scale + self.cell[0] / 2.0
        y = self.baseline - (pts[:, 1] - self.floor_y) * self.scale
        return np.stack([x, y], axis=1)


# ---------------------------------------------------------------------------
# The whole thing: one clip -> one keypoints.json payload
# ---------------------------------------------------------------------------
KEYPOINTS_SCHEMA = "anim.keypoints/1"


def sample_times(duration: float, fps: int) -> np.ndarray:
    """Evenly spaced times from 0 to the clip's end, both ends included.

    `round(duration * fps) + 1` samples over exactly [0, duration], so the
    first and last poses of the clip are always frames (the picker keeps both
    endpoints) and the spacing is within half a frame of 1/fps.
    """
    if duration <= 0:
        return np.zeros(1)
    n = max(2, int(round(duration * fps)) + 1)
    return np.linspace(0.0, duration, n)


def clip_keypoints(rig: Rig, clip_name: str, *, fps: int,
                   cell: tuple[int, int], mirror: bool = False,
                   frame: RigFrame | None = None,
                   times: Sequence[float] | None = None) -> dict[str, Any]:
    """Sample a clip and return the `anim.keypoints/1` payload `anim pose` writes.

    Every score is 1.0 and `person.score` is 1.0: a rig has no detector.
    `source_size` is the cell itself, so `anim.posekit.draw` places the points
    1:1. Each frame also carries its sample `time` in seconds.
    """
    frame = frame or reference_frame(rig, mirror=mirror)
    clip = rig.clip(clip_name)
    proj = Projection.for_rig(frame, cell)
    ts = np.asarray(times, dtype=float) if times is not None \
        else sample_times(clip.duration, fps)

    entries: list[dict[str, Any]] = []
    for index, t in enumerate(ts):
        world = rig.pose_at(clip, float(t))
        joints = openpose_points(rig, world, frame)
        pts2d = proj.to_2d(joints)
        entry: dict[str, Any] = {
            "index": index,
            "file": f"{index + 1:04d}.png",
            "time": round(float(t), 4),
            "keypoints": [[round(float(x), 2), round(float(y), 2)] for x, y in pts2d],
            "scores": [1.0] * len(OPENPOSE18),
            "person": {"score": 1.0},
        }
        if frame.props:
            anchor = proj.anchor_for(joints)
            entry["props"] = [
                {"kind": kind,
                 "polygons": [[[round(float(x), 2), round(float(y), 2)]
                               for x, y in proj.to_2d(poly, anchor=anchor)]
                              for poly in polys]}
                for kind, polys in prop_polygons(rig, world, frame)]
        entries.append(entry)
    payload_props = [p.kind for p in frame.props]
    return {
        "schema": KEYPOINTS_SCHEMA,
        "rig": "openpose18",
        "names": list(OPENPOSE18),
        "model": "rig",
        "mode": frame.map.name,
        "kpt_thr": 0.4,
        "source_size": [int(cell[0]), int(cell[1])],
        "motion": {
            "file": rig.path.name, "clip": clip.name,
            "duration": round(clip.duration, 4), "fps": fps,
            "mirror": bool(frame.mirror),
            "forward": [round(float(v), 4) for v in frame.forward],
            "floor_y": round(frame.floor_y, 4), "height": round(frame.height, 4),
            "head_length": round(frame.head_length, 4),
            "scale_px_per_unit": round(proj.scale, 4),
            "baseline_row": round(proj.baseline, 2),
            "props": payload_props,
        },
        "frames": entries,
    }
