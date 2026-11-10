"""The first guess by hand (reco_engine/user_guess.py): the shapes and their file, the coarse slider scales, the live hint
on a known overlap, the pick, and `refine` on an analytic case. No Qt, no FreeCAD."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from scipy.spatial.transform import Rotation  # noqa: E402

from reco_engine import user_guess as ug  # noqa: E402
from reco_engine.result_schema import Primitive  # noqa: E402
from reco_engine.user_guess import Scales, UserShape  # noqa: E402

trimesh = pytest.importorskip("trimesh")


def box_mesh(extents, centre=(0, 0, 0), rotation=None):
    m = trimesh.creation.box(extents=extents)
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    T[:3, 3] = centre
    m.apply_transform(T)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


TILT = Rotation.from_euler("xyz", [20, -35, 50], degrees=True)
CENTRE = np.array([30.0, -12.0, 7.0])
HALF = np.array([20.0, 8.0, 5.0])


def test_shape_primitive_and_file_round_trip(tmp_path):
    s = UserShape("cut", "cylinder", (1.0, 2.0, 3.0), ug.euler_quaternion((90, 0, 0)), {"radius": 2.0, "half_length": 5.0},
                  "bore")
    p = s.primitive()
    assert isinstance(p, Primitive) and p.kind == "cylinder" and p.center_mm == (1.0, 2.0, 3.0) and p.label == "bore"
    assert p.dims_mm == {"radius": 2.0, "half_length": 5.0} and p.sources[0].mode == "user"
    assert UserShape.from_primitive("cut", p) == s
    a = UserShape("add", "box", (0, 0, 0), (0, 0, 0, 1), {"x_half": 1, "y_half": 2, "z_half": 3})
    path = ug.save_first_guess(tmp_path, [s, a], ordered=False, input_path="part.step")
    assert path == tmp_path / "first_guess.json"
    data = json.loads(path.read_text())
    assert data["schema"] == 1 and data["input"] == "part.step" and [e["op"] for e in data["shapes"]] == ["cut", "add"]
    shapes, meta = ug.load_first_guess(path)
    assert shapes == [s, a] and meta["ordered"] is False and meta["refined"] == [False, False]
    assert ug.load_first_guess(tmp_path)[0] == [s, a]                                  # the directory works too
    # the three forms of the contract; adds before cuts unless the user ordered them
    for form in (path, [s, a], [s.as_dict(), a.as_dict()], [("cut", s.primitive()), ("add", a.primitive())]):
        ops = ug.as_operations(form)
        assert [(op, p.kind) for op, p in ops] == [("add", "box"), ("cut", "cylinder")]
    assert [op for op, _ in ug.as_operations([s, a], ordered=True)] == ["cut", "add"]
    ug.save_first_guess(path, [s, a], ordered=True)
    assert [op for op, _ in ug.as_operations(path)] == ["cut", "add"]
    assert ug.as_operations(None) == []
    with pytest.raises(ValueError):
        UserShape.from_dict(dict(op="intersect", kind="box"))


def test_slider_scales_are_coarse():
    V, _ = box_mesh((100.0, 50.0, 20.0), centre=(10, 0, 0))
    s = Scales.of(V)
    assert s.lo == pytest.approx([-40, -25, -10]) and s.hi == pytest.approx([60, 25, 10])
    # position: 1/50 of the extent per step, a quarter of the extent beyond the box on each side
    assert [s.position_step(a) for a in range(3)] == pytest.approx([2.0, 1.0, 0.4])
    assert s.position_count() == 75
    assert s.position(0, 0) == pytest.approx(-65.0) and s.position(0, 75) == pytest.approx(85.0)
    assert s.position(0, 1) - s.position(0, 0) == pytest.approx(2.0) and s.position(0, 999) == s.position(0, 75)
    assert s.position_index(0, 10.9) == 38 and s.position(0, 38) == pytest.approx(11.0)       # the nearest step
    assert all(s.position_index(a, s.position(a, i)) == i for a in range(3) for i in (0, 7, 40, 75))
    # size: logarithmic, 40 steps from diagonal / 100 to the diagonal
    d = s.diagonal
    assert d == pytest.approx(np.sqrt(100 ** 2 + 50 ** 2 + 20 ** 2))
    assert s.size(0) == pytest.approx(d / 100) and s.size(40) == pytest.approx(d) and s.size(20) == pytest.approx(d / 10)
    ratios = [s.size(i + 1) / s.size(i) for i in range(40)]
    assert ratios == pytest.approx([100 ** (1 / 40)] * 40)                                     # each step x 1.122
    assert all(s.size_index(s.size(i)) == i for i in range(41)) and s.size_index(1e-9) == 0 and s.size_index(1e9) == 40
    # angles: 15 degrees
    assert Scales.angle_count() == 24 and Scales.angle(0) == -180 and Scales.angle(12) == 0 and Scales.angle(13) == 15
    assert Scales.angle_index(44) == 15 and Scales.angle(Scales.angle_index(44)) == 45 and Scales.angle_index(190) == 1
    q = ug.euler_quaternion((0, 0, 90))
    assert Rotation.from_quat(q).apply([1, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)
    assert ug.quaternion_euler(ug.euler_quaternion((30, -45, 105))) == pytest.approx((30, -45, 105))
    shape = s.default_shape("cylinder", "cut", centre=(1, 2, 3))
    assert shape.op == "cut" and shape.center_mm == (1.0, 2.0, 3.0) and set(shape.dims_mm) == {"radius", "half_length"}
    assert shape.dims_mm["half_length"] == pytest.approx(s.size(s.size_index(0.1 * d)))
    for kind in ug.KINDS:
        assert ug.sample_inside(s.default_shape(kind), 200).shape == (200, 3)


def test_align_with_the_part():
    V, F = box_mesh(2 * HALF, CENTRE, TILT.as_matrix())
    frames = dict(ug.part_frames(V, F))
    assert {"world x", "world z", "bounding box", "box axis 1"} <= set(frames)
    for name, R in frames.items():
        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-9) and np.linalg.det(R) == pytest.approx(1.0), name
    long_axis = TILT.as_matrix()[:, 0]
    assert abs(frames["box axis 1"][:, 2] @ long_axis) == pytest.approx(1.0, abs=1e-6)       # the box's long edge
    assert any(abs(R[:, 2] @ long_axis) > 0.9999 for n, R in frames.items() if n.startswith("member direction"))


def test_live_hint_on_a_known_overlap():
    V, F = box_mesh((20.0, 20.0, 20.0))                                                  # the input: a cube about 0
    box = lambda cx, op="add": UserShape(op, "box", (cx, 0, 0), (0, 0, 0, 1), {"x_half": 5, "y_half": 5, "z_half": 5})  # noqa: E731
    assert ug.inside_share(box(0), V, F) == 1.0
    half = ug.inside_share(box(10), V, F)                                                # half of it sticks out
    assert half == pytest.approx(0.5, abs=0.03)
    assert ug.inside_share(box(7.5), V, F) == pytest.approx(0.75, abs=0.03)
    assert ug.inside_share(box(30), V, F) == 0.0
    assert ug.inside_share(box(10, "cut"), V, F) == pytest.approx(1.0 - half)            # a cut: the share outside
    assert ug.inside_share(box(30, "cut"), V, F) == 1.0
    sphere = UserShape("add", "sphere", (10, 0, 0), (0, 0, 0, 1), {"radius": 4.0})
    assert ug.inside_share(sphere, V, F) == pytest.approx(0.5, abs=0.03)
    assert ug.inside_share(box(10), V, F) == half                                        # seeded: the same number
    assert [ug.hint_label(x) for x in (1.0, 0.9, 0.6, 0.2, None)] == ["good", "fair", "rough", "poor", "no volume"]
    colours = [ug.hint_colour(x) for x in (0.0, 0.5, 1.0)]
    assert colours[0] != colours[2] and all(c[2] >= c[0] and c[2] >= c[1] for c in colours)   # one hue: blue
    lightness = [sum(c) for c in colours]
    assert lightness[0] > lightness[1] > lightness[2]                                    # ... darker is better


def test_pick_lands_on_the_surface_under_the_cursor():
    V, F = box_mesh((4.0, 2.0, 6.0), centre=(7.0, -3.0, 11.0))                           # off-centre, asymmetric
    hit = ug.ray_mesh_hit((7.5, -3.2, 100.0), (0, 0, -1), V, F)
    assert hit == pytest.approx([7.5, -3.2, 14.0])                                       # the top face, not the bottom
    assert ug.ray_mesh_hit((7.5, -3.2, 100.0), (0, 0, 1), V, F) is None                  # looking away
    assert ug.ray_mesh_hit((20.0, -3.0, 100.0), (0, 0, -1), V, F) is None
    assert ug.drop_point((100.0, -3.5, 12.0), (-1, 0, 0), V, F) == pytest.approx([9.0, -3.5, 12.0])
    miss = ug.drop_point((20.0, 5.0, 100.0), (0, 0, -1), V, F)                           # beside the part: its centre's depth
    assert miss == pytest.approx([20.0, 5.0, 11.0])


def test_refine_brings_a_rough_box_back_onto_the_input():
    """The input: an off-centre, tilted box. The user's box: 2 slider steps off along every axis, 20 % too large, 10
    degrees off. Refined it lies on the input; the report says what changed."""
    R = TILT.as_matrix()
    V, F = box_mesh(2 * HALF, CENTRE, R)
    scales = Scales.of(V)
    step = np.array([scales.position_step(a) for a in range(3)])
    off = Rotation.from_rotvec(np.radians(10) * np.array([0.0, 0.6, 0.8])) * TILT
    given = UserShape("add", "box", tuple(CENTRE + 2 * step * [1, -1, 1]), tuple(off.as_quat()),
                      dict(zip(("x_half", "y_half", "z_half"), 1.2 * HALF)), "block")
    (r,) = ug.refine([given], V, F)
    got = r.shape
    assert got.op == "add" and got.kind == "box" and got.label == "block" and r.given is given
    assert np.asarray(got.center_mm) == pytest.approx(CENTRE, abs=0.05)
    assert [got.dims_mm[k] for k in ("x_half", "y_half", "z_half")] == pytest.approx(HALF, abs=0.05)
    turn = (Rotation.from_quat(got.quaternion) * TILT.inv()).magnitude()
    assert np.degrees(turn) < 0.2
    assert r.moved_steps == pytest.approx(2.0, abs=0.05) and r.moved_mm == pytest.approx(np.linalg.norm(2 * step), abs=0.1)
    assert r.turned_deg == pytest.approx(10.0, abs=0.3)
    assert list(r.scaled.values()) == pytest.approx([1 / 1.2] * 3, abs=0.005) and r.at_limit == [] and r.points > 1000
    assert r.precision_given < 0.75 and r.precision > 0.99 and r.surface_given < 0.2 and r.surface > 0.95
    assert r.passes and r.reason == "" and r.thresholds == dict(member_min_precision=0.95, boolean_min_surface=0.2)
    text = r.summary()
    assert "moved" in text and "2.0 steps" in text and "x0.83" in text and text.endswith("passes")
    json.dumps(r.as_dict())


def test_refine_stays_in_its_window_and_reports_a_failure():
    V, F = box_mesh(2 * HALF, CENTRE, TILT.as_matrix())
    scales = Scales.of(V)
    step = np.array([scales.position_step(a) for a in range(3)])
    far = UserShape("add", "box", tuple(CENTRE + 12 * step), tuple(TILT.as_quat()),
                    dict(zip(("x_half", "y_half", "z_half"), 0.5 * HALF)))
    (r,) = ug.refine([far], V, F)
    moved = np.abs(np.asarray(r.shape.center_mm) - np.asarray(far.center_mm)) / step
    assert (moved <= ug.REFINE_POSITION_STEPS + 1e-6).all()
    assert all(1 / ug.REFINE_SIZE_FACTOR - 1e-6 <= v <= ug.REFINE_SIZE_FACTOR + 1e-6 for v in r.scaled.values())
    assert r.turned_deg <= ug.REFINE_ANGLE + 1e-6
    assert not r.passes and "inside the input" in r.reason and r.at_limit and "FAILS" in r.summary()
    nowhere = UserShape("cut", "sphere", (500.0, 500.0, 500.0), (0, 0, 0, 1), {"radius": 1.0})
    (n,) = ug.refine([nowhere], V, F)                                  # no surface near it: it stays, and fails
    assert n.shape.center_mm == nowhere.center_mm and n.points == 0 and n.precision == 1.0 and n.surface is None
    assert not n.passes and "nearby surface" in n.reason
    assert ug.refine([], V, F) == []


def test_refine_a_cut_cylinder_and_snap_to_exact_faces():
    """A bore through the tilted box: the rough cylinder (radius 25 % too large, a step aside) comes onto the bore; with
    the input's exact faces it is snapped (feature_stage.snap_to_faces is called, not copied)."""
    R = TILT.as_matrix()
    block = trimesh.creation.box(extents=2 * HALF)
    bore = trimesh.creation.cylinder(radius=3.0, height=30.0, sections=96)
    try:
        part = block.difference(bore)
    except Exception:
        pytest.skip("no boolean backend for the test part")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, CENTRE
    part.apply_transform(T)
    V, F = np.asarray(part.vertices, float), np.asarray(part.faces, np.int64)
    scales = Scales.of(V)
    step = np.array([scales.position_step(a) for a in range(3)])
    given = UserShape("cut", "cylinder", tuple(CENTRE + step * [1, 1, 0]), tuple(TILT.as_quat()),
                      {"radius": 3.75, "half_length": 6.0}, "bore")
    (r,) = ug.refine([given], V, F)
    axis = R[:, 2]
    aside = np.asarray(r.shape.center_mm) - CENTRE
    assert np.linalg.norm(aside - axis * (aside @ axis)) < 0.1                         # on the bore's axis
    assert r.shape.dims_mm["radius"] == pytest.approx(3.0, abs=0.05)
    assert r.surface > r.surface_given and r.precision == pytest.approx(1.0, abs=0.02)
    faces = [dict(type="cylinder", axis=list(axis), center=list(CENTRE), radius=3.0, convex=False, area=100.0,
                  normal=None)]
    (s,) = ug.refine([given], V, F, exact_faces=faces)
    assert s.snapped and "error" not in s.snapped[0], s.snapped
    assert s.shape.dims_mm["radius"] == pytest.approx(3.0, abs=1e-9) and "snapped to" in s.summary()
