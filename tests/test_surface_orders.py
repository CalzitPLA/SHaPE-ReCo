"""
Containment and surface coverage by order (reco_engine/surface_orders.py) on parts whose answer is known: a block with
a bore, tilted and off-centre, its exact faces written by hand, its mesh from the SDF tree's own solid.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("igl")

from reco_engine import surface_orders as so  # noqa: E402
from reco_engine.result_schema import Primitive  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp, quaternion_of  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

R = Rotation.from_euler("zxy", [35.0, -20.0, 110.0], degrees=True).as_matrix()
T = np.array([40.0, -25.0, 310.0])
HALF = np.array([30.0, 20.0, 15.0])
BORE = np.array([8.0, -3.0, 0.0])                      # the bore's axis in the block's frame: along z, through this point


def block(half=HALF):
    return Primitive(kind="box", quaternion=quaternion_of(R), center_mm=tuple(T), dims_mm=dict(zip(("x_half", "y_half", "z_half"), half)))


def cylinder(r, half_length=20.0, centre=BORE):
    return Primitive(kind="cylinder", quaternion=quaternion_of(R), center_mm=tuple(T + R @ centre),
                     dims_mm=dict(radius=r, half_length=half_length))


def part_mesh(r=6.0):
    """The block with a through bore of radius r as a mesh (a box mesh minus a cylinder mesh, by trimesh's primitives
    placed like the part), and its exact faces as FreeCAD would give them."""
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, T
    box = trimesh.creation.box(extents=2 * HALF, transform=M)
    B = np.eye(4)
    B[:3, :3], B[:3, 3] = R, T + R @ BORE
    wall = trimesh.creation.cylinder(radius=r, height=2 * HALF[2], sections=256, transform=B)
    keep = np.abs(np.abs((wall.triangles_center - B[:3, 3]) @ R[:, 2]) - HALF[2]) > 1e-6     # the wall without its ends
    wall = trimesh.Trimesh(wall.vertices, wall.faces[keep], process=False)
    # the block's faces with the bore's discs left in: good enough for a surface sample (the discs are 3 % of the area)
    V = np.vstack([box.vertices, wall.vertices])
    F = np.vstack([box.faces, wall.faces + len(box.vertices)])
    faces = []
    lo, hi = V.min(axis=0), V.max(axis=0)
    for i in range(3):
        for s in (-1.0, 1.0):
            faces.append(dict(type="plane", area=1.0, normal=list(s * R[:, i]), point=list(T + s * HALF[i] * R[:, i]),
                              origin="input", bbox=[list(lo), list(hi)]))
    faces.append(dict(type="cylinder", area=1.0, axis=list(R[:, 2]), center=list(T + R @ BORE), radius=r, origin="input",
                      bbox=[list(lo), list(hi)]))
    return V, F, faces


def test_the_analytic_surfaces_have_their_normals_and_curvatures():
    P = T + (BORE + np.array([[6.0, 0.0, 3.0], [0.0, -6.0, -7.0]])) @ R.T
    cyl = dict(type="cylinder", center=T + R @ BORE, axis=R[:, 2], radius=6.0)
    assert np.allclose(so.distance(cyl, P), 0.0, atol=1e-9)
    n = so.normal(cyl, P, 1e-3)
    assert np.allclose(n, np.array([R[:, 0], -R[:, 1]]), atol=1e-6)
    S = so.shape_operator(cyl, P, 1e-3)
    assert np.allclose(np.linalg.eigvalsh(S), [[0.0, 0.0, 1 / 6.0]] * 2, atol=1e-5)      # flat along the axis, 1/r across
    plane = dict(type="plane", point=T, normal=R[:, 0])
    assert np.allclose(so.shape_operator(plane, P, 1e-3), 0.0, atol=1e-9)
    ball = dict(type="sphere", center=T, radius=4.0)
    Q = T + 4.0 * np.array([R[:, 0], R[:, 1]])
    assert np.allclose(np.linalg.eigvalsh(so.shape_operator(ball, Q, 1e-3)), [[0.0, 0.25, 0.25]] * 2, atol=1e-5)
    torus = dict(type="torus", center=T, axis=R[:, 2], major_radius=20.0, minor_radius=5.0)
    outer = (T + 25.0 * R[:, 0])[None]
    assert np.allclose(np.linalg.eigvalsh(so.shape_operator(torus, outer, 1e-3)), [[0.0, 1 / 25.0, 1 / 5.0]], atol=1e-5)
    cone = dict(type="cone", center=T, axis=R[:, 2], semi_angle=math.radians(30.0))
    on = (T + 10.0 * R[:, 2] + 10.0 * math.tan(math.radians(30.0)) * R[:, 0])[None]
    assert abs(so.distance(cone, on)[0]) < 1e-9
    k = np.linalg.eigvalsh(so.shape_operator(cone, on, 1e-3))[0]
    assert k[:2] == pytest.approx([0.0, 0.0], abs=1e-5) and k[2] == pytest.approx(math.cos(math.radians(30.0)) / (10.0 * math.tan(math.radians(30.0))), rel=1e-4)


def test_a_primitive_names_its_surfaces():
    assert [s["type"] for s in so.primitive_surfaces(block())] == ["plane"] * 6
    assert sorted(s["type"] for s in so.primitive_surfaces(cylinder(6.0))) == ["cylinder", "plane", "plane"]
    tip = Primitive(kind="cone", quaternion=quaternion_of(R), center_mm=tuple(T), dims_mm=dict(radius1=3.0, radius2=0.0, half_length=1.0))
    s = so.primitive_surfaces(tip)
    assert sorted(x["type"] for x in s) == ["cone", "plane"]
    cone = [x for x in s if x["type"] == "cone"][0]
    assert np.allclose(cone["center"], T + 1.0 * R[:, 2]) and cone["semi_angle"] == pytest.approx(math.atan2(3.0, 2.0))
    sector = Primitive(kind="cylinder", quaternion=quaternion_of(R), center_mm=tuple(T), dims_mm=dict(radius=6.0, half_length=5.0),
                       u=(0.0, math.pi))
    assert sorted(x["type"] for x in so.primitive_surfaces(sector)) == ["cylinder"] + ["plane"] * 4


def test_containment_is_the_share_of_the_inputs_surface_on_or_inside():
    V, F, _ = part_mesh()
    assert so.containment(primitive_to_sp(block()), V, F)["inside"] == 1.0
    short = so.containment(primitive_to_sp(block(HALF * [0.9, 1.0, 1.0])), V, F)
    assert 0.5 < short["inside"] < 0.95 and short["worst"] == pytest.approx(3.0, abs=0.05)
    assert len(short["outside"]) == round((1 - short["inside"]) * short["points"])


def test_the_block_alone_covers_the_planes_exactly_and_nothing_of_the_bore():
    V, F, faces = part_mesh()
    c = so.coverage(block(), faces, V, F)
    # the bore's discs are left in the mesh: 2 x pi 36 of the block's 10 400 mm2 and of the sample
    wall = 2 * math.pi * 6.0 * 30.0
    planes = 2 * (60 * 40 + 60 * 30 + 40 * 30)
    assert c["input_on_model"][2] == pytest.approx(planes / (planes + wall), abs=0.02)
    assert c["input_on_model"] == pytest.approx([c["exact"]] * 3, abs=0.01)
    assert c["model_on_input"][2] == pytest.approx(1.0, abs=0.01)


def test_the_bores_radius_decides_order_two():
    V, F, faces = part_mesh(6.0)
    both = Primitive(kind="composite", center_mm=tuple(T), dims_mm={}, parts=[block(), cylinder(6.0)], combine="cut")
    exact = so.coverage(both, faces, V, F)
    assert exact["exact"] == pytest.approx(1.0, abs=0.03)                 # planes and wall, to every order (the mesh's
    #                                                                       two discs over the bore are 2 % of it)
    # a bore of radius 5 that touches the real wall (radius 6) along a line: near that line it agrees in position,
    # nearer still in the normal, nowhere in curvature (1 / 5 against 1 / 6)
    r = 5.0
    touching = cylinder(r, centre=BORE + np.array([6.0 - r, 0.0, 0.0]))
    near = Primitive(kind="composite", center_mm=tuple(T), dims_mm={}, parts=[block(), touching], combine="cut")
    c = so.coverage(near, faces, V, F, tol=0.05, n=40000)
    area = 2 * (60 * 40 + 60 * 30 + 40 * 30) + 2 * math.pi * 6.0 * 30.0
    wall, discs = 2 * math.pi * 6.0 * 30.0 / area, 2 * math.pi * 36.0 / area
    theta0 = math.acos(1.0 - 0.05 / (6.0 - r))                          # the gap (6 - r)(1 - cos theta) is the tolerance
    assert c["input_on_model"][0] - c["input_on_model"][2] == pytest.approx(wall * theta0 / math.pi, rel=0.15)
    assert c["input_on_model"][0] > c["input_on_model"][1] > c["input_on_model"][2]
    assert c["input_on_model"][2] == pytest.approx(1.0 - wall - discs, abs=0.02)   # the planes, nothing of the wall
    # a radius that is right within 0.5 % passes; the tolerance of order 2 is 1 % of the curvature
    fine = Primitive(kind="composite", center_mm=tuple(T), dims_mm={}, parts=[block(), cylinder(6.02)], combine="cut")
    assert so.coverage(fine, faces, V, F, tol=0.05)["exact"] == pytest.approx(1.0 - discs, abs=0.02)
