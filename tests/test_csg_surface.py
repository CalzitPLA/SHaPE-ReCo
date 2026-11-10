"""reco_engine/csg_surface.py: boundary samples of sp trees (area, on-surface, normals, internal and coincident faces)
and the bracketed distances, against analytic values."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reco_engine.csg_surface import (boundary_samples, compile_program, compile_tree, finite_bounds,  # noqa: E402
                                     surface_distances)
from reco_engine.result_to_program import sp  # noqa: E402

PI = math.pi


def placed(prim, t=(0.0, 0.0, 0.0), R=None):
    return sp.Placed(prim, np.eye(3) if R is None else np.asarray(R, float), np.asarray(t, float))


def rot(axis, angle):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * angle).as_matrix()


# (name, tree, analytic boundary area); every tree tilted and moved off the origin (asymmetric placement)
R0 = rot((1, 2, 3), 0.7)
T0 = (3.0, -1.5, 2.0)
CASES = [
    ("box", sp.Box((1.0, 2.0, 3.0)), 8 * (2 + 6 + 3)),
    ("cylinder", sp.Cylinder(2.0, 5.0), 28 * PI),
    ("tube", sp.Cylinder(2.0, 5.0, inner_radius=1.0), 36 * PI),
    ("cylinder sector", sp.Cylinder(2.0, 5.0, u=(0.0, PI / 2)), 7 * PI + 20),
    ("cone", sp.Cone(3.0, 1.0, 4.0), PI * 4 * math.sqrt(20) + 10 * PI),
    ("cone up", sp.Cone(1.0, 3.0, 4.0), PI * 4 * math.sqrt(20) + 10 * PI),
    ("cone sector", sp.Cone(3.0, 1.0, 4.0, u=(0.0, PI)), 0.5 * (PI * 4 * math.sqrt(20) + 10 * PI) + 2 * 0.5 * (3 + 1) * 4),
    ("sphere", sp.Sphere(2.0), 16 * PI),
    ("hollow sphere", sp.Sphere(2.0, inner_radius=1.0), 20 * PI),
    ("hemisphere cone", sp.Sphere(2.0, v=(0.0, PI / 2)), 12 * PI),
    ("hemisphere plane", sp.Sphere(2.0, v=(0.0, PI / 2), latitude="plane"), 12 * PI),
    ("sphere sector", sp.Sphere(2.0, u=(0.0, PI)), 8 * PI + 4 * PI),
    ("torus", sp.Torus(3.0, 1.0), 12 * PI ** 2),
    ("torus sector", sp.Torus(3.0, 1.0, u=(0.0, PI)), 6 * PI ** 2 + 2 * PI),
    ("hollow torus", sp.Torus(3.0, 1.0, inner_minor=0.5), 12 * PI ** 2 + 6 * PI ** 2),
]


@pytest.mark.parametrize("name,prim,area", CASES, ids=[c[0] for c in CASES])
def test_leaf_area_surface_and_normals(name, prim, area):
    tree = compile_tree(placed(prim, T0, R0))
    s = boundary_samples(tree, n_target=60_000, seed=1)
    assert len(s.points) > 20_000, name
    # area within 3 % (Poisson error ~0.5 %; the candidate surfaces are exact)
    assert s.area == pytest.approx(area, rel=0.03), name
    lo, hi = finite_bounds(tree.leaves)
    diag = float(np.linalg.norm(hi - lo))
    assert np.max(np.abs(tree.sdf(s.points))) < 1e-7 * diag, name
    # outward normals: a small step along the normal leaves the solid, a step against it enters it
    eps = 1e-4 * diag
    assert np.mean(tree.sdf(s.points + eps * s.normals) > 0) > 0.995, name
    assert np.mean(tree.sdf(s.points - eps * s.normals) < 0) > 0.995, name


def test_shared_internal_face_is_dropped():
    a = placed(sp.Box((0.5, 0.5, 0.5)), (0.5, 0.0, 0.0))
    b = placed(sp.Box((0.5, 0.5, 0.5)), (1.5, 0.0, 0.0))
    s = boundary_samples(compile_tree(sp.Union(a, b)), n_target=40_000, seed=2)
    assert s.area == pytest.approx(10.0, rel=0.03)                       # 2 x 1 x 1 bar, not 12
    assert np.sum(np.abs(s.points[:, 0] - 1.0) < 1e-9) == 0             # nothing on the plane x = 1 inside the bar


def test_coincident_boundary_faces_do_not_double():
    box = sp.Box((1.0, 1.0, 1.0))
    s1 = boundary_samples(compile_tree(box), n_target=40_000, seed=3)
    s2 = boundary_samples(compile_tree(sp.Union(box, box)), n_target=40_000, seed=3)
    assert s2.area == pytest.approx(24.0, rel=0.03) and s1.area == pytest.approx(24.0, rel=0.03)


def test_box_minus_cylinder_hole_wall_normals_point_to_the_axis():
    hole = placed(sp.Cylinder(0.5, 4.0), (0.0, 0.0, -2.0))
    tree = compile_tree(sp.Subtraction(sp.Box((1.0, 1.0, 1.0)), hole))
    s = boundary_samples(tree, n_target=60_000, seed=4)
    area = 24.0 - 2 * PI * 0.25 + 2 * PI * 0.5 * 2.0
    assert s.area == pytest.approx(area, rel=0.03)
    wall = np.abs(np.hypot(s.points[:, 0], s.points[:, 1]) - 0.5) < 1e-9
    assert wall.sum() > 1000
    radial = s.points[wall, :2] / 0.5
    assert np.all((s.normals[wall, :2] * radial).sum(1) < -0.99)         # into the hole = towards the axis
    roles = {leaf.role for leaf in tree.leaves}
    assert roles == {"add", "cut"}


def test_halfspace_clip_and_program_owners():
    ops = [("add", sp.Box((1.0, 1.0, 1.0))), ("intersect", sp.HalfSpace((0.0, 0.0, 1.0), 0.25)),
           ("cut", placed(sp.Cylinder(0.3, 4.0), (0.0, 0.0, -2.0)))]
    tree = compile_program(ops)
    assert [leaf.owner for leaf in tree.leaves] == [0, 1, 2]
    assert [leaf.role for leaf in tree.leaves] == ["add", "clip", "cut"]
    s = boundary_samples(tree, n_target=60_000, seed=5)
    top = np.abs(s.points[:, 2] - 0.25) < 1e-9
    assert set(np.unique(s.owner[top])) == {1}                          # the clipped top face is the half space's
    h = 1.25
    area = 4 * 2 * h + 4 + 4 - 2 * PI * 0.09 + 2 * PI * 0.3 * h        # 4 side faces 2 x 1.25, bottom, top, hole
    assert s.area == pytest.approx(area, rel=0.03)


def test_distance_bracket_on_a_shifted_box():
    a = compile_tree(sp.Box((1.0, 1.0, 1.0)))
    b = compile_tree(placed(sp.Box((1.0, 1.0, 1.0)), (0.1, 0.0, 0.0)))
    sa = boundary_samples(a, n_target=30_000, seed=6)
    sb = boundary_samples(b, n_target=30_000, seed=7)
    d = surface_distances(sa.points, sa.normals, b, sb)
    assert np.all(d["lo"] <= d["hi"] + 1e-12)
    face = (np.abs(sa.points[:, 0] - 1.0) < 1e-9) & (np.abs(sa.points[:, 1]) < 0.8) & (np.abs(sa.points[:, 2]) < 0.8)
    assert np.allclose(d["hi"][face], 0.1, atol=1e-6)                   # Newton reaches the true distance
    assert np.all(d["signed"][face] < 0)                                # A's +x face lies inside B
    far = (np.abs(sa.points[:, 0] + 1.0) < 1e-9) & (np.abs(sa.points[:, 1]) < 0.8) & (np.abs(sa.points[:, 2]) < 0.8)
    assert np.allclose(d["hi"][far], 0.1, atol=1e-6) and np.all(d["signed"][far] > 0)


def test_true_distance_lies_in_the_bracket_at_a_concave_edge():
    # an L: the unit cube minus its upper +x quarter; points above the concave edge
    L = compile_tree(sp.Subtraction(sp.Box((1.0, 1.0, 1.0)), placed(sp.Box((0.5, 2.0, 0.5)), (0.5, 0.0, 0.5))))
    sL = boundary_samples(L, n_target=60_000, seed=8)
    P = np.array([[0.2, 0.0, 0.3], [0.1, 0.3, 0.1], [0.3, -0.2, 0.2]])   # in the notch
    true = np.array([min(0.3, 0.2), min(0.1, 0.1), min(0.2, 0.3)])      # distance to the walls x = 0 or z = 0
    d = surface_distances(P, np.tile([0.0, 0.0, 1.0], (3, 1)), L, sL)
    assert np.all(d["lo"] <= true + 1e-9)
    assert np.all(d["hi"] >= true - 1e-9)
    assert np.all(d["hi"] <= true + 3 * sL.spacing)
