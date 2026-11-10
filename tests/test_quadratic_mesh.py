"""dataset_gen/src/quadratic_mesh: six-node triangles, their flat subdivision and the closest-point query on the patches."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src.freecad_path import find_freecadcmd  # noqa: E402
from src.quadratic_mesh import (NODE_BARY, QuadraticSurface, mesh_step, shape_derivatives, shape_functions,  # noqa: E402
                                signed_volume, subdivide)

CENTRE = np.array([3.0, -2.0, 5.0])                 # off the origin: a sign or axis mix-up shows
RADII = np.array([10.0, 7.0, 4.0])                  # an ellipsoid, no two axes alike


def ellipsoid_mesh(level=2, radii=RADII, centre=CENTRE):
    """Six-node triangles on an ellipsoid: a subdivided octahedron, corners and edge midpoints projected onto it."""
    v = [np.array(x, float) for x in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    f = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    cache = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = v[a] + v[b]
            v.append(m / np.linalg.norm(m))
            cache[key] = len(v) - 1
        return cache[key]
    for _ in range(level):
        f = [t for a, b, c in f for t in ((a, mid(a, b), mid(c, a)), (mid(a, b), b, mid(b, c)),
                                          (mid(c, a), mid(b, c), c), (mid(a, b), mid(b, c), mid(c, a)))]
    tri6 = np.array([(a, b, c, mid(a, b), mid(b, c), mid(c, a)) for a, b, c in f])
    return np.array(v) * radii + centre, tri6


def test_shape_functions_interpolate_the_nodes_and_sum_to_one():
    n = shape_functions(NODE_BARY)
    assert np.allclose(n, np.eye(6))
    rng = np.random.default_rng(0)
    b = rng.dirichlet((1, 1, 1), 50)
    assert np.allclose(shape_functions(b).sum(1), 1.0)
    du, dv = shape_derivatives(b)
    assert np.allclose(du.sum(1), 0.0) and np.allclose(dv.sum(1), 0.0)
    h = 1e-6                                                             # the derivatives against finite differences
    bu = b + h * np.array([-1.0, 1.0, 0.0])
    bv = b + h * np.array([-1.0, 0.0, 1.0])
    assert np.allclose((shape_functions(bu) - shape_functions(b)) / h, du, atol=1e-5)
    assert np.allclose((shape_functions(bv) - shape_functions(b)) / h, dv, atol=1e-5)


def test_subdivision_keeps_orientation_and_approaches_the_volume():
    nodes, tri6 = ellipsoid_mesh(2)
    true = 4.0 / 3.0 * np.pi * RADII.prod()
    volumes = []
    for levels in (1, 2, 3):
        v, f, parent, bary = subdivide(nodes, tri6, levels)
        assert len(f) == len(tri6) * 4 ** levels and parent.max() == len(tri6) - 1 and bary.shape == (len(f), 3, 3)
        volumes.append(signed_volume(v - CENTRE, f))
    assert all(x > 0 for x in volumes)                                   # outward, as the octahedron was
    err = [abs(x - true) / true for x in volumes]
    assert err[2] < err[1] < err[0] and err[2] < 5e-3            # measured 2.3e-2, 6.6e-3, 2.4e-3 (128 elements)


def test_query_is_closer_to_the_true_surface_than_the_flat_mesh():
    nodes, tri6 = ellipsoid_mesh(2, radii=np.array([10.0, 10.0, 10.0]))   # a sphere: the truth is analytic
    s = QuadraticSurface(nodes, tri6)
    rng = np.random.default_rng(1)
    d = rng.normal(size=(3000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = 10.0 + rng.uniform(-2, 2, 3000)
    q = s.query(CENTRE + d * r[:, None])
    e_quad, e_flat = np.abs(q.distance - np.abs(r - 10.0)), np.abs(q.linear_distance - np.abs(r - 10.0))
    assert e_quad.mean() < 0.05 * e_flat.mean() and e_quad.max() < 0.2 * e_flat.max()
    angle = np.degrees(np.arccos(np.clip((q.normal * d).sum(1), -1, 1)))  # signed: the normal points outward
    assert angle.mean() < 0.5 and angle.max() < 2.0                      # measured 0.27 and 1.11 degrees (128 elements)
    assert np.allclose(np.linalg.norm(q.closest - CENTRE, axis=1), 10.0, atol=2e-2)   # measured max 1.3e-2 (128 elements)
    assert np.allclose(q.bary.sum(1), 1.0) and (q.bary >= -1e-12).all()


def test_query_on_the_ellipsoid_finds_points_of_the_patches():
    nodes, tri6 = ellipsoid_mesh(2)
    s = QuadraticSurface(nodes, tri6)
    rng = np.random.default_rng(2)
    face = rng.integers(0, len(tri6), 500)
    bary = rng.dirichlet((2, 2, 2), 500)
    on = s.evaluate(face, bary)
    q = s.query(on + 0.3 * s.normals(face, bary))                         # 0.3 above the patch along its normal
    assert np.abs(q.distance - 0.3).max() < 1e-3
    assert np.linalg.norm(q.closest - on, axis=1).max() < 2e-2
    x, y, z = ((q.closest - CENTRE) / RADII).T                            # on the ellipsoid itself, within the patch error
    assert np.abs(x * x + y * y + z * z - 1.0).max() < 5e-3


def test_neighbours_share_an_edge():
    nodes, tri6 = ellipsoid_mesh(1)
    s = QuadraticSurface(nodes, tri6)
    assert (s._neighbour >= 0).all()                                      # closed
    for e in range(len(tri6)):
        for k in range(3):
            assert tri6[e, [4, 5, 3][k]] in tri6[s._neighbour[e, k], 3:]


@pytest.mark.skipif(shutil.which("gmsh") is None or find_freecadcmd() is None, reason="needs gmsh and freecadcmd")
def test_gmsh_meshes_a_step_with_nodes_on_the_surface(tmp_path):
    import subprocess
    (tmp_path / "mk.py").write_text("import FreeCAD, Part\nPart.makeCylinder(10, 30, FreeCAD.Vector(3, -2, 5)).exportStep('c.step')\n")
    subprocess.run([find_freecadcmd(), "mk.py"], cwd=tmp_path, capture_output=True, timeout=120, check=True)
    nodes, tri6 = mesh_step(tmp_path / "c.step", 4.0, order=2)
    assert tri6.shape[1] == 6
    lateral = (nodes[:, 2] > 5.01) & (nodes[:, 2] < 34.99)
    assert lateral.sum() > 50
    assert np.abs(np.hypot(nodes[lateral, 0] - 3.0, nodes[lateral, 1] + 2.0) - 10.0).max() < 1e-9
    v, f, _, _ = subdivide(nodes, tri6, 3)
    assert abs(signed_volume(v, f) - np.pi * 100 * 30) / (np.pi * 100 * 30) < 1e-3


@pytest.mark.skipif(shutil.which("gmsh") is None or find_freecadcmd() is None, reason="needs gmsh and freecadcmd")
def test_load_field_order_2_is_a_surface_field_with_better_distances(tmp_path, monkeypatch):
    import subprocess
    from src.closest_point_field import SurfaceField, load_field
    (tmp_path / "mk.py").write_text("import FreeCAD, Part\nPart.makeSphere(10, FreeCAD.Vector(3, -2, 5)).exportStep('s.step')\n")
    subprocess.run([find_freecadcmd(), "mk.py"], cwd=tmp_path, capture_output=True, timeout=120, check=True)
    monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", "igl")
    first = load_field(tmp_path / "s.step")
    assert type(first) is SurfaceField                                    # the default is unchanged
    monkeypatch.setenv("SHAPE_RECO_MESH_ORDER", "2")
    second = load_field(tmp_path / "s.step")
    assert isinstance(second, SurfaceField) and second.order == 2 and second.is_closed()
    rng = np.random.default_rng(3)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = 10.0 + rng.uniform(-2, 2, 2000)
    p = CENTRE + d * r[:, None]
    q1, q2 = first.query(p), second.query(p)
    truth = 10.0 - r                                                      # signed: positive inside
    assert np.abs(q2.signed_distance - truth).mean() < 0.2 * np.abs(q1.signed_distance - truth).mean()
    assert (np.sign(q2.signed_distance) == np.sign(truth)).mean() > 0.999
    assert ((q2.normal * d).sum(1) > 0.99).all()                          # outward
    assert q2.face.max() < len(second.faces)
