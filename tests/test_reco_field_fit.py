"""reco_field/fit.py: primitives fitted to the closest-point field of a mesh recover the true parameters (synthetic meshes with a
known pose) and pick the right kind; on the example STLs the base primitive is found and the rest of the surface is left over."""
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.fit import fit_best, fit_box, fit_cone, fit_cylinder, fit_sphere, fit_torus, make_samples  # noqa: E402
from reco_field.peel import mesh_from_sdf  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField, load_field  # noqa: E402


def posed(mesh, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    x, y, z, w = q / np.linalg.norm(q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    t = rng.uniform(-5, 5, 3)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    mesh = mesh.copy()
    mesh.apply_transform(T)
    return SurfaceField(mesh.vertices, mesh.faces), R, t


@pytest.mark.parametrize("seed", range(3))
def test_box_is_recovered_with_pose_and_size(seed):
    field, R, t = posed(trimesh.creation.box(extents=(12.0, 7.0, 4.0)), seed)
    r = fit_box(make_samples(field))
    assert r.explained > 0.99
    assert np.allclose(sorted([r.params["hx"], r.params["hy"], r.params["hz"]]), [2.0, 3.5, 6.0], atol=0.05)
    assert np.allclose([r.params["cx"], r.params["cy"], r.params["cz"]], t, atol=0.05)


@pytest.mark.parametrize("seed", range(3))
def test_sphere_is_recovered(seed):
    field, R, t = posed(trimesh.creation.icosphere(subdivisions=4, radius=7.0), seed)
    r = fit_sphere(make_samples(field))
    assert r.explained > 0.99 and abs(r.params["radius"] - 7.0) < 0.05
    assert np.allclose([r.params["cx"], r.params["cy"], r.params["cz"]], t, atol=0.05)


@pytest.mark.parametrize("seed", range(3))
def test_cylinder_is_recovered(seed):
    field, R, t = posed(trimesh.creation.cylinder(radius=3.0, height=9.0, sections=128), seed)
    r = fit_cylinder(make_samples(field))
    assert r.explained > 0.99 and abs(r.params["radius"] - 3.0) < 0.05 and abs(r.params["half_length"] - 4.5) < 0.05
    assert np.allclose([r.params["cx"], r.params["cy"], r.params["cz"]], t, atol=0.05)


@pytest.mark.parametrize("kind,mesh", [("box", trimesh.creation.box(extents=(9.0, 6.0, 5.0))),
                                       ("sphere", trimesh.creation.icosphere(subdivisions=4, radius=6.0)),
                                       ("cylinder", trimesh.creation.cylinder(radius=3.0, height=10.0, sections=96))])
def test_the_right_kind_wins(kind, mesh):
    field, _, _ = posed(mesh, 11)
    ranked = fit_best(field, stop_at=2.0)                                     # fit every kind, no early stop
    # a cone contains the cylinder as its degenerate case (equal radii): it ties, and loses on the parameter count
    others = [r.explained for r in ranked[1:] if not (kind == "cylinder" and r.kind == "cone")]
    assert ranked[0].kind == kind and ranked[0].explained > 0.99 and max(others) < 0.8


# ------------------------------------------------------------------------------ cone, torus and the segments (u, v)

def solid(prim, seed=0, resolution=150):
    """The closest-point field of an analytic solid, placed at a random pose: (field, R, t)."""
    rng = np.random.default_rng(seed)
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=seed).as_matrix()
    t = rng.uniform(-4, 4, 3)
    v, f = mesh_from_sdf(sp.Placed(prim, R, t), resolution)
    return SurfaceField(v, f), R, t


@pytest.mark.parametrize("seed", range(2))
def test_cone_is_recovered_with_pose_and_size(seed):
    field, R, t = solid(sp.Cone(9.0, 3.0, 14.0), seed)
    r = fit_cone(make_samples(field))
    assert r.explained > 0.99 and r.u is None and r.n_params == 9
    assert np.allclose(sorted([r.params["radius1"], r.params["radius2"]]), [3.0, 9.0], atol=0.06)
    assert abs(r.params["half_height"] - 7.0) < 0.06
    assert np.allclose(r.centre, t + R @ [0, 0, 7.0], atol=0.06)               # the middle of the axis segment
    assert abs(abs(r.frame[:, 2] @ R[:, 2]) - 1.0) < 1e-3                       # the axis


@pytest.mark.parametrize("seed", range(2))
def test_torus_is_recovered_with_pose_and_size(seed):
    field, R, t = solid(sp.Torus(8.0, 3.0), seed)
    r = fit_torus(make_samples(field))
    assert r.explained > 0.99 and r.u is None and r.v is None
    assert abs(r.params["major"] - 8.0) < 0.06 and abs(r.params["minor"] - 3.0) < 0.06
    assert np.allclose(r.centre, t, atol=0.06) and abs(abs(r.frame[:, 2] @ R[:, 2]) - 1.0) < 1e-3


SEGMENTS = [
    ("cylinder", sp.Cylinder(6.0, 12.0, u=(0.0, np.radians(120.0))), fit_cylinder, 120.0),
    ("cylinder", sp.Cylinder(6.0, 12.0, u=(np.radians(40.0), np.radians(300.0))), fit_cylinder, 260.0),
    ("cone", sp.Cone(8.0, 3.0, 12.0, u=(np.radians(-20.0), np.radians(180.0))), fit_cone, 200.0),
    ("torus", sp.Torus(9.0, 3.0, u=(0.0, np.radians(90.0))), fit_torus, 90.0),
    ("torus", sp.Torus(9.0, 3.0, u=(np.radians(30.0), np.radians(210.0))), fit_torus, 180.0),
    ("sphere", sp.Sphere(8.0, u=(0.0, np.radians(270.0))), fit_sphere, 270.0),
]


@pytest.mark.parametrize("kind,prim,fitter,sweep", SEGMENTS)
def test_sweep_angle_u_is_recovered(kind, prim, fitter, sweep):
    field, R, t = solid(prim, 5)
    r = fitter(make_samples(field))
    assert r.kind == kind and r.explained > 0.99 and r.u is not None
    assert r.u[0] == 0.0 and abs(np.degrees(r.u[1]) - sweep) < 1.5
    assert "u_sweep" in r.params


def test_minor_circle_range_v_of_a_torus_is_recovered():
    prim = sp.Torus(9.0, 3.0, v=(np.radians(-60.0), np.radians(150.0)))          # a tube with a wedge taken out of its profile
    field, _, _ = solid(prim, 2)
    r = fit_torus(make_samples(field))
    assert r.explained > 0.99 and r.v is not None and r.u is None
    assert abs(np.degrees(r.v[1] - r.v[0]) - 210.0) < 2.0


@pytest.mark.parametrize("latitude,v", [("plane", (-20.0, 90.0)), ("cone", (-20.0, 90.0)), ("cone", (10.0, 70.0))])
def test_latitude_range_v_of_a_sphere_is_recovered(latitude, v):
    prim = sp.Sphere(8.0, v=tuple(np.radians(v)), latitude=latitude)
    field, _, _ = solid(prim, 1)
    r = fit_sphere(make_samples(field))
    assert r.explained > 0.99 and r.v is not None and r.latitude == latitude
    lo, hi = sorted(np.degrees(r.v))
    if abs(hi - 90.0) < 5.0 or abs(lo + 90.0) < 5.0:                            # a pole end is unobservable to a few degrees
        assert abs((hi - lo) - (v[1] - v[0])) < 6.0
    else:
        assert abs((hi - lo) - (v[1] - v[0])) < 1.5


def test_full_bodies_get_no_segment():
    for prim, fitter in ((sp.Cylinder(5.0, 10.0), fit_cylinder), (sp.Cone(8.0, 3.0, 12.0), fit_cone), (sp.Torus(9.0, 3.0), fit_torus),
                         (sp.Sphere(7.0), fit_sphere)):
        r = fitter(make_samples(solid(prim, 3)[0]))
        assert r.explained > 0.99 and r.u is None and r.v is None, prim


def test_a_cylinder_with_a_flat_is_not_a_sector():
    """The flat is a plane that does not pass through the axis: the empty arc has no side faces, so there is no segment."""
    field = load_field(ROOT / "dataset_gen" / "examples" / "cy_flat.stl")
    r = fit_cylinder(make_samples(field))
    assert r.u is None and 0.6 < r.explained < 0.99


def test_a_body_far_larger_than_the_part_is_a_plane_in_disguise():
    from reco_field.fit import _disguised
    assert _disguised("sphere", np.array([500.0]), 30.0) and _disguised("cylinder", np.array([500.0, 500.0, 5.0]), 30.0)
    assert not _disguised("sphere", np.array([20.0]), 30.0) and not _disguised("box", np.array([500.0]), 30.0)


EX = ROOT / "dataset_gen" / "examples"


@pytest.mark.skipif(not (EX / "bs_box_sphere_pocket.stl").is_file(), reason="example STLs missing")
def test_example_parts_base_primitive_and_leftover():
    box = fit_best(load_field(EX / "bs_box.stl"))[0]
    assert box.kind == "box" and box.explained > 0.99
    assert np.allclose(sorted([box.params["hx"], box.params["hy"], box.params["hz"]]), [6.0, 8.0, 10.0], atol=0.05)
    sph = fit_best(load_field(EX / "bs_sphere.stl"))[0]
    assert sph.kind == "sphere" and sph.explained > 0.99 and abs(sph.params["radius"] - 10.0) < 0.05
    pocket = fit_best(load_field(EX / "bs_box_sphere_pocket.stl"))[0]
    assert pocket.kind == "box" and 0.85 < pocket.explained < 0.97                 # the pocket's surface is left over
    assert np.allclose(sorted([pocket.params["hx"], pocket.params["hy"], pocket.params["hz"]]), [10, 10, 10], atol=0.1)
    assert pocket.rms_explained < 0.1
