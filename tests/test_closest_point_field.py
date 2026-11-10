"""dataset_gen/src/closest_point_field.py: the closest-surface-point vector field of a mesh (STL) or a STEP model, against
trimesh's exact closest_point as an independent oracle and against the analytic SDFs of dataset_gen/src/sdf_primitives.py."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField, closest_points_on_triangles, load_field, voxel_surface_mesh  # noqa: E402


def sphere_field(radius=10.0, subdivisions=4):
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return SurfaceField(m.vertices, m.faces), m


def box_field(extents=(8.0, 5.0, 3.0)):
    m = trimesh.creation.box(extents=extents)
    return SurfaceField(m.vertices, m.faces), m


def test_point_triangle_projection_matches_trimesh_in_every_region():
    tri = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 3.0, 0.0]])
    rng = np.random.default_rng(0)
    P = rng.uniform(-4, 8, (3000, 3))                                     # face interior, edges and vertices all occur
    mine = closest_points_on_triangles(P, np.tile(tri[0], (3000, 1)), np.tile(tri[1], (3000, 1)), np.tile(tri[2], (3000, 1)))
    oracle = trimesh.triangles.closest_point(np.tile(tri, (3000, 1, 1)), P)
    assert np.abs(mine - oracle).max() < 1e-9


def test_query_agrees_with_trimesh_exact_closest_point():
    field, mesh = sphere_field()
    P = np.random.default_rng(1).uniform(-16, 16, (300, 3))
    q = field.query(P)
    cp, d, _ = trimesh.proximity.closest_point(mesh, P)
    assert np.abs(q.distance - d).max() < 1e-6                             # the distance is exact (ties only move the point)
    assert np.mean(np.linalg.norm(q.closest - cp, axis=1) < 1e-6) > 0.99


def test_box_field_is_the_exact_box_distance_and_points_along_the_face_normal():
    field, _ = box_field()
    rng = np.random.default_rng(2)
    P = rng.uniform(-7, 7, (2000, 3))
    q = field.query(P)
    exact = np.abs(sp.Box((4.0, 2.5, 1.5)).sdf(P))
    assert np.abs(q.distance - exact).max() < 1e-9
    outside = sp.Box((4.0, 2.5, 1.5)).sdf(P) > 0.05
    # outside a face's shadow the vector is the outward normal reversed; check the points nearest a face interior
    near_face = outside & (np.abs(np.abs(P[:, 0]) - 4) < 0.5) & (np.abs(P[:, 1]) < 2) & (np.abs(P[:, 2]) < 1.2)
    assert near_face.any() and (np.abs(np.abs((q.vector[near_face] / np.linalg.norm(q.vector[near_face], axis=1)[:, None]) @ [1, 0, 0]) - 1) < 1e-9).all()
    assert np.allclose(q.closest, P + q.vector) and np.allclose(np.linalg.norm(q.vector, axis=1), q.distance)


def test_signed_distance_from_the_stl_normal_is_negative_outside_and_positive_inside():
    """The opposite convention from sdf_primitives.py's analytic SDF (there: negative inside) - flip its sign for the oracle."""
    field, _ = box_field()
    rng = np.random.default_rng(8)
    P = rng.uniform(-7, 7, (3000, 3))
    q = field.query(P)
    assert np.allclose(np.abs(q.signed_distance), q.distance)
    oracle = -sp.Box((4.0, 2.5, 1.5)).sdf(P)                                      # sp.sdf: negative inside; flipped to match
    away_from_edges = np.abs(sp.Box((4.0, 2.5, 1.5)).sdf(P)) > 0.05               # near an edge/corner the nearest face is ambiguous
    assert (np.sign(q.signed_distance[away_from_edges]) == np.sign(oracle[away_from_edges])).all()
    assert (q.signed_distance[away_from_edges] < 0).any() and (q.signed_distance[away_from_edges] > 0).any()      # both signs occur


def test_signed_distance_on_a_sphere_matches_the_flipped_analytic_sdf():
    field, _ = sphere_field(10.0, subdivisions=5)
    rng = np.random.default_rng(9)
    P = rng.normal(size=(500, 3))
    P = P / np.linalg.norm(P, axis=1)[:, None] * rng.uniform(5.0, 16.0, (500, 1))       # both sides of the surface, off the medial axis
    oracle = -sp.Sphere(10.0).sdf(P)
    assert np.abs(field.query(P).signed_distance - oracle).max() < 0.05                 # icosphere tessellation error


def test_vector_is_minus_sdf_times_gradient_on_a_sphere():
    """Where the SDF is differentiable, V = -f grad f (the field carries the SDF and its gradient). Checked in the band around
    the surface: deep inside a FACETED sphere the nearest facet is ambiguous and the closest point can jump by an edge length."""
    field, _ = sphere_field(10.0, subdivisions=5)
    rng = np.random.default_rng(3)
    P = rng.normal(size=(500, 3))
    P = P / np.linalg.norm(P, axis=1)[:, None] * rng.uniform(5.0, 16.0, (500, 1))
    prim = sp.Sphere(10.0)
    f = prim.sdf(P)
    eps = 1e-5
    grad = np.stack([(prim.sdf(P + eps * e) - prim.sdf(P - eps * e)) / (2 * eps) for e in np.eye(3)], axis=1)
    got = field.query(P).vector
    err = np.linalg.norm(got - (-f[:, None] * grad), axis=1)
    # the icosphere's tessellation error: a transverse (direction) error that grows with the distance from the surface and
    # halves each time the mesh is refined (measured 0.017 within 1 unit of the surface at this resolution)
    assert (err < 0.02 + 0.03 * np.abs(f)).all()
    assert err[np.abs(f) < 1.0].max() < 0.03


def test_sample_points_stay_within_the_band():
    field, _ = box_field()
    P = field.sample_points(500, band=1.0, seed=4)
    assert P.shape == (500, 3) and field.query(P).distance.max() <= 1.0 + 1e-9
    assert (field.query(P).vector * 0 == 0).all() and np.abs(sp.Box((4.0, 2.5, 1.5)).sdf(P)).max() <= 1.0 + 1e-9    # both sides of the surface
    assert (sp.Box((4.0, 2.5, 1.5)).sdf(P) < 0).any() and (sp.Box((4.0, 2.5, 1.5)).sdf(P) > 0).any()


def test_grid_vectors_use_the_projects_array_frame():
    """Array axes [z, y reversed, x], components in voxels: outside the sphere the vector points back to the surface."""
    field, _ = sphere_field(10.0, subdivisions=4)
    g = field.grid_vectors(64, 40.0, anchor_centre=np.zeros(3))
    assert g.shape == (3, 64, 64, 64)
    v = 40.0 / 64
    i = int(round(15.0 / v + 32 - 0.5))                                                  # a voxel ~15 mm from the centre
    mid = 32
    assert g[2, mid, mid, i] < -3                                                        # +x side: points to -x (array axis 2)
    assert g[0, i, mid, mid] < -3                                                        # +z side: points to -z (array axis 0)
    j = 63 - i                                                                           # +y is the LOW index of array axis 1
    assert g[1, mid, j, mid] > 3                                                         # +y side: points to -y = +array axis 1
    assert np.abs(np.linalg.norm(g[:, mid, mid, i]) * v - (np.linalg.norm([(i + 0.5 - 32) * v, 0.5 * v, 0.5 * v]) - 10.0)) < 0.5


def test_a_closed_mesh_is_recognised_and_an_open_one_is_not():
    assert box_field()[0].is_closed()
    m = trimesh.creation.box(extents=(1, 1, 1))
    open_mesh = SurfaceField(m.vertices, m.faces[:-1])
    assert not open_mesh.is_closed()


def test_genus_of_a_sphere_a_box_a_torus_and_an_open_mesh():
    """Genus 0 for a simply-connected solid, 1 for a single ring/handle (like a torus or a picture frame), NaN when not
    closed - see docs/FRAME_FAILURE.md for a real part with genus 17 (a lattice of bridging rods, not a plain ring)."""
    assert sphere_field(10.0, subdivisions=3)[0].genus() == 0
    assert box_field()[0].genus() == 0
    m = trimesh.creation.torus(major_radius=10.0, minor_radius=3.0)
    assert SurfaceField(m.vertices, m.faces).genus() == 1
    m = trimesh.creation.box(extents=(1, 1, 1))
    open_mesh = SurfaceField(m.vertices, m.faces[:-1])
    assert open_mesh.genus() != open_mesh.genus()                                     # NaN != NaN


def test_voxel_surface_mesh_of_a_solid_block_and_a_shell_with_a_cavity():
    """A solid block is one closed genus-0 shell; a block with an internal cavity is TWO disjoint closed shells (its own
    skin and the cavity's wall), each genus 0 - not a single surface with a nonsensical negative genus."""
    v, f = voxel_surface_mesh(np.ones((6, 6, 6), bool))
    field = SurfaceField(v, f)
    assert field.is_closed() and field.genus() == 0

    occ = np.ones((10, 10, 10), bool)
    occ[2:8, 2:8, 2:8] = False                                                        # a hollow cube: one cavity, no tunnel
    v, f = voxel_surface_mesh(occ)
    field = SurfaceField(v, f)
    assert field.is_closed() and field.genus() == 0

    assert voxel_surface_mesh(np.zeros((4, 4, 4), bool)) == (None, None)


def test_voxel_surface_mesh_of_a_digital_torus_and_two_tunnels():
    n = 24
    yy, xx = np.indices((n, n))
    r = np.sqrt((yy - (n - 1) / 2) ** 2 + (xx - (n - 1) / 2) ** 2)
    washer = (r > 4) & (r < 10)
    occ = np.broadcast_to(washer, (6, n, n))
    v, f = voxel_surface_mesh(occ)
    assert SurfaceField(v, f).genus() == 1

    occ = np.ones((20, 20, 20), bool)
    zz, yy, xx = np.indices((20, 20, 20))
    for cx in (6, 14):
        occ &= ~(((yy - 10) ** 2 + (xx - cx) ** 2) < 9)
    v, f = voxel_surface_mesh(occ)
    assert SurfaceField(v, f).genus() == 2


def test_coarse_genus_resolves_a_tunnel_only_once_the_voxels_are_fine_enough():
    """`coarse_genus` on the exact box-minus-two-tubes mesh (genus 2): too coarse to see a tunnel merges it away (0);
    fine enough recovers 2. Not monotonic in between (aliasing right at the resolving scale) - see genus_by_resolution's
    docstring; this only checks the coarse and the fine ends, which are the reliable ones."""
    m = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    for cx in (-5.0, 5.0):
        tube = trimesh.creation.cylinder(radius=2.0, height=30.0, sections=32)
        tube.apply_translation([cx, 0, 0])
        m = m.difference(tube)
    field = SurfaceField(m.vertices, m.faces)
    assert field.genus() == 2
    assert field.coarse_genus(4) == 0                                                 # a 5 mm voxel swallows a 2 mm-radius tunnel
    assert field.coarse_genus(32) == 2                                                # well-resolved


def test_coarse_genus_cannot_see_a_handle_sealed_entirely_inside_a_cavity():
    """Known, documented limitation (coarse_genus's own docstring): a hollow box with a strut spanning its cavity has
    exact genus 1 (the strut is a real handle), but .fill() erases the whole sealed cavity - strut included - as solid
    material at every resolution, so coarse_genus never converges to it. genus() itself is unaffected (per-shell, never
    fills), which is how the mismatch is caught rather than silently trusted."""
    outer, inner = trimesh.creation.box(extents=(20.0, 20.0, 20.0)), trimesh.creation.box(extents=(12.0, 12.0, 12.0))
    strut = trimesh.creation.cylinder(radius=1.5, height=12.0, sections=24)             # both ends inside the cavity
    part = outer.difference(inner).union(strut)
    field = SurfaceField(part.vertices, part.faces)
    assert field.is_closed() and field.genus() == 1
    for r in (16, 32, 64):
        assert field.coarse_genus(r) == 0                                              # never reaches the true 1


def test_two_bridging_handles_add_two_to_the_genus():
    """A box with two independent tunnels drilled through it (genus 0 -> 2), built as an explicit boolean difference so
    the ground truth is known, not fitted."""
    m = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    for cx in (-5.0, 5.0):
        tube = trimesh.creation.cylinder(radius=2.0, height=30.0, sections=32)
        tube.apply_translation([cx, 0, 0])
        m = m.difference(tube)
    field = SurfaceField(m.vertices, m.faces)
    assert field.is_closed() and field.genus() == 2


def test_stl_file_round_trip(tmp_path):
    m = trimesh.creation.box(extents=(6.0, 4.0, 2.0))
    p = tmp_path / "b.stl"
    m.export(p)
    field = load_field(p)
    assert np.allclose(field.query(np.array([[10.0, 0, 0]])).distance, 7.0)              # 10 - 3


def _freecad():
    from src.freecad_path import find_freecadcmd  # $SHAPE_RECO_FREECADCMD, PATH, /usr/bin/freecadcmd (T-04)
    return find_freecadcmd()


@pytest.mark.skipif(_freecad() is None, reason="needs FreeCAD")
def test_step_model_is_tessellated_and_gives_the_analytic_cylinder_field():
    tmp = Path(tempfile.mkdtemp())
    script = tmp / "make.py"
    script.write_text(f'import Part\nPart.makeCylinder(3.0, 8.0).exportStep(r"{tmp}/cyl.step")\n')
    subprocess.run([_freecad(), str(script)], capture_output=True, timeout=120)
    assert (tmp / "cyl.step").is_file()
    field = load_field(tmp / "cyl.step", deflection=0.01)
    P = np.random.default_rng(5).uniform(-6, 6, (800, 3))
    P[:, 2] = np.random.default_rng(6).uniform(-3, 11, 800)
    exact = np.abs(sp.Cylinder(3.0, 8.0).sdf(P))
    assert np.abs(field.query(P).distance - exact).max() < 0.03                          # the tessellation's deflection


def test_approximate_query_is_close_and_exact_flag_is_honoured():
    field, mesh = sphere_field(10.0, subdivisions=4)
    P = np.random.default_rng(7).uniform(-16, 16, (500, 3))
    exact, fast = field.query(P), field.query(P, exact=False)
    assert np.abs(fast.distance - exact.distance).max() < 0.5                      # one pass: never worse than about a triangle
    assert (fast.distance >= exact.distance - 1e-9).all()                          # it can only miss a nearer triangle


@pytest.mark.parametrize("make", [lambda: box_field()[1], lambda: sphere_field(subdivisions=3)[1],
                                  lambda: trimesh.creation.cylinder(radius=4.0, height=9.0, sections=48)])
def test_torch_one_pass_query_matches_the_numpy_one_pass_query(make, monkeypatch):
    """exact=False: the dense torch backend gives the numpy path's answer (large triangles included: a raw box and a
    cylinder's caps go to the brute-force list)."""
    pytest.importorskip("torch")
    m = make()
    pts = np.random.default_rng(3).uniform(-12, 12, (5000, 3))
    monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", "numpy")
    ref = SurfaceField(m.vertices, m.faces).query(pts, exact=False)
    for backend in ("torch", "cuda"):
        monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", backend)
        got = SurfaceField(m.vertices, m.faces).query(pts, exact=False)
        np.testing.assert_allclose(got.distance, ref.distance, atol=1e-9)
        np.testing.assert_allclose(got.closest, ref.closest, atol=1e-9)


def test_gpu_knn_gives_the_kdtree_candidate_sets():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from scipy.spatial import cKDTree
    from src.closest_point_field import _knn_gpu
    rng = np.random.default_rng(5)
    C, P = rng.uniform(-10, 10, (3000, 3)), rng.uniform(-20, 20, (5000, 3))
    _, ref = cKDTree(C).query(P, k=64)
    got = _knn_gpu(C, P, 64)
    assert all(set(a) == set(b) for a, b in zip(ref, got))
