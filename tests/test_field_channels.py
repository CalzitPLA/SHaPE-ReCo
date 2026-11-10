"""dataset_gen/src/field_channels.py: the one field encoding (occupancy, sdf positive inside, V = c(x) - x) shared by the
generator (--with-field) and the residual loop - against the analytic SDFs of sdf_primitives (sign flipped to this
convention), trimesh's `contains`, the peel grid convention and the .bin axis order."""
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import field_channels as fc  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField, load_field  # noqa: E402


def field_of(mesh):
    return SurfaceField(mesh.vertices, mesh.faces)


def test_grid_centres_match_peel_convention():
    from reco_field.peel import _grid_centres
    spec = fc.grid_spec(np.array([-3.0, -1.0, 2.0]), np.array([5.0, 7.0, 10.0]), 12)
    assert np.array_equal(spec.centres(), _grid_centres(np.asarray(spec.lo), np.asarray(spec.hi), 12))


def test_grid_spec_is_a_cube_about_the_centre():
    spec = fc.grid_spec([0, 0, 0], [4, 10, 6], 8)
    assert np.allclose(np.subtract(spec.hi, spec.lo), 10.0)
    assert np.allclose((np.add(spec.hi, spec.lo)) / 2, [2, 5, 3])


def test_offcentre_noncubic_box_sdf_v_occupancy_and_axis_order():
    half, centre = np.array([6.0, 3.5, 2.0]), np.array([2.3, -4.1, 1.7])
    mesh = trimesh.creation.box(extents=2 * half)
    mesh.apply_translation(centre)
    spec = fc.grid_spec(-12, 12, 32)
    ch = fc.mesh_channels(field_of(mesh), spec)
    assert ch.shape == (5, 32, 32, 32) and ch.dtype == np.float32 and np.isfinite(ch).all()
    P = spec.centres()
    analytic = -sp.Box(half).sdf(P - centre)                       # sdf_primitives is negative inside: flip
    assert np.abs(ch[1] - analytic).max() < 1e-4
    assert np.array_equal(ch[0] > 0, analytic > 0)
    # V lands on the surface
    on = P + np.moveaxis(ch[2:5], 0, -1)
    assert np.abs(sp.Box(half).sdf(on - centre)).max() < 1e-4
    # axis order is world [x, y, z]: the occupied extent along array axis i is the box's extent along world axis i
    idx = np.argwhere(ch[0] > 0)
    ext = (idx.max(0) - idx.min(0) + 1) * spec.pitch
    assert np.allclose(ext, 2 * half, atol=spec.pitch)
    mid = (idx.max(0) + idx.min(0) + 1) / 2 * spec.pitch + np.asarray(spec.lo)
    assert np.allclose(mid, centre, atol=spec.pitch)


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_curved_meshes_match_analytic_sdf(kind):
    if kind == "sphere":
        mesh, prim, tol = trimesh.creation.icosphere(subdivisions=5, radius=7.0), sp.Sphere(7.0), 0.02
        offset = np.array([1.0, -2.0, 0.5])
        local = lambda p: p - offset  # noqa: E731
    else:
        mesh = trimesh.creation.cylinder(radius=5.0, height=12.0, sections=256)     # centred, from z=-6 to 6
        prim, tol, offset = sp.Cylinder(5.0, 12.0), 0.01, np.array([-1.5, 2.0, 0.3])
        local = lambda p: p - offset + np.array([0.0, 0.0, 6.0])  # noqa: E731    sp.Cylinder stands on z=0
    mesh.apply_translation(offset)
    spec = fc.grid_spec(-12, 12, 32)
    ch = fc.mesh_channels(field_of(mesh), spec)
    analytic = -prim.sdf(local(spec.centres()))
    err = np.abs(ch[1] - analytic)
    # tessellation error of the mesh (tol), plus the one-pass query's rare misses on sliver triangles (field_channels.
    # mesh_channels: measured up to ~0.05-0.09 mm on a handful of centres of this 256-section cylinder)
    assert np.quantile(err, 0.999) < tol and err.max() < 0.1
    near = np.abs(analytic) > tol
    assert np.array_equal((ch[0] > 0)[near], (analytic > 0)[near])
    assert np.allclose(np.linalg.norm(ch[2:5], axis=0), np.abs(ch[1]), atol=1e-4)     # |V| = |sdf|


def test_cone_example_occupancy_matches_trimesh_contains():
    """co_cone: the face-normal sign was measured wrong in 1465 voxels even with exact queries (apex / edges); the
    default parity sign must agree with an independent ray test everywhere."""
    field = load_field(ROOT / "dataset_gen" / "examples" / "co_cone.stl")
    lo, hi = field.bounds
    spec = fc.grid_spec(lo - 2.0, hi + 2.0, 40)
    ch = fc.mesh_channels(field, spec)
    inside = trimesh.Trimesh(field.vertices, field.faces).contains(spec.centres().reshape(-1, 3)).reshape(ch[0].shape)
    assert np.array_equal(ch[0] > 0, inside)
    assert np.array_equal(fc.parity_occupancy(field, spec), inside)


def test_voxel_channels_sign_distance_and_v():
    occ = np.zeros((16, 16, 16), bool)
    occ[2:9, 3:12, 4:14] = True
    occ[4:9, 5:8, 4:7] = False                                      # a notch, so the surface is not convex
    spec = fc.grid_spec(-4, 4, 16)
    ch = fc.voxel_channels(occ, spec)
    assert np.isfinite(ch).all()
    assert np.array_equal(ch[0] > 0, occ) and np.array_equal(ch[1] > 0, occ)
    assert np.abs(ch[1]).min() >= 0.5 * spec.pitch - 1e-9           # centres are half a voxel off the blocky surface
    # the outward-oriented voxel surface gives the same sign by its face normals
    q = fc.voxel_field(occ, spec).query(spec.centres().reshape(-1, 3))
    assert np.array_equal(q.signed_distance.reshape(occ.shape) > 0, occ)
    empty = fc.voxel_channels(np.zeros_like(occ), spec)
    assert np.isfinite(empty).all() and (empty[1] < 0).all() and not empty[0].any()


def test_surface_points_on_surface_with_outward_normals():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=6.0)
    field = field_of(mesh)
    pts = fc.surface_points(field, 500, seed=1)
    assert pts.shape == (500, 6)
    assert np.abs(np.linalg.norm(pts[:, :3], axis=1) - 6.0).max() < 0.05
    assert np.allclose(np.linalg.norm(pts[:, 3:], axis=1), 1.0, atol=1e-5)
    assert ((pts[:, :3] * pts[:, 3:]).sum(1) > 0).all()             # outward on a sphere


def test_bin_world_round_trip_and_convention():
    """bin_to_world is peel.real_voxel_occupancy's flip: an array-frame [z, -y, x] grid of an off-centre box (from
    sdf_primitives' own voxel_centres) becomes the world-indexed occupancy of the same box on the same grid."""
    half, centre = np.array([6.0, 3.0, 2.0]), np.array([4.0, -5.0, 2.0])
    grid_mm, r = 40.0, 32
    raw = (sp.Box(half).sdf(sp.voxel_centres(r, grid_mm) - centre) <= 0)
    world = fc.bin_to_world(raw)
    spec = fc.inscale_spec(grid_mm, r)
    assert np.array_equal(world, sp.Box(half).sdf(spec.centres() - centre) <= 0)
    assert np.array_equal(fc.world_to_bin(world), raw)


def test_network_input_matches_fused_layout():
    """vf_exact's network input has the fused kind's sign (negative inside), scaling and array frame."""
    from training.field_input import encode
    half = np.array([7.0, 4.0, 3.0])
    spec = fc.inscale_spec(40.0, 32)
    ch = fc.voxel_channels(sp.Box(half).sdf(spec.centres() - np.array([2.0, -3.0, 1.0])) <= 0, spec)
    x = fc.network_input(ch, spec)
    fused = encode(fc.world_to_bin(ch[0]) * 2.0 - 1.0, "fused")
    assert x.shape == fused.shape
    assert np.array_equal(x[0], fused[0])
    assert np.array_equal(np.sign(x[1]), np.sign(fused[1]))
    body = np.abs(fused[1]) < 0.9                                   # not truncated: vectors comparable
    cos = (x[2:] * fused[2:]).sum(0)[body] / np.maximum(np.linalg.norm(x[2:], axis=0) * np.linalg.norm(fused[2:], axis=0), 1e-9)[body]
    assert np.median(cos) > 0.95                                    # same direction convention, same axes


def test_rotate_network_input_turns_vectors():
    from training.rotation_symmetry import GROUP, rotate_volume
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 6, 6, 6)).astype(np.float32)
    g = GROUP[7]
    y = fc.rotate_network_input(x, g)
    assert np.allclose(y[0], rotate_volume(x[0], g))
    moved = np.stack([rotate_volume(c, g) for c in x[2:5]])
    assert np.allclose(y[2:5], np.einsum("ij,j...->i...", g, moved), atol=1e-6)


def test_npz_round_trip(tmp_path):
    spec = fc.inscale_spec(40.0, 8)
    ch = np.random.default_rng(0).normal(size=(5, 8, 8, 8)).astype(np.float32)
    fc.save_field_npz(tmp_path / "a.field.npz", ch, spec, points=np.zeros((4, 6)), source="mesh")
    ch2, spec2, data = fc.load_field_npz(tmp_path / "a.field.npz")
    assert spec2 == spec and np.allclose(ch2, ch, atol=1e-2) and str(data["source"]) == "mesh"
